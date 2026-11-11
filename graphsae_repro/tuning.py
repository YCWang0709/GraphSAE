"""Validation-only SAE and complete explanation-pipeline selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
import math
import torch

from .activations import ActivationPack
from .evaluation import deletion_metrics, representation_metrics
from .explainer import explain
from .policies import ExtractionPolicy
from .sae import SAEConfig, train_sae


def parameter_grid(base: SAEConfig, grid: dict):
    """Enumerate an explicit, caller-supplied training search space."""
    unknown = set(grid) - set(asdict(base))
    if unknown or any(not values for values in grid.values()):
        raise ValueError(f"Invalid parameter grid: {unknown}")
    keys = list(grid)
    return [
        replace(base, **dict(zip(keys, values)))
        for values in product(*(grid[k] for k in keys))
    ]


def require_split(pack, split):
    declared = {r.get("split") for r in pack.metadata if r.get("split") is not None}
    if pack.info.get("split"):
        declared.add(pack.info["split"])
    if declared != {split}:
        raise ValueError(f"Explicit split={split!r} is required, found {declared}")


def check_disjoint(train, validation, expected_split="val"):
    require_split(train, "train")
    require_split(validation, expected_split)
    for key in ("node_id", "graph_id", "sample_id"):
        a = {r[key] for r in train.metadata if key in r}
        b = {r[key] for r in validation.metadata if key in r}
        if a & b:
            raise ValueError(f"Overlapping training/validation {key}")


def subset_pack(pack, limit=2048, seed=0):
    if len(pack.z) <= limit:
        return pack
    ids = (
        torch.randperm(len(pack.z), generator=torch.Generator().manual_seed(seed))[
            :limit
        ]
        .sort()
        .values
    )
    return ActivationPack(
        pack.z[ids],
        [pack.metadata[i] for i in ids.tolist()],
        pack.task,
        pack.source,
        {**pack.info, "num_rows": len(ids)},
    )


def fit_sae(
    activations, config=None, *, groups=None, prediction_head=None, device="cpu"
):
    """General tensor API; reconstruction-only by default, with optional groups/head."""
    config = config or SAEConfig(lambda_ground=0.0, lambda_faith=0.0, lambda_coh=0.0)
    metadata = [
        {
            "sample_id": i,
            "split": "train",
            **({"group": int(groups[i])} if groups is not None else {}),
        }
        for i in range(len(activations))
    ]
    pack = ActivationPack(
        activations, metadata, "representation", info={"split": "train"}
    )
    return train_sae(pack, config, prediction_head=prediction_head, device=device)


def tune_sae(
    train, validation, configs, *, prediction_head=None, criterion="nmse", device="cpu"
):
    """Fit candidates on train; minimize a named validation representation metric."""
    check_disjoint(train, validation)
    if not configs:
        raise ValueError("At least one SAE configuration is required")
    trials, best = [], None
    for index, config in enumerate(configs):
        artifact = train_sae(
            train, config, prediction_head=prediction_head, device=device
        )
        metrics = representation_metrics(
            artifact, validation, prediction_head=prediction_head, device=device
        )
        value = metrics.get(criterion)
        if value is None or not math.isfinite(value):
            raise ValueError(f"Invalid selection metric {criterion}")
        trials.append({"index": index, "config": asdict(config), "validation": metrics})
        if best is None or value < best[0]:
            best = value, artifact, index
    return best[1], {
        "criterion": criterion,
        "selected_index": best[2],
        "trials": trials,
    }


@dataclass
class ValidationTarget:
    data: object
    target_index: int = 0
    truth: list | None = None
    split: str = "val"


def tune_pipeline(
    model,
    train,
    validation_targets,
    configs,
    policies,
    *,
    budgets=(8, 12, 16),
    task_type="classification",
    recovery_weight=1.0,
    size_weight=0.0,
    device="cpu",
):
    """Select one training+ranking policy, shared across all requested budgets.

    Utility = mean deletion faithfulness + recovery_weight * mean Jaccard
    - size_weight * mean selected nodes. Regression uses absolute prediction
    change and no recovery term. These coefficients must be fixed before search.
    Native-size selection uses budgets=None and each policy's native settings.
    """
    require_split(train, "train")
    if not configs or not policies or not validation_targets:
        raise ValueError(
            "Nonempty configurations, policies and validation targets required"
        )
    if any(t.split != "val" for t in validation_targets):
        raise ValueError("Only validation targets may enter pipeline selection")
    for target in validation_targets:
        if model.task == "node":
            mask = getattr(target.data, "val_mask", None)
            if mask is None or mask.ndim != 1 or not bool(mask[target.target_index]):
                raise ValueError("Node validation targets must belong to val_mask")
        else:
            graph_id = getattr(target.data, "graph_id", None)
            train_ids = {row["graph_id"] for row in train.metadata if "graph_id" in row}
            if graph_id is None or int(graph_id[target.target_index]) in train_ids:
                raise ValueError(
                    "Graph validation targets require disjoint stable graph_id values"
                )
    trials, best = [], None
    for config_id, config in enumerate(configs):
        config = replace(config, seed=0, task_type=task_type)
        artifact = train_sae(
            train, config, prediction_head=model.classifier, device=device
        )
        for policy_id, policy in enumerate(policies):
            if budgets and (policy.mode != "fixed" or policy.refine_steps):
                raise ValueError(
                    "Matched-budget selection requires fixed ranking policies"
                )
            rows = []
            for target in validation_targets:
                for budget in budgets if budgets is not None else (policy.budget,):
                    selected_policy = replace(policy, budget=budget)
                    result = explain(
                        model,
                        target.data,
                        artifact,
                        target_index=target.target_index,
                        policy=selected_policy,
                        task_type=task_type,
                        device=device,
                    )
                    rows.append(
                        deletion_metrics(
                            model,
                            target.data,
                            result["selected_nodes"],
                            target_index=target.target_index,
                            task_type=task_type,
                            truth=target.truth,
                            device=device,
                            stable_sage=selected_policy.stable_sage,
                        )
                    )
            primary = (
                "absolute_change"
                if task_type == "regression"
                else "probability_decrease"
            )
            faith = sum(r[primary] for r in rows) / len(rows)
            annotated = [r["jaccard"] for r in rows if "jaccard" in r]
            recovery = sum(annotated) / len(annotated) if annotated else 0.0
            size = sum(r["num_nodes"] for r in rows) / len(rows)
            utility = (
                faith
                + (recovery_weight * recovery if task_type == "classification" else 0.0)
                - size_weight * size
            )
            if not math.isfinite(utility):
                raise ValueError("Nonfinite validation utility")
            trials.append(
                {
                    "config_id": config_id,
                    "policy_id": policy_id,
                    "utility": utility,
                    "faithfulness": faith,
                    "recovery": recovery,
                    "nodes": size,
                }
            )
            if best is None or utility > best[0]:
                best = utility, config, policy
    return (
        best[1],
        best[2],
        {
            "budgets": budgets,
            "selection_seed": 0,
            "recovery_weight": recovery_weight,
            "size_weight": size_weight,
            "selected_config": asdict(best[1]),
            "selected_policy": asdict(best[2]),
            "trials": trials,
        },
    )


def refit_seeds(train, config, *, prediction_head=None, seeds=(0, 1, 2), device="cpu"):
    """Reuse selected settings without reselecting on held-out test targets."""
    for seed in seeds:
        yield seed, train_sae(
            train,
            replace(config, seed=seed),
            prediction_head=prediction_head,
            device=device,
        )


def tune_l1(train, validation, *, prediction_head=None, device="cpu"):
    """Supplementary L1+Path: 12--20 active preferred, else closest log-ratio to 16."""
    check_disjoint(train, validation)
    candidates = []
    for coefficient in (0.001, 0.01, 0.1, 1.0):
        config = SAEConfig(
            encoder_type="relu",
            l1_weight=coefficient,
            lambda_ground=0.0,
            lambda_faith=0.0,
            lambda_coh=0.0,
            seed=0,
        )
        artifact = train_sae(train, config, device=device)
        metrics = representation_metrics(
            artifact, validation, prediction_head=prediction_head, device=device
        )
        active = metrics["mean_active_features"]
        preferred = 12 <= active <= 20
        key = (
            (0, metrics["nmse"])
            if preferred
            else (1, abs(math.log(max(active, 1e-12) / 16)), metrics["nmse"])
        )
        candidates.append((key, artifact, metrics))
    selected = min(range(len(candidates)), key=lambda i: candidates[i][0])
    return candidates[selected][1], {
        "selected_index": selected,
        "trials": [
            {"coefficient": c[1].config.l1_weight, "validation": c[2]}
            for c in candidates
        ],
    }
