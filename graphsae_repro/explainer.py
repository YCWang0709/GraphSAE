"""SAE-GNNExplainer: local source decomposition and active decoder projection."""

from __future__ import annotations

from dataclasses import asdict, replace
import torch
from torch import Tensor
import torch.nn.functional as F

from .contributions import source_scores
from .policies import ExtractionPolicy, select_nodes, refine_nodes, score_blocks
from .sae import SAEArtifact


def target_direction(
    artifact: SAEArtifact,
    representation: Tensor,
    *,
    weighting="activity",
    prediction_gradient: Tensor | None = None,
    num_features: int | None = None,
):
    codes = artifact.encode(representation.reshape(1, -1))[0]
    active = torch.where(codes > 1e-8)[0]
    active = active[torch.argsort(codes[active], descending=True, stable=True)]
    if num_features is not None:
        active = active[:num_features]
    decoder = artifact.sae.decoder.weight.detach().cpu()[:, active]
    directions = F.normalize(decoder, dim=0)
    weights = codes[active]
    if weighting == "prediction":
        if prediction_gradient is None:
            raise ValueError(
                "prediction weighting requires an original-space head gradient"
            )
        effects = (
            directions * artifact.std.cpu().reshape(-1, 1)
        ).T @ prediction_gradient.cpu()
        weights = weights * effects
    elif weighting != "activity":
        raise ValueError("weighting must be activity or prediction")
    direction = (directions @ weights) / artifact.std.cpu().reshape(-1)
    features = [
        {"feature_id": int(k), "activation": float(codes[k]), "weight": float(w)}
        for k, w in zip(active, weights)
    ]
    return direction, features


def _induced_subgraph(edge_index, nodes):
    ids = torch.tensor(nodes, dtype=torch.long, device=edge_index.device)
    keep = torch.isin(edge_index[0], ids) & torch.isin(edge_index[1], ids)
    return {"nodes": nodes, "edges": edge_index[:, keep].T.cpu().tolist()}


def explain(
    model,
    data,
    artifact,
    *,
    target_index=0,
    top_k=8,
    num_features=None,
    policy: ExtractionPolicy | None = None,
    task_type="classification",
    device="cpu",
    evaluator=None,
    blocks=None,
):
    model = model.to(device).eval()
    data = data.to(device)
    policy = policy or ExtractionPolicy(budget=top_k)
    with torch.no_grad():
        logits, capture = model(
            data.x, data.edge_index, batch=getattr(data, "batch", None), capture=True
        )
    h = capture.graph if model.task == "graph" else capture.node
    raw = h[target_index].detach()
    gradient = None
    if policy.weighting == "prediction":
        with torch.enable_grad():
            value = raw.clone().requires_grad_(True)
            output = model.classifier(value)
            if task_type == "regression":
                score = output.reshape(-1)[0]
            else:
                from .evaluation import class_logits

                row = class_logits(output.reshape(1, -1))[0]
                score = row[row.detach().argmax()]
            (gradient,) = torch.autograd.grad(score, value)
    direction, features = target_direction(
        artifact,
        raw,
        weighting=policy.weighting,
        prediction_gradient=gradient,
        num_features=num_features,
    )
    candidates, scores = source_scores(
        model, data, direction, target_index=target_index, device=device
    )
    variants = {
        key: select_nodes(
            candidates,
            scores,
            replace(policy, mode="fixed", refine_steps=0, evidence=sign),
        )
        for key, sign in [
            ("sae_positive", "positive"),
            ("sae_negative", "negative"),
            ("sae_signed", "mixed"),
        ]
    }
    if policy.mode == "one_hop":
        if model.task != "node":
            raise ValueError("one_hop extraction applies to node targets")
        neighbors = data.edge_index[0, data.edge_index[1] == target_index].tolist()
        selected = sorted(set(neighbors) & set(candidates))
    else:
        selected = select_nodes(candidates, scores, policy)
    if policy.refine_steps:
        if policy.mode == "fixed":
            raise ValueError("Refinement changes cardinality; use native-size policies")
        if evaluator is None:
            from .evaluation import deletion_metrics

            evaluator = lambda ids: deletion_metrics(
                model,
                data,
                ids,
                target_index=target_index,
                task_type=task_type,
                device=device,
                stable_sage=policy.stable_sage,
            )[
                (
                    "absolute_change"
                    if task_type == "regression"
                    else "probability_decrease"
                )
            ]
        if policy.refinement == "blocks" and blocks is None:
            blocks = score_blocks(
                candidates,
                scores,
                data.edge_index,
                target_index=target_index if model.task == "node" else None,
            )
        selected = refine_nodes(
            selected,
            candidates,
            scores,
            evaluator,
            steps=policy.refine_steps,
            edge_index=data.edge_index,
            blocks=blocks,
            size_penalty=policy.size_penalty,
        )
    return {
        "task": model.task,
        "target_index": target_index,
        **(
            {"target_node": target_index}
            if model.task == "node"
            else {"graph_index": target_index}
        ),
        "reference_class": (
            int(logits[target_index].argmax())
            if logits.shape[-1] > 1
            else int(logits[target_index].item() > 0)
        ),
        "active_features": features,
        "policy": asdict(policy),
        "node_budget": policy.budget,
        "selected_nodes": selected,
        **variants,
        "node_scores": [
            {
                "node_id": u,
                "signed_score": float(s),
                "positive_score": max(0.0, float(s)),
                "negative_score": max(0.0, -float(s)),
            }
            for u, s in zip(candidates, scores)
        ],
        "subgraphs": {
            key: _induced_subgraph(data.edge_index, nodes)
            for key, nodes in {**variants, "selected": selected}.items()
        },
    }


def explain_node(model, data, artifact, *, target_node, **kwargs):
    if model.task != "node":
        raise ValueError("A node backbone is required")
    return explain(model, data, artifact, target_index=target_node, **kwargs)


def explain_graph(model, data, artifact, *, graph_index=0, **kwargs):
    if model.task != "graph":
        raise ValueError("A graph backbone is required")
    return explain(model, data, artifact, target_index=graph_index, **kwargs)
