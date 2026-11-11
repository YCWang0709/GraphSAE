"""Semantic and causal alignment measurements for SAE features."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import torch
from torch import Tensor
import torch.nn.functional as F

from .activations import ActivationPack
from .sae import SAEArtifact


def top_active_feature_ids(
    codes: Tensor, *, limit: int = 16, min_activation: float = 1e-8
) -> list[int]:
    active = torch.nonzero(codes.detach().cpu() > min_activation, as_tuple=False).view(
        -1
    )
    if active.numel() == 0:
        return []
    order = torch.argsort(codes.detach().cpu()[active], descending=True)
    return [int(fid) for fid in active[order[:limit]].tolist()]


def feature_alignment_table(
    pack: ActivationPack,
    artifact: SAEArtifact,
    *,
    min_activation: float = 1e-8,
    top_q: int = 50,
    max_features: int = 128,
) -> pd.DataFrame:
    """Measure class/motif purity of every SAE feature in an activation pack."""
    codes = artifact.encode(pack.z)
    labels = [row.get("label") for row in pack.metadata]
    if artifact.config.task_type != "classification" or any(
        label is None for label in labels
    ):
        raise ValueError("Class purity requires classification labels")
    valid_labels = sorted({int(label) for label in labels if label is not None})
    rows: list[dict] = []
    for feature_id in range(codes.size(1)):
        active = codes[:, feature_id] > min_activation
        indices = torch.nonzero(active, as_tuple=False).view(-1).tolist()
        purity_ids = (
            indices
            if pack.task == "graph"
            else torch.argsort(codes[:, feature_id], descending=True, stable=True)[
                : min(top_q, len(labels))
            ].tolist()
        )
        active_labels = [int(labels[idx]) for idx in purity_ids] if indices else []
        counts = {label: active_labels.count(label) for label in valid_labels}
        majority_label, majority_count = (
            (None, 0) if not counts else max(counts.items(), key=lambda item: item[1])
        )
        rows.append(
            {
                "feature_id": int(feature_id),
                "active_count": len(indices),
                "active_fraction": len(indices) / max(1, codes.size(0)),
                "mean_activation_when_active": (
                    float(codes[active, feature_id].mean().item()) if indices else 0.0
                ),
                "majority_label": majority_label,
                "majority_count": int(majority_count),
                "label_purity": float(majority_count / max(1, len(active_labels))),
                **{
                    f"label_{label}_count": int(count)
                    for label, count in counts.items()
                },
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["active_count", "feature_id"], ascending=[False, True])
        .head(max_features)
        .reset_index(drop=True)
    )


def _prediction_record(
    logits: Tensor, index: int = 0, reference_class: int | None = None
) -> dict:
    row = logits.detach().cpu()[int(index)]
    if row.numel() == 1:
        row = torch.cat([torch.zeros_like(row.reshape(-1)), row.reshape(-1)])
    probabilities = F.softmax(row, dim=-1)
    prediction = int(row.argmax().item())
    reference = prediction if reference_class is None else int(reference_class)
    return {
        "prediction": prediction,
        "reference_class": reference,
        "reference_logit": float(row[reference].item()),
        "reference_probability": float(probabilities[reference].item()),
    }


def _comparison(before: dict, after: dict) -> dict:
    return {
        "prediction_changed": bool(before["prediction"] != after["prediction"]),
        "delta_reference_logit": float(
            after["reference_logit"] - before["reference_logit"]
        ),
        "delta_reference_probability": float(
            after["reference_probability"] - before["reference_probability"]
        ),
    }


@torch.no_grad()
def run_node_feature_ablation(
    model,
    data,
    artifact: SAEArtifact,
    *,
    target_node: int,
    feature_ids: Sequence[int] | None = None,
    num_features: int = 16,
    device: torch.device | str = "cpu",
) -> dict:
    """Ablate target-active SAE features and patch only the target's last-layer state."""
    device = torch.device(device)
    model = model.to(device).eval()
    data = data.to(device)
    logits, capture = model(
        data.x, data.edge_index, batch=getattr(data, "batch", None), capture=True
    )
    codes = artifact.encode(capture.node, device=device)
    target_node = int(target_node)
    selected = (
        list(feature_ids)
        if feature_ids is not None
        else top_active_feature_ids(codes[target_node], limit=num_features)
    )
    if not selected:
        raise ValueError(
            "The target node has no active SAE features under the current threshold."
        )
    modified_codes = codes[target_node : target_node + 1].clone()
    modified_codes[:, list(map(int, selected))] = 0.0
    patched_last = capture.node.detach().clone()
    baseline_last = patched_last.clone()
    baseline_last[target_node] = artifact.decode(
        codes[target_node : target_node + 1], device=device
    ).to(device)[0]
    patched_last[target_node] = artifact.decode(modified_codes, device=device).to(
        device
    )[0]
    modified_logits = model(
        data.x,
        data.edge_index,
        batch=getattr(data, "batch", None),
        patch_last=patched_last,
    )
    original_prediction = _prediction_record(logits, target_node)
    baseline_logits = model(
        data.x,
        data.edge_index,
        batch=getattr(data, "batch", None),
        patch_last=baseline_last,
    )
    original = _prediction_record(
        baseline_logits, target_node, original_prediction["reference_class"]
    )
    modified = _prediction_record(
        modified_logits, target_node, original["reference_class"]
    )
    return {
        "task": "node",
        "target_node": target_node,
        "feature_ids": [int(fid) for fid in selected],
        "original_prediction": original_prediction,
        "unablated_reconstruction": original,
        "modified": modified,
        **_comparison(original, modified),
    }


@torch.no_grad()
def run_graph_feature_ablation(
    model,
    data,
    artifact: SAEArtifact,
    *,
    graph_index: int = 0,
    feature_ids: Sequence[int] | None = None,
    num_features: int = 16,
    device: torch.device | str = "cpu",
) -> dict:
    """Ablate graph-level SAE features and patch the pooled final-layer activation."""
    device = torch.device(device)
    model = model.to(device).eval()
    data = data.to(device)
    logits, capture = model(
        data.x, data.edge_index, batch=getattr(data, "batch", None), capture=True
    )
    if capture.graph is None:
        raise ValueError(
            "run_graph_feature_ablation requires a graph-classification backbone"
        )
    graph_index = int(graph_index)
    codes = artifact.encode(capture.graph, device=device)
    selected = (
        list(feature_ids)
        if feature_ids is not None
        else top_active_feature_ids(codes[graph_index], limit=num_features)
    )
    if not selected:
        raise ValueError(
            "The graph has no active SAE features under the current threshold."
        )
    modified_codes = codes[graph_index : graph_index + 1].clone()
    modified_codes[:, list(map(int, selected))] = 0.0
    patched_graph = capture.graph.detach().clone()
    baseline_graph = patched_graph.clone()
    baseline_graph[graph_index] = artifact.decode(
        codes[graph_index : graph_index + 1], device=device
    ).to(device)[0]
    patched_graph[graph_index] = artifact.decode(modified_codes, device=device).to(
        device
    )[0]
    modified_logits = model(
        data.x,
        data.edge_index,
        batch=getattr(data, "batch", None),
        patch_graph=patched_graph,
    )
    original_prediction = _prediction_record(logits, graph_index)
    baseline_logits = model(
        data.x,
        data.edge_index,
        batch=getattr(data, "batch", None),
        patch_graph=baseline_graph,
    )
    original = _prediction_record(
        baseline_logits, graph_index, original_prediction["reference_class"]
    )
    modified = _prediction_record(
        modified_logits, graph_index, original["reference_class"]
    )
    return {
        "task": "graph",
        "graph_index": graph_index,
        "feature_ids": [int(fid) for fid in selected],
        "original_prediction": original_prediction,
        "unablated_reconstruction": original,
        "modified": modified,
        **_comparison(original, modified),
    }
