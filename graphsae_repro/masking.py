"""Prediction-space masking experiments for GraphSAE explanations."""

from __future__ import annotations

from copy import copy
from typing import Iterable

import torch
from torch import Tensor
import torch.nn.functional as F


def clone_with_node_mask(
    data, node_ids: Iterable[int], *, mode: str = "both", fill_value: float = 0.0
):
    """Mask node features, incident edges, or both without changing the original Data object."""
    mode = str(mode).lower()
    if mode not in {"features", "edges", "both"}:
        raise ValueError("mode must be 'features', 'edges', or 'both'")
    out = data.clone() if hasattr(data, "clone") else copy(data)
    ids = torch.tensor(
        sorted(set(map(int, node_ids))), dtype=torch.long, device=out.x.device
    )
    if ids.numel() == 0:
        return out
    if int(ids.min()) < 0 or int(ids.max()) >= out.x.size(0):
        raise IndexError("node_ids contain an invalid node index")
    if mode in {"features", "both"}:
        out.x[ids] = float(fill_value)
    if mode in {"edges", "both"}:
        keep = ~(
            torch.isin(out.edge_index[0], ids) | torch.isin(out.edge_index[1], ids)
        )
        edge_count = out.edge_index.size(1)
        for key in list(out.keys()):
            value = out[key]
            if (
                key != "edge_index"
                and isinstance(value, Tensor)
                and out.is_edge_attr(key)
                and value.ndim
                and value.shape[0] == edge_count
            ):
                out[key] = value[keep]
        out.edge_index = out.edge_index[:, keep]
    return out


def _state(logits: Tensor, *, index: int, reference_class: int | None = None) -> dict:
    row = logits.detach().cpu()[int(index)]
    if row.numel() == 1:
        row = torch.cat([torch.zeros_like(row.reshape(-1)), row.reshape(-1)])
    probs = F.softmax(row, dim=-1)
    prediction = int(row.argmax().item())
    reference = prediction if reference_class is None else int(reference_class)
    return {
        "prediction": prediction,
        "reference_class": reference,
        "reference_logit": float(row[reference].item()),
        "reference_probability": float(probs[reference].item()),
    }


@torch.no_grad()
def run_mask_experiment(
    model,
    data,
    node_ids: Iterable[int],
    *,
    task: str,
    target_index: int = 0,
    mode: str = "both",
    device: torch.device | str = "cpu",
) -> dict:
    """Evaluate the reference prediction before and after masking explanation nodes."""
    if task not in {"node", "graph"}:
        raise ValueError("task must be 'node' or 'graph'")
    device = torch.device(device)
    node_ids = list(node_ids)
    if task == "node" and int(target_index) in node_ids:
        raise ValueError("The target node is excluded from removable sources")
    model = model.to(device).eval()
    original_data = data.to(device)
    masked_data = clone_with_node_mask(original_data, node_ids, mode=mode)
    original_logits = model(
        original_data.x,
        original_data.edge_index,
        batch=getattr(original_data, "batch", None),
    )
    masked_logits = model(
        masked_data.x, masked_data.edge_index, batch=getattr(masked_data, "batch", None)
    )
    original = _state(original_logits, index=target_index)
    masked = _state(
        masked_logits, index=target_index, reference_class=original["reference_class"]
    )
    return {
        "task": task,
        "target_index": int(target_index),
        "mask_mode": mode,
        "masked_node_ids": sorted(set(map(int, node_ids))),
        "num_masked_nodes": len(set(map(int, node_ids))),
        "original": original,
        "masked": masked,
        "prediction_changed": bool(original["prediction"] != masked["prediction"]),
        "delta_reference_logit": float(
            masked["reference_logit"] - original["reference_logit"]
        ),
        "delta_reference_probability": float(
            masked["reference_probability"] - original["reference_probability"]
        ),
    }
