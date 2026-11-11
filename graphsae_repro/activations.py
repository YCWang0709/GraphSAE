"""Last-layer activation extraction for node and graph classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor


@dataclass
class ActivationPack:
    z: Tensor
    metadata: list[dict[str, Any]]
    task: str
    source: str = ""
    info: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "z": self.z.detach().cpu(),
                "metadata": self.metadata,
                "task": self.task,
                "source": self.source,
                "info": self.info,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ActivationPack":
        obj = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls(
            z=obj["z"],
            metadata=obj["metadata"],
            task=obj["task"],
            source=obj.get("source", ""),
            info=obj.get("info", {}),
        )


@torch.no_grad()
def extract_node_activations(
    model,
    data,
    *,
    split: str | None = None,
    task_type: str = "classification",
    device: torch.device | str = "cpu",
) -> ActivationPack:
    """Extract one SAE input row per node from the final GNN layer."""
    device = torch.device(device)
    model = model.to(device).eval()
    data = data.to(device)
    logits, capture = model(
        data.x, data.edge_index, batch=getattr(data, "batch", None), capture=True
    )
    indices = torch.arange(capture.node.size(0), device=device)
    if split is not None:
        mask = getattr(data, f"{split}_mask", None)
        if mask is None:
            raise ValueError(f"Data has no {split}_mask")
        indices = torch.nonzero(mask, as_tuple=False).view(-1)
    labels = getattr(data, "y", None)
    groups = None
    if task_type == "regression":
        from .grounding import city_roads_groups

        groups = city_roads_groups(data.edge_index, data.num_nodes)
    metadata = []
    for node_id in indices.detach().cpu().tolist():
        label = None if labels is None else labels[node_id].item()
        row = {
            "unit_type": "node",
            "node_id": int(node_id),
            "label": label,
            "split": split,
        }
        if task_type == "regression":
            row["group"] = int(groups[node_id])
        metadata.append(row)
    return ActivationPack(
        z=capture.node[indices].detach().cpu(),
        metadata=metadata,
        task="node",
        source="last_message_passing_layer",
        info={
            "num_rows": int(indices.numel()),
            "hidden_dim": int(capture.node.size(1)),
            "split": split,
        },
    )


@torch.no_grad()
def extract_graph_activations(
    model,
    loader: Iterable,
    *,
    split: str | None = None,
    task_type: str = "classification",
    device: torch.device | str = "cpu",
) -> ActivationPack:
    """Extract one pooled final-layer activation per graph in a PyG loader."""
    device = torch.device(device)
    model = model.to(device).eval()
    rows: list[Tensor] = []
    metadata: list[dict[str, Any]] = []
    graph_offset = 0
    for batch in loader:
        batch = batch.to(device)
        logits, capture = model(
            batch.x, batch.edge_index, batch=batch.batch, capture=True
        )
        if capture.graph is None:
            raise ValueError(
                "extract_graph_activations requires a graph-classification backbone"
            )
        rows.append(capture.graph.detach().cpu())
        labels = batch.y.view(-1).detach().cpu()
        predictions = logits.argmax(dim=-1).detach().cpu()
        graph_ids = getattr(batch, "graph_id", None)
        for local_idx, (label, prediction) in enumerate(
            zip(labels.tolist(), predictions.tolist())
        ):
            graph_id = (
                graph_offset + local_idx
                if graph_ids is None
                else int(graph_ids[local_idx].item())
            )
            row = {
                "unit_type": "graph",
                "graph_id": graph_id,
                "label": label,
                "split": split,
            }
            if task_type == "regression":
                from .grounding import logarithmic_groups

                row["group"] = int(
                    logarithmic_groups([(batch.batch == local_idx).sum().item()])[0]
                )
            metadata.append(row)
        graph_offset += int(labels.numel())
    z = torch.cat(rows, dim=0) if rows else torch.empty((0, 0))
    return ActivationPack(
        z=z,
        metadata=metadata,
        task="graph",
        source="pooled_last_message_passing_layer",
        info={
            "num_rows": int(z.size(0)),
            "hidden_dim": int(z.size(1)) if z.ndim == 2 else 0,
            "split": split,
        },
    )
