"""Capture-compatible GCN, GAT, GIN, and GraphSAGE backbones.

The final message-passing activation is captured after its nonlinearity and
before the task readout.  This is the activation supplied to the SAE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv, global_mean_pool


MODEL_NAMES = ("GCN", "GAT", "GIN", "GraphSAGE")
_ALIASES = {
    "gcn": "GCN",
    "gat": "GAT",
    "gin": "GIN",
    "sage": "GraphSAGE",
    "graphsage": "GraphSAGE",
    "graph_sage": "GraphSAGE",
}


@dataclass
class LastLayerCapture:
    node: Tensor
    graph: Tensor | None


def canonical_model_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if key in _ALIASES:
        return _ALIASES[key]
    if name in MODEL_NAMES:
        return str(name)
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}.")


def _gin_mlp(in_channels: int, hidden_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_channels, hidden_channels),
        nn.ReLU(),
        nn.Linear(hidden_channels, hidden_channels),
    )


class _BaseBackbone(nn.Module):
    """Shared output handling for node and graph classification."""

    model_name: str

    def __init__(self, out_channels: int, task: str, dropout: float) -> None:
        super().__init__()
        if task not in {"node", "graph"}:
            raise ValueError("task must be 'node' or 'graph'")
        self.task = task
        self.dropout = float(dropout)
        self.classifier = nn.Linear(self.hidden_channels, out_channels)
        self.num_layers = 2

    def _finish(
        self,
        last_h: Tensor,
        batch: Tensor | None,
        capture: bool,
        patch_last: Tensor | None,
        patch_graph: Tensor | None,
    ) -> Tensor | tuple[Tensor, LastLayerCapture]:
        if patch_last is not None:
            if patch_last.shape != last_h.shape:
                raise ValueError(f"patch_last has shape {tuple(patch_last.shape)}, expected {tuple(last_h.shape)}")
            last_h = patch_last

        graph_h = None
        if self.task == "graph":
            if batch is None:
                batch = torch.zeros(last_h.size(0), dtype=torch.long, device=last_h.device)
            graph_h = global_mean_pool(last_h, batch)
            if patch_graph is not None:
                if patch_graph.shape != graph_h.shape:
                    raise ValueError(f"patch_graph has shape {tuple(patch_graph.shape)}, expected {tuple(graph_h.shape)}")
                graph_h = patch_graph
            logits = self.classifier(graph_h)
        else:
            if patch_graph is not None:
                raise ValueError("patch_graph is only valid for graph classification")
            logits = self.classifier(last_h)

        if capture:
            return logits, LastLayerCapture(node=last_h.detach(), graph=None if graph_h is None else graph_h.detach())
        return logits


class GCNBackbone(_BaseBackbone):
    model_name = "GCN"

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, task: str, dropout: float) -> None:
        self.hidden_channels = int(hidden_channels)
        super().__init__(out_channels, task, dropout)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor | None = None, *, capture: bool = False, patch_last: Tensor | None = None, patch_graph: Tensor | None = None):
        h = F.dropout(F.relu(self.conv1(x, edge_index)), p=self.dropout, training=self.training)
        h = F.dropout(F.relu(self.conv2(h, edge_index)), p=self.dropout, training=self.training)
        return self._finish(h, batch, capture, patch_last, patch_graph)


class GraphSAGEBackbone(_BaseBackbone):
    model_name = "GraphSAGE"

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, task: str, dropout: float) -> None:
        self.hidden_channels = int(hidden_channels)
        super().__init__(out_channels, task, dropout)
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor | None = None, *, capture: bool = False, patch_last: Tensor | None = None, patch_graph: Tensor | None = None):
        h = F.dropout(F.relu(self.conv1(x, edge_index)), p=self.dropout, training=self.training)
        h = F.dropout(F.relu(self.conv2(h, edge_index)), p=self.dropout, training=self.training)
        return self._finish(h, batch, capture, patch_last, patch_graph)


class GINBackbone(_BaseBackbone):
    model_name = "GIN"

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, task: str, dropout: float) -> None:
        self.hidden_channels = int(hidden_channels)
        super().__init__(out_channels, task, dropout)
        self.conv1 = GINConv(_gin_mlp(in_channels, hidden_channels))
        self.conv2 = GINConv(_gin_mlp(hidden_channels, hidden_channels))

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor | None = None, *, capture: bool = False, patch_last: Tensor | None = None, patch_graph: Tensor | None = None):
        h = F.dropout(F.relu(self.conv1(x, edge_index)), p=self.dropout, training=self.training)
        h = F.dropout(F.relu(self.conv2(h, edge_index)), p=self.dropout, training=self.training)
        return self._finish(h, batch, capture, patch_last, patch_graph)


class GATBackbone(_BaseBackbone):
    model_name = "GAT"

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, task: str, dropout: float, heads: int) -> None:
        self.hidden_channels = int(hidden_channels)
        super().__init__(out_channels, task, dropout)
        self.heads = int(heads)
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # The final GAT layer averages heads, producing the common SAE input dimension.
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor | None = None, *, capture: bool = False, patch_last: Tensor | None = None, patch_graph: Tensor | None = None):
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = F.dropout(F.elu(self.conv1(h, edge_index)), p=self.dropout, training=self.training)
        h = F.dropout(F.elu(self.conv2(h, edge_index)), p=self.dropout, training=self.training)
        return self._finish(h, batch, capture, patch_last, patch_graph)


def build_backbone(
    model_name: str,
    in_channels: int,
    out_channels: int,
    *,
    task: str = "node",
    hidden_channels: int = 64,
    dropout: float = 0.3,
    gat_heads: int = 4,
) -> _BaseBackbone:
    name = canonical_model_name(model_name)
    common = dict(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, task=task, dropout=dropout)
    if name == "GCN":
        return GCNBackbone(**common)
    if name == "GAT":
        return GATBackbone(**common, heads=gat_heads)
    if name == "GIN":
        return GINBackbone(**common)
    return GraphSAGEBackbone(**common)


def logits_only(model: nn.Module, data) -> Tensor:
    out = model(data.x, data.edge_index, batch=getattr(data, "batch", None))
    return out[0] if isinstance(out, tuple) else out


def train_full_batch_node(
    model: nn.Module,
    data,
    *,
    epochs: int = 300,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    device: torch.device | str = "cpu",
) -> list[dict]:
    """Train a node-classification backbone using ``train_mask`` on one PyG Data object."""
    device = torch.device(device)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict] = []
    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        logits = logits_only(model, data)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == int(epochs):
            model.eval()
            with torch.no_grad():
                logits = logits_only(model, data)
                record = {"epoch": epoch + 1, "loss": float(loss.item())}
                for split in ("train", "val", "test"):
                    mask = getattr(data, f"{split}_mask", None)
                    if mask is not None and int(mask.sum()) > 0:
                        record[f"{split}_accuracy"] = float((logits[mask].argmax(dim=-1) == data.y[mask]).float().mean().item())
                history.append(record)
    model.eval()
    return history


def train_graph_classifier(
    model: nn.Module,
    train_loader: Iterable,
    *,
    epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    device: torch.device | str = "cpu",
) -> list[dict]:
    """Minimal mini-batch trainer for graph-classification backbones."""
    device = torch.device(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict] = []
    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0.0
        total_graphs = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = logits_only(model, batch)
            y = batch.y.view(-1)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(y.numel())
            total_graphs += int(y.numel())
        history.append({"epoch": epoch + 1, "loss": total_loss / max(total_graphs, 1)})
    model.eval()
    return history
