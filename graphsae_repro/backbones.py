"""GNN backbones with access to final-layer node and graph representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv, global_mean_pool


MODEL_NAMES = ("GCN", "GAT", "GIN", "GraphSAGE")
MODEL_PRESETS = {
    "GCN": {"hidden_channels": 128, "num_layers": 2},
    "GIN": {"hidden_channels": 128, "num_layers": 2},
    "GAT": {"hidden_channels": 256, "num_layers": 3},
    "GraphSAGE": {"hidden_channels": 256, "num_layers": 3},
}
_ALIASES = {
    "gcn": "GCN",
    "gat": "GAT",
    "gin": "GIN",
    "sage": "GraphSAGE",
    "graphsage": "GraphSAGE",
}


@dataclass
class LayerCapture:
    """Quantities that define the observed local linear region of one GNN layer."""

    input: Tensor
    pre_activation: Tensor
    node: Tensor
    attention_edge_index: Tensor | None = None
    attention_weights: Tensor | None = None


@dataclass
class LastLayerCapture:
    """Representations and local-forward information used by GraphSAE."""

    node: Tensor
    graph: Tensor | None
    layers: tuple[LayerCapture, ...]
    batch: Tensor | None


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


class GNNBackbone(nn.Module):
    """Message-passing classifier with a common final-representation interface."""

    def __init__(
        self,
        model_name: str,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        num_layers: int,
        task: str,
        dropout: float,
        gat_heads: int,
    ) -> None:
        super().__init__()
        if task not in {"node", "graph"}:
            raise ValueError("task must be 'node' or 'graph'")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        self.model_name = canonical_model_name(model_name)
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.task = task
        self.dropout = float(dropout)
        self.gat_heads = int(gat_heads)
        self.convs = nn.ModuleList()

        for layer_index in range(self.num_layers):
            input_dim = int(in_channels) if layer_index == 0 else self.hidden_channels
            if self.model_name == "GCN":
                conv = GCNConv(input_dim, self.hidden_channels)
            elif self.model_name == "GraphSAGE":
                conv = SAGEConv(input_dim, self.hidden_channels)
            elif self.model_name == "GIN":
                conv = GINConv(_gin_mlp(input_dim, self.hidden_channels))
            else:
                if self.hidden_channels % self.gat_heads:
                    raise ValueError("hidden_channels must be divisible by gat_heads")
                conv = GATConv(
                    input_dim,
                    self.hidden_channels // self.gat_heads,
                    heads=self.gat_heads,
                    concat=True,
                    dropout=self.dropout,
                )
            self.convs.append(conv)

        self.classifier = nn.Linear(self.hidden_channels, int(out_channels))

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor | None = None,
        *,
        capture: bool = False,
        patch_last: Tensor | None = None,
        patch_graph: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, LastLayerCapture]:
        layer_captures: list[LayerCapture] = []
        h = x

        for conv in self.convs:
            layer_input = h
            attention_edge_index = None
            attention_weights = None
            if capture and isinstance(conv, GATConv):
                pre_activation, attention = conv(
                    h, edge_index, return_attention_weights=True
                )
                attention_edge_index, attention_weights = attention
            else:
                pre_activation = conv(h, edge_index)

            h = F.relu(pre_activation)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if capture:
                layer_captures.append(
                    LayerCapture(
                        input=layer_input.detach(),
                        pre_activation=pre_activation.detach(),
                        node=h.detach(),
                        attention_edge_index=(
                            None
                            if attention_edge_index is None
                            else attention_edge_index.detach()
                        ),
                        attention_weights=(
                            None
                            if attention_weights is None
                            else attention_weights.detach()
                        ),
                    )
                )

        if patch_last is not None:
            if patch_last.shape != h.shape:
                raise ValueError(
                    f"patch_last has shape {tuple(patch_last.shape)}, expected {tuple(h.shape)}"
                )
            h = patch_last

        graph_h = None
        resolved_batch = batch
        if self.task == "graph":
            if resolved_batch is None:
                resolved_batch = torch.zeros(
                    h.size(0), dtype=torch.long, device=h.device
                )
            graph_h = global_mean_pool(h, resolved_batch)
            if patch_graph is not None:
                if patch_graph.shape != graph_h.shape:
                    raise ValueError(
                        f"patch_graph has shape {tuple(patch_graph.shape)}, expected {tuple(graph_h.shape)}"
                    )
                graph_h = patch_graph
            logits = self.classifier(graph_h)
        else:
            if patch_graph is not None:
                raise ValueError("patch_graph is only valid for graph classification")
            logits = self.classifier(h)

        if not capture:
            return logits
        return logits, LastLayerCapture(
            node=h,
            graph=graph_h,
            layers=tuple(layer_captures),
            batch=None if resolved_batch is None else resolved_batch.detach(),
        )


def build_backbone(
    model_name: str,
    in_channels: int,
    out_channels: int,
    *,
    task: str = "node",
    hidden_channels: int | None = None,
    num_layers: int | None = None,
    dropout: float = 0.5,
    gat_heads: int = 4,
) -> GNNBackbone:
    name = canonical_model_name(model_name)
    preset = MODEL_PRESETS[name]
    return GNNBackbone(
        name,
        in_channels,
        int(
            hidden_channels
            if hidden_channels is not None
            else preset["hidden_channels"]
        ),
        out_channels,
        num_layers=int(num_layers if num_layers is not None else preset["num_layers"]),
        task=task,
        dropout=dropout,
        gat_heads=gat_heads,
    )


def logits_only(model: nn.Module, data) -> Tensor:
    output = model(data.x, data.edge_index, batch=getattr(data, "batch", None))
    return output[0] if isinstance(output, tuple) else output


def train_full_batch_node(
    model: nn.Module,
    data,
    *,
    epochs: int = 300,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    device: torch.device | str = "cpu",
) -> list[dict]:
    """Train a node classifier on the dataset-provided training split."""
    device = torch.device(device)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict] = []

    best_loss, best_state = float("inf"), None

    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        logits = logits_only(model, data)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation = logits_only(model, data)
            validation_loss = float(
                F.cross_entropy(validation[data.val_mask], data.y[data.val_mask])
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == int(epochs):
            model.eval()
            with torch.no_grad():
                logits = logits_only(model, data)
                record = {"epoch": epoch + 1, "loss": float(loss.item())}
                for split in ("train", "val"):
                    mask = getattr(data, f"{split}_mask", None)
                    if mask is not None and int(mask.sum()) > 0:
                        accuracy = (
                            (logits[mask].argmax(dim=-1) == data.y[mask]).float().mean()
                        )
                        record[f"{split}_accuracy"] = float(accuracy.item())
                history.append(record)

    if best_state is not None:
        model.load_state_dict(best_state)
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
    validation_loader: Iterable | None = None,
) -> list[dict]:
    """Train a graph classifier with mean-pooled final-layer representations."""
    device = torch.device(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict] = []
    best_loss, best_state = float("inf"), None

    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0.0
        total_graphs = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = logits_only(model, batch)
            labels = batch.y.view(-1)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(labels.numel())
            total_graphs += int(labels.numel())
        record = {"epoch": epoch + 1, "loss": total_loss / max(1, total_graphs)}
        if validation_loader is not None:
            model.eval()
            value, count = 0.0, 0
            with torch.no_grad():
                for batch in validation_loader:
                    batch = batch.to(device)
                    logits = logits_only(model, batch)
                    value += float(
                        F.cross_entropy(logits, batch.y.view(-1), reduction="sum")
                    )
                    count += batch.y.numel()
            if not count:
                raise ValueError("Validation loader is empty")
            record["val_loss"] = value / count
            if record["val_loss"] < best_loss:
                best_loss = record["val_loss"]
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
        history.append(record)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return history
