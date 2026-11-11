"""Frozen Cora predictor architectures used by the bundled checkpoints."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, remove_self_loops, degree, softmax


@dataclass
class LayerCapture:
    """One captured GNN layer.

    ``node`` is the post-nonlinearity hidden state used by node/subgraph SAE
    packs.  ``message`` and ``message_edge_index`` are optional because custom
    models may only expose node activations.
    """

    node: Tensor
    pre_activation: Tensor
    message: Optional[Tensor]
    message_edge_index: Optional[Tensor]


class CapturingMessagePassing(MessagePassing):
    def __init__(self, aggr: str = "add", node_dim: int = -2):
        super().__init__(aggr=aggr, node_dim=node_dim)
        self.last_messages: Optional[Tensor] = None
        self.last_message_edge_index: Optional[Tensor] = None
        self._message_override: Optional[Tensor] = None

    def clear_capture(self) -> None:
        self.last_messages = None
        self.last_message_edge_index = None
        self._message_override = None

    def set_message_override(self, messages: Optional[Tensor]) -> None:
        self._message_override = messages


class CapturingGCNConv(CapturingMessagePassing):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(
        self, x: Tensor, edge_index: Tensor, message_override: Optional[Tensor] = None
    ) -> Tensor:
        self.clear_capture()
        self.set_message_override(message_override)

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype).clamp(min=1)
        norm = deg.pow(-0.5)[row] * deg.pow(-0.5)[col]

        x = self.lin(x)
        self.last_message_edge_index = edge_index.detach()
        out = self.propagate(edge_index, x=x, norm=norm)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        msg = norm.view(-1, 1) * x_j
        if self._message_override is not None:
            if self._message_override.shape != msg.shape:
                raise ValueError(
                    f"message_override shape {tuple(self._message_override.shape)} != {tuple(msg.shape)}"
                )
            msg = self._message_override
        self.last_messages = msg.detach()
        return msg


class CapturingSAGEConv(CapturingMessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="mean")
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=True)

    def forward(
        self, x: Tensor, edge_index: Tensor, message_override: Optional[Tensor] = None
    ) -> Tensor:
        self.clear_capture()
        self.set_message_override(message_override)
        self.last_message_edge_index = edge_index.detach()
        neigh = self.propagate(edge_index, x=x)
        return self.lin_self(x) + self.lin_neigh(neigh)

    def message(self, x_j: Tensor) -> Tensor:
        captured = self.lin_neigh(x_j)
        msg_for_aggregation = x_j
        if self._message_override is not None:
            captured = self._message_override
            if self._message_override.shape[-1] == x_j.shape[-1]:
                msg_for_aggregation = self._message_override
        self.last_messages = captured.detach()
        return msg_for_aggregation


class CapturingGINConv(CapturingMessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        eps: float = 0.0,
        train_eps: bool = True,
    ):
        super().__init__(aggr="add")
        self.eps = nn.Parameter(
            torch.tensor([eps], dtype=torch.float32), requires_grad=train_eps
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(
        self, x: Tensor, edge_index: Tensor, message_override: Optional[Tensor] = None
    ) -> Tensor:
        self.clear_capture()
        self.set_message_override(message_override)
        self.last_message_edge_index = edge_index.detach()
        out = self.propagate(edge_index, x=x)
        return self.mlp((1 + self.eps) * x + out)

    def message(self, x_j: Tensor) -> Tensor:
        msg = x_j
        if self._message_override is not None:
            if self._message_override.shape != x_j.shape:
                raise ValueError(
                    f"message_override shape {tuple(self._message_override.shape)} != {tuple(x_j.shape)}"
                )
            msg = self._message_override
        self.last_messages = msg.detach()
        return msg


class CapturingGATConv(CapturingMessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(heads, out_channels))
        bias_dim = heads * out_channels if concat else out_channels
        self.bias = nn.Parameter(torch.zeros(bias_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(
        self, x: Tensor, edge_index: Tensor, message_override: Optional[Tensor] = None
    ) -> Tensor:
        self.clear_capture()
        self.set_message_override(message_override)

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        h = self.lin(x).view(-1, self.heads, self.out_channels)
        alpha_src = (h * self.att_src).sum(dim=-1)
        alpha_dst = (h * self.att_dst).sum(dim=-1)

        self.last_message_edge_index = edge_index.detach()
        out = self.propagate(edge_index, x=h, alpha=(alpha_src, alpha_dst))
        if self.concat:
            out = out.reshape(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        return out + self.bias

    def message(
        self,
        x_j: Tensor,
        alpha_j: Tensor,
        alpha_i: Tensor,
        index: Tensor,
        ptr=None,
        size_i=None,
    ) -> Tensor:
        alpha = F.leaky_relu(alpha_j + alpha_i, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        msg = x_j * alpha.unsqueeze(-1)
        msg_flat = msg.reshape(msg.size(0), -1)
        if self._message_override is not None:
            if self._message_override.shape != msg_flat.shape:
                raise ValueError(
                    f"message_override shape {tuple(self._message_override.shape)} != {tuple(msg_flat.shape)}"
                )
            msg_flat = self._message_override
            msg = msg_flat.view(-1, self.heads, self.out_channels)
        self.last_messages = msg_flat.detach()
        return msg


class CapturingGNN(nn.Module):
    def __init__(
        self,
        model_name: str,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        task: str = "node",
        gat_heads: int = 4,
    ):
        super().__init__()
        self.model_name = model_name.lower()
        self.task = task
        self.dropout = float(dropout)
        self.num_layers = int(num_layers)
        self.hidden_channels = int(hidden_channels)
        self.convs = nn.ModuleList()

        for layer in range(self.num_layers):
            in_dim = in_channels if layer == 0 else hidden_channels
            if self.model_name == "gcn":
                conv = CapturingGCNConv(in_dim, hidden_channels)
            elif self.model_name in {"sage", "graphsage"}:
                conv = CapturingSAGEConv(in_dim, hidden_channels)
            elif self.model_name == "gin":
                conv = CapturingGINConv(in_dim, hidden_channels)
            elif self.model_name == "gat":
                if hidden_channels % gat_heads != 0:
                    raise ValueError(
                        "hidden_channels must be divisible by gat_heads for CapturingGATConv."
                    )
                conv = CapturingGATConv(
                    in_channels=in_dim,
                    out_channels=hidden_channels // gat_heads,
                    heads=gat_heads,
                    concat=True,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"Unknown model_name={model_name!r}")
            self.convs.append(conv)

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
        capture: bool = False,
        patch_layer: Optional[int] = None,
        patch_node_activation: Optional[Tensor] = None,
        patch_message_activation: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[int, LayerCapture]]:
        captures: Dict[int, LayerCapture] = {}
        h = x

        for layer_idx, conv in enumerate(self.convs):
            msg_override = (
                patch_message_activation if patch_layer == layer_idx else None
            )
            pre = conv(h, edge_index, message_override=msg_override)
            h = F.relu(pre)
            if patch_layer == layer_idx and patch_node_activation is not None:
                if patch_node_activation.shape != h.shape:
                    raise ValueError(
                        f"patch_node_activation shape {tuple(patch_node_activation.shape)} != {tuple(h.shape)}"
                    )
                h = patch_node_activation
            h = F.dropout(h, p=self.dropout, training=self.training)

            if capture:
                captures[layer_idx] = LayerCapture(
                    node=h.detach(),
                    pre_activation=pre.detach(),
                    message=conv.last_messages,
                    message_edge_index=conv.last_message_edge_index,
                )

        if self.task == "graph":
            if batch is None:
                batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            h = global_mean_pool(h, batch)

        logits = self.classifier(h)
        return logits, captures

    @torch.no_grad()
    def predict(self, data) -> Tensor:
        self.eval()
        logits, _ = self(
            data.x,
            data.edge_index,
            batch=getattr(data, "batch", None),
            capture=False,
        )
        return logits
