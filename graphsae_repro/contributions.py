"""Model-specific source-node contribution decompositions for SAE-GNNExplainer."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import k_hop_subgraph

from .backbones import GNNBackbone, LastLayerCapture


def _sum_aggregate(edge_index: Tensor, source_values: Tensor, num_nodes: int) -> Tensor:
    source, target = edge_index
    output = torch.zeros(
        num_nodes,
        source_values.size(-1),
        dtype=source_values.dtype,
        device=source_values.device,
    )
    output.index_add_(0, target, source_values[source])
    return output


def _gcn_layer(conv: GCNConv, contribution: Tensor, edge_index: Tensor) -> Tensor:
    normalized_edges, edge_weight = gcn_norm(
        edge_index,
        None,
        contribution.size(0),
        conv.improved,
        conv.add_self_loops,
        conv.flow,
        contribution.dtype,
    )
    source, target = normalized_edges
    transformed = F.linear(contribution, conv.lin.weight, bias=None)
    messages = transformed[source] * edge_weight.view(-1, 1)
    output = torch.zeros_like(transformed)
    output.index_add_(0, target, messages)
    return output


def _sage_layer(conv: SAGEConv, contribution: Tensor, edge_index: Tensor) -> Tensor:
    source, target = edge_index
    neighbor_sum = torch.zeros_like(contribution)
    neighbor_sum.index_add_(0, target, contribution[source])
    degree = (
        torch.bincount(target, minlength=contribution.size(0))
        .to(contribution.dtype)
        .clamp_min(1)
    )
    neighbor_mean = neighbor_sum / degree.view(-1, 1)
    output = F.linear(neighbor_mean, conv.lin_l.weight, bias=None)
    if conv.root_weight and conv.lin_r is not None:
        output = output + F.linear(contribution, conv.lin_r.weight, bias=None)
    return output


def _gin_aggregate(conv: GINConv, values: Tensor, edge_index: Tensor) -> Tensor:
    return (
        _sum_aggregate(edge_index, values, values.size(0)) + (1.0 + conv.eps) * values
    )


def _gin_layer(
    conv: GINConv,
    contribution: Tensor,
    original_input: Tensor,
    edge_index: Tensor,
) -> Tensor:
    base = _gin_aggregate(conv, original_input, edge_index)
    tangent = _gin_aggregate(conv, contribution, edge_index)
    _, output = torch.autograd.functional.jvp(
        conv.nn,
        (base.detach().requires_grad_(True),),
        (tangent,),
        create_graph=torch.is_grad_enabled(),
        strict=False,
    )
    return output


def _gat_projection(conv: GATConv, values: Tensor) -> Tensor:
    linear = conv.lin if conv.lin is not None else conv.lin_src
    if linear is None:
        raise NotImplementedError(
            "The GAT contribution rule requires a homogeneous linear projection"
        )
    projected = F.linear(values, linear.weight, bias=None)
    return projected.view(-1, conv.heads, conv.out_channels)


def _gat_layer(
    conv: GATConv,
    contribution: Tensor,
    attention_edge_index: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    source, target = attention_edge_index
    source_values = (
        contribution if isinstance(contribution, Tensor) else contribution[0]
    )
    target_values = (
        contribution if isinstance(contribution, Tensor) else contribution[1]
    )
    if target_values is None:
        raise ValueError(
            "Bipartite GAT source attribution requires destination features"
        )
    projected = _gat_projection(conv, source_values)
    alpha = attention_weights
    if alpha.ndim == 1:
        alpha = alpha.view(-1, 1)
    messages = projected[source] * alpha.to(projected.dtype).unsqueeze(-1)
    output_heads = torch.zeros(
        target_values.size(0),
        conv.heads,
        conv.out_channels,
        dtype=projected.dtype,
        device=projected.device,
    )
    output_heads.index_add_(0, target, messages)
    output = (
        output_heads.reshape(target_values.size(0), -1)
        if conv.concat
        else output_heads.mean(dim=1)
    )
    residual = getattr(conv, "res", None)
    if residual is not None:
        output = output + F.linear(target_values, residual.weight, bias=None)
    return output


def _propagate_one_source(
    model: GNNBackbone,
    data,
    capture: LastLayerCapture,
    source_node: int,
) -> Tensor:
    """Propagate one source term through the observed local GNN linearization."""
    device = data.x.device
    contribution = torch.zeros_like(data.x, device=device)
    contribution[int(source_node)] = data.x[int(source_node)]

    return _linearized_forward(model, data, capture, contribution)


def _linearized_forward(model, data, capture, contribution):
    device = contribution.device
    for conv, layer in zip(model.convs, capture.layers):
        if isinstance(conv, GCNConv):
            pre_contribution = _gcn_layer(conv, contribution, data.edge_index)
        elif isinstance(conv, SAGEConv):
            pre_contribution = _sage_layer(conv, contribution, data.edge_index)
        elif isinstance(conv, GINConv):
            pre_contribution = _gin_layer(
                conv, contribution, layer.input.to(device), data.edge_index
            )
        elif isinstance(conv, GATConv):
            if layer.attention_edge_index is None or layer.attention_weights is None:
                raise RuntimeError(
                    "GAT source decomposition requires captured attention coefficients"
                )
            pre_contribution = _gat_layer(
                conv,
                contribution,
                layer.attention_edge_index.to(device),
                layer.attention_weights.to(device),
            )
        else:
            raise NotImplementedError(
                f"Unsupported message-passing layer: {type(conv).__name__}"
            )

        # The paper fixes the ReLU gates observed on the frozen forward pass.
        gate = (layer.pre_activation.to(device) > 0).to(pre_contribution.dtype)
        contribution = pre_contribution * gate

    return contribution


def source_scores(
    model,
    data,
    direction: Tensor,
    *,
    target_index: int = 0,
    device: torch.device | str = "cpu",
) -> tuple[list[int], Tensor]:
    """Project local source terms in one reverse pass, holding GAT attention fixed.

    For the local linear map L, <L(e_u x_u), a> equals
    <x_u, [L^T a]_u>. This avoids storing a sources-by-nodes Jacobian.
    The original full graph supplies normalization degrees and boundary messages.
    """
    device = torch.device(device)
    model = model.to(device).eval()
    data = data.to(device)
    with torch.no_grad():
        _, capture = model(
            data.x, data.edge_index, batch=getattr(data, "batch", None), capture=True
        )
    with torch.enable_grad():
        tangent = torch.zeros_like(data.x, requires_grad=True)
        output = _linearized_forward(model, data, capture, tangent)
        if model.task == "graph":
            nodes = torch.where(capture.batch == target_index)[0]
            representation = output[nodes].mean(0)
            candidates = nodes.tolist()
        else:
            representation = output[target_index]
            nodes, _, _, _ = k_hop_subgraph(
                target_index,
                model.num_layers,
                data.edge_index,
                num_nodes=data.num_nodes,
            )
            candidates = [u for u in nodes.tolist() if u != target_index]
        (gradient,) = torch.autograd.grad(
            (representation * direction.to(device)).sum(), tangent
        )
        scores = (gradient * data.x).sum(-1).detach().cpu()
    candidates = sorted(candidates)
    return candidates, scores[candidates]


def node_source_contributions(
    model: GNNBackbone,
    data,
    *,
    target_node: int,
    candidate_nodes: Sequence[int] | None = None,
    exclude_target: bool = True,
    device: torch.device | str = "cpu",
) -> tuple[list[int], Tensor]:
    """Return ``c_{u;t} = r_{u->t}^{(L)}`` for candidate source nodes."""
    device = torch.device(device)
    model = model.to(device).eval()
    data = data.to(device)
    _, capture = model(
        data.x, data.edge_index, batch=getattr(data, "batch", None), capture=True
    )
    target_node = int(target_node)

    if candidate_nodes is None:
        nodes, _, _, _ = k_hop_subgraph(
            target_node,
            model.num_layers,
            data.edge_index,
            relabel_nodes=False,
        )
        candidates = [int(node) for node in nodes.detach().cpu().tolist()]
    else:
        candidates = [int(node) for node in candidate_nodes]
    if exclude_target:
        candidates = [node for node in candidates if node != target_node]

    rows = [
        _propagate_one_source(model, data, capture, source_node)[target_node]
        .detach()
        .cpu()
        for source_node in candidates
    ]
    contributions = (
        torch.stack(rows) if rows else torch.empty((0, capture.node.size(1)))
    )
    return candidates, contributions


def graph_source_contributions(
    model: GNNBackbone,
    data,
    *,
    graph_index: int = 0,
    candidate_nodes: Sequence[int] | None = None,
    device: torch.device | str = "cpu",
) -> tuple[list[int], Tensor]:
    """Return ``c_{u;G} = |V|^{-1} sum_v r_{u->v}^{(L)}`` for one graph."""
    device = torch.device(device)
    model = model.to(device).eval()
    data = data.to(device)
    _, capture = model(
        data.x, data.edge_index, batch=getattr(data, "batch", None), capture=True
    )
    if capture.graph is None:
        raise ValueError(
            "graph_source_contributions requires a graph-classification backbone"
        )

    graph_index = int(graph_index)
    batch = capture.batch
    if batch is None:
        batch = torch.zeros(capture.node.size(0), dtype=torch.long, device=device)
    graph_nodes = torch.nonzero(batch == graph_index, as_tuple=False).view(-1)
    graph_node_ids = [int(node) for node in graph_nodes.detach().cpu().tolist()]
    candidates = (
        graph_node_ids
        if candidate_nodes is None
        else [int(node) for node in candidate_nodes]
    )
    invalid = sorted(set(candidates) - set(graph_node_ids))
    if invalid:
        raise ValueError(
            f"Candidate nodes do not belong to graph {graph_index}: {invalid}"
        )

    rows = [
        _propagate_one_source(model, data, capture, source_node)[graph_nodes]
        .mean(dim=0)
        .detach()
        .cpu()
        for source_node in candidates
    ]
    contributions = (
        torch.stack(rows) if rows else torch.empty((0, capture.node.size(1)))
    )
    return candidates, contributions
