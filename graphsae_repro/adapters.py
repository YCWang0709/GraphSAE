"""Frozen predictor interface for custom, relation-aware, and heterogeneous GNNs."""

from contextlib import contextmanager
import types
import torch
from torch_geometric.nn import GATConv
from .contributions import _gat_layer


@contextmanager
def frozen_attention(model, forward):
    """Retain observed GAT coefficients while differentiating source features.

    Each GAT module must be called once per forward. Edge-conditioned
    GATs require a model-specific adapter; they are rejected explicitly.
    The ordinary predictor is restored before any deletion evaluation.
    """
    traces, handles, originals = {}, [], {}

    def record(module, args, kwargs):
        if module in traces:
            raise ValueError(
                "Shared/recurrent GAT modules require a custom frozen-attention adapter"
            )
        x = args[0] if args else kwargs["x"]
        edge = args[1] if len(args) > 1 else kwargs["edge_index"]
        if module.edge_dim is not None:
            raise ValueError("Edge-conditioned GAT requires a custom adapter")
        values = (
            x.detach()
            if isinstance(x, torch.Tensor)
            else tuple(v.detach() if v is not None else None for v in x)
        )
        traces[module] = values, edge

    try:
        for module in model.modules():
            if isinstance(module, GATConv):
                handles.append(
                    module.register_forward_pre_hook(record, with_kwargs=True)
                )
        with torch.no_grad():
            forward()
        for handle in handles:
            handle.remove()
        handles.clear()
        for module, (x, edge) in traces.items():
            with torch.no_grad():
                _, (observed_edge, alpha) = module(
                    x, edge, return_attention_weights=True
                )
            originals[module] = module.forward

            def fixed(
                self,
                x,
                edge_index,
                edge_attr=None,
                size=None,
                return_attention_weights=None,
                _edge=observed_edge,
                _alpha=alpha,
            ):
                result = _gat_layer(self, x, _edge, _alpha)
                if self.bias is not None:
                    result = result + self.bias
                return (
                    (result, (_edge, _alpha))
                    if isinstance(return_attention_weights, bool)
                    else result
                )

            module.forward = types.MethodType(fixed, module)
        yield
    finally:
        for handle in handles:
            handle.remove()
        for module, original in originals.items():
            module.forward = original


def typed_deletion(data, nodes):
    """Zero typed node inputs and remove incident edges from every relation."""
    out = data.clone()
    masked = {kind: set() for kind in out.node_types}
    for kind, index in nodes:
        if kind not in masked or not 0 <= index < out[kind].num_nodes:
            raise IndexError(f"Unknown typed node {(kind, index)}")
        masked[kind].add(index)
    for kind, ids in masked.items():
        if ids:
            out[kind].x[sorted(ids)] = 0
    for relation in out.edge_types:
        source, _, target = relation
        store = out[relation]
        src_ids = torch.tensor(
            sorted(masked[source]), device=store.edge_index.device, dtype=torch.long
        )
        dst_ids = torch.tensor(
            sorted(masked[target]), device=store.edge_index.device, dtype=torch.long
        )
        keep = ~torch.isin(store.edge_index[0], src_ids) & ~torch.isin(
            store.edge_index[1], dst_ids
        )
        edge_count = store.edge_index.shape[1]
        for key in list(store.keys()):
            if (
                key != "edge_index"
                and store.is_edge_attr(key)
                and store[key].shape[0] == edge_count
            ):
                store[key] = store[key][keep]
        store.edge_index = store.edge_index[:, keep]
    return out


class FrozenModelAdapter:
    """Adapt a frozen model using differentiable representation and predictor calls.

    representation(data, target) returns a single raw target vector. candidates
    returns removable global IDs, or (node_type, local_id) pairs for HeteroData.
    All normalization/attention context remains on the original graph. For input
    encoders, source_inputs and with_inputs can instead expose encoded vectors.
    """

    def __init__(
        self,
        model,
        representation,
        predict,
        candidates,
        *,
        source_inputs=None,
        with_inputs=None,
        delete=None,
        attention_context=None,
    ):
        self.model = model.eval()
        self.representation = representation
        self.predict = predict
        self.candidates = candidates
        self.source_inputs = source_inputs or (
            lambda d: d.x_dict if hasattr(d, "node_types") else {None: d.x}
        )
        self.with_inputs = with_inputs or self._with_inputs
        self.delete = delete or self._delete
        self.attention_context = attention_context

    @staticmethod
    def _with_inputs(data, inputs):
        out = data.clone()
        for kind, value in inputs.items():
            if kind is None:
                out.x = value
            else:
                out[kind].x = value
        return out

    @staticmethod
    def _delete(data, nodes):
        if hasattr(data, "node_types"):
            return typed_deletion(data, nodes)
        from .masking import clone_with_node_mask

        return clone_with_node_mask(data, nodes, mode="both")

    def project_sources(self, data, target, direction):
        self.model.eval()
        inputs = {
            kind: value.detach().clone().requires_grad_(True)
            for kind, value in self.source_inputs(data).items()
        }
        patched = self.with_inputs(data, inputs)
        forward = lambda: self.representation(patched, target)
        context = (
            self.attention_context(self.model, forward)
            if self.attention_context
            else frozen_attention(self.model, forward)
        )
        with torch.enable_grad(), context:
            representation = forward()
            gradient = torch.autograd.grad(
                (representation * direction.to(representation.device)).sum(),
                tuple(inputs.values()),
                allow_unused=True,
            )
        scores = {
            kind: (
                (grad * inputs[kind]).sum(-1).detach().cpu()
                if grad is not None
                else torch.zeros(len(inputs[kind]))
            )
            for kind, grad in zip(inputs, gradient)
        }
        candidates = sorted(self.candidates(data, target))
        values = [
            scores[u[0]][u[1]] if isinstance(u, tuple) else scores[None][u]
            for u in candidates
        ]
        return candidates, torch.stack(values) if values else torch.empty(0)


def heterogeneous_receptive_field(data, target, hops):
    """Directed incoming-message field with typed IDs and no dense adjacency."""
    kind, index = target
    visited = {name: set() for name in data.node_types}
    frontier = {name: set() for name in data.node_types}
    visited[kind].add(index)
    frontier[kind].add(index)
    for _ in range(hops):
        next_frontier = {name: set() for name in data.node_types}
        for source, relation, destination in data.edge_types:
            edge = data[source, relation, destination].edge_index
            destinations = torch.tensor(
                sorted(frontier[destination]), dtype=torch.long, device=edge.device
            )
            predecessors = set(edge[0, torch.isin(edge[1], destinations)].tolist())
            next_frontier[source].update(predecessors - visited[source])
        for name in visited:
            visited[name].update(next_frontier[name])
        frontier = next_frontier
    visited[kind].discard(index)
    return sorted((name, i) for name, ids in visited.items() for i in ids)
