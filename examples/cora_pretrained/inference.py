"""Frozen SAE source scoring and validation-selected Cora extraction rules."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import (
    add_self_loops,
    degree,
    k_hop_subgraph,
    remove_self_loops,
    softmax,
)

try:
    from .models import CapturingGNN
except ImportError:
    from models import CapturingGNN


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def sparse_transpose(indices, values, num_nodes):
    with torch.sparse.check_sparse_tensor_invariants():
        return torch.sparse_coo_tensor(
            indices, values, (num_nodes, num_nodes)
        ).coalesce()


def load_manifest(directory):
    manifest = json.loads(
        (Path(directory) / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported checkpoint manifest")
    return manifest


def checked_load(directory, entry):
    path = Path(directory) / entry["file"]
    if file_hash(path) != entry["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if tensor_hash(payload["state_dict"]) != entry["state_tensor_sha256"]:
        raise ValueError(f"Checkpoint tensor checksum mismatch: {path}")
    return payload


def load_data(root, manifest):
    data = Planetoid(
        root=str(root), name="Cora", split="public", transform=NormalizeFeatures()
    )[0]
    values = {
        key: data[key]
        for key in ("x", "edge_index", "y", "train_mask", "val_mask", "test_mask")
    }
    if tensor_hash(values) != manifest["data_tensor_sha256"]:
        raise ValueError("Cora tensors differ from the frozen evaluation data")
    for entry in manifest["models"].values():
        if not all(bool(data.test_mask[i]) for i in entry["test_targets"]):
            raise ValueError("Evaluation target is outside the public test split")
    return data


def load_predictor(directory, entry):
    payload = checked_load(directory, entry["backbone"])
    model = CapturingGNN(**payload["architecture"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model


class FrozenSAE(nn.Module):
    def __init__(self, payload):
        super().__init__()
        state = payload["state_dict"]
        latent_dim, input_dim = state["encoder.weight"].shape
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.load_state_dict(state, strict=True)
        self.register_buffer("mean", payload["mean"].reshape(-1))
        self.register_buffer("std", payload["std"].reshape(-1))
        self.k_active = int(payload["k_active"])
        self.eval()
        self.requires_grad_(False)

    @torch.no_grad()
    def direction(self, representation, class_gradient):
        code = F.relu(self.encoder((representation - self.mean) / self.std))
        values, ids = code.topk(self.k_active)
        sparse = torch.zeros_like(code).scatter(0, ids, values)
        decoder = self.decoder.weight
        decoder = decoder / decoder.norm(dim=0, keepdim=True).clamp_min(1e-8)
        weighted = sparse * (decoder.T @ (class_gradient * self.std))
        return (decoder @ weighted) / self.std


class SourceProjection:
    """Transpose the observed local message-passing map, with fixed attention."""

    def __init__(self, model, data, layer):
        self.model, self.data, self.layer = model, data, layer
        with torch.no_grad():
            self.logits, self.captures = model(data.x, data.edge_index, capture=True)
        self.ops = []
        previous = data.x
        for index in range(layer + 1):
            conv, capture = model.convs[index], self.captures[index]
            op = {"conv": conv, "gate": (capture.pre_activation > 0).float()}
            if model.model_name == "gcn":
                edges, _ = remove_self_loops(data.edge_index)
                edges, _ = add_self_loops(edges, num_nodes=data.num_nodes)
                source, target = edges
                count = degree(target, data.num_nodes, dtype=data.x.dtype).clamp_min(1)
                norm = count.pow(-0.5)[source] * count.pow(-0.5)[target]
                op["transpose"] = sparse_transpose(
                    torch.stack([source, target]),
                    norm,
                    data.num_nodes,
                )
            elif model.model_name == "gat":
                edges, _ = remove_self_loops(data.edge_index)
                edges, _ = add_self_loops(edges, num_nodes=data.num_nodes)
                source, target = edges
                projected = conv.lin(previous).view(-1, conv.heads, conv.out_channels)
                alpha_source = (projected * conv.att_src).sum(-1)
                alpha_target = (projected * conv.att_dst).sum(-1)
                alpha = softmax(
                    F.leaky_relu(
                        alpha_source[source] + alpha_target[target], conv.negative_slope
                    ),
                    target,
                    num_nodes=data.num_nodes,
                ).detach()
                op["transpose"] = [
                    sparse_transpose(
                        torch.stack([source, target]),
                        alpha[:, head],
                        data.num_nodes,
                    )
                    for head in range(conv.heads)
                ]
            elif model.model_name == "gin":
                source, target = data.edge_index
                total = torch.zeros_like(previous)
                total.index_add_(0, target, previous[source])
                base = (total + (1 + conv.eps) * previous).detach().requires_grad_(True)
                with torch.enable_grad():
                    op["base"], op["mlp_output"] = base, conv.mlp(base)
            self.ops.append(op)
            previous = capture.node
        self.deletion_cache = first_projection_cache(model, data.x)

    def candidates(self, target):
        ids, _, _, _ = k_hop_subgraph(
            target,
            self.model.num_layers,
            self.data.edge_index,
            num_nodes=self.data.num_nodes,
        )
        return sorted(int(u) for u in ids if int(u) != target)

    def project(self, target, direction):
        gradient = torch.zeros_like(self.captures[self.layer].node)
        gradient[target] = direction
        source, destination = self.data.edge_index
        for op in reversed(self.ops):
            conv = op["conv"]
            gradient = gradient * op["gate"]
            if self.model.model_name == "gcn":
                gradient = torch.sparse.mm(op["transpose"], gradient) @ conv.lin.weight
            elif self.model.model_name == "gin":
                gradient = torch.autograd.grad(
                    op["mlp_output"],
                    op["base"],
                    grad_outputs=gradient,
                    retain_graph=True,
                )[0]
                neighbors = torch.zeros_like(gradient)
                neighbors.index_add_(0, source, gradient[destination])
                gradient = (1 + conv.eps.detach()) * gradient + neighbors
            elif self.model.model_name == "gat":
                heads = gradient.view(gradient.size(0), conv.heads, conv.out_channels)
                gradient = (
                    torch.stack(
                        [
                            torch.sparse.mm(matrix, heads[:, head])
                            for head, matrix in enumerate(op["transpose"])
                        ],
                        1,
                    ).reshape(gradient.size(0), -1)
                    @ conv.lin.weight
                )
            else:
                neighbor_gradient = gradient @ conv.lin_neigh.weight
                count = (
                    torch.bincount(destination, minlength=gradient.size(0))
                    .float()
                    .clamp_min(1)
                )
                backward = torch.zeros_like(neighbor_gradient)
                backward.index_add_(
                    0, source, neighbor_gradient[destination] / count[destination, None]
                )
                gradient = backward + gradient @ conv.lin_self.weight
        return (gradient * self.data.x).sum(-1).detach()


class ProjectionSlice(nn.Module):
    def __init__(self, start, stop, bias):
        super().__init__()
        self.start, self.stop = start, stop
        self.register_buffer("bias", None if bias is None else bias.detach().clone())

    def forward(self, x):
        selected = x[..., self.start : self.stop]
        return selected if self.bias is None else selected + self.bias


def first_projection_cache(model, inputs):
    """Cache bias-free first affine projections for repeated native deletions."""
    cached = copy.deepcopy(model).eval()
    first = cached.convs[0]
    pieces, width = [], 0

    def cache(linear):
        nonlocal width
        with torch.no_grad():
            value = F.linear(inputs, linear.weight, None)
        start = width
        width += value.shape[-1]
        pieces.append(value)
        return ProjectionSlice(start, width, getattr(linear, "bias", None))

    if model.model_name in {"gcn", "gat"}:
        first.lin = cache(first.lin)
    elif model.model_name == "gin":
        first.mlp[0] = cache(first.mlp[0])
    else:
        first.lin_neigh = cache(first.lin_neigh)
        first.lin_self = cache(first.lin_self)
    return cached, torch.cat(pieces, -1)


class DeletionOracle:
    """Greedy search queries plus independent full-predictor final replay."""

    def __init__(self, model, data, original_logits, target, projection_cache=None):
        self.model, self.data, self.target = model, data, int(target)
        self.original = original_logits[target].detach()
        self.reference_class = int(self.original.argmax())
        self.original_probability_tensor = self.original.softmax(-1)[
            self.reference_class
        ]
        self.original_probability = float(self.original_probability_tensor)
        self.cache, self.local_cache = {}, {}
        self.cached_model, self.features = projection_cache or first_projection_cache(
            model, data.x
        )
        self.local = None
        if model.model_name in {"gat", "sage", "graphsage"}:
            ids, edges, mapping, _ = k_hop_subgraph(
                target,
                model.num_layers,
                data.edge_index,
                relabel_nodes=True,
                num_nodes=data.num_nodes,
            )
            self.local = (
                data.x[ids],
                edges,
                int(mapping),
                {int(u): i for i, u in enumerate(ids)},
            )

    @torch.no_grad()
    def native_value(self, nodes):
        key = tuple(sorted(nodes))
        if key not in self.cache:
            keep = torch.ones(self.data.num_nodes, dtype=torch.bool)
            keep[list(key)] = False
            x = self.features.clone()
            x[~keep] = 0
            edges = self.data.edge_index[
                :, keep[self.data.edge_index[0]] & keep[self.data.edge_index[1]]
            ]
            logits, _ = self.cached_model(x, edges)
            self.cache[key] = float(
                self.original_probability_tensor
                - logits[self.target].softmax(-1)[self.reference_class]
            )
        return self.cache[key]

    @torch.no_grad()
    def value(self, nodes):
        if self.local is None:
            return self.native_value(nodes)
        key = tuple(sorted(nodes))
        if key not in self.local_cache:
            x, edges, index, lookup = self.local
            keep = torch.ones(len(x), dtype=torch.bool)
            keep[[lookup[u] for u in key]] = False
            masked = x.clone()
            masked[~keep] = 0
            edges = edges[:, keep[edges[0]] & keep[edges[1]]]
            logits, _ = self.model(masked, edges)
            self.local_cache[key] = float(
                self.original_probability_tensor
                - logits[index].softmax(-1)[self.reference_class]
            )
        return self.local_cache[key]

    @torch.no_grad()
    def replay(self, nodes):
        nodes = sorted(set(nodes))
        if self.target in nodes:
            raise ValueError("The target is excluded from removable sources")
        keep = torch.ones(self.data.num_nodes, dtype=torch.bool)
        keep[nodes] = False
        x = self.data.x.clone()
        x[~keep] = 0
        edges = self.data.edge_index[
            :, keep[self.data.edge_index[0]] & keep[self.data.edge_index[1]]
        ]
        logits, _ = self.model(x, edges)
        masked = logits[self.target]
        return {
            "logit_drop": float(
                self.original[self.reference_class] - masked[self.reference_class]
            ),
            "probability_drop": float(
                self.original_probability_tensor
                - masked.softmax(-1)[self.reference_class]
            ),
            "num_nodes": len(nodes),
            "reference_class": self.reference_class,
            "original_probability": self.original_probability,
            "masked_probability": float(masked.softmax(-1)[self.reference_class]),
        }


def extract_nodes(candidates, scores, oracle, rule, pool_size=32):
    """Use the archived validation-selected rule; no test-driven size selection."""
    ranked = sorted(
        zip(candidates, scores.tolist()), key=lambda item: (-max(0.0, item[1]), item[0])
    )
    remaining = [u for u, _ in ranked[:pool_size]]
    chosen = []
    previous = oracle.native_value([])
    if rule.startswith("greedy_"):
        maximum = min(int(rule.split("_", 1)[1]), len(remaining))
        tolerance = None
    elif rule.startswith("gainstop_"):
        maximum = len(remaining)
        tolerance = float(rule.split("_", 1)[1])
    else:
        raise ValueError(f"Unsupported selected extraction rule {rule}")
    for _ in range(maximum):
        evaluated = [
            (oracle.value(chosen + [u]), -j, u) for j, u in enumerate(remaining)
        ]
        if oracle.local is not None:
            best_local = max(value for value, _, _ in evaluated)
            evaluated = [
                (
                    (
                        oracle.native_value(chosen + [u])
                        if value >= best_local - 1e-5
                        else value
                    ),
                    order,
                    u,
                )
                for value, order, u in evaluated
            ]
        value, _, node = max(evaluated, key=lambda item: (item[0], item[1]))
        if tolerance is not None and value - previous <= tolerance:
            break
        chosen.append(node)
        remaining.remove(node)
        previous = value
    return sorted(chosen)


def explain_target(projection, sae, entry, target):
    predicted_class = int(projection.logits[target].argmax())
    direction = sae.direction(
        projection.captures[projection.layer].node[target],
        projection.model.classifier.weight[predicted_class],
    )
    candidates = projection.candidates(target)
    scores = projection.project(target, direction)[candidates]
    oracle = DeletionOracle(
        projection.model,
        projection.data,
        projection.logits,
        target,
        projection.deletion_cache,
    )
    nodes = extract_nodes(
        candidates, scores, oracle, entry["rule"], entry["candidate_pool"]
    )
    metrics = oracle.replay(nodes)
    return {
        "target_node": target,
        "selected_nodes": nodes,
        "node_scores": [
            {"node_id": u, "score": float(s)} for u, s in zip(candidates, scores)
        ],
        "rule": entry["rule"],
        "metrics": metrics,
        "search_queries": len(oracle.cache) + len(oracle.local_cache),
    }
