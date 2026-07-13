"""SAE-GNNExplainer for node and graph classification.

Node classification scores k-hop source nodes through target-active SAE codes.
Graph classification instead allocates an active pooled SAE direction back to
the final-layer node states used by mean readout.  The different treatment is
necessary because graph logits are computed after pooling whereas node logits
are computed directly from each node's final representation.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor
from torch_geometric.utils import k_hop_subgraph

from .alignment import top_active_feature_ids
from .sae import SAEArtifact


def _classifier_direction(model, reference_class: int) -> Tensor:
    if not hasattr(model, "classifier"):
        raise TypeError("SAE-GNNExplainer requires a backbone exposing .classifier")
    return model.classifier.weight.detach().cpu()[int(reference_class)]


def _feature_rows(artifact: SAEArtifact, codes: Tensor, reference_direction: Tensor, *, num_features: int) -> list[dict]:
    feature_ids = top_active_feature_ids(codes, limit=num_features)
    decoder = artifact.sae.decoder.weight.detach().cpu()
    rows = []
    for rank, feature_id in enumerate(feature_ids, start=1):
        class_effect = float(torch.dot(decoder[:, feature_id], reference_direction).item())
        rows.append(
            {
                "rank": rank,
                "feature_id": int(feature_id),
                "activation": float(codes[feature_id].item()),
                "classifier_effect": class_effect,
                "signed_feature_effect": float(codes[feature_id].item() * class_effect),
            }
        )
    return rows


def _select(scores: Tensor, candidate_mask: Tensor, *, top_k: int, positive: bool) -> list[int]:
    values = scores.clone()
    values[~candidate_mask] = float("-inf") if positive else float("inf")
    order = torch.argsort(values, descending=positive)
    selected: list[int] = []
    for index in order.tolist():
        if not bool(candidate_mask[index]):
            continue
        value = float(scores[index].item())
        if (positive and value <= 0) or ((not positive) and value >= 0):
            continue
        selected.append(int(index))
        if len(selected) >= int(top_k):
            break
    return selected


@torch.no_grad()
def explain_node(
    model,
    data,
    artifact: SAEArtifact,
    *,
    target_node: int,
    num_features: int = 16,
    top_k: int = 8,
    device: torch.device | str = "cpu",
) -> dict:
    """Return SAE+, SAE-, and SAE+/- explanation nodes for one target node."""
    device = torch.device(device)
    model = model.to(device).eval()
    data = data.to(device)
    logits, capture = model(data.x, data.edge_index, batch=getattr(data, "batch", None), capture=True)
    target_node = int(target_node)
    reference_class = int(logits[target_node].argmax().item())
    codes = artifact.encode(capture.node, device=device)
    features = _feature_rows(artifact, codes[target_node], _classifier_direction(model, reference_class), num_features=num_features)
    if not features:
        return {"target_node": target_node, "reference_class": reference_class, "active_features": [], "node_scores": [], "sae_positive": [], "sae_negative": [], "sae_signed": []}

    feature_ids = torch.tensor([row["feature_id"] for row in features], dtype=torch.long)
    feature_weights = torch.tensor([row["classifier_effect"] for row in features], dtype=torch.float32)
    signed_scores = (codes[:, feature_ids] * feature_weights).sum(dim=-1)
    candidate_nodes, _, _, _ = k_hop_subgraph(target_node, int(getattr(model, "num_layers", 2)), data.edge_index, relabel_nodes=False)
    candidate_mask = torch.zeros(codes.size(0), dtype=torch.bool)
    candidate_mask[candidate_nodes.detach().cpu()] = True
    candidate_mask[target_node] = False
    positive = _select(signed_scores, candidate_mask, top_k=top_k, positive=True)
    negative = _select(signed_scores, candidate_mask, top_k=top_k, positive=False)
    ranked = torch.argsort(signed_scores.abs(), descending=True)
    node_scores = [
        {
            "node_id": int(node_id),
            "in_receptive_field": bool(candidate_mask[node_id]),
            "signed_score": float(signed_scores[node_id].item()),
            "positive_score": float(max(0.0, signed_scores[node_id].item())),
            "negative_score": float(max(0.0, -signed_scores[node_id].item())),
        }
        for node_id in ranked.tolist()
        if bool(candidate_mask[node_id])
    ]
    return {
        "task": "node",
        "target_node": target_node,
        "reference_class": reference_class,
        "active_features": features,
        "node_scores": node_scores,
        "sae_positive": positive,
        "sae_negative": negative,
        "sae_signed": sorted(set(positive) | set(negative)),
    }


@torch.no_grad()
def explain_graph(
    model,
    data,
    artifact: SAEArtifact,
    *,
    graph_index: int = 0,
    num_features: int = 16,
    top_k: int = 8,
    device: torch.device | str = "cpu",
) -> dict:
    """Allocate graph-active SAE features back to final-layer node states.

    For mean pooling, each node contributes ``h_v / |V|`` to the graph hidden
    state.  We project this contribution onto each active SAE decoder direction
    and weight it by that feature's class-logit effect.
    """
    device = torch.device(device)
    model = model.to(device).eval()
    data = data.to(device)
    logits, capture = model(data.x, data.edge_index, batch=getattr(data, "batch", None), capture=True)
    if capture.graph is None:
        raise ValueError("explain_graph requires a graph-classification backbone")
    graph_index = int(graph_index)
    reference_class = int(logits[graph_index].argmax().item())
    graph_codes = artifact.encode(capture.graph, device=device)
    reference_direction = _classifier_direction(model, reference_class)
    features = _feature_rows(artifact, graph_codes[graph_index], reference_direction, num_features=num_features)
    batch = getattr(data, "batch", None)
    if batch is None:
        batch = torch.zeros(capture.node.size(0), dtype=torch.long, device=device)
    graph_nodes = torch.nonzero(batch == graph_index, as_tuple=False).view(-1)
    node_h = capture.node[graph_nodes].detach().cpu()
    decoder = artifact.sae.decoder.weight.detach().cpu()
    signed_scores = torch.zeros(node_h.size(0))
    for feature in features:
        direction = decoder[:, int(feature["feature_id"])]
        node_projection = node_h @ direction
        signed_scores += node_projection * float(feature["signed_feature_effect"]) / max(1, node_h.size(0))
    candidate_mask = torch.ones(node_h.size(0), dtype=torch.bool)
    positive_local = _select(signed_scores, candidate_mask, top_k=top_k, positive=True)
    negative_local = _select(signed_scores, candidate_mask, top_k=top_k, positive=False)
    order = torch.argsort(signed_scores.abs(), descending=True)
    node_scores = [
        {
            "node_id": int(graph_nodes[local_id].item()),
            "signed_score": float(signed_scores[local_id].item()),
            "positive_score": float(max(0.0, signed_scores[local_id].item())),
            "negative_score": float(max(0.0, -signed_scores[local_id].item())),
        }
        for local_id in order.tolist()
    ]
    to_global = lambda ids: [int(graph_nodes[index].item()) for index in ids]
    return {
        "task": "graph",
        "graph_index": graph_index,
        "reference_class": reference_class,
        "active_features": features,
        "node_scores": node_scores,
        "sae_positive": to_global(positive_local),
        "sae_negative": to_global(negative_local),
        "sae_signed": sorted(set(to_global(positive_local)) | set(to_global(negative_local))),
    }
