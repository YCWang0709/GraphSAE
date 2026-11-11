"""Synthetic dataset construction and structural reference masks."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence
import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data


@dataclass(frozen=True)
class BAGraphConfig:
    name: str
    description: str
    class_names: Tuple[str, ...]
    seed: int
    num_nodes: int
    ba_m: int


DATASET_CONFIGS: Dict[str, BAGraphConfig] = {
    "BATriadic": BAGraphConfig(
        name="BATriadic",
        description="Triadic-closure BA graph. Labels are structural role buckets from degree and triangle participation.",
        class_names=("low_triadic_role", "middle_triadic_role", "high_triadic_role"),
        seed=37,
        num_nodes=2000,
        ba_m=3,
    ),
    "BAPathRole": BAGraphConfig(
        name="BAPathRole",
        description="BA backbone with many attached path motifs. Labels are background, path-internal, or path-end role.",
        class_names=("background_ba", "path_internal", "path_end"),
        seed=43,
        num_nodes=2000,
        ba_m=3,
    ),
    "BABiFanRole": BAGraphConfig(
        name="BABiFanRole",
        description="BA backbone with many attached bi-fan motifs. Labels are background, small-partition, or large-partition role.",
        class_names=("background_ba", "bifan_small_partition", "bifan_large_partition"),
        seed=47,
        num_nodes=2000,
        ba_m=3,
    ),
}


def make_ba_triadic(config: BAGraphConfig, triangle_probability: float = 0.35) -> Data:
    """Triadic-closure BA graph with structural-role labels.

    Construction:
    - Generate a Holme-Kim powerlaw-cluster graph, which starts from the BA
      preferential-attachment idea and adds triadic closure with probability p.
    - This keeps the heavy-tailed degree distribution but creates more triangles
      than a plain BA graph.

    Label:
    - Compute degree rank r_deg and triangle-count rank r_tri in [0, 1].
    - Define role_score = 0.65 * r_deg + 0.35 * r_tri.
    - Split role_score into three balanced buckets:
        0 low triadic role, 1 middle triadic role, 2 high triadic role.
    """
    rng = np.random.default_rng(config.seed)
    graph = nx.powerlaw_cluster_graph(
        n=config.num_nodes,
        m=config.ba_m,
        p=triangle_probability,
        seed=config.seed,
    )

    degree = _degree_array(graph).astype(np.float64)
    triangles = np.array(
        [nx.triangles(graph, node) for node in range(graph.number_of_nodes())],
        dtype=np.float64,
    )

    r_deg = _rank_percentile(np.log1p(degree), rng=rng)
    r_tri = _rank_percentile(np.log1p(triangles), rng=rng)
    role_score = 0.65 * r_deg + 0.35 * r_tri
    y = _rank_bucket_labels(role_score, num_classes=3, rng=rng)

    extra_arrays = {
        "log_triangles_z": _zscore(np.log1p(triangles)),
        "triangles_norm": triangles / (triangles.max() + 1e-8),
        "role_score": role_score,
    }
    x = _structural_features(graph, rng=rng, extra_arrays=extra_arrays, noise_dim=10)
    return _to_pyg_data(graph, x=x, y=y, seed=config.seed)


def make_ba_path_role(
    config: BAGraphConfig, backbone_nodes: int = 750, num_motifs: int = 250
) -> Data:
    """BA graph with attached path-5 motifs and role labels.

    Construction:
    - Start from a BA backbone.
    - Attach many path-5 motifs to random/high-degree backbone anchors.
    - Label background BA nodes as class 0.
    - For each path motif, label internal nodes as class 1 and path endpoints as class 2.
    """
    rng = np.random.default_rng(config.seed)
    graph = nx.barabasi_albert_graph(n=backbone_nodes, m=config.ba_m, seed=config.seed)
    labels = np.zeros(backbone_nodes, dtype=np.int64)

    for motif_idx in range(num_motifs):
        path_nodes = list(range(graph.number_of_nodes(), graph.number_of_nodes() + 5))
        graph.add_nodes_from(path_nodes)
        graph.add_edges_from((path_nodes[i], path_nodes[i + 1]) for i in range(4))
        labels = np.concatenate([labels, np.array([2, 1, 1, 1, 2], dtype=np.int64)])
        anchor = _sample_backbone_anchor(graph, backbone_nodes, rng)
        attach_to = int(path_nodes[2 if motif_idx % 2 == 0 else 1])
        graph.add_edge(anchor, attach_to)

    x = _structural_features(graph, rng=rng, extra_arrays=None, noise_dim=8)
    return _to_pyg_data(graph, x=x, y=labels, seed=config.seed)


def make_ba_bifan_role(
    config: BAGraphConfig, backbone_nodes: int = 750, num_motifs: int = 250
) -> Data:
    """BA graph with attached K2,3 motifs and bipartite role labels."""
    rng = np.random.default_rng(config.seed)
    graph = nx.barabasi_albert_graph(n=backbone_nodes, m=config.ba_m, seed=config.seed)
    labels = np.zeros(backbone_nodes, dtype=np.int64)

    for motif_idx in range(num_motifs):
        motif_nodes = list(range(graph.number_of_nodes(), graph.number_of_nodes() + 5))
        left = motif_nodes[:2]
        right = motif_nodes[2:]
        graph.add_nodes_from(motif_nodes)
        for u in left:
            for v in right:
                graph.add_edge(u, v)
        labels = np.concatenate([labels, np.array([1, 1, 2, 2, 2], dtype=np.int64)])
        anchor = _sample_backbone_anchor(graph, backbone_nodes, rng)
        attach_to = int(left[motif_idx % 2])
        graph.add_edge(anchor, attach_to)

    x = _structural_features(graph, rng=rng, extra_arrays=None, noise_dim=8)
    return _to_pyg_data(graph, x=x, y=labels, seed=config.seed)


def _to_pyg_data(graph: nx.Graph, x: np.ndarray, y: np.ndarray, seed: int) -> Data:
    num_nodes = graph.number_of_nodes()
    edge_index = _edge_index_from_nx(graph)
    train_mask, val_mask, test_mask = _stratified_masks(y, seed=seed)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    data.num_nodes = num_nodes
    return data


def _edge_index_from_nx(graph: nx.Graph) -> torch.Tensor:
    edges = []
    for u, v in graph.edges():
        if u == v:
            continue
        edges.append((int(u), int(v)))
        edges.append((int(v), int(u)))
    edges = sorted(edges)
    if not edges:
        raise ValueError("Generated graph has no edges.")
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _degree_array(graph: nx.Graph) -> np.ndarray:
    return np.array(
        [graph.degree(i) for i in range(graph.number_of_nodes())], dtype=np.float32
    )


def _sample_backbone_anchor(
    graph: nx.Graph, backbone_nodes: int, rng: np.random.Generator
) -> int:
    degrees = np.array(
        [graph.degree(i) for i in range(backbone_nodes)], dtype=np.float64
    )
    prob = degrees + 1.0
    prob = prob / prob.sum()
    if rng.random() < 0.55:
        return int(rng.choice(np.arange(backbone_nodes), p=prob))
    return int(rng.integers(0, backbone_nodes))


def _structural_features(
    graph: nx.Graph,
    rng: np.random.Generator,
    extra_arrays: Dict[str, np.ndarray] | None,
    noise_dim: int,
) -> np.ndarray:
    n = graph.number_of_nodes()
    degree = _degree_array(graph).astype(np.float64)
    log_degree = np.log1p(degree)

    clustering_dict = nx.clustering(graph)
    clustering = np.array([clustering_dict[i] for i in range(n)], dtype=np.float64)

    avg_neighbor_degree_dict = nx.average_neighbor_degree(graph)
    avg_neighbor_degree = np.array(
        [avg_neighbor_degree_dict[i] for i in range(n)], dtype=np.float64
    )

    try:
        core_dict = nx.core_number(graph)
        core_number = np.array([core_dict[i] for i in range(n)], dtype=np.float64)
    except nx.NetworkXError:
        core_number = np.zeros(n, dtype=np.float64)

    base_features = [
        _zscore(log_degree),
        degree / (degree.max() + 1e-8),
        clustering,
        _zscore(avg_neighbor_degree),
        core_number / (core_number.max() + 1e-8),
    ]

    if extra_arrays is not None:
        for _, values in extra_arrays.items():
            values = np.asarray(values, dtype=np.float64)
            if values.shape[0] != n:
                raise ValueError(
                    "Every extra feature array must have length num_nodes."
                )
            base_features.append(values)

    structural_x = np.stack(base_features, axis=1).astype(np.float32)
    noise = rng.normal(0.0, 1.0, size=(n, noise_dim)).astype(np.float32)
    return np.concatenate([structural_x, noise], axis=1).astype(np.float32)


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / (values.std() + 1e-8)


def _rank_percentile(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    jitter = rng.uniform(low=-1e-9, high=1e-9, size=values.shape[0])
    order = np.argsort(values + jitter)
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(values.shape[0], dtype=np.float64)
    if values.shape[0] == 1:
        return np.zeros_like(ranks)
    return ranks / (values.shape[0] - 1)


def _rank_bucket_labels(
    values: np.ndarray, num_classes: int, rng: np.random.Generator
) -> np.ndarray:
    ranks = _rank_percentile(values, rng=rng)
    labels = np.floor(ranks * num_classes).astype(np.int64)
    labels = np.clip(labels, 0, num_classes - 1)
    return labels


def _stratified_masks(
    y: np.ndarray, seed: int, train_ratio: float = 0.60, val_ratio: float = 0.20
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y = np.asarray(y, dtype=np.int64)
    rng = np.random.default_rng(seed)
    n = y.shape[0]

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    for cls in sorted(np.unique(y).tolist()):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)

        n_train = max(1, int(round(train_ratio * len(idx))))
        n_val = max(1, int(round(val_ratio * len(idx))))
        if n_train + n_val >= len(idx):
            n_train = max(1, len(idx) - 2)
            n_val = 1

        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

    return (
        torch.tensor(train_mask, dtype=torch.bool),
        torch.tensor(val_mask, dtype=torch.bool),
        torch.tensor(test_mask, dtype=torch.bool),
    )
