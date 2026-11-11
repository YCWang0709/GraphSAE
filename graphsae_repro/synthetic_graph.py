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
class BAShapeGraphConfig:
    name: str
    description: str
    class_names: Tuple[str, ...]
    seed: int = 42
    num_graphs: int = 900
    ba_min_nodes: int = 22
    ba_max_nodes: int = 38
    ba_m: int = 2


DATASET_CONFIGS: Dict[str, BAShapeGraphConfig] = {
    "BAHouse": BAShapeGraphConfig(
        name="BAHouse",
        description="BA backbone plus a variable number of house motifs. Graph label is the number of inserted house motifs: 0, 1, or 2.",
        class_names=("no_house", "one_house", "two_houses"),
        seed=101,
        num_graphs=900,
    ),
    "BACycle": BAShapeGraphConfig(
        name="BACycle",
        description="BA backbone plus a variable number of 6-cycle motifs. Graph label is the number of inserted cycle motifs: 0, 1, or 2.",
        class_names=("no_cycle", "one_cycle", "two_cycles"),
        seed=202,
        num_graphs=900,
    ),
    "BAPath": BAShapeGraphConfig(
        name="BAPath",
        description="BA backbone plus a variable number of path-5 motifs. Graph label is the number of inserted path motifs: 0, 1, or 2.",
        class_names=("no_path", "one_path", "two_paths"),
        seed=404,
        num_graphs=900,
    ),
}


def make_ba_house_dataset(config: BAShapeGraphConfig) -> List[Data]:
    """Three classes: 0, 1, or 2 inserted house motifs."""

    rng = np.random.default_rng(config.seed)
    data_list: List[Data] = []
    labels = [i % 3 for i in range(config.num_graphs)]
    rng.shuffle(labels)

    for graph_id, label in enumerate(labels):
        data = make_single_graph_with_count_label(
            dataset_name=config.name,
            graph_id=graph_id,
            rng=rng,
            config=config,
            motif_kind="house",
            num_motifs=int(label),
            y=int(label),
        )
        data_list.append(data)
    return data_list


def make_ba_cycle_dataset(config: BAShapeGraphConfig) -> List[Data]:
    """Three classes: 0, 1, or 2 inserted six-cycle motifs."""

    rng = np.random.default_rng(config.seed)
    data_list: List[Data] = []
    labels = [i % 3 for i in range(config.num_graphs)]
    rng.shuffle(labels)

    for graph_id, label in enumerate(labels):
        data = make_single_graph_with_count_label(
            dataset_name=config.name,
            graph_id=graph_id,
            rng=rng,
            config=config,
            motif_kind="cycle6",
            num_motifs=int(label),
            y=int(label),
        )
        data_list.append(data)
    return data_list


def make_ba_path_dataset(config: BAShapeGraphConfig) -> List[Data]:
    """Three classes: 0, 1, or 2 inserted path-5 motifs."""

    rng = np.random.default_rng(config.seed)
    data_list: List[Data] = []
    labels = [i % 3 for i in range(config.num_graphs)]
    rng.shuffle(labels)

    for graph_id, label in enumerate(labels):
        data = make_single_graph_with_count_label(
            dataset_name=config.name,
            graph_id=graph_id,
            rng=rng,
            config=config,
            motif_kind="path5",
            num_motifs=int(label),
            y=int(label),
        )
        data_list.append(data)
    return data_list


def make_single_graph_with_count_label(
    dataset_name: str,
    graph_id: int,
    rng: np.random.Generator,
    config: BAShapeGraphConfig,
    motif_kind: str,
    num_motifs: int,
    y: int,
    attachment_mode: str = "single",
) -> Data:
    n_backbone = int(rng.integers(config.ba_min_nodes, config.ba_max_nodes + 1))
    ba_seed = int(rng.integers(0, 2**31 - 1))
    graph = nx.barabasi_albert_graph(n=n_backbone, m=config.ba_m, seed=ba_seed)

    motif_nodes_all: List[int] = []
    motif_edges_internal: List[Tuple[int, int]] = []
    attachment_edges: List[Tuple[int, int]] = []
    motif_id = np.full(n_backbone, -1, dtype=np.int64)

    for motif_pos in range(int(num_motifs)):
        start = graph.number_of_nodes()
        nodes, edges = motif_template(motif_kind, start=start)
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        if graph.number_of_nodes() > len(nodes):
            motif_attachment_targets = motif_attachment_targets_for_kind(
                motif_kind, nodes
            )
            attachment_pairs = sample_attachment_pairs(
                rng=rng,
                n_backbone=n_backbone,
                graph=graph,
                motif_targets=motif_attachment_targets,
                attachment_mode=attachment_mode,
            )
            for anchor, attach_to in attachment_pairs:
                graph.add_edge(anchor, attach_to)
                attachment_edges.append((anchor, attach_to))

        old = motif_id
        motif_id = np.concatenate(
            [old, np.full(len(nodes), motif_pos, dtype=np.int64)], axis=0
        )
        motif_nodes_all.extend(nodes)
        motif_edges_internal.extend(edges)

    x = structural_node_features(graph)
    edge_index = directed_edge_index(graph)
    node_mask = torch.zeros(graph.number_of_nodes(), dtype=torch.bool)
    if motif_nodes_all:
        node_mask[torch.tensor(motif_nodes_all, dtype=torch.long)] = True

    edge_mask_internal = edge_mask_for_edges(edge_index, motif_edges_internal)
    edge_mask_with_attachment = edge_mask_for_edges(
        edge_index, motif_edges_internal + attachment_edges
    )

    motif_node_tensor = torch.tensor(motif_nodes_all, dtype=torch.long)
    motif_edge_index = undirected_edge_pairs_to_directed_tensor(motif_edges_internal)
    attach_edge_index = undirected_edge_pairs_to_directed_tensor(attachment_edges)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([int(y)], dtype=torch.long),
        gt_explanation_node_mask=node_mask,
        gt_explanation_edge_mask=edge_mask_internal,
        gt_explanation_edge_mask_with_attachment=edge_mask_with_attachment,
        motif_node_id=torch.tensor(motif_id, dtype=torch.long),
        gt_motif_nodes=motif_node_tensor,
        gt_motif_edge_index=motif_edge_index,
        gt_attachment_edge_index=attach_edge_index,
        graph_id=torch.tensor([int(graph_id)], dtype=torch.long),
        dataset_id=torch.tensor([dataset_name_to_id(dataset_name)], dtype=torch.long),
        motif_type_id=torch.tensor([motif_name_to_id(motif_kind)], dtype=torch.long),
        num_inserted_motifs=torch.tensor([int(num_motifs)], dtype=torch.long),
        num_backbone_nodes=torch.tensor([int(n_backbone)], dtype=torch.long),
    )
    return data


def motif_template(kind: str, start: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    kind = str(kind).lower()
    if kind == "house":
        # 0--1--2--3--0 is a square; 1--4--2 is the roof.
        nodes = list(range(start, start + 5))
        rel_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (4, 2)]
    elif kind in {"cycle", "cycle6"}:
        nodes = list(range(start, start + 6))
        rel_edges = [(i, (i + 1) % 6) for i in range(6)]
    elif kind in {"star", "star5"}:
        nodes = list(range(start, start + 6))  # center + 5 leaves
        rel_edges = [(0, i) for i in range(1, 6)]
    elif kind in {"clique", "clique4"}:
        nodes = list(range(start, start + 4))
        rel_edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    elif kind in {"path", "path5"}:
        nodes = list(range(start, start + 5))
        rel_edges = [(i, i + 1) for i in range(4)]
    elif kind in {"bifan", "bifan23"}:
        nodes = list(range(start, start + 5))
        rel_edges = [(u, v) for u in range(2) for v in range(2, 5)]
    else:
        raise ValueError(f"Unknown motif kind {kind!r}")
    edges = [(start + u, start + v) for u, v in rel_edges]
    return nodes, edges


def motif_attachment_targets_for_kind(kind: str, nodes: Sequence[int]) -> List[int]:
    kind = str(kind).lower()
    nodes = [int(x) for x in nodes]
    if kind == "house":
        return [nodes[0], nodes[2], nodes[4]]
    if kind in {"cycle", "cycle6"}:
        return [nodes[0], nodes[2], nodes[4]]
    if kind in {"star", "star5"}:
        return [nodes[0], nodes[1], nodes[3]]
    if kind in {"clique", "clique4"}:
        return [nodes[0], nodes[1], nodes[2]]
    if kind in {"path", "path5"}:
        return [nodes[1], nodes[2], nodes[3]]
    if kind in {"bifan", "bifan23"}:
        return [nodes[0], nodes[1], nodes[2]]
    raise ValueError(f"Unknown motif kind {kind!r}")


def sample_attachment_pairs(
    rng: np.random.Generator,
    n_backbone: int,
    graph: nx.Graph,
    motif_targets: Sequence[int],
    attachment_mode: str,
) -> List[Tuple[int, int]]:
    attachment_mode = str(attachment_mode).lower()
    count_lookup = {"single": 1, "double": 2, "triple": 3}
    count = count_lookup.get(attachment_mode, 1)
    if n_backbone <= 0:
        return []
    available = np.arange(n_backbone)
    degrees = np.array([graph.degree(v) for v in range(n_backbone)], dtype=np.float64)
    prob = degrees + 1.0
    prob = prob / prob.sum()
    replace = bool(count > n_backbone)
    anchors = rng.choice(available, size=count, replace=replace, p=prob)
    pairs: List[Tuple[int, int]] = []
    for idx, anchor in enumerate(np.atleast_1d(anchors).tolist()):
        target = int(motif_targets[min(idx, len(motif_targets) - 1)])
        pairs.append((int(anchor), target))
    return pairs


def structural_node_features(graph: nx.Graph) -> Tensor:
    """Feature matrix computed from observable graph structure only.

    The ground-truth motif mask is deliberately not used as an input feature.
    The features are simple enough for GCN/GIN/GAT/GraphSAGE to train but do not
    directly reveal the class label.
    """

    n = graph.number_of_nodes()
    degree = np.array([graph.degree(i) for i in range(n)], dtype=np.float32)
    clustering = np.array([nx.clustering(graph, i) for i in range(n)], dtype=np.float32)
    triangles = np.array([nx.triangles(graph, i) for i in range(n)], dtype=np.float32)
    try:
        core = np.array([nx.core_number(graph)[i] for i in range(n)], dtype=np.float32)
    except Exception:
        core = np.zeros(n, dtype=np.float32)
    try:
        pagerank_dict = nx.pagerank(graph, alpha=0.85, max_iter=100)
        pagerank = np.array([pagerank_dict[i] for i in range(n)], dtype=np.float32)
    except Exception:
        pagerank = np.ones(n, dtype=np.float32) / max(1, n)

    def norm(arr: np.ndarray) -> np.ndarray:
        mx = float(arr.max()) if arr.size else 0.0
        return arr / mx if mx > 0 else arr

    x = np.stack(
        [
            np.ones(n, dtype=np.float32),
            norm(degree),
            clustering,
            norm(triangles),
            norm(core),
            norm(pagerank),
        ],
        axis=1,
    )
    return torch.tensor(x, dtype=torch.float32)


def directed_edge_index(graph: nx.Graph) -> Tensor:
    edges: List[Tuple[int, int]] = []
    for u, v in sorted(graph.edges()):
        u = int(u)
        v = int(v)
        if u == v:
            continue
        edges.append((u, v))
        edges.append((v, u))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def undirected_edge_pairs_to_directed_tensor(
    edges: Sequence[Tuple[int, int]]
) -> Tensor:
    directed: List[Tuple[int, int]] = []
    for u, v in edges:
        if int(u) == int(v):
            continue
        directed.append((int(u), int(v)))
        directed.append((int(v), int(u)))
    if not directed:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(directed, dtype=torch.long).t().contiguous()


def edge_mask_for_edges(
    edge_index: Tensor, undirected_edges: Sequence[Tuple[int, int]]
) -> Tensor:
    edge_set = set()
    for u, v in undirected_edges:
        edge_set.add((int(u), int(v)))
        edge_set.add((int(v), int(u)))
    mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
    for pos in range(edge_index.size(1)):
        u = int(edge_index[0, pos].item())
        v = int(edge_index[1, pos].item())
        if (u, v) in edge_set:
            mask[pos] = True
    return mask


def stratified_split_indices(
    labels: Sequence[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 0,
) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    labels_np = np.asarray(labels, dtype=np.int64)
    train: List[int] = []
    val: List[int] = []
    test: List[int] = []
    for cls in sorted(np.unique(labels_np).tolist()):
        idx = np.flatnonzero(labels_np == cls)
        rng.shuffle(idx)
        n = int(idx.size)
        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        if n >= 3:
            n_train = min(max(1, n_train), n - 2)
            n_val = min(max(1, n_val), n - n_train - 1)
        else:
            n_train = min(n_train, n)
            n_val = min(n_val, max(0, n - n_train))
        train.extend(int(x) for x in idx[:n_train])
        val.extend(int(x) for x in idx[n_train : n_train + n_val])
        test.extend(int(x) for x in idx[n_train + n_val :])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return {"train": train, "val": val, "test": test}


def dataset_name_to_id(name: str) -> int:
    names = tuple(DATASET_CONFIGS.keys())
    return names.index(str(name)) if str(name) in names else -1


def motif_name_to_id(name: str) -> int:
    names = ("none", "house", "cycle6", "star5", "clique4", "path5", "bifan23")
    return names.index(str(name)) if str(name) in names else -1
