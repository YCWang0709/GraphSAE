"""Synthetic benchmark generators and molecular reference-mask utilities."""

import torch
from .synthetic import GRAPH_DATASETS, NODE_DATASETS, generate_synthetic


def synthetic_node_dataset(name, **options):
    if name not in NODE_DATASETS:
        raise ValueError(f"Expected a node dataset from {NODE_DATASETS}")
    return generate_synthetic(name, **options).data


def synthetic_graph_dataset(name, *, num_graphs=900, **options):
    if name not in GRAPH_DATASETS:
        raise ValueError(f"Expected a graph dataset from {GRAPH_DATASETS}")
    dataset = generate_synthetic(name, num_graphs=num_graphs, **options)
    return dataset.graphs, dataset.splits


def node_reference(name, data, target, candidates, *, split="test", backbone_nodes=750):
    """Same-label reference nodes within the supplied source field.

    Validation role references use only train/validation source labels.
    """
    if name not in NODE_DATASETS:
        raise ValueError(f"Expected a node dataset from {NODE_DATASETS}")
    eligible = candidates
    if split == "val":
        eligible = [
            u for u in candidates if bool(data.train_mask[u] | data.val_mask[u])
        ]
    return [u for u in eligible if u != target and data.y[u] == data.y[target]]


def mutag_proxy_mask(atom_types, edge_index, *, nitrogen=1, oxygen=4):
    """NH2 / NO2 proxy on hydrogen-suppressed MUTAG atom types/connectivity."""
    n = len(atom_types)
    adjacency = [set() for _ in range(n)]
    for u, v in edge_index.T.tolist():
        if u != v:
            adjacency[u].add(v)
            adjacency[v].add(u)
    selected = set()
    for u in range(n):
        if int(atom_types[u]) != nitrogen:
            continue
        if len(adjacency[u]) == 1:
            selected.add(u)
        if len(adjacency[u]) == 3:
            terminal_oxygen = [
                v
                for v in adjacency[u]
                if int(atom_types[v]) == oxygen and len(adjacency[v]) == 1
            ]
            if len(terminal_oxygen) >= 2:
                selected.update([u, *terminal_oxygen])
    mask = torch.zeros(n, dtype=torch.bool)
    mask[sorted(selected)] = True
    return mask


def sdf_reference_mask(atom_count, atom_indices):
    """Use task-supplied SDF atom indices without sorting/remapping atom order."""
    indices = torch.as_tensor(atom_indices, dtype=torch.long)
    if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= atom_count):
        raise IndexError("Reference atom index outside the original molecular graph")
    mask = torch.zeros(atom_count, dtype=torch.bool)
    mask[indices] = True
    return mask
