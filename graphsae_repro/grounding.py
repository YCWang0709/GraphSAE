"""Deterministic regression groups, computed before propagation/interventions."""

import torch


def logarithmic_groups(counts):
    counts = torch.as_tensor(counts)
    if (counts < 0).any():
        raise ValueError("Structural counts must be nonnegative")
    return torch.floor(torch.log2(1 + counts.double())).long()


def city_roads_groups(edge_index, num_nodes):
    return logarithmic_groups(torch.bincount(edge_index[1], minlength=num_nodes))


def relsc_group(graph):
    """Count every node type for HeteroData, or every node for homogeneous Data."""
    count = sum(store.num_nodes for store in graph.node_stores)
    return int(logarithmic_groups([count])[0])
