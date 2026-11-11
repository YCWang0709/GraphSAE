"""Construction checks against the supplementary synthetic-data protocol."""

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

from graphsae_repro import SYNTHETIC_DATASETS, SyntheticDataset, generate_synthetic
from graphsae_repro.synthetic_node import _sample_backbone_anchor


SMALL_OPTIONS = {
    "BATriadic": {"num_nodes": 30},
    "BAPathRole": {"backbone_nodes": 20, "num_motifs": 6},
    "BABiFanRole": {"backbone_nodes": 20, "num_motifs": 6},
    "BAHouse": {"num_graphs": 12},
    "BAPath": {"num_graphs": 12},
    "BACycle": {"num_graphs": 12},
}


@pytest.mark.parametrize("name", SYNTHETIC_DATASETS)
def test_seeded_generation_and_storage_preserve_data(name, tmp_path):
    first = generate_synthetic(name, **SMALL_OPTIONS[name])
    second = generate_synthetic(name, **SMALL_OPTIONS[name])
    assert first.splits == second.splits
    a = [first.data] if first.task == "node" else first.graphs
    b = [second.data] if second.task == "node" else second.graphs
    for left, right in zip(a, b):
        assert torch.equal(left.x, right.x)
        assert torch.equal(left.edge_index, right.edge_index)
        assert torch.equal(left.y, right.y)
        assert torch.isfinite(left.x).all()
    population_size = (
        first.data.num_nodes if first.task == "node" else len(first.graphs)
    )
    flattened = [i for ids in first.splits.values() for i in ids]
    assert sorted(flattened) == list(range(population_size))
    assert all(first.splits[split] for split in ("train", "val", "test"))
    restored = SyntheticDataset.load(first.save(tmp_path / name))
    assert restored.config == first.config and restored.splits == first.splits
    restored_population = [restored.data] if first.task == "node" else restored.graphs
    for original, loaded in zip(a, restored_population):
        assert set(original.keys()) == set(loaded.keys())
        for key in original.keys():
            if isinstance(original[key], torch.Tensor):
                assert torch.equal(original[key], loaded[key])


@pytest.mark.parametrize(
    "name,role_labels,internal_edges",
    [
        ("BAPathRole", [2, 1, 1, 1, 2], 4),
        ("BABiFanRole", [1, 1, 2, 2, 2], 6),
    ],
)
def test_node_motif_topology_labels_and_reference_masks(
    name, role_labels, internal_edges
):
    generated = generate_synthetic(name, **SMALL_OPTIONS[name])
    data = generated.data
    assert data.x.shape == (50, 13)
    assert data.y[:20].eq(0).all()
    assert torch.equal(data.y[20:], torch.tensor(role_labels).repeat(6))
    assert data.gt_explanation_node_mask.sum() == 30
    assert data.gt_explanation_edge_mask.sum() == 6 * internal_edges * 2
    assert (
        data.gt_explanation_edge_mask_with_attachment.sum()
        == 6 * (internal_edges + 1) * 2
    )
    assert generated.reference(20) == [21, 22, 23, 24]
    assert generated.reference(0) == []
    assert generated.reference(20, [19, 20, 21, 27]) == [21]
    edges = set(map(tuple, data.edge_index.T.tolist()))
    if name == "BAPathRole":
        expected = {(20 + i, 21 + i) for i in range(4)}
    else:
        expected = {(u, v) for u in (20, 21) for v in (22, 23, 24)}
    assert expected | {(v, u) for u, v in expected} <= edges


@pytest.mark.parametrize(
    "name,motif_nodes,motif_edges",
    [("BAHouse", 5, 6), ("BAPath", 5, 4), ("BACycle", 6, 6)],
)
def test_graph_count_labels_and_masks(name, motif_nodes, motif_edges):
    generated = generate_synthetic(name, **SMALL_OPTIONS[name])
    labels = torch.tensor([int(data.y.item()) for data in generated.graphs])
    assert torch.bincount(labels).tolist() == [4, 4, 4]
    for split in generated.splits.values():
        assert set(labels[split].tolist()) == {0, 1, 2}
    for i, data in enumerate(generated.graphs):
        count = int(data.y.item())
        assert 22 <= int(data.num_backbone_nodes.item()) <= 38
        assert data.num_features == 6
        assert data.gt_explanation_node_mask.sum() == count * motif_nodes
        assert data.gt_explanation_edge_mask.sum() == count * motif_edges * 2
        assert (
            data.gt_explanation_edge_mask_with_attachment.sum()
            == count * (motif_edges + 1) * 2
        )
        assert (
            generated.reference(i)
            == torch.where(data.gt_explanation_node_mask)[0].tolist()
        )


def test_triadic_generator_and_rank_weighting_match_protocol():
    generated = generate_synthetic("BATriadic", num_nodes=30)
    original = nx.powerlaw_cluster_graph(30, 3, 0.35, seed=37)
    expected_edges = {(int(u), int(v)) for u, v in original.edges()}
    expected_edges |= {(v, u) for u, v in expected_edges}
    assert set(map(tuple, generated.data.edge_index.T.tolist())) == expected_edges
    rng = np.random.default_rng(37)

    def ranks(values):
        order = np.argsort(np.asarray(values) + rng.uniform(-1e-9, 1e-9, 30))
        result = np.empty(30)
        result[order] = np.arange(30) / 29
        return result

    degree_rank = ranks(np.log1p([original.degree(i) for i in range(30)]))
    triangle_rank = ranks(np.log1p([nx.triangles(original, i) for i in range(30)]))
    scores = 0.65 * degree_rank + 0.35 * triangle_rank
    labels = np.minimum(np.floor(ranks(scores) * 3), 2).astype(np.int64)
    assert torch.equal(generated.data.y, torch.tensor(labels))
    assert generated.data.num_features == 18
    target = generated.splits["val"][0]
    candidates = list(range(30))
    expected = [
        u
        for u in candidates
        if u != target
        and generated.data.y[u] == generated.data.y[target]
        and bool(generated.data.train_mask[u] | generated.data.val_mask[u])
    ]
    assert generated.reference(target, candidates, split="val") == expected


def test_anchor_mixture_uses_degree_plus_one_or_uniform():
    class ControlledRNG:
        def __init__(self, value):
            self.value = value
            self.probabilities = None
            self.uniform_bounds = None

        def random(self):
            return self.value

        def choice(self, nodes, p):
            self.probabilities = p
            return nodes[0]

        def integers(self, low, high):
            self.uniform_bounds = low, high
            return low

    backbone = nx.path_graph(4)
    preferential, uniform = ControlledRNG(0.549), ControlledRNG(0.55)
    _sample_backbone_anchor(backbone, 4, preferential)
    _sample_backbone_anchor(backbone, 4, uniform)
    assert np.allclose(preferential.probabilities, np.array([2, 3, 3, 2]) / 10)
    assert uniform.uniform_bounds == (0, 4) and uniform.probabilities is None


def test_default_node_sizes_and_stratified_ratios():
    for name in ("BATriadic", "BAPathRole", "BABiFanRole"):
        generated = generate_synthetic(name)
        assert generated.data.num_nodes == 2000
        for label in range(3):
            total = int((generated.data.y == label).sum())
            assert int(
                (generated.data.y[generated.splits["train"]] == label).sum()
            ) == round(0.6 * total)
            assert int(
                (generated.data.y[generated.splits["val"]] == label).sum()
            ) == round(0.2 * total)


def test_graph_default_splits_and_class_populations():
    from graphsae_repro.synthetic_graph import stratified_split_indices

    labels = [i % 3 for i in range(900)]
    splits = stratified_split_indices(labels, seed=101)
    assert [len(splits[k]) for k in ("train", "val", "test")] == [630, 135, 135]
    for ids in splits.values():
        assert len({labels[i] for i in ids}) == 3


def test_all_dataset_cli_and_protocol_configuration(tmp_path):
    from graphsae_repro.cli import main

    defaults = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/synthetic_datasets.json"
        ).read_text()
    )
    seeds = {
        "BATriadic": 37,
        "BAPathRole": 43,
        "BABiFanRole": 47,
        "BAHouse": 101,
        "BAPath": 404,
        "BACycle": 202,
    }
    assert {
        name: options["seed"] for name, options in defaults["datasets"].items()
    } == seeds
    for name in ("BAHouse", "BAPath", "BACycle"):
        assert defaults["datasets"][name]["num_graphs"] == 900
    config = tmp_path / "options.json"
    config.write_text(json.dumps({"datasets": SMALL_OPTIONS}), encoding="utf-8")
    main(
        [
            "generate-data",
            "--dataset",
            "all",
            "--config",
            str(config),
            "--output",
            str(tmp_path / "data"),
        ]
    )
    for name in SYNTHETIC_DATASETS:
        loaded = SyntheticDataset.load(tmp_path / "data" / name)
        metadata = json.loads((tmp_path / "data" / name / "metadata.json").read_text())
        assert metadata["name"] == loaded.name == name
        assert metadata["config"]["seed"] == seeds[name]


@pytest.mark.parametrize(
    "name,options",
    [
        ("unknown", {}),
        ("BATriadic", {"triangle_probability": 1.1}),
        ("BATriadic", {"seed": -1}),
        ("BAPathRole", {"num_motifs": 0}),
        ("BAPathRole", {"backbone_nodes": 3}),
        ("BAHouse", {"num_graphs": 10}),
        ("BACycle", {"ba_min_nodes": 40, "ba_max_nodes": 38}),
        ("BAPath", {"triangle_probability": 0.35}),
    ],
)
def test_invalid_construction_options_are_rejected(name, options):
    with pytest.raises(ValueError):
        generate_synthetic(name, **options)
