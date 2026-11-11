"""Checks for the pretrained Cora inference artifact and its evaluation scope."""

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from examples.cora_pretrained.inference import (
    FrozenSAE,
    checked_load,
    extract_nodes,
    load_manifest,
    load_predictor,
)
from examples.cora_pretrained.run import aggregate, main


HERE = Path(__file__).resolve().parents[1] / "examples/cora_pretrained"


@pytest.mark.parametrize("name", ["GCN", "GIN", "GAT", "GraphSAGE"])
def test_bundled_predictor_and_three_sae_checkpoints(name):
    manifest = load_manifest(HERE)
    entry = manifest["models"][name]
    model = load_predictor(HERE, entry)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert len(entry["test_targets"]) == len(set(entry["test_targets"])) == 40
    assert not set(entry["test_targets"]) & set(entry["validation_targets"])
    assert entry["sae_layer"] == model.num_layers - 1
    for seed in (0, 1, 2):
        payload = checked_load(HERE, entry["saes"][str(seed)])
        sae = FrozenSAE(payload)
        assert payload["seed"] == seed
        assert sae.encoder.in_features == model.hidden_channels
        assert sae.k_active == entry["saes"][str(seed)]["k_active"]
        assert (sae.std > 0).all()


def test_partial_run_cannot_claim_paper_row():
    with pytest.raises(SystemExit) as error:
        main(["--models", "GCN", "--check-paper"])
    assert error.value.code == 2


def test_equal_backbone_and_within_target_seed_aggregation():
    def row(model, target, seed, value):
        return {
            "model": model,
            "target_node": target,
            "seed": seed,
            "logit_drop": value,
            "probability_drop": value / 100,
            "num_nodes": value,
        }

    rows = [
        row("A", 0, 0, 0),
        row("A", 0, 1, 2),
        row("A", 1, 0, 10),
        row("B", 3, 0, 100),
    ]
    models, overall = aggregate(rows)
    assert models["A"]["logit_drop"] == 5.5
    assert overall["logit_drop"] == 52.75
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate(rows + rows[:1])


def test_greedy_rules_and_deterministic_score_ties():
    class Oracle:
        local = None
        values = {
            (): 0.0,
            (1,): 0.3,
            (2,): 0.2,
            (3,): 0.1,
            (1, 2): 0.4,
            (1, 3): 0.35,
            (1, 2, 3): 0.4,
        }

        def native_value(self, nodes):
            return self.values[tuple(sorted(nodes))]

        value = native_value

    scores = torch.tensor([1.0, 1.0, -1.0])
    assert extract_nodes([2, 1, 3], scores, Oracle(), "gainstop_0") == [1, 2]
    assert extract_nodes([2, 1, 3], scores, Oracle(), "greedy_3") == [1, 2, 3]


def test_one_hop_visualization_exports_both_formats(tmp_path):
    from examples.cora_pretrained.visualize import draw_explanation

    data = Data(
        x=torch.ones(4, 2),
        y=torch.tensor([0, 0, 1, 1]),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
    )
    record = {"model": "GCN", "seed": 0, "target_node": 0, "selected_nodes": [1, 2]}
    paths = draw_explanation(data, record, tmp_path)
    assert {p.suffix for p in paths} == {".png", ".svg"}
    assert all(p.stat().st_size > 1000 for p in paths)
