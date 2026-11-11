from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch, Data

from graphsae_repro.backbones import build_backbone
from graphsae_repro.contributions import graph_source_contributions, source_scores
from graphsae_repro.datasets import node_reference
from graphsae_repro.explainer import explain_graph, target_direction
from graphsae_repro.policies import ExtractionPolicy
from graphsae_repro.sae import SAEArtifact, SAEConfig, TopKSAE


@pytest.mark.parametrize("name", ["BATriadic", "BAPathRole", "BABiFanRole"])
def test_same_label_reference_excludes_target_and_validation_test_sources(name):
    data = Data(
        y=torch.tensor([2, 1, 2, 2, 1, 2]),
        train_mask=torch.tensor([True, True, False, False, False, False]),
        val_mask=torch.tensor([False, False, True, True, False, False]),
        test_mask=torch.tensor([False, False, False, False, True, True]),
        motif_node_id=torch.tensor([-1, -1, 0, 1, 1, 2]),
    )
    assert node_reference(name, data, 2, range(6)) == [0, 3, 5]
    assert node_reference(name, data, 2, range(6), split="val") == [0, 3]
    assert node_reference(name, data, 2, [1, 2, 4]) == []


@pytest.mark.parametrize("name", ["GCN", "GIN", "GAT", "GraphSAGE"])
@pytest.mark.parametrize("graph_index", [0, 1])
def test_readout_mode_uses_final_nodes_and_per_graph_mean(name, graph_index):
    torch.manual_seed(9)
    data = Batch.from_data_list(
        [
            Data(
                x=torch.rand(3, 2),
                edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            ),
            Data(x=torch.rand(2, 2), edge_index=torch.tensor([[0, 1], [1, 0]])),
        ]
    )
    model = build_backbone(
        name,
        2,
        2,
        hidden_channels=4,
        num_layers=2,
        task="graph",
        dropout=0.0,
        gat_heads=2,
    ).eval()
    with torch.no_grad():
        _, capture = model(data.x, data.edge_index, batch=data.batch, capture=True)
    ids, terms = graph_source_contributions(
        model, data, graph_index=graph_index, source_mode="readout"
    )
    expected_ids = [0, 1, 2] if graph_index == 0 else [3, 4]
    assert ids == expected_ids
    torch.testing.assert_close(terms * len(ids), capture.node[ids])
    torch.testing.assert_close(terms.sum(0), capture.graph[graph_index])
    subset_ids, subset = graph_source_contributions(
        model,
        data,
        graph_index=graph_index,
        candidate_nodes=ids[:1],
        source_mode="readout",
    )
    assert subset_ids == ids[:1]
    torch.testing.assert_close(subset, terms[:1])
    with pytest.raises(ValueError, match="do not belong"):
        graph_source_contributions(
            model,
            data,
            graph_index=graph_index,
            candidate_nodes=[3 if graph_index == 0 else 0],
            source_mode="readout",
        )
    direction = torch.tensor([0.4, -0.7, 1.2, 0.8])
    scored_ids, scores = source_scores(
        model, data, direction, target_index=graph_index, source_mode="readout"
    )
    assert scored_ids == ids
    torch.testing.assert_close(scores, terms @ direction)
    artifact = SAEArtifact(
        TopKSAE(4, 8, 3),
        torch.zeros(1, 4),
        torch.ones(1, 4),
        SAEConfig(latent_dim=8, k_active=3),
    )
    policy = ExtractionPolicy(budget=1, source_mode="readout")
    result = explain_graph(
        model, data, artifact, graph_index=graph_index, policy=policy
    )
    target, _ = target_direction(artifact, capture.graph[graph_index])
    torch.testing.assert_close(
        torch.tensor([s["signed_score"] for s in result["node_scores"]]),
        terms @ target,
    )
    assert set(result["selected_nodes"]) <= set(ids)
    assert len(result["selected_nodes"]) == 1
    assert result["policy"] == asdict(policy)


def test_readout_and_input_modes_keep_distinct_contributions():
    model = build_backbone(
        "GCN", 2, 2, hidden_channels=2, num_layers=1, task="graph", dropout=0.0
    ).eval()
    with torch.no_grad():
        model.convs[0].lin.weight.copy_(torch.eye(2))
        model.convs[0].bias.fill_(2.0)
    data = Data(
        x=torch.tensor([[1.0, 0.0], [0.0, 3.0]]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
    )
    _, inputs = graph_source_contributions(model, data)
    _, readout = graph_source_contributions(model, data, source_mode="readout")
    assert not torch.allclose(inputs, readout)
    torch.testing.assert_close(readout.sum(0) - inputs.sum(0), torch.full((2,), 2.0))
    node_model = build_backbone("GCN", 2, 2, hidden_channels=2, num_layers=1)
    with pytest.raises(ValueError, match="graph-level"):
        source_scores(node_model, data, torch.ones(2), source_mode="readout")
    with pytest.raises(ValueError, match="source_mode"):
        ExtractionPolicy(source_mode="unknown")


def test_readout_search_configuration_loads_as_extraction_policies():
    root = Path(__file__).resolve().parents[1]
    for name in ["policies_graph_readout.json", "policies_synthetic_graph_native.json"]:
        policies = [
            ExtractionPolicy(**p)
            for p in json.loads((root / "configs" / name).read_text())
        ]
        assert policies and all(p.source_mode == "readout" for p in policies)
