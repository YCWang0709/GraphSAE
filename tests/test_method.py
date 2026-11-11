"""Equation-level regression tests for the GraphSAE reproduction."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from graphsae_repro.backbones import build_backbone
from graphsae_repro.contributions import node_source_contributions
from graphsae_repro.explainer import explain_node
from graphsae_repro.sae import SAEArtifact, SAEConfig
from graphsae_repro.sae import TopKSAE, decoder_coherence_loss, grounding_entropy_loss


def test_topk_sae_enforces_nonnegative_budget() -> None:
    sae = TopKSAE(input_dim=4, latent_dim=12, k_active=3)
    codes = sae.encode(torch.randn(7, 4))
    assert bool((codes >= 0).all())
    assert bool(((codes > 0).sum(dim=-1) <= 3).all())


def test_grounding_entropy_rewards_group_concentration() -> None:
    groups = torch.tensor([0, 0, 1, 1])
    concentrated = torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 1.0]])
    mixed = torch.ones_like(concentrated)
    assert grounding_entropy_loss(concentrated, groups) < grounding_entropy_loss(
        mixed, groups
    )


def test_decoder_coherence_is_zero_for_orthogonal_columns() -> None:
    decoder = torch.eye(4)
    assert torch.isclose(decoder_coherence_loss(decoder), torch.tensor(0.0))


def test_gcn_source_terms_reconstruct_bias_free_target_state() -> None:
    model = build_backbone(
        "GCN",
        in_channels=2,
        out_channels=2,
        hidden_channels=2,
        num_layers=1,
        dropout=0.0,
        task="node",
    )
    with torch.no_grad():
        model.convs[0].lin.weight.copy_(torch.eye(2))
        model.convs[0].bias.zero_()
    data = Data(
        x=torch.tensor([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
    )
    model.eval()
    _, capture = model(data.x, data.edge_index, capture=True)
    _, contributions = node_source_contributions(
        model,
        data,
        target_node=1,
        exclude_target=False,
    )
    assert torch.allclose(contributions.sum(dim=0), capture.node[1], atol=1e-6)


@pytest.mark.parametrize("model_name", ["GCN", "GAT", "GIN", "GraphSAGE"])
def test_node_explainer_supports_each_paper_backbone(model_name: str) -> None:
    model = build_backbone(
        model_name,
        in_channels=3,
        out_channels=2,
        hidden_channels=4,
        num_layers=2,
        dropout=0.0,
        gat_heads=2,
        task="node",
    ).eval()
    artifact = SAEArtifact(
        sae=TopKSAE(4, 8, 3),
        mean=torch.zeros(1, 4),
        std=torch.ones(1, 4),
        config=SAEConfig(latent_dim=8, k_active=3),
    )
    data = Data(
        x=torch.rand(4, 3),
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]],
            dtype=torch.long,
        ),
    )
    result = explain_node(model, data, artifact, target_node=0, top_k=2)
    assert result["task"] == "node"
    assert len(result["node_scores"]) == 3
