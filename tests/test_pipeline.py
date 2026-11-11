"""Numerical and protocol checks independent of experiment results."""

from dataclasses import replace
import torch
import pytest
from torch_geometric.data import Data, Batch, HeteroData

from graphsae_repro.activations import ActivationPack
from graphsae_repro.backbones import build_backbone
from graphsae_repro.contributions import (
    node_source_contributions,
    graph_source_contributions,
    source_scores,
)
from graphsae_repro.explainer import explain_node, explain_graph, target_direction
from graphsae_repro.sae import (
    SAEConfig,
    SAEArtifact,
    TopKSAE,
    train_sae,
    faithfulness_loss,
)
from graphsae_repro.tuning import fit_sae, tune_sae, tune_pipeline, ValidationTarget
from graphsae_repro.policies import ExtractionPolicy, select_nodes, refine_nodes
from graphsae_repro.masking import clone_with_node_mask
from graphsae_repro.adapters import FrozenModelAdapter, typed_deletion
from graphsae_repro.ablations import loss_configurations, pure_path_scores
from graphsae_repro.grounding import logarithmic_groups


def graph():
    return Data(
        x=torch.tensor(
            [[1.0, 0.5, 0.2], [0.3, 0.8, 0.4], [0.9, 0.1, 0.7], [0.2, 0.4, 0.6]]
        ),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
    )


@pytest.mark.parametrize("name", ["GCN", "GAT", "GIN", "GraphSAGE"])
@pytest.mark.parametrize("task", ["node", "graph"])
def test_reverse_projection_equals_explicit_source_recursion(name, task):
    torch.manual_seed(19)
    model = build_backbone(
        name, 3, 2, hidden_channels=4, num_layers=2, gat_heads=2, dropout=0.0, task=task
    ).eval()
    data = graph()
    direction = torch.tensor([0.4, -0.7, 1.2, 0.8])
    if task == "node":
        ids, terms = node_source_contributions(model, data, target_node=1)
        scored_ids, scores = source_scores(model, data, direction, target_index=1)
    else:
        ids, terms = graph_source_contributions(model, data)
        scored_ids, scores = source_scores(model, data, direction)
    assert ids == scored_ids
    assert torch.allclose(scores, terms @ direction, atol=2e-6)


@pytest.mark.parametrize("name", ["GCN", "GAT", "GIN", "GraphSAGE"])
def test_generic_adapter_agrees_with_explicit_decomposition(name):
    torch.manual_seed(4)
    model = build_backbone(
        name, 3, 2, hidden_channels=4, num_layers=2, gat_heads=2, dropout=0.0
    ).eval()
    data = graph()

    def representation(d, target):
        # patch_last capture is detached; evaluate the message-passing layers
        # directly to expose a differentiable representation for custom models.
        h = d.x
        for conv in model.convs:
            h = conv(h, d.edge_index).relu()
        return h[target]

    adapter = FrozenModelAdapter(
        model,
        representation,
        lambda d: model(d.x, d.edge_index),
        lambda d, t: [u for u in range(4) if u != t],
    )
    direction = torch.tensor([0.3, 1.0, -0.6, 0.8])
    ids, scores = adapter.project_sources(data, 1, direction)
    expected_ids, expected = source_scores(model, data, direction, target_index=1)
    assert ids == expected_ids
    assert torch.allclose(scores, expected, atol=2e-6)
    assert torch.allclose(adapter.predict(data), model(data.x, data.edge_index))


def pack(split, offset=0):
    torch.manual_seed(offset)
    return ActivationPack(
        torch.randn(12, 4),
        [{"sample_id": i + offset, "label": i % 2, "split": split} for i in range(12)],
        "node",
        info={"split": split},
    )


@pytest.mark.parametrize("encoder", ["topk", "relu", "batchtopk", "jumprelu", "gated"])
def test_training_and_checkpoint_roundtrip(encoder, tmp_path):
    config = SAEConfig(
        latent_dim=8,
        k_active=3,
        epochs=2,
        batch_size=6,
        encoder_type=encoder,
        l1_weight=0.01 if encoder in {"relu", "gated"} else 0.0,
    )
    head = torch.nn.Linear(4, 2)
    before = {k: v.clone() for k, v in head.state_dict().items()}
    artifact = train_sae(pack("train"), config, prediction_head=head)
    assert all(p.requires_grad for p in head.parameters())
    assert all(torch.equal(v, head.state_dict()[k]) for k, v in before.items())
    assert torch.isfinite(torch.tensor([r["loss"] for r in artifact.history])).all()
    assert torch.allclose(
        artifact.sae.decoder.weight.norm(dim=0), torch.ones(8), atol=1e-6
    )
    path = artifact.save(tmp_path / "sae.pt")
    loaded = SAEArtifact.load(path)
    assert torch.equal(artifact.encode(pack("val").z), loaded.encode(pack("val").z))


def test_train_only_normalization_and_validation_search():
    train, val = pack("train"), pack("val", 100)
    val.z += 1000
    config = SAEConfig(
        latent_dim=8,
        k_active=3,
        epochs=1,
        lambda_ground=0.0,
        lambda_faith=0.0,
        lambda_coh=0.0,
    )
    artifact, protocol = tune_sae(train, val, [config, replace(config, lr=0.01)])
    assert torch.equal(artifact.mean, train.z.mean(0, keepdim=True))
    values = [row["validation"]["nmse"] for row in protocol["trials"]]
    assert protocol["selected_index"] == values.index(min(values))
    with pytest.raises(ValueError, match="split"):
        tune_sae(train, pack("test", 100), [config])
    with pytest.raises(ValueError, match="training"):
        train_sae(val, config)


def test_general_sae_needs_neither_labels_nor_head():
    artifact = fit_sae(
        torch.randn(9, 4),
        SAEConfig(
            epochs=1, k_active=3, lambda_ground=0.0, lambda_faith=0.0, lambda_coh=0.0
        ),
    )
    assert artifact.encode(torch.randn(2, 4)).shape == (2, 32)


def test_regression_and_binary_faithfulness_have_gradients():
    reconstructed = torch.tensor([[1.0], [2.0]], requires_grad=True)
    loss = faithfulness_loss(torch.zeros(2, 1), reconstructed, "regression")
    assert torch.isclose(loss, torch.tensor(2.5))
    loss.backward()
    assert reconstructed.grad.abs().sum() > 0
    assert faithfulness_loss(reconstructed.detach(), reconstructed.detach()) < 1e-6


@pytest.mark.parametrize("evidence", ["positive", "negative", "mixed"])
def test_matched_budget_and_deterministic_shortage_padding(evidence):
    ids, scores = [4, 1, 7, 2, 9], torch.tensor([2.0, -1.0, 0.0, 0.0, 0.0])
    selected = select_nodes(ids, scores, ExtractionPolicy(budget=4, evidence=evidence))
    assert len(selected) == len(set(selected)) == 4
    shuffled = [3, 1, 0, 4, 2]
    assert selected == select_nodes(
        [ids[i] for i in shuffled],
        scores[shuffled],
        ExtractionPolicy(budget=4, evidence=evidence),
    )
    assert (
        select_nodes(ids, scores, ExtractionPolicy(budget=0, evidence=evidence)) == []
    )


def test_deletion_keeps_slots_and_edge_attribute_alignment():
    data = graph()
    data.edge_attr = torch.arange(data.num_edges)[:, None]
    data.batch = torch.zeros(4, dtype=torch.long)
    out = clone_with_node_mask(data, [1])
    assert out.num_nodes == data.num_nodes
    assert torch.equal(out.batch, data.batch)
    assert out.x[1].count_nonzero() == 0 and data.x[1].count_nonzero() > 0
    assert not (out.edge_index == 1).any()
    assert out.edge_attr.shape[0] == out.num_edges


def test_typed_deletion_preserves_all_relations():
    data = HeteroData()
    data["a"].x, data["b"].x = torch.ones(3, 2), torch.ones(2, 2)
    data["a", "r", "b"].edge_index = torch.tensor([[0, 1, 2], [0, 1, 0]])
    data["b", "s", "a"].edge_index = torch.tensor([[0, 1], [1, 2]])
    out = typed_deletion(data, [("a", 1)])
    assert out.node_types == data.node_types and out.edge_types == data.edge_types
    assert out["a"].num_nodes == 3 and out["a"].x[1].sum() == 0
    assert out["a", "r", "b"].num_edges == 2
    assert out["b", "s", "a"].num_edges == 1


def test_pure_path_is_standardized_target_projection():
    model = build_backbone(
        "GCN", 3, 2, hidden_channels=4, num_layers=1, dropout=0.0
    ).eval()
    artifact = SAEArtifact(
        TopKSAE(4, 8, 3),
        torch.ones(1, 4),
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        SAEConfig(),
    )
    data = graph()
    _, cap = model(data.x, data.edge_index, capture=True)
    ids, terms = node_source_contributions(model, data, target_node=1)
    actual_ids, actual = pure_path_scores(model, data, artifact, target_index=1)
    expected = (terms / artifact.std) @ (
        (cap.node[1] - artifact.mean[0]) / artifact.std[0]
    )
    assert actual_ids == ids and torch.allclose(actual, expected, atol=1e-6)


def test_logarithmic_bucket_boundaries():
    assert logarithmic_groups([0, 1, 2, 3, 6, 7, 14, 15]).tolist() == [
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
    ]


def test_no_full_baseline_is_duplicated_in_loss_ablations():
    assert "full" not in loss_configurations()
    assert len(loss_configurations()) == 5


def test_end_to_end_selection_and_refinement():
    from examples.toy import load

    experiment = load(epochs=1)
    config = SAEConfig(epochs=1, expansion=4, k_active=3)
    selected, policy, protocol = tune_pipeline(
        experiment.model,
        experiment.train,
        experiment.validation_targets,
        [config],
        [ExtractionPolicy()],
        budgets=(2, 3),
    )
    artifact = train_sae(
        experiment.train, selected, prediction_head=experiment.model.classifier
    )
    target = experiment.test_targets[0]
    result = explain_node(
        experiment.model,
        target.data,
        artifact,
        target_node=target.target_index,
        top_k=3,
    )
    assert (
        len(result["sae_positive"]) == 3
        and target.target_index not in result["selected_nodes"]
    )
    with pytest.raises(ValueError, match="validation"):
        tune_pipeline(
            experiment.model,
            experiment.train,
            experiment.test_targets,
            [config],
            [policy],
        )
    assert refine_nodes(
        [],
        [0, 1, 2],
        [3.0, 2.0, 1.0],
        lambda ids: float(1 in ids) - 0.1 * len(ids),
        steps=4,
    ) == [1]


def test_graph_batch_explanation_does_not_select_other_graph_nodes():
    from examples.toy import load

    experiment = load(task="graph", epochs=1, model_name="GAT")
    artifact = train_sae(
        experiment.train,
        SAEConfig(epochs=1, k_active=3),
        prediction_head=experiment.model.classifier,
    )
    data = Batch.from_data_list([t.data for t in experiment.test_targets])
    result = explain_graph(experiment.model, data, artifact, graph_index=1, top_k=3)
    assert all(int(data.batch[u]) == 1 for u in result["selected_nodes"])


def test_feature_ablation_uses_unablated_reconstruction_as_reference():
    from graphsae_repro.alignment import run_node_feature_ablation

    model = build_backbone(
        "GCN", 3, 2, hidden_channels=4, num_layers=1, dropout=0.0
    ).eval()
    artifact = SAEArtifact(
        TopKSAE(4, 8, 3), torch.zeros(1, 4), torch.ones(1, 4), SAEConfig()
    )
    with torch.no_grad():
        artifact.sae.encoder.weight.zero_()
        artifact.sae.encoder.bias.zero_()
    result = run_node_feature_ablation(
        model, graph(), artifact, target_node=1, feature_ids=[0]
    )
    assert result["delta_reference_logit"] == 0.0
    assert result["delta_reference_probability"] == 0.0


def test_regression_training_uses_structural_groups():
    train = pack("train")
    for i, row in enumerate(train.metadata):
        row["label"] = float(i) / 7
        row["group"] = i % 3
    config = SAEConfig(epochs=1, k_active=3, task_type="regression")
    artifact = train_sae(train, config, prediction_head=torch.nn.Linear(4, 1))
    assert len(artifact.group_vocab) == 3
    assert artifact.history[0]["faith"] >= 0.0


def test_stable_sage_matches_direct_surviving_neighbor_mean():
    from graphsae_repro.numerics import stable_sage_deletion

    model = build_backbone(
        "GraphSAGE", 3, 2, hidden_channels=4, num_layers=2, dropout=0.0
    ).eval()
    data = clone_with_node_mask(graph(), [1])
    original = model(data.x, data.edge_index)
    with stable_sage_deletion(model):
        replay = model(data.x, data.edge_index)
    assert torch.allclose(original, replay, atol=1e-6)


def test_synthetic_generators_preserve_masks_and_splits():
    from graphsae_repro.datasets import synthetic_node_dataset, synthetic_graph_dataset

    data = synthetic_node_dataset("BAPathRole", backbone_nodes=30, num_motifs=6)
    assert len(data.x) == 60
    assert (
        data.train_mask.int() + data.val_mask.int() + data.test_mask.int() == 1
    ).all()
    graphs, splits = synthetic_graph_dataset("BAHouse", num_graphs=12)
    assert len(graphs) == 12
    assert sorted(splits["train"] + splits["val"] + splits["test"]) == list(range(12))
    assert all(g.gt_explanation_edge_mask.shape[0] == g.num_edges for g in graphs)
    assert all(g.num_features == 6 for g in graphs)


def test_cli_training_search_and_explanation(tmp_path):
    import json
    from graphsae_repro.cli import main

    train_path, val_path = pack("train").save(tmp_path / "train.pt"), pack(
        "val", 100
    ).save(tmp_path / "val.pt")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"epochs": 1, "k_active": 3, "lambda_faith": 0.0}), encoding="utf-8"
    )
    main(
        [
            "train-sae",
            "--train",
            str(train_path),
            "--config",
            str(config),
            "--output",
            str(tmp_path / "train"),
        ]
    )
    assert (tmp_path / "train/sae.pt").exists()
    main(
        [
            "tune-explainer",
            "--factory",
            "examples.toy:load",
            "--space",
            "configs/smoke_search.json",
            "--policies",
            "configs/policies_fixed.json",
            "--output",
            str(tmp_path / "tuned"),
        ]
    )
    assert all((tmp_path / f"tuned/sae_seed_{s}.pt").exists() for s in range(3))
    main(
        [
            "explain",
            "--factory",
            "examples.toy:load",
            "--sae",
            str(tmp_path / "tuned/sae_seed_0.pt"),
            "--policy",
            str(tmp_path / "tuned/selected_policy.json"),
            "--output",
            str(tmp_path / "explain"),
        ]
    )
    assert (tmp_path / "explain/explanations.json").exists()
    main(
        [
            "tune-sae",
            "--train",
            str(train_path),
            "--validation",
            str(val_path),
            "--space",
            "configs/smoke_search.json",
            "--head",
            str(_save_head(tmp_path)),
            "--output",
            str(tmp_path / "search"),
        ]
    )
    main(
        [
            "ablate",
            "--train",
            str(train_path),
            "--evaluation",
            str(val_path),
            "--config",
            str(config),
            "--study",
            "losses",
            "--seeds",
            "0",
            "--output",
            str(tmp_path / "ablations"),
        ]
    )
    assert (tmp_path / "ablations/metrics.json").exists()
    for study in ("pure-path", "signed"):
        main(
            [
                "ablate-explanations",
                "--factory",
                "examples.toy:load",
                "--sae",
                str(tmp_path / "tuned/sae_seed_0.pt"),
                "--study",
                study,
                "--output",
                str(tmp_path / study),
            ]
        )
        assert (tmp_path / study / "explanations.json").exists()


def _save_head(path):
    target = path / "head.pt"
    torch.save(torch.nn.Linear(4, 2).state_dict(), target)
    return target


def test_heterogeneous_graphsae_example():
    from examples.heterogeneous import run

    result, original, deleted = run(epochs=1)
    assert ("person", 4) not in result["selected_nodes"]
    assert result["selected_nodes"] == [("item", 0)]
    assert torch.isfinite(original).all() and torch.isfinite(deleted).all()


def test_bipartite_gat_attention_context_preserves_forward():
    from graphsae_repro.adapters import frozen_attention
    from torch_geometric.nn import GATConv

    model = GATConv((3, 2), 4, heads=2, add_self_loops=False).eval()
    x = (torch.rand(4, 3), torch.rand(2, 2))
    edges = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    forward = lambda: model(x, edges)
    original = forward()
    with frozen_attention(model, forward):
        fixed = forward()
    assert torch.allclose(original, fixed, atol=1e-6)


@pytest.mark.parametrize("name", ["GCN", "GAT", "GIN", "GraphSAGE"])
def test_bias_free_source_vectors_add_to_the_target_state(name):
    torch.manual_seed(51)
    model = build_backbone(
        name, 3, 2, hidden_channels=4, num_layers=2, gat_heads=2, dropout=0.0
    ).eval()
    with torch.no_grad():
        for key, parameter in model.named_parameters():
            if "bias" in key:
                parameter.zero_()
    data = graph()
    _, capture = model(data.x, data.edge_index, capture=True)
    _, terms = node_source_contributions(
        model, data, target_node=1, exclude_target=False
    )
    assert torch.allclose(terms.sum(0), capture.node[1], atol=2e-6)
