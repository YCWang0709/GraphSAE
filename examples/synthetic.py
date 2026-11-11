"""Factories for the six synthetic tasks, using generated data only."""

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph
from graphsae_repro.synthetic import SyntheticDataset, generate_synthetic
from graphsae_repro.backbones import (
    build_backbone,
    train_full_batch_node,
    train_graph_classifier,
)
from graphsae_repro.activations import (
    extract_node_activations,
    extract_graph_activations,
)
from graphsae_repro.experiment import Experiment
from graphsae_repro.tuning import ValidationTarget


def load(
    *,
    dataset="BAPathRole",
    model_name="GCN",
    epochs=100,
    hidden_channels=64,
    num_layers=2,
    num_graphs=900,
    targets_per_split=25,
    device="cpu",
    dataset_path=None,
    generator_options=None,
):
    torch.manual_seed(0)
    options = dict(generator_options or {})
    if dataset in {"BAHouse", "BAPath", "BACycle"}:
        options.setdefault("num_graphs", num_graphs)
    generated = (
        SyntheticDataset.load(dataset_path)
        if dataset_path
        else generate_synthetic(dataset, **options)
    )
    if generated.name != dataset:
        raise ValueError("dataset_path does not match the requested dataset")
    if generated.task == "node":
        data = generated.data
        model = build_backbone(
            model_name,
            data.num_features,
            3,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=0.3,
        ).to(device)
        train_full_batch_node(model, data, epochs=epochs, device=device)
        pack = extract_node_activations(model, data, split="train", device=device)

        def targets(split):
            result = []
            for i in torch.where(getattr(data, f"{split}_mask"))[0][
                :targets_per_split
            ].tolist():
                ids, _, _, _ = k_hop_subgraph(
                    i, model.num_layers, data.edge_index, num_nodes=data.num_nodes
                )
                truth = generated.reference(i, ids.tolist(), split=split)
                result.append(ValidationTarget(data, i, truth, split=split))
            return result

    else:
        graphs, splits = generated.graphs, generated.splits
        model = build_backbone(
            model_name,
            graphs[0].num_features,
            3,
            task="graph",
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=0.3,
        ).to(device)
        train = DataLoader(
            [graphs[i] for i in splits["train"]], batch_size=64, shuffle=True
        )
        val = DataLoader([graphs[i] for i in splits["val"]], batch_size=64)
        train_graph_classifier(
            model, train, validation_loader=val, epochs=epochs, device=device
        )
        pack = extract_graph_activations(
            model,
            DataLoader([graphs[i] for i in splits["train"]], batch_size=64),
            split="train",
            device=device,
        )

        def targets(split):
            return [
                ValidationTarget(
                    graphs[i],
                    0,
                    torch.where(graphs[i].gt_explanation_node_mask)[0].tolist(),
                    split=split,
                )
                for i in splits[split][:targets_per_split]
            ]

    return Experiment(model, pack, targets("val"), targets("test"))
