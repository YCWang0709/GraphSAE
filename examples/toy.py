"""Small deterministic node/graph tasks for exercising the complete workflow."""

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from graphsae_repro.activations import (
    extract_node_activations,
    extract_graph_activations,
)
from graphsae_repro.backbones import (
    build_backbone,
    train_full_batch_node,
    train_graph_classifier,
)
from graphsae_repro.experiment import Experiment
from graphsae_repro.tuning import ValidationTarget


def load(*, device="cpu", model_name="GCN", task="node", epochs=2):
    torch.manual_seed(42)
    edges = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]]
    )
    model = build_backbone(
        model_name,
        3,
        2,
        task=task,
        hidden_channels=8,
        num_layers=2,
        gat_heads=2,
        dropout=0.0,
    )
    if task == "node":
        data = Data(
            x=torch.rand(6, 3),
            edge_index=edges,
            y=torch.tensor([0, 1, 0, 1, 0, 1]),
            train_mask=torch.tensor([True, True, True, False, False, False]),
            val_mask=torch.tensor([False, False, False, True, False, False]),
            test_mask=torch.tensor([False, False, False, False, True, True]),
        )
        train_full_batch_node(model, data, epochs=epochs, device=device)
        pack = extract_node_activations(model, data, split="train", device=device)
        return Experiment(
            model,
            pack,
            [ValidationTarget(data, 3)],
            [ValidationTarget(data, i, split="test") for i in (4, 5)],
        )
    graphs = [
        Data(
            x=torch.rand(6, 3),
            edge_index=edges,
            y=torch.tensor([i % 2]),
            graph_id=torch.tensor([i]),
        )
        for i in range(8)
    ]
    train = DataLoader(graphs[:4], batch_size=4)
    val = DataLoader(graphs[4:6], batch_size=2)
    train_graph_classifier(
        model, train, validation_loader=val, epochs=epochs, device=device
    )
    pack = extract_graph_activations(model, train, split="train", device=device)
    return Experiment(
        model,
        pack,
        [ValidationTarget(g, 0) for g in graphs[4:6]],
        [ValidationTarget(g, 0, split="test") for g in graphs[6:]],
    )
