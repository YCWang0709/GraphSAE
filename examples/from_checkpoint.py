"""Factory for an existing PyG node graph or a split graph collection."""

import torch
from torch_geometric.loader import DataLoader
from graphsae_repro.activations import (
    extract_node_activations,
    extract_graph_activations,
)
from graphsae_repro.backbones import build_backbone
from graphsae_repro.experiment import Experiment
from graphsae_repro.tuning import ValidationTarget


def load(
    *,
    checkpoint,
    data_path,
    device="cpu",
    task_type="classification",
    targets_per_split=25,
):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_backbone(**payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    model = model.to(device).eval()
    bundle = torch.load(data_path, map_location="cpu", weights_only=False)
    if model.task == "node":
        data = bundle["data"]
        pack = extract_node_activations(
            model, data, split="train", task_type=task_type, device=device
        )

        def targets(split):
            return [
                ValidationTarget(data, int(i), split=split)
                for i in torch.where(getattr(data, f"{split}_mask"))[0][
                    :targets_per_split
                ]
            ]

    else:
        graphs, splits = bundle["graphs"], bundle["splits"]
        for i, graph in enumerate(graphs):
            graph.graph_id = torch.tensor([i])
        pack = extract_graph_activations(
            model,
            DataLoader([graphs[i] for i in splits["train"]], batch_size=64),
            split="train",
            task_type=task_type,
            device=device,
        )

        def targets(split):
            result = []
            for i in splits[split][:targets_per_split]:
                graph = graphs[i]
                mask = getattr(graph, "gt_explanation_node_mask", None)
                truth = torch.where(mask)[0].tolist() if mask is not None else None
                result.append(ValidationTarget(graph, 0, truth=truth, split=split))
            return result

    return Experiment(model, pack, targets("val"), targets("test"), task_type=task_type)
