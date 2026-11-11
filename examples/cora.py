"""Load a selected Cora predictor without retraining during explanation search."""

from pathlib import Path
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from graphsae_repro.activations import extract_node_activations
from graphsae_repro.backbones import build_backbone
from graphsae_repro.experiment import Experiment
from graphsae_repro.tuning import ValidationTarget


def load(*, checkpoint, device="cpu", data_root="data", targets_per_split=25):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_backbone(
        payload["model_name"],
        payload["in_channels"],
        payload["out_channels"],
        hidden_channels=payload["hidden_channels"],
        num_layers=payload["num_layers"],
        dropout=payload["dropout"],
        gat_heads=payload["gat_heads"],
    )
    model.load_state_dict(payload["state_dict"])
    model = model.to(device).eval()
    data = Planetoid(
        root=str(Path(data_root)), name="Cora", transform=NormalizeFeatures()
    )[0]
    pack = extract_node_activations(model, data, split="train", device=device)

    def targets(split):
        ids = torch.where(getattr(data, f"{split}_mask"))[0][
            :targets_per_split
        ].tolist()
        return [ValidationTarget(data, i, split=split) for i in ids]

    return Experiment(model, pack, targets("val"), targets("test"))
