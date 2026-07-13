"""Shared Cora data, checkpoint, and artifact helpers for the reproduction scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphsae_repro.backbones import build_backbone, canonical_model_name


def resolve_device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value))


def artifact_dir(value: str | None) -> Path:
    path = Path(value) if value else ROOT / "artifacts" / "cora"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_cora(data_root: str | None = None):
    root = Path(data_root) if data_root else ROOT / "data"
    dataset = Planetoid(root=str(root), name="Cora", transform=NormalizeFeatures())
    return dataset, dataset[0]


def checkpoint_path(out_dir: Path, model_name: str) -> Path:
    return out_dir / f"backbone_{canonical_model_name(model_name).lower()}.pt"


def sae_path(out_dir: Path, model_name: str) -> Path:
    return out_dir / f"sae_{canonical_model_name(model_name).lower()}.pt"


def activation_path(out_dir: Path, model_name: str, split: str) -> Path:
    return out_dir / f"activations_{canonical_model_name(model_name).lower()}_{split}.pt"


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def load_backbone(path: Path, *, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_backbone(
        checkpoint["model_name"],
        int(checkpoint["in_channels"]),
        int(checkpoint["out_channels"]),
        task="node",
        hidden_channels=int(checkpoint["hidden_channels"]),
        dropout=float(checkpoint["dropout"]),
        gat_heads=int(checkpoint["gat_heads"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def choose_correct_test_node(model, data, *, device: torch.device) -> int:
    model = model.to(device).eval()
    data = data.to(device)
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    candidates = torch.nonzero(data.test_mask & (logits.argmax(dim=-1) == data.y), as_tuple=False).view(-1)
    if candidates.numel() == 0:
        candidates = torch.nonzero(data.test_mask, as_tuple=False).view(-1)
    return int(candidates[0].item())
