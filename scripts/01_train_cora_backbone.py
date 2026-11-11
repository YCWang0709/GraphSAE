from __future__ import annotations

import argparse

import torch

from _cora import artifact_dir, checkpoint_path, load_cora, resolve_device, save_json
from graphsae_repro.backbones import (
    MODEL_NAMES,
    build_backbone,
    canonical_model_name,
    train_full_batch_node,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one capture-compatible GNN backbone on Cora."
    )
    parser.add_argument("--model", default="GCN", choices=MODEL_NAMES)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = artifact_dir(args.output_dir)
    dataset, data = load_cora(args.data_root)
    model_name = canonical_model_name(args.model)
    model = build_backbone(
        model_name,
        dataset.num_features,
        dataset.num_classes,
        task="node",
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gat_heads=args.gat_heads,
    )
    history = train_full_batch_node(model, data, epochs=args.epochs, device=device)
    path = checkpoint_path(out_dir, model_name)
    torch.save(
        {
            "model_name": model_name,
            "in_channels": int(dataset.num_features),
            "out_channels": int(dataset.num_classes),
            "hidden_channels": args.hidden_channels,
            "dropout": args.dropout,
            "gat_heads": args.gat_heads,
            "num_layers": args.num_layers,
            "state_dict": model.cpu().state_dict(),
        },
        path,
    )
    save_json(out_dir / f"backbone_{model_name.lower()}_history.json", history)
    torch.save(model.classifier.state_dict(), out_dir / f"head_{model_name.lower()}.pt")
    print(path)


if __name__ == "__main__":
    main()
