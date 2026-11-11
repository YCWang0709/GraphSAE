from __future__ import annotations

import argparse

from _cora import (
    activation_path,
    artifact_dir,
    checkpoint_path,
    load_backbone,
    resolve_device,
    sae_path,
    save_json,
)
from graphsae_repro.activations import ActivationPack
from graphsae_repro.backbones import MODEL_NAMES
from graphsae_repro.sae import SAEConfig, train_sae


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a TopK SAE on extracted final-layer activations."
    )
    parser.add_argument("--model", default="GCN", choices=MODEL_NAMES)
    parser.add_argument("--activation-split", default="train", choices=["train"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--expansion", type=int, default=8)
    parser.add_argument("--k-active", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l1-weight", type=float, default=0.0)
    parser.add_argument("--lambda-ground", type=float, default=0.01)
    parser.add_argument("--lambda-faith", type=float, default=0.1)
    parser.add_argument("--lambda-coh", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    out_dir = artifact_dir(args.output_dir)
    pack = ActivationPack.load(
        activation_path(out_dir, args.model, args.activation_split)
    )
    config = SAEConfig(
        epochs=args.epochs,
        expansion=args.expansion,
        k_active=args.k_active,
        batch_size=args.batch_size,
        lr=args.lr,
        l1_weight=args.l1_weight,
        lambda_ground=args.lambda_ground,
        lambda_faith=args.lambda_faith,
        lambda_coh=args.lambda_coh,
        seed=args.seed,
    )
    model, _ = load_backbone(
        checkpoint_path(out_dir, args.model), device=resolve_device(args.device)
    )
    artifact = train_sae(
        pack,
        config,
        prediction_head=model.classifier,
        device=resolve_device(args.device),
    )
    path = artifact.save(sae_path(out_dir, args.model))
    save_json(
        out_dir / f"sae_{args.model.lower()}_metrics.json",
        {"metrics": artifact.metrics, "history": artifact.history},
    )
    print(path)


if __name__ == "__main__":
    main()
