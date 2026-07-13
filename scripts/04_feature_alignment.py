from __future__ import annotations

import argparse

import pandas as pd
import torch

from _cora import activation_path, artifact_dir, checkpoint_path, choose_correct_test_node, load_backbone, load_cora, resolve_device, sae_path, save_json
from graphsae_repro.activations import ActivationPack
from graphsae_repro.alignment import feature_alignment_table, run_node_feature_ablation
from graphsae_repro.backbones import MODEL_NAMES
from graphsae_repro.sae import SAEArtifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure SAE-label alignment and run target-level SAE feature ablations.")
    parser.add_argument("--model", default="GCN", choices=MODEL_NAMES)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--activation-split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--target-node", type=int, default=None)
    parser.add_argument("--num-features", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    out_dir = artifact_dir(args.output_dir)
    _, data = load_cora(args.data_root)
    model, _ = load_backbone(checkpoint_path(out_dir, args.model), device=device)
    artifact = SAEArtifact.load(sae_path(out_dir, args.model), device=device)
    pack = ActivationPack.load(activation_path(out_dir, args.model, args.activation_split))
    table = feature_alignment_table(pack, artifact)
    table.to_csv(out_dir / f"feature_alignment_{args.model.lower()}.csv", index=False)
    target = choose_correct_test_node(model, data, device=device) if args.target_node is None else int(args.target_node)
    intervention = run_node_feature_ablation(model, data, artifact, target_node=target, num_features=args.num_features, device=device)
    save_json(out_dir / f"feature_ablation_{args.model.lower()}_node_{target}.json", intervention)
    print(pd.DataFrame([intervention]).to_string(index=False))


if __name__ == "__main__":
    main()
