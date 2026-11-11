from __future__ import annotations

import argparse

from _cora import (
    artifact_dir,
    checkpoint_path,
    choose_correct_test_node,
    load_backbone,
    load_cora,
    resolve_device,
    sae_path,
    save_json,
)
from graphsae_repro.backbones import MODEL_NAMES
from graphsae_repro.explainer import explain_node
from graphsae_repro.masking import run_mask_experiment
from graphsae_repro.sae import SAEArtifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mask SAE+ / SAE- / SAE+/- explanation nodes on Cora."
    )
    parser.add_argument("--model", default="GCN", choices=MODEL_NAMES)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--target-node", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--num-features", type=int, default=None)
    parser.add_argument(
        "--mask-mode", default="both", choices=["features", "edges", "both"]
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    out_dir = artifact_dir(args.output_dir)
    _, data = load_cora(args.data_root)
    model, _ = load_backbone(checkpoint_path(out_dir, args.model), device=device)
    artifact = SAEArtifact.load(sae_path(out_dir, args.model), device=device)
    target = (
        choose_correct_test_node(model, data, device=device)
        if args.target_node is None
        else int(args.target_node)
    )
    explanation = explain_node(
        model,
        data,
        artifact,
        target_node=target,
        num_features=args.num_features,
        top_k=args.top_k,
        device=device,
    )
    results = {}
    for name, nodes in (
        ("sae_positive", explanation["sae_positive"]),
        ("sae_negative", explanation["sae_negative"]),
        ("sae_signed", explanation["sae_signed"]),
    ):
        results[name] = run_mask_experiment(
            model,
            data,
            nodes,
            task="node",
            target_index=target,
            mode=args.mask_mode,
            device=device,
        )
    path = out_dir / f"mask_experiment_{args.model.lower()}_node_{target}.json"
    save_json(path, {"explanation": explanation, "mask_results": results})
    print(path)


if __name__ == "__main__":
    main()
