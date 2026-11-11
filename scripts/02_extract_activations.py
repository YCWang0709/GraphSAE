from __future__ import annotations

import argparse

from _cora import (
    activation_path,
    artifact_dir,
    checkpoint_path,
    load_backbone,
    load_cora,
    resolve_device,
)
from graphsae_repro.activations import extract_node_activations
from graphsae_repro.backbones import MODEL_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract final-layer Cora activations for SAE training or evaluation."
    )
    parser.add_argument("--model", default="GCN", choices=MODEL_NAMES)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    out_dir = artifact_dir(args.output_dir)
    _, data = load_cora(args.data_root)
    model, _ = load_backbone(checkpoint_path(out_dir, args.model), device=device)
    pack = extract_node_activations(model, data, split=args.split, device=device)
    path = pack.save(activation_path(out_dir, args.model, args.split))
    print(path)


if __name__ == "__main__":
    main()
