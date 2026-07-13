from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from _cora import ROOT, resolve_device, save_json
from graphsae_repro.activations import extract_graph_activations
from graphsae_repro.alignment import feature_alignment_table, run_graph_feature_ablation
from graphsae_repro.backbones import MODEL_NAMES, build_backbone, train_graph_classifier
from graphsae_repro.explainer import explain_graph
from graphsae_repro.masking import run_mask_experiment
from graphsae_repro.sae import SAEConfig, train_sae


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal graph-classification GraphSAE reproduction on a TUDataset.")
    parser.add_argument("--dataset", default="MUTAG")
    parser.add_argument("--model", default="GCN", choices=MODEL_NAMES)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--sae-epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    base_dir = Path(args.output_dir) if args.output_dir else ROOT / "artifacts" / "graphs"
    out_dir = base_dir / f"{args.dataset.lower()}_{args.model.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = TUDataset(root=str(args.data_root or (out_dir / "data")), name=args.dataset)
    if dataset.num_features <= 0:
        raise ValueError("This minimal demo requires a TUDataset with node features, such as MUTAG.")
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_end = max(1, int(0.8 * len(order)))
    train_data, test_data = dataset[order[:train_end]], dataset[order[train_end:]]
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    model = build_backbone(args.model, dataset.num_features, dataset.num_classes, task="graph")
    history = train_graph_classifier(model, train_loader, epochs=args.epochs, device=device)
    train_pack = extract_graph_activations(model, DataLoader(train_data, batch_size=64), device=device)
    train_pack.save(out_dir / "train_activations.pt")
    artifact = train_sae(train_pack, SAEConfig(epochs=args.sae_epochs, seed=args.seed), device=device)
    artifact.save(out_dir / "sae.pt")
    feature_alignment_table(train_pack, artifact).to_csv(out_dir / "feature_alignment.csv", index=False)
    example = test_data[0]
    explanation = explain_graph(model, example, artifact, device=device)
    ablation = run_graph_feature_ablation(model, example, artifact, device=device)
    masking = run_mask_experiment(model, example, explanation["sae_signed"], task="graph", mode="both", device=device)
    save_json(out_dir / "summary.json", {"training_history": history, "explanation": explanation, "feature_ablation": ablation, "masking": masking})
    print(out_dir)


if __name__ == "__main__":
    main()
