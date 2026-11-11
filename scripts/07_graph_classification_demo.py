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
from graphsae_repro.synthetic_graph import stratified_split_indices
from graphsae_repro.datasets import mutag_proxy_mask


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GraphSAE training and explanation on a TUDataset."
    )
    parser.add_argument("--dataset", default="MUTAG")
    parser.add_argument("--model", default="GCN", choices=MODEL_NAMES)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--sae-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    base_dir = (
        Path(args.output_dir) if args.output_dir else ROOT / "artifacts" / "graphs"
    )
    out_dir = base_dir / f"{args.dataset.lower()}_{args.model.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = TUDataset(
        root=str(args.data_root or (out_dir / "data")), name=args.dataset
    )
    if dataset.num_features <= 0:
        raise ValueError("A TUDataset with node features is required, such as MUTAG.")
    graphs = [dataset[i] for i in range(len(dataset))]
    for i, graph in enumerate(graphs):
        graph.graph_id = torch.tensor([i])
        if args.dataset == "MUTAG":
            graph.gt_explanation_node_mask = mutag_proxy_mask(
                graph.x.argmax(-1), graph.edge_index
            )
    splits = stratified_split_indices([int(g.y) for g in graphs], seed=args.seed)
    train_data = [graphs[i] for i in splits["train"]]
    val_data = [graphs[i] for i in splits["val"]]
    test_data = [graphs[i] for i in splits["test"]]
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    model = build_backbone(
        args.model, dataset.num_features, dataset.num_classes, task="graph"
    )
    history = train_graph_classifier(
        model,
        train_loader,
        validation_loader=DataLoader(val_data, batch_size=64),
        epochs=args.epochs,
        device=device,
    )
    model_config = dict(
        model_name=args.model,
        in_channels=dataset.num_features,
        out_channels=dataset.num_classes,
        task="graph",
        hidden_channels=model.hidden_channels,
        num_layers=model.num_layers,
        dropout=model.dropout,
        gat_heads=model.gat_heads,
    )
    torch.save(
        {"model_config": model_config, "state_dict": model.cpu().state_dict()},
        out_dir / "backbone.pt",
    )
    torch.save({"graphs": graphs, "splits": splits}, out_dir / "dataset.pt")
    torch.save(model.classifier.state_dict(), out_dir / "head.pt")
    train_pack = extract_graph_activations(
        model, DataLoader(train_data, batch_size=64), split="train", device=device
    )
    train_pack.save(out_dir / "train_activations.pt")
    artifact = train_sae(
        train_pack,
        SAEConfig(epochs=args.sae_epochs, seed=args.seed),
        prediction_head=model.classifier,
        device=device,
    )
    artifact.save(out_dir / "sae.pt")
    for split, population in (("val", val_data), ("test", test_data)):
        extract_graph_activations(
            model, DataLoader(population, batch_size=64), split=split, device=device
        ).save(out_dir / f"{split}_activations.pt")
    test_pack = extract_graph_activations(
        model, DataLoader(test_data, batch_size=64), split="test", device=device
    )
    feature_alignment_table(test_pack, artifact).to_csv(
        out_dir / "feature_alignment.csv", index=False
    )
    example = test_data[0]
    explanation = explain_graph(model, example, artifact, device=device)
    ablation = run_graph_feature_ablation(model, example, artifact, device=device)
    masking = run_mask_experiment(
        model,
        example,
        explanation["sae_signed"],
        task="graph",
        mode="both",
        device=device,
    )
    save_json(
        out_dir / "summary.json",
        {
            "training_history": history,
            "explanation": explanation,
            "feature_ablation": ablation,
            "masking": masking,
        },
    )
    print(out_dir)


if __name__ == "__main__":
    main()
