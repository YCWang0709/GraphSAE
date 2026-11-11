"""Render the original one-hop field and its visible selected source subgraph."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import torch


def draw_explanation(data, record, output):
    target = record["target_node"]
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))
    graph.add_edges_from(data.edge_index.T.tolist())
    context = nx.ego_graph(graph, target, radius=1)
    selected = set(record["selected_nodes"])
    visible = selected & set(context)
    explanation = context.subgraph(visible | {target}).copy()
    positions = nx.spring_layout(context, seed=42)
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)
    palette = plt.get_cmap("tab10")
    for axis, panel, title in zip(
        axes,
        (context, explanation),
        ("Original one-hop neighborhood", "Selected sources within one hop"),
    ):
        nx.draw_networkx_edges(
            panel, positions, ax=axis, edge_color="#9ca3af", width=1.1, alpha=0.8
        )
        neighbors = sorted(set(panel) - {target})
        if neighbors:
            nx.draw_networkx_nodes(
                panel,
                positions,
                ax=axis,
                nodelist=neighbors,
                node_color=[
                    palette(int(data.y[u])) if u in visible else "#e5e7eb"
                    for u in neighbors
                ],
                edgecolors=[
                    "#be5d14" if u in visible else "#9ca3af" for u in neighbors
                ],
                linewidths=1.8,
                node_size=460,
            )
        nx.draw_networkx_nodes(
            panel,
            positions,
            ax=axis,
            nodelist=[target],
            node_color="#f3b61f",
            edgecolors="#3f3f46",
            node_shape="*",
            linewidths=1.5,
            node_size=850,
        )
        nx.draw_networkx_labels(
            panel, positions, ax=axis, labels={u: str(u) for u in panel}, font_size=8
        )
        axis.set_title(title)
        axis.set_axis_off()
        axis.margins(0.18)
    figure.suptitle(
        f"{record['model']} · SAE seed {record['seed']} · Target {target}\n"
        f"Full explanation: {len(selected)} sources; visible within one hop: {len(visible)}"
    )
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="*",
                color="none",
                markerfacecolor="#f3b61f",
                markeredgecolor="#3f3f46",
                markersize=13,
                label="Target",
            ),
            Line2D(
                [],
                [],
                marker="o",
                color="none",
                markerfacecolor="#4c78a8",
                markeredgecolor="#be5d14",
                markersize=8,
                label="Selected source (fill = class)",
            ),
            Line2D(
                [],
                [],
                marker="o",
                color="none",
                markerfacecolor="#e5e7eb",
                markeredgecolor="#9ca3af",
                markersize=8,
                label="Other neighbor",
            ),
        ],
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{record['model']}_seed{record['seed']}_node{target}_1hop"
    paths = [output / f"{stem}.png", output / f"{stem}.svg"]
    for path in paths:
        figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return paths


def main(argv=None):
    if __package__:
        from .inference import (
            FrozenSAE,
            SourceProjection,
            checked_load,
            explain_target,
            load_data,
            load_manifest,
            load_predictor,
        )
    else:
        from inference import (
            FrozenSAE,
            SourceProjection,
            checked_load,
            explain_target,
            load_data,
            load_manifest,
            load_predictor,
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["GCN", "GIN", "GAT", "GraphSAGE"], default="GCN"
    )
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--target-node", type=int)
    parser.add_argument("--data-root", default="data/cora_pretrained")
    parser.add_argument("--output", default="artifacts/cora_pretrained/figures")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    here = Path(__file__).resolve().parent
    manifest = load_manifest(here)
    entry = manifest["models"][args.model]
    data = load_data(args.data_root, manifest)
    target = entry["test_targets"][0] if args.target_node is None else args.target_node
    if not 0 <= target < data.num_nodes:
        parser.error("--target-node is outside the graph")
    model = load_predictor(here, entry)
    projection = SourceProjection(model, data, entry["sae_layer"])
    sae = FrozenSAE(checked_load(here, entry["saes"][str(args.seed)]))
    record = explain_target(projection, sae, entry, target)
    record.update(model=args.model, seed=args.seed)
    for path in draw_explanation(data, record, args.output):
        print(path.resolve())


if __name__ == "__main__":
    main()
