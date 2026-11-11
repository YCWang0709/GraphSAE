"""Reproduce the Cora row from bundled frozen predictors and SAE weights."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
import time

import torch

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


HERE = Path(__file__).resolve().parent
METRICS = ("logit_drop", "probability_drop", "num_nodes")
PAPER_ROW = {"logit_drop": "7.08", "probability_drop": "0.51", "num_nodes": "8.07"}


def aggregate(rows):
    """Average seeds within target, targets within backbone, then backbones."""
    grouped = {}
    seen = set()
    for row in rows:
        key = row["model"], row["target_node"], row["seed"]
        if key in seen:
            raise ValueError(f"Duplicate evaluation row: {key}")
        seen.add(key)
        grouped.setdefault(key[:2], []).append(row)
    targets = {
        key: {metric: mean(r[metric] for r in values) for metric in METRICS}
        for key, values in grouped.items()
    }
    backbones = {}
    for (model, _), values in targets.items():
        backbones.setdefault(model, []).append(values)
    per_backbone = {
        model: {metric: mean(row[metric] for row in values) for metric in METRICS}
        for model, values in backbones.items()
    }
    overall = {
        metric: mean(row[metric] for row in per_backbone.values()) for metric in METRICS
    }
    return per_backbone, overall


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/cora_pretrained")
    parser.add_argument("--output", default="artifacts/cora_pretrained")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["GCN", "GIN", "GAT", "GraphSAGE"],
        default=["GCN", "GIN", "GAT", "GraphSAGE"],
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, choices=[0, 1, 2], default=[0, 1, 2]
    )
    parser.add_argument(
        "--target-limit",
        type=int,
        help="Run a development prefix of the archived cohort",
    )
    parser.add_argument(
        "--check-paper",
        action="store_true",
        help="Require the full cohort and compare its computed rounded row",
    )
    parser.add_argument("--skip-visualization", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(
        args.seeds
    ):
        parser.error("Models and seeds must not be repeated")
    if args.target_limit is not None and args.target_limit < 1:
        parser.error("--target-limit must be positive")
    complete = (
        set(args.models) == {"GCN", "GIN", "GAT", "GraphSAGE"}
        and set(args.seeds) == {0, 1, 2}
        and args.target_limit is None
    )
    if args.check_paper and not complete:
        parser.error(
            "--check-paper requires all four backbones, all three seeds, and the complete target cohorts"
        )
    torch.set_num_threads(1)
    manifest = load_manifest(HERE)
    data = load_data(args.data_root, manifest)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, visualizations = [], []
    start = time.perf_counter()
    with (output / "explanations.jsonl").open("w", encoding="utf-8") as explanations:
        for name in args.models:
            entry = manifest["models"][name]
            model = load_predictor(HERE, entry)
            projection = SourceProjection(model, data, entry["sae_layer"])
            targets = entry["test_targets"][: args.target_limit]
            for seed in args.seeds:
                sae = FrozenSAE(checked_load(HERE, entry["saes"][str(seed)]))
                for index, target in enumerate(targets):
                    record = explain_target(projection, sae, entry, target)
                    record.update(model=name, seed=seed)
                    explanations.write(json.dumps(record, allow_nan=False) + "\n")
                    explanations.flush()
                    rows.append(
                        {
                            "model": name,
                            "seed": seed,
                            "target_node": target,
                            **record["metrics"],
                            "search_queries": record["search_queries"],
                        }
                    )
                    if (
                        index == 0
                        and seed == args.seeds[0]
                        and not args.skip_visualization
                    ):
                        if __package__:
                            from .visualize import draw_explanation
                        else:
                            from visualize import draw_explanation
                        visualizations.extend(
                            str(p)
                            for p in draw_explanation(data, record, output / "figures")
                        )
                    if (index + 1) % 10 == 0 or index + 1 == len(targets):
                        print(
                            f"{name} seed {seed}: {index + 1}/{len(targets)} targets",
                            flush=True,
                        )
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    backbones, overall = aggregate(rows)
    rounded = {key: f"{value:.2f}" for key, value in overall.items()}
    summary = {
        "dataset": "Cora",
        "full_paper_cohort": complete,
        "models": args.models,
        "seeds": args.seeds,
        "rows": len(rows),
        "target_predictor_pairs": len({(r["model"], r["target_node"]) for r in rows}),
        "aggregation": manifest["aggregation"],
        "per_backbone": backbones,
        "overall": overall,
        "rounded": rounded,
        "paper_reference": PAPER_ROW,
        "matches_paper_rounding": complete and rounded == PAPER_ROW,
        "seconds": time.perf_counter() - start,
        "visualizations": visualizations,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("\nModel              Logit drop   Probability drop   Mean nodes")
    for model, values in backbones.items():
        print(
            f"{model:<18} {values['logit_drop']:>9.4f} {values['probability_drop']:>18.4f} {values['num_nodes']:>12.4f}"
        )
    label = "SAE (Ours)" if complete else "Partial cohort"
    print(
        f"{label:<18} {rounded['logit_drop']:>9} {rounded['probability_drop']:>18} {rounded['num_nodes']:>12}"
    )
    print(f"Outputs: {output.resolve()}")
    if args.check_paper and rounded != PAPER_ROW:
        raise SystemExit(
            "Computed metrics differ from the published rounded row; see summary.json and per-target outputs."
        )
    return summary


if __name__ == "__main__":
    main()
