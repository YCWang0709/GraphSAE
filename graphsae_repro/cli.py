"""Command-line training, validation selection, explanation, and ablations."""

from __future__ import annotations
import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import torch

from .activations import ActivationPack
from .ablations import (
    loss_configurations,
    sensitivity_configurations,
    encoder_configurations,
    train_ablations,
    pure_path_scores,
)
from .evaluation import representation_metrics, deletion_metrics
from .explainer import explain
from .policies import ExtractionPolicy, select_nodes
from .sae import SAEArtifact, SAEConfig, train_sae
from .synthetic import GRAPH_DATASETS, SYNTHETIC_DATASETS, generate_synthetic
from .tuning import (
    parameter_grid,
    tune_sae,
    tune_l1,
    tune_pipeline,
    refit_seeds,
    subset_pack,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def load_head(path):
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    weight = payload["weight"]
    head = torch.nn.Linear(weight.shape[1], weight.shape[0], bias="bias" in payload)
    head.load_state_dict(payload)
    return head.eval()


def load_factory(args):
    module_name, name = args.factory.split(":", 1)
    factory = getattr(importlib.import_module(module_name), name)
    options = read_json(args.factory_options) if args.factory_options else {}
    return factory(device=args.device, **options)


def candidates(path):
    space = read_json(path)
    return parameter_grid(SAEConfig(**space["base"]), space["grid"])


def main(argv=None):
    parser = argparse.ArgumentParser(prog="graphsae")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser(
        "generate-data", help="Construct synthetic datasets without training"
    )
    generate.add_argument(
        "--dataset", required=True, choices=[*SYNTHETIC_DATASETS, "all"]
    )
    generate.add_argument(
        "--config", help="Generator options, or a JSON object with a datasets mapping"
    )
    generate.add_argument(
        "--seed", type=int, help="Override the construction/split seed"
    )
    generate.add_argument(
        "--num-graphs", type=int, help="Override graph-task collection size"
    )
    generate.add_argument("--output", default="data/synthetic")
    for name in ("train-sae", "tune-sae", "ablate"):
        sub = commands.add_parser(name)
        sub.add_argument("--train", required=True, help="Training ActivationPack (.pt)")
        sub.add_argument("--head", help="Frozen Linear head state_dict (.pt)")
        sub.add_argument(
            "--config", help="SAEConfig JSON; default is reconstruction-only"
        )
        sub.add_argument("--output", required=True)
        sub.add_argument("--device", default="cpu")
        if name == "tune-sae":
            sub.add_argument("--validation", required=True)
        if name == "ablate":
            sub.add_argument(
                "--evaluation",
                required=True,
                help="Measured val/test pack; never used for selection",
            )
        if name == "tune-sae":
            sub.add_argument("--space")
            sub.add_argument(
                "--criterion", default="nmse", choices=["nmse", "mse", "fidelity"]
            )
            sub.add_argument("--l1", action="store_true")
        if name == "ablate":
            sub.add_argument(
                "--study",
                choices=["losses", "sensitivity", "encoders"],
                default="losses",
            )
            sub.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
            sub.add_argument(
                "--sensitivity-scope",
                choices=["individual", "joint"],
                default="individual",
            )
    for name in ("tune-explainer", "explain", "ablate-explanations"):
        sub = commands.add_parser(name)
        sub.add_argument(
            "--factory",
            required=True,
            help="Importable module:function returning an Experiment",
        )
        sub.add_argument("--factory-options")
        sub.add_argument("--output", required=True)
        sub.add_argument("--device", default="cpu")
        if name == "tune-explainer":
            sub.add_argument("--space", required=True)
            sub.add_argument("--policies", required=True)
            sub.add_argument("--native", action="store_true")
            sub.add_argument("--recovery-weight", type=float, default=1.0)
            sub.add_argument("--size-weight", type=float, default=0.0)
        else:
            sub.add_argument("--sae", required=True)
            sub.add_argument("--split", choices=["val", "test"], default="test")
            if name == "explain":
                sub.add_argument("--policy", required=True)
            else:
                sub.add_argument(
                    "--study", choices=["pure-path", "signed"], required=True
                )
                sub.add_argument("--budgets", nargs="+", type=int, default=[8, 12, 16])
    args = parser.parse_args(argv)
    output = Path(args.output)
    if args.command == "generate-data":
        names = SYNTHETIC_DATASETS if args.dataset == "all" else (args.dataset,)
        specification = read_json(args.config) if args.config else {}
        if not isinstance(specification, dict):
            parser.error("Generator configuration must be a JSON object")
        if "datasets" in specification:
            specification = specification["datasets"]
            if not isinstance(specification, dict) or set(specification) - set(
                SYNTHETIC_DATASETS
            ):
                parser.error(
                    "datasets must map supported dataset names to generator options"
                )
            per_dataset = True
        else:
            per_dataset = False
            if args.dataset == "all" and specification:
                parser.error("Use a datasets mapping to configure all six datasets")
        if args.num_graphs is not None and not any(
            name in GRAPH_DATASETS for name in names
        ):
            parser.error("--num-graphs applies to graph-classification datasets")
        for name in names:
            options = specification.get(name, {}) if per_dataset else specification
            if not isinstance(options, dict):
                parser.error(f"Generator options for {name} must be a JSON object")
            options = dict(options)
            if args.seed is not None:
                options["seed"] = args.seed
            if args.num_graphs is not None and name in GRAPH_DATASETS:
                options["num_graphs"] = args.num_graphs
            try:
                dataset = generate_synthetic(name, **options)
            except ValueError as error:
                parser.error(str(error))
            print(dataset.save(output / name).resolve())
        return
    output.mkdir(parents=True, exist_ok=True)
    if args.command in {"train-sae", "tune-sae", "ablate"}:
        train = ActivationPack.load(args.train)
        head = load_head(args.head)
        config = (
            SAEConfig(**read_json(args.config))
            if args.config
            else (
                SAEConfig()
                if args.command == "ablate"
                else SAEConfig(lambda_ground=0.0, lambda_faith=0.0, lambda_coh=0.0)
            )
        )
        if args.command == "train-sae":
            artifact = train_sae(
                train, config, prediction_head=head, device=args.device
            )
            artifact.save(output / "sae.pt")
        else:
            validation = ActivationPack.load(
                args.validation if args.command == "tune-sae" else args.evaluation
            )
            if args.command == "tune-sae":
                if args.l1:
                    artifact, protocol = tune_l1(
                        train, validation, prediction_head=head, device=args.device
                    )
                else:
                    if not args.space:
                        parser.error("tune-sae requires --space or --l1")
                    artifact, protocol = tune_sae(
                        train,
                        validation,
                        candidates(args.space),
                        prediction_head=head,
                        criterion=args.criterion,
                        device=args.device,
                    )
                artifact.save(output / "sae.pt")
                write_json(output / "selection.json", protocol)
                write_json(output / "selected_config.json", asdict(artifact.config))
            else:
                from .tuning import check_disjoint

                split = validation.info.get("split") or validation.metadata[0].get(
                    "split"
                )
                if split not in {"val", "test"}:
                    parser.error("ablate --evaluation must declare val or test")
                check_disjoint(train, validation, expected_split=split)
                configs = {
                    "losses": loss_configurations,
                    "sensitivity": sensitivity_configurations,
                    "encoders": encoder_configurations,
                }[args.study](config)
                if args.study == "sensitivity":
                    joint = args.sensitivity_scope == "joint"
                    if split != ("val" if joint else "test"):
                        parser.error(
                            "Joint sensitivity uses val; individual sensitivity uses test"
                        )
                    configs = {
                        name: c
                        for name, c in configs.items()
                        if name.startswith("joint_") == joint
                    }
                rows = []
                validation = subset_pack(validation)
                for name, seed, artifact in train_ablations(
                    train,
                    configs,
                    prediction_head=head,
                    seeds=args.seeds,
                    device=args.device,
                ):
                    artifact.save(output / name / f"seed_{seed}.pt")
                    rows.append(
                        {
                            "variant": name,
                            "seed": seed,
                            "metrics": representation_metrics(
                                artifact,
                                validation,
                                prediction_head=head,
                                train_pack=subset_pack(train),
                                device=args.device,
                            ),
                        }
                    )
                write_json(output / "metrics.json", rows)
    else:
        experiment = load_factory(args)
        if args.command == "tune-explainer":
            policies = [ExtractionPolicy(**p) for p in read_json(args.policies)]
            config, policy, protocol = tune_pipeline(
                experiment.model,
                experiment.train,
                experiment.validation_targets,
                candidates(args.space),
                policies,
                budgets=None if args.native else (8, 12, 16),
                task_type=experiment.task_type,
                recovery_weight=args.recovery_weight,
                size_weight=args.size_weight,
                device=args.device,
            )
            write_json(output / "selection.json", protocol)
            write_json(output / "selected_config.json", asdict(config))
            write_json(output / "selected_policy.json", asdict(policy))
            for seed, artifact in refit_seeds(
                experiment.train,
                config,
                prediction_head=experiment.model.classifier,
                device=args.device,
            ):
                artifact.save(output / f"sae_seed_{seed}.pt")
        else:
            artifact = SAEArtifact.load(args.sae, device=args.device)
            targets = (
                experiment.validation_targets
                if args.split == "val"
                else experiment.test_targets
            )
            rows = []
            for target in targets:
                if args.command == "explain":
                    policy = ExtractionPolicy(**read_json(args.policy))
                    result = explain(
                        experiment.model,
                        target.data,
                        artifact,
                        target_index=target.target_index,
                        policy=policy,
                        task_type=experiment.task_type,
                        device=args.device,
                    )
                    metrics = deletion_metrics(
                        experiment.model,
                        target.data,
                        result["selected_nodes"],
                        target_index=target.target_index,
                        task_type=experiment.task_type,
                        truth=target.truth,
                        device=args.device,
                        stable_sage=policy.stable_sage,
                    )
                    rows.append({"explanation": result, "metrics": metrics})
                elif args.study == "pure-path":
                    ids, scores = pure_path_scores(
                        experiment.model,
                        target.data,
                        artifact,
                        target_index=target.target_index,
                        device=args.device,
                    )
                    selected = select_nodes(
                        ids, scores, ExtractionPolicy(mode="support")
                    )
                    rows.append(
                        {
                            "target": target.target_index,
                            "variant": "pure_path",
                            "selected_nodes": selected,
                            "metrics": deletion_metrics(
                                experiment.model,
                                target.data,
                                selected,
                                target_index=target.target_index,
                                task_type=experiment.task_type,
                                truth=target.truth,
                                device=args.device,
                            ),
                        }
                    )
                else:
                    for budget in args.budgets:
                        result = explain(
                            experiment.model,
                            target.data,
                            artifact,
                            target_index=target.target_index,
                            top_k=budget,
                            task_type=experiment.task_type,
                            device=args.device,
                        )
                        for key in ("sae_positive", "sae_negative", "sae_signed"):
                            rows.append(
                                {
                                    "target": target.target_index,
                                    "variant": key,
                                    "budget": budget,
                                    "selected_nodes": result[key],
                                    "metrics": deletion_metrics(
                                        experiment.model,
                                        target.data,
                                        result[key],
                                        target_index=target.target_index,
                                        task_type=experiment.task_type,
                                        truth=target.truth,
                                        device=args.device,
                                    ),
                                }
                            )
            write_json(output / "explanations.json", rows)
    print(output.resolve())


if __name__ == "__main__":
    main()
