"""Construction, reference annotations, and storage of the six synthetic tasks."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import torch
from torch_geometric.data import Data

from . import synthetic_node as node
from . import synthetic_graph as graph


NODE_DATASETS = tuple(node.DATASET_CONFIGS)
GRAPH_DATASETS = tuple(graph.DATASET_CONFIGS)
SYNTHETIC_DATASETS = NODE_DATASETS + GRAPH_DATASETS


@dataclass
class SyntheticDataset:
    name: str
    task: str
    config: dict
    splits: dict[str, list[int]]
    data: Data | None = None
    graphs: list[Data] | None = None

    def reference(self, target: int, candidates=None, *, split="test") -> list[int]:
        """Return a target's reference nodes in the original node indexing.

        For graph tasks, target is the graph's collection index. BATriadic uses
        role agreement within the supplied field, not a planted motif mask.
        """
        if self.task == "graph":
            data = self.graphs[target]
            nodes = torch.where(data.gt_explanation_node_mask)[0].tolist()
            return nodes if candidates is None else sorted(set(nodes) & set(candidates))
        from .datasets import node_reference

        if candidates is None:
            candidates = range(self.data.num_nodes)
        return node_reference(self.name, self.data, target, candidates, split=split)

    def summary(self) -> dict:
        population = [self.data] if self.task == "node" else self.graphs
        return {
            "name": self.name,
            "task": self.task,
            "config": self.config,
            "split_unit": "node" if self.task == "node" else "graph",
            "split_sizes": {key: len(ids) for key, ids in self.splits.items()},
            "num_graphs": len(population),
            "num_nodes": sum(data.num_nodes for data in population),
            "num_directed_edges": sum(data.num_edges for data in population),
            "num_features": population[0].num_features,
            "reference": (
                "role_agreement" if self.name == "BATriadic" else "planted_motifs"
            ),
        }

    def save(self, directory: str | Path) -> Path:
        """Write a PyG bundle compatible with examples.from_checkpoint."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "dataset.pt"
        payload = {
            "schema_version": 1,
            "name": self.name,
            "task": self.task,
            "config": self.config,
            "splits": self.splits,
        }
        payload["data" if self.task == "node" else "graphs"] = (
            self.data if self.task == "node" else self.graphs
        )
        torch.save(payload, path)
        (directory / "metadata.json").write_text(
            json.dumps(self.summary(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SyntheticDataset":
        path = Path(path)
        if path.is_dir():
            path = path / "dataset.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported synthetic dataset schema")
        return cls(
            name=payload["name"],
            task=payload["task"],
            config=payload["config"],
            splits=payload["splits"],
            data=payload.get("data"),
            graphs=payload.get("graphs"),
        )


def _integer(options, key, minimum):
    value = options[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")


def _motif_annotations(data: Data, backbone_nodes: int, num_motifs: int):
    """Store planted membership and edge masks separately from input features."""
    membership = torch.full((data.num_nodes,), -1, dtype=torch.long)
    membership[backbone_nodes:] = torch.arange(num_motifs).repeat_interleave(5)
    source, target = data.edge_index
    internal = (membership[source] >= 0) & (membership[source] == membership[target])
    attachment = (membership[source] >= 0) ^ (membership[target] >= 0)
    data.motif_node_id = membership
    data.num_backbone_nodes = torch.tensor([backbone_nodes])
    data.num_inserted_motifs = torch.tensor([num_motifs])
    data.gt_explanation_node_mask = membership >= 0
    data.gt_explanation_edge_mask = internal
    data.gt_explanation_edge_mask_with_attachment = internal | attachment


def generate_synthetic(name: str, **overrides) -> SyntheticDataset:
    """Construct a task with supplementary defaults and explicit size/seed overrides.

    BATriadic: seed, num_nodes, ba_m, triangle_probability.
    Role tasks: seed, ba_m, backbone_nodes, num_motifs.
    Graph tasks: seed, num_graphs, ba_m, ba_min_nodes, ba_max_nodes.
    """
    if name not in SYNTHETIC_DATASETS:
        raise ValueError(
            f"Unknown synthetic dataset {name!r}; choose from {SYNTHETIC_DATASETS}"
        )
    if name in NODE_DATASETS:
        default = node.DATASET_CONFIGS[name]
        options = {"seed": default.seed, "ba_m": default.ba_m}
        if name == "BATriadic":
            options.update(num_nodes=default.num_nodes, triangle_probability=0.35)
        else:
            options.update(backbone_nodes=750, num_motifs=250)
    else:
        default = graph.DATASET_CONFIGS[name]
        options = {
            key: getattr(default, key)
            for key in ("seed", "num_graphs", "ba_m", "ba_min_nodes", "ba_max_nodes")
        }
    unknown = set(overrides) - options.keys()
    if unknown:
        raise ValueError(f"Unsupported options for {name}: {sorted(unknown)}")
    options.update(overrides)
    _integer(options, "seed", 0)
    _integer(options, "ba_m", 1)
    if name == "BATriadic":
        _integer(options, "num_nodes", 9)
        if options["ba_m"] >= options["num_nodes"]:
            raise ValueError("ba_m must be smaller than num_nodes")
        probability = options["triangle_probability"]
        if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
            raise ValueError("triangle_probability must be in [0, 1]")
        config = replace(
            default,
            seed=options["seed"],
            ba_m=options["ba_m"],
            num_nodes=options["num_nodes"],
        )
        data = node.make_ba_triadic(config, triangle_probability=probability)
    elif name in NODE_DATASETS:
        _integer(options, "backbone_nodes", 3)
        _integer(options, "num_motifs", 2)
        if options["ba_m"] >= options["backbone_nodes"]:
            raise ValueError("ba_m must be smaller than backbone_nodes")
        config = replace(
            default,
            seed=options["seed"],
            ba_m=options["ba_m"],
            num_nodes=options["backbone_nodes"] + 5 * options["num_motifs"],
        )
        constructor = (
            node.make_ba_path_role if name == "BAPathRole" else node.make_ba_bifan_role
        )
        data = constructor(
            config,
            backbone_nodes=options["backbone_nodes"],
            num_motifs=options["num_motifs"],
        )
        _motif_annotations(data, options["backbone_nodes"], options["num_motifs"])
    else:
        _integer(options, "num_graphs", 9)
        _integer(options, "ba_min_nodes", 2)
        _integer(options, "ba_max_nodes", 2)
        if options["num_graphs"] % 3:
            raise ValueError(
                "num_graphs must be divisible by three for balanced count labels"
            )
        if not options["ba_m"] < options["ba_min_nodes"] <= options["ba_max_nodes"]:
            raise ValueError("Require ba_m < ba_min_nodes <= ba_max_nodes")
        config = replace(default, **options)
        constructor = {
            "BAHouse": graph.make_ba_house_dataset,
            "BAPath": graph.make_ba_path_dataset,
            "BACycle": graph.make_ba_cycle_dataset,
        }[name]
        graphs = constructor(config)
        splits = graph.stratified_split_indices(
            [int(data.y.item()) for data in graphs], seed=config.seed
        )
        return SyntheticDataset(name, "graph", options, splits, graphs=graphs)
    splits = {
        split: torch.where(getattr(data, f"{split}_mask"))[0].tolist()
        for split in ("train", "val", "test")
    }
    return SyntheticDataset(name, "node", options, splits, data=data)
