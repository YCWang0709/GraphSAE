"""Explicit train/validation/test contract for experiment factories."""

from dataclasses import dataclass
from .activations import ActivationPack
from .tuning import ValidationTarget


@dataclass
class Experiment:
    model: object
    train: ActivationPack
    validation_targets: list[ValidationTarget]
    test_targets: list[ValidationTarget]
    task_type: str = "classification"

    def __post_init__(self):
        from .tuning import require_split

        require_split(self.train, "train")
        cohorts = []
        for split, targets in (
            ("val", self.validation_targets),
            ("test", self.test_targets),
        ):
            ids = set()
            for target in targets:
                if target.split != split:
                    raise ValueError("Explanation cohort split mismatch")
                if self.model.task == "node":
                    if not bool(
                        getattr(target.data, f"{split}_mask")[target.target_index]
                    ):
                        raise ValueError("Target is outside its declared split")
                    ids.add(target.target_index)
                else:
                    graph_id = getattr(target.data, "graph_id", None)
                    if graph_id is None:
                        raise ValueError("Graph targets require stable graph_id values")
                    ids.add(int(graph_id[target.target_index]))
            cohorts.append(ids)
        if cohorts[0] & cohorts[1]:
            raise ValueError("Validation and test targets overlap")
        key = "node_id" if self.model.task == "node" else "graph_id"
        train_ids = {row[key] for row in self.train.metadata if key in row}
        if train_ids & (cohorts[0] | cohorts[1]):
            raise ValueError("Training and explanation targets overlap")
