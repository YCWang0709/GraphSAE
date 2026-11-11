"""GraphSAE dictionary training and source explanation for frozen model adapters."""

from dataclasses import dataclass
import torch
from .sae import SAEArtifact, SAEConfig, train_sae
from .explainer import target_direction
from .policies import ExtractionPolicy, select_nodes, refine_nodes


@dataclass
class GraphSAE:
    artifact: SAEArtifact

    @classmethod
    def fit(cls, train_pack, prediction_head, config=None, *, device="cpu"):
        return cls(
            train_sae(
                train_pack,
                config or SAEConfig(),
                prediction_head=prediction_head,
                device=device,
            )
        )

    @classmethod
    def load(cls, path, *, device="cpu"):
        return cls(SAEArtifact.load(path, device=device))

    def save(self, path):
        return self.artifact.save(path)

    def explain(
        self,
        adapter,
        data,
        target,
        *,
        policy=None,
        prediction_head=None,
        evaluator=None,
        blocks=None,
    ):
        policy = policy or ExtractionPolicy()
        adapter.model.eval()
        with torch.no_grad():
            raw = adapter.representation(data, target).detach()
        gradient = None
        if policy.weighting == "prediction":
            if prediction_head is None:
                raise ValueError(
                    "prediction weighting requires a head for the target representation"
                )
            with torch.enable_grad():
                value = raw.clone().requires_grad_(True)
                prediction = prediction_head(value)
                if self.artifact.config.task_type == "regression":
                    score = prediction.flatten()[0]
                else:
                    from .evaluation import class_logits

                    row = class_logits(prediction.reshape(1, -1))[0]
                    score = row[row.detach().argmax()]
                (gradient,) = torch.autograd.grad(score, value)
        direction, features = target_direction(
            self.artifact, raw, weighting=policy.weighting, prediction_gradient=gradient
        )
        candidates, scores = adapter.project_sources(data, target, direction)
        # Integer ranking positions preserve typed/global identifiers unchanged.
        positions = list(range(len(candidates)))
        if policy.mode == "one_hop":
            if isinstance(target, tuple):
                from .adapters import heterogeneous_receptive_field

                neighborhood = set(heterogeneous_receptive_field(data, target, 1))
            else:
                neighborhood = set(
                    data.edge_index[0, data.edge_index[1] == target].tolist()
                )
            order = [i for i, u in enumerate(candidates) if u in neighborhood]
        else:
            order = select_nodes(positions, scores, policy)
        if policy.refine_steps:
            if policy.mode == "fixed" or evaluator is None:
                raise ValueError("Native refinement requires a deletion evaluator")
            if policy.refinement == "blocks" and blocks is None:
                raise ValueError(
                    "Custom-model block refinement requires source-ID blocks"
                )
            lookup = {u: i for i, u in enumerate(candidates)}
            block_positions = (
                None
                if blocks is None
                else [{lookup[u] for u in block if u in lookup} for block in blocks]
            )
            order = refine_nodes(
                order,
                positions,
                scores,
                lambda ids: evaluator([candidates[i] for i in ids]),
                steps=policy.refine_steps,
                blocks=block_positions,
                size_penalty=policy.size_penalty,
            )
        return {
            "selected_nodes": [candidates[i] for i in order],
            "active_features": features,
            "node_scores": list(zip(candidates, scores.tolist())),
        }
