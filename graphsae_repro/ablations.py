"""Ablation deltas reuse the main training and explanation implementations."""

from dataclasses import replace
from itertools import product
import torch
from .sae import SAEConfig, train_sae
from .tuning import subset_pack
from .contributions import source_scores


LOSS_ABLATIONS = {
    "no_reconstruction": {"lambda_rec": 0.0},
    "no_grounding": {"lambda_ground": 0.0},
    "no_faithfulness": {"lambda_faith": 0.0},
    "no_coherence": {"lambda_coh": 0.0},
    "reconstruction_only": {
        "lambda_ground": 0.0,
        "lambda_faith": 0.0,
        "lambda_coh": 0.0,
    },
}


def loss_configurations(base=None):
    base = base or SAEConfig()
    return {name: replace(base, **delta) for name, delta in LOSS_ABLATIONS.items()}


def sensitivity_configurations(base=None):
    base = base or SAEConfig()
    configs = {}
    for key in ("lambda_ground", "lambda_faith", "lambda_coh"):
        for multiplier in (0.1, 0.3, 1.0, 3.0, 10.0):
            if multiplier != 1:
                configs[f"{key}_{multiplier:g}"] = replace(
                    base, **{key: getattr(base, key) * multiplier}
                )
    for expansion in (2, 4, 8, 16):
        if expansion != base.expansion:
            configs[f"width_{expansion}"] = replace(base, expansion=expansion)
    for g, f in product((0.1, 1.0, 10.0), repeat=2):
        if (g, f) != (1.0, 1.0):
            configs[f"joint_{g:g}_{f:g}"] = replace(
                base,
                lambda_ground=base.lambda_ground * g,
                lambda_faith=base.lambda_faith * f,
            )
    return configs


def train_ablations(
    pack, configurations, *, prediction_head=None, seeds=(0, 1, 2), device="cpu"
):
    """Identical sampled training population and train-only normalizer across deltas."""
    train = subset_pack(pack, limit=2048, seed=0)
    for name, config in configurations.items():
        for seed in seeds:
            yield name, seed, train_sae(
                train,
                replace(config, seed=seed),
                prediction_head=prediction_head,
                device=device,
            )


def pure_path_scores(
    model, data, artifact, *, target_index=0, device="cpu", source_mode="input"
):
    """Q(u)=<c_u/rho, (s-mu)/rho>; native selection keeps Q(u)>0."""
    with torch.no_grad():
        model = model.to(device).eval()
        data = data.to(device)
        _, capture = model(
            data.x, data.edge_index, batch=getattr(data, "batch", None), capture=True
        )
    target = (capture.graph if model.task == "graph" else capture.node)[
        target_index
    ].cpu()
    direction = (target - artifact.mean.reshape(-1)) / artifact.std.reshape(-1).square()
    return source_scores(
        model,
        data,
        direction,
        target_index=target_index,
        device=device,
        source_mode=source_mode,
    )


def encoder_configurations(base=None):
    """Alternative sparse encoders; the full TopK setting stays in the main pipeline."""
    base = base or SAEConfig()
    return {
        name: replace(
            base,
            encoder_type=name,
            l1_weight=1e-3 if name in {"relu", "gated", "jumprelu"} else 0.0,
        )
        for name in ("batchtopk", "jumprelu", "gated", "relu")
    }
