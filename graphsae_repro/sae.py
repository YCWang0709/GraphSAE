"""Graph-grounded Top-K sparse autoencoder and its training objectives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .activations import ActivationPack


@dataclass
class SAEConfig:
    """Hyperparameters for the GraphSAE objective in the paper."""

    expansion: int = 8
    latent_dim: int | None = None
    k_active: int = 16
    epochs: int = 100
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.0
    standardize: bool = True
    lambda_rec: float = 1.0
    lambda_ground: float = 0.01
    lambda_stab: float = 0.0
    lambda_faith: float = 0.1
    lambda_coh: float = 0.01
    regularizer_ramp_epochs: int = 5
    stability_noise_std: float = 0.01
    normalize_decoder: bool = True
    encoder_type: str = "topk"
    l1_weight: float = 0.0
    l0_weight: float = 1e-3
    threshold_bandwidth: float = 1e-3
    task_type: str = "classification"
    seed: int = 0


class TopKSAE(nn.Module):
    """Nonnegative Top-K SAE with decoder matrix ``D`` stored column-wise."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        k_active: int,
        *,
        normalize_decoder: bool = True,
        encoder_type: str = "topk",
        threshold_bandwidth: float = 1e-3,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.k_active = int(k_active)
        self.normalize_decoder = bool(normalize_decoder)
        if encoder_type not in {"topk", "relu", "batchtopk", "jumprelu", "gated"}:
            raise ValueError("Unknown encoder_type")
        if min(input_dim, latent_dim, k_active) < 1 or k_active > latent_dim:
            raise ValueError(
                "Require positive dimensions and 1 <= k_active <= latent_dim"
            )
        self.encoder_type = encoder_type
        self.threshold_bandwidth = threshold_bandwidth
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        if encoder_type == "jumprelu":
            self.log_threshold = nn.Parameter(torch.full((latent_dim,), -2.0))
        if encoder_type == "gated":
            self.log_magnitude_scale = nn.Parameter(torch.zeros(latent_dim))
            self.magnitude_bias = nn.Parameter(torch.zeros(latent_dim))
        if encoder_type == "batchtopk":
            self.register_buffer("activation_threshold", torch.tensor(0.0))
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight, a=5**0.5)
        nn.init.zeros_(self.decoder.bias)
        self.normalize_decoder_columns()

    @torch.no_grad()
    def normalize_decoder_columns(self) -> None:
        if not self.normalize_decoder:
            return
        weights = self.decoder.weight.data
        self.decoder.weight.data = weights / weights.norm(
            dim=0, keepdim=True
        ).clamp_min(1e-8)

    def encode(self, standardized: Tensor) -> Tensor:
        """Compute ``z_tau = TopK(ReLU(W_enc s_tilde + b_enc))``."""
        if self.encoder_type == "gated":
            centered = standardized - self.decoder.bias
            projection = F.linear(centered, self.encoder.weight)
            magnitude = F.relu(
                projection * self.log_magnitude_scale.exp() + self.magnitude_bias
            )
            return magnitude * (projection + self.encoder.bias > 0)
        codes = F.relu(self.encoder(standardized))
        if self.encoder_type == "jumprelu":
            from .sparse_ops import JumpReLU

            threshold = self.log_threshold.exp()
            codes = JumpReLU.apply(codes, threshold, self.threshold_bandwidth)
        elif self.encoder_type == "batchtopk" and self.training:
            values, indices = codes.flatten().topk(
                min(codes.numel(), codes.size(0) * self.k_active)
            )
            codes = (
                torch.zeros_like(codes)
                .flatten()
                .scatter(0, indices, values)
                .view_as(codes)
            )
        elif self.encoder_type == "batchtopk":
            codes = codes * (codes > self.activation_threshold)
        elif self.encoder_type == "topk" and 0 < self.k_active < codes.size(-1):
            values, indices = torch.topk(codes, k=self.k_active, dim=-1)
            sparse = torch.zeros_like(codes)
            sparse.scatter_(dim=-1, index=indices, src=values)
            codes = sparse
        return codes

    def forward(self, standardized: Tensor) -> tuple[Tensor, Tensor]:
        codes = self.encode(standardized)
        return self.decoder(codes), codes


def grounding_entropy_loss(
    codes: Tensor,
    group_ids: Tensor,
    *,
    num_groups: int | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Compute the mean semantic entropy over features active in the batch."""
    if codes.ndim != 2 or group_ids.ndim != 1 or codes.size(0) != group_ids.numel():
        raise ValueError("codes and group_ids must have shapes [N, F] and [N]")
    num_groups = int(group_ids.max().item()) + 1 if num_groups is None else num_groups
    group_mass = torch.zeros(
        num_groups, codes.size(1), dtype=codes.dtype, device=codes.device
    )
    group_mass.index_add_(0, group_ids, codes)
    feature_mass = group_mass.sum(dim=0)
    probabilities = (group_mass + eps) / (feature_mass.unsqueeze(0) + num_groups * eps)
    entropies = -(probabilities * probabilities.log()).sum(dim=0)
    active = feature_mass > 0
    return entropies[active].mean() if bool(active.any()) else codes.sum() * 0.0


def stability_loss(
    clean_codes: Tensor,
    noisy_codes: Tensor,
    clean_reconstruction: Tensor,
    noisy_reconstruction: Tensor,
) -> Tensor:
    """Match clean and perturbed codes and reconstructions with stop-gradient targets."""
    return F.mse_loss(noisy_codes, clean_codes.detach()) + F.mse_loss(
        noisy_reconstruction,
        clean_reconstruction.detach(),
    )


def faithfulness_loss(
    clean_logits: Tensor,
    reconstructed_logits: Tensor,
    task_type: str = "classification",
) -> Tensor:
    """Compute ``KL(p(s_tau) || p(s_hat_tau))`` for the frozen prediction head."""
    if task_type == "regression":
        return F.mse_loss(reconstructed_logits, clean_logits.detach())
    if clean_logits.ndim == 1 or clean_logits.shape[-1] == 1:
        clean_logits = torch.stack(
            (torch.zeros_like(clean_logits.reshape(-1)), clean_logits.reshape(-1)), -1
        )
        reconstructed_logits = torch.stack(
            (
                torch.zeros_like(reconstructed_logits.reshape(-1)),
                reconstructed_logits.reshape(-1),
            ),
            -1,
        )
    target_probabilities = F.softmax(clean_logits.detach(), dim=-1)
    reconstructed_log_probabilities = F.log_softmax(reconstructed_logits, dim=-1)
    return F.kl_div(
        reconstructed_log_probabilities, target_probabilities, reduction="batchmean"
    )


def decoder_coherence_loss(decoder_weight: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Compute the mean squared off-diagonal cosine similarity of decoder columns."""
    feature_count = int(decoder_weight.size(1))
    if feature_count < 2:
        return decoder_weight.sum() * 0.0
    directions = decoder_weight / decoder_weight.norm(dim=0, keepdim=True).clamp_min(
        eps
    )
    gram = directions @ directions.T
    diagonal = directions.square().sum(0).square().sum()
    return (gram.square().sum() - diagonal).clamp_min(0) / (
        feature_count * (feature_count - 1)
    )


@dataclass
class SAEArtifact:
    sae: TopKSAE
    mean: Tensor
    std: Tensor
    config: SAEConfig
    group_vocab: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def _standardize(self, raw: Tensor) -> Tensor:
        return (raw - self.mean.to(raw.device)) / self.std.to(raw.device)

    def encode(self, raw: Tensor, *, device: torch.device | str = "cpu") -> Tensor:
        device = torch.device(device)
        self.sae = self.sae.to(device).eval()
        with torch.no_grad():
            return (
                self.sae.encode(self._standardize(raw.to(device).float()))
                .detach()
                .cpu()
            )

    def decode(self, codes: Tensor, *, device: torch.device | str = "cpu") -> Tensor:
        device = torch.device(device)
        self.sae = self.sae.to(device).eval()
        with torch.no_grad():
            standardized = self.sae.decoder(codes.to(device).float())
            return (
                (standardized * self.std.to(device) + self.mean.to(device))
                .detach()
                .cpu()
            )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.sae.state_dict(),
                "input_dim": self.sae.input_dim,
                "latent_dim": self.sae.latent_dim,
                "k_active": self.sae.k_active,
                "normalize_decoder": self.sae.normalize_decoder,
                "mean": self.mean.detach().cpu(),
                "std": self.std.detach().cpu(),
                "config": asdict(self.config),
                "group_vocab": self.group_vocab,
                "metrics": self.metrics,
                "history": self.history,
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls, path: str | Path, *, device: torch.device | str = "cpu"
    ) -> "SAEArtifact":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        sae = TopKSAE(
            int(payload["input_dim"]),
            int(payload["latent_dim"]),
            int(payload["k_active"]),
            normalize_decoder=bool(payload.get("normalize_decoder", True)),
            encoder_type=payload["config"].get("encoder_type", "topk"),
            threshold_bandwidth=payload["config"].get("threshold_bandwidth", 1e-3),
        )
        sae.load_state_dict(payload["state_dict"])
        sae.to(device).eval()
        return cls(
            sae=sae,
            mean=payload["mean"].float(),
            std=payload["std"].float(),
            config=SAEConfig(**payload["config"]),
            group_vocab=dict(payload.get("group_vocab", {})),
            metrics=dict(payload.get("metrics", {})),
            history=list(payload.get("history", [])),
        )


def _encode_groups(pack: ActivationPack) -> tuple[Tensor, dict[str, int]]:
    labels = [row.get("group", row.get("label")) for row in pack.metadata]
    if any(label is None for label in labels):
        raise ValueError(
            "Every activation row must have a label when lambda_ground is positive"
        )
    values = sorted({str(label) for label in labels})
    vocabulary = {value: index for index, value in enumerate(values)}
    return (
        torch.tensor([vocabulary[str(label)] for label in labels], dtype=torch.long),
        vocabulary,
    )


def _active_weights(epoch: int, config: SAEConfig) -> dict[str, float]:
    ramp = min(1.0, (epoch + 1) / max(1, int(config.regularizer_ramp_epochs)))
    return {
        "rec": float(config.lambda_rec),
        "ground": float(config.lambda_ground) * ramp,
        "stab": float(config.lambda_stab) * ramp,
        "faith": float(config.lambda_faith) * ramp,
        "coh": float(config.lambda_coh) * ramp,
    }


def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def train_sae(
    pack: ActivationPack,
    config: SAEConfig | None = None,
    *,
    prediction_head: nn.Module | None = None,
    device: torch.device | str = "cpu",
) -> SAEArtifact:
    """Train on training representations; optional losses require groups/a frozen head."""
    config = SAEConfig() if config is None else config
    if config.epochs < 1 or config.batch_size < 1 or config.lr <= 0:
        raise ValueError("epochs, batch_size and lr must be positive")
    if any(
        value < 0
        for value in (
            config.lambda_rec,
            config.lambda_ground,
            config.lambda_stab,
            config.lambda_faith,
            config.lambda_coh,
            config.l1_weight,
            config.l0_weight,
        )
    ):
        raise ValueError("Loss weights must be nonnegative")
    if config.task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be classification or regression")
    if config.lambda_faith and prediction_head is None:
        raise ValueError("A frozen prediction_head is required for faithfulness")
    if pack.info.get("split") in {"val", "validation", "test"} or any(
        row.get("split") in {"val", "validation", "test"} for row in pack.metadata
    ):
        raise ValueError("SAE fitting requires training representations")
    if pack.z.ndim != 2 or pack.z.size(0) == 0:
        raise ValueError(
            "ActivationPack.z must be a non-empty [units, hidden_dim] tensor"
        )
    if len(pack.metadata) != pack.z.size(0):
        raise ValueError("ActivationPack metadata must contain one row per activation")

    torch.manual_seed(int(config.seed))
    device = torch.device(device)
    raw = pack.z.detach().float().cpu()
    mean = (
        raw.mean(dim=0, keepdim=True)
        if config.standardize
        else torch.zeros((1, raw.size(1)))
    )
    std = (
        raw.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        if config.standardize
        else torch.ones((1, raw.size(1)))
    )
    standardized = (raw - mean) / std
    if not torch.isfinite(raw).all():
        raise ValueError("Training representations must be finite")
    group_ids, group_vocab = (
        _encode_groups(pack)
        if config.lambda_ground
        else (torch.zeros(len(raw), dtype=torch.long), {})
    )
    latent_dim = (
        int(config.latent_dim)
        if config.latent_dim is not None
        else raw.size(1) * int(config.expansion)
    )
    sae = TopKSAE(
        raw.size(1),
        latent_dim,
        config.k_active,
        normalize_decoder=config.normalize_decoder,
        encoder_type=config.encoder_type,
        threshold_bandwidth=config.threshold_bandwidth,
    ).to(device)
    if prediction_head is not None:
        from copy import deepcopy

        prediction_head = _freeze(deepcopy(prediction_head).to(device))
    optimizer = torch.optim.Adam(
        sae.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(int(config.seed))
    loader = DataLoader(
        TensorDataset(standardized, group_ids),
        batch_size=min(config.batch_size, standardized.size(0)),
        shuffle=True,
        generator=generator,
    )
    history: list[dict] = []

    for epoch in range(int(config.epochs)):
        sae.train()
        weights = _active_weights(epoch, config)
        totals = {
            "loss": 0.0,
            "rec": 0.0,
            "ground": 0.0,
            "stab": 0.0,
            "faith": 0.0,
            "coh": 0.0,
            "magnitude": 0.0,
            "encoder_aux": 0.0,
            "l0": 0.0,
        }
        seen = 0

        for standardized_batch, group_batch in loader:
            standardized_batch = standardized_batch.to(device)
            group_batch = group_batch.to(device)
            optimizer.zero_grad()
            reconstruction, codes = sae(standardized_batch)

            losses = {
                "rec": F.mse_loss(reconstruction, standardized_batch),
                "ground": (
                    grounding_entropy_loss(
                        codes, group_batch, num_groups=max(1, len(group_vocab))
                    )
                    if config.lambda_ground
                    else codes.sum() * 0
                ),
                "coh": decoder_coherence_loss(sae.decoder.weight),
            }
            losses["stab"] = codes.sum() * 0
            if config.lambda_stab:
                noise = torch.randn_like(standardized_batch) * float(
                    config.stability_noise_std
                )
                noisy_reconstruction, noisy_codes = sae(standardized_batch + noise)
                losses["stab"] = stability_loss(
                    codes, noisy_codes, reconstruction, noisy_reconstruction
                )

            clean_raw = standardized_batch * std.to(device) + mean.to(device)
            reconstructed_raw = reconstruction * std.to(device) + mean.to(device)
            losses["faith"] = (
                faithfulness_loss(
                    prediction_head(clean_raw),
                    prediction_head(reconstructed_raw),
                    config.task_type,
                )
                if prediction_head is not None and config.lambda_faith
                else codes.sum() * 0
            )
            losses["magnitude"] = (codes * sae.decoder.weight.norm(dim=0)).sum() / (
                codes.size(0) * raw.size(1)
            )
            weights["magnitude"] = config.l1_weight
            losses["encoder_aux"] = codes.sum() * 0
            losses["l0"] = codes.sum() * 0
            weights["encoder_aux"] = 1.0
            weights["l0"] = (
                config.l0_weight if config.encoder_type == "jumprelu" else 0.0
            )
            if config.encoder_type == "jumprelu":
                from .sparse_ops import Step

                pre = F.relu(sae.encoder(standardized_batch))
                losses["l0"] = (
                    Step.apply(pre, sae.log_threshold.exp(), config.threshold_bandwidth)
                    .sum(-1)
                    .mean()
                )
            if config.encoder_type == "gated":
                gate_acts = F.relu(sae.encoder(standardized_batch - sae.decoder.bias))
                losses["magnitude"] = (
                    gate_acts * sae.decoder.weight.detach().norm(dim=0)
                ).sum() / (codes.size(0) * raw.size(1))
                via_gate = F.linear(
                    gate_acts, sae.decoder.weight.detach(), sae.decoder.bias.detach()
                )
                losses["encoder_aux"] = F.mse_loss(via_gate, standardized_batch)
            total_loss = sum(weights[name] * losses[name] for name in losses)
            total_loss.backward()
            optimizer.step()
            sae.normalize_decoder_columns()

            count = int(standardized_batch.size(0))
            totals["loss"] += float(total_loss.detach().item()) * count
            for name, value in losses.items():
                totals[name] += float(value.detach().item()) * count
            seen += count

        history.append(
            {
                "epoch": epoch + 1,
                **{name: value / max(1, seen) for name, value in totals.items()},
                **{f"lambda_{name}": value for name, value in weights.items()},
            }
        )

    if config.encoder_type == "batchtopk":
        with torch.no_grad():
            cutoffs = []
            for batch in standardized.split(config.batch_size):
                pre = F.relu(sae.encoder(batch.to(device)))
                kept = (
                    pre.flatten()
                    .topk(min(pre.numel(), len(batch) * config.k_active))
                    .values
                )
                cutoffs.append(kept.min())
            sae.activation_threshold.copy_(torch.stack(cutoffs).mean())
    sae.eval()
    with torch.no_grad():
        error, active = 0.0, 0
        for batch in standardized.split(config.batch_size):
            batch = batch.to(device)
            reconstruction, codes = sae(batch)
            error += float(F.mse_loss(reconstruction, batch, reduction="sum"))
            active += int((codes > 0).sum())
        metrics = {
            "normalized_reconstruction_mse": error / standardized.numel(),
            "active_fraction": active / (len(raw) * latent_dim),
            "mean_active_features": active / len(raw),
            "input_dim": float(raw.size(1)),
            "latent_dim": float(latent_dim),
        }
    return SAEArtifact(
        sae=sae.cpu(),
        mean=mean,
        std=std,
        config=config,
        group_vocab=group_vocab,
        metrics=metrics,
        history=history,
    )
