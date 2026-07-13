"""Top-k sparse autoencoder training and portable SAE artifacts."""

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
    expansion: int = 8
    latent_dim: int | None = None
    k_active: int = 16
    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    l1_weight: float = 1e-4
    weight_decay: float = 0.0
    standardize: bool = True
    seed: int = 0


class TopKSAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, k_active: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.k_active = int(k_active)
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight, a=5**0.5)
        nn.init.zeros_(self.decoder.bias)
        self.normalize_decoder_columns()

    @torch.no_grad()
    def normalize_decoder_columns(self) -> None:
        weights = self.decoder.weight.data
        self.decoder.weight.data = weights / weights.norm(dim=0, keepdim=True).clamp_min(1e-8)

    def encode(self, x: Tensor) -> Tensor:
        codes = F.relu(self.encoder(x))
        if 0 < self.k_active < codes.size(-1):
            values, indices = torch.topk(codes, k=self.k_active, dim=-1)
            sparse = torch.zeros_like(codes)
            sparse.scatter_(dim=-1, index=indices, src=values)
            codes = sparse
        return codes

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        codes = self.encode(x)
        return self.decoder(codes), codes


@dataclass
class SAEArtifact:
    sae: TopKSAE
    mean: Tensor
    std: Tensor
    config: SAEConfig
    metrics: dict[str, float] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def _standardize(self, x: Tensor) -> Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def encode(self, x: Tensor, *, device: torch.device | str = "cpu") -> Tensor:
        device = torch.device(device)
        self.sae = self.sae.to(device).eval()
        with torch.no_grad():
            return self.sae.encode(self._standardize(x.to(device).float())).detach().cpu()

    def decode(self, codes: Tensor, *, device: torch.device | str = "cpu") -> Tensor:
        device = torch.device(device)
        self.sae = self.sae.to(device).eval()
        with torch.no_grad():
            reconstructed = self.sae.decoder(codes.to(device).float())
            return (reconstructed * self.std.to(device) + self.mean.to(device)).detach().cpu()

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.sae.state_dict(),
                "input_dim": self.sae.input_dim,
                "latent_dim": self.sae.latent_dim,
                "k_active": self.sae.k_active,
                "mean": self.mean.detach().cpu(),
                "std": self.std.detach().cpu(),
                "config": asdict(self.config),
                "metrics": self.metrics,
                "history": self.history,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, *, device: torch.device | str = "cpu") -> "SAEArtifact":
        obj = torch.load(Path(path), map_location="cpu", weights_only=False)
        sae = TopKSAE(int(obj["input_dim"]), int(obj["latent_dim"]), int(obj["k_active"]))
        sae.load_state_dict(obj["state_dict"])
        sae.to(device).eval()
        return cls(
            sae=sae,
            mean=obj["mean"].float(),
            std=obj["std"].float(),
            config=SAEConfig(**obj["config"]),
            metrics=dict(obj.get("metrics", {})),
            history=list(obj.get("history", [])),
        )


def train_sae(pack: ActivationPack, config: SAEConfig | None = None, *, device: torch.device | str = "cpu") -> SAEArtifact:
    """Train a TopK SAE with MSE reconstruction plus a small L1 penalty."""
    config = SAEConfig() if config is None else config
    if pack.z.ndim != 2 or pack.z.size(0) == 0:
        raise ValueError("ActivationPack.z must be a non-empty [units, hidden_dim] tensor")
    torch.manual_seed(int(config.seed))
    device = torch.device(device)
    raw = pack.z.detach().float().cpu()
    mean = raw.mean(dim=0, keepdim=True) if config.standardize else torch.zeros((1, raw.size(1)))
    std = raw.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6) if config.standardize else torch.ones((1, raw.size(1)))
    normalized = (raw - mean) / std
    latent_dim = int(config.latent_dim) if config.latent_dim is not None else raw.size(1) * int(config.expansion)
    sae = TopKSAE(raw.size(1), latent_dim, config.k_active).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loader = DataLoader(TensorDataset(normalized), batch_size=min(config.batch_size, normalized.size(0)), shuffle=True)
    history: list[dict] = []

    for epoch in range(int(config.epochs)):
        sae.train()
        loss_sum = rec_sum = sparse_sum = 0.0
        seen = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed, codes = sae(batch)
            reconstruction = F.mse_loss(reconstructed, batch)
            sparsity = codes.abs().mean()
            loss = reconstruction + float(config.l1_weight) * sparsity
            loss.backward()
            optimizer.step()
            sae.normalize_decoder_columns()
            count = int(batch.size(0))
            loss_sum += float(loss.item()) * count
            rec_sum += float(reconstruction.item()) * count
            sparse_sum += float(sparsity.item()) * count
            seen += count
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == int(config.epochs):
            history.append({"epoch": epoch + 1, "loss": loss_sum / seen, "reconstruction": rec_sum / seen, "mean_abs_code": sparse_sum / seen})

    sae.eval()
    with torch.no_grad():
        reconstructed, codes = sae(normalized.to(device))
        active_fraction = float((codes > 0).float().mean().item())
        metrics = {
            "normalized_reconstruction_mse": float(F.mse_loss(reconstructed, normalized.to(device)).item()),
            "active_fraction": active_fraction,
            "mean_active_features": float((codes > 0).sum(dim=-1).float().mean().item()),
            "input_dim": float(raw.size(1)),
            "latent_dim": float(latent_dim),
        }
    return SAEArtifact(sae=sae.cpu(), mean=mean, std=std, config=config, metrics=metrics, history=history)
