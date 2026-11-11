"""Native-size extraction, matched budgets, and frozen-model local refinement."""

from dataclasses import dataclass, replace
import math
import torch


@dataclass(frozen=True)
class ExtractionPolicy:
    weighting: str = "activity"
    mode: str = "fixed"
    budget: int = 12
    evidence: str = "positive"
    threshold: float = 0.0
    mass_fraction: float = 0.9
    refine_steps: int = 0
    size_penalty: float = 0.0
    refinement: str = "local"
    stable_sage: bool = False
    source_mode: str = "input"

    def __post_init__(self):
        if self.source_mode not in {"input", "readout"}:
            raise ValueError("source_mode must be input or readout")
        if self.mode not in {
            "fixed",
            "support",
            "relative",
            "mass",
            "prefix",
            "one_hop",
        }:
            raise ValueError("Unknown extraction mode")
        if self.evidence not in {
            "positive",
            "negative",
            "mixed",
        } or self.weighting not in {"activity", "prediction"}:
            raise ValueError("Unknown scoring policy")
        if self.refinement not in {"local", "blocks"}:
            raise ValueError("refinement must be local or blocks")
        if self.evidence == "mixed" and self.mode != "fixed":
            raise ValueError("Mixed evidence is defined for matched fixed budgets")
        if self.refinement == "blocks" and self.refine_steps > 8:
            raise ValueError("Block refinement permits at most eight improving moves")
        if (
            self.budget < 0
            or not 0 <= self.threshold <= 1
            or not 0 < self.mass_fraction <= 1
            or self.refine_steps < 0
        ):
            raise ValueError("Invalid extraction parameters")

    def with_evidence(self, evidence):
        return replace(self, evidence=evidence)


def select_nodes(candidates, scores, policy):
    """Ties use global node ID; fixed budgets include deterministic zero padding."""
    pairs = sorted(zip(candidates, map(float, scores)), key=lambda p: p[0])
    if (
        len(pairs) != len(candidates)
        or len({u for u, _ in pairs}) != len(pairs)
        or not all(math.isfinite(s) for _, s in pairs)
    ):
        raise ValueError(
            "Candidates must be unique and scores finite with equal lengths"
        )
    budget = min(policy.budget, len(pairs))
    if policy.evidence == "mixed":
        pos = sorted((p for p in pairs if p[1] > 0), key=lambda p: (-p[1], p[0]))
        neg = sorted((p for p in pairs if p[1] < 0), key=lambda p: (p[1], p[0]))
        selected = [u for u, _ in pos[: (budget + 1) // 2] + neg[: budget // 2]]
        used = set(selected)
        remaining = sorted(
            (p for p in pairs if p[0] not in used), key=lambda p: (-abs(p[1]), p[0])
        )
        return selected + [u for u, _ in remaining[: budget - len(selected)]]
    sign = 1 if policy.evidence == "positive" else -1
    ranked = sorted(
        [(u, max(0.0, sign * s)) for u, s in pairs], key=lambda p: (-p[1], p[0])
    )
    if policy.mode == "fixed":
        return [u for u, _ in ranked[:budget]]
    positive = [(u, s) for u, s in ranked if s > 0]
    if not positive:
        return []
    if policy.mode == "support":
        return [u for u, _ in positive]
    if policy.mode == "relative":
        return [u for u, s in positive if s >= positive[0][1] * policy.threshold]
    if policy.mode == "prefix":
        return [u for u, _ in positive[:budget]]
    if policy.mode == "mass":
        cutoff = sum(s for _, s in positive) * policy.mass_fraction
        total, selected = 0.0, []
        for u, s in positive:
            selected.append(u)
            total += s
            if total >= cutoff:
                break
        return selected
    raise ValueError("one_hop masks require topology")


def score_blocks(candidates, scores, edge_index, target_index=None, quantiles=8):
    """Blocks from sign/score quantiles, direct neighbors, and degree strata."""
    scores = torch.as_tensor(scores).cpu()
    ids = torch.tensor(candidates, dtype=torch.long)
    edge_index = edge_index.cpu()
    blocks = []
    for sign in (scores > 0, scores < 0, scores == 0):
        positions = torch.where(sign)[0]
        positions = positions[torch.argsort(scores[positions], stable=True)]
        blocks.extend(
            [
                set(ids[p].tolist())
                for p in torch.tensor_split(positions, quantiles)
                if p.numel()
            ]
        )
    degree = torch.bincount(
        edge_index.flatten(), minlength=max(candidates, default=0) + 1
    )
    strata = torch.floor(torch.log2(1 + degree[ids].float())).long()
    blocks.extend(set(ids[strata == k].tolist()) for k in strata.unique())
    if target_index is not None:
        neighbors = edge_index[0, edge_index[1] == target_index].tolist()
        blocks.append(set(neighbors) & set(candidates))
    ranked = ids[torch.argsort(scores, descending=True, stable=True)]
    blocks.extend(
        set(ranked[:k].tolist()) for k in (8, 16, 32, 64, 128) if k <= len(ranked)
    )
    return [b for b in blocks if b]


def refine_nodes(
    initial,
    candidates,
    scores,
    evaluator,
    *,
    steps=32,
    edge_index=None,
    blocks=None,
    tolerance=1e-6,
    size_penalty=0.0,
):
    """Greedy additions/removals evaluated through the frozen predictor."""
    eligible = set(candidates)
    selected = set(initial) & eligible
    if blocks is None:
        pool = set(initial)
        ranked = sorted(
            zip(candidates, map(float, scores)), key=lambda p: (-abs(p[1]), p[0])
        )
        pool.update(u for u, _ in ranked[:128])
        if edge_index is not None:
            edges = edge_index.cpu()
            ids = torch.tensor(sorted(pool), dtype=torch.long)
            mask = torch.isin(edges[0], ids) | torch.isin(edges[1], ids)
            pool.update(edges[:, mask].flatten().tolist())
        blocks = [{u} for u in sorted(pool & eligible)]
    moves = [set(b) & eligible for b in blocks]
    cache = {}

    def objective(nodes):
        key = tuple(sorted(nodes))
        if key not in cache:
            cache[key] = float(evaluator(list(key))) - size_penalty * len(key)
        return cache[key]

    best = objective(selected)
    for _ in range(steps):
        next_nodes, next_value = selected, best
        for block in moves:
            for trial in (selected | block, selected - block):
                value = objective(trial)
                if value > next_value + tolerance:
                    next_nodes, next_value = trial, value
        if next_value <= best + tolerance:
            break
        selected, best = next_nodes, next_value
    return sorted(selected)
