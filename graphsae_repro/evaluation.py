"""Common deletion and representation metrics for classification and regression."""

import torch
import torch.nn.functional as F
from .masking import clone_with_node_mask
from .sae import faithfulness_loss, decoder_coherence_loss


def class_logits(output):
    if output.ndim == 1:
        output = output[:, None]
    return (
        torch.cat([torch.zeros_like(output), output], -1)
        if output.shape[-1] == 1
        else output
    )


def recovery_metrics(selected, truth):
    a, b = set(selected), set(truth)
    return {
        "precision": len(a & b) / len(a) if a else 0.0,
        "recall": len(a & b) / len(b) if b else 0.0,
        "jaccard": len(a & b) / len(a | b) if a | b else 0.0,
    }


@torch.no_grad()
def deletion_metrics(
    model,
    data,
    nodes,
    *,
    target_index=0,
    task_type="classification",
    truth=None,
    device="cpu",
    stable_sage=False,
):
    model = model.to(device).eval()
    data = data.to(device)
    nodes = list(nodes)
    if model.task == "node" and target_index in nodes:
        raise ValueError("The target node cannot be deleted")
    masked = clone_with_node_mask(data, nodes, mode="both")

    def predict(d):
        out = model(d.x, d.edge_index, batch=getattr(d, "batch", None))
        return out[0] if isinstance(out, tuple) else out

    before = predict(data)
    if stable_sage:
        from .numerics import stable_sage_deletion

        with stable_sage_deletion(model):
            after = predict(masked)
    else:
        after = predict(masked)
    result = {"num_nodes": len(set(nodes))}
    if task_type == "regression":
        result["absolute_change"] = float(
            (before[target_index] - after[target_index]).abs().mean()
        )
    else:
        before, after = class_logits(before), class_logits(after)
        c = before[target_index].argmax()
        result.update(
            logit_decrease=float(before[target_index, c] - after[target_index, c]),
            probability_decrease=float(
                before.softmax(-1)[target_index, c] - after.softmax(-1)[target_index, c]
            ),
        )
    if truth is not None:
        result.update(recovery_metrics(nodes, truth))
    return result


@torch.no_grad()
def representation_metrics(
    artifact, pack, *, prediction_head=None, train_pack=None, device="cpu"
):
    artifact.sae.to(device).eval()
    raw = pack.z.float().to(device)
    mean, std = artifact.mean.to(device), artifact.std.to(device)
    x = (raw - mean) / std
    reconstruction, codes = artifact.sae(x)
    mse = F.mse_loss(reconstruction, x)
    metrics = {
        "mse": float(mse),
        "nmse": float(mse / x.var(0, unbiased=False).mean().clamp_min(1e-12)),
        "mean_active_features": float((codes > 1e-8).sum(-1).float().mean()),
        "coherence": float(decoder_coherence_loss(artifact.sae.decoder.weight)),
    }
    if prediction_head is not None:
        prediction_head = prediction_head.to(device).eval()
        before, after = prediction_head(raw), prediction_head(
            reconstruction * std + mean
        )
        metrics["fidelity"] = float(
            faithfulness_loss(before, after, artifact.config.task_type)
        )
        if artifact.config.task_type == "classification":
            metrics["agreement"] = float(
                (class_logits(before).argmax(-1) == class_logits(after).argmax(-1))
                .float()
                .mean()
            )
        else:
            metrics["normalized_fidelity"] = metrics["fidelity"] / max(
                float(before.var(unbiased=False)), 1e-12
            )
    if train_pack is not None and artifact.config.task_type == "classification":
        training_codes = artifact.encode(train_pack.z).to(device)
        eligible = ((training_codes > 1e-8).sum(0) >= 3) & ((codes > 1e-8).sum(0) >= 5)
        labels = torch.tensor([r["label"] for r in pack.metadata], device=device)
        purity = []
        for k in torch.where(eligible)[0]:
            ids = codes[:, k].topk(5).indices
            purity.append(float(labels[ids].unique(return_counts=True)[1].max()) / 5)
        metrics["purity_at_5"] = sum(purity) / len(purity) if purity else None
    return metrics
