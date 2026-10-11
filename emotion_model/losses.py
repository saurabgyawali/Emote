"""Losses that respect what each corpus can and cannot say.

masked_cross_entropy normalises only over the clip's own corpus label set, so CREMA-D
(six classes) is never trained to push "surprise" down. ccc_loss is 1 - concordance
correlation, the standard objective for arousal/valence/dominance, computed only over
clips that carry dimensional labels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor, class_mask: torch.Tensor,
                         class_weight: torch.Tensor | None = None, smoothing: float = 0.0) -> torch.Tensor | None:
    """Mean loss over clips with target >= 0; None if the batch has none."""
    valid = target >= 0
    if not valid.any():
        return None
    logits = logits[valid].float().masked_fill(~class_mask[valid], -1e4)
    t = target[valid]
    m = class_mask[valid].float()
    logp = F.log_softmax(logits, dim=-1)
    nll = -logp.gather(1, t.unsqueeze(1)).squeeze(1)
    smooth = -(logp * m).sum(-1) / m.sum(-1)
    loss = (1 - smoothing) * nll + smoothing * smooth
    if class_weight is None:
        return loss.mean()
    w = class_weight[t]
    return (loss * w).sum() / w.sum()


def ccc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pm, tm = pred.mean(), target.mean()
    pv, tv = pred.var(unbiased=False), target.var(unbiased=False)
    cov = ((pred - pm) * (target - tm)).mean()
    return 2 * cov / (pv + tv + (pm - tm) ** 2 + 1e-8)


def ccc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    """1 - mean CCC over the dimensions with at least 2 labelled clips in the batch."""
    losses = []
    for j in range(target.shape[1]):
        ok = torch.isfinite(target[:, j])
        if ok.sum() >= 2:
            losses.append(1 - ccc(pred[ok, j].float(), target[ok, j].float()))
    return torch.stack(losses).mean() if losses else None
