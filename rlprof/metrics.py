"""Rollout-training mismatch statistics (the quantities in Figures 1, 3 and 4 of the paper)."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def mismatch_stats(logp_roll: torch.Tensor, logp_train: torch.Tensor, mask: torch.Tensor,
                   tis_cap: float = 2.0) -> Dict[str, float]:
    """Compare pi_vllm and pi_fsdp on the sampled tokens (same parameters theta_old).

    Samples come from pi_vllm, so with r = pi_fsdp/pi_vllm:
        kl_k1 = E[log pi_vllm - log pi_fsdp]              unbiased estimate of KL(pi_vllm || pi_fsdp)
        kl_k3 = E[r - 1 - log r]                           low-variance, always >= 0
    """
    m = mask.bool()
    lr, lt = logp_roll[m].double(), logp_train[m].double()
    if lr.numel() == 0:
        return {}
    pr, pt = lr.exp(), lt.exp()
    d = (pr - pt).abs()
    log_r = (lt - lr).clamp(-30, 30)
    r = log_r.exp()
    # sequence-level log ratio (sum over each sequence's tokens)
    seq_lr = ((logp_train - logp_roll) * mask).sum(1)
    corr = torch.corrcoef(torch.stack([pr, pt]))[0, 1] if pr.numel() > 1 else torch.tensor(float("nan"))
    return dict(
        tokens=int(lr.numel()),
        prob_absdiff_mean=float(d.mean()), prob_absdiff_max=float(d.max()),
        prob_absdiff_p99=float(torch.quantile(d.float(), 0.99)) if d.numel() < 16_000_000 else float("nan"),
        frac_absdiff_gt_0p1=float((d > 0.1).double().mean()),
        frac_absdiff_gt_0p5=float((d > 0.5).double().mean()),
        prob_corr=float(corr),
        kl_k1=float((lr - lt).mean()), kl_k3=float((r - 1 - log_r).mean()),
        ratio_mean=float(r.mean()), ratio_max=float(r.max()), ratio_min=float(r.min()),
        frac_ratio_gt_cap=float((r > tis_cap).double().mean()),
        frac_ratio_lt_inv_cap=float((r < 1 / tis_cap).double().mean()),
        seq_log_ratio_mean=float(seq_lr.mean()), seq_log_ratio_absmax=float(seq_lr.abs().max()),
    )


def prob_hist2d(logp_roll: torch.Tensor, logp_train: torch.Tensor, mask: torch.Tensor, bins: int = 50) -> np.ndarray:
    """bins x bins counts of (p_vllm, p_fsdp) per token; rows = p_vllm, cols = p_fsdp."""
    m = mask.bool()
    pr = logp_roll[m].float().exp().clamp(0, 1).cpu().numpy()
    pt = logp_train[m].float().exp().clamp(0, 1).cpu().numpy()
    h, _, _ = np.histogram2d(pr, pt, bins=bins, range=[[0, 1], [0, 1]])
    return h.astype(np.int64)


def group_advantages(rewards: torch.Tensor, group_size: int, scale_std: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """GRPO-style: reward minus the mean of its prompt's group, optionally divided by the group std."""
    g = rewards.view(-1, group_size)
    a = g - g.mean(1, keepdim=True)
    if scale_std:
        a = a / (g.std(1, keepdim=True) + eps)
    return a.view(-1)
