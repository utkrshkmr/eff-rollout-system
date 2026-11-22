"""Policy-gradient losses with the rollout-training mismatch corrections of Yao et al. (2025).

Notation (all per token, in log space, shape [B, T], masked by `mask`):
    logp       log pi_fsdp(a, theta)       trainer, current parameters (has grad)
    logp_old   log pi_fsdp(a, theta_old)   trainer, parameters that produced the rollouts
    logp_roll  log pi_vllm(a, theta_old)   rollout engine, same parameters

correction:
    none        standard PPO clip; ignores that samples came from pi_vllm (the "mismatched" update)
    tis         truncated importance sampling: weight = min(pi_fsdp_old / pi_vllm_old, C), detached,
                multiplying the PPO objective whose ratio stays pi_fsdp(theta)/pi_fsdp(theta_old)
    ppo_is      PPO-IS: put pi_vllm_old in the PPO ratio denominator, clip that ratio
    vanilla_is  untruncated importance weight pi_fsdp_old / pi_vllm_old
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

CORRECTIONS = ("none", "tis", "ppo_is", "vanilla_is")


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp_min(1)


def policy_loss(logp: torch.Tensor, logp_old: torch.Tensor, logp_roll: torch.Tensor, adv: torch.Tensor,
                mask: torch.Tensor, correction: str = "tis", clip_low: float = 0.2, clip_high: float = 0.2,
                tis_cap: float = 2.0, denom: float | None = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Token-level PPO loss (sum over tokens / denom). `adv` is per token ([B, T]) or per sequence ([B]).

    `denom` is the number of valid tokens in the whole mini-batch so micro-batches can be summed
    (token-mean aggregation as in DAPO). Defaults to mask.sum().
    """
    if correction not in CORRECTIONS:
        raise ValueError(f"correction must be one of {CORRECTIONS}")
    if adv.dim() == 1:
        adv = adv[:, None].expand_as(logp)
    mask = mask.to(logp.dtype)
    denom = float(mask.sum().item()) if denom is None else denom
    base = logp_roll if correction == "ppo_is" else logp_old
    log_ratio = (logp - base).clamp(-20, 20)
    ratio = torch.exp(log_ratio)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_low, 1 + clip_high) * adv
    pg = -torch.minimum(surr1, surr2)

    is_log = (logp_old - logp_roll).clamp(-20, 20)
    w = torch.exp(is_log).detach()
    if correction == "tis":
        weight = torch.clamp(w, max=tis_cap)
        pg = pg * weight
    elif correction == "vanilla_is":
        weight = w
        pg = pg * weight
    else:
        weight = torch.ones_like(w)

    loss = (pg * mask).sum() / max(denom, 1.0)
    with torch.no_grad():
        clipped = ((surr2 < surr1) & (mask > 0)).float()
        stats = dict(
            ppo_clip_frac=float(clipped.sum() / mask.sum().clamp_min(1)),
            ppo_ratio_mean=float(masked_mean(ratio, mask)),
            is_weight_mean=float(masked_mean(weight, mask)),
            is_weight_max=float((weight * mask).max()),
            tis_truncated_frac=float(masked_mean((w > tis_cap).float(), mask)),
        )
    return loss, stats
