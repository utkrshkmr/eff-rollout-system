"""FSDP2 training policy (fp32 master weights, bf16 compute) on one GPU.

FSDP2 (`torch.distributed.fsdp.fully_shard`) is applied per decoder layer and at the root, as in
multi-GPU RL trainers; with one GPU the shards are whole tensors but the same code path (unshard /
reshard, mixed-precision casts, gradient reduction) runs and is visible in the CUPTI profile.

The lm_head is replaced by a function that returns per-token log-probs of the target tokens only,
computed in chunks with activation checkpointing, so the [tokens x vocab] logits never exist all at
once. It runs inside the root module's forward, where FSDP has the head weight unsharded.
"""
from __future__ import annotations

import os
import types
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _ensure_process_group():
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
    dist.init_process_group("nccl", rank=0, world_size=1)


def _chunk_logprob(h, w, tgt, inv_temp, fp32):
    logits = F.linear(h.float(), w.float()) if fp32 else F.linear(h, w).float()
    if inv_temp != 1.0:
        logits = logits * inv_temp
    return logits.log_softmax(-1).gather(-1, tgt[:, None]).squeeze(-1)


def _chunk_logprob_entropy(h, w, tgt, inv_temp, fp32):
    """No-grad path: log p(target) and entropy with two [chunk, vocab] fp32 buffers.
    entropy = logsumexp(z) - sum softmax(z) * z."""
    logits = F.linear(h.float(), w.float()) if fp32 else F.linear(h, w).float()
    if inv_temp != 1.0:
        logits.mul_(inv_temp)
    lse = torch.logsumexp(logits, -1)
    lp = logits.gather(-1, tgt[:, None]).squeeze(-1) - lse
    p = torch.exp(logits - lse[:, None])
    ent = lse - (p.mul_(logits)).sum(-1)
    return lp, ent


def _head_forward(self, hidden):
    """Replacement lm_head.forward: per-token log-probs of ctx targets (and entropy without grad)."""
    ctx = self._rlprof_ctx
    pos, tgt = ctx["pos"], ctx["tgt"]            # flat indices into [B*T] and their target ids
    B, T, H = hidden.shape
    h = hidden.reshape(B * T, H)[pos]
    w = self.weight
    out_lp, out_ent = [], []
    chunk = ctx["chunk"]
    for i in range(0, h.shape[0], chunk):
        hs, ts = h[i:i + chunk], tgt[i:i + chunk]
        if ctx["entropy"]:
            with torch.no_grad():
                lp, ent = _chunk_logprob_entropy(hs, w, ts, ctx["inv_temp"], ctx["fp32"])
            out_ent.append(ent)
        elif torch.is_grad_enabled():
            lp = checkpoint(_chunk_logprob, hs, w, ts, ctx["inv_temp"], ctx["fp32"], use_reentrant=False)
        else:
            lp = _chunk_logprob(hs, w, ts, ctx["inv_temp"], ctx["fp32"])
        out_lp.append(lp)
    ctx["out_entropy"] = torch.cat(out_ent) if out_ent else None
    return torch.cat(out_lp) if out_lp else hidden.new_zeros(0)


class Policy:
    def __init__(self, model_path: str, lr: float = 1e-6, weight_decay: float = 0.0, betas=(0.9, 0.999),
                 gradient_checkpointing: bool = True, fp32_head: bool = False, head_chunk: int = 2048,
                 attn_impl: str = "sdpa"):
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
        from transformers import AutoModelForCausalLM

        _ensure_process_group()
        self.device = torch.device("cuda", torch.cuda.current_device())
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32, attn_implementation=attn_impl)
        model.config.use_cache = False
        # from_pretrained returns eval mode, and HF gradient checkpointing only runs in train mode
        # (the Qwen/Llama families used here have no dropout, so train mode changes nothing else)
        model.train()
        if gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.to(self.device)
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in model.model.layers:
            fully_shard(layer, mp_policy=mp)
        fully_shard(model, mp_policy=mp)
        head = model.get_output_embeddings()
        head.forward = types.MethodType(_head_forward, head)
        self.head = head
        self.model = model
        self.fp32_head = fp32_head
        self.head_chunk = head_chunk
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)

    # ------------------------------------------------------------------ data
    @staticmethod
    def make_microbatches(lengths: List[int], max_tokens: int) -> List[List[int]]:
        """Group sequence indices (sorted by length) so that padded B*T_max <= max_tokens."""
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        mbs, cur, cur_max = [], [], 0
        for i in order:
            L = lengths[i]
            if cur and max(cur_max, L) * (len(cur) + 1) > max_tokens:
                mbs.append(cur); cur, cur_max = [], 0
            cur.append(i); cur_max = max(cur_max, L)
        if cur:
            mbs.append(cur)
        return mbs

    def _collate(self, seqs: List[Tuple[List[int], List[int]]]):
        """(prompt_ids, response_ids) -> right-padded input ids, attention mask, response mask."""
        T = max(len(p) + len(r) for p, r in seqs)
        B = len(seqs)
        ids = torch.zeros(B, T, dtype=torch.long)
        attn = torch.zeros(B, T, dtype=torch.long)
        rmask = torch.zeros(B, T, dtype=torch.float32)   # 1 where position t predicts a response token
        for b, (p, r) in enumerate(seqs):
            s = p + r
            ids[b, :len(s)] = torch.tensor(s)
            attn[b, :len(s)] = 1
            rmask[b, len(p) - 1:len(s) - 1] = 1
        return ids.to(self.device, non_blocking=True), attn.to(self.device, non_blocking=True), rmask.to(self.device, non_blocking=True)

    def forward_logprobs(self, seqs, temperature: float = 1.0, entropy: bool = False):
        """Per-position log-prob of the next token, [B, T] (zeros outside the response)."""
        ids, attn, rmask = self._collate(seqs)
        B, T = ids.shape
        tgt_full = torch.roll(ids, -1, dims=1)
        pos = rmask.reshape(-1).nonzero().squeeze(-1)
        self.head._rlprof_ctx = dict(pos=pos, tgt=tgt_full.reshape(-1)[pos], inv_temp=1.0 / temperature,
                                     fp32=self.fp32_head, chunk=self.head_chunk, entropy=entropy)
        out = self.model(input_ids=ids, attention_mask=attn, use_cache=False)
        lp_flat = out.logits
        lp = torch.zeros(B * T, device=self.device, dtype=lp_flat.dtype)
        lp = lp.index_put((pos,), lp_flat).view(B, T)
        ent = None
        if entropy:
            ent = torch.zeros(B * T, device=self.device).index_put((pos,), self.head._rlprof_ctx["out_entropy"]).view(B, T)
        self.head._rlprof_ctx = None
        return lp, rmask, ent

    @staticmethod
    def response_view(x: torch.Tensor, seqs) -> List[torch.Tensor]:
        """[B, T] per-position values -> list of per-response-token 1D tensors."""
        out = []
        for b, (p, r) in enumerate(seqs):
            out.append(x[b, len(p) - 1:len(p) - 1 + len(r)])
        return out

    # --------------------------------------------------------------- weights
    @torch.no_grad()
    def export_weights(self, dtype=torch.bfloat16) -> Iterator[Tuple[str, torch.Tensor]]:
        """Full (unsharded) parameters in HF naming, cast for the rollout engine."""
        from torch.distributed.tensor import DTensor
        for name, p in self.model.named_parameters():
            t = p.full_tensor() if isinstance(p, DTensor) else p
            yield name, t.to(dtype)

    def clip_and_step(self, max_norm: float) -> float:
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        if hasattr(norm, "full_tensor"):
            norm = norm.full_tensor()
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return float(norm)
