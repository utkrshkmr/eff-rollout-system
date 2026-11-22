"""RL fine-tuning loop: vLLM rollouts -> rewards -> FSDP2 PPO update, with every phase CUPTI-profiled.

One training step (each item is a named profiler phase):
    weight_sync/{gather, quantize, vllm_load}   FSDP params -> bf16 (-> int8) -> vLLM reload
    rollout                                      vLLM sampling; returns log pi_vllm(theta_old)
    reward                                       rule-based verification (CPU)
    advantage                                    group-normalized advantages, drop uniform groups
    old_logprob                                  FSDP no-grad forward; log pi_fsdp(theta_old), entropy
    mismatch                                     pi_vllm vs pi_fsdp statistics + histogram
    train/{forward, backward, optimizer}         PPO loss with the chosen mismatch correction
"""
from __future__ import annotations

import json
import os
import random
import shutil
import time
from typing import Dict, List

import numpy as np
import torch

from . import data as data_mod
from . import paths, quant
from .config import Config
from .cupti_profiler import CuptiProfiler
from .losses import policy_loss
from .metrics import group_advantages, mismatch_stats, prob_hist2d
from .registry import resolve_model
from .rewards import score


def _pad2d(rows: List[torch.Tensor], seqs, T: int, device) -> torch.Tensor:
    """Per-response-token 1D tensors -> [B, T] aligned with Policy._collate's response mask."""
    out = torch.zeros(len(rows), T, device=device)
    for b, ((p, r), v) in enumerate(zip(seqs, rows)):
        out[b, len(p) - 1:len(p) - 1 + len(r)] = v.to(device)
    return out


class Trainer:
    def __init__(self, cfg: Config):
        from transformers import AutoTokenizer

        from .policy import Policy
        from .rollout import RolloutEngine

        self.cfg = cfg
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        self.name = cfg.run_name or cfg.default_run_name()
        self.out = os.path.join(paths.RUNS, self.name)
        for sub in ("profile", "hist", "samples"):  # a run with an existing name starts clean
            shutil.rmtree(os.path.join(self.out, sub), ignore_errors=True)
        os.makedirs(os.path.join(self.out, "profile"), exist_ok=True)
        os.makedirs(os.path.join(self.out, "hist"), exist_ok=True)
        os.makedirs(os.path.join(self.out, "samples"), exist_ok=True)
        with open(os.path.join(self.out, "config.json"), "w") as f:
            f.write(cfg.to_json())
        self.metrics_f = open(os.path.join(self.out, "metrics.jsonl"), "w")

        self.model_path = resolve_model(cfg.model)
        self.tok = AutoTokenizer.from_pretrained(self.model_path)
        rollout_path = self.model_path
        if cfg.rollout_quant == "int8":
            rollout_path = quant.int8_checkpoint_dir(paths.MODELS, self.model_path)
            if not os.path.exists(os.path.join(rollout_path, "config.json")):
                raise FileNotFoundError(f"{rollout_path} missing: run `python scripts/04_make_int8_checkpoint.py "
                                        f"--model {cfg.model}`")
        self.prof = CuptiProfiler(record_api=cfg.profile_api)

        self.train_rows = self._tokenize(data_mod.load(cfg.dataset, "train", shuffle=True, seed=cfg.seed))
        ev = cfg.eval_dataset or cfg.dataset
        self.eval_rows = self._tokenize(data_mod.load(ev, "test", limit=cfg.eval_limit or None))
        self.cursor = 0
        self.weights_stale = False
        max_len = cfg.max_prompt_tokens + max(cfg.max_new_tokens, cfg.eval_max_new_tokens or 0)

        t = time.time()
        self.engine = RolloutEngine(rollout_path, self.model_path, cfg.rollout_quant,
                                    cfg.vllm_gpu_memory_utilization, max_len, cfg.logprobs_mode,
                                    cfg.vllm_fp32_head, cfg.enforce_eager, cfg.seed, cfg.vllm_sleep)
        t_vllm = time.time() - t
        t = time.time()
        self.policy = Policy(self.model_path, lr=cfg.lr, weight_decay=cfg.weight_decay,
                             gradient_checkpointing=cfg.gradient_checkpointing, fp32_head=cfg.train_fp32_head,
                             head_chunk=cfg.head_chunk_tokens)
        self._log(dict(event="init", vllm_init_s=t_vllm, policy_init_s=time.time() - t,
                       train_prompts=len(self.train_rows), eval_prompts=len(self.eval_rows),
                       rollout_model=rollout_path, gpu=torch.cuda.get_device_name()))

    # ---------------------------------------------------------------- utils
    def _tokenize(self, rows):
        out = []
        for r in rows:
            ids = self.tok(data_mod.render_prompt(self.tok, r), add_special_tokens=False)["input_ids"]
            if len(ids) <= self.cfg.max_prompt_tokens:
                r["prompt_ids"] = ids
                out.append(r)
        return out

    def _log(self, rec: Dict):
        rec["time"] = time.time()
        self.metrics_f.write(json.dumps(rec) + "\n")
        self.metrics_f.flush()
        keys = [k for k in ("step", "reward_mean", "eval_acc", "kl_k3", "prob_absdiff_mean", "step_s") if k in rec]
        print(f"[{self.name}] " + (rec.get("event", "") + " ") + " ".join(
            f"{k}={rec[k]:.4g}" if isinstance(rec[k], float) else f"{k}={rec[k]}" for k in keys), flush=True)

    def _next_prompts(self, n):
        if self.cursor + n > len(self.train_rows):
            random.shuffle(self.train_rows)
            self.cursor = 0
        rows = self.train_rows[self.cursor:self.cursor + n]
        self.cursor += n
        return rows

    def _microbatches(self, seqs):
        return self.policy.make_microbatches([len(p) + len(r) for p, r in seqs], self.cfg.micro_batch_tokens)

    # --------------------------------------------------------------- phases
    def sync_weights(self):
        P = self.prof
        with P.phase("weight_sync"):
            if self.engine.asleep:
                with P.phase("vllm_wake_weights"):
                    self.engine.wake_weights()
            with P.phase("gather"):
                items = list(self.policy.export_weights(torch.bfloat16))
            if self.cfg.rollout_quant == "int8":
                with P.phase("quantize"):
                    items = list(quant.int8_items(items))
            with P.phase("vllm_load"):
                self.engine.load_weights(items)
            del items
            if self.engine.asleep:
                with P.phase("vllm_wake_kv_cache"):
                    self.engine.wake_kv_cache()

    def evaluate(self, step: int):
        cfg = self.cfg
        if self.weights_stale or self.engine.asleep:  # engine has pre-update weights, or is asleep
            self.sync_weights()
        with self.prof.phase("eval"):
            res = self.engine.generate([r["prompt_ids"] for r in self.eval_rows], n=1,
                                       max_tokens=cfg.eval_max_new_tokens or cfg.max_new_tokens,
                                       temperature=0.0, top_p=1.0, want_logprobs=False)
            acc = [score(r, g[0].text) for r, g in zip(self.eval_rows, res)]
            lens = [len(g[0].response_ids) for g in res]
        with open(os.path.join(self.out, "samples", f"eval_{step:05d}.jsonl"), "w") as f:
            for r, g, a in zip(self.eval_rows, res, acc):
                f.write(json.dumps(dict(id=r["id"], answer=r["answer"], reward=a, response=g[0].text)) + "\n")
        rec = dict(event="eval", step=step, eval_acc=float(np.mean(acc)), eval_n=len(acc),
                   eval_resp_len=float(np.mean(lens)), eval_dataset=cfg.eval_dataset or cfg.dataset)
        return rec

    @torch.no_grad()
    def old_logprobs(self, seqs):
        lp_out, ent_out = [None] * len(seqs), [None] * len(seqs)
        for mb in self._microbatches(seqs):
            s = [seqs[i] for i in mb]
            lp, _, ent = self.policy.forward_logprobs(s, self.cfg.temperature, entropy=True)
            for i, a, e in zip(mb, self.policy.response_view(lp, s), self.policy.response_view(ent, s)):
                lp_out[i], ent_out[i] = a.clone(), e.clone()
        return lp_out, ent_out

    def train_on(self, seqs, adv, lp_old, lp_roll):
        cfg, P = self.cfg, self.prof
        stats_acc: Dict[str, List[float]] = {}
        grad_norms = []
        idx = list(range(len(seqs)))
        for _ in range(cfg.ppo_epochs):
            random.shuffle(idx)
            n_mb = max(1, min(cfg.ppo_minibatches, len(idx)))
            for k in range(n_mb):
                mbatch = idx[k::n_mb]
                denom = float(sum(len(seqs[i][1]) for i in mbatch))
                sub = [seqs[i] for i in mbatch]
                for micro in self._microbatches(sub):
                    ids = [mbatch[j] for j in micro]
                    s = [seqs[i] for i in ids]
                    with P.phase("forward"):
                        logp, rmask, _ = self.policy.forward_logprobs(s, cfg.temperature)
                        T = logp.shape[1]
                        old = _pad2d([lp_old[i] for i in ids], s, T, logp.device)
                        roll = _pad2d([lp_roll[i] for i in ids], s, T, logp.device)
                        a = torch.tensor([adv[i] for i in ids], device=logp.device, dtype=torch.float32)
                        loss, st = policy_loss(logp, old, roll, a, rmask, cfg.correction, cfg.clip_low,
                                               cfg.clip_high, cfg.tis_cap, denom=denom)
                    with P.phase("backward"):
                        loss.backward()
                    for kk, v in st.items():
                        stats_acc.setdefault(kk, []).append(v)
                with P.phase("optimizer"):
                    grad_norms.append(self.policy.clip_and_step(cfg.grad_clip))
        out = {k: float(np.mean(v)) for k, v in stats_acc.items()}
        out["grad_norm"] = float(np.mean(grad_norms)) if grad_norms else 0.0
        out["optimizer_steps"] = len(grad_norms)
        return out

    # ----------------------------------------------------------------- step
    def step(self, step: int):
        cfg, P = self.cfg, self.prof
        rec: Dict = dict(event="step", step=step)
        t_step = time.time()
        self.sync_weights()
        self.weights_stale = False

        with P.phase("rollout"):
            rows = self._next_prompts(cfg.prompts_per_step)
            groups = self.engine.generate([r["prompt_ids"] for r in rows], n=cfg.samples_per_prompt,
                                          max_tokens=cfg.max_new_tokens, temperature=cfg.temperature,
                                          top_p=cfg.top_p, seed=cfg.seed * 100003 + step)
        if cfg.vllm_sleep:
            with P.phase("vllm_sleep"):
                self.engine.sleep()
        samples = [s for g in groups for s in g]
        gen_tokens = sum(len(s.response_ids) for s in samples)

        with P.phase("reward"):
            rewards = torch.tensor([score(r, s.text) for r, g in zip(rows, groups) for s in g], dtype=torch.float32)

        with open(os.path.join(self.out, "samples", f"step_{step:05d}.jsonl"), "w") as f:
            for j, s in enumerate(samples[:cfg.save_samples]):
                f.write(json.dumps(dict(id=rows[j // cfg.samples_per_prompt]["id"], reward=float(rewards[j]),
                                        finish=s.finish_reason, response=s.text)) + "\n")

        with P.phase("advantage"):
            adv = group_advantages(rewards, cfg.samples_per_prompt, cfg.scale_adv_by_std)
            g = rewards.view(-1, cfg.samples_per_prompt)
            keep_group = (g.max(1).values != g.min(1).values) if cfg.drop_uniform_groups else torch.ones(len(g), dtype=torch.bool)
            keep = keep_group.repeat_interleave(cfg.samples_per_prompt) & torch.tensor([len(s.response_ids) > 0 for s in samples])
            seqs = [(s.prompt_ids, s.response_ids) for s in samples]
            lp_roll = [torch.tensor(s.logprobs, dtype=torch.float32) for s in samples]

        with P.phase("old_logprob"):
            nonempty = [i for i, s in enumerate(seqs) if len(s[1]) > 0]
            lp_o, ent_o = self.old_logprobs([seqs[i] for i in nonempty])
            lp_old, ent = [torch.zeros(0)] * len(seqs), [torch.zeros(0)] * len(seqs)
            for j, i in enumerate(nonempty):
                lp_old[i], ent[i] = lp_o[j], ent_o[j]

        with P.phase("mismatch"):
            T = max(len(s[1]) for s in seqs) if seqs else 1
            dev = self.policy.device
            mask = torch.zeros(len(seqs), T, device=dev)
            R = torch.zeros(len(seqs), T, device=dev); O = torch.zeros(len(seqs), T, device=dev)
            for b in nonempty:
                L = len(seqs[b][1])
                mask[b, :L] = 1; R[b, :L] = lp_roll[b].to(dev); O[b, :L] = lp_old[b].to(dev)
            mm = mismatch_stats(R, O, mask, cfg.tis_cap)
            if cfg.save_hist_every and step % cfg.save_hist_every == 0:
                np.save(os.path.join(self.out, "hist", f"step_{step:05d}.npy"), prob_hist2d(R, O, mask))
            ent_mean = float(torch.cat([e for e in ent if e.numel()]).mean()) if nonempty else 0.0

        train_idx = [i for i in range(len(seqs)) if keep[i]]
        if train_idx:
            with P.phase("train"):
                tstats = self.train_on([seqs[i] for i in train_idx], [float(adv[i]) for i in train_idx],
                                       [lp_old[i] for i in train_idx], [lp_roll[i] for i in train_idx])
            self.weights_stale = True
        else:
            tstats = dict(skipped=1)

        rec.update(mm)
        rec.update(tstats)
        walls = P.wall_times()
        rec.update(
            reward_mean=float(rewards.mean()), solve_all_frac=float((g.min(1).values == 1).float().mean()),
            solve_none_frac=float((g.max(1).values == 0).float().mean()),
            trained_seqs=len(train_idx), samples=len(samples), gen_tokens=gen_tokens,
            resp_len_mean=gen_tokens / max(1, len(samples)),
            truncated_frac=float(np.mean([s.finish_reason == "length" for s in samples])),
            entropy_mean=ent_mean,
            rollout_tok_per_s=gen_tokens / max(1e-9, walls.get("rollout", 0.0)),
            phase_wall_s={k: v for k, v in walls.items()},
            step_s=time.time() - t_step,
            mem_max_reserved_GB=torch.cuda.max_memory_reserved() / 1e9,
        )
        return rec

    # ------------------------------------------------------------------ run
    def _is_profiled(self, step: int) -> bool:
        c = self.cfg
        return step in c.profile_steps or (c.profile_every > 0 and step % c.profile_every == 0)

    @staticmethod
    def compact_profile(summ: dict) -> dict:
        """Per-phase numbers small enough to keep for every step in metrics.jsonl."""
        out = {}
        for name, p in summ["phases"].items():
            if name == "(outside phases)" and not p.get("kernels"):
                continue
            out[name] = dict(wall_ms=round(p.get("wall_ms", 0.0), 3), gpu_busy_ms=round(p.get("gpu_busy_ms", 0.0), 3),
                             kernels=p.get("kernels", 0), kernel_ms=round(p.get("kernel_ms", 0.0), 3),
                             mem_peak_GB=round(p.get("mem_peak_GB", 0.0), 3),
                             api_calls=p.get("api_calls", 0),
                             memcpy_MB={k: round(v["MB"], 3) for k, v in (p.get("memcpy") or {}).items()},
                             cat_ms={k: round(v["ms"], 3) for k, v in (p.get("by_category") or {}).items()})
        return out

    def _profiled(self, fn, tag: str, active: bool, trace: bool = False):
        P = self.prof
        P.reset()
        if active:
            P.start()
        try:
            rec = fn()
        finally:
            if active:
                P.stop()
        if active:
            summ = P.summarize()
            summ["tag"] = tag
            with open(os.path.join(self.out, "profile", f"{tag}.json"), "w") as f:
                json.dump(summ, f, indent=1)
            if isinstance(rec, dict):
                rec["cupti"] = self.compact_profile(summ)
                rec["cupti_totals"] = dict(kernels=summ["total_kernels"], copies=summ["total_copies"],
                                           api_calls=summ["total_api_calls"])
            if self.cfg.chrome_trace and trace:
                P.write_chrome_trace(os.path.join(self.out, "profile", f"trace_{tag}.json.gz"))
        return rec

    def run(self):
        cfg = self.cfg
        best, since_best, stop_reason = -1.0, 0, "max_steps"
        if cfg.eval_every:
            ev = self._profiled(lambda: self.evaluate(0), "eval_00000", cfg.profile_eval, trace=0 in cfg.trace_steps)
            self._log(ev)
            best = ev["eval_acc"]
        for step in range(1, cfg.steps + 1):
            prof = self._is_profiled(step)
            rec = self._profiled(lambda: self.step(step), f"step_{step:05d}", prof, trace=step in cfg.trace_steps)
            rec["profiled"] = prof
            self._log(rec)
            if cfg.eval_every and (step % cfg.eval_every == 0 or step == cfg.steps):
                ev = self._profiled(lambda: self.evaluate(step), f"eval_{step:05d}", cfg.profile_eval and prof)
                self._log(ev)
                if ev["eval_acc"] > best + cfg.converge_min_delta:
                    best, since_best = ev["eval_acc"], 0
                else:
                    since_best += 1
                if cfg.converge_patience and step >= cfg.min_steps and since_best >= cfg.converge_patience:
                    stop_reason = "converged"
                    break
        self._log(dict(event="end", step=step if cfg.steps else 0, stop_reason=stop_reason, best_eval_acc=best))
        self.metrics_f.close()
