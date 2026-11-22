"""Run configuration: a dataclass filled from a YAML file and `--set key=value` overrides."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class Config:
    # --- what to train
    model: str = "qwen2.5-0.5b-instruct"        # registry name, HF repo id or local path (an SFT model)
    dataset: str = "gsm8k"                      # see rlprof.registry.DATASETS
    eval_dataset: Optional[str] = None          # defaults to `dataset` (its test split)
    run_name: Optional[str] = None              # defaults to a name built from the settings below

    # --- rollout engine (vLLM)
    rollout_quant: str = "bf16"                 # bf16 | fp8 | int8
    logprobs_mode: str = "processed_logprobs"   # processed_logprobs (true sampling probs) | raw_logprobs
    vllm_fp32_head: bool = False                # run vLLM's lm_head in fp32 (paper section 3.1)
    vllm_gpu_memory_utilization: float = 0.30   # fraction of GPU memory vLLM may use (weights + KV cache)
    vllm_sleep: bool = False                    # free vLLM weights + KV cache during training (sleep level 2)
    enforce_eager: bool = False                 # disable CUDA graphs / torch.compile in vLLM
    temperature: float = 1.0
    top_p: float = 1.0
    max_prompt_tokens: int = 512                # longer prompts are skipped
    max_new_tokens: int = 1024

    # --- algorithm
    correction: str = "tis"                     # none | tis | ppo_is | vanilla_is
    tis_cap: float = 2.0                        # C in min(pi_fsdp / pi_vllm, C)
    clip_low: float = 0.2
    clip_high: float = 0.28                     # DAPO clip-higher; set 0.2 for symmetric PPO
    prompts_per_step: int = 32
    samples_per_prompt: int = 8                 # group size for group-normalized advantages
    scale_adv_by_std: bool = True
    drop_uniform_groups: bool = True            # skip groups whose samples all got the same reward
    ppo_minibatches: int = 2                    # optimizer steps per rollout batch (>1 makes PPO clip matter)
    ppo_epochs: int = 1
    steps: int = 100                            # maximum number of steps
    # --- stop at convergence: after min_steps, stop when the best eval accuracy has not improved by
    # converge_min_delta for converge_patience consecutive evals (needs eval_every > 0; 0 disables)
    converge_patience: int = 0
    converge_min_delta: float = 0.005
    min_steps: int = 0

    # --- trainer (FSDP2)
    lr: float = 1e-6
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    micro_batch_tokens: int = 16384             # padded tokens per forward/backward
    gradient_checkpointing: bool = True
    train_fp32_head: bool = False               # compute the trainer's lm_head in fp32
    head_chunk_tokens: int = 2048               # tokens per lm_head/log-softmax chunk (memory ~ chunk x vocab x 4 B)

    # --- evaluation
    eval_every: int = 25                        # 0 disables; eval also runs before step 1 and after the last
    eval_limit: int = 500                       # number of test problems (0 = all)
    eval_max_new_tokens: Optional[int] = None   # defaults to max_new_tokens

    # --- CUPTI profiling
    profile_every: int = 1                      # CUPTI-profile every Nth step (1 = every step, 0 = only profile_steps)
    profile_steps: List[int] = field(default_factory=list)   # extra 1-based steps to profile
    profile_eval: bool = True                   # profile evaluation passes too
    trace_steps: List[int] = field(default_factory=lambda: [2])   # steps that also get a Chrome trace
    chrome_trace: bool = True                   # write profile/trace_*.json.gz for trace_steps
    profile_api: bool = True                    # also record CUDA runtime/driver API calls

    seed: int = 0
    save_samples: int = 16                      # rollouts per step written to samples/step_XXXXX.jsonl
    save_hist_every: int = 1                    # write the (p_vllm, p_fsdp) histogram every N steps

    def default_run_name(self) -> str:
        m = self.model.split("/")[-1]
        extra = "".join([f"_raw" if self.logprobs_mode == "raw_logprobs" else "",
                         "_vhead32" if self.vllm_fp32_head else "", "_thead32" if self.train_fp32_head else ""])
        return f"{self.dataset}_{m}_{self.rollout_quant}_{self.correction}{extra}_s{self.seed}"

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)


def _coerce(cur, val: str):
    if isinstance(cur, bool):
        return val.lower() in ("1", "true", "yes", "on")
    if isinstance(cur, int) and not isinstance(cur, bool):
        return int(val)
    if isinstance(cur, float):
        return float(val)
    if isinstance(cur, list):
        return [int(x) for x in val.split(",") if x.strip()] if val.strip() else []
    if val.lower() in ("none", "null"):
        return None
    try:
        return yaml.safe_load(val)
    except Exception:
        return val


def load_config(path: Optional[str], overrides: List[str]) -> Config:
    cfg = Config()
    names = {f.name for f in dataclasses.fields(Config)}
    if path:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        bad = set(data) - names
        if bad:
            raise KeyError(f"unknown config keys in {path}: {sorted(bad)}")
        for k, v in data.items():
            setattr(cfg, k, v)
    for ov in overrides:
        k, _, v = ov.partition("=")
        if k not in names:
            raise KeyError(f"unknown config key {k!r}")
        setattr(cfg, k, _coerce(getattr(cfg, k), v))
    return cfg
