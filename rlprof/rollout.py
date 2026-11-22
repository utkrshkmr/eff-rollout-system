"""vLLM rollout engine running in the trainer's process (colocated on one GPU).

VLLM_ENABLE_V1_MULTIPROCESSING=0 keeps vLLM's engine core in this process, so (a) CUPTI sees its
kernels and (b) weights can be handed over as CUDA tensors without IPC.
"""
from __future__ import annotations

from . import paths  # noqa: F401  (env defaults for vLLM)

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch

from . import quant


@dataclass
class Sample:
    prompt_ids: List[int]
    response_ids: List[int]
    logprobs: List[float]        # log pi_vllm(token) of each sampled token
    text: str
    finish_reason: str


def _reload(worker, items):
    # runs inside the vLLM worker (same process): layer-wise reload, re-quantizing if needed
    worker.reload_weights(weights_iterator=iter(items))


class RolloutEngine:
    def __init__(self, model_path: str, tokenizer_path: str, quant_mode: str = "bf16",
                 gpu_memory_utilization: float = 0.3, max_model_len: int = 2048,
                 logprobs_mode: str = "processed_logprobs", fp32_head: bool = False,
                 enforce_eager: bool = False, seed: int = 0, sleep_mode: bool = False):
        from vllm import LLM
        if quant_mode not in quant.VLLM_QUANT_ARG:
            raise ValueError(f"rollout_quant must be one of {list(quant.VLLM_QUANT_ARG)}")
        kw = {}
        if quant.VLLM_QUANT_ARG[quant_mode]:
            kw["quantization"] = quant.VLLM_QUANT_ARG[quant_mode]
        if fp32_head:
            kw["hf_overrides"] = {"head_dtype": "float32"}
        self.quant_mode = quant_mode
        self.sleep_mode = sleep_mode
        self.asleep = False
        # enable_extensible_kv_cache=False: vLLM's extensible KV cache caps torch's allocator for
        # the whole process (torch.cuda.set_per_process_memory_fraction), which would starve the
        # colocated trainer. A plain KV cache sized by gpu_memory_utilization leaves the rest free.
        self.llm = LLM(model=model_path, tokenizer=tokenizer_path, dtype="bfloat16",
                       gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
                       logprobs_mode=logprobs_mode, enforce_eager=enforce_eager, seed=seed,
                       enable_prefix_caching=True, enable_extensible_kv_cache=False,
                       enable_sleep_mode=sleep_mode, **kw)

    def generate(self, prompt_ids: List[List[int]], n: int, max_tokens: int, temperature: float,
                 top_p: float, seed: Optional[int] = None, want_logprobs: bool = True) -> List[List[Sample]]:
        from vllm import SamplingParams
        sp = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                            logprobs=0 if want_logprobs else None, flat_logprobs=True, seed=seed)
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompt_ids], sp, use_tqdm=False)
        res = []
        for o in outs:
            group = []
            for c in o.outputs:
                ids = list(c.token_ids)
                lps: List[float] = []
                if want_logprobs:
                    fl = c.logprobs
                    if hasattr(fl, "start_indices"):
                        # logprobs=0 -> exactly the sampled token at each position
                        lps = [fl.logprobs[s] for s in fl.start_indices]
                    else:
                        lps = [d[t].logprob for d, t in zip(fl, ids)]
                    assert len(lps) == len(ids), (len(lps), len(ids))
                group.append(Sample(list(o.prompt_token_ids), ids, lps, c.text, c.finish_reason or ""))
            res.append(group)
        return res

    def sleep(self):
        """Level 2: drop weights and KV cache (the next sync reloads weights anyway)."""
        if self.sleep_mode and not self.asleep:
            self.llm.sleep(level=2)
            self.asleep = True

    def wake_weights(self):
        if self.asleep:
            self.llm.wake_up(tags=["weights"])

    def wake_kv_cache(self):
        if self.asleep:
            self.llm.wake_up(tags=["kv_cache", "scheduling"])
            self.asleep = False

    def load_weights(self, items: List[Tuple[str, torch.Tensor]]):
        """Load checkpoint-format tensors: bf16 for bf16/fp8 engines (vLLM re-quantizes fp8 itself),
        int8 + scales from `quant.int8_items` for the int8 engine."""
        self.llm.collective_rpc(_reload, args=(items,))
        self.llm.reset_prefix_cache()  # cached KV was computed with the old weights
