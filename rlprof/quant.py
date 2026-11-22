"""Quantized rollout support.

bf16  plain vLLM; the trainer's bf16 weights are loaded as is.
fp8   vLLM online FP8 (`quantization="fp8_per_tensor"`): vLLM quantizes the bf16 weights we send at
      every weight sync. On GPUs without FP8 tensor cores (A100) vLLM runs these as weight-only FP8
      (Marlin W8A16 kernels); on Hopper/Blackwell it uses FP8 GEMMs.
int8  INT8 W8A8 (per-channel symmetric int8 weights, dynamic per-token int8 activations) in the
      compressed-tensors format, run by vLLM's Cutlass INT8 kernels. vLLM 0.34 has no online INT8
      for dense layers, so scripts/04_make_int8_checkpoint.py writes an INT8 copy of the SFT model
      once, and at every weight sync the trainer quantizes its weights the same way (round to
      nearest, scale = absmax/127 per output channel) and sends int8 weights + scales.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Iterator, Tuple

import torch

QUANT_LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

INT8_W8A8_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "int-quantized",
    "quantization_status": "compressed",
    "ignore": ["lm_head"],
    "config_groups": {"group_0": {
        "targets": ["Linear"],
        "weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel",
                    "dynamic": False, "group_size": None, "observer": "minmax"},
        "input_activations": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "token",
                              "dynamic": True, "group_size": None, "observer": "memoryless"},
    }},
}

VLLM_QUANT_ARG = {"bf16": None, "fp8": "fp8_per_tensor", "int8": None}


def is_quant_linear(name: str) -> bool:
    parts = name.split(".")
    return len(parts) >= 2 and parts[-1] == "weight" and parts[-2] in QUANT_LINEARS


def quantize_int8_per_channel(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    wf = w.float()
    scale = (wf.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-8)
    q = torch.round(wf / scale).clamp_(-128, 127).to(torch.int8)
    return q, scale.to(w.dtype)


def int8_items(items: Iterable[Tuple[str, torch.Tensor]]) -> Iterator[Tuple[str, torch.Tensor]]:
    """bf16 (name, tensor) -> compressed-tensors INT8 W8A8 checkpoint items."""
    for name, t in items:
        if is_quant_linear(name):
            q, s = quantize_int8_per_channel(t)
            yield name, q
            yield name + "_scale", s
        else:
            yield name, t


def int8_checkpoint_dir(models_dir: str, model_path: str) -> str:
    base = model_path.rstrip("/").replace("/", "--")
    return os.path.join(models_dir, f"{base}-w8a8-int8")


def write_int8_checkpoint(model_path: str, out_dir: str) -> Dict[str, int]:
    """Write an INT8 W8A8 (compressed-tensors) copy of an HF model for vLLM rollouts."""
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    cfg = AutoConfig.from_pretrained(model_path)
    os.makedirs(out_dir, exist_ok=True)
    tied = bool(getattr(cfg, "tie_word_embeddings", False))
    sd = {k: v.contiguous() for k, v in int8_items(model.state_dict().items())
          if not (tied and k == "lm_head.weight")}
    # shard at ~2 GB so large models do not need one giant file
    shards, cur, size, limit = [], {}, 0, 2 << 30
    for k, v in sd.items():
        nb = v.numel() * v.element_size()
        if cur and size + nb > limit:
            shards.append(cur); cur, size = {}, 0
        cur[k] = v; size += nb
    shards.append(cur)
    index = {}
    for i, sh in enumerate(shards):
        fn = "model.safetensors" if len(shards) == 1 else f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(sh, os.path.join(out_dir, fn), metadata={"format": "pt"})
        index.update({k: fn for k in sh})
    if len(shards) > 1:
        json.dump({"metadata": {}, "weight_map": index}, open(os.path.join(out_dir, "model.safetensors.index.json"), "w"))
    c = cfg.to_dict()
    c["quantization_config"] = INT8_W8A8_CONFIG
    json.dump(c, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    AutoTokenizer.from_pretrained(model_path).save_pretrained(out_dir)
    try:
        GenerationConfig.from_pretrained(model_path).save_pretrained(out_dir)
    except Exception:
        pass
    n_q = sum(1 for k in sd if k.endswith("_scale"))
    return dict(tensors=len(sd), int8_linears=n_q, shards=len(shards))
