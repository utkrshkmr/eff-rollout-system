# eff-rollout-system: CUPTI-profiled RL fine-tuning

This repo runs RL fine-tuning on one GPU and profiles every step with
[CUPTI](https://docs.nvidia.com/cupti/). It also measures the *rollout-training mismatch*
described in Yao, Liu, Zhang, Dong, Shang and Gao, "On the Rollout-Training Mismatch in Modern
RL Systems" (NeurIPS 2025 Efficient Reasoning workshop).

As in production RL frameworks, there are two engines with the same weights:

* **vLLM** generates the rollouts and reports log π<sub>vllm</sub>(token).
* **PyTorch FSDP2** re-scores those rollouts to get log π<sub>fsdp</sub>(token), then computes the gradients.

Both run in one process, so a single CUPTI session records every kernel, memcpy, memset and
CUDA runtime/driver call. Each record is assigned to the training phase that launched it. Training
starts from an instruction-tuned (SFT) checkpoint and continues until evaluation accuracy stops
improving.

The results page is the static site in [`docs/`](docs/index.html), served by GitHub Pages (see
[Publishing the results page](#publishing-the-results-page)).

## What one training step does

| phase | what happens | paper |
|---|---|---|
| `weight_sync/gather` | FSDP2 parameters are unsharded and cast to bf16 | hybrid engine |
| `weight_sync/quantize` | INT8 rollouts only: per-channel int8 weights and scales | §4.2 quantized rollouts |
| `weight_sync/vllm_load` | vLLM layer-wise `reload_weights` (re-quantizes FP8 on the fly) | |
| `weight_sync/vllm_wake_*`, `vllm_sleep` | with `vllm_sleep: true`, vLLM frees weights and KV cache during training | |
| `rollout` | vLLM samples `samples_per_prompt` responses per prompt and returns the sampled-token logprobs (`processed_logprobs` by default) | π<sub>vllm</sub>(θ<sub>old</sub>) |
| `reward` | math-verify (math) or an exact equation check (Countdown); runs on the CPU | |
| `advantage` | group-normalized advantages; groups where every sample got the same reward are dropped | |
| `old_logprob` | FSDP forward pass without gradients over every rollout: log π<sub>fsdp</sub>(θ<sub>old</sub>) and entropy | π<sub>fsdp</sub>(θ<sub>old</sub>) |
| `mismatch` | KL(π<sub>vllm</sub>‖π<sub>fsdp</sub>) (k1, k3), \|p<sub>vllm</sub>−p<sub>fsdp</sub>\|, ratio extremes, a 50×50 (p<sub>vllm</sub>, p<sub>fsdp</sub>) histogram | Fig. 1, 3, 4 |
| `train/forward`, `train/backward` | PPO-clip loss over `ppo_minibatches` mini-batches, with the selected `correction` | §3.2, §3.3, §4.3 |
| `train/optimizer` | gradient clipping, AdamW step | |
| `eval` | greedy decoding on the test split every `eval_every` steps | AIME/GSM8K eval |

### Mismatch corrections

These are the `correction` setting, all from the paper. Notation:
w = π<sub>fsdp</sub>(θ<sub>old</sub>) / π<sub>vllm</sub>(θ<sub>old</sub>) and r = π<sub>fsdp</sub>(θ) / π<sub>fsdp</sub>(θ<sub>old</sub>).

* `none`: plain PPO-clip on r, ignoring that the samples came from π<sub>vllm</sub>.
* `tis`: truncated importance sampling, min(w, C) × PPO-clip(r). C is `tis_cap`, default 2.
* `ppo_is`: π<sub>vllm</sub>(θ<sub>old</sub>) replaces π<sub>fsdp</sub>(θ<sub>old</sub>) inside the PPO ratio and its clip.
* `vanilla_is`: w × PPO-clip(r), with no truncation.

The two fixes the paper tried in §3.1 are also switches:

* `logprobs_mode: processed_logprobs`, the true sampling probabilities. `raw_logprobs` gives pre-temperature logprobs instead.
* `vllm_fp32_head: true` runs vLLM's lm_head in FP32. `train_fp32_head: true` does the same for the trainer.

### Rollout precision

The `rollout_quant` setting:

* `bf16`
* `fp8`: vLLM online FP8 quantization. On A100 this runs as weight-only Marlin kernels; on H100 and newer it uses FP8 GEMMs.
* `int8`: W8A8 with per-channel symmetric int8 weights, dynamic per-token int8 activations, and vLLM's Cutlass INT8 GEMMs.

vLLM 0.34 has no online INT8 for dense layers. Instead, step 4 writes a compressed-tensors INT8 copy
of the SFT model once, and every weight sync re-quantizes the trainer's current weights the same way.

## How the CUPTI profiling works

The profiler is [`rlprof/cupti_profiler.py`](rlprof/cupti_profiler.py). It uses NVIDIA's
`cupti-python` Activity API, with no nsys and no torch.profiler. It records
`CONCURRENT_KERNEL`, `MEMCPY`, `MEMSET`, `RUNTIME`, `DRIVER` and `EXTERNAL_CORRELATION` activity.

Phase attribution:

* **External correlation.** `with prof.phase("rollout"):` pushes an external correlation id. CUPTI
  links every API call made inside the phase to that id, and every kernel or copy to the API call
  that launched it. For vLLM's CUDA graphs, that call is the `cudaGraphLaunch`. This is exact even
  though kernels run asynchronously.
* **Time-window fallback.** Work launched from another thread has no correlation id. The main case
  is the autograd engine's backward thread. Such work is assigned by GPU start time to the innermost
  phase window. Every phase is bracketed by `torch.cuda.synchronize()`, so these windows are exact
  for this sequential, single-GPU pipeline. The summary reports how many kernels were attributed
  each way.

For every phase, the profile records:

* host wall time
* GPU busy time: the union of kernel and copy intervals
* kernel count and time, both by kernel and by category (GEMM, quantized GEMM, attention, quantize, optimizer, sampling, inductor/Triton, ...)
* how many kernels came from CUDA graphs
* memcpy bytes and bandwidth by direction
* memset bytes
* CUDA API call counts and time
* peak allocated memory

With `profile_every: 1` (the default), every training step is profiled:

* `profile/step_XXXXX.json` holds the full summary of each step.
* A compact per-phase copy goes into each step's line in `metrics.jsonl`.
* Steps listed in `trace_steps` also get a Chrome/Perfetto trace, `profile/trace_step_XXXXX.json.gz`; open it at https://ui.perfetto.dev.

Do not run nsys or ncu at the same time: CUPTI allows only one subscriber per process.

## Setup, step by step

Tested on an NVIDIA A100 80GB PCIe (driver 595.91), Python 3.12, vLLM 0.34.0, torch 2.14.0 (CUDA
13.0), transformers 5.22.2 and cupti-python 13.0.1. The setup needs a driver for CUDA 13
(>= 580) and about 10 GB of disk for the default model and data.

Everything large goes under `RLPROF_HOME` (default `./workdir`): the Hugging Face cache,
prepared data, derived checkpoints, runs, and the vLLM/Triton/FlashInfer caches.

```bash
export RLPROF_HOME=/path/to/big/disk/rlprof      # optional
```

**0. Environment.** Creates `.venv` from the pinned `requirements.lock.txt` with [uv](https://docs.astral.sh/uv/).

```bash
bash scripts/00_create_env.sh
source .venv/bin/activate
```

**1. Check the GPU, the libraries and that CUPTI records kernels in this process.**

```bash
python scripts/01_check_env.py
```

**2. Download the SFT starting checkpoints.** `--list` shows the tested ones.

```bash
python scripts/02_download_models.py --models qwen2.5-0.5b-instruct qwen2.5-1.5b-instruct
```

**3. Download and normalize the RL datasets.** A self-check asserts that the gold answers score reward 1.

```bash
python scripts/03_prepare_data.py --datasets gsm8k math dapo_math deepscaler countdown
```

**4. INT8 rollouts only: write the INT8 W8A8 copy of the model.** The script then loads it in vLLM to check it.

```bash
python scripts/04_make_int8_checkpoint.py --model qwen2.5-0.5b-instruct
```

**5. Optional: baseline accuracy and step-0 mismatch of the untrained SFT model.**

```bash
python scripts/05_eval.py --model qwen2.5-0.5b-instruct --dataset gsm8k --rollout-quant int8
```

**6. Train until convergence, with CUPTI on every step.** Choose the GPU with `CUDA_VISIBLE_DEVICES`.

```bash
python scripts/06_train.py --config configs/smoke.yaml        # ~3 minutes, checks everything
python scripts/06_train.py --config configs/gsm8k_qwen2.5-0.5b.yaml --set rollout_quant=bf16 correction=tis
python scripts/06_train.py --config configs/gsm8k_qwen2.5-0.5b.yaml --set rollout_quant=int8 correction=none
```

**7. Markdown report with plots for any set of runs.**

```bash
python scripts/07_report.py --all --out report/
```

**8. Rebuild the GitHub Pages data in `docs/` from the finished runs.**

```bash
python scripts/08_build_site.py
```

Every setting is a field of [`rlprof/config.py`](rlprof/config.py). Set it in a YAML file or with `--set key=value`.

## Models and datasets

| model (`--models`, `model:`) | HF repo |
|---|---|
| `qwen2.5-0.5b-instruct` (default) | Qwen/Qwen2.5-0.5B-Instruct |
| `qwen2.5-1.5b-instruct` | Qwen/Qwen2.5-1.5B-Instruct |
| `qwen2.5-math-1.5b-instruct` | Qwen/Qwen2.5-Math-1.5B-Instruct |
| `qwen3-0.6b` | Qwen/Qwen3-0.6B (thinking turned off in the prompt) |

Any other Hugging Face causal LM with a chat template should work if both vLLM and transformers
load it; pass its repo id or a local path.

| dataset | train | test | reward |
|---|---|---|---|
| `gsm8k` | openai/gsm8k train (7,473) | openai/gsm8k test (1,319) | final number / `\boxed{}` |
| `math` | EleutherAI/hendrycks_math, 7 subjects (7,496) | HuggingFaceH4/MATH-500 | math-verify |
| `dapo_math` | BytedTsinghua-SIA/DAPO-Math-17k, deduplicated (17,255) | HuggingFaceH4/aime_2024 (30) | math-verify |
| `deepscaler` | agentica-org/DeepScaleR-Preview-Dataset (40,315) | HuggingFaceH4/aime_2024 | math-verify |
| `countdown` | Jiayi-Pan/Countdown-Tasks-3to4 (489,852) | last 512 rows | equation in `<answer>` uses each number once and hits the target |

Configs: `configs/gsm8k_qwen2.5-0.5b.yaml` (the paper's §4.2 setting), `math_qwen2.5-1.5b.yaml`,
`dapo_qwen2.5-math-1.5b.yaml` (DAPO-style, AIME24 eval), `countdown_qwen2.5-1.5b.yaml`,
`deepscaler_qwen3-0.6b.yaml`, and `smoke.yaml`.

## Outputs

Each run writes to `$RLPROF_HOME/runs/<run_name>/`:

```
config.json            resolved settings
metrics.jsonl          per step: rewards, mismatch statistics, PPO/IS statistics, phase wall times and
                       a compact CUPTI summary per phase; per eval: accuracy; last line: stop reason
profile/step_*.json    full CUPTI summary of each profiled step (top kernels, API calls, memcpy, ...)
profile/eval_*.json    the same for evaluation passes
profile/trace_*.json.gz  Chrome/Perfetto traces for trace_steps
hist/step_*.npy        (p_vllm, p_fsdp) histogram of the sampled tokens
samples/*.jsonl        a few rollouts and all eval generations, with rewards
```

## Design notes and limits

* **One GPU, one process.** FSDP2 is applied per decoder layer and at the root, as in multi-GPU
  trainers. With one rank, nothing is sharded and there is no NCCL traffic, but the unshard,
  reshard, mixed-precision casts and gradient reduction still run, and the profile shows them.
* **vLLM settings.** vLLM runs in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) with its
  extensible KV cache turned off. That cache caps PyTorch's allocator for the whole process, which
  would starve the trainer. The FlashInfer sampler is also turned off, because it would
  JIT-compile with whatever `nvcc` is on `PATH`.
* **Advantages.** These are group-normalized (GRPO/DAPO style) instead of a PPO critic. The
  importance-sampling corrections are exactly those of the paper.
* **INT8 quantization.** It is round-to-nearest with no calibration (no SmoothQuant or GPTQ), so
  it adds more mismatch than a calibrated checkpoint would.
* **Memory.** The trainer computes the lm_head and log-softmax in token chunks (`head_chunk_tokens`)
  with activation checkpointing, so the full [tokens × vocab] logits never exist at once.
* **CUPTI overhead.** A Python callback runs for each activity record, which adds a few hundred
  milliseconds per step for the 0.5B model. Set `profile_every: 0` to time steps without it.
* **Shared GPUs.** Timings depend on other jobs running on the same GPU; check `nvidia-smi` before
  comparing numbers.

## Publishing the results page

On GitHub, open the repository's Settings → Pages, choose "Deploy from a branch", then branch `main`
and folder `/docs`. To update the page, rerun step 8 and push.

## Tests

```bash
pytest -q tests
```

The tests cover the losses, rewards, mismatch metrics, INT8 quantization, micro-batching,
categorization, config parsing and, when a GPU is present, CUPTI phase attribution.
