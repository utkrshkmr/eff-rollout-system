"""Where everything is stored. Set RLPROF_HOME to put data, models and runs on a big disk.

    $RLPROF_HOME/hf        Hugging Face cache (models, datasets), unless HF_HOME is already set
    $RLPROF_HOME/data      prepared jsonl datasets (scripts/03_prepare_data.py)
    $RLPROF_HOME/models    derived checkpoints, e.g. INT8 W8A8 rollout copies (scripts/04_...)
    $RLPROF_HOME/runs      training runs, metrics and CUPTI profiles (scripts/06_train.py)
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.abspath(os.environ.get("RLPROF_HOME", os.path.join(REPO, "workdir")))
os.environ.setdefault("HF_HOME", os.path.join(HOME, "hf"))
# keep compiler/JIT caches next to the data instead of ~/.cache
CACHE = os.path.join(HOME, "cache")
os.environ.setdefault("VLLM_CACHE_ROOT", os.path.join(CACHE, "vllm"))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(CACHE, "triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(CACHE, "inductor"))
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", CACHE)
# vLLM engine core in this process (CUPTI sees its kernels; weights are passed as CUDA tensors)
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# vLLM's FlashInfer sampler is JIT-compiled with whatever nvcc is on PATH; the PyTorch sampler
# needs no compiler and keeps sampling numerics independent of the local CUDA toolkit
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
DATA = os.path.join(HOME, "data")
MODELS = os.path.join(HOME, "models")
RUNS = os.path.join(HOME, "runs")
CONFIGS = os.path.join(REPO, "configs")
