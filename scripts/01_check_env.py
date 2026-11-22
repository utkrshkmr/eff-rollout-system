"""Step 1: check the GPU, the libraries and that CUPTI tracing works in this process.

    python scripts/01_check_env.py
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlprof import paths  # noqa: E402  (sets HF_HOME under RLPROF_HOME unless already set)


def main():
    ok = True
    import torch
    print(f"python      {sys.version.split()[0]}")
    print(f"torch       {torch.__version__}  CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("FAIL: torch sees no CUDA GPU"); return 1
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"gpu {i}       {p.name}  {p.total_memory / 2**30:.0f} GiB  sm_{p.major}{p.minor}")
    free, total = torch.cuda.mem_get_info()
    print(f"gpu mem     {free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB on the current device")
    if free / total < 0.6:
        print("WARN: less than 60% of GPU memory is free; lower vllm_gpu_memory_utilization or pick another GPU")
    cc = torch.cuda.get_device_capability()
    print(f"fp8 rollout {'FP8 tensor-core GEMMs' if cc >= (8, 9) else 'weight-only FP8 (Marlin W8A16) on this GPU'}")

    try:
        import vllm
        print(f"vllm        {vllm.__version__}")
    except Exception as e:
        print(f"FAIL: import vllm: {e!r}"); ok = False
    import transformers, datasets
    print(f"transformers {transformers.__version__}, datasets {datasets.__version__}")
    try:
        import math_verify  # noqa: F401
        print("math-verify ok")
    except Exception as e:
        print(f"FAIL: import math_verify: {e!r}"); ok = False

    from rlprof.cupti_profiler import CuptiProfiler, cupti_available
    good, msg = cupti_available()
    print(f"cupti       {msg}")
    if not good:
        return 1
    prof = CuptiProfiler()
    prof.start()
    with prof.phase("matmul"):
        x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        y = x @ x
    with prof.phase("copy"):
        y.cpu()
    prof.stop()
    s = prof.summarize()["phases"]
    mm, cp = s.get("matmul", {}), s.get("copy", {})
    print(f"cupti test  matmul phase: {mm.get('kernels', 0)} kernels, {mm.get('kernel_ms', 0):.3f} ms, "
          f"categories {list(mm.get('by_category', {}))}; copy phase memcpy {list(cp.get('memcpy', {}))}")
    if not mm.get("kernels") or "DTOH" not in cp.get("memcpy", {}):
        print("FAIL: CUPTI recorded no kernels / copies (is another profiler attached? nsys/ncu must not run)")
        ok = False
    if mm.get("kernels_attributed_by_correlation", 0) == 0:
        print("WARN: no kernel was attributed by external correlation; falling back to time windows")

    print(f"RLPROF_HOME {paths.HOME}\nHF_HOME     {os.environ['HF_HOME']}")
    os.makedirs(paths.HOME, exist_ok=True)
    du = shutil.disk_usage(paths.HOME)
    print(f"disk        {du.free / 2**30:.0f} GiB free at RLPROF_HOME (models+data need ~10 GB for the defaults)")
    print("OK" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
