"""Step 6: RL fine-tuning with CUPTI profiling of every phase.

    python scripts/06_train.py --config configs/gsm8k_qwen0.5b_bf16_tis.yaml
    python scripts/06_train.py --config configs/gsm8k_qwen0.5b_bf16_tis.yaml --set steps=5 correction=none
    CUDA_VISIBLE_DEVICES=1 python scripts/06_train.py --config ...          # pick a GPU

Outputs in $RLPROF_HOME/runs/<run_name>/:
    config.json             resolved settings
    metrics.jsonl           one record per step / eval (rewards, mismatch stats, losses, phase wall times)
    profile/step_XXXXX.json CUPTI summary per phase for the profiled steps (profile_steps)
    profile/trace_*.json.gz Chrome/Perfetto trace of the profiled steps
    hist/step_XXXXX.npy     50x50 histogram of (p_vllm, p_fsdp) for sampled tokens
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
from rlprof import paths  # noqa: E402,F401
from rlprof.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--print-config", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    print(cfg.to_json())
    if args.print_config:
        return 0
    from rlprof.train import Trainer
    t = Trainer(cfg)
    t.run()
    print(f"done: {t.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
