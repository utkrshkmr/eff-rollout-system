"""Step 4 (only for INT8 rollouts): write an INT8 W8A8 copy of the SFT model for vLLM.

    python scripts/04_make_int8_checkpoint.py --model qwen2.5-0.5b-instruct

BF16 and FP8 rollouts need no extra checkpoint (vLLM quantizes FP8 online). The INT8 copy only
seeds the vLLM engine; at every weight sync the trainer re-quantizes its current weights the same
way (per-channel round-to-nearest) and loads them, so rollouts always use the latest policy.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlprof import paths, quant, registry  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
    ap.add_argument("--check", action="store_true", default=True,
                    help="load the result in vLLM and generate a few tokens (default on)")
    ap.add_argument("--no-check", dest="check", action="store_false")
    args = ap.parse_args()
    repo = registry.resolve_model(args.model)
    out = quant.int8_checkpoint_dir(paths.MODELS, repo)
    info = quant.write_int8_checkpoint(repo, out)
    print(f"wrote {out}: {info}")
    if args.check:
        from vllm import LLM, SamplingParams
        llm = LLM(out, dtype="bfloat16", gpu_memory_utilization=0.25, max_model_len=1024, enforce_eager=True)
        txt = llm.generate(["The capital of France is"], SamplingParams(max_tokens=8, temperature=0))[0].outputs[0].text
        print(f"vLLM INT8 check: 'The capital of France is' -> {txt!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
