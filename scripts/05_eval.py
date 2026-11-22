"""Step 5 (optional): baseline accuracy of the SFT model before RL, with the rollout engine profiled.

    python scripts/05_eval.py --model qwen2.5-0.5b-instruct --dataset gsm8k --limit 500
    python scripts/05_eval.py --dataset math --rollout-quant int8

Also measures the rollout-training mismatch of the untouched SFT model: samples responses with
vLLM, scores them with the FSDP trainer, and prints the statistics the paper plots (no training).
Writes $RLPROF_HOME/runs/eval_<model>_<dataset>_<quant>/{eval.json, profile_eval.json}.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
from rlprof import paths  # noqa: E402
from rlprof.config import Config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--rollout-quant", default="bf16", choices=["bf16", "fp8", "int8"])
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--mismatch-prompts", type=int, default=32, help="0 skips the mismatch measurement")
    ap.add_argument("--samples", type=int, default=4)
    args = ap.parse_args()

    from rlprof.train import Trainer
    cfg = Config(model=args.model, dataset=args.dataset, rollout_quant=args.rollout_quant,
                 eval_limit=args.limit, max_new_tokens=args.max_new_tokens, steps=0,
                 prompts_per_step=args.mismatch_prompts or 1, samples_per_prompt=args.samples,
                 run_name=f"eval_{args.model.split('/')[-1]}_{args.dataset}_{args.rollout_quant}",
                 profile_steps=[], chrome_trace=False)
    t = Trainer(cfg)
    ev = t._profiled(lambda: t.evaluate(0), "eval_00000", True)
    t._log(ev)
    out = dict(eval=ev)
    if args.mismatch_prompts:
        # one step's worth of rollouts + trainer scoring, without the update
        import torch
        from rlprof.metrics import mismatch_stats
        rows = t._next_prompts(args.mismatch_prompts)
        groups = t.engine.generate([r["prompt_ids"] for r in rows], n=args.samples,
                                   max_tokens=args.max_new_tokens, temperature=1.0, top_p=1.0, seed=1)
        samples = [s for g in groups for s in g if s.response_ids]
        seqs = [(s.prompt_ids, s.response_ids) for s in samples]
        lp_old, _ = t.old_logprobs(seqs)
        T = max(len(s[1]) for s in seqs)
        R, O, M = (torch.zeros(len(seqs), T) for _ in range(3))
        for b, s in enumerate(samples):
            L = len(s.response_ids)
            R[b, :L] = torch.tensor(s.logprobs); O[b, :L] = lp_old[b].cpu(); M[b, :L] = 1
        out["mismatch"] = mismatch_stats(R, O, M)
    with open(os.path.join(t.out, "eval.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
