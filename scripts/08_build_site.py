"""Step 8: build the static results page (GitHub Pages) from finished runs.

    python scripts/08_build_site.py --all                  # every run under $RLPROF_HOME/runs
    python scripts/08_build_site.py <run_dir> <run_dir> ...

Writes docs/data/runs.json (per-step metrics + per-step CUPTI phase summaries), docs/data/
profile_<run>.json (full CUPTI summary of one representative step) and docs/img/probs_<run>.png.
docs/index.html is a fixed page that renders those files; serve docs/ with GitHub Pages
(Settings -> Pages -> Deploy from a branch -> main, /docs).
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlprof import paths  # noqa: E402

DOCS = os.path.join(paths.REPO, "docs")
STEP_KEYS = ["step", "reward_mean", "kl_k1", "kl_k3", "prob_absdiff_mean", "prob_absdiff_max", "frac_absdiff_gt_0p1",
             "ratio_max", "frac_ratio_gt_cap", "tis_truncated_frac", "is_weight_mean", "entropy_mean", "resp_len_mean",
             "truncated_frac", "grad_norm", "ppo_clip_frac", "step_s", "rollout_tok_per_s", "gen_tokens",
             "trained_seqs", "mem_max_reserved_GB"]


def load(d):
    recs = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl"))]
    cfg = json.load(open(os.path.join(d, "config.json")))
    name = os.path.basename(d.rstrip("/"))
    # metrics.jsonl is appended to; keep only the last run of this directory
    starts = [i for i, r in enumerate(recs) if r.get("event") == "init"]
    recs = recs[starts[-1]:] if starts else recs
    steps, evals = [], []
    for r in recs:
        if r.get("event") == "step":
            s = {k: r[k] for k in STEP_KEYS if k in r}
            if "cupti" in r:
                s["cupti"] = {ph: dict(w=v["wall_ms"], b=v["gpu_busy_ms"], k=v["kernels"], m=v.get("mem_peak_GB", 0),
                                       c=v.get("cat_ms", {})) for ph, v in r["cupti"].items()}
                s["cupti_totals"] = r.get("cupti_totals")
            steps.append(s)
        elif r.get("event") == "eval":
            evals.append(dict(step=r["step"], acc=r["eval_acc"], n=r["eval_n"], len=r.get("eval_resp_len")))
    init = next((r for r in recs if r.get("event") == "init"), {})
    end = next((r for r in recs if r.get("event") == "end"), None)
    return dict(name=name, dir=d, cfg=cfg, steps=steps, evals=evals, end=end,
                init=dict(gpu=init.get("gpu"), vllm_init_s=init.get("vllm_init_s"), policy_init_s=init.get("policy_init_s"),
                          rollout_model=init.get("rollout_model")))


def probs_png(run, out):
    hs = sorted(glob.glob(os.path.join(run["dir"], "hist", "*.npy")))
    if not hs:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick = [hs[0], hs[-1]] if len(hs) > 1 else [hs[0]]
    fig, axes = plt.subplots(1, len(pick), figsize=(4.6 * len(pick), 4.2))
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, pick):
        im = ax.imshow(np.log10(np.load(h).astype(float).T + 1), origin="lower", extent=[0, 1, 0, 1], cmap="magma")
        ax.plot([0, 1], [0, 1], color="w", lw=0.6, ls="--")
        ax.set_xlabel("p_vllm (rollout engine)"); ax.set_ylabel("p_fsdp (trainer)")
        ax.set_title(f"step {int(os.path.basename(h)[5:10])}", fontsize=9)
    fig.colorbar(im, ax=list(axes), label="log10(1 + tokens)")
    fig.savefig(out, dpi=100, bbox_inches="tight"); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--include-smoke", action="store_true", help="also list runs whose name starts with smoke/eval")
    args = ap.parse_args()
    dirs = args.runs or sorted(glob.glob(os.path.join(paths.RUNS, "*")))
    if not args.include_smoke and not args.runs:
        dirs = [d for d in dirs if not os.path.basename(d).startswith(("smoke", "eval_"))]
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, "metrics.jsonl"))]
    os.makedirs(os.path.join(DOCS, "data"), exist_ok=True)
    os.makedirs(os.path.join(DOCS, "img"), exist_ok=True)
    runs = []
    for d in dirs:
        r = load(d)
        if not r["steps"]:
            continue
        prof = sorted(glob.glob(os.path.join(d, "profile", "step_*.json")))
        if prof:  # representative step: step 2 if present (after warm-up), else the first
            pick = next((p for p in prof if p.endswith("step_00002.json")), prof[0])
            shutil.copy(pick, os.path.join(DOCS, "data", f"profile_{r['name']}.json"))
            r["profile_step"] = int(os.path.basename(pick)[5:10])
        if probs_png(r, os.path.join(DOCS, "img", f"probs_{r['name']}.png")):
            r["probs_img"] = f"img/probs_{r['name']}.png"
        r.pop("dir")
        runs.append(r)
        print(f"{r['name']}: {len(r['steps'])} steps, {len(r['evals'])} evals")
    with open(os.path.join(DOCS, "data", "runs.json"), "w") as f:
        json.dump(dict(runs=runs), f, separators=(",", ":"))
    print(f"wrote {os.path.join(DOCS, 'data', 'runs.json')} ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
