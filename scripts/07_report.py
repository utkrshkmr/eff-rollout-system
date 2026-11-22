"""Step 7: tables and figures from one or more runs.

    python scripts/07_report.py $RLPROF_HOME/runs/<run_a> $RLPROF_HOME/runs/<run_b> --out report/
    python scripts/07_report.py --all --out report/          # every run under $RLPROF_HOME/runs

Writes <out>/report.md plus:
    curves.png         reward, eval accuracy, mismatch KL, |p_vllm - p_fsdp|, entropy vs step
    phases_<run>.png   per-phase wall time vs CUPTI GPU-busy time, and GPU time by kernel category
    probs_<run>.png    (p_vllm, p_fsdp) token density for the first logged step (cf. paper Fig. 1/3)
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlprof import paths  # noqa: E402

TOP_PHASES = ["weight_sync", "rollout", "reward", "advantage", "old_logprob", "mismatch", "train", "eval"]


def load_run(d):
    recs = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl"))] if os.path.exists(os.path.join(d, "metrics.jsonl")) else []
    starts = [i for i, r in enumerate(recs) if r.get("event") == "init"]
    recs = recs[starts[-1]:] if starts else recs   # only the latest run written to this directory
    profs = {os.path.basename(p)[:-5]: json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "profile", "*.json")))}
    cfg = json.load(open(os.path.join(d, "config.json"))) if os.path.exists(os.path.join(d, "config.json")) else {}
    return dict(name=os.path.basename(d.rstrip("/")), dir=d, steps=[r for r in recs if r.get("event") == "step"],
                evals=[r for r in recs if r.get("event") == "eval"], init=[r for r in recs if r.get("event") == "init"],
                profiles=profs, cfg=cfg)


def rollup(phases: dict) -> dict:
    """Leaf phases -> top-level totals (wall from the top-level phase itself, GPU sums over children)."""
    out = defaultdict(lambda: defaultdict(float))
    for name, p in phases.items():
        top = name.split("/")[0]
        o = out[top]
        if name == top:
            o["wall_ms"] = p.get("wall_ms", 0.0)
        for k in ("gpu_busy_ms", "kernel_ms", "kernels", "api_calls", "api_ms"):
            o[k] += p.get(k, 0) or 0
        for c, v in (p.get("by_category") or {}).items():
            o["cat:" + c] += v["ms"]
        for kind, v in (p.get("memcpy") or {}).items():
            o["memcpy_MB:" + kind] += v["MB"]
        o["mem_peak_GB"] = max(o["mem_peak_GB"], p.get("mem_peak_GB", 0.0))
    return out


def fmt(x, nd=1):
    return f"{x:,.{nd}f}" if isinstance(x, (int, float)) else str(x)


def phase_table(prof: dict) -> str:
    ph = prof["phases"]
    lines = ["| phase | wall ms | GPU busy ms | busy % | kernels | kernel ms | top categories (ms) | memcpy MB | peak mem GB | CUDA API calls |",
             "|---|---:|---:|---:|---:|---:|---|---|---:|---:|"]
    for name in sorted(ph, key=lambda n: (TOP_PHASES.index(n.split("/")[0]) if n.split("/")[0] in TOP_PHASES else 99, n)):
        p = ph[name]
        cats = ", ".join(f"{c} {v['ms']:.1f}" for c, v in list((p.get("by_category") or {}).items())[:3])
        mc = ", ".join(f"{k} {v['MB']:.0f}" for k, v in (p.get("memcpy") or {}).items())
        busy = p.get("gpu_busy_frac")
        lines.append(f"| {name} | {fmt(p.get('wall_ms', 0))} | {fmt(p.get('gpu_busy_ms', 0))} | "
                     f"{'' if busy is None else fmt(100 * busy, 0)} | {p.get('kernels', 0):,} | {fmt(p.get('kernel_ms', 0))} | "
                     f"{cats} | {mc} | {fmt(p.get('mem_peak_GB', 0), 2)} | {p.get('api_calls', 0):,} |")
    return "\n".join(lines)


def top_kernels_table(prof: dict, phases=("rollout", "old_logprob", "train/forward", "train/backward", "train/optimizer",
                                         "weight_sync/vllm_load"), k=5) -> str:
    out = []
    for name in phases:
        p = prof["phases"].get(name)
        if not p or not p.get("top_kernels"):
            continue
        out.append(f"\n**{name}** ({p['kernels']:,} kernels, {p['kernel_ms']:.1f} ms; "
                   f"{p.get('graph_kernels', 0):,} from CUDA graphs)\n")
        out.append("| kernel | category | count | ms |\n|---|---|---:|---:|")
        for kk in p["top_kernels"][:k]:
            nm = kk["name"] if len(kk["name"]) < 90 else kk["name"][:87] + "..."
            out.append(f"| `{nm.replace('|', '/')}` | {kk['category']} | {kk['count']:,} | {kk['ms']:.2f} |")
    return "\n".join(out)


def plot_curves(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [("reward_mean", "train reward"), ("eval_acc", "eval accuracy"), ("kl_k3", "KL(pi_vllm||pi_fsdp), k3"),
            ("prob_absdiff_mean", "mean |p_vllm - p_fsdp|"), ("entropy_mean", "policy entropy"),
            ("tis_truncated_frac", "fraction of tokens with ratio > C")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (k, title) in zip(axes.flat, keys):
        for r in runs:
            src = r["evals"] if k == "eval_acc" else r["steps"]
            xs = [x["step"] for x in src if k in x]
            ys = [x[k] for x in src if k in x]
            if xs:
                ax.plot(xs, ys, marker="o" if len(xs) < 30 else None, ms=3, label=r["name"])
        ax.set_title(title); ax.set_xlabel("step"); ax.grid(alpha=0.3)
        if k.startswith("kl"):
            ax.set_yscale("symlog", linthresh=1e-4)
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def plot_phases(run, prof, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    roll = rollup(prof["phases"])
    names = [n for n in TOP_PHASES if n in roll]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 4.8))
    x = np.arange(len(names))
    a1.bar(x - 0.2, [roll[n]["wall_ms"] for n in names], 0.4, label="wall (host)")
    a1.bar(x + 0.2, [roll[n]["gpu_busy_ms"] for n in names], 0.4, label="GPU busy (CUPTI)")
    a1.set_xticks(x, names, rotation=30); a1.set_ylabel("ms"); a1.set_yscale("log"); a1.legend()
    a1.set_title(f"{run['name']}  {prof.get('tag', '')}", fontsize=9)
    cats = sorted({k[4:] for n in names for k in roll[n] if k.startswith("cat:")},
                  key=lambda c: -sum(roll[n].get("cat:" + c, 0) for n in names))
    bottom = np.zeros(len(names))
    for c in cats:
        v = np.array([roll[n].get("cat:" + c, 0.0) for n in names])
        a2.bar(x, v, bottom=bottom, label=c); bottom += v
    a2.set_xticks(x, names, rotation=30); a2.set_ylabel("GPU kernel ms"); a2.legend(fontsize=7, ncol=2)
    a2.set_title("GPU time by kernel category")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def plot_probs(run, out):
    hs = sorted(glob.glob(os.path.join(run["dir"], "hist", "*.npy")))
    if not hs:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = np.load(hs[0]).astype(float)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(np.log10(h.T + 1), origin="lower", extent=[0, 1, 0, 1], cmap="viridis", aspect="equal")
    ax.plot([0, 1], [0, 1], "w--", lw=0.8)
    ax.set_xlabel("p_vllm (rollout engine)"); ax.set_ylabel("p_fsdp (trainer)")
    ax.set_title(f"{run['name']}\n{os.path.basename(hs[0])[:-4]}: token probabilities", fontsize=8)
    fig.colorbar(im, label="log10(1 + tokens)")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="report")
    args = ap.parse_args()
    dirs = args.runs or (sorted(glob.glob(os.path.join(paths.RUNS, "*"))) if args.all else [])
    if not dirs:
        ap.error("give run directories or --all")
    runs = [load_run(d) for d in dirs]
    os.makedirs(args.out, exist_ok=True)
    md = ["# RL fine-tuning profile report\n"]

    md.append("## Runs\n\n| run | rollout | correction | steps | final train reward | first → last eval acc | mean KL k3 | mean abs dp | step s (median) |\n|---|---|---|---:|---:|---|---:|---:|---:|")
    for r in runs:
        st, ev, c = r["steps"], r["evals"], r["cfg"]
        m = lambda k: float(np.mean([s[k] for s in st if k in s])) if st else float("nan")
        evs = f"{ev[0]['eval_acc']:.3f} → {ev[-1]['eval_acc']:.3f}" if ev else ""
        md.append(f"| {r['name']} | {c.get('rollout_quant')} | {c.get('correction')} | {len(st)} | "
                  f"{st[-1]['reward_mean']:.3f} | {evs} | {m('kl_k3'):.2e} | {m('prob_absdiff_mean'):.4f} | "
                  f"{np.median([s['step_s'] for s in st]) if st else float('nan'):.1f} |" if st else
                  f"| {r['name']} | {c.get('rollout_quant')} | {c.get('correction')} | 0 | | {evs} | | | |")
    try:
        plot_curves(runs, os.path.join(args.out, "curves.png"))
        md.append("\n![curves](curves.png)\n")
    except Exception as e:  # matplotlib missing, etc.
        md.append(f"\n(curves not plotted: {e!r})\n")

    for r in runs:
        md.append(f"\n## {r['name']}\n")
        if r["init"]:
            i = r["init"][-1]
            md.append(f"GPU: {i.get('gpu')}; vLLM init {i.get('vllm_init_s', 0):.1f} s, FSDP policy init "
                      f"{i.get('policy_init_s', 0):.1f} s; rollout model `{i.get('rollout_model')}`.\n")
        p = plot_probs(r, os.path.join(args.out, f"probs_{r['name']}.png"))
        if p:
            md.append(f"![probs](probs_{r['name']}.png)\n")
        for tag, prof in r["profiles"].items():
            md.append(f"\n### CUPTI profile `{tag}`\n")
            md.append(f"{prof['total_kernels']:,} kernels, {prof['total_copies']:,} copies, "
                      f"{prof['total_api_calls']:,} CUDA API calls recorded.\n")
            md.append(phase_table(prof))
            md.append(top_kernels_table(prof))
            try:
                fn = f"phases_{r['name']}_{tag}.png"
                plot_phases(r, prof, os.path.join(args.out, fn))
                md.append(f"\n![phases]({fn})\n")
            except Exception as e:
                md.append(f"\n(phase plot failed: {e!r})\n")
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {os.path.join(args.out, 'report.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
