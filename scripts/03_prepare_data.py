"""Step 3: download RL datasets and write $RLPROF_HOME/data/<name>/{train,test}.jsonl.

    python scripts/03_prepare_data.py                     # all datasets
    python scripts/03_prepare_data.py --datasets gsm8k math
    python scripts/03_prepare_data.py --list
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlprof import data, paths, registry, rewards  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(registry.DATASETS))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for k, v in registry.DATASETS.items():
            print(f"{k:12s} train: {v['train']}\n{'':12s} test:  {v['test']}")
        return 0
    for name in args.datasets:
        counts = data.prepare(name)
        # self-check: a response containing the gold answer must get reward 1.0
        checked = 0
        for r in data.load(name, "test", limit=5):
            if r["task"] == "countdown":
                expr = _countdown_solution(r["nums"], r["target"])
                if expr is None:
                    continue
                resp = f"<answer>{expr}</answer>"
            else:
                resp = f"The answer is \\boxed{{{r['answer']}}}."
            assert rewards.score(r, resp) == 1.0, (name, r["answer"], resp)
            checked += 1
        assert checked, f"{name}: reward self-check ran on no rows"
        print(f"ok  {name}: {counts} -> {os.path.join(paths.DATA, name)} (reward self-check on {checked} rows)")
    return 0


def _countdown_solution(nums, target):
    """Brute-force a left-nested expression ((a o b) o c) o d that hits target, or None."""
    import itertools
    from fractions import Fraction
    for perm in itertools.permutations(nums):
        for ops in itertools.product("+-*/", repeat=len(nums) - 1):
            val, expr = Fraction(perm[0]), str(perm[0])
            try:
                for o, n in zip(ops, perm[1:]):
                    val = {"+": val + n, "-": val - n, "*": val * n, "/": val / n if n else None}[o]
                    if val is None:
                        raise ZeroDivisionError
                    expr = f"({expr} {o} {n})"
            except ZeroDivisionError:
                continue
            if val == target:
                return expr
    return None


if __name__ == "__main__":
    sys.exit(main())
