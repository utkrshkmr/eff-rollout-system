"""Rule-based rewards: 1.0 for a verified-correct final answer, else 0.0."""
from __future__ import annotations

import re
from fractions import Fraction

from .data import last_boxed

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _math_verify(gold: str, pred: str) -> bool:
    try:
        from math_verify import parse, verify
        g = parse(f"${gold}$") or parse(gold)
        p = parse(f"${pred}$") or parse(pred)
        return bool(g) and bool(p) and verify(g, p)
    except Exception:
        return False


def _norm(s: str) -> str:
    s = s.strip().strip("$").replace("\\!", "").replace(" ", "").replace("dfrac", "frac").replace("tfrac", "frac")
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    return s.rstrip(".").replace(",", "") if re.fullmatch(r"-?[\d,]+(\.\d+)?", s) else s


def math_reward(response: str, gold: str) -> float:
    pred = last_boxed(response)
    if pred is None:  # fall back to the last number in the response
        nums = _NUM.findall(response)
        if not nums:
            return 0.0
        pred = nums[-1]
    if _norm(pred) == _norm(gold):
        return 1.0
    return 1.0 if _math_verify(gold, pred) else 0.0


_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)
_ALLOWED = re.compile(r"^[\d+\-*/().\s]+$")


def countdown_reward(response: str, nums, target: int) -> float:
    m = _ANSWER.findall(response)
    if not m:
        return 0.0
    expr = m[-1].strip()
    if "=" in expr:
        expr = expr.split("=")[0]
    if not _ALLOWED.match(expr):
        return 0.0
    used = sorted(int(x) for x in re.findall(r"\d+", expr))
    if used != sorted(int(n) for n in nums):
        return 0.0
    try:
        # Fractions keep 6/4*... exact; the regex above admits only arithmetic
        val = eval(re.sub(r"(\d+)", r"Fraction(\1)", expr), {"__builtins__": {}}, {"Fraction": Fraction})
    except Exception:
        return 0.0
    return 1.0 if val == target else 0.0


def score(row: dict, response: str) -> float:
    if row["task"] == "countdown":
        return countdown_reward(response, row["nums"], row["target"])
    return math_reward(response, row["answer"])
