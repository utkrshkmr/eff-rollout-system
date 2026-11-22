"""Download RL datasets from the Hugging Face Hub and normalize them to jsonl.

Each record: {"id": str, "prompt": str (the user turn, without instructions), "answer": str,
"task": "math" | "countdown", ...task fields}. Instructions are added at rollout time by
`build_messages`, so the same data works with any chat model.
"""
from __future__ import annotations

import json
import os
import random
import re
from typing import Dict, Iterable, List

from . import paths

MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
COUNTDOWN_INSTRUCTION = (
    "Using the numbers {nums}, create an equation that equals {target}. You can use basic arithmetic "
    "operations (+, -, *, /) and each number must be used exactly once. Show your work, then give the "
    "final equation in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
)


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (or \\fbox{...}) in text, braces balanced."""
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def _gsm8k(split: str) -> Iterable[dict]:
    from datasets import load_dataset
    for i, r in enumerate(load_dataset("openai/gsm8k", "main", split=split)):
        ans = r["answer"].split("####")[-1].strip().replace(",", "")
        yield dict(id=f"gsm8k-{split}-{i}", prompt=r["question"], answer=ans, task="math")


MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus"]


def _math_train() -> Iterable[dict]:
    from datasets import load_dataset
    for subj in MATH_SUBJECTS:
        for i, r in enumerate(load_dataset("EleutherAI/hendrycks_math", subj, split="train")):
            ans = last_boxed(r["solution"])
            if ans:
                yield dict(id=f"math-{subj}-{i}", prompt=r["problem"], answer=ans, task="math",
                           level=r["level"])


def _math500() -> Iterable[dict]:
    from datasets import load_dataset
    for r in load_dataset("HuggingFaceH4/MATH-500", split="test"):
        yield dict(id=r["unique_id"], prompt=r["problem"], answer=r["answer"], task="math",
                   level=str(r["level"]))


DAPO_PREFIX = re.compile(r"^Solve the following math problem step by step\..*?problem\.\s*", re.S)
DAPO_SUFFIX = re.compile(r"\s*Remember to put your answer on its own line after \"Answer:\"\.?\s*$", re.S)


def _dapo() -> Iterable[dict]:
    """DAPO-Math-17k ships each problem ~100 times; keep one copy and drop its answer-format text."""
    from datasets import load_dataset
    seen = set()
    for r in load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train"):
        text = r["prompt"][0]["content"]
        prob = DAPO_SUFFIX.sub("", DAPO_PREFIX.sub("", text)).strip()
        if prob in seen:
            continue
        seen.add(prob)
        yield dict(id=f"dapo-{r['extra_info']['index']}", prompt=prob,
                   answer=str(r["reward_model"]["ground_truth"]), task="math")


def _aime24() -> Iterable[dict]:
    from datasets import load_dataset
    for r in load_dataset("HuggingFaceH4/aime_2024", split="train"):
        yield dict(id=f"aime24-{r['id']}", prompt=r["problem"], answer=str(r["answer"]), task="math")


def _deepscaler() -> Iterable[dict]:
    from datasets import load_dataset
    for i, r in enumerate(load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")):
        yield dict(id=f"deepscaler-{i}", prompt=r["problem"], answer=str(r["answer"]), task="math")


def _countdown(test_size: int = 512):
    from datasets import load_dataset
    d = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    rows = [dict(id=f"countdown-{i}", prompt="", answer=str(r["target"]), task="countdown",
                 nums=list(r["nums"]), target=int(r["target"])) for i, r in enumerate(d)]
    return rows[:-test_size], rows[-test_size:]


def build(name: str) -> Dict[str, List[dict]]:
    if name == "gsm8k":
        return dict(train=list(_gsm8k("train")), test=list(_gsm8k("test")))
    if name == "math":
        return dict(train=list(_math_train()), test=list(_math500()))
    if name == "dapo_math":
        return dict(train=list(_dapo()), test=list(_aime24()))
    if name == "deepscaler":
        return dict(train=list(_deepscaler()), test=list(_aime24()))
    if name == "countdown":
        tr, te = _countdown()
        return dict(train=tr, test=te)
    raise KeyError(f"unknown dataset {name!r}")


def prepare(name: str, out_dir: str | None = None) -> Dict[str, int]:
    out_dir = out_dir or os.path.join(paths.DATA, name)
    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for split, rows in build(name).items():
        with open(os.path.join(out_dir, f"{split}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        counts[split] = len(rows)
    return counts


def load(name: str, split: str, limit: int | None = None, seed: int = 0, shuffle: bool = False) -> List[dict]:
    path = os.path.join(paths.DATA, name, f"{split}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run `python scripts/03_prepare_data.py --datasets {name}`")
    with open(path) as f:
        rows = [json.loads(l) for l in f]
    if shuffle:
        random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def build_messages(row: dict) -> List[dict]:
    if row["task"] == "countdown":
        return [{"role": "user", "content": COUNTDOWN_INSTRUCTION.format(nums=row["nums"], target=row["target"])}]
    return [{"role": "user", "content": f"{row['prompt']}\n\n{MATH_INSTRUCTION}"}]


def render_prompt(tokenizer, row: dict) -> str:
    kw = {}
    if "enable_thinking" in (tokenizer.chat_template or ""):
        kw["enable_thinking"] = False  # Qwen3: keep responses short enough for small-GPU RL
    return tokenizer.apply_chat_template(build_messages(row), tokenize=False, add_generation_prompt=True, **kw)
