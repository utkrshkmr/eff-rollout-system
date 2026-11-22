"""SFT (instruction-tuned) starting checkpoints and RL datasets known to work with this repo.

Models: any Hugging Face causal LM that both vLLM and transformers load works; these are the
ones that were tested end to end. Sizes are chosen to fit on one GPU together with the vLLM
rollout engine (the trainer keeps fp32 master weights + AdamW state: ~16 bytes/param).
"""

MODELS = {
    # name: (hf repo, notes)
    "qwen2.5-0.5b-instruct": ("Qwen/Qwen2.5-0.5B-Instruct", "default; ~10 GB trainer state"),
    "qwen2.5-1.5b-instruct": ("Qwen/Qwen2.5-1.5B-Instruct", "~25 GB trainer state"),
    "qwen2.5-math-1.5b-instruct": ("Qwen/Qwen2.5-Math-1.5B-Instruct", "math SFT model, 4k context"),
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", "post-trained (SFT+RL) model; thinking disabled in the prompt"),
}

# Datasets: name -> train source(s) and eval source(s). scripts/03_prepare_data.py turns each
# into $RLPROF_HOME/data/<name>/{train,test}.jsonl with fields id, prompt, answer, task.
DATASETS = {
    "gsm8k": dict(train="openai/gsm8k (main, train)", test="openai/gsm8k (main, test)", task="math"),
    "math": dict(train="EleutherAI/hendrycks_math (7 subjects, train)", test="HuggingFaceH4/MATH-500",
                 task="math"),
    "dapo_math": dict(train="BytedTsinghua-SIA/DAPO-Math-17k (deduplicated)",
                      test="HuggingFaceH4/aime_2024", task="math"),
    "deepscaler": dict(train="agentica-org/DeepScaleR-Preview-Dataset", test="HuggingFaceH4/aime_2024",
                       task="math"),
    "countdown": dict(train="Jiayi-Pan/Countdown-Tasks-3to4 (all but last 512)",
                      test="Jiayi-Pan/Countdown-Tasks-3to4 (last 512)", task="countdown"),
}


def resolve_model(name_or_repo: str) -> str:
    """Registry name -> HF repo id; anything else (repo id or local path) is returned as is."""
    return MODELS[name_or_repo][0] if name_or_repo in MODELS else name_or_repo
