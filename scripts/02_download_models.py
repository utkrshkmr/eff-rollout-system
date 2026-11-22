"""Step 2: download SFT (instruction-tuned) starting checkpoints and check that they load.

    python scripts/02_download_models.py                       # default: qwen2.5-0.5b-instruct
    python scripts/02_download_models.py --models qwen2.5-0.5b-instruct qwen2.5-1.5b-instruct
    python scripts/02_download_models.py --models some-org/some-sft-model   # any HF repo id
    python scripts/02_download_models.py --list
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlprof import paths, registry  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen2.5-0.5b-instruct"])
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for k, (repo, note) in registry.MODELS.items():
            print(f"{k:28s} {repo:36s} {note}")
        return 0
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer
    for name in args.models:
        repo = registry.resolve_model(name)
        path = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
        cfg = AutoConfig.from_pretrained(repo)
        tok = AutoTokenizer.from_pretrained(repo)
        files = [f for f in os.listdir(path) if f.endswith(".safetensors")]
        size = sum(os.path.getsize(os.path.join(path, f)) for f in files) / 1e9
        assert files, f"no safetensors in {path}"
        assert tok.chat_template, f"{repo} has no chat template; use an instruction-tuned (SFT) model"
        print(f"ok  {repo}: {cfg.model_type}, {cfg.num_hidden_layers} layers, hidden {cfg.hidden_size}, "
              f"{size:.2f} GB weights -> {path}")
    print(f"cache: {os.environ['HF_HOME']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
