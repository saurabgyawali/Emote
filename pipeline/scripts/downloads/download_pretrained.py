"""Download the pretrained encoders into the data root's Hugging Face cache, once.

    python download_pretrained.py                 # the defaults the core experiments need
    python download_pretrained.py --extended      # + HuBERT, wav2vec2 (encoder comparison)
    python download_pretrained.py --large         # + WavLM-Large (needs a >= 24 GB GPU to fine-tune)

Training reads the same cache (<data root>/pretrained, forced over any HF_HOME the machine sets),
so nothing is fetched mid-run.
"""

from __future__ import annotations

import argparse
import os

from _fetch import data_root, use_data_cache

CORE = ["microsoft/wavlm-base-plus", "j-hartmann/emotion-english-distilroberta-base"]
EXTENDED = ["facebook/hubert-base-ls960", "facebook/wav2vec2-base"]
LARGE = ["microsoft/wavlm-large"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default=None)
    p.add_argument("--extended", action="store_true")
    p.add_argument("--large", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cache = use_data_cache(data_root(args.data_root))
    repos = CORE + (EXTENDED if args.extended else []) + (LARGE if args.large else [])
    if args.dry_run:
        print(f"would download {repos} into {cache}")
        return
    from huggingface_hub import snapshot_download

    for repo in repos:
        # These repos ship pytorch_model.bin (no safetensors) next to TensorFlow weights we skip.
        path = snapshot_download(repo, allow_patterns=["*.json", "*.txt", "*.safetensors", "pytorch_model.bin"])
        print(f"  {repo} -> {path}")
    print(f"pretrained models ready in {cache}")


if __name__ == "__main__":
    main()
