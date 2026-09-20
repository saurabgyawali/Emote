#!/usr/bin/env python3
"""Download a faster-whisper model into the project's models/ folder (idempotent).

    python scripts/download_model.py                  # model named in config.toml
    python scripts/download_model.py --model base.en --yes
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / ".hf_cache"))

from transcriber.config import load_config  # noqa: E402
from transcriber.whisper_backend import model_is_downloaded  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None, help="Model name (default: model.name from config.toml)")
    p.add_argument("--config", default=str(PROJECT_DIR / "config.toml"))
    p.add_argument("--yes", action="store_true", help="Do not ask for confirmation")
    p.add_argument("--force", action="store_true", help="Re-download even if present")
    args = p.parse_args()

    config = load_config(args.config)
    model = args.model or config.model_name
    models_dir = config.model_path
    if model_is_downloaded(model, models_dir) and not args.force:
        print(f"Model '{model}' already present in {models_dir}; nothing to do.")
        return 0
    if not args.yes:
        answer = input(f"Download '{model}' (hundreds of MB) into {models_dir}? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            return 0
    models_dir.mkdir(parents=True, exist_ok=True)
    os.environ.pop("HF_HUB_OFFLINE", None)
    from faster_whisper import WhisperModel

    print(f"Downloading '{model}' -> {models_dir} ...")
    WhisperModel(model, device="cpu", compute_type="int8", download_root=str(models_dir))
    if not model_is_downloaded(model, models_dir):
        print("Download finished but model.bin was not found where expected.", file=sys.stderr)
        return 1
    print(f"Done: '{model}' is available offline in {models_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
