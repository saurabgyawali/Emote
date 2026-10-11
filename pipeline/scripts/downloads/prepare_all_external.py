"""Download every open corpus and the pretrained encoders, each step its own idempotent subprocess.

    python prepare_all_external.py                 # RAVDESS, CREMA-D, TESS, MELD + core encoders
    python prepare_all_external.py --skip-meld     # without the 10.9 GB MELD download
    python prepare_all_external.py --extended-models --large-model
    python prepare_all_external.py --dry-run

Licensed corpora (IEMOCAP, MSP-Podcast) cannot be fetched by script: see NEXT_STEPS.md.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default=None)
    p.add_argument("--skip-meld", action="store_true")
    p.add_argument("--extended-models", action="store_true")
    p.add_argument("--large-model", action="store_true")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    common = [f"--data-root={args.data_root}"] if args.data_root else []
    common += ["--force"] if args.force else []
    common += ["--dry-run"] if args.dry_run else []
    steps = [["download_pretrained.py", *(["--extended"] if args.extended_models else []),
              *(["--large"] if args.large_model else [])],
             ["download_ravdess.py"], ["download_crema_d.py", f"--workers={args.workers}"],
             ["download_tess.py", f"--workers={min(args.workers, 4)}"]]
    if not args.skip_meld:
        steps.append(["download_meld.py"])
    failed = []
    for step in steps:
        cmd = [sys.executable, str(HERE / step[0]), *step[1:], *[c for c in common if not (step[0] == "download_pretrained.py" and c == "--force")]]
        print(f"\n==> {' '.join(step)}", flush=True)
        if subprocess.run(cmd).returncode:
            failed.append(step[0])
    if failed:
        raise SystemExit(f"\nfailed: {failed}. Every step resumes where it stopped: just run this again.")
    print("\nall downloads complete")


if __name__ == "__main__":
    main()
