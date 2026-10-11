"""Run the experiment suite: every experiment x seed as its own subprocess, finished runs skipped.

    python train_suite.py                          # configs/suite.yaml default groups x seeds
    python train_suite.py --groups core extended   # more groups
    python train_suite.py --experiments E3_audio_finetune --seeds 0 1 2 3 4
    python train_suite.py --dry-run                # list what would run
    python train_suite.py --override train.epochs=3 --force

A failing run is logged and the suite moves on; the exit code is non-zero if any failed.
Each run's console output is kept in pipeline/results/<experiment>/seed<k>.log.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

EXP_DIR = Path(__file__).resolve().parent
PIPELINE = EXP_DIR.parent.parent
CONFIGS = PIPELINE / "configs"
RESULTS = PIPELINE / "results"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--suite", default=str(CONFIGS / "suite.yaml"))
    p.add_argument("--groups", nargs="*", default=None)
    p.add_argument("--experiments", nargs="*", default=None, help="explicit list; overrides --groups")
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--data-root", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    suite = yaml.safe_load(Path(args.suite).read_text())
    groups = args.groups or suite["default_groups"]
    experiments = args.experiments or [e for g in groups for e in suite["groups"][g]]
    seeds = args.seeds if args.seeds is not None else suite["seeds"]
    runs = [(e, s) for e in experiments for s in seeds]
    print(f"suite: {len(experiments)} experiment(s) x {len(seeds)} seed(s) = {len(runs)} run(s)")
    failed = []
    for n, (exp, seed) in enumerate(runs, 1):
        cmd = [sys.executable, str(EXP_DIR / "train.py"), "--experiment", exp, "--seed", str(seed)]
        if args.override:
            cmd += ["--override", *args.override]
        for flag, value in (("--data-root", args.data_root), ("--device", args.device)):
            if value:
                cmd += [flag, value]
        if args.force:
            cmd.append("--force")
        if args.dry_run:
            cmd.append("--dry-run")
        print(f"\n[{n}/{len(runs)}] {exp} seed {seed}", flush=True)
        log = RESULTS / exp / f"seed{seed}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with log.open("a") as fh:
            fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
            fh.flush()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                sys.stdout.write(line)
                fh.write(line)
            rc = proc.wait()
        if rc:
            failed.append(f"{exp} seed {seed} (exit {rc}, see {log})")
        print(f"  [{exp} seed {seed}] exit {rc} after {(time.time() - t0) / 60:.1f} min", flush=True)
    if failed:
        print("\nFAILED runs:\n  " + "\n  ".join(failed))
        raise SystemExit(1)
    print("\nsuite finished")


if __name__ == "__main__":
    main()
