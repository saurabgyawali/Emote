"""Map a SLURM array index to one (experiment, seed) run of the suite in configs/suite.yaml.

    python task_list.py --groups core                    # table: index experiment seed
    python task_list.py --groups core extended --count   # number of array tasks
    python task_list.py --groups core --index 7          # "E2_audio_frozen 1"
    python task_list.py --experiments E5_audio_text_prosody --seeds 0 1 2 3 4 --count

Index order is experiment-major, so the array index is stable while the suite file is unchanged.
Stdlib + PyYAML only, so it runs on a login node before any job starts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

SLURM_DIR = Path(__file__).resolve().parent
PIPELINE = SLURM_DIR.parent


def runs(groups: list[str] | None, experiments: list[str] | None, seeds: list[int] | None,
         suite_path: Path) -> list[tuple[str, int]]:
    suite = yaml.safe_load(suite_path.read_text())
    groups = groups or suite["default_groups"]
    unknown = [g for g in groups if g not in suite["groups"]]
    if unknown:
        raise SystemExit(f"unknown group(s) {unknown}; suite.yaml has {sorted(suite['groups'])}")
    exps = experiments or [e for g in groups for e in suite["groups"][g]]
    missing = [e for e in exps if not (PIPELINE / "configs" / "experiments" / f"{e}.yaml").exists()]
    if missing:
        raise SystemExit(f"no config file for {missing} in configs/experiments/")
    seeds = seeds if seeds is not None else suite["seeds"]
    return [(e, s) for e in exps for s in seeds]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--suite", default=str(PIPELINE / "configs" / "suite.yaml"))
    p.add_argument("--groups", nargs="*", default=None)
    p.add_argument("--experiments", nargs="*", default=None, help="explicit list; overrides --groups")
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--index", type=int, default=None)
    p.add_argument("--count", action="store_true")
    args = p.parse_args()

    todo = runs(args.groups, args.experiments, args.seeds, Path(args.suite))
    if args.count:
        print(len(todo))
    elif args.index is not None:
        if not 0 <= args.index < len(todo):
            raise SystemExit(f"index {args.index} outside 0-{len(todo) - 1}")
        exp, seed = todo[args.index]
        print(exp, seed)
    else:
        for i, (exp, seed) in enumerate(todo):
            print(f"{i:3d}  {exp:28s} seed {seed}")


if __name__ == "__main__":
    main()
