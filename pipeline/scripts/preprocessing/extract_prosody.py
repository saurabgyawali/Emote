"""Compute the 42 prosody features for every manifest clip, once, in parallel.

    python extract_prosody.py [--data-root DIR] [--workers 16] [--force] [--dry-run]

Writes <data root>/features/prosody.parquet (utt_id + PROSODY_FEATURES). Clips already
in the table are skipped, so an interrupted run resumes; --force recomputes everything.
pYIN pitch tracking dominates the cost: ~25 k clips take ~1 h on 16 cores.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
PIPELINE = EXP_DIR.parent.parent
EMOTE = PIPELINE.parent
sys.path.insert(0, str(EMOTE))
sys.path.insert(0, str(PIPELINE / "scripts" / "downloads"))

from _fetch import data_root  # noqa: E402
from emotion_model.prosody import PROSODY_FEATURES  # noqa: E402

_ROOT: Path | None = None


def _init(root: str) -> None:
    global _ROOT
    _ROOT = Path(root)
    sys.path.insert(0, str(EMOTE))


def _one(item: tuple[str, str]) -> tuple[str, np.ndarray]:
    from emotion_model.audio import load_audio
    from emotion_model.prosody import prosody_vector

    utt, path = item
    try:
        return utt, prosody_vector(load_audio(str(_ROOT / path)))
    except Exception:  # noqa: BLE001 - an unreadable clip gets NaNs, which standardise to the mean
        return utt, np.full(len(PROSODY_FEATURES), np.nan, dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk", type=int, default=2000, help="clips between checkpoints of the parquet")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    root = data_root(args.data_root)
    manifest = pd.read_csv(root / "manifests" / "manifest.csv")
    out = root / "features" / "prosody.parquet"
    done = pd.DataFrame(columns=["utt_id", *PROSODY_FEATURES])
    if out.exists() and not args.force:
        done = pd.read_parquet(out)
    todo = manifest[~manifest["utt_id"].isin(set(done["utt_id"]))]
    print(f"prosody: {len(done)} cached, {len(todo)} to compute with {args.workers} workers")
    if args.dry_run or todo.empty:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    items = list(zip(todo["utt_id"], todo["path"]))
    # librosa's pYIN is numba-compiled on first use and cached to disk. Many workers
    # compiling at once race on that cache and die (32 workers, cold cache: crashed 2/2,
    # 2026-10-10). Compiling once here means forked workers inherit the compiled code.
    _init(str(root))
    _one(items[0])
    for start in range(0, len(items), args.chunk):
        chunk = items[start : start + args.chunk]
        try:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(str(root),)) as pool:
                batch = list(pool.map(_one, chunk, chunksize=16))
        except BrokenProcessPool:
            print(f"  a worker died; computing clips {start}-{start + len(chunk)} in this process instead", flush=True)
            batch = [_one(item) for item in chunk]
        new = pd.DataFrame([v for _, v in batch], columns=list(PROSODY_FEATURES))
        new.insert(0, "utt_id", [u for u, _ in batch])
        done = pd.concat([done, new], ignore_index=True)
        tmp = out.with_suffix(".tmp.parquet")
        done.to_parquet(tmp, index=False)
        tmp.replace(out)
        print(f"  {min(start + args.chunk, len(items))}/{len(items)}", flush=True)
    nan_rate = float(np.isnan(done[list(PROSODY_FEATURES)].to_numpy(dtype=float)).mean())
    print(f"prosody table: {len(done)} clips, {100 * nan_rate:.1f}% undefined values -> {out}")


if __name__ == "__main__":
    main()
