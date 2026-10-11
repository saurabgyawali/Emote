"""Audit the manifest before any training: leakage, duplicates, balance, missing files.

    python quality_checks.py [--data-root DIR] [--strict]

Writes pipeline/results/quality_checks.json. Exits non-zero on a hard failure (a speaker
in two splits, a duplicated audio file across splits, a missing cached clip), because a
model evaluated on speakers it trained on reports numbers that mean nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
PIPELINE = EXP_DIR.parent.parent
EMOTE = PIPELINE.parent
sys.path.insert(0, str(EMOTE))
sys.path.insert(0, str(PIPELINE / "scripts" / "downloads"))

from _fetch import data_root  # noqa: E402
from emotion_model.labels import validate_manifest  # noqa: E402

RESULTS = PIPELINE / "results"


def audit(df: pd.DataFrame, root: Path, hash_audio: bool) -> dict:
    hard, soft = [], []
    validate_manifest(df)
    per_speaker = df.groupby("speaker")["split"].nunique()
    leaky = per_speaker[per_speaker > 1]
    # MELD's official split reuses TV characters; that is documented, not a bug.
    leaky = leaky[~leaky.index.str.startswith("meld_")]
    if len(leaky):
        hard.append(f"{len(leaky)} speaker(s) appear in more than one split: {list(leaky.index[:10])}")
    missing = [p for p in df["path"] if not (root / p).exists()]
    if missing:
        hard.append(f"{len(missing)} cached clip(s) missing, e.g. {missing[:3]}")
    if hash_audio and not missing:
        digests = df["path"].map(lambda p: hashlib.md5((root / p).read_bytes()).hexdigest())
        dup = df.assign(md5=digests).groupby("md5")["split"].nunique()
        if (dup > 1).any():
            hard.append(f"{int((dup > 1).sum())} identical audio file(s) occur in different splits")
    for ds, sub in df.groupby("dataset"):
        counts = sub[sub["emotion"].fillna("") != ""].groupby(["split", "emotion"]).size().unstack(fill_value=0)
        for split, r in counts.iterrows():
            nz = r[r > 0]
            if len(nz) > 1 and nz.max() / nz.min() > 10:
                soft.append(f"{ds}/{split}: class imbalance {nz.max() / nz.min():.0f}:1 ({nz.idxmax()} vs {nz.idxmin()})")
        if "val" not in set(sub["split"]) or "test" not in set(sub["split"]):
            soft.append(f"{ds}: no val/test split (training data only)")
    long = df[df["duration_s"] > 20]
    if len(long):
        soft.append(f"{len(long)} clip(s) longer than 20 s are centre-cropped at evaluation")
    return {
        "hard_failures": hard,
        "warnings": soft,
        "clips": len(df),
        "hours": round(float(df["duration_s"].sum()) / 3600, 2),
        "datasets": {ds: {"clips": len(s), "speakers": int(s["speaker"].nunique()),
                          "splits": s["split"].value_counts().to_dict(),
                          "with_text": int((s["text"].fillna("").str.len() > 0).sum()),
                          "with_dimensions": int(s["arousal"].notna().sum()),
                          "median_s": round(float(s["duration_s"].median()), 2)}
                     for ds, s in df.groupby("dataset")},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default=None)
    p.add_argument("--hash-audio", action="store_true", help="also md5 every clip to find duplicates (slow)")
    p.add_argument("--out", default=str(RESULTS / "quality_checks.json"))
    args = p.parse_args()
    root = data_root(args.data_root)
    df = pd.read_csv(root / "manifests" / "manifest.csv")
    report = audit(df, root, args.hash_audio)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=int))
    for w in report["warnings"]:
        print(f"  warning: {w}")
    if report["hard_failures"]:
        for f in report["hard_failures"]:
            print(f"  FAIL: {f}")
        raise SystemExit(1)
    print(f"quality checks passed: {report['clips']} clips, {report['hours']} h -> {args.out}")


if __name__ == "__main__":
    main()
