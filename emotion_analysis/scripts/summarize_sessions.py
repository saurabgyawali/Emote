#!/usr/bin/env python3
"""Print a per-session summary table (utterances, minutes, mean prosody) from features.csv.

    python scripts/summarize_sessions.py [--features results/features.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FEATURES = ANALYSIS_DIR / "results" / "features.csv"

SUMMARY_COLS = ["duration_s", "word_count", "speech_rate_wps", "pause_fraction", "f0_mean", "f0_std", "rms_mean", "spectral_centroid"]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    kept = df[df["kept"]] if "kept" in df else df
    g = kept.groupby("session")
    out = pd.DataFrame({"utterances": g.size(), "minutes_speech": g["duration_s"].sum() / 60.0})
    for col in SUMMARY_COLS[1:]:
        if col in kept:
            out[f"mean_{col}"] = g[col].mean()
    return out.round(3)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", default=str(DEFAULT_FEATURES))
    args = p.parse_args(argv)
    path = Path(args.features)
    if not path.exists():
        print(f"{path} not found; run scripts/extract_features.py first")
        return 1
    df = pd.read_csv(path)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summarize(df).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
