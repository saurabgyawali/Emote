#!/usr/bin/env python3
"""Unsupervised baseline over emotion_analysis/results/features.csv.

No labelled emotion data exists yet, so this is NOT an emotion classifier. It
clusters utterances by their acoustic pattern (pitch, energy, timing, timbre)
so we can sanity-check that the features separate distinguishable vocal styles
before any real labels are collected. Once self-report labels exist, this
script is the place a supervised classifier replaces the clustering step.

    python scripts/train_baseline.py                    # -> results/model/
    python scripts/train_baseline.py --clusters 5 --force

Designed to run unattended in the background (see run_pipeline.sh): it logs
progress to results/logs/train_<timestamp>.log and writes a PID-stamped status
file so a caller can poll whether it is still running.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FEATURES = ANALYSIS_DIR / "results" / "features.csv"
DEFAULT_OUT = ANALYSIS_DIR / "results" / "model"

NON_FEATURE_COLS = {"session", "index", "start_s", "end_s", "kept", "text", "clip"}

logger = logging.getLogger("train_baseline")


def setup_logging(log_dir: Path, verbose: bool) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.handlers[:] = [ch, fh]
    return path


def write_status(path: Path, **fields) -> None:
    path.write_text(json.dumps({"pid": os.getpid(), "updated": datetime.now().isoformat(timespec="seconds"), **fields}, indent=2))


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return numeric


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default=str(DEFAULT_FEATURES))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--clusters", type=int, default=4, help="number of vocal-style clusters (default 4; no meaning assigned yet)")
    p.add_argument("--per-speaker-norm", default="session", choices=["none", "session"], help="standardize features within each session before clustering (reduces speaker/recording-level differences)")
    p.add_argument("--min-rows", type=int, default=8, help="refuse to train on fewer than this many utterances")
    p.add_argument("--force", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(out_dir.parent / "logs", args.verbose)
    status_path = out_dir / "STATUS.json"
    write_status(status_path, state="starting", log=str(log_path))
    logger.info("train_baseline starting: features=%s out=%s clusters=%d", args.features, out_dir, args.clusters)

    model_path = out_dir / "model.joblib"
    if model_path.exists() and not args.force:
        logger.info("Model already exists at %s; nothing to do (use --force to retrain)", model_path)
        write_status(status_path, state="skipped_existing", model=str(model_path))
        return 0

    features_path = Path(args.features)
    if not features_path.exists():
        logger.error("No feature table at %s. Run scripts/extract_features.py first.", features_path)
        write_status(status_path, state="failed", error="features.csv missing")
        return 1

    df = pd.read_csv(features_path)
    if "kept" in df.columns:
        df = df[df["kept"]].reset_index(drop=True)
    n = len(df)
    logger.info("Loaded %d kept utterance(s) from %s", n, features_path)
    if n < args.min_rows:
        logger.warning(
            "Only %d utterance(s); need at least %d to train anything meaningful. "
            "This is expected before real data collection starts -- run more sessions first.",
            n, args.min_rows,
        )
        write_status(status_path, state="insufficient_data", rows=n, min_rows=args.min_rows)
        return 0

    cols = feature_columns(df)
    logger.info("Using %d numeric feature columns", len(cols))
    X = df[cols].copy()
    na_frac = X.isna().mean()
    dropped = na_frac[na_frac > 0.5].index.tolist()
    if dropped:
        logger.warning("Dropping %d column(s) >50%% missing: %s", len(dropped), dropped)
        cols = [c for c in cols if c not in dropped]
        X = X[cols]
    X = X.fillna(X.median(numeric_only=True))

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    if args.per_speaker_norm == "session" and "session" in df.columns and df["session"].nunique() > 1:
        logger.info("Standardizing within each of %d session(s) (speaker/recording normalisation)", df["session"].nunique())
        Xn = X.copy()
        for sess, idx in df.groupby("session").groups.items():
            sub = X.loc[idx]
            std = sub.std(ddof=0).replace(0, 1.0)
            Xn.loc[idx] = (sub - sub.mean()) / std
        Xn = Xn.fillna(0.0)
    else:
        logger.info("Global standardization (single session or per-speaker norm disabled)")
        scaler = StandardScaler()
        Xn = pd.DataFrame(scaler.fit_transform(X), columns=cols, index=X.index)

    k = min(args.clusters, max(2, n // 3))
    if k != args.clusters:
        logger.warning("Reducing clusters from %d to %d: too few utterances for the requested count", args.clusters, k)
    t0 = time.time()
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(Xn.values)
    fit_s = time.time() - t0
    sil = silhouette_score(Xn.values, labels) if len(set(labels)) > 1 else float("nan")
    logger.info("KMeans(k=%d) fit in %.2fs; silhouette score %.3f (near 0 = clusters overlap, near 1 = well separated)", k, fit_s, sil)

    pca = PCA(n_components=min(2, Xn.shape[1]))
    coords = pca.fit_transform(Xn.values)
    explained = pca.explained_variance_ratio_.sum() if hasattr(pca, "explained_variance_ratio_") else float("nan")

    import joblib

    joblib.dump({"kmeans": km, "feature_columns": cols, "per_speaker_norm": args.per_speaker_norm}, model_path)

    out = df[["session", "index", "text"] if "text" in df.columns else ["session", "index"]].copy()
    out["cluster"] = labels
    out["pca_1"] = coords[:, 0]
    out["pca_2"] = coords[:, 1] if coords.shape[1] > 1 else 0.0
    assignments_path = out_dir / "cluster_assignments.csv"
    out.to_csv(assignments_path, index=False)
    logger.info("Wrote %s", assignments_path)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(out["pca_1"], out["pca_2"], c=out["cluster"], cmap="tab10", s=28, alpha=0.85)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% var)" if hasattr(pca, "explained_variance_ratio_") else "PC1")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}% var)" if coords.shape[1] > 1 and hasattr(pca, "explained_variance_ratio_") else "PC2")
        ax.set_title(f"Unsupervised vocal-style clusters (k={k}, silhouette={sil:.2f})")
        plt.colorbar(sc, ax=ax, label="cluster")
        plt.tight_layout()
        plot_path = out_dir / "clusters.png"
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.info("Wrote %s", plot_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not render cluster plot: %s", exc)
        plot_path = None

    summary = {
        "rows": n,
        "feature_columns": len(cols),
        "clusters": k,
        "silhouette_score": None if np.isnan(sil) else round(float(sil), 4),
        "pca_explained_variance": None if np.isnan(explained) else round(float(explained), 4),
        "model": str(model_path),
        "assignments": str(assignments_path),
        "plot": str(plot_path) if plot_path else None,
        "note": "Unsupervised baseline only -- no emotion labels exist yet. This checks that acoustic features separate distinguishable vocal styles; it does not predict emotions.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_status(status_path, state="done", **summary)
    logger.info("Done. Summary: %s", json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
