#!/usr/bin/env python3
"""Turn transcriber sessions into one per-utterance feature table for emotion analysis.

Reads every ``sessions/<name>/utterances.jsonl`` written by the transcriber,
computes acoustic (prosody + spectral) and text-timing features for each kept
utterance from its 16 kHz clip, and writes a CSV (and Parquet when available).

    python scripts/extract_features.py                       # ../transcriber/sessions -> results/features.csv
    python scripts/extract_features.py --sessions-dir DIR --out results/features.csv --force

Feature groups (one row per utterance):
  prosody   : f0 mean/std/min/max/range (Hz, voiced frames), voiced fraction
  energy    : rms mean/std/max, dynamic range (dB)
  spectral  : centroid, bandwidth, rolloff, flatness, zero-crossing rate (means)
  timbre    : mfcc 1-13 mean and std
  timing    : speech rate (words/s, syllable proxy), mean/max pause between words,
              articulation rate, utterance duration
  asr       : avg_logprob, no_speech_prob, mean word probability, word count
Rows are idempotent per (session, index); rerun with --force to recompute.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = EXP_DIR.parent
EMOTE_DIR = ANALYSIS_DIR.parent
DEFAULT_SESSIONS = EMOTE_DIR / "transcriber" / "sessions"
DEFAULT_OUT = ANALYSIS_DIR / "results" / "features.csv"

SR = 16000
N_MFCC = 13
logger = logging.getLogger("extract_features")


def setup_logging(log_dir: Path, verbose: bool) -> Path:
    """Console INFO + a DEBUG file per run under results/logs/."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.handlers[:] = [ch, fh]
    logging.getLogger("numba").setLevel(logging.WARNING)
    return path


def acoustic_features(y: np.ndarray, sr: int = SR) -> dict[str, float]:
    import librosa

    feats: dict[str, float] = {}
    if y.size < sr // 10:
        return feats
    y = y.astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, voiced_flag, _ = librosa.pyin(y, fmin=60, fmax=500, sr=sr, frame_length=1024, hop_length=256)
    f0v = f0[np.isfinite(f0)] if f0 is not None else np.array([])
    feats["f0_mean"] = float(f0v.mean()) if f0v.size else np.nan
    feats["f0_std"] = float(f0v.std()) if f0v.size else np.nan
    feats["f0_min"] = float(f0v.min()) if f0v.size else np.nan
    feats["f0_max"] = float(f0v.max()) if f0v.size else np.nan
    feats["f0_range"] = feats["f0_max"] - feats["f0_min"] if f0v.size else np.nan
    feats["f0_slope"] = float(np.polyfit(np.arange(f0v.size), f0v, 1)[0]) if f0v.size > 2 else np.nan
    feats["voiced_fraction"] = float(np.mean(voiced_flag)) if voiced_flag is not None and voiced_flag.size else np.nan

    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    feats["rms_mean"] = float(rms.mean())
    feats["rms_std"] = float(rms.std())
    feats["rms_max"] = float(rms.max())
    db = librosa.amplitude_to_db(rms + 1e-8)
    feats["dynamic_range_db"] = float(np.percentile(db, 95) - np.percentile(db, 5))

    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    feats["spectral_centroid"] = float(librosa.feature.spectral_centroid(S=S, sr=sr).mean())
    feats["spectral_bandwidth"] = float(librosa.feature.spectral_bandwidth(S=S, sr=sr).mean())
    feats["spectral_rolloff"] = float(librosa.feature.spectral_rolloff(S=S, sr=sr).mean())
    feats["spectral_flatness"] = float(librosa.feature.spectral_flatness(S=S).mean())
    feats["zcr"] = float(librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=256).mean())

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=1024, hop_length=256)
    for i in range(N_MFCC):
        feats[f"mfcc{i + 1}_mean"] = float(mfcc[i].mean())
        feats[f"mfcc{i + 1}_std"] = float(mfcc[i].std())
    return feats


def timing_features(record: dict) -> dict[str, float]:
    words = record.get("words") or []
    dur = float(record.get("duration_s") or 0.0)
    feats: dict[str, float] = {
        "duration_s": dur,
        "speech_ms": float(record.get("speech_ms") or 0.0),
        "word_count": float(len(words)),
        "char_count": float(len(record.get("text") or "")),
    }
    if words:
        starts = np.array([w["start_s"] for w in words])
        ends = np.array([w["end_s"] for w in words])
        probs = np.array([w.get("probability", np.nan) for w in words], dtype=float)
        spoken = float(ends[-1] - starts[0]) if ends[-1] > starts[0] else dur
        gaps = np.clip(starts[1:] - ends[:-1], 0, None)
        word_lens = np.clip(ends - starts, 0, None)
        feats["speech_rate_wps"] = len(words) / spoken if spoken > 0 else np.nan
        feats["articulation_rate_wps"] = len(words) / word_lens.sum() if word_lens.sum() > 0 else np.nan
        feats["pause_mean_s"] = float(gaps.mean()) if gaps.size else 0.0
        feats["pause_max_s"] = float(gaps.max()) if gaps.size else 0.0
        feats["pause_fraction"] = float(gaps.sum() / spoken) if spoken > 0 else np.nan
        feats["word_prob_mean"] = float(np.nanmean(probs)) if probs.size else np.nan
        feats["syllable_rate_proxy"] = sum(_syllables(w["word"]) for w in words) / spoken if spoken > 0 else np.nan
    else:
        for k in ("speech_rate_wps", "articulation_rate_wps", "pause_mean_s", "pause_max_s", "pause_fraction", "word_prob_mean", "syllable_rate_proxy"):
            feats[k] = np.nan
    feats["avg_logprob"] = float(record.get("avg_logprob", np.nan))
    feats["no_speech_prob"] = float(record.get("no_speech_prob", np.nan))
    return feats


def _syllables(word: str) -> int:
    w = "".join(ch for ch in word.lower() if ch.isalpha())
    if not w:
        return 0
    groups = 0
    prev = False
    for ch in w:
        vowel = ch in "aeiouy"
        if vowel and not prev:
            groups += 1
        prev = vowel
    if w.endswith("e") and groups > 1:
        groups -= 1
    return max(1, groups)


def load_records(sessions_dir: Path) -> list[tuple[str, Path, dict]]:
    out = []
    for jsonl in sorted(sessions_dir.glob("*/utterances.jsonl")):
        session = jsonl.parent.name
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append((session, jsonl.parent, json.loads(line)))
    return out


def build_table(sessions_dir: Path, existing: pd.DataFrame | None, include_dropped: bool) -> pd.DataFrame:
    # Imported here, not at the top: the Jetson imports acoustic_features() for the emotion model
    # from the transcriber venv, which has no pandas (found 2026-11-08).
    import pandas as pd

    import soundfile as sf

    done = set(zip(existing["session"], existing["index"])) if existing is not None and len(existing) else set()
    rows = []
    records = load_records(sessions_dir)
    skipped_done = skipped_dropped = missing_clips = 0
    t_start = time.monotonic()
    for n, (session, sdir, rec) in enumerate(records, 1):
        if (session, rec["index"]) in done:
            skipped_done += 1
            continue
        if not rec.get("kept", True) and not include_dropped:
            skipped_dropped += 1
            continue
        row = {
            "session": session,
            "index": rec["index"],
            "start_s": rec.get("start_s"),
            "end_s": rec.get("end_s"),
            "kept": rec.get("kept", True),
            "text": rec.get("text", ""),
            "clip": rec.get("clip"),
        }
        row.update(timing_features(rec))
        clip = sdir / rec["clip"] if rec.get("clip") else None
        t0 = time.monotonic()
        if clip and clip.exists():
            y, sr = sf.read(str(clip), dtype="float32", always_2d=True)
            if sr != SR:
                logger.warning("%s/%s: clip rate %d Hz, expected %d", session, rec["clip"], sr, SR)
            row.update(acoustic_features(y.mean(axis=1), sr))
            logger.debug("%s #%d: %.2fs clip, %d words, f0=%.0f Hz, rms=%.3f, features in %.2fs, text=%r",
                         session, rec["index"], y.shape[0] / sr, len(rec.get("words") or []),
                         row.get("f0_mean") or float("nan"), row.get("rms_mean") or float("nan"),
                         time.monotonic() - t0, (rec.get("text") or "")[:60])
        else:
            missing_clips += 1
            logger.warning("%s #%d: clip missing (%s); acoustic features left empty", session, rec["index"], clip)
        rows.append(row)
        if n % 25 == 0:
            logger.info("  %d/%d utterances (%.1fs elapsed)", n, len(records), time.monotonic() - t_start)
    logger.info("featurized %d new utterance(s) in %.1fs; skipped %d already done, %d dropped-by-transcriber, %d missing clips",
                len(rows), time.monotonic() - t_start, skipped_done, skipped_dropped, missing_clips)
    new = pd.DataFrame(rows)
    if len(new):
        nan_cols = new.columns[new.isna().any()].tolist()
        if nan_cols:
            logger.debug("columns with NaN in new rows: %s", nan_cols)
    if existing is not None and len(existing):
        return pd.concat([existing, new], ignore_index=True) if len(new) else existing
    return new


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sessions-dir", default=str(DEFAULT_SESSIONS))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--include-dropped", action="store_true", help="also featurize utterances the transcriber filtered out")
    p.add_argument("--force", action="store_true", help="recompute everything instead of appending new utterances")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true", help="DEBUG lines on the console (always in the log file)")
    args = p.parse_args(argv)

    sessions_dir = Path(args.sessions_dir)
    out = Path(args.out)
    log_path = setup_logging(out.parent / "logs", args.verbose)
    logger.info("extract_features: sessions=%s out=%s force=%s include_dropped=%s log=%s", sessions_dir, out, args.force, args.include_dropped, log_path)
    import librosa

    logger.info("python %s, numpy %s, pandas %s, librosa %s", sys.version.split()[0], np.__version__, pd.__version__, librosa.__version__)
    if not sessions_dir.exists():
        logger.error("No sessions directory at %s", sessions_dir)
        return 1
    existing = None
    if out.exists() and not args.force:
        existing = pd.read_csv(out)
        logger.info("existing table: %d rows (will append only new utterances; --force recomputes all)", len(existing))
    records = load_records(sessions_dir)
    sessions = sorted(set(s for s, _, _ in records))
    logger.info("%d utterance record(s) in %d session(s): %s", len(records), len(sessions), ", ".join(sessions[:10]) + (" ..." if len(sessions) > 10 else ""))
    for jsonl in sorted(sessions_dir.glob("*/utterances.jsonl")):
        meta = jsonl.parent / "session.json"
        if meta.exists():
            m = json.loads(meta.read_text())
            logger.debug("session %s: mode=%s model=%s kept=%s dropped=%s", jsonl.parent.name, m.get("mode"), (m.get("model") or {}).get("name"), m.get("utterances_kept"), m.get("utterances_dropped"))
        else:
            logger.warning("session %s has no session.json (transcriber did not shut down cleanly?)", jsonl.parent.name)
    if args.dry_run:
        logger.info("dry run: would write %s", out)
        return 0
    table = build_table(sessions_dir, existing, args.include_dropped)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    try:
        table.to_parquet(out.with_suffix(".parquet"), index=False)
    except Exception:  # noqa: BLE001 - pyarrow optional
        pass
    logger.info("wrote %s (%d rows, %d columns)", out, len(table), table.shape[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
