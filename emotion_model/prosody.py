"""The prosody tower's input: the 42 hand-crafted features stage 2 already computes.

Reuses emotion_analysis/scripts/extract_features.acoustic_features unchanged, on
loudness-normalised 16 kHz audio, so the features a model is trained on are exactly
the ones the existing analysis stage and the Jetson produce. pYIN pitch tracking is
slow (~0.3-1 s per clip), so features are computed once per corpus by
pipeline/scripts/preprocessing/extract_prosody.py and cached.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .audio import normalize_loudness

EMOTE_DIR = Path(__file__).resolve().parent.parent
_EXTRACTOR_DIR = EMOTE_DIR / "emotion_analysis" / "scripts"
N_MFCC = 13
PROSODY_FEATURES: tuple[str, ...] = (
    "f0_mean", "f0_std", "f0_min", "f0_max", "f0_range", "f0_slope", "voiced_fraction",
    "rms_mean", "rms_std", "rms_max", "dynamic_range_db",
    "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "spectral_flatness", "zcr",
    *[f"mfcc{i}_{s}" for i in range(1, N_MFCC + 1) for s in ("mean", "std")],
)


def _extractor():
    if str(_EXTRACTOR_DIR) not in sys.path:
        sys.path.insert(0, str(_EXTRACTOR_DIR))
    import extract_features  # noqa: PLC0415 - emotion_analysis/scripts is not a package

    return extract_features.acoustic_features


def prosody_vector(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Feature vector in PROSODY_FEATURES order; NaN where a feature is undefined (no voicing, too short)."""
    feats = _extractor()(normalize_loudness(y), sr)
    return np.array([feats.get(name, np.nan) for name in PROSODY_FEATURES], dtype=np.float32)


def fit_standardizer(matrix: np.ndarray) -> dict[str, list[float]]:
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    return {"mean": mean.tolist(), "std": std.tolist(), "features": list(PROSODY_FEATURES)}


def standardize(matrix: np.ndarray, stats: dict[str, list[float]]) -> np.ndarray:
    """z-score with training statistics; undefined features become 0, the training mean."""
    z = (matrix - np.asarray(stats["mean"], dtype=np.float32)) / np.asarray(stats["std"], dtype=np.float32)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
