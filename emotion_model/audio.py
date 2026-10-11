"""Audio in: 16 kHz mono float32, loudness-normalised so corpora recorded at different gains look alike.

Corpora differ by 20 dB or more in recording level, and the Jetson microphone differs
again (its speech sat near -48 dBFS before the gain was fixed). Without normalisation a
model can learn "loud recording = CREMA-D = anger". Every clip, in training and on the
device, is scaled to TARGET_DBFS RMS over its louder frames before anything sees it.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
TARGET_DBFS = -23.0
_FRAME = 400  # 25 ms


def load_audio(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    import soundfile as sf

    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != sample_rate:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(sr, sample_rate)
        y = resample_poly(y, sample_rate // g, sr // g).astype(np.float32)
    return y


def active_rms(y: np.ndarray) -> float:
    """RMS over the loudest 50% of 25 ms frames, so leading silence does not dilute it."""
    n = y.size // _FRAME
    if n == 0:
        return float(np.sqrt(np.mean(y**2))) if y.size else 0.0
    frames = y[: n * _FRAME].reshape(n, _FRAME)
    energy = np.mean(frames**2, axis=1)
    loud = np.sort(energy)[n // 2 :]
    return float(np.sqrt(np.mean(loud)))


def normalize_loudness(y: np.ndarray, target_dbfs: float = TARGET_DBFS) -> np.ndarray:
    rms = active_rms(y)
    if rms < 1e-6:
        return y.astype(np.float32)
    gain = 10 ** (target_dbfs / 20) / rms
    return np.clip(y * gain, -1.0, 1.0).astype(np.float32)


def crop(y: np.ndarray, max_samples: int, rng: np.random.Generator | None) -> np.ndarray:
    """Random window in training (rng given), centre window in evaluation."""
    if y.size <= max_samples:
        return y
    start = int(rng.integers(0, y.size - max_samples + 1)) if rng is not None else (y.size - max_samples) // 2
    return y[start : start + max_samples]


def augment(y: np.ndarray, rng: np.random.Generator, noise_snr_db: tuple[float, float], gain_db: tuple[float, float],
            p_noise: float) -> np.ndarray:
    """Cheap, label-preserving: random gain and, sometimes, white-ish noise at a random SNR.

    Pitch or tempo shifts are deliberately absent: both change exactly the prosody
    that carries emotion.
    """
    y = y * 10 ** (rng.uniform(*gain_db) / 20)
    if rng.random() < p_noise:
        snr = rng.uniform(*noise_snr_db)
        power = np.mean(y**2) + 1e-12
        noise = rng.standard_normal(y.size).astype(np.float32)
        y = y + noise * np.sqrt(power / 10 ** (snr / 10))
    return np.clip(y, -1.0, 1.0).astype(np.float32)
