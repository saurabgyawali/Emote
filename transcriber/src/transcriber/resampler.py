"""Stream-safe conversion of microphone audio to 16 kHz mono.

The previous implementation resampled each 100 ms block independently, so the
polyphase filter saw zeros at both edges of every block and left a small dip at
each boundary (10 clicks per second in the stream Whisper heard). This version
keeps a short context of already-seen input on both sides of the region it
emits, so consecutive blocks tile seamlessly. Emitted audio lags the input by
``delay_samples`` input samples; callers subtract that when computing times.
"""

from __future__ import annotations

import logging
from math import gcd

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.signal import resample_poly

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


class Resampler:
    """Resample a continuous stream of float32 mono blocks."""

    def __init__(self, orig_sr: int, target_sr: int = 16000) -> None:
        self.orig_sr = int(orig_sr)
        self.target_sr = int(target_sr)
        if self.orig_sr <= 0 or self.target_sr <= 0:
            raise ValueError(f"sample rates must be positive, got {orig_sr}->{target_sr}")
        divisor = gcd(self.orig_sr, self.target_sr)
        self._up = self.target_sr // divisor
        self._down = self.orig_sr // divisor
        # Context on each side must cover the filter half-length (about
        # 10*max(up,down) taps at the upsampled rate). Round up to a multiple of
        # ``down`` so the context maps to a whole number of output samples.
        half_len_in = (10 * max(self._up, self._down)) // self._up + 1
        self._ctx = ((half_len_in + self._down - 1) // self._down) * self._down
        self._carry = np.zeros(2 * self._ctx, dtype=np.float32)
        self._in_total = 0     # input samples whose output has been emitted
        self._out_total = 0    # output samples emitted so far
        if not _HAVE_SCIPY and self.needed:
            logger.warning("SciPy not available; using linear-interpolation resampler.")

    @property
    def needed(self) -> bool:
        return self.orig_sr != self.target_sr

    @property
    def backend(self) -> str:
        if not self.needed:
            return "passthrough"
        return "scipy.resample_poly" if _HAVE_SCIPY else "numpy.interp"

    @property
    def delay_samples(self) -> int:
        """Input-rate samples by which the emitted stream lags the input."""
        return self._ctx if self.needed else 0

    def process(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not self.needed or samples.size == 0:
            return samples
        x = np.concatenate([self._carry, samples])
        y = self._convert(x)
        # Emit the output that corresponds to input region [ctx, ctx + B) of x.
        # Global input positions of that region: [in_total, in_total + B).
        end_out = ((self._in_total + samples.size) * self._up) // self._down
        start_local = (self._ctx * self._up) // self._down
        count = end_out - self._out_total
        out = y[start_local : start_local + count]
        self._in_total += samples.size
        self._out_total += out.size
        keep = 2 * self._ctx
        self._carry = x[-keep:] if x.size >= keep else np.concatenate(
            [np.zeros(keep - x.size, dtype=np.float32), x]
        )
        return np.ascontiguousarray(out, dtype=np.float32)

    def flush(self) -> np.ndarray:
        """Emit the audio still held back as right-hand context."""
        if not self.needed:
            return np.zeros(0, dtype=np.float32)
        return self.process(np.zeros(self._ctx, dtype=np.float32))

    def _convert(self, x: np.ndarray) -> np.ndarray:
        if _HAVE_SCIPY:
            return np.asarray(resample_poly(x, self._up, self._down), dtype=np.float32)
        n_out = int(np.ceil(x.size * self._up / self._down))
        x_old = np.arange(x.size, dtype=np.float64)
        x_new = np.arange(n_out, dtype=np.float64) * self._down / self._up
        return np.interp(x_new, x_old, x).astype(np.float32)


def float_to_pcm16(samples: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
