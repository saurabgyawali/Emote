import numpy as np

from transcriber.resampler import Resampler, float_to_pcm16, pcm16_to_float


def _tone(sr, seconds=2.0, hz=440.0):
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_passthrough():
    r = Resampler(16000, 16000)
    x = np.ones(100, dtype=np.float32)
    assert r.backend == "passthrough" and r.delay_samples == 0
    assert np.array_equal(r.process(x), x)


def test_total_length_matches_rate_ratio():
    for sr in (44100, 48000, 22050, 32000):
        r = Resampler(sr, 16000)
        x = _tone(sr)
        out = np.concatenate([r.process(b) for b in np.array_split(x, 37)] + [r.flush()])
        delay_out = r.delay_samples * 16000 // sr
        assert abs(out.size - delay_out - 32000) <= 2, (sr, out.size)


def test_blockwise_equals_whole_signal():
    # Streaming through 100 ms blocks must match one-shot resampling in the interior.
    for sr in (44100, 48000):
        x = _tone(sr, seconds=1.5, hz=1000.0)
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, 16000)
        whole = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
        r = Resampler(sr, 16000)
        block = int(sr * 0.1)
        blocks = [x[i : i + block] for i in range(0, x.size, block)]
        stream = np.concatenate([r.process(b) for b in blocks] + [r.flush()])
        delay_out = r.delay_samples * 16000 // sr
        n = min(whole.size - 400, stream.size - delay_out - 400)
        a = whole[200 : 200 + n]
        b = stream[delay_out + 200 : delay_out + 200 + n]
        assert np.max(np.abs(a - b)) < 1e-3, sr


def test_no_click_at_block_boundaries():
    sr = 44100
    x = _tone(sr, seconds=1.0, hz=200.0)
    r = Resampler(sr, 16000)
    block = int(sr * 0.1)
    out = np.concatenate([r.process(x[i : i + block]) for i in range(0, x.size, block)])
    # A 200 Hz tone at 16 kHz changes by at most ~0.04 per sample; a boundary
    # dip from zero-padding would be far larger.
    diffs = np.abs(np.diff(out[500:]))
    assert diffs.max() < 0.06


def test_pcm_roundtrip():
    src = np.linspace(-0.9, 0.9, 1000, dtype=np.float32)
    back = pcm16_to_float(float_to_pcm16(src))
    assert np.max(np.abs(back - src)) < 1e-3
