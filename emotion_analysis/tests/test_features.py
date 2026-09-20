import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import extract_features as ef  # noqa: E402


def _tone_clip(path: Path, seconds=1.0, hz=180.0):
    t = np.arange(int(16000 * seconds)) / 16000
    y = (0.4 * np.sin(2 * np.pi * hz * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
        wf.writeframes((y * 32767).astype("<i2").tobytes())
    return y


def test_syllables():
    assert ef._syllables("hello") == 2
    assert ef._syllables("the") == 1
    assert ef._syllables("cake") == 1
    assert ef._syllables("!!!") == 0


def test_timing_features():
    rec = {"duration_s": 2.0, "speech_ms": 1500, "text": "hi there friend", "avg_logprob": -0.2, "no_speech_prob": 0.1,
           "words": [{"word": "hi", "start_s": 0.0, "end_s": 0.2, "probability": 0.9},
                     {"word": "there", "start_s": 0.5, "end_s": 0.8, "probability": 0.8},
                     {"word": "friend", "start_s": 0.9, "end_s": 1.2, "probability": 0.7}]}
    f = ef.timing_features(rec)
    assert f["word_count"] == 3
    assert abs(f["speech_rate_wps"] - 3 / 1.2) < 1e-9
    assert abs(f["pause_max_s"] - 0.3) < 1e-9
    assert abs(f["word_prob_mean"] - 0.8) < 1e-9


def test_acoustic_features_on_tone(tmp_path):
    y = _tone_clip(tmp_path / "c.wav")
    f = ef.acoustic_features(y)
    assert 150 < f["f0_mean"] < 210
    assert f["rms_mean"] > 0.1
    assert "mfcc13_std" in f


def test_end_to_end_table(tmp_path):
    sdir = tmp_path / "sessions" / "20260101_000000"
    (sdir / "clips").mkdir(parents=True)
    _tone_clip(sdir / "clips" / "utt_0001.wav")
    rec = {"index": 1, "start_s": 0.5, "end_s": 1.5, "duration_s": 1.0, "speech_ms": 900, "text": "hello world", "kept": True,
           "avg_logprob": -0.3, "no_speech_prob": 0.05, "clip": "clips/utt_0001.wav",
           "words": [{"word": "hello", "start_s": 0.5, "end_s": 0.9, "probability": 0.9}, {"word": "world", "start_s": 1.0, "end_s": 1.4, "probability": 0.9}]}
    dropped = dict(rec, index=2, kept=False, text="you", clip=None, words=[])
    (sdir / "utterances.jsonl").write_text(json.dumps(rec) + "\n" + json.dumps(dropped) + "\n")
    out = tmp_path / "features.csv"
    assert ef.main(["--sessions-dir", str(tmp_path / "sessions"), "--out", str(out)]) == 0
    import pandas as pd

    df = pd.read_csv(out)
    assert len(df) == 1 and df.loc[0, "word_count"] == 2 and df.loc[0, "f0_mean"] > 100
    # Re-running appends nothing new (idempotent).
    assert ef.main(["--sessions-dir", str(tmp_path / "sessions"), "--out", str(out)]) == 0
    assert len(pd.read_csv(out)) == 1
