import numpy as np
import pytest

from emotion_model.audio import TARGET_DBFS, active_rms, crop, normalize_loudness
from emotion_model.prosody import PROSODY_FEATURES, fit_standardizer, prosody_vector, standardize


def test_loudness_normalisation_ignores_leading_silence():
    y = np.concatenate([np.zeros(16000), 0.01 * np.sin(np.arange(16000) / 5)]).astype(np.float32)
    out = normalize_loudness(y)
    assert 20 * np.log10(active_rms(out)) == pytest.approx(TARGET_DBFS, abs=0.5)
    assert np.all(normalize_loudness(np.zeros(100, np.float32)) == 0)


def test_crop_is_random_in_training_and_centred_otherwise():
    y = np.arange(100, dtype=np.float32)
    assert crop(y, 10, None)[0] == 45
    assert len(crop(y, 10, np.random.default_rng(0))) == 10
    assert len(crop(y, 1000, None)) == 100


def test_prosody_vector_matches_the_stage2_extractor_names():
    pytest.importorskip("librosa")
    t = np.arange(16000) / 16000
    v = prosody_vector((0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32))
    assert v.shape == (len(PROSODY_FEATURES),)
    assert 150 < v[PROSODY_FEATURES.index("f0_mean")] < 210
    stats = fit_standardizer(np.stack([v, v * 1.1]))
    z = standardize(np.stack([v, np.full_like(v, np.nan)]), stats)
    assert np.isfinite(z).all() and np.all(z[1] == 0)


def test_text_normalisation_matches_whisper_style():
    from emotion_model.text import normalize_text

    assert normalize_text("Oh my God, he’s lost it…  “Really” — yes") == "Oh my God, he's lost it... \"Really\" - yes"
    assert normalize_text("") == ""
