from pathlib import Path

import pytest

from transcriber.config import Config, load_config

PROJECT = Path(__file__).resolve().parents[1]


def test_shipped_config_is_valid():
    cfg = load_config(PROJECT / "config.toml")
    assert cfg.model_name == "small.en"
    assert cfg.device == "auto"
    assert cfg.microphone == "USB"
    assert cfg.model_path == PROJECT / "models"
    assert cfg.sessions_path == PROJECT / "sessions"


def test_defaults_and_overrides(tmp_path):
    cfg = Config.from_dict({"audio": {"microphone": "", "capture_sample_rate": "48000"}, "model": {"device": "CPU"}}, tmp_path)
    assert cfg.microphone is None
    assert cfg.capture_sample_rate == 48000
    assert cfg.device == "cpu"
    assert cfg.word_timestamps is True


@pytest.mark.parametrize(
    "data",
    [
        {"vad": {"frame_duration_ms": 25}},
        {"vad": {"aggressiveness": 5}},
        {"audio": {"target_sample_rate": 8000}},
        {"model": {"device": "gpu"}},
        {"output": {"session_wav_source": "raw"}},
        {"audio": {"latency": "fast"}},
    ],
)
def test_validate_rejects(data):
    with pytest.raises(ValueError):
        Config.from_dict(data).validate()


def test_missing_file_is_clear(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml")
