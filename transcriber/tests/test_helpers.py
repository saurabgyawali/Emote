import json
import wave

from transcriber.configure import set_toml_value
from transcriber.context_window import ContextWindow
from transcriber.session import SessionWriter
from transcriber.whisper_backend import (
    TranscriptResult,
    Word,
    _cpu_compute_type,
    _looks_like_cuda_problem,
    cpu_fallback_choice,
    looks_hallucinated,
    model_is_downloaded,
    model_repo,
)


def test_context_window_caps():
    cw = ContextWindow(True, 20)
    cw.add("a" * 15)
    cw.add("b" * 15)
    p = cw.get_prompt()
    assert len(p) <= 20 and "b" * 15 in p
    assert ContextWindow(False, 20).get_prompt() == ""


def test_fallback_helpers():
    assert _looks_like_cuda_problem(RuntimeError("not compiled with CUDA support"))
    assert not _looks_like_cuda_problem(RuntimeError("Unable to open file model.bin"))
    assert _cpu_compute_type("int8_float16") == "int8"
    assert _cpu_compute_type("float32") == "float32"
    assert cpu_fallback_choice("medium.en", "base.en") == "base.en"
    assert cpu_fallback_choice("small.en", "base.en") == "small.en"


def test_hallucination_filter():
    good = TranscriptResult(text="hello there", avg_logprob=-0.3, no_speech_prob=0.1, compression_ratio=1.2)
    assert looks_hallucinated(good, 0.6, -1.2, 2.4) is None
    assert looks_hallucinated(TranscriptResult(text="..."), 0.6, -1.2, 2.4) == "empty"
    assert "no_speech_prob" in looks_hallucinated(TranscriptResult(text="you", no_speech_prob=0.9), 0.6, -1.2, 2.4)
    assert "avg_logprob" in looks_hallucinated(TranscriptResult(text="you", avg_logprob=-2.0), 0.6, -1.2, 2.4)


def test_model_is_downloaded(tmp_path):
    assert not model_is_downloaded("small.en", tmp_path)
    snap = tmp_path / "models--Systran--faster-whisper-small.en" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "model.bin").write_bytes(b"x")
    assert model_is_downloaded("small.en", tmp_path)


def test_set_toml_value_keeps_comments():
    text = '[audio]\nmicrophone = "USB"   # mic\n\n[model]\nname = "small.en"\ndevice = "auto"  # d\n'
    out = set_toml_value(text, "model", "device", "cuda")
    assert 'device = "cuda"  # d' in out
    assert 'microphone = "USB"   # mic' in out
    out = set_toml_value(out, "audio", "microphone", 3)
    assert "microphone = 3   # mic" in out


def test_session_writer_roundtrip(tmp_path):
    s = SessionWriter(tmp_path, "unit")
    s.start_wav(44100)
    s.write_wav(b"\x00\x00" * 44100)
    res = TranscriptResult(text="hi there", words=[Word("hi", 0.0, 0.2, 0.9), Word("there", 0.3, 0.6, 0.8)], avg_logprob=-0.2, no_speech_prob=0.05, compression_ratio=1.1, latency_s=0.1)
    rec = s.add_utterance(1.0, 2.0, 600, b"\x00\x00" * 16000, res, None, True, False)
    s.add_utterance(3.0, 3.5, 300, b"\x00\x00" * 8000, TranscriptResult(text="you", no_speech_prob=0.9), "no_speech_prob", True, False)
    s.close()
    assert rec["words"][1]["start_s"] == 1.3
    lines = (s.dir / "utterances.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[1])["kept"] is False
    assert (s.dir / "transcript.txt").read_text() == "[00:00:01.000] hi there\n"
    with wave.open(str(s.dir / "clips" / "utt_0001.wav")) as wf:
        assert wf.getframerate() == 16000 and wf.getnframes() == 16000
    with wave.open(str(s.dir / "session.wav")) as wf:
        assert wf.getframerate() == 44100 and wf.getnframes() == 44100
    meta = json.loads((s.dir / "session.json").read_text())
    assert meta["utterances_kept"] == 1 and meta["utterances_dropped"] == 1


def test_model_repo_names():
    assert model_repo("small.en") == "Systran/faster-whisper-small.en"
    assert model_repo("distil-large-v3") == "Systran/faster-distil-whisper-large-v3"
    assert model_repo("someorg/custom-model") == "someorg/custom-model"
