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


def test_resolve_device_reuses_given_devices(monkeypatch):
    from transcriber import audio_capture

    monkeypatch.setattr(
        audio_capture, "list_input_devices",
        lambda: (_ for _ in ()).throw(AssertionError("PortAudio was queried again")),
    )
    devices = [
        {"index": 0, "name": "USB PnP Sound Device: Audio (hw:0,0)", "max_input_channels": 1,
         "default_samplerate": 44100, "hostapi": "ALSA"},
        {"index": 5, "name": "NVIDIA Jetson Orin Nano APE: - (hw:2,0)", "max_input_channels": 16,
         "default_samplerate": 44100, "hostapi": "ALSA"},
    ]
    assert audio_capture.resolve_device("USB", devices) == 0
    assert audio_capture.resolve_device(5, devices) == 5


def test_doctor_report_reaches_the_log(caplog, tmp_path, capsys):
    import logging

    from transcriber.doctor import run_doctor

    with caplog.at_level(logging.INFO, logger="transcriber.doctor"):
        rc = run_doctor(tmp_path / "missing.toml")
    logged = caplog.text
    printed = capsys.readouterr().out
    assert rc == 1
    assert "doctor report" in logged
    assert "python" in logged and "ctranslate2 CUDA devices" in logged
    assert "doctor STATUS: PROBLEMS FOUND" in logged
    assert "STATUS:" in printed


def test_stop_key_line_mode(monkeypatch):
    """Without a terminal (pipe, nohup) Enter/q still stop the run."""
    import io
    import threading

    from transcriber.main import Pipeline

    class Fake:
        stop_event = threading.Event()

    monkeypatch.setattr("sys.stdin", io.StringIO("\n"))
    assert Pipeline._read_stop_key(Fake()) is True

    monkeypatch.setattr("sys.stdin", io.StringIO("q\n"))
    assert Pipeline._read_stop_key(Fake()) is True

    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert Pipeline._read_stop_key(Fake()) is False
