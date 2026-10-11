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
    rec = s.add_utterance(1.0, 2.0, 600, b"\x00\x00" * 16000, res, None, True)
    s.add_utterance(3.0, 3.5, 300, b"\x00\x00" * 8000, TranscriptResult(text="you", no_speech_prob=0.9), "no_speech_prob", True)
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


def test_dbfs_scale():
    from transcriber.main import _dbfs

    assert _dbfs(0.0) == "-inf dBFS"
    assert _dbfs(1.0) == "0 dBFS"
    assert _dbfs(0.003).startswith("-50")   # the level seen on the silent Jetson run


def test_run_log_dir_is_day_then_hour(tmp_path):
    from datetime import datetime

    from transcriber.logging_utils import run_log_dir

    d = run_log_dir(tmp_path, datetime(2026, 9, 20, 9, 21, 3))
    assert d == tmp_path / "2026-09-20" / "09"


def test_cpu_compute_type_respects_what_the_build_supports():
    from transcriber.whisper_backend import _cpu_compute_type

    # The Jetson's source-built CTranslate2 has no efficient int8 CPU backend.
    assert _cpu_compute_type("int8_float16", {"float32"}) == "float32"
    assert _cpu_compute_type("int8_float16", {"int8", "float32"}) == "int8"
    assert _cpu_compute_type("float32", {"float32"}) == "float32"
    assert _cpu_compute_type("int8_float16", set()) == "int8"   # unknown: keep the old behaviour


def test_builder_counts_discarded_short_bursts():
    """Speech the VAD starts but that never reaches min_speech_ms must be counted."""
    from transcriber.utterance_builder import UtteranceParams, UtteranceBuilder

    params = UtteranceParams(frame_duration_ms=30, speech_start_frames=1, silence_end_ms=60,
                             min_speech_ms=300, preroll_ms=0, max_utterance_s=15.0)
    b = UtteranceBuilder(params)
    frame = b"\x00\x00" * params.frame_samples
    assert b.discarded_short == 0
    for is_speech in (True, True, False, False, False):   # 60 ms of speech, then silence
        b.add_frame(frame, is_speech)
    assert b.discarded_short == 1
    assert b.discarded_speech_ms == 60


def test_input_gain_must_be_positive(tmp_path):
    from transcriber.config import load_config

    cfg = tmp_path / "c.toml"
    cfg.write_text('[audio]\nmicrophone = "USB"\ninput_gain = 0\n', encoding="utf-8")
    try:
        load_config(cfg)
    except ValueError as exc:
        assert "input_gain" in str(exc)
    else:
        raise AssertionError("input_gain = 0 should be rejected")


def test_input_gain_defaults_to_one(tmp_path):
    from transcriber.config import load_config

    cfg = tmp_path / "c.toml"
    cfg.write_text('[audio]\nmicrophone = "USB"\n', encoding="utf-8")
    assert load_config(cfg).input_gain == 1.0


def test_suggest_gain_uses_rms_and_keeps_headroom():
    from transcriber.main import suggest_gain

    # The real Jetson recording: one loud bump at 0.094, speech rms 0.0042.
    # Keyed off the peak this would suggest 2; the speech needs about 10.
    assert suggest_gain(0.0943, 0.0042) == 9
    # A peak close to full scale caps the gain no matter how quiet the rms.
    assert suggest_gain(0.9, 0.001) == 1
    assert suggest_gain(0.0, 0.0) == 0


def test_warm_up_runs_the_model_without_the_vad_filter():
    import numpy as np
    from transcriber.whisper_backend import WhisperBackend

    calls = []

    class Info:
        language, language_probability, duration, duration_after_vad = "en", 1.0, 1.0, 1.0

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            calls.append(kwargs)
            return iter(()), Info()

    backend = WhisperBackend("small.en", "models", vad_filter=True, temperature=0.0)
    backend._model = FakeModel()
    backend.warm_up()
    assert calls and calls[0]["vad_filter"] is False
    backend.transcribe(np.zeros(16000, dtype=np.float32))
    assert calls[1]["vad_filter"] is True and calls[1]["temperature"] == 0.0


def test_cuda_out_of_memory_is_retried_not_lost(monkeypatch):
    import numpy as np
    import transcriber.whisper_backend as wb

    monkeypatch.setattr(wb, "OOM_RETRY_S", 0.0)

    class Info:
        language, language_probability, duration, duration_after_vad = "en", 1.0, 1.0, 1.0

    class Seg:
        start, end, avg_logprob, no_speech_prob, compression_ratio, text, words = 0.0, 1.0, -0.2, 0.0, 1.0, " hi", []

    calls = []

    class FlakyModel:
        def transcribe(self, audio, **kwargs):
            calls.append(kwargs["beam_size"])
            if len(calls) == 1:
                raise RuntimeError("CUDA failed with error out of memory")
            return iter([Seg()]), Info()

    backend = wb.WhisperBackend("small.en", "models", beam_size=5)
    backend._model = FlakyModel()
    assert backend.transcribe(np.zeros(16000, dtype=np.float32)).text == "hi"
    assert calls == [5, 5]

    class OutOfMemoryUntilGreedy:
        def transcribe(self, audio, **kwargs):
            calls.append(kwargs["beam_size"])
            if kwargs["beam_size"] > 1:
                raise RuntimeError("CUDA failed with error out of memory")
            return iter([Seg()]), Info()

    calls.clear()
    backend._model = OutOfMemoryUntilGreedy()
    assert backend.transcribe(np.zeros(16000, dtype=np.float32)).text == "hi"
    assert calls == [5, 5, 5, 1]

    class BrokenModel:
        def transcribe(self, audio, **kwargs):
            raise RuntimeError("some other failure")

    backend._model = BrokenModel()
    try:
        backend.transcribe(np.zeros(16000, dtype=np.float32))
        raise AssertionError("expected the non-memory error to propagate")
    except RuntimeError as exc:
        assert "other failure" in str(exc)


def test_memory_hogs_sums_processes_by_name(tmp_path):
    from transcriber.whisper_backend import memory_hogs

    for pid, name, kb in [(1, "codium", 700000), (2, "codium", 400000), (3, "Xorg", 600000), (4, "bash", 4000)]:
        d = tmp_path / str(pid)
        d.mkdir()
        (d / "status").write_text(f"Name:\t{name}\nVmRSS:\t   {kb} kB\n")
    (tmp_path / "5").mkdir()
    (tmp_path / "5" / "status").write_text("Name:\tkthreadd\n")
    assert memory_hogs(2, tmp_path) == [("codium", 1073), ("Xorg", 585)]
