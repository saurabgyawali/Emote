import threading
import time

from transcriber.config import Config
from transcriber.main import Pipeline
from transcriber.utterance_builder import Utterance


def test_last_utterance_survives_a_stop_that_arrives_first(tmp_path):
    """Esc sets stop_event before the audio worker flushes its tail; the tail must still be transcribed."""
    config = Config(sessions_dir=str(tmp_path), base_dir=tmp_path)
    pipeline = Pipeline(config, session_name="race")
    handled = []
    pipeline._handle_utterance = lambda utt, offset: handled.append(utt)
    worker = threading.Thread(target=pipeline._transcription_worker)
    pipeline.stop_event.set()
    worker.start()
    time.sleep(0.3)
    assert worker.is_alive()
    pipeline.utterance_queue.put((Utterance(pcm=b"", start_frame=0, end_frame=1, speech_frames=1), 0.0))
    pipeline.audio_done.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive() and len(handled) == 1


def test_empty_session_folder_is_discarded(tmp_path):
    config = Config(sessions_dir=str(tmp_path), base_dir=tmp_path)
    pipeline = Pipeline(config, session_name="empty")
    assert pipeline.session.dir.exists()
    assert pipeline.discard_if_empty() is True
    assert not pipeline.session.dir.exists()
    assert pipeline.discard_if_empty() is False


def test_level_is_judged_only_on_speech_frames():
    import numpy as np
    from transcriber.main import MIN_SPEECH_FRAMES, SpeechLevel

    def frame(amplitude):
        t = np.arange(480) / 16000
        return (np.sin(2 * np.pi * 220 * t) * amplitude * 32767).astype("<i2").tobytes()

    level = SpeechLevel()
    assert not level.too_quiet  # nobody has spoken yet: not a quiet mic
    for _ in range(MIN_SPEECH_FRAMES):
        level.add(frame(0.006))
    assert level.too_quiet
    level.add(frame(0.1))
    assert not level.too_quiet


def test_looped_utterance_is_decoded_again_without_prompt_or_vocabulary(tmp_path):
    from transcriber.whisper_backend import TranscriptResult

    pipeline = Pipeline(Config(sessions_dir=str(tmp_path), base_dir=tmp_path), session_name="loop")
    pipeline.context.add("earlier confident text")
    calls, prompts = [], []

    class Backend:
        hotwords = "Jetson, CUDA"
        active_device = "cuda"

        def transcribe(self, audio, initial_prompt=None, use_hotwords=True):
            calls.append(use_hotwords)
            prompts.append(initial_prompt)
            if use_hotwords:
                return TranscriptResult(text="good start viv, viv, viv", avg_logprob=-0.2, compression_ratio=20.6, latency_s=1.0)
            return TranscriptResult(text="good start and a good end", avg_logprob=-0.3, compression_ratio=1.1, latency_s=1.0)

    pipeline.backend = Backend()
    utt = Utterance(pcm=b"\x10\x00" * 16000, start_frame=0, end_frame=33, speech_frames=30)
    pipeline._handle_utterance(utt, 0.0)
    assert calls == [True, False]
    assert prompts == ["earlier confident text", "earlier confident text"]  # context kept, only vocabulary dropped
    assert pipeline.stats["utterances_kept"] == 1
    assert "good end" in (pipeline.session.dir / "transcript.txt").read_text()
