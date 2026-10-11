"""Entry point: microphone (or a WAV file) -> utterances -> Whisper -> session folder.

Threads and queues
------------------
  sounddevice callback  --audio_queue-->  audio worker  --utterance_queue-->  transcription worker
  (copy block only)                        (resample, VAD,                    (Whisper, filter,
                                            session WAV)                       clips, JSONL, stdout)

Every run writes a DEBUG log (logs/run_<timestamp>.log, also copied into the
session folder as run.log) containing: environment, full config, device
discovery, sample-rate probing, a heartbeat every few seconds (mic level,
queue depth, VAD activity, overflow counts), one line per utterance with
Whisper's confidences, and a final summary. Any worker exception is logged
with its traceback and stops the run cleanly. Stop with Esc / Enter / q /
Ctrl+C / SIGTERM; the session folder is always kept.

Console output has four modes, all writing the same full DEBUG log to disk:
the default screen (mascot, live status line, transcript, warnings -- see ui.py),
--log (INFO log lines), --verbose (DEBUG log lines) and --quiet (transcript only).
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import queue
import select
import shutil
import signal
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

from . import __version__
from .audio_capture import LOSS_WARN, AudioCapture, AudioDeviceError, format_devices, list_input_devices, pick_samplerate, resolve_device
from .config import DEFAULT_CONFIG_PATH, PROJECT_DIR, Config, load_config
from .context_window import ContextWindow
from . import logging_utils
from .logging_utils import cleanup_crash_file, log_environment, setup_logging
from .resampler import Resampler, float_to_pcm16, pcm16_to_float
from .session import SessionWriter, fmt_time
from .ui import Console, dbfs
from .utterance_builder import Utterance, UtteranceBuilder, UtteranceParams
from .vad_gate import VadGate
from .vocabulary import hotwords, load_vocabulary
from .whisper_backend import LOW_MEMORY_MB, WhisperBackend, available_mb, describe_hogs, looks_hallucinated

logger = logging.getLogger(__name__)
QUIET_RMS = 0.015     # loudest speech frame below this (-36 dBFS) and the VAD fragments speech
MIN_SPEECH_FRAMES = 10  # judge the level only after this many VAD speech frames (0.3 s): silence before
                        # anyone talks is not a quiet mic, and one bump's peak (0.093 over speech at
                        # -48 dBFS, 2026-09-20) says nothing about the speech
TARGET_RMS = 0.07     # for the loudest speech frame, about -23 dBFS; average speech then sits near -30
HEADROOM = 0.9        # never let the suggested gain drive the loudest sample into clipping


def frame_rms(frame: bytes) -> float:
    samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0


class SpeechLevel:
    """Loudest RMS over frames the VAD called speech: the level that decides whether speech survives."""

    def __init__(self) -> None:
        self.loudest = 0.0
        self.frames = 0

    def add(self, frame: bytes) -> None:
        self.frames += 1
        self.loudest = max(self.loudest, frame_rms(frame))

    @property
    def judged(self) -> bool:
        return self.frames >= MIN_SPEECH_FRAMES

    @property
    def too_quiet(self) -> bool:
        return self.judged and self.loudest < QUIET_RMS


def suggest_gain(peak: float, rms: float) -> int:
    """Software gain that lifts speech to a usable level without clipping.

    Keyed off RMS, not peak: a single door slam or mic bump sets the peak and
    would otherwise suggest a gain far too small for the actual speech.
    """
    if rms <= 0.0:
        return 0
    by_rms = TARGET_RMS / rms
    by_peak = HEADROOM / peak if peak > 0 else by_rms
    return max(1, int(min(by_rms, by_peak)))
HEARTBEAT_S = 5.0


def build_utterance_params(config: Config) -> UtteranceParams:
    return UtteranceParams(
        sample_rate=config.target_sample_rate,
        frame_duration_ms=config.frame_duration_ms,
        speech_start_frames=config.speech_start_frames,
        silence_end_ms=config.silence_end_ms,
        min_speech_ms=config.min_speech_ms,
        max_utterance_s=config.max_utterance_s,
        preroll_ms=config.preroll_ms,
    )


def log_config(config: Config) -> None:
    logger.info("=== config (%s) ===", config.base_dir / "config.toml")
    for f in dataclasses.fields(config):
        logger.info("  %-24s = %r", f.name, getattr(config, f.name))
    logger.info("  %-24s = %s", "model_path", config.model_path)
    logger.info("  %-24s = %s", "sessions_path", config.sessions_path)
    logger.info("=== end config ===")


class Pipeline:
    def __init__(self, config: Config, session_name: str | None = None, quiet: bool = False,
                 console: Console | None = None) -> None:
        self.config = config
        self.quiet = quiet
        self.console = console
        self.params = build_utterance_params(config)
        logger.debug(
            "Utterance params: frame=%d ms (%d samples, %d bytes), start after %d speech frames, "
            "end after %d ms silence, min speech %d ms, chunk at %.1fs, preroll %d ms",
            self.params.frame_duration_ms, self.params.frame_samples, self.params.frame_bytes,
            self.params.speech_start_frames, self.params.silence_end_ms, self.params.min_speech_ms,
            self.params.max_utterance_s, self.params.preroll_ms,
        )
        self.stop_event = threading.Event()
        self.audio_done = threading.Event()
        self.failed = False
        self._level_peak = 0.0
        self._level_rms = 0.0
        self._speech = SpeechLevel()
        self._level_warned = False
        self._loss_warned = False
        self._builder = None
        # Quiet mode and the default screen show no log lines, so the transcript must
        # print regardless of config; it is the only thing the user would otherwise see.
        self.print_text = config.print_transcript or self.quiet or console is not None
        self.audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=config.input_queue_max_blocks)
        self.utterance_queue: "queue.Queue[tuple[Utterance, float]]" = queue.Queue()
        self.context = ContextWindow(config.context_enabled, config.context_max_chars)
        self.vocabulary = load_vocabulary(config.vocabulary_path) if config.vocabulary_file else []
        logger.info("Vocabulary: %d term(s) from %s%s", len(self.vocabulary), config.vocabulary_path,
                    f": {', '.join(self.vocabulary[:12])}{' ...' if len(self.vocabulary) > 12 else ''}" if self.vocabulary else "")
        self.backend = WhisperBackend(
            model_name=config.model_name,
            model_dir=config.model_path,
            device=config.device,
            compute_type=config.compute_type,
            language=config.language,
            beam_size=config.beam_size,
            word_timestamps=config.word_timestamps,
            vad_filter=config.vad_filter,
            require_cuda=config.require_cuda,
            cpu_fallback_model=config.cpu_fallback_model,
            temperature=config.temperature,
            hotwords=hotwords(self.vocabulary),
        )
        self.session = SessionWriter(config.sessions_path, session_name)
        self._capture: AudioCapture | None = None
        self.stats = {
            "frames": 0, "speech_frames": 0, "utterances_detected": 0, "utterances_transcribed": 0,
            "utterances_kept": 0, "utterances_dropped": 0, "transcribe_seconds": 0.0, "audio_seconds": 0.0,
            "max_utterance_queue": 0, "errors": 0,
        }

    # --- shared helpers ------------------------------------------------------
    def _make_gate_and_builder(self) -> tuple[VadGate, UtteranceBuilder]:
        gate = VadGate(self.config.vad_aggressiveness, self.config.target_sample_rate, self.config.frame_duration_ms)
        logger.debug("VAD gate ready: aggressiveness=%d frame_bytes=%d", self.config.vad_aggressiveness, gate.frame_bytes)
        return gate, UtteranceBuilder(self.params)

    def _handle_utterance(self, utt: Utterance, time_offset_s: float) -> None:
        """Transcribe one utterance and persist it (runs on the transcription thread)."""
        start_s = max(0.0, utt.start_s(self.params) - time_offset_s)
        end_s = max(start_s, utt.end_s(self.params) - time_offset_s)
        speech_ms = utt.speech_frames * self.params.frame_duration_ms
        audio = pcm16_to_float(utt.pcm)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
        prompt = self.context.get_prompt() or None
        queued = self.utterance_queue.qsize()
        logger.info(
            "Transcribing utterance #%d: %.2f-%.2fs (%.2fs audio, %d ms speech, peak=%.3f rms=%.4f, "
            "continued=%s, %d more queued)",
            self.session.next_index, start_s, end_s, end_s - start_s, speech_ms, peak, rms, utt.continued, queued,
        )
        logger.debug("  prompt=%r", prompt)
        self._ui(busy=True, queued=queued)
        try:
            result = self.backend.transcribe(audio, initial_prompt=prompt)
            reason = self._judge(result)
            if reason and reason.startswith("compression_ratio") and (prompt or self.backend.hotwords):
                # A prompt can push Whisper into a loop ("viv, viv, viv, ..." from the vocabulary in
                # the 2026-10-10 benchmark), and the filter then drops the whole utterance, correct
                # half included. One plain decode usually recovers it; only failures pay for it.
                # Drop only the vocabulary when there is one: decoding with no context at all
                # invited stock hallucinations ("For more information, visit www.FEMA.gov").
                with_vocab = bool(self.backend.hotwords)
                logger.info("Utterance looped (%s); decoding once more without the %s.", reason,
                            "vocabulary" if with_vocab else "context prompt")
                plain = self.backend.transcribe(audio, initial_prompt=prompt if with_vocab else None, use_hotwords=False)
                plain.latency_s += result.latency_s
                plain_reason = self._judge(plain)
                if plain_reason is None or not plain_reason.startswith("compression_ratio"):
                    result, reason = plain, plain_reason
        finally:
            self._ui(busy=False, queued=self.utterance_queue.qsize())
        record = self.session.add_utterance(
            start_s=start_s,
            end_s=end_s,
            speech_ms=speech_ms,
            pcm16k=utt.pcm,
            result=result,
            dropped_reason=reason,
            save_clip=self.config.save_clips,
        )
        self._show_transcript(start_s, result.text, reason)
        if reason is None and result.avg_logprob >= self.config.context_min_logprob:
            self.context.add(result.text)
        self.stats["utterances_transcribed"] += 1
        self.stats["utterances_kept" if reason is None else "utterances_dropped"] += 1
        self.stats["transcribe_seconds"] += result.latency_s
        self.stats["audio_seconds"] += end_s - start_s
        rtf = result.latency_s / max(1e-6, end_s - start_s)
        logger.info(
            "Utterance #%d done in %.2fs (RTF %.2f): %s | text=%r | logprob=%.2f no_speech=%.2f "
            "compression=%.2f words=%d segments=%d",
            record["index"], result.latency_s, rtf, "KEPT" if reason is None else f"DROPPED ({reason})",
            result.text[:120], result.avg_logprob, result.no_speech_prob, result.compression_ratio,
            len(result.words), result.segment_count,
        )
        self._ui(kept=self.stats["utterances_kept"], dropped=self.stats["utterances_dropped"])
        if rtf > 1.0:
            if self.backend.active_device == "cpu":
                hint = "On CPU consider model.name = base.en; on Jetson make sure ctranslate2 reports a CUDA device (bash run.sh --doctor)."
            else:
                hint = ("On CUDA this is usually Whisper struggling with weak or noisy input, not the GPU: "
                        "check the level with bash go.sh mic.")
            logger.warning("Transcription slower than real time (RTF %.2f). %s", rtf, hint)

    def _judge(self, result) -> str | None:  # noqa: ANN001
        return looks_hallucinated(result, self.config.no_speech_prob_max, self.config.avg_logprob_min,
                                  self.config.compression_ratio_max)

    def _ui(self, **fields) -> None:
        if self.console is not None:
            self.console.update(**fields)

    def _show_transcript(self, start_s: float, text: str, reason: str | None) -> None:
        stamp = fmt_time(start_s)
        if self.console is not None:
            if reason is None:
                self.console.transcript(stamp, text)
            else:
                self.console.dropped(stamp, text[:60], reason)
        elif reason is None and self.print_text:
            print(f"[{stamp}] {text}", flush=True)

    def _fail(self, where: str, exc: BaseException) -> None:
        logger.exception("%s failed: %s: %s", where, type(exc).__name__, exc)
        self.stats["errors"] += 1
        self.failed = True
        self.stop_event.set()

    # --- workers --------------------------------------------------------------
    def _audio_worker(self, input_sr: int) -> None:
        try:
            resampler = Resampler(input_sr, self.config.target_sample_rate)
            gate, builder = self._make_gate_and_builder()
            self._builder = builder
            gain = self.config.input_gain
            frame_bytes = self.params.frame_bytes
            delay_s = resampler.delay_samples / input_sr
            logger.debug("Resampler %d->%d Hz backend=%s delay=%d samples (%.1f ms)",
                         input_sr, self.config.target_sample_rate, resampler.backend, resampler.delay_samples, delay_s * 1000)
            pending = bytearray()
            was_in_speech = False
            last_beat = time.monotonic()
            beat_frames = beat_speech = 0
            blocks = 0
            first_block_logged = False
            logger.info("listening (waiting for the first audio block)")
            while not self.stop_event.is_set():
                now = time.monotonic()
                if now - last_beat >= HEARTBEAT_S:
                    self._heartbeat(beat_frames, beat_speech, builder.in_speech)
                    beat_frames = beat_speech = 0
                    last_beat = now
                try:
                    block = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                blocks += 1
                raw_peak = float(np.max(np.abs(block))) if block.size else 0.0
                if self.console is not None:
                    self._ui(level_db=max(dbfs(min(1.0, raw_peak * gain)), self.console.status.level_db - 3.0))
                if gain != 1.0 and block.size:
                    block = np.clip(block * gain, -1.0, 1.0)
                if not first_block_logged:
                    first_block_logged = True
                    logger.info("First audio block received: %d samples, peak=%.4f%s",
                                block.size, raw_peak,
                                f" (x{gain:g} gain -> {float(np.max(np.abs(block))):.4f})" if gain != 1.0 else "")
                self._level_peak = max(self._level_peak, raw_peak)
                if block.size:
                    self._level_rms = max(self._level_rms, float(np.sqrt(np.mean(block.astype(np.float64) ** 2))) / gain)
                resampled = resampler.process(block)
                block_pcm = float_to_pcm16(resampled)
                pending += block_pcm
                if self.config.save_session_wav:
                    self.session.write_wav(float_to_pcm16(block) if self.config.session_wav_source == "device" else block_pcm)
                while len(pending) >= frame_bytes:
                    frame = bytes(pending[:frame_bytes])
                    del pending[:frame_bytes]
                    is_speech = gate.is_speech(frame)
                    if is_speech:
                        self._speech.add(frame)
                    beat_frames += 1
                    beat_speech += is_speech
                    self.stats["frames"] += 1
                    self.stats["speech_frames"] += is_speech
                    utt = builder.add_frame(frame, is_speech)
                    if builder.in_speech and not was_in_speech:
                        logger.info("speech started at %.2fs", builder.frame_index * self.params.frame_duration_ms / 1000 - delay_s)
                    was_in_speech = builder.in_speech
                    self._ui(speaking=builder.in_speech)
                    if utt is not None:
                        self.stats["utterances_detected"] += 1
                        logger.info(
                            "speech %s at %.2fs: utterance of %.2fs (%d speech frames) queued for transcription",
                            "chunked (still speaking)" if utt.continued else "ended",
                            utt.end_s(self.params) - delay_s,
                            (utt.end_frame - utt.start_frame) * self.params.frame_duration_ms / 1000,
                            utt.speech_frames,
                        )
                        self.utterance_queue.put((utt, delay_s))
                        self.stats["max_utterance_queue"] = max(self.stats["max_utterance_queue"], self.utterance_queue.qsize())
                        self._ui(queued=self.utterance_queue.qsize())
            tail = builder.flush()
            if tail is not None:
                logger.info("Flushing %.2fs of speech still buffered at shutdown", (tail.end_frame - tail.start_frame) * self.params.frame_duration_ms / 1000)
                self.utterance_queue.put((tail, delay_s))
            else:
                logger.debug("Nothing buffered at shutdown (builder.in_speech=%s)", builder.in_speech)
            logger.info("Audio worker finished: %d blocks processed", blocks)
        except BaseException as exc:  # noqa: BLE001
            self._fail("audio worker", exc)
        finally:
            self.audio_done.set()

    def _heartbeat(self, frames: int, speech: int, in_speech: bool) -> None:
        cap = self._capture
        mic = cap.health() if cap is not None else "no capture"
        pct = 100.0 * speech / frames if frames else 0.0
        logger.info(
            "heartbeat | mic: %s | vad: %d frames, %.0f%% speech, in_speech=%s | utterance_queue=%d | transcribed=%d kept=%d dropped=%d",
            mic, frames, pct, in_speech, self.utterance_queue.qsize(),
            self.stats["utterances_transcribed"], self.stats["utterances_kept"], self.stats["utterances_dropped"],
        )
        if self._speech.too_quiet and not self._level_warned:
            self._level_warned = True
            logger.warning(
                "Input is very quiet (loudest speech %s; VAD hears %.0f%% speech). Much of what you say "
                "may be discarded before it reaches Whisper. Raise the capture gain "
                "(alsamixer -c 0, F4) or set audio.input_gain in config.toml, then check with "
                "bash go.sh mic.",
                _dbfs(self._speech.loudest), pct,
            )
        self._ui(worried=self._speech.too_quiet)
        if cap is not None and not self._loss_warned and cap.audio_loss() > LOSS_WARN and cap.samples_received > 10 * cap.samplerate:
            self._loss_warned = True
            logger.warning("The mic is losing %.0f%% of the audio (input overflows); words will be cut. "
                           "Raise audio.latency in config.toml (currently %s).", 100 * cap.audio_loss(), self.config.latency)
        if cap is not None and cap.blocks_received and cap.last_block_at and time.monotonic() - cap.last_block_at > 2.0:
            logger.warning("No audio block for %.1fs: the mic stream appears stalled (USB disconnect? device grabbed by another app?)",
                           time.monotonic() - cap.last_block_at)
        if cap is not None and cap.blocks_received and "SILENT?" in mic:
            logger.warning("Mic level is ~0 for the last %.0fs: check the mic is unmuted / the right device is selected.", HEARTBEAT_S)

    def _transcription_worker(self) -> None:
        try:
            logger.debug("Transcription worker started")
            while not (self.audio_done.is_set() and self.utterance_queue.empty()):
                try:
                    utt, offset = self.utterance_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    self._handle_utterance(utt, offset)
                except Exception as exc:  # noqa: BLE001 - one bad utterance must not end the session
                    self.stats["errors"] += 1
                    logger.exception("Transcription error (utterance skipped): %s", exc)
            logger.debug("Transcription worker finished")
        except BaseException as exc:  # noqa: BLE001
            self._fail("transcription worker", exc)

    # --- microphone run --------------------------------------------------------
    def run_microphone(self) -> int:
        console = self.console
        if console is not None:
            console.banner(f"Emote  v{__version__}", [
                f"model {self.config.model_name} · mic matching {self.config.microphone!r}",
                f"session {self.session.name}"
                + (f" · {len(self.vocabulary)} vocabulary words" if self.vocabulary else ""),
            ])
            console.update(detail=self.config.model_name)
            console.attach_logging(logging_utils.CONSOLE_HANDLER)
            console.start()
        try:
            return self._run_microphone()
        finally:
            if console is not None:
                console.stop()

    def _run_microphone(self) -> int:
        logger.info("=== microphone run: session %s ===", self.session.name)
        free_mb = available_mb()
        if free_mb < LOW_MEMORY_MB:
            # The Orin's GPU shares this memory. At 1.6-2.0 GB available (VSCodium alone held
            # 2.1 GB, 2026-10-10) CUDA allocations failed and had to be retried.
            logger.warning("Only %.1f GB of memory is free, and the GPU shares it. Biggest users: %s. "
                           "Close what you do not need (an editor or browser) for a smooth session.",
                           free_mb / 1024, describe_hogs())
        self.backend.load()
        self._ui(detail="and stretching")
        self.backend.warm_up()
        devices = list_input_devices()
        device = resolve_device(self.config.microphone, devices)
        if self.config.capture_sample_rate:
            input_sr = self.config.capture_sample_rate
            logger.info("Capture sample rate forced by config: %d Hz", input_sr)
        else:
            input_sr = pick_samplerate(device, self.config.target_sample_rate)
        blocksize = 0 if self.config.block_duration_ms <= 0 else max(1, int(input_sr * self.config.block_duration_ms / 1000))
        resampler = Resampler(input_sr, self.config.target_sample_rate)
        dev_name = next((d["name"] for d in devices if d["index"] == device), "system default")
        logger.info(
            "Audio plan: device=[%s] %s, capture=%d Hz -> %d Hz (%s), block=%d samples (%s), VAD frame=%d ms, queue max=%d blocks (%.1fs)",
            device, dev_name, input_sr, self.config.target_sample_rate, resampler.backend, blocksize,
            "PortAudio chooses" if blocksize == 0 else f"{self.config.block_duration_ms} ms",
            self.config.frame_duration_ms, self.config.input_queue_max_blocks,
            self.config.input_queue_max_blocks * self.config.block_duration_ms / 1000,
        )
        logger.info(
            "Whisper plan: model=%s device=%s compute=%s beam=%d word_timestamps=%s vad_filter=%s",
            self.backend.active_model, self.backend.active_device, self.backend.active_compute,
            self.config.beam_size, self.config.word_timestamps, self.config.vad_filter,
        )
        if self.console is not None:
            self.console.note(f"model {self.backend.active_model} on {self.backend.active_device} ({self.backend.active_compute})"
                              f" · mic [{device}] {dev_name} at {input_sr} Hz")
        wav_rate = input_sr if self.config.session_wav_source == "device" else self.config.target_sample_rate
        if self.config.save_session_wav:
            self.session.start_wav(wav_rate)
        self.session.write_metadata(
            mode="microphone",
            microphone={"selector": self.config.microphone, "index": device, "name": dev_name},
            capture_sample_rate=input_sr,
            input_gain=self.config.input_gain,
            session_wav_sample_rate=wav_rate if self.config.save_session_wav else None,
            model={"name": self.backend.active_model, "device": self.backend.active_device, "compute_type": self.backend.active_compute},
            config=_config_summary(self.config),
            version=__version__,
            run_log=str(logging_utils.RUN_LOG) if logging_utils.RUN_LOG else None,
        )

        capture = AudioCapture(self.audio_queue, device, input_sr, blocksize, self.config.latency_value)
        self._capture = capture
        audio_thread = threading.Thread(target=self._audio_worker, args=(input_sr,), name="audio", daemon=True)
        trans_thread = threading.Thread(target=self._transcription_worker, name="transcribe", daemon=True)
        interactive = sys.stdin is not None and sys.stdin.isatty()
        logger.debug("stdin interactive=%s", interactive)

        def _on_signal(signum, _frame):  # noqa: ANN001
            logger.info("Signal %s received, stopping...", signal.Signals(signum).name)
            self.stop_event.set()

        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGINT, _on_signal)

        audio_thread.start()
        trans_thread.start()
        control_thread = threading.Thread(target=self._control_loop, name="control", daemon=True) if interactive else None
        if control_thread is not None:
            control_thread.start()
        t_start = time.monotonic()
        try:
            capture.start()
            hint = "Press Esc or Enter to stop (q and Ctrl+C also work)." if interactive else "Stop with Ctrl+C or SIGTERM."
            logger.info("Session %s running. %s", self.session.name, hint)
            self._ui(phase="running", started=time.monotonic())
            if self.console is not None:
                self.console.note(hint)
            elif self.quiet:
                print(f"Listening. {hint} Transcript:", flush=True)
            while not self.stop_event.is_set():
                time.sleep(0.2)
        except AudioDeviceError as exc:
            self._fail("microphone", exc)
        finally:
            logger.info("Stopping after %.1fs ...", time.monotonic() - t_start)
            self.stop_event.set()
            self._ui(phase="stopping", speaking=False)
            capture.stop()
            audio_thread.join(timeout=5.0)
            if audio_thread.is_alive():
                logger.warning("Audio worker did not stop within 5s")
            self._drain_transcriptions(trans_thread)
            if control_thread is not None:
                control_thread.join(timeout=1.0)
            self._finish()
        return 1 if self.failed else 0

    def _drain_transcriptions(self, trans_thread: threading.Thread) -> None:
        pending = self.utterance_queue.qsize()
        if pending:
            logger.info("Finishing %d queued utterance(s)...", pending)
        trans_thread.join(timeout=60.0 + 10.0 * pending)
        if trans_thread.is_alive():
            logger.warning("Transcription worker did not finish in time; some utterances were not saved.")

    def _control_loop(self) -> None:
        if self._read_stop_key():
            logger.info("Stop requested from console.")
        self.stop_event.set()

    def _read_stop_key(self) -> bool:
        """Block until the user asks to stop. True if they did, False if stdin died.

        On a terminal this reads single keypresses so Esc alone stops the run;
        line mode (a pipe, nohup) still works but needs Enter.
        """
        try:
            import termios
            import tty

            fd = sys.stdin.fileno()
            interactive = sys.stdin.isatty()
        except Exception:  # noqa: BLE001 - no stdin at all
            interactive = False

        if not interactive:
            try:
                for line in sys.stdin:
                    if line.strip().lower() in ("", "q", "quit", "stop", "exit"):
                        return True
                    logger.info("Unknown command %r. Press Enter or type q to stop.", line.strip())
            except Exception as exc:  # noqa: BLE001 - stdin closed
                logger.debug("control loop ended: %s", exc)
            return False

        saved = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self.stop_event.is_set():
                if not select.select([fd], [], [], 0.2)[0]:
                    continue
                key = os.read(fd, 1)
                if not key:
                    return False
                if key == b"\x1b":
                    # Esc alone stops; an escape sequence (arrow key, mouse) does not.
                    if select.select([fd], [], [], 0.05)[0]:
                        os.read(fd, 16)
                        continue
                    return True
                if key in (b"\r", b"\n", b"q", b"Q", b"\x03"):
                    return True
        except Exception as exc:  # noqa: BLE001 - terminal went away
            logger.debug("control loop ended: %s", exc)
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            except Exception:  # noqa: BLE001
                pass
        return False

    def _finish(self) -> None:
        self._log_summary()
        if self.discard_if_empty():
            return
        self.session.close()
        self._copy_run_log()
        self._farewell()

    def discard_if_empty(self) -> bool:
        """Remove the session folder when nothing was captured. True if it was removed."""
        if not self.session.dir.exists() or not self.session.is_empty():
            return False
        self.session.close()
        logger.info("Nothing was captured; removing empty session folder %s", self.session.dir)
        _rmtree(self.session.dir)
        if self.console is not None:
            self.console.note("Nothing was captured, so no session folder was kept.")
        return True

    def _farewell(self) -> None:
        console = self.console
        if console is None:
            return
        s = self.stats
        console.note(f"Saved {self.session.dir}")
        console.note(f"{s['utterances_kept']} heard, {s['utterances_dropped']} dropped, "
                     f"{s['transcribe_seconds']:.1f}s of transcribing for {s['audio_seconds']:.1f}s of audio")
        builder = self._builder
        if builder is not None and builder.discarded_short:
            console.note(f"{builder.discarded_short} short burst(s) too brief to transcribe (min_speech_ms={self.params.min_speech_ms})")
        if self._speech.too_quiet:
            gain = suggest_gain(self._level_peak, self._speech.loudest / self.config.input_gain)
            console.alert(logging.WARNING, f"The mic was too quiet (loudest speech {_dbfs(self._speech.loudest)}). Raise the capture gain "
                          f"(alsamixer -c 0), or set audio.input_gain = {gain} in config.toml. Check with: bash go.sh mic")

    def _copy_run_log(self) -> None:
        run_log = logging_utils.RUN_LOG
        if run_log and run_log.exists():
            for h in logging.getLogger().handlers:
                h.flush()
            try:
                shutil.copy2(run_log, self.session.dir / "run.log")
                logger.debug("Run log copied to %s", self.session.dir / "run.log")
            except OSError as exc:
                logger.warning("Could not copy run log into session: %s", exc)

    def _log_summary(self) -> None:
        s = self.stats
        frames = s["frames"]
        logger.info("=== run summary ===")
        logger.info("  VAD frames         : %d (%.1fs), speech %.1f%%", frames, frames * self.params.frame_duration_ms / 1000, 100.0 * s["speech_frames"] / frames if frames else 0.0)
        logger.info("  utterances         : detected=%d transcribed=%d kept=%d dropped=%d", s["utterances_detected"], s["utterances_transcribed"], s["utterances_kept"], s["utterances_dropped"])
        logger.info("  transcription time : %.1fs for %.1fs of audio (RTF %.2f), max queue depth %d",
                    s["transcribe_seconds"], s["audio_seconds"], s["transcribe_seconds"] / s["audio_seconds"] if s["audio_seconds"] else 0.0, s["max_utterance_queue"])
        logger.info("  errors             : %d", s["errors"])
        builder = self._builder
        if builder is not None and builder.discarded_short:
            logger.info(
                "  discarded          : %d short speech burst(s), %.1fs total, below min_speech_ms=%d",
                builder.discarded_short, builder.discarded_speech_ms / 1000, self.params.min_speech_ms,
            )
        if self._level_peak > 0.0:
            gain_hint = ""
            if self._speech.too_quiet:
                suggested = suggest_gain(self._level_peak, self._speech.loudest / self.config.input_gain)
                gain_hint = (f"  TOO QUIET -- raise the capture gain, or set audio.input_gain = {suggested} "
                             f"in config.toml (current {self.config.input_gain:g})")
            elif self._level_peak * self.config.input_gain > 0.95:
                gain_hint = "  CLIPPING -- lower the capture gain or audio.input_gain"
            logger.info("  input level        : peak %.4f (%s), loudest rms %.4f (%s), loudest speech %s over %d speech frames%s",
                        self._level_peak, _dbfs(self._level_peak),
                        self._level_rms, _dbfs(self._level_rms),
                        _dbfs(self._speech.loudest), self._speech.frames, gain_hint)
        if self._capture is not None:
            logger.info("  mic                : %s", self._capture.health())
        logger.info("  model used         : %s on %s (%s)", self.backend.active_model, self.backend.active_device, self.backend.active_compute)
        logger.info("=== end summary ===")

    # --- file run ----------------------------------------------------------------
    def run_wav_file(self, wav_path: Path) -> int:
        logger.info("=== file run: %s -> session %s ===", wav_path, self.session.name)
        samples, file_sr = _read_wav_mono(wav_path)
        logger.info("Read %s: %d samples, %.2fs @ %d Hz, peak=%.3f", wav_path.name, samples.size, samples.size / file_sr, file_sr, float(np.max(np.abs(samples))) if samples.size else 0.0)
        self.backend.load()
        resampler = Resampler(file_sr, self.config.target_sample_rate)
        logger.info("Resampling %d -> %d Hz (%s)", file_sr, self.config.target_sample_rate, resampler.backend)
        gate, builder = self._make_gate_and_builder()
        pcm = bytearray(float_to_pcm16(np.concatenate([resampler.process(samples), resampler.flush()])))
        frame_bytes = self.params.frame_bytes
        utterances: list[Utterance] = []
        while len(pcm) >= frame_bytes:
            frame = bytes(pcm[:frame_bytes])
            del pcm[:frame_bytes]
            is_speech = gate.is_speech(frame)
            self.stats["frames"] += 1
            self.stats["speech_frames"] += is_speech
            utt = builder.add_frame(frame, is_speech)
            if utt is not None:
                utterances.append(utt)
        tail = builder.flush()
        if tail is not None:
            utterances.append(tail)
        self.stats["utterances_detected"] = len(utterances)
        logger.info("VAD found %d utterance(s): %s", len(utterances),
                    ", ".join(f"{u.start_s(self.params):.1f}-{u.end_s(self.params):.1f}s" for u in utterances[:20]) + (" ..." if len(utterances) > 20 else ""))
        self.session.write_metadata(
            mode="wav_file",
            source_wav=str(wav_path.resolve()),
            source_sample_rate=file_sr,
            model={"name": self.backend.active_model, "device": self.backend.active_device, "compute_type": self.backend.active_compute},
            config=_config_summary(self.config),
            version=__version__,
            run_log=str(logging_utils.RUN_LOG) if logging_utils.RUN_LOG else None,
        )
        delay_s = resampler.delay_samples / file_sr
        for utt in utterances:
            try:
                self._handle_utterance(utt, delay_s)
            except Exception as exc:  # noqa: BLE001
                self.stats["errors"] += 1
                logger.exception("Transcription error (utterance skipped): %s", exc)
        self._log_summary()
        self.session.close()
        self._copy_run_log()
        return 0


def _rmtree(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        child.unlink() if child.is_file() else child.rmdir()
    path.rmdir()


def _config_summary(config: Config) -> dict:
    return {f.name: getattr(config, f.name) for f in dataclasses.fields(config) if f.name != "base_dir"}


def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        logger.debug("soundfile info: %s", info)
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if data.shape[1] > 1:
            logger.info("Mixing %d channels down to mono", data.shape[1])
        return data.mean(axis=1).astype(np.float32), int(sr)
    except ImportError:
        logger.warning("soundfile not installed; falling back to the wave module (16-bit PCM WAV only)")
        with wave.open(str(path), "rb") as wf:
            sr, n_ch, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        if width != 2:
            raise ValueError("Without soundfile, only 16-bit PCM WAV is supported.")
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if n_ch > 1:
            data = data.reshape(-1, n_ch).mean(axis=1)
        return data, sr


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="transcriber", description="Local English speech-to-text (Emote).")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="TOML config (default: config.toml)")
    p.add_argument("--wav", default=None, help="Transcribe this audio file instead of the microphone.")
    p.add_argument("--name", default=None, help="Optional label appended to the session folder name.")
    p.add_argument("--list-devices", action="store_true", help="List input devices and exit.")
    p.add_argument("--doctor", action="store_true", help="Print an environment health report and exit.")
    p.add_argument("--mic-check", nargs="?", type=float, const=10.0, default=None, metavar="SECONDS",
                   help="Record for a few seconds (default 10) and show a live level meter plus what the VAD hears, then exit.")
    p.add_argument("--log", action="store_true", help="Show the INFO log on the console instead of the mascot screen.")
    p.add_argument("--verbose", "-v", action="store_true", help="Show DEBUG log lines on the console instead of the mascot screen (they are always in logs/run_*.log).")
    p.add_argument("--quiet", "-q", action="store_true", help="Print only the transcript; no log lines on the console (logs/run_*.log is unchanged).")
    p.add_argument("--version", action="version", version=f"transcriber {__version__}")
    return p.parse_args(argv)


def console_mode(args: argparse.Namespace) -> str:
    """ui (mascot screen, the default for microphone runs) | log | verbose | quiet."""
    if args.quiet:
        return "quiet"
    if args.verbose:
        return "verbose"
    if args.log or args.wav:
        return "log"
    return "ui"


CONSOLE_LEVELS = {"quiet": None, "verbose": logging.DEBUG, "log": logging.INFO, "ui": logging.WARNING}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mode = console_mode(args)
    run_log = setup_logging(PROJECT_DIR / "logs", console_level=CONSOLE_LEVELS[mode])
    logger.info("transcriber %s starting: argv=%s", __version__, sys.argv[1:] if argv is None else argv)
    logger.info("Full DEBUG log for this run: %s", run_log)
    rc = 1
    try:
        rc = _dispatch(args, mode)
    except (AudioDeviceError, FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        logger.debug("traceback:", exc_info=True)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    except Exception as exc:  # noqa: BLE001 - last resort: never die without a logged traceback
        logger.exception("Unexpected error: %s", exc)
    finally:
        logger.info("Exit code %d. Log: %s", rc, run_log)
        cleanup_crash_file()
    return rc


def mic_check(config: Config, seconds: float) -> int:
    """Live level meter plus VAD verdict: the only way to see why speech is missed.

    A silent-looking run is almost always input gain, not the pipeline: WebRTC VAD
    needs a real signal, and a peak near 0.003 (-50 dBFS) reads as silence.
    """
    devices = list_input_devices()
    device = resolve_device(config.microphone, devices)
    input_sr = config.capture_sample_rate or pick_samplerate(device, config.target_sample_rate)
    blocksize = 0 if config.block_duration_ms <= 0 else max(1, int(input_sr * config.block_duration_ms / 1000))
    resampler = Resampler(input_sr, config.target_sample_rate)
    gate = VadGate(config.vad_aggressiveness, config.target_sample_rate, config.frame_duration_ms)
    frame_bytes = gate.frame_bytes

    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=config.input_queue_max_blocks)
    capture = AudioCapture(audio_queue, device, input_sr, blocksize, config.latency_value)
    dev_name = next((d["name"] for d in devices if d["index"] == device), "system default")
    gain_note = f", audio.input_gain = {config.input_gain:g}" if config.input_gain != 1.0 else ""
    print(f"Listening on [{device}] {dev_name} at {input_sr} Hz for {seconds:.0f}s{gain_note}. Speak normally.\n")

    peak_all = 0.0
    speech = SpeechLevel()
    total_frames = 0
    pending = b""
    capture.start()
    t_end = time.monotonic() + seconds
    try:
        while time.monotonic() < t_end:
            try:
                block = audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if config.input_gain != 1.0 and block.size:
                block = np.clip(block * config.input_gain, -1.0, 1.0)
            peak = float(np.max(np.abs(block))) if block.size else 0.0
            peak_all = max(peak_all, peak)
            pending += float_to_pcm16(resampler.process(block))
            while len(pending) >= frame_bytes:
                frame, pending = pending[:frame_bytes], pending[frame_bytes:]
                total_frames += 1
                if gate.is_speech(frame):
                    speech.add(frame)
            bars = min(40, int(peak * 200))
            print(f"\r  level [{'#' * bars:<40}] peak {peak:.4f}", end="", flush=True)
    finally:
        print(flush=True)
        capture.stop()

    speech_pct = 100.0 * speech.frames / total_frames if total_frames else 0.0
    print(f"\n  peak      : {peak_all:.4f} ({_dbfs(peak_all)})")
    print(f"  speech    : {_dbfs(speech.loudest)} at its loudest (aim for about -25 to -10 dBFS)")
    print(f"  VAD       : {speech_pct:.1f}% of {total_frames} frames counted as speech "
          f"(aggressiveness {config.vad_aggressiveness})")
    print(f"  lost audio: {100 * capture.audio_loss():.1f}% ({capture.overflow_count} overflows, audio.latency = {config.latency})")

    if capture.audio_loss() > LOSS_WARN:
        print(f"\n  VERDICT: the mic stream is dropping audio ({100 * capture.audio_loss():.0f}%), so words get cut out.")
        print("  Raise audio.latency in config.toml (for example \"1.0\") and run this check again.")
        return 1
    if not speech.judged:
        print("\n  VERDICT: no speech heard. Speak normally for the whole check, and check the mic is the")
        print("  selected device (bash run.sh --list-devices) and not muted (alsamixer -c 0).")
        return 1
    if speech.too_quiet:
        gain = config.input_gain
        suggested = suggest_gain(peak_all / gain, speech.loudest / gain)
        print("\n  VERDICT: too quiet. The VAD fragments speech this weak into bursts shorter than")
        print(f"  min_speech_ms = {config.min_speech_ms}, and they are discarded before Whisper sees them.")
        print("  Fix it at the source first:")
        print("    alsamixer -c 0                   # F4 for capture, arrow up, Esc")
        print("    amixer -c 0 sset Mic 80% cap     # or 'Capture' on some devices")
        if suggested:
            print(f"  If the hardware gain is already at maximum, set in config.toml:")
            print(f"    [audio]")
            print(f"    input_gain = {suggested}       # software gain, current {config.input_gain:g}")
        return 1
    if peak_all > 0.95:
        print("\n  VERDICT: clipping. Lower the capture gain or move back from the mic.")
        return 1
    print("\n  VERDICT: levels look usable. Run: bash run.sh")
    return 0


def _dbfs(x: float) -> str:
    import math

    return "-inf dBFS" if x <= 0 else f"{20 * math.log10(x):.0f} dBFS"


def _dispatch(args: argparse.Namespace, mode: str) -> int:
    if args.list_devices:
        listing = format_devices(list_input_devices())
        logger.info("Input devices:\n%s", listing)
        print(listing)
        return 0
    if args.doctor:
        from .doctor import run_doctor

        return run_doctor(Path(args.config))
    if args.mic_check is not None:
        return mic_check(load_config(args.config), args.mic_check)
    log_environment(logger)
    config = load_config(args.config)
    log_config(config)
    console = Console() if mode == "ui" and not args.wav else None
    pipeline = Pipeline(config, session_name=args.name or (Path(args.wav).stem if args.wav else None),
                        quiet=args.quiet, console=console)
    try:
        if args.wav:
            wav_path = Path(args.wav)
            if not wav_path.exists():
                logger.error("Audio file not found: %s", wav_path)
                return 1
            return pipeline.run_wav_file(wav_path)
        return pipeline.run_microphone()
    finally:
        pipeline.discard_if_empty()


if __name__ == "__main__":
    raise SystemExit(main())
