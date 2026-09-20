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
with its traceback and stops the run cleanly. Stop with Enter / q / Ctrl+C /
SIGTERM; the session folder is always kept.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import queue
import shutil
import signal
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

from . import __version__
from .audio_capture import AudioCapture, AudioDeviceError, format_devices, list_input_devices, pick_samplerate, resolve_device
from .config import DEFAULT_CONFIG_PATH, PROJECT_DIR, Config, load_config
from .context_window import ContextWindow
from . import logging_utils
from .logging_utils import cleanup_crash_file, log_environment, setup_logging
from .resampler import Resampler, float_to_pcm16, pcm16_to_float
from .session import SessionWriter
from .utterance_builder import Utterance, UtteranceBuilder, UtteranceParams
from .vad_gate import VadGate
from .whisper_backend import WhisperBackend, looks_hallucinated

logger = logging.getLogger(__name__)
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
    def __init__(self, config: Config, session_name: str | None = None) -> None:
        self.config = config
        self.params = build_utterance_params(config)
        logger.debug(
            "Utterance params: frame=%d ms (%d samples, %d bytes), start after %d speech frames, "
            "end after %d ms silence, min speech %d ms, chunk at %.1fs, preroll %d ms",
            self.params.frame_duration_ms, self.params.frame_samples, self.params.frame_bytes,
            self.params.speech_start_frames, self.params.silence_end_ms, self.params.min_speech_ms,
            self.params.max_utterance_s, self.params.preroll_ms,
        )
        self.stop_event = threading.Event()
        self.failed = False
        self.audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=config.input_queue_max_blocks)
        self.utterance_queue: "queue.Queue[tuple[Utterance, float]]" = queue.Queue()
        self.context = ContextWindow(config.context_enabled, config.context_max_chars)
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
        result = self.backend.transcribe(audio, initial_prompt=prompt)
        reason = looks_hallucinated(
            result, self.config.no_speech_prob_max, self.config.avg_logprob_min, self.config.compression_ratio_max
        )
        record = self.session.add_utterance(
            start_s=start_s,
            end_s=end_s,
            speech_ms=speech_ms,
            pcm16k=utt.pcm,
            result=result,
            dropped_reason=reason,
            save_clip=self.config.save_clips,
            print_text=self.config.print_transcript,
        )
        if reason is None:
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
        if rtf > 1.0:
            logger.warning(
                "Transcription slower than real time (RTF %.2f). On CPU consider model.name = base.en; "
                "on Jetson make sure ctranslate2 reports a CUDA device (bash run.sh --doctor).", rtf,
            )

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
                if not first_block_logged:
                    first_block_logged = True
                    logger.info("First audio block received: %d samples, peak=%.4f", block.size, float(np.max(np.abs(block))) if block.size else 0.0)
                resampled = resampler.process(block)
                block_pcm = float_to_pcm16(resampled)
                pending += block_pcm
                if self.config.save_session_wav:
                    self.session.write_wav(float_to_pcm16(block) if self.config.session_wav_source == "device" else block_pcm)
                while len(pending) >= frame_bytes:
                    frame = bytes(pending[:frame_bytes])
                    del pending[:frame_bytes]
                    is_speech = gate.is_speech(frame)
                    beat_frames += 1
                    beat_speech += is_speech
                    self.stats["frames"] += 1
                    self.stats["speech_frames"] += is_speech
                    utt = builder.add_frame(frame, is_speech)
                    if builder.in_speech and not was_in_speech:
                        logger.info("speech started at %.2fs", builder.frame_index * self.params.frame_duration_ms / 1000 - delay_s)
                    was_in_speech = builder.in_speech
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
            tail = builder.flush()
            if tail is not None:
                logger.info("Flushing %.2fs of speech still buffered at shutdown", (tail.end_frame - tail.start_frame) * self.params.frame_duration_ms / 1000)
                self.utterance_queue.put((tail, delay_s))
            else:
                logger.debug("Nothing buffered at shutdown (builder.in_speech=%s)", builder.in_speech)
            logger.info("Audio worker finished: %d blocks processed", blocks)
        except BaseException as exc:  # noqa: BLE001
            self._fail("audio worker", exc)

    def _heartbeat(self, frames: int, speech: int, in_speech: bool) -> None:
        cap = self._capture
        mic = cap.health() if cap is not None else "no capture"
        pct = 100.0 * speech / frames if frames else 0.0
        logger.info(
            "heartbeat | mic: %s | vad: %d frames, %.0f%% speech, in_speech=%s | utterance_queue=%d | transcribed=%d kept=%d dropped=%d",
            mic, frames, pct, in_speech, self.utterance_queue.qsize(),
            self.stats["utterances_transcribed"], self.stats["utterances_kept"], self.stats["utterances_dropped"],
        )
        if cap is not None and cap.blocks_received and cap.last_block_at and time.monotonic() - cap.last_block_at > 2.0:
            logger.warning("No audio block for %.1fs: the mic stream appears stalled (USB disconnect? device grabbed by another app?)",
                           time.monotonic() - cap.last_block_at)
        if cap is not None and cap.blocks_received and "SILENT?" in mic:
            logger.warning("Mic level is ~0 for the last %.0fs: check the mic is unmuted / the right device is selected.", HEARTBEAT_S)

    def _transcription_worker(self) -> None:
        try:
            logger.debug("Transcription worker started")
            while not self.stop_event.is_set() or not self.utterance_queue.empty():
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
        logger.info("=== microphone run: session %s ===", self.session.name)
        self.backend.load()
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
        wav_rate = input_sr if self.config.session_wav_source == "device" else self.config.target_sample_rate
        if self.config.save_session_wav:
            self.session.start_wav(wav_rate)
        self.session.write_metadata(
            mode="microphone",
            microphone={"selector": self.config.microphone, "index": device, "name": dev_name},
            capture_sample_rate=input_sr,
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
        if interactive:
            threading.Thread(target=self._control_loop, name="control", daemon=True).start()
        t_start = time.monotonic()
        try:
            capture.start()
            logger.info("Session %s running. %s", self.session.name, "Press Enter or type q to stop; Ctrl+C also works." if interactive else "Stop with Ctrl+C or SIGTERM.")
            while not self.stop_event.is_set():
                time.sleep(0.2)
        except AudioDeviceError as exc:
            self._fail("microphone", exc)
        finally:
            logger.info("Stopping after %.1fs ...", time.monotonic() - t_start)
            self.stop_event.set()
            capture.stop()
            audio_thread.join(timeout=5.0)
            if audio_thread.is_alive():
                logger.warning("Audio worker did not stop within 5s")
            self._drain_transcriptions(trans_thread)
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
        try:
            for line in sys.stdin:
                if line.strip().lower() in ("", "q", "quit", "stop", "exit"):
                    break
                logger.info("Unknown command %r. Press Enter or type q to stop.", line.strip())
        except Exception as exc:  # noqa: BLE001 - stdin closed
            logger.debug("control loop ended: %s", exc)
        logger.info("Stop requested from console.")
        self.stop_event.set()

    def _finish(self) -> None:
        self._log_summary()
        if self.session.is_empty():
            self.session.close()
            logger.info("Nothing was captured; removing empty session folder %s", self.session.dir)
            _rmtree(self.session.dir)
            return
        self.session.close()
        self._copy_run_log()

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
    p.add_argument("--verbose", "-v", action="store_true", help="Show DEBUG log lines on the console too (they are always in logs/run_*.log).")
    p.add_argument("--version", action="version", version=f"transcriber {__version__}")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_log = setup_logging(PROJECT_DIR / "logs", verbose=args.verbose)
    logger.info("transcriber %s starting: argv=%s", __version__, sys.argv[1:] if argv is None else argv)
    logger.info("Full DEBUG log for this run: %s", run_log)
    rc = 1
    try:
        rc = _dispatch(args)
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


def _dispatch(args: argparse.Namespace) -> int:
    if args.list_devices:
        listing = format_devices(list_input_devices())
        logger.info("Input devices:\n%s", listing)
        print(listing)
        return 0
    if args.doctor:
        from .doctor import run_doctor

        return run_doctor(Path(args.config))
    log_environment(logger)
    config = load_config(args.config)
    log_config(config)
    pipeline = Pipeline(config, session_name=args.name or (Path(args.wav).stem if args.wav else None))
    if args.wav:
        wav_path = Path(args.wav)
        if not wav_path.exists():
            logger.error("Audio file not found: %s", wav_path)
            return 1
        return pipeline.run_wav_file(wav_path)
    return pipeline.run_microphone()


if __name__ == "__main__":
    raise SystemExit(main())
