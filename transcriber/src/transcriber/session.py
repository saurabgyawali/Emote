"""Everything one run writes to disk, under ``sessions/<timestamp>/``.

    session.json       metadata (config, device, rates, model, timing)
    session.wav        whole session, mono 16-bit (native mic rate by default)
    utterances.jsonl   one record per utterance: times, text, words, confidences
    transcript.txt     human-readable transcript with timestamps
    clips/utt_0001.wav per-utterance 16 kHz clips (what Whisper heard)

Writes are incremental so a crash or Ctrl+C loses nothing already spoken.
"""

from __future__ import annotations

import json
import logging
import threading
import wave
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .whisper_backend import TranscriptResult

logger = logging.getLogger(__name__)


class SessionWriter:
    def __init__(self, sessions_dir: Path, name: str | None = None) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.name = f"{stamp}_{name}" if name else stamp
        self.dir = Path(sessions_dir) / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir = self.dir / "clips"
        self.started = datetime.now()
        self._lock = threading.Lock()
        self._wav: wave.Wave_write | None = None
        self._wav_rate = 0
        self._wav_frames = 0
        self._count = 0
        self._kept = 0
        self._dropped = 0
        self.metadata: dict[str, Any] = {"session": self.name, "started": self.started.isoformat(timespec="seconds")}
        logger.info("Session folder: %s", self.dir)

    @property
    def next_index(self) -> int:
        return self._count + 1

    # --- metadata ----------------------------------------------------------
    def write_metadata(self, **fields: Any) -> None:
        self.metadata.update(fields)
        tmp = self.dir / "session.json.tmp"
        tmp.write_text(json.dumps(self.metadata, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.dir / "session.json")
        logger.debug("session.json updated (%s)", ", ".join(fields))

    # --- whole-session WAV ---------------------------------------------------
    def start_wav(self, sample_rate: int) -> None:
        self._wav_rate = sample_rate
        self._wav = wave.open(str(self.dir / "session.wav"), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(sample_rate)
        logger.info("Writing session.wav at %d Hz mono 16-bit", sample_rate)

    def write_wav(self, pcm: bytes) -> None:
        if self._wav is not None and pcm:
            self._wav.writeframes(pcm)
            self._wav_frames += len(pcm) // 2

    @property
    def wav_duration_s(self) -> float:
        return self._wav_frames / self._wav_rate if self._wav_rate else 0.0

    # --- utterances ----------------------------------------------------------
    def add_utterance(
        self,
        start_s: float,
        end_s: float,
        speech_ms: int,
        pcm16k: bytes,
        result: TranscriptResult,
        dropped_reason: str | None,
        save_clip: bool,
        print_text: bool,
    ) -> dict[str, Any]:
        with self._lock:
            self._count += 1
            idx = self._count
            clip_rel = None
            if save_clip and pcm16k:
                self.clips_dir.mkdir(exist_ok=True)
                clip_rel = f"clips/utt_{idx:04d}.wav"
                with wave.open(str(self.dir / clip_rel), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(pcm16k)
                logger.debug("wrote %s (%.2fs)", clip_rel, len(pcm16k) / 32000)
            record = {
                "index": idx,
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "duration_s": round(end_s - start_s, 3),
                "speech_ms": speech_ms,
                "text": result.text,
                "kept": dropped_reason is None,
                "dropped_reason": dropped_reason,
                "avg_logprob": round(result.avg_logprob, 4),
                "no_speech_prob": round(result.no_speech_prob, 4),
                "compression_ratio": round(result.compression_ratio, 3),
                "transcribe_latency_s": round(result.latency_s, 3),
                "words": [
                    {
                        "word": w.word,
                        "start_s": round(start_s + w.start, 3),
                        "end_s": round(start_s + w.end, 3),
                        "probability": round(w.probability, 3),
                    }
                    for w in result.words
                ],
                "clip": clip_rel,
                "wall_time": datetime.now().isoformat(timespec="milliseconds"),
            }
            with (self.dir / "utterances.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.debug("utterances.jsonl += #%d (%d words)", idx, len(record["words"]))
            if dropped_reason is None:
                self._kept += 1
                stamp = _fmt_time(start_s)
                with (self.dir / "transcript.txt").open("a", encoding="utf-8") as fh:
                    fh.write(f"[{stamp}] {result.text}\n")
                if print_text:
                    print(f"[{stamp}] {result.text}", flush=True)
            else:
                self._dropped += 1
                logger.info("Dropped utterance %d (%s): %r", idx, dropped_reason, result.text[:60])
            return record

    # --- lifecycle -------------------------------------------------------------
    def close(self) -> None:
        if self._wav is not None:
            self._wav.close()
            self._wav = None
        ended = datetime.now()
        self.write_metadata(
            ended=ended.isoformat(timespec="seconds"),
            wall_seconds=round((ended - self.started).total_seconds(), 1),
            session_wav_seconds=round(self.wav_duration_s, 2),
            utterances_total=self._count,
            utterances_kept=self._kept,
            utterances_dropped=self._dropped,
        )
        audio = f", {self.wav_duration_s:.1f}s session audio" if self._wav_rate else ""
        logger.info("Session saved: %s (%d utterance(s) kept, %d dropped%s)", self.dir, self._kept, self._dropped, audio)

    def is_empty(self) -> bool:
        return self._count == 0 and self._wav_frames == 0


def _fmt_time(seconds: float) -> str:
    m, s = divmod(max(0.0, seconds), 60.0)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def result_to_dict(result: TranscriptResult) -> dict[str, Any]:
    return asdict(result)
