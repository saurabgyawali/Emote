"""Turn per-frame speech/silence decisions into whole utterances with positions.

Pure Python state machine (no audio libraries) so it is unit-testable anywhere.
Audio is 16-bit little-endian mono PCM ``bytes``; positions are frame indices
since the start of the stream, so callers can convert to seconds.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass
class UtteranceParams:
    sample_rate: int = 16000
    frame_duration_ms: int = 30
    speech_start_frames: int = 3
    silence_end_ms: int = 700
    min_speech_ms: int = 300
    max_utterance_s: float = 15.0
    preroll_ms: int = 300

    def frames_for_ms(self, ms: float) -> int:
        return max(1, math.ceil(ms / self.frame_duration_ms))

    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate * self.frame_duration_ms / 1000)

    @property
    def frame_bytes(self) -> int:
        return self.frame_samples * 2


@dataclass
class Utterance:
    pcm: bytes
    start_frame: int          # inclusive, includes pre-roll
    end_frame: int            # exclusive
    speech_frames: int
    continued: bool = False   # True when this is a chunk of ongoing long speech

    def start_s(self, params: UtteranceParams) -> float:
        return self.start_frame * params.frame_duration_ms / 1000.0

    def end_s(self, params: UtteranceParams) -> float:
        return self.end_frame * params.frame_duration_ms / 1000.0


class UtteranceBuilder:
    def __init__(self, params: UtteranceParams | None = None) -> None:
        self.params = params or UtteranceParams()
        self._silence_end_frames = self.params.frames_for_ms(self.params.silence_end_ms)
        self._min_speech_frames = self.params.frames_for_ms(self.params.min_speech_ms)
        self._max_frames = self.params.frames_for_ms(self.params.max_utterance_s * 1000)
        # The pre-roll must also hold the frames that trigger speech start,
        # otherwise the first syllable is lost whenever preroll < trigger.
        self._preroll_frames = self.params.frames_for_ms(self.params.preroll_ms) + self.params.speech_start_frames

        self.in_speech = False
        self.frame_index = -1          # index of the most recent frame fed in
        self._preroll: deque[bytes] = deque(maxlen=self._preroll_frames)
        self._buffer: list[bytes] = []
        self._buffer_start = 0
        self._speech_frames = 0
        self._trailing_silence = 0
        self._leading_run = 0

    def add_frame(self, frame: bytes, is_speech: bool) -> Utterance | None:
        self.frame_index += 1
        if not self.in_speech:
            return self._handle_idle(frame, is_speech)
        return self._handle_active(frame, is_speech)

    def _handle_idle(self, frame: bytes, is_speech: bool) -> Utterance | None:
        self._preroll.append(frame)
        if is_speech:
            self._leading_run += 1
            if self._leading_run >= self.params.speech_start_frames:
                self._start_utterance()
        else:
            self._leading_run = 0
        return None

    def _handle_active(self, frame: bytes, is_speech: bool) -> Utterance | None:
        self._buffer.append(frame)
        if is_speech:
            self._speech_frames += 1
            self._trailing_silence = 0
        else:
            self._trailing_silence += 1

        if self._trailing_silence >= self._silence_end_frames:
            return self._finalize(reset_to_idle=True)
        if len(self._buffer) >= self._max_frames:
            return self._finalize(reset_to_idle=False)
        return None

    def _start_utterance(self) -> None:
        self.in_speech = True
        self._buffer = list(self._preroll)
        self._buffer_start = self.frame_index - len(self._buffer) + 1
        self._speech_frames = self._leading_run
        self._trailing_silence = 0
        self._preroll.clear()
        self._leading_run = 0

    def _finalize(self, reset_to_idle: bool) -> Utterance | None:
        utt = Utterance(
            pcm=b"".join(self._buffer),
            start_frame=self._buffer_start,
            end_frame=self.frame_index + 1,
            speech_frames=self._speech_frames,
            continued=not reset_to_idle,
        )
        if reset_to_idle:
            self._reset_idle()
        else:
            self._buffer = []
            self._buffer_start = self.frame_index + 1
            self._speech_frames = 0
            self._trailing_silence = 0
        if utt.speech_frames < self._min_speech_frames:
            return None
        return utt

    def _reset_idle(self) -> None:
        self.in_speech = False
        self._buffer = []
        self._speech_frames = 0
        self._trailing_silence = 0
        self._leading_run = 0
        self._preroll.clear()

    def flush(self) -> Utterance | None:
        """Emit whatever speech is buffered (shutdown / end of file)."""
        if not self.in_speech or not self._buffer:
            self._reset_idle()
            return None
        utt = Utterance(
            pcm=b"".join(self._buffer),
            start_frame=self._buffer_start,
            end_frame=self.frame_index + 1,
            speech_frames=self._speech_frames,
        )
        self._reset_idle()
        if utt.speech_frames < self._min_speech_frames:
            return None
        return utt
