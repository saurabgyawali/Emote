"""Thin wrapper around WebRTC VAD: ``gate.is_speech(frame)`` on 16-bit frames."""

from __future__ import annotations


class VadGate:
    def __init__(self, aggressiveness: int = 2, sample_rate: int = 16000, frame_duration_ms: int = 30) -> None:
        if aggressiveness not in (0, 1, 2, 3):
            raise ValueError("aggressiveness must be 0, 1, 2 or 3")
        if frame_duration_ms not in (10, 20, 30):
            raise ValueError("frame_duration_ms must be 10, 20 or 30")
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("WebRTC VAD supports 8/16/32/48 kHz only")
        import webrtcvad  # lazy so the package imports without it

        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_samples = int(sample_rate * frame_duration_ms / 1000)
        self.frame_bytes = self.frame_samples * 2
        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame: bytes) -> bool:
        if len(frame) != self.frame_bytes:
            raise ValueError(f"VAD frame must be {self.frame_bytes} bytes, got {len(frame)}")
        return self._vad.is_speech(frame, self.sample_rate)
