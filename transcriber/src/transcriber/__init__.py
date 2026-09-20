"""Local, offline English speech-to-text for the Emote project.

microphone -> resampler (16 kHz) -> WebRTC VAD -> utterance builder
           -> faster-whisper -> per-session outputs (clips, JSONL, WAV, text)
"""

__version__ = "1.0.0"
