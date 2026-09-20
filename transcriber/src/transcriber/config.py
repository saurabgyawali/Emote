"""Load ``config.toml`` into one typed ``Config`` and validate it.

Every tunable knob lives in the TOML file so nobody edits Python to change
behaviour. Relative paths in the file are resolved against the directory the
config lives in (the project root).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib as _toml
except ModuleNotFoundError:
    import tomli as _toml  # type: ignore[no-redef]

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.toml"


@dataclass
class Config:
    # [audio]
    microphone: str | int | None = "USB"
    capture_sample_rate: int = 0          # 0 = auto (prefer 16 kHz, else device default)
    target_sample_rate: int = 16000
    block_duration_ms: int = 100
    input_queue_max_blocks: int = 100
    latency: str = "high"

    # [vad]
    frame_duration_ms: int = 30
    vad_aggressiveness: int = 2
    speech_start_frames: int = 3
    silence_end_ms: int = 700
    min_speech_ms: int = 300
    max_utterance_s: float = 15.0
    preroll_ms: int = 300

    # [model]
    model_name: str = "small.en"
    model_dir: str = "models"
    device: str = "auto"                  # cuda | cpu | auto
    compute_type: str = "int8_float16"
    language: str = "en"
    beam_size: int = 1
    word_timestamps: bool = True
    vad_filter: bool = True
    require_cuda: bool = False
    cpu_fallback_model: str = "base.en"

    # [filter]
    no_speech_prob_max: float = 0.6
    avg_logprob_min: float = -1.2
    compression_ratio_max: float = 2.4

    # [context]
    context_enabled: bool = True
    context_max_chars: int = 400

    # [output]
    sessions_dir: str = "sessions"
    save_session_wav: bool = True
    session_wav_source: str = "device"    # device | processed
    save_clips: bool = True
    print_transcript: bool = True

    base_dir: Path = field(default_factory=lambda: PROJECT_DIR)

    @property
    def model_path(self) -> Path:
        return self._abs(self.model_dir)

    @property
    def sessions_path(self) -> Path:
        return self._abs(self.sessions_dir)

    def _abs(self, p: str) -> Path:
        path = Path(p).expanduser()
        return path if path.is_absolute() else (self.base_dir / path)

    @property
    def latency_value(self) -> str | float:
        if self.latency in ("high", "low"):
            return self.latency
        return float(self.latency)

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Path | None = None) -> "Config":
        audio = data.get("audio", {})
        vad = data.get("vad", {})
        model = data.get("model", {})
        filt = data.get("filter", {})
        context = data.get("context", {})
        output = data.get("output", {})

        mic = audio.get("microphone", "USB")
        if isinstance(mic, str) and mic.strip() == "":
            mic = None

        cap = audio.get("capture_sample_rate", 0)
        if isinstance(cap, str):
            cap = 0 if cap.strip().lower() in ("", "auto") else int(cap)

        defaults = cls()
        return cls(
            microphone=mic,
            capture_sample_rate=int(cap),
            target_sample_rate=int(audio.get("target_sample_rate", defaults.target_sample_rate)),
            block_duration_ms=int(audio.get("block_duration_ms", defaults.block_duration_ms)),
            input_queue_max_blocks=int(audio.get("input_queue_max_blocks", defaults.input_queue_max_blocks)),
            latency=str(audio.get("latency", defaults.latency)),
            frame_duration_ms=int(vad.get("frame_duration_ms", defaults.frame_duration_ms)),
            vad_aggressiveness=int(vad.get("aggressiveness", defaults.vad_aggressiveness)),
            speech_start_frames=int(vad.get("speech_start_frames", defaults.speech_start_frames)),
            silence_end_ms=int(vad.get("silence_end_ms", defaults.silence_end_ms)),
            min_speech_ms=int(vad.get("min_speech_ms", defaults.min_speech_ms)),
            max_utterance_s=float(vad.get("max_utterance_s", defaults.max_utterance_s)),
            preroll_ms=int(vad.get("preroll_ms", defaults.preroll_ms)),
            model_name=str(model.get("name", defaults.model_name)),
            model_dir=str(model.get("download_dir", defaults.model_dir)),
            device=str(model.get("device", defaults.device)).lower(),
            compute_type=str(model.get("compute_type", defaults.compute_type)),
            language=str(model.get("language", defaults.language)),
            beam_size=int(model.get("beam_size", defaults.beam_size)),
            word_timestamps=bool(model.get("word_timestamps", defaults.word_timestamps)),
            vad_filter=bool(model.get("vad_filter", defaults.vad_filter)),
            require_cuda=bool(model.get("require_cuda", defaults.require_cuda)),
            cpu_fallback_model=str(model.get("cpu_fallback_model", defaults.cpu_fallback_model)),
            no_speech_prob_max=float(filt.get("no_speech_prob_max", defaults.no_speech_prob_max)),
            avg_logprob_min=float(filt.get("avg_logprob_min", defaults.avg_logprob_min)),
            compression_ratio_max=float(filt.get("compression_ratio_max", defaults.compression_ratio_max)),
            context_enabled=bool(context.get("enabled", defaults.context_enabled)),
            context_max_chars=int(context.get("max_chars", defaults.context_max_chars)),
            sessions_dir=str(output.get("sessions_dir", defaults.sessions_dir)),
            save_session_wav=bool(output.get("save_session_wav", defaults.save_session_wav)),
            session_wav_source=str(output.get("session_wav_source", defaults.session_wav_source)),
            save_clips=bool(output.get("save_clips", defaults.save_clips)),
            print_transcript=bool(output.get("print_transcript", defaults.print_transcript)),
            base_dir=base_dir or PROJECT_DIR,
        )

    def validate(self) -> None:
        problems = []
        if self.frame_duration_ms not in (10, 20, 30):
            problems.append(f"vad.frame_duration_ms must be 10, 20 or 30 (got {self.frame_duration_ms})")
        if self.vad_aggressiveness not in (0, 1, 2, 3):
            problems.append(f"vad.aggressiveness must be 0..3 (got {self.vad_aggressiveness})")
        if self.target_sample_rate != 16000:
            problems.append("audio.target_sample_rate must be 16000 (Whisper + WebRTC VAD)")
        if self.block_duration_ms < 0:
            problems.append("audio.block_duration_ms must be >= 0")
        if self.input_queue_max_blocks < 1:
            problems.append("audio.input_queue_max_blocks must be >= 1")
        if self.latency not in ("high", "low"):
            try:
                float(self.latency)
            except (TypeError, ValueError):
                problems.append(f"audio.latency must be 'high', 'low' or seconds (got {self.latency!r})")
        if self.device not in ("cuda", "cpu", "auto"):
            problems.append(f"model.device must be cuda, cpu or auto (got {self.device!r})")
        if self.beam_size < 1:
            problems.append("model.beam_size must be >= 1")
        if self.session_wav_source not in ("device", "processed"):
            problems.append(f"output.session_wav_source must be device or processed (got {self.session_wav_source!r})")
        if self.max_utterance_s <= 0:
            problems.append("vad.max_utterance_s must be > 0")
        if problems:
            raise ValueError("Invalid config:\n  - " + "\n  - ".join(problems))


def load_config(path: str | Path | None = None) -> Config:
    """Parse and validate a TOML config; defaults to the project's config.toml."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("rb") as fh:
        data = _toml.load(fh)
    config = Config.from_dict(data, base_dir=path.resolve().parent)
    config.validate()
    return config


def main(argv: list[str] | None = None) -> int:
    """CLI helper for shell scripts: ``python -m transcriber.config get model.name``."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) >= 2 and argv[0] == "get":
        path = Path(argv[2]) if len(argv) > 2 else DEFAULT_CONFIG_PATH
        with path.open("rb") as fh:
            data = _toml.load(fh)
        node: Any = data
        for part in argv[1].split("."):
            node = node[part]
        print(node)
        return 0
    print("usage: python -m transcriber.config get <section.key> [config.toml]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
