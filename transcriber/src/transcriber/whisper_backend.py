"""faster-whisper wrapper: load once (CUDA with safe CPU fallback), transcribe utterances.

Returns rich results (words, confidences) so downstream emotion analysis can use
speaking rate and pauses, and so hallucinated segments can be filtered.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
OOM_ATTEMPTS = 4
OOM_RETRY_S = 1.0      # waits 1, 2, 3 s: the kernel frees cached pages within seconds
LOW_MEMORY_MB = 3000   # below this at start, CUDA allocations on the Orin start failing


@dataclass
class Word:
    word: str
    start: float
    end: float
    probability: float


@dataclass
class TranscriptResult:
    text: str
    words: list[Word] = field(default_factory=list)
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 0.0
    latency_s: float = 0.0
    segment_count: int = 0


def _memory_summary() -> str:
    try:
        fields = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            fields[key] = int(rest.split()[0]) // 1024
        return f"{fields.get('MemAvailable', 0)} MB available of {fields.get('MemTotal', 0)} MB"
    except Exception:  # noqa: BLE001
        return "memory unknown"


def available_mb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 0


def memory_hogs(n: int = 3, proc: Path = Path("/proc")) -> list[tuple[str, int]]:
    """Biggest programs by resident memory, summed per name (VSCodium runs as 7 processes)."""
    totals: dict[str, int] = {}
    for status in proc.glob("[0-9]*/status"):
        try:
            fields = dict(line.split(":", 1) for line in status.read_text().splitlines() if ":" in line)
            rss_kb = int(fields.get("VmRSS", "0 kB").split()[0])
        except (OSError, ValueError):
            continue
        name = fields.get("Name", "?").strip()
        totals[name] = totals.get(name, 0) + rss_kb // 1024
    return sorted(totals.items(), key=lambda kv: -kv[1])[:n]


def describe_hogs() -> str:
    return ", ".join(f"{name} {mb / 1024:.1f} GB" for name, mb in memory_hogs())


def _looks_like_cuda_problem(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(k in text for k in ("cuda", "cublas", "cudnn", "gpu"))


def _is_missing_model_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(k in text for k in ("cached snapshot", "local_files_only", "no such file", "could not find", "cannot find"))


def supported_compute_types(device: str) -> set[str]:
    try:
        import ctranslate2

        return set(ctranslate2.get_supported_compute_types(device))
    except Exception:  # noqa: BLE001 - ctranslate2 missing or too old to ask
        return set()


def _cpu_compute_type(requested: str, supported: set[str] | None = None) -> str:
    """A CPU compute type this build can actually run.

    The source-built CTranslate2 on the Jetson has no efficient int8 backend
    (built without Ruy/oneDNN), so the old unconditional "int8" made the CPU
    fallback die with 'Requested int8 compute type, but the target device or
    backend do not support efficient int8 computation' -- turning a recoverable
    CUDA failure into a failed run (Jetson, 2026-09-20 09:20).
    """
    want = "int8" if requested in ("int8_float16", "float16", "int8_bfloat16", "bfloat16") else requested
    if supported is None:
        supported = supported_compute_types("cpu")
    if not supported or want in supported:
        return want
    for candidate in ("int8", "int8_float32", "float32"):
        if candidate in supported:
            return candidate
    return "float32"


def _is_cpu_friendly(model_name: str) -> bool:
    return model_name.lower().startswith(("tiny", "base", "small"))


def cpu_fallback_choice(configured: str, fallback: str) -> str:
    return configured if _is_cpu_friendly(configured) else fallback


def cuda_device_count() -> int:
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # noqa: BLE001
        return 0


_MODEL_REPOS = {"distil-large-v3": "Systran/faster-distil-whisper-large-v3"}


def model_repo(model_name: str) -> str:
    """Hugging Face repo faster-whisper downloads for a short model name."""
    if "/" in model_name:
        return model_name
    return _MODEL_REPOS.get(model_name, f"Systran/faster-whisper-{model_name}")


def model_is_downloaded(model_name: str, model_dir: Path) -> bool:
    """True if faster-whisper's HF snapshot for ``model_name`` has a model.bin."""
    folder = model_dir / ("models--" + model_repo(model_name).replace("/", "--")) / "snapshots"
    return any(p.is_file() for p in folder.glob("*/model.bin")) if folder.exists() else False


class WhisperBackend:
    def __init__(
        self,
        model_name: str,
        model_dir: str | Path,
        device: str = "auto",
        compute_type: str = "int8_float16",
        language: str = "en",
        beam_size: int = 1,
        word_timestamps: bool = True,
        vad_filter: bool = True,
        require_cuda: bool = False,
        cpu_fallback_model: str = "base.en",
        temperature: float = 0.0,
        hotwords: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.word_timestamps = word_timestamps
        self.vad_filter = vad_filter
        self.require_cuda = require_cuda
        self.cpu_fallback_model = cpu_fallback_model
        self.temperature = temperature
        self.hotwords = hotwords
        self.active_model = model_name
        self.active_device = device
        self.active_compute = compute_type
        self._model = None

    # --- loading -----------------------------------------------------------
    def load(self) -> None:
        if self._model is not None:
            return
        self.model_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self.model_dir.parent / ".hf_cache"))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        import faster_whisper
        from faster_whisper import WhisperModel

        logger.info("faster_whisper %s; model dir %s (exists=%s); HF_HOME=%s HF_HUB_OFFLINE=%s",
                    getattr(faster_whisper, "__version__", "?"), self.model_dir, self.model_dir.exists(),
                    os.environ.get("HF_HOME"), os.environ.get("HF_HUB_OFFLINE"))
        for name in ("tiny.en", "base.en", "small.en", "medium.en", "distil-large-v3"):
            logger.debug("  model %-16s downloaded=%s", name, model_is_downloaded(name, self.model_dir))
        device = self.device
        if device == "auto":
            n = cuda_device_count()
            device = "cuda" if n > 0 else "cpu"
            logger.info("model.device = auto -> %s (ctranslate2 sees %d CUDA device(s))", device, n)
            if device == "cpu" and self.require_cuda:
                raise RuntimeError(
                    "model.require_cuda = true but ctranslate2 reports no CUDA device. "
                    "Run: bash install.sh (it builds CUDA CTranslate2 on Jetson) or set require_cuda = false."
                )
        supported = supported_compute_types(device)
        logger.info("compute types supported on %s: %s", device, sorted(supported) or "unknown")
        if device == "cuda":
            compute = self.compute_type if not supported or self.compute_type in supported else next(
                (c for c in ("int8_float16", "float16", "float32") if c in supported), "float32"
            )
        else:
            compute = _cpu_compute_type(self.compute_type, supported)
        if compute != self.compute_type:
            logger.info("compute_type %s is unavailable on %s; using %s.", self.compute_type, device, compute)

        try:
            self._model = self._build(WhisperModel, self.model_name, device, compute)
            self.active_model, self.active_device, self.active_compute = self.model_name, device, compute
        except Exception as exc:  # noqa: BLE001
            if device == "cuda" and _looks_like_cuda_problem(exc):
                self._load_cpu_fallback(WhisperModel, exc)
            else:
                raise RuntimeError(
                    f"Could not load model '{self.model_name}' from {self.model_dir}. "
                    f"If it is not downloaded yet run: bash install.sh --only-model  "
                    f"(or python scripts/download_model.py --model {self.model_name} --yes). "
                    f"Original error: {exc}"
                ) from exc

    def _load_cpu_fallback(self, whisper_model_cls, exc: Exception) -> None:
        if "out of memory" in str(exc).lower():
            # Orin shares one pool between CPU and GPU: whatever the desktop holds is
            # unavailable to CUDA. Seen with 1.8 GB free of 7.6 GB (Jetson, 2026-09-20).
            logger.error(
                "CUDA ran out of memory loading '%s'. On Orin the GPU shares system RAM: %s",
                self.model_name, _memory_summary(),
            )
            logger.error(
                "Close other applications (browser, IDE, other Python processes), or use a "
                "smaller model.name (base.en, tiny.en). `tegrastats` shows what is holding RAM."
            )
        if self.require_cuda:
            raise RuntimeError(
                f"model.require_cuda = true but CUDA is unusable ({exc}). Re-run bash install.sh "
                "to build CUDA CTranslate2, or set require_cuda = false."
            ) from exc
        cpu_compute = _cpu_compute_type(self.compute_type)
        original = self.model_name
        preferred = cpu_fallback_choice(original, self.cpu_fallback_model)
        candidates: list[str] = []
        for name in (preferred, "base.en", "small.en", "tiny.en", original):
            if name and name not in candidates:
                candidates.append(name)
        last_exc = exc
        for cand in candidates:
            try:
                model = self._build(whisper_model_cls, cand, "cpu", cpu_compute)
            except Exception as build_exc:  # noqa: BLE001
                if _is_missing_model_error(build_exc):
                    logger.info("CPU fallback model '%s' is not downloaded; trying next.", cand)
                    last_exc = build_exc
                    continue
                raise
            self._model = model
            self.active_model, self.active_device, self.active_compute = cand, "cpu", cpu_compute
            logger.warning(
                "CUDA is not usable (%s). Running '%s' on CPU%s.",
                exc, cand, "" if cand == original else f" instead of '{original}'",
            )
            return
        raise RuntimeError(
            "CUDA is unavailable and no CPU fallback model is downloaded "
            f"(tried: {', '.join(candidates)}). Run: python scripts/download_model.py --model base.en --yes"
        ) from last_exc

    def _build(self, whisper_model_cls, model_name: str, device: str, compute_type: str):
        logger.info(
            "Loading faster-whisper '%s' (device=%s, compute_type=%s) from %s",
            model_name, device, compute_type, self.model_dir,
        )
        t0 = time.monotonic()
        model = whisper_model_cls(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=str(self.model_dir),
            local_files_only=True,
        )
        logger.info("Model ready in %.1fs", time.monotonic() - t0)
        try:
            _log_memory_after_load()
        except Exception:  # noqa: BLE001
            pass
        return model

    def warm_up(self) -> None:
        """One throwaway inference so CUDA kernels and weights are resident before the first utterance.

        Runs with the VAD filter off: with it on, faster-whisper strips the silent
        clip to nothing and returns in 0.04 s without ever running the model, which
        made the earlier version of this a no-op.
        """
        if self._model is None:
            return
        t0 = time.monotonic()
        try:
            self.transcribe(np.zeros(16000, dtype=np.float32), vad_filter=False)
            logger.info("Model warm-up finished in %.1fs", time.monotonic() - t0)
        except Exception as exc:  # noqa: BLE001 - warm-up must never stop a run
            logger.warning("Model warm-up failed (harmless, the first utterance will be slower): %s", exc)

    # --- inference ---------------------------------------------------------
    def transcribe(self, audio: np.ndarray, initial_prompt: str | None = None, vad_filter: bool | None = None,
                   use_hotwords: bool = True) -> TranscriptResult:
        if self._model is None:
            self.load()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        vad_filter = self.vad_filter if vad_filter is None else vad_filter
        logger.debug("whisper input: %d samples (%.2fs @16k), beam=%d vad_filter=%s word_timestamps=%s temperature=%s prompt_chars=%d",
                     audio.size, audio.size / 16000, self.beam_size, vad_filter, self.word_timestamps, self.temperature, len(initial_prompt or ""))
        t0 = time.monotonic()
        for attempt in range(1, OOM_ATTEMPTS + 1):
            try:
                # The last attempt drops to greedy decoding: five beams need more memory than one.
                beam = self.beam_size if attempt < OOM_ATTEMPTS else 1
                return self._decode(audio, initial_prompt, vad_filter, beam, self.hotwords if use_hotwords else None, t0)
            except RuntimeError as exc:
                if attempt == OOM_ATTEMPTS or "out of memory" not in str(exc).lower():
                    raise
                # Orin shares RAM between CPU and GPU, so a momentary squeeze from the desktop
                # can fail one allocation (warm-up on 2026-10-10 19:26, 2.1 GB available) and
                # succeed a second later; giving up would lose that utterance's text.
                wait = OOM_RETRY_S * attempt
                logger.warning("CUDA ran out of memory while transcribing (%s; biggest: %s); retrying in %.0fs (attempt %d of %d).",
                               _memory_summary(), describe_hogs(), wait, attempt + 1, OOM_ATTEMPTS)
                gc.collect()
                time.sleep(wait)
        raise AssertionError("unreachable")

    def _decode(self, audio: np.ndarray, initial_prompt: str | None, vad_filter: bool, beam_size: int,
                hotwords: str | None, t0: float) -> TranscriptResult:
        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=self.word_timestamps,
            temperature=self.temperature,
            hotwords=hotwords,
            initial_prompt=initial_prompt or None,
            condition_on_previous_text=False,
        )
        texts: list[str] = []
        words: list[Word] = []
        logprobs: list[float] = []
        no_speech: list[float] = []
        ratios: list[float] = []
        n = 0
        for seg in segments:
            n += 1
            logger.debug("  segment %d: %.2f-%.2fs logprob=%.2f no_speech=%.2f compression=%.2f text=%r",
                         n, seg.start, seg.end, seg.avg_logprob, seg.no_speech_prob, seg.compression_ratio, seg.text)
            texts.append(seg.text)
            logprobs.append(float(seg.avg_logprob))
            no_speech.append(float(seg.no_speech_prob))
            ratios.append(float(seg.compression_ratio))
            for w in seg.words or []:
                words.append(Word(w.word.strip(), float(w.start), float(w.end), float(w.probability)))
        elapsed = time.monotonic() - t0
        logger.debug("whisper done in %.2fs: %d segment(s), %d word(s), language=%s(p=%.2f), duration=%.2fs%s",
                     elapsed, n, len(words), info.language, info.language_probability, info.duration,
                     f", vad kept {info.duration_after_vad:.2f}s" if getattr(info, "duration_after_vad", None) is not None else "")
        if n == 0:
            logger.info("Whisper returned no segments (vad_filter removed everything or audio too short)")
        return TranscriptResult(
            text="".join(texts).strip(),
            words=words,
            avg_logprob=float(np.mean(logprobs)) if logprobs else 0.0,
            no_speech_prob=float(np.min(no_speech)) if no_speech else 1.0,
            compression_ratio=float(np.max(ratios)) if ratios else 0.0,
            latency_s=elapsed,
            segment_count=n,
        )


def looks_hallucinated(
    result: TranscriptResult,
    no_speech_prob_max: float,
    avg_logprob_min: float,
    compression_ratio_max: float,
) -> str | None:
    """Return a reason string if the result should be discarded, else None."""
    if not result.text or not any(ch.isalnum() for ch in result.text):
        return "empty"
    if result.no_speech_prob > no_speech_prob_max:
        return f"no_speech_prob {result.no_speech_prob:.2f} > {no_speech_prob_max}"
    if result.avg_logprob < avg_logprob_min:
        return f"avg_logprob {result.avg_logprob:.2f} < {avg_logprob_min}"
    if result.compression_ratio > compression_ratio_max:
        return f"compression_ratio {result.compression_ratio:.2f} > {compression_ratio_max}"
    return None


def _log_memory_after_load() -> None:
    import resource

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    avail = None
    try:
        for line in open("/proc/meminfo", encoding="utf-8"):
            if line.startswith("MemAvailable"):
                avail = int(line.split()[1]) // 1024
    except OSError:
        pass
    logger.info("Process peak RSS %.0f MB; system MemAvailable %s MB", rss_mb, avail)
