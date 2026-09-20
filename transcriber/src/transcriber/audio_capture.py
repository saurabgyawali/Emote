"""Microphone capture with sounddevice: minimal callback, everything else downstream.

Device selection is by name substring (e.g. "USB") or index. Every step of
device discovery and stream opening is logged so a failed run explains itself:
which devices existed, which matched, which sample rates were tried, and what
PortAudio said.
"""

from __future__ import annotations

import logging
import queue
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

PREFERRED_RATES = (16000, 48000, 44100, 32000, 22050, 8000)


class AudioDeviceError(RuntimeError):
    pass


def _sd():
    try:
        import sounddevice as sd
    except OSError as exc:
        logger.error("sounddevice import failed: %s", exc)
        raise AudioDeviceError(
            f"PortAudio is not available ({exc}). On Ubuntu: sudo apt-get install "
            "libportaudio2 portaudio19-dev, then reinstall sounddevice."
        ) from exc
    return sd


def list_input_devices() -> list[dict[str, Any]]:
    sd = _sd()
    logger.debug("sounddevice %s, PortAudio: %s", sd.__version__, sd.get_portaudio_version()[1])
    all_devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    devices = []
    for idx, dev in enumerate(all_devices):
        entry = {
            "index": idx,
            "name": dev["name"],
            "max_input_channels": int(dev["max_input_channels"]),
            "default_samplerate": int(dev.get("default_samplerate", 0)),
            "hostapi": hostapis[dev.get("hostapi", 0)]["name"],
        }
        logger.debug(
            "device [%2d] %-40s in=%d out=%d default=%d Hz api=%s in_latency=%.3f/%.3f",
            idx, dev["name"], dev["max_input_channels"], dev["max_output_channels"],
            dev.get("default_samplerate", 0), entry["hostapi"],
            dev.get("default_low_input_latency", 0), dev.get("default_high_input_latency", 0),
        )
        if entry["max_input_channels"] > 0:
            devices.append(entry)
    try:
        logger.debug("system default input device index: %s", sd.default.device[0])
    except Exception:  # noqa: BLE001
        pass
    logger.info("Found %d input-capable device(s) out of %d total", len(devices), len(all_devices))
    return devices


def format_devices(devices: list[dict[str, Any]]) -> str:
    if not devices:
        return "  (no input-capable devices found)"
    return "\n".join(
        f"  [{d['index']:>2}] {d['name']}  ({d['max_input_channels']} ch, "
        f"{d['default_samplerate']} Hz, {d['hostapi']})"
        for d in devices
    )


def resolve_device(selector: str | int | None) -> int | None:
    """Map a config selector to a PortAudio input-device index (None = default)."""
    logger.info("Resolving microphone selector %r", selector)
    if selector is None or selector == "":
        logger.info("No microphone configured; using the system default input.")
        return None
    devices = list_input_devices()
    logger.info("Input devices:\n%s", format_devices(devices))
    if isinstance(selector, int) or str(selector).strip().isdigit():
        idx = int(selector)
        if any(d["index"] == idx for d in devices):
            logger.info("Microphone index %d is a valid input device.", idx)
            return idx
        raise AudioDeviceError(
            f"microphone = {idx} is not an input device. Input devices present:\n"
            f"{format_devices(devices)}\nSet audio.microphone in config.toml to one of "
            "these indices or to a name substring like \"USB\"."
        )
    text = str(selector).strip().lower()
    matches = [d for d in devices if text in d["name"].lower()]
    logger.info("%d device(s) match %r: %s", len(matches), selector, [(d["index"], d["name"]) for d in matches])
    if not matches:
        raise AudioDeviceError(
            f"No input device name contains {selector!r}. Input devices present:\n"
            f"{format_devices(devices)}"
        )
    if len(matches) > 1:
        logger.info("Using the first match [%d] %s", matches[0]["index"], matches[0]["name"])
    return matches[0]["index"]


def pick_samplerate(device: int | None, preferred: int) -> int:
    """First rate the device accepts from: preferred, its default, common rates."""
    sd = _sd()
    info = sd.query_devices(device, "input")
    default = int(info.get("default_samplerate", 0) or 0)
    logger.info("Probing sample rates for [%s] %s (device default %d Hz)", device, info.get("name"), default)
    candidates = [preferred, default, *PREFERRED_RATES]
    tried: list[int] = []
    for rate in candidates:
        if rate <= 0 or rate in tried:
            continue
        tried.append(rate)
        try:
            sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype="float32")
            logger.info("  %d Hz: accepted", rate)
            if rate != preferred:
                logger.info("Device does not support %d Hz; capturing at %d Hz (resampling on).", preferred, rate)
            return rate
        except Exception as exc:  # noqa: BLE001 - PortAudio reports unsupported settings this way
            logger.info("  %d Hz: rejected (%s)", rate, exc)
    raise AudioDeviceError(
        f"Device {device} ({info.get('name')}) accepted none of the sample rates {tried}."
    )


class AudioCapture:
    """Open an input stream; the callback only copies blocks onto a queue."""

    def __init__(
        self,
        audio_queue: "queue.Queue[np.ndarray]",
        device: int | None,
        samplerate: int,
        blocksize: int,
        latency: str | float = "high",
    ) -> None:
        self.audio_queue = audio_queue
        self.device = device
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.latency = latency
        self._stream = None
        self.overflow_count = 0
        self.dropped_count = 0
        self.blocks_received = 0
        self.samples_received = 0
        self.first_block_at: float | None = None
        self.last_block_at: float | None = None
        self.peak_recent = 0.0
        self.status_flags: set[str] = set()

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        now = time.monotonic()
        if status:
            self.overflow_count += 1
            self.status_flags.add(str(status).strip())
        if self.first_block_at is None:
            self.first_block_at = now
        self.last_block_at = now
        self.blocks_received += 1
        self.samples_received += frames
        block = indata[:, 0].copy()
        peak = float(np.max(np.abs(block))) if block.size else 0.0
        if peak > self.peak_recent:
            self.peak_recent = peak
        try:
            self.audio_queue.put_nowait(block)
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.audio_queue.put_nowait(block)
            except queue.Full:
                pass
            self.dropped_count += 1

    def start(self) -> None:
        sd = _sd()
        logger.info(
            "Opening mic stream: device=%s samplerate=%d blocksize=%d latency=%s dtype=float32 channels=1",
            self.device, self.samplerate, self.blocksize, self.latency,
        )
        try:
            self._stream = sd.InputStream(
                device=self.device,
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                channels=1,
                dtype="float32",
                latency=self.latency,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001
            logger.error("InputStream failed: %s: %s", type(exc).__name__, exc)
            raise AudioDeviceError(f"Could not open input device {self.device}: {exc}") from exc
        logger.info(
            "Mic stream active: actual samplerate=%.0f Hz, reported input latency=%.1f ms, active=%s",
            self._stream.samplerate, self._stream.latency * 1000.0, self._stream.active,
        )

    def health(self) -> str:
        """One-line status for periodic heartbeats; resets the recent peak."""
        peak = self.peak_recent
        self.peak_recent = 0.0
        age = (time.monotonic() - self.last_block_at) if self.last_block_at else float("nan")
        level = "SILENT?" if peak < 1e-4 else f"peak={peak:.3f}"
        return (
            f"blocks={self.blocks_received} audio={self.samples_received / self.samplerate:.1f}s "
            f"last_block={age * 1000:.0f}ms ago {level} overflow={self.overflow_count} "
            f"dropped={self.dropped_count} queue={self.audio_queue.qsize()}/{self.audio_queue.maxsize}"
        )

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None
        logger.info(
            "Mic stream closed: %d blocks, %.1fs audio, %d overflow(s), %d dropped, status flags=%s",
            self.blocks_received, self.samples_received / self.samplerate, self.overflow_count,
            self.dropped_count, sorted(self.status_flags) or "none",
        )
        if self.blocks_received == 0:
            logger.error(
                "The microphone delivered NO audio blocks. Check the device is not in use by "
                "another program (pulseaudio/pipewire), the USB cable, and audio.microphone in config.toml."
            )
        if self.overflow_count:
            logger.warning(
                "PortAudio reported %d input overflow(s); %d block(s) dropped app-side. "
                "If frequent: set audio.block_duration_ms = 0 or audio.latency = \"high\".",
                self.overflow_count, self.dropped_count,
            )
