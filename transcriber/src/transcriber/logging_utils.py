"""Logging that leaves a complete trail of every run.

Files (under ``logs/``):
    transcriber.log            INFO+, appended across runs (quick history)
    run_<timestamp>.log        DEBUG, one file per run (the full rundown)
    latest.log                 symlink to the newest run log
    crash_<timestamp>.log      native crash dump (segfault in PortAudio / CTranslate2)

The console level is chosen per mode (WARNING for the default screen, INFO for
--log, DEBUG for --verbose, none for --quiet) and goes to stderr; stdout stays
clean for transcript text. Uncaught exceptions in any thread are logged with tracebacks.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import platform
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

RUN_LOG: Path | None = None
CONSOLE_HANDLER: logging.Handler | None = None
_CRASH_FILE = None


def run_log_dir(log_dir: Path, when: datetime | None = None) -> Path:
    """Per-run logs go in logs/<YYYY-MM-DD>/<HH>/ so a day's runs stay together.

    A flat folder became unreadable fast: 15 run logs from one morning, in three
    unrelated debugging sessions, with nothing but a timestamp to tell them apart.
    """
    when = when or datetime.now()
    return log_dir / when.strftime("%Y-%m-%d") / when.strftime("%H")


def setup_logging(log_dir: Path, console_level: int | None = logging.INFO) -> Path:
    """Configure root logging; returns the per-run DEBUG log path."""
    global RUN_LOG, CONSOLE_HANDLER, _CRASH_FILE
    now = datetime.now()
    log_dir.mkdir(parents=True, exist_ok=True)
    day_dir = run_log_dir(log_dir, now)
    day_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%d_%H%M%S")
    run_log = day_dir / f"run_{stamp}.log"
    RUN_LOG = run_log

    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(threadName)-10s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    CONSOLE_HANDLER = None
    if console_level is not None:
        ch = logging.StreamHandler(stream=sys.stderr)
        ch.setLevel(console_level)
        ch.setFormatter(fmt)
        root.addHandler(ch)
        CONSOLE_HANDLER = ch

    history = logging.FileHandler(log_dir / "transcriber.log", encoding="utf-8")
    history.setLevel(logging.INFO)
    history.setFormatter(fmt)
    root.addHandler(history)

    detail = logging.FileHandler(run_log, encoding="utf-8")
    detail.setLevel(logging.DEBUG)
    detail.setFormatter(fmt)
    root.addHandler(detail)

    # latest.log points at the newest run log; on filesystems without symlink
    # support (network shares) it is a one-line text file with the path instead.
    latest = log_dir / "latest.log"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_log.relative_to(log_dir))
    except OSError:
        try:
            latest.write_text(str(run_log) + "\n", encoding="utf-8")
        except OSError:
            pass

    for noisy in ("huggingface_hub", "urllib3", "numba", "filelock"):
        logging.getLogger(noisy).setLevel(logging.INFO)

    # Native crashes (segfaults inside PortAudio/CTranslate2) bypass Python
    # logging entirely; faulthandler writes the thread stacks to this file.
    crash_path = day_dir / f"crash_{stamp}.log"
    _CRASH_FILE = crash_path.open("w", encoding="utf-8")
    faulthandler.enable(file=_CRASH_FILE, all_threads=True)

    sys.excepthook = _log_uncaught
    threading.excepthook = _log_thread_uncaught
    logging.getLogger(__name__).debug("Logging to %s (DEBUG) and %s (INFO)", run_log, log_dir / "transcriber.log")
    return run_log


def cleanup_crash_file() -> None:
    """Remove the crash file if nothing was written to it (normal exit)."""
    global _CRASH_FILE
    if _CRASH_FILE is None:
        return
    path = Path(_CRASH_FILE.name)
    faulthandler.disable()
    _CRASH_FILE.close()
    _CRASH_FILE = None
    try:
        if path.stat().st_size == 0:
            path.unlink()
    except OSError:
        pass


def _log_uncaught(exc_type, exc, tb) -> None:  # noqa: ANN001
    logging.getLogger("uncaught").critical(
        "Uncaught exception in main thread:\n%s", "".join(traceback.format_exception(exc_type, exc, tb))
    )


def _log_thread_uncaught(args) -> None:  # noqa: ANN001
    logging.getLogger("uncaught").critical(
        "Uncaught exception in thread %s:\n%s",
        args.thread.name if args.thread else "?",
        "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)),
    )


def log_environment(logger: logging.Logger) -> None:
    """One block that tells you what machine / interpreter / packages ran."""
    logger.info("=== environment ===")
    logger.info("python      : %s (%s)", sys.version.split()[0], sys.executable)
    logger.info("platform    : %s %s, %s", platform.system(), platform.machine(), platform.platform())
    tegra = Path("/etc/nv_tegra_release")
    if tegra.exists():
        logger.info("jetson      : %s", tegra.read_text(errors="replace").splitlines()[0].strip())
    model = Path("/proc/device-tree/model")
    if model.exists():
        logger.info("device model: %s", model.read_text(errors="replace").strip("\x00\n"))
    logger.info("cwd         : %s", os.getcwd())
    for var in ("HF_HOME", "HF_HUB_OFFLINE", "PYTHONPATH", "VIRTUAL_ENV", "CUDA_HOME", "LD_LIBRARY_PATH"):
        logger.info("env %-12s: %s", var, os.environ.get(var, "<unset>"))
    try:
        import resource

        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        logger.debug("open-file limit: %s", soft)
    except Exception:  # noqa: BLE001
        pass
    _log_memory(logger)
    for pkg in ("numpy", "scipy", "sounddevice", "webrtcvad", "faster_whisper", "ctranslate2", "soundfile", "onnxruntime"):
        try:
            mod = __import__(pkg)
            logger.info("pkg %-13s: %s", pkg, getattr(mod, "__version__", "present"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("pkg %-13s: NOT IMPORTABLE (%s: %s)", pkg, type(exc).__name__, exc)
    try:
        import ctranslate2

        n = ctranslate2.get_cuda_device_count()
        logger.info("ctranslate2 : CUDA devices=%d, cpu compute types=%s", n, sorted(ctranslate2.get_supported_compute_types("cpu")))
        if n:
            logger.info("ctranslate2 : cuda compute types=%s", sorted(ctranslate2.get_supported_compute_types("cuda")))
    except Exception as exc:  # noqa: BLE001
        logger.warning("ctranslate2 : query failed: %s", exc)
    logger.info("=== end environment ===")


def _log_memory(logger: logging.Logger) -> None:
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemAvailable", "SwapFree"):
                info[key] = int(rest.split()[0]) // 1024
        logger.info("memory      : total=%s MB, available=%s MB, swap free=%s MB", info.get("MemTotal"), info.get("MemAvailable"), info.get("SwapFree"))
    except Exception:  # noqa: BLE001
        pass
    try:
        st = os.statvfs(os.getcwd())
        logger.info("disk free   : %.1f GB at %s", st.f_bavail * st.f_frsize / 1e9, os.getcwd())
    except Exception:  # noqa: BLE001
        pass
