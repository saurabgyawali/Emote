"""Environment health report: Python, packages, CUDA, model, microphone, config."""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

from .config import Config, load_config
from .whisper_backend import cuda_device_count, model_is_downloaded


def _pkg_version(name: str) -> str:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        return f"MISSING ({type(exc).__name__}: {exc})"
    return str(getattr(mod, "__version__", "ok"))


def run_doctor(config_path: Path) -> int:
    ok = True
    rows: list[tuple[str, str]] = [
        ("python", f"{sys.version.split()[0]} ({sys.executable})"),
        ("platform", f"{platform.system()} {platform.machine()}"),
        ("jetson", "yes" if Path("/etc/nv_tegra_release").exists() else "no"),
    ]
    for pkg in ("numpy", "scipy", "webrtcvad", "faster_whisper", "ctranslate2", "soundfile"):
        v = _pkg_version(pkg)
        ok &= not v.startswith("MISSING")
        rows.append((pkg, v))
    n_cuda = cuda_device_count()
    rows.append(("ctranslate2 CUDA devices", str(n_cuda)))

    config: Config | None = None
    try:
        config = load_config(config_path)
        rows.append(("config", f"{config_path} (valid)"))
    except Exception as exc:  # noqa: BLE001
        ok = False
        rows.append(("config", f"INVALID: {exc}"))

    if config is not None:
        downloaded = model_is_downloaded(config.model_name, config.model_path)
        ok &= downloaded
        rows.append(("model", f"{config.model_name} in {config.model_path}: {'downloaded' if downloaded else 'MISSING (bash install.sh --only-model)'}"))
        if config.device == "cuda" and n_cuda == 0:
            rows.append(("warning", "model.device = cuda but no CUDA device visible; will fall back to CPU unless require_cuda"))
        try:
            from .audio_capture import format_devices, list_input_devices, resolve_device

            devices = list_input_devices()
            rows.append(("input devices", "\n" + format_devices(devices)))
            try:
                idx = resolve_device(config.microphone)
                rows.append(("microphone", f"{config.microphone!r} -> index {idx if idx is not None else 'default'}"))
            except Exception as exc:  # noqa: BLE001
                ok = False
                rows.append(("microphone", f"PROBLEM: {exc}"))
        except Exception as exc:  # noqa: BLE001
            is_jetson = Path("/etc/nv_tegra_release").exists()
            ok &= not is_jetson
            rows.append(("input devices", f"{'PROBLEM: ' if is_jetson else ''}unavailable ({exc})"))

    width = max(len(k) for k, _ in rows)
    for key, val in rows:
        print(f"{key:<{width}}  {val}")
    print("\nSTATUS:", "OK" if ok else "PROBLEMS FOUND (see above)")
    return 0 if ok else 1
