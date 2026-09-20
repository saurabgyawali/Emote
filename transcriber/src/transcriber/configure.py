"""Interactive wizard that edits config.toml IN PLACE (comments preserved).

    python -m transcriber.configure            # asks mic / device / model
    bash install.sh --interactive              # same, at the end of install
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH

MODELS = ("tiny.en", "base.en", "small.en", "medium.en", "distil-large-v3")


def set_toml_value(text: str, section: str, key: str, value: str | int | float | bool) -> str:
    """Replace ``key = ...`` inside ``[section]``; keep the trailing comment."""
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    elif isinstance(value, (int, float)):
        rendered = str(value)
    else:
        rendered = '"' + str(value).replace('"', '\\"') + '"'
    lines = text.splitlines(keepends=True)
    in_section = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("["):
            in_section = stripped == f"[{section}]"
            continue
        if not in_section:
            continue
        m = re.match(r"^(\s*)" + re.escape(key) + r"\s*=\s*([^#]*?)(\s*#.*)?\r?\n?$", line)
        if m:
            indent, comment = m.group(1), m.group(3) or ""
            lines[i] = f"{indent}{key} = {rendered}{comment}\n"
            return "".join(lines)
    raise KeyError(f"{section}.{key} not found in config")


def _ask(prompt: str, default: str) -> str:
    try:
        answer = input(f"{prompt} [{default}]: ").strip()
    except EOFError:
        return default
    return answer or default


def run_wizard(config_path: Path = DEFAULT_CONFIG_PATH) -> int:
    text = config_path.read_text(encoding="utf-8")
    print(f"\nEditing {config_path}\n")

    try:
        from .audio_capture import format_devices, list_input_devices

        print("Input devices:\n" + format_devices(list_input_devices()) + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"(could not list audio devices: {exc})\n")
    current_mic = re.search(r'^\s*microphone\s*=\s*"?([^"#\n]*)"?', text, re.M)
    mic = _ask("Microphone (name substring like USB, or index; blank = system default)", current_mic.group(1).strip() if current_mic else "USB")
    text = set_toml_value(text, "audio", "microphone", int(mic) if mic.isdigit() else mic)

    device = _ask("Whisper device: auto | cuda | cpu", "auto").lower()
    if device not in ("auto", "cuda", "cpu"):
        device = "auto"
    text = set_toml_value(text, "model", "device", device)

    print("Models: " + ", ".join(MODELS))
    model = _ask("Model", "small.en")
    text = set_toml_value(text, "model", "name", model)

    if model.startswith(("medium", "large", "distil")):
        print("Note: heavier model chosen; on CPU it will fall back to cpu_fallback_model.")

    config_path.write_text(text, encoding="utf-8")
    print(f"\nSaved {config_path}. Run: bash run.sh")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else DEFAULT_CONFIG_PATH
    return run_wizard(path)


if __name__ == "__main__":
    raise SystemExit(main())
