"""Experiment configuration: configs/base.yaml, then an experiment file, then dotted CLI overrides.

    cfg = load_config("configs/experiments/audio_text.yaml", ["train.epochs=3", "model.audio.encoder=microsoft/wavlm-large"])

Experiment files name their parent with `inherit:` (a path relative to themselves);
nested mappings merge key by key, everything else replaces.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _parse_value(text: str) -> Any:
    value = yaml.safe_load(text)
    # YAML 1.1 reads "1e-3" (no decimal point) as a string; learning rates are written that way.
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like section.key=value")
        dotted, raw = item.split("=", 1)
        node = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise KeyError(f"override {dotted!r}: no section {part!r}")
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"override {dotted!r}: unknown key {parts[-1]!r} (known: {sorted(node)})")
        node[parts[-1]] = _parse_value(raw)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("inherit", None)
    cfg = _merge(load_config(path.parent / parent), raw) if parent else raw
    # The experiment is named after the file actually loaded, never after the file it inherits from.
    cfg["experiment"] = raw.get("experiment", path.stem)
    return apply_overrides(cfg, overrides or [])


def fingerprint(cfg: dict) -> str:
    """Short hash of everything except the seed, so seeds of one experiment share it."""
    stripped = {k: v for k, v in cfg.items() if k != "seed"}
    return hashlib.sha1(json.dumps(stripped, sort_keys=True, default=str).encode()).hexdigest()[:12]
