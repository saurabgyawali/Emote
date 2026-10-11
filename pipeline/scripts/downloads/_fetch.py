"""Resumable, verified downloads with the standard library only (runs before any pip install).

fetch() writes to <dest>.part, resumes with an HTTP Range request after an interruption,
retries with backoff, and renames only when the size matches, so a re-run never trusts a
half-written file and never re-downloads a complete one.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

UA = {"User-Agent": "emote-pipeline/0.1 (research download script)"}


def data_root(cli_value: str | None = None) -> Path:
    return Path(cli_value or os.environ.get("EMOTE_DATA") or Path.home() / "emote_data").expanduser().resolve()


def use_data_cache(root: Path) -> Path:
    """Point every Hugging Face cache variable at <data root>/pretrained, overriding the machine's.

    A shell profile that exports HF_HOME / HUGGINGFACE_HUB_CACHE for another project (as the
    dev server's ~/.bashrc does) otherwise wins, and models land in that project's cache
    instead of ours -- 3.8 GB did on 2026-10-10 before this existed.
    """
    cache = Path(root) / "pretrained"
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache / "hub")
    os.environ.pop("TRANSFORMERS_CACHE", None)
    return cache


def get_json(url: str, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(url, headers={**UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def fetch(url: str, dest: Path, expected_size: int | None = None, retries: int = 6, quiet: bool = False) -> Path:
    dest = Path(dest)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = dict(UA)
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                if have and r.status != 206:
                    have = 0  # server ignored the range: start over
                total = int(r.headers.get("Content-Length", 0)) + have if r.headers.get("Content-Length") else None
                mode = "ab" if have else "wb"
                done, last = have, time.monotonic()
                with open(part, mode) as fh:
                    while chunk := r.read(1 << 20):
                        fh.write(chunk)
                        done += len(chunk)
                        if not quiet and total and time.monotonic() - last > 10:
                            print(f"    {dest.name}: {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", flush=True)
                            last = time.monotonic()
            size = part.stat().st_size
            if expected_size is not None and size != expected_size:
                raise OSError(f"size {size} != expected {expected_size}")
            part.replace(dest)
            return dest
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError(f"download failed after {retries} attempts: {url}: {exc}") from exc
            wait = min(60, 2**attempt)
            print(f"    retry {attempt}/{retries} for {dest.name} in {wait}s ({exc})", file=sys.stderr, flush=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


def fetch_many(jobs: list[tuple[str, Path]], workers: int = 16, label: str = "files") -> list[str]:
    """Download (url, dest) pairs in parallel; returns the failures instead of stopping at the first."""
    failures: list[str] = []
    todo = [(u, d) for u, d in jobs if not Path(d).exists()]
    print(f"  {len(jobs) - len(todo)} of {len(jobs)} {label} already present; downloading {len(todo)}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, u, d, None, 6, True): d for u, d in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001 - collect and report every failure
                failures.append(f"{futures[fut].name}: {exc}")
            if i % 500 == 0 or i == len(todo):
                print(f"    {i}/{len(todo)} {label}", flush=True)
    return failures


def is_wav(path: Path) -> bool:
    """A real audio file, not an HTML error page or an LFS pointer.

    Accepts AIFF too: two TESS files (YAF_germ_angry, YAF_neat_fear) are AIFF named .wav,
    and rejecting them made the download fail on every re-run (2026-10-10).
    """
    with open(path, "rb") as fh:
        head = fh.read(12)
    return head[:4] == b"RIFF" or (head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"))


def write_marker(directory: Path, info: dict) -> None:
    (directory / "DOWNLOAD_OK.json").write_text(json.dumps({**info, "when": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
