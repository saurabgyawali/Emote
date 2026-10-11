"""Download RAVDESS speech audio (24 actors, 1,440 clips, 208 MB) from Zenodo and unpack it.

License CC BY-NC-SA 4.0 (non-commercial; cite Livingstone & Russo 2018).

    python download_ravdess.py [--data-root DIR] [--force] [--dry-run]
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from _fetch import data_root, fetch, write_marker

URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
SIZE = 208468073
EXPECTED_WAVS = 1440


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out = data_root(args.data_root) / "raw" / "ravdess"
    if (out / "DOWNLOAD_OK.json").exists() and not args.force:
        print(f"RAVDESS already at {out}")
        return
    if args.dry_run:
        print(f"would download {URL} ({SIZE / 1e6:.0f} MB) and unpack into {out}")
        return
    if args.force and out.exists():
        shutil.rmtree(out)
    zpath = fetch(URL, out / "Audio_Speech_Actors_01-24.zip", SIZE)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out)
    wavs = list(out.rglob("*.wav"))
    if len(wavs) != EXPECTED_WAVS:
        raise SystemExit(f"RAVDESS: expected {EXPECTED_WAVS} wav files, found {len(wavs)} in {out}")
    zpath.unlink()
    write_marker(out, {"dataset": "ravdess", "files": len(wavs), "source": URL})
    print(f"RAVDESS ready: {len(wavs)} clips in {out}")


if __name__ == "__main__":
    main()
