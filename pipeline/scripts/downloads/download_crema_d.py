"""Download CREMA-D (91 actors, 7,442 clips, ~600 MB) plus its crowd-rating and demographics tables.

The audio lives in Git LFS on GitHub (CheyneyComputerScience/CREMA-D); files are fetched
one by one from the LFS media host, listed via the GitHub tree API. If the repository's
LFS bandwidth quota is exhausted the host returns small pointer files instead of audio;
those are detected (no RIFF header) and reported, and the Kaggle mirror is the fallback.
License: Open Database License (ODbL); cite Cao et al. 2014.

    python download_crema_d.py [--data-root DIR] [--workers 16] [--force] [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _fetch import data_root, fetch, fetch_many, get_json, is_wav, write_marker

REPO = "CheyneyComputerScience/CREMA-D"
TREE = f"https://api.github.com/repos/{REPO}/git/trees/master?recursive=1"
MEDIA = f"https://media.githubusercontent.com/media/{REPO}/master/"
RAW = f"https://raw.githubusercontent.com/{REPO}/master/"
TABLES = ["processedResults/summaryTable.csv", "VideoDemographics.csv", "SentenceFilenames.csv"]
EXPECTED_WAVS = 7442


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out = data_root(args.data_root) / "raw" / "crema_d"
    if (out / "DOWNLOAD_OK.json").exists() and not args.force:
        print(f"CREMA-D already at {out}")
        return
    tree = get_json(TREE)
    wavs = [i["path"] for i in tree["tree"] if i["path"].startswith("AudioWAV/") and i["path"].endswith(".wav")]
    if args.dry_run:
        print(f"would download {len(wavs)} wav files + {len(TABLES)} tables into {out}")
        return
    for t in TABLES:
        fetch(RAW + t, out / Path(t).name)
    failures = fetch_many([(MEDIA + w, out / w) for w in wavs], workers=args.workers, label="CREMA-D clips")
    pointers = [w for w in wavs if (out / w).exists() and not is_wav(out / w)]
    for w in pointers:
        (out / w).unlink()
    have = sum(1 for w in wavs if (out / w).exists())
    if failures or pointers or have != EXPECTED_WAVS:
        hint = (" Git LFS quota may be exhausted; download the Kaggle mirror 'ejlok1/cremad' and unpack its "
                f"AudioWAV/ into {out / 'AudioWAV'}, then re-run this script." if pointers else "")
        raise SystemExit(f"CREMA-D incomplete: {have}/{EXPECTED_WAVS} clips, {len(failures)} failed downloads, "
                         f"{len(pointers)} LFS pointer files.{hint} Re-running resumes.")
    write_marker(out, {"dataset": "crema_d", "files": have, "source": f"github.com/{REPO}"})
    print(f"CREMA-D ready: {have} clips in {out}")


if __name__ == "__main__":
    main()
