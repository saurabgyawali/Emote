"""Download TESS (Toronto Emotional Speech Set: 2 speakers, 2,800 clips, 281 MB) from Borealis.

Files are listed through the Dataverse API and fetched one by one (the server refuses
HEAD requests but serves GET normally). License CC BY-NC 4.0; cite Dupuis & Pichora-Fuller 2010.

    python download_tess.py [--data-root DIR] [--workers 16] [--force] [--dry-run]
"""

from __future__ import annotations

import argparse

from _fetch import data_root, fetch_many, get_json, is_wav, write_marker

DOI = "doi:10.5683/SP2/E8H2MF"
LIST = f"https://borealisdata.ca/api/datasets/:persistentId/?persistentId={DOI}"
FILE = "https://borealisdata.ca/api/access/datafile/{}"
EXPECTED_WAVS = 2800


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default=None)
    p.add_argument("--workers", type=int, default=4,
                   help="Borealis resets connections under heavier parallelism (16 workers: ~400 retries, 2026-10-10)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out = data_root(args.data_root) / "raw" / "tess"
    if (out / "DOWNLOAD_OK.json").exists() and not args.force:
        print(f"TESS already at {out}")
        return
    files = get_json(LIST)["data"]["latestVersion"]["files"]
    jobs = [(FILE.format(f["dataFile"]["id"]), out / f["dataFile"]["filename"]) for f in files
            if f["dataFile"]["filename"].lower().endswith(".wav")]
    if args.dry_run:
        print(f"would download {len(jobs)} wav files into {out}")
        return
    failures = fetch_many(jobs, workers=args.workers, label="TESS clips")
    bad = [d for _, d in jobs if d.exists() and not is_wav(d)]
    for d in bad:
        d.unlink()
    have = sum(1 for _, d in jobs if d.exists())
    if failures or bad or have != EXPECTED_WAVS:
        raise SystemExit(f"TESS incomplete: {have}/{EXPECTED_WAVS}, {len(failures)} failed, {len(bad)} not audio. Re-running resumes.")
    write_marker(out, {"dataset": "tess", "files": have, "source": DOI})
    print(f"TESS ready: {have} clips in {out}")


if __name__ == "__main__":
    main()
