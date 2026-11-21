"""Download MELD (Friends TV dialogue: ~13.7 k utterances with transcripts, 7 emotions; 10.9 GB) and unpack it.

The archive holds video clips (mp4) plus train/dev/test CSVs with the transcript of every
utterance, which is why MELD is the one open corpus where the text tower can learn
something. Audio is extracted later by build_manifest.py. Source: Hugging Face mirror of
declare-lab/MELD; license GPL-3.0; cite Poria et al. 2019.

Needs ~25 GB free while unpacking (archive + videos); the archives are deleted afterwards.

    python download_meld.py [--data-root DIR] [--keep-archives] [--force] [--dry-run]
"""

from __future__ import annotations

import argparse
import shutil
import tarfile

from _fetch import data_root, fetch, write_marker

URL = "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz"
SIZE = 10878146150
CSV_URL = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/{}_sent_emo.csv"
# The folder names build_manifest.parse_meld matches videos in, one per official split.
VIDEO_DIRS = ("train_splits", "dev_splits_complete", "output_repeated_splits_test")


def _extract(archive, dest) -> None:  # noqa: ANN001
    with tarfile.open(archive) as t:
        t.extractall(dest)


def _video_dirs(out):  # noqa: ANN001, ANN202
    found = {}
    for name in VIDEO_DIRS:
        hit = next((p for p in out.rglob(name) if p.is_dir() and any(p.glob("*.mp4"))), None)
        if hit:
            found[name] = hit
    return found


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default=None)
    p.add_argument("--keep-archives", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out = data_root(args.data_root) / "raw" / "meld"
    if (out / "DOWNLOAD_OK.json").exists() and not args.force:
        print(f"MELD already at {out}")
        return
    if args.dry_run:
        print(f"would download {URL} ({SIZE / 1e9:.1f} GB) and unpack into {out}")
        return
    free = shutil.disk_usage(out.parent if out.parent.exists() else data_root(args.data_root)).free
    if free < 25e9:
        raise SystemExit(f"MELD needs ~25 GB free to unpack; {free / 1e9:.0f} GB available under {out.parent}")
    archive = out / "MELD.Raw.tar.gz"
    # The full archive was first unpacked on ACCRE (2026-11-04): it does not hold the
    # *_sent_emo.csv tables the code assumed, so nothing here may depend on its layout.
    # A re-run after a partial unpack skips whatever is already extracted (and the download).
    inners = [p for p in out.rglob("*.tar.gz") if p != archive]
    if not inners and len(_video_dirs(out)) < len(VIDEO_DIRS):
        fetch(URL, archive, SIZE)
        print("  unpacking MELD.Raw.tar.gz ...", flush=True)
        _extract(archive, out)
        inners = [p for p in out.rglob("*.tar.gz") if p != archive]
    for inner in sorted(inners):
        print(f"  unpacking {inner.relative_to(out)} ...", flush=True)
        _extract(inner, inner.parent)
        if not args.keep_archives:
            inner.unlink()
    for name in ("train", "dev", "test"):
        if not any(out.rglob(f"{name}_sent_emo.csv")):
            print(f"  {name}_sent_emo.csv not in the archive; fetching it from github.com/declare-lab/MELD", flush=True)
            fetch(CSV_URL.format(name), out / "csv" / f"{name}_sent_emo.csv")
    found = _video_dirs(out)
    missing = sorted(set(VIDEO_DIRS) - set(found))
    if missing:
        listing = "\n    ".join(sorted(str(p.relative_to(out)) for p in out.glob("*/*")) or ["(empty)"])
        raise SystemExit(f"MELD: no {missing} video folder(s) after unpacking. Two levels of {out}:\n    {listing}")
    if not args.keep_archives:
        archive.unlink(missing_ok=True)
    videos = len(list(out.rglob("*.mp4")))
    write_marker(out, {"dataset": "meld", "videos": videos, "video_dirs": {k: str(v.relative_to(out)) for k, v in found.items()},
                       "source": URL})
    print(f"MELD ready: {videos} video clips under {out}")


if __name__ == "__main__":
    main()
