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


def _extract(archive, dest) -> None:  # noqa: ANN001
    with tarfile.open(archive) as t:
        t.extractall(dest)


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
    archive = fetch(URL, out / "MELD.Raw.tar.gz", SIZE)
    print("  unpacking MELD.Raw.tar.gz ...", flush=True)
    _extract(archive, out)
    root = next(out.rglob("train_sent_emo.csv")).parent
    for inner in sorted(root.glob("*.tar.gz")):
        print(f"  unpacking {inner.name} ...", flush=True)
        _extract(inner, root)
        if not args.keep_archives:
            inner.unlink()
    if not args.keep_archives:
        archive.unlink()
    videos = len(list(root.rglob("*.mp4")))
    write_marker(out, {"dataset": "meld", "videos": videos, "root": str(root.relative_to(out)), "source": URL})
    print(f"MELD ready: {videos} video clips under {root}")


if __name__ == "__main__":
    main()
