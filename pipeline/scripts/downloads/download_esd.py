"""Download ESD (Emotional Speech Dataset, NUS/SUTD) and keep its 10 English speakers (0011-0020).

350 parallel sentences x 5 emotions (neutral, happy, angry, sad, surprise) per speaker, acted;
17,500 English clips. Internal non-commercial research only. The licence (clause 5) allows use only
after it is signed by an authorised signatory and returned as a dated PDF to haizhou.li@nus.edu.sg,
so this script refuses to run until you say it was sent. Clause 6 asks that models built on it be
made freely available and sent to the licensor; clause 10 fixes the acknowledgement and citation
(Zhou, Sisman, Liu & Li, "Emotional voice conversion: theory, databases and ESD", Speech Communication).

    python download_esd.py --licence-sent [--data-root DIR] [--keep-mandarin] [--force] [--dry-run]

Google Drive throttles popular files ("too many users"). If that happens, download the zip in a
browser, copy it to <data root>/raw/esd/, and re-run: a zip already there is used as is.
"""

from __future__ import annotations

import argparse
import re
import zipfile

from _fetch import data_root, write_marker

DRIVE_ID = "1scuFwqh8s7KIYAfZW1Eu6088ZAK2SI-v"
ENGLISH = {f"{i:04d}" for i in range(11, 21)}
EXPECTED_ENGLISH_WAVS = 17500


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default=None)
    p.add_argument("--licence-sent", action="store_true", help="the signed ESD licence was emailed to haizhou.li@nus.edu.sg")
    p.add_argument("--keep-mandarin", action="store_true")
    p.add_argument("--keep-archives", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out = data_root(args.data_root) / "raw" / "esd"
    if (out / "DOWNLOAD_OK.json").exists() and not args.force:
        print(f"ESD already at {out}")
        return
    if not args.licence_sent:
        raise SystemExit("ESD: the licence (https://github.com/HLTSingapore/Emotional-Speech-Data) must be signed and "
                         "returned as a dated PDF to haizhou.li@nus.edu.sg first; then re-run with --licence-sent")
    if args.dry_run:
        print(f"would download Google Drive file {DRIVE_ID} into {out}, unpack, keep speakers {sorted(ENGLISH)}")
        return
    out.mkdir(parents=True, exist_ok=True)
    archive = next(iter(sorted(out.glob("*.zip"))), None)
    if archive is None:
        import gdown

        archive = out / "ESD.zip"
        part = out / "ESD.zip.part"
        print("  downloading ESD from Google Drive ...", flush=True)
        if gdown.download(id=DRIVE_ID, output=str(part), quiet=False, resume=True) is None:
            raise SystemExit("ESD: Google Drive refused the download (often a quota limit). Download the zip in a browser, "
                             f"copy it to {out}/ and re-run.")
        part.replace(archive)
    print(f"  unpacking {archive.name} ...", flush=True)
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.namelist() if "__MACOSX" not in m and not m.rsplit("/", 1)[-1].startswith("._")]
        if not args.keep_mandarin:
            members = [m for m in members if not _speaker(m) or _speaker(m) in ENGLISH]
        z.extractall(out, members)
    if not args.keep_archives:
        archive.unlink()
    wavs = [w for w in out.rglob("*.wav") if _speaker(str(w.relative_to(out))) in ENGLISH]
    if not wavs:
        shown = "\n    ".join(sorted(str(q.relative_to(out)) for q in out.glob("*/*"))[:40]) or "(empty)"
        raise SystemExit(f"ESD: no English speaker wavs after unpacking. Two levels of {out}:\n    {shown}")
    if len(wavs) != EXPECTED_ENGLISH_WAVS:
        print(f"  note: {len(wavs)} English wavs, the paper describes {EXPECTED_ENGLISH_WAVS}")
    write_marker(out, {"dataset": "esd", "english_wavs": len(wavs), "drive_id": DRIVE_ID})
    print(f"ESD ready: {len(wavs)} English clips under {out}")


def _speaker(path: str) -> str:
    """The 4-digit speaker folder (0001-0020) on a path, or ''. Mandarin is 0001-0010."""
    m = re.search(r"(?:^|/)(00(?:0[1-9]|1\d|20))(?:/|_|$)", path)
    return m.group(1) if m else ""


if __name__ == "__main__":
    main()
