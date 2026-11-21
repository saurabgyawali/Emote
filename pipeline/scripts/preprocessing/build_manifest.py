"""Turn every downloaded corpus into one validated manifest and a 16 kHz mono FLAC cache.

    python build_manifest.py                         # every corpus found under <data root>/raw
    python build_manifest.py --datasets ravdess tess --crema-label voice_vote --workers 16
    python build_manifest.py --dry-run               # parse and report, write nothing

Output: <data root>/audio16k/<dataset>/<stem>.flac, <data root>/manifests/manifest.csv
(schema: emotion_model.labels.MANIFEST_SCHEMA) and manifests/summary.json.

Upstream representations, checked on first read and documented here because the loss
and the splits depend on them:

  RAVDESS   raw/ravdess/Actor_NN/03-01-EE-II-SS-RR-AA.wav, 48 kHz; field 3 is the emotion
            (01 neutral 02 calm 03 happy 04 sad 05 angry 06 fearful 07 disgust 08 surprised),
            field 5 the sentence, field 7 the actor (odd = male). Speech-only subset.
  CREMA-D   raw/crema_d/AudioWAV/AAAA_SSS_EEE_LL.wav, 16 kHz; EEE in ANG DIS FEA HAP NEU SAD;
            summaryTable.csv holds crowd votes (VoiceVote = audio-only perception, ties "A:N").
  TESS      raw/tess/{OAF,YAF}_<word>_<emotion>.wav, 24.4 kHz; one file is named "OA_..."
            (a typo for OAF, not a third speaker); "ps" = pleasant surprise.
  MELD      raw/meld/**/{train,dev,test}_sent_emo.csv + dia<D>_utt<U>.mp4; transcripts are
            cp1252-mangled UTF-8 in places ("Iâ\\x80\\x99m") and are repaired.
  ESD       raw/esd/**/00SS/<Emotion>/[train|evaluation|test/]00SS_NNNNNN.wav, 16 kHz, English
            speakers 0011-0020 only; transcripts in 00SS.txt as "00SS_NNNNNN<TAB>text<TAB>emotion".
            Layout taken from the ESD README/paper, not yet seen on real files: the parser accepts
            the emotion folder at any depth and fails loudly if nothing matches.
  IEMOCAP   raw/iemocap/IEMOCAP_full_release/Session{1-5}/dialog/EmoEvaluation/*.txt lines
            "[start - end]  TURN  emo  [V, A, D]" with V/A/D on 1-5; wavs under sentences/wav.
  MSP-Pod   raw/msp_podcast/Labels/labels_consensus.csv (EmoClass, EmoAct/Val/Dom on 1-7,
            SpkrID, Split_Set) + Audios/ + Transcripts/.

Splits are speaker-independent wherever the corpus allows: RAVDESS actors 19-20 val and
21-24 test; CREMA-D actors split 70/15/15 within each sex by a fixed seed; IEMOCAP
sessions 1-3 / 4 / 5; MELD and MSP-Podcast use their official splits (MELD's recur
speakers -- it is a TV show). TESS has only two speakers and is training data only. ESD's
own train/evaluation/test folders split sentences, not speakers, so they are ignored:
speakers 0011-0018 train, 0019 val, 0020 test.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
PIPELINE = EXP_DIR.parent.parent
EMOTE = PIPELINE.parent
sys.path.insert(0, str(EMOTE))
sys.path.insert(0, str(PIPELINE / "scripts" / "downloads"))

from _fetch import data_root  # noqa: E402
from emotion_model.labels import DIMENSIONS, MANIFEST_SCHEMA, validate_manifest  # noqa: E402

SPLIT_SEED = 20261010
ALL = ("ravdess", "crema_d", "tess", "meld", "esd", "iemocap", "msp_podcast")

RAVDESS_EMO = {"01": "neutral", "02": "neutral", "03": "happy", "04": "sad", "05": "angry", "06": "fear",
               "07": "disgust", "08": "surprise"}
RAVDESS_SRC = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry", "06": "fearful",
               "07": "disgust", "08": "surprised"}
RAVDESS_TEXT = {"01": "Kids are talking by the door.", "02": "Dogs are sitting by the door."}
CREMA_EMO = {"ANG": "angry", "DIS": "disgust", "FEA": "fear", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
CREMA_VOTE = {"A": "angry", "D": "disgust", "F": "fear", "H": "happy", "N": "neutral", "S": "sad"}
CREMA_TEXT = {
    "IEO": "It's eleven o'clock.", "TIE": "That is exactly what happened.", "IOM": "I'm on my way to the meeting.",
    "IWW": "I wonder what this is about.", "TAI": "The airplane is almost full.",
    "MTI": "Maybe tomorrow it will be cold.", "IWL": "I would like a new alarm clock.",
    "ITH": "I think I have a doctor's appointment.", "DFA": "Don't forget a jacket.",
    "ITS": "I think I've seen this before.", "TSI": "The surface is slick.", "WSI": "We'll stop in a couple of minutes.",
}
TESS_EMO = {"angry": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy", "neutral": "neutral",
            "ps": "surprise", "sad": "sad"}
MELD_EMO = {"neutral": "neutral", "joy": "happy", "sadness": "sad", "anger": "angry", "fear": "fear",
            "disgust": "disgust", "surprise": "surprise"}
MELD_SPLIT = {"train": "train", "dev": "val", "test": "test"}
ESD_EMO = {"neutral": "neutral", "happy": "happy", "sad": "sad", "angry": "angry", "surprise": "surprise"}
ESD_ENGLISH = tuple(f"{i:04d}" for i in range(11, 21))
ESD_SPLIT = {"0019": "val", "0020": "test"}
IEMOCAP_EMO = {"neu": "neutral", "hap": "happy", "exc": "happy", "sad": "sad", "ang": "angry"}
MSP_EMO = {"N": "neutral", "H": "happy", "S": "sad", "A": "angry", "F": "fear", "D": "disgust", "U": "surprise"}
MSP_SPLIT = {"Train": "train", "Development": "val", "Test1": "test"}


def row(dataset, stem, src, speaker, gender, split, emotion, text, source_label, dims=(np.nan,) * 3) -> dict:  # noqa: ANN001
    return {"utt_id": f"{dataset}/{stem}", "dataset": dataset, "src": str(src),
            "path": f"audio16k/{dataset}/{stem}.flac", "speaker": f"{dataset}_{speaker}", "gender": gender,
            "split": split, "emotion": emotion, **dict(zip(DIMENSIONS, dims)), "text": text,
            "source_label": source_label}


# --- parsers ---------------------------------------------------------------------
def parse_ravdess(raw: Path, _args) -> list[dict]:  # noqa: ANN001
    rows = []
    for wav in sorted(raw.rglob("*.wav")):
        f = wav.stem.split("-")
        if len(f) != 7 or f[0] != "03" or f[1] != "01":
            continue
        actor = int(f[6])
        split = "test" if actor >= 21 else "val" if actor >= 19 else "train"
        rows.append(row("ravdess", wav.stem, wav, f"{actor:02d}", "M" if actor % 2 else "F", split,
                        RAVDESS_EMO[f[2]], RAVDESS_TEXT[f[4]], RAVDESS_SRC[f[2]]))
    return rows


def _speaker_split(speakers_by_group: dict[str, list[str]], fractions=(0.7, 0.15)) -> dict[str, str]:
    rng = np.random.default_rng(SPLIT_SEED)
    out = {}
    for group in sorted(speakers_by_group):
        spk = sorted(speakers_by_group[group])
        rng.shuffle(spk)
        n_train = round(len(spk) * fractions[0])
        n_val = round(len(spk) * fractions[1])
        for i, s in enumerate(spk):
            out[s] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return out


def parse_crema_d(raw: Path, args) -> list[dict]:  # noqa: ANN001
    sex = {}
    with open(raw / "VideoDemographics.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            sex[r["ActorID"]] = "M" if r["Sex"].strip().lower().startswith("m") else "F"
    votes = {}
    if args.crema_label == "voice_vote":
        with open(raw / "summaryTable.csv", newline="") as fh:
            for r in csv.DictReader(fh):
                votes[r["FileName"]] = r["VoiceVote"].strip()
    wavs = sorted((raw / "AudioWAV").glob("*.wav"))
    groups: dict[str, list[str]] = {}
    for w in wavs:
        a = w.stem.split("_")[0]
        groups.setdefault(sex.get(a, "U"), [])
        if a not in groups[sex.get(a, "U")]:
            groups[sex.get(a, "U")].append(a)
    split = _speaker_split(groups)
    rows = []
    for w in wavs:
        actor, sent, emo, _level = w.stem.split("_")
        if args.crema_label == "voice_vote":
            vote = votes.get(w.stem, "")
            if vote not in CREMA_VOTE:  # missing or a tie such as "A:N"
                continue
            emotion, src = CREMA_VOTE[vote], f"voice_vote:{vote}"
        else:
            emotion, src = CREMA_EMO[emo], emo
        rows.append(row("crema_d", w.stem, w, actor, sex.get(actor, "U"), split[actor], emotion, CREMA_TEXT[sent], src))
    return rows


def parse_tess(raw: Path, _args) -> list[dict]:  # noqa: ANN001
    rows = []
    for w in sorted(raw.glob("*.wav")):
        parts = w.stem.split("_")
        speaker = "OAF" if parts[0].upper().startswith("OA") else "YAF"
        label = parts[-1].lower()
        word = "_".join(parts[1:-1])
        rows.append(row("tess", f"{speaker}_{word}_{label}", w, speaker, "F", "train", TESS_EMO[label],
                        f"Say the word {word}.", label))
    return rows


def _fix_mojibake(text: str) -> str:
    if any(c in text for c in "âÂ\x92\x93\x94\x85"):
        try:
            text = text.encode("cp1252", errors="ignore").decode("utf-8", errors="ignore") or text
        except UnicodeError:
            pass
    return text.replace("\x92", "'").replace("\x93", '"').replace("\x94", '"').replace("\x85", "...").strip()


MELD_VIDEO_DIRS = {"train": "train_splits", "dev": "dev_splits_complete", "test": "output_repeated_splits_test"}


def parse_meld(raw: Path, _args) -> list[dict]:  # noqa: ANN001
    # Each split numbers its dialogues from 0, so dev's dia9_utt5.mp4 and train's dia9_utt5.mp4
    # are different clips. Indexing videos by file name alone matched dev and test rows to
    # training videos -- wrong labels and train audio inside the test set (found 2026-10-10
    # on a real MELD slice). Videos are matched only within their split's own folder.
    rows, missing = [], 0
    for name, split in MELD_SPLIT.items():
        csv_path = next(raw.rglob(f"{name}_sent_emo.csv"))
        folder = next((p for p in raw.rglob(MELD_VIDEO_DIRS[name]) if p.is_dir()), None)
        videos = {p.name: p for p in folder.glob("*.mp4") if not p.name.startswith("._")} if folder else {}
        if folder is None:
            print(f"  meld: no {MELD_VIDEO_DIRS[name]}/ folder; {name} split skipped")
        df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")
        for r in df.itertuples(index=False):
            stem = f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}"
            video = videos.get(f"{stem}.mp4")
            if video is None:
                missing += 1
                continue
            speaker = re.sub(r"[^A-Za-z0-9]+", "", str(r.Speaker)) or "unknown"
            rows.append(row("meld", f"{name}_{stem}", video, speaker, "U", split, MELD_EMO[str(r.Emotion).lower()],
                            _fix_mojibake(str(r.Utterance)), str(r.Emotion).lower()))
    if missing:
        print(f"  meld: {missing} utterance(s) listed in the CSVs have no video and were skipped")
    return rows


def _read_text_any(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def parse_esd(raw: Path, _args) -> list[dict]:  # noqa: ANN001
    texts = {}
    for t in raw.rglob("00??.txt"):
        if t.stem not in ESD_ENGLISH:
            continue
        for line in _read_text_any(t).splitlines():
            parts = [x.strip() for x in line.split("\t") if x.strip()]
            if len(parts) >= 2:
                texts[parts[0]] = parts[1]
    rows, seen = [], set()
    for w in sorted(raw.rglob("*.wav")):
        if "__MACOSX" in w.parts or w.name.startswith("._"):
            continue
        speaker = w.stem.split("_")[0]
        emotion = next((ESD_EMO[p.lower()] for p in reversed(w.relative_to(raw).parts[:-1]) if p.lower() in ESD_EMO), None)
        if speaker not in ESD_ENGLISH or emotion is None or w.stem in seen:
            continue
        seen.add(w.stem)
        rows.append(row("esd", w.stem, w, speaker, "U", ESD_SPLIT.get(speaker, "train"), emotion,
                        texts.get(w.stem, ""), emotion))
    if not rows:
        raise SystemExit(f"ESD: no English clips with an emotion folder found under {raw}")
    if not texts:
        print("  esd: no transcript files matched; text left empty")
    return rows


def parse_iemocap(raw: Path, args) -> list[dict]:  # noqa: ANN001
    base = next((p for p in raw.rglob("Session1") if p.is_dir()), None)
    if base is None:
        raise SystemExit(f"IEMOCAP: no Session1 directory under {raw}")
    base = base.parent
    line_re = re.compile(r"^\[(\d+\.\d+) - (\d+\.\d+)\]\t(\S+)\t(\S+)\t\[(\S+), (\S+), (\S+)\]")
    rows = []
    for s in range(1, 6):
        sess = base / f"Session{s}"
        texts = {}
        for t in (sess / "dialog" / "transcriptions").glob("*.txt"):
            for line in t.read_text(errors="replace").splitlines():
                m = re.match(r"^(\S+) \[[^\]]*\]:\s*(.*)$", line)
                if m:
                    texts[m.group(1)] = m.group(2).strip()
        for ev in sorted((sess / "dialog" / "EmoEvaluation").glob("*.txt")):
            for line in ev.read_text(errors="replace").splitlines():
                m = line_re.match(line)
                if not m:
                    continue
                turn, emo = m.group(3), m.group(4)
                v, a, d = (float(m.group(i)) for i in (5, 6, 7))
                wav = sess / "sentences" / "wav" / turn.rsplit("_", 1)[0] / f"{turn}.wav"
                if not wav.exists():
                    continue
                gender = turn.rsplit("_", 1)[1][0]
                split = "test" if s == args.iemocap_test_session else "val" if s == args.iemocap_val_session else "train"
                dims = ((a - 1) / 4, (v - 1) / 4, (d - 1) / 4)  # file order is V, A, D; ours is A, V, D
                rows.append(row("iemocap", turn, wav, f"s{s}{gender}", gender if gender in "MF" else "U", split,
                                IEMOCAP_EMO.get(emo, ""), texts.get(turn, ""), emo, dims))
    return rows


def parse_msp_podcast(raw: Path, args) -> list[dict]:  # noqa: ANN001
    labels = next(raw.rglob("labels_consensus.csv"))
    audios = next(p for p in raw.rglob("Audios") if p.is_dir())
    tdir = next((p for p in raw.rglob("Transcripts") if p.is_dir()), None)
    df = pd.read_csv(labels)
    rows = []
    for r in df.itertuples(index=False):
        split = MSP_SPLIT.get(str(r.Split_Set))
        if split is None and args.msp_include_test2 and str(r.Split_Set).startswith("Test"):
            split = "test"
        if split is None:
            continue
        wav = audios / r.FileName
        if not wav.exists():
            continue
        stem = Path(r.FileName).stem
        spk = str(r.SpkrID)
        speaker = f"unknown_{split}" if spk.lower() in ("unknown", "nan", "") else spk
        g = str(r.Gender).strip().upper()[:1]
        text = ""
        if tdir is not None and (tdir / f"{stem}.txt").exists():
            text = (tdir / f"{stem}.txt").read_text(errors="replace").strip()
        dims = tuple((float(x) - 1) / 6 for x in (r.EmoAct, r.EmoVal, r.EmoDom))
        rows.append(row("msp_podcast", stem, wav, speaker, g if g in "MF" else "U", split,
                        MSP_EMO.get(str(r.EmoClass), ""), text, str(r.EmoClass), dims))
    return rows


PARSERS = {"ravdess": parse_ravdess, "crema_d": parse_crema_d, "tess": parse_tess, "meld": parse_meld,
           "esd": parse_esd, "iemocap": parse_iemocap, "msp_podcast": parse_msp_podcast}


# --- audio cache -----------------------------------------------------------------
def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def convert(job: tuple[str, str]) -> tuple[str, float, str]:
    """src -> 16 kHz mono FLAC; returns (dest, duration_s, error)."""
    src, dest = job
    import soundfile as sf

    dest_p = Path(dest)
    try:
        if not dest_p.exists():
            dest_p.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest_p.with_suffix(".tmp.flac")
            if src.endswith(".mp4"):
                subprocess.run([_ffmpeg(), "-nostdin", "-loglevel", "error", "-y", "-i", src, "-vn", "-ac", "1",
                                "-ar", "16000", "-sample_fmt", "s16", str(tmp)], check=True, timeout=300)
            else:
                from emotion_model.audio import load_audio

                sf.write(str(tmp), load_audio(src), 16000, subtype="PCM_16", format="FLAC")
            tmp.replace(dest_p)
        info = sf.info(str(dest_p))
        return dest, info.frames / info.samplerate, ""
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop a corpus
        return dest, 0.0, f"{type(exc).__name__}: {exc}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default=None)
    p.add_argument("--datasets", nargs="*", default=None, help=f"subset of {ALL}; default: every corpus under raw/")
    p.add_argument("--crema-label", choices=("acted", "voice_vote"), default="acted",
                   help="acted = what the actor was asked to portray; voice_vote = what listeners heard (ties dropped)")
    p.add_argument("--iemocap-val-session", type=int, default=4)
    p.add_argument("--iemocap-test-session", type=int, default=5)
    p.add_argument("--msp-include-test2", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--min-seconds", type=float, default=0.3)
    p.add_argument("--force", action="store_true", help="rebuild the manifest (the audio cache is always reused)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    root = data_root(args.data_root)
    sys.path.insert(0, str(EMOTE))
    out = root / "manifests" / "manifest.csv"
    if out.exists() and not args.force and not args.dry_run:
        print(f"manifest exists: {out} (use --force to rebuild)")
        return
    wanted = args.datasets or [d for d in ALL if (root / "raw" / d).is_dir()]
    if not wanted:
        raise SystemExit(f"no corpora under {root / 'raw'}; run the download scripts first")
    rows: list[dict] = []
    for ds in wanted:
        raw = root / "raw" / ds
        if not raw.is_dir():
            raise SystemExit(f"{ds}: {raw} does not exist")
        parsed = PARSERS[ds](raw, args)
        print(f"  {ds:12} {len(parsed):6d} clips parsed")
        rows += parsed
    df = pd.DataFrame(rows)
    if args.dry_run:
        print(df.groupby(["dataset", "split"]).size().unstack(fill_value=0))
        return

    jobs = [(s, str(root / pth)) for s, pth in zip(df["src"], df["path"])]
    print(f"  caching {len(jobs)} clips as 16 kHz FLAC with {args.workers} workers ...", flush=True)
    durations, errors = {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (dest, dur, err) in enumerate(pool.map(convert, jobs, chunksize=32), 1):
            durations[dest] = dur
            if err:
                errors.append(f"{dest}: {err}")
            if i % 2000 == 0:
                print(f"    {i}/{len(jobs)}", flush=True)
    df["duration_s"] = [durations[str(root / pth)] for pth in df["path"]]
    short = df["duration_s"] < args.min_seconds
    if errors:
        print(f"  {len(errors)} clip(s) failed to convert and are excluded, e.g. {errors[:3]}")
    if short.any():
        print(f"  {int(short.sum())} clip(s) shorter than {args.min_seconds}s excluded")
    df = df[~short].drop(columns=["src"])
    df = df[list(MANIFEST_SCHEMA)]
    validate_manifest(df)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summary = {
        "clips": len(df),
        "hours": round(df["duration_s"].sum() / 3600, 2),
        "by_dataset_split": df.groupby(["dataset", "split"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "by_dataset_emotion": df.groupby(["dataset", "emotion"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "speakers": df.groupby("dataset")["speaker"].nunique().to_dict(),
        "crema_label": args.crema_label,
        "conversion_failures": errors,
    }
    (out.parent / "summary.json").write_text(json.dumps(summary, indent=2, default=int))
    print(f"manifest: {len(df)} clips, {summary['hours']} h -> {out}")


if __name__ == "__main__":
    main()
