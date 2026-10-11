"""End-to-end smoke test on synthetic corpora: every pipeline stage, real code paths, minutes not days.

    python smoke_test.py                 # in a temporary data root, removed afterwards
    python smoke_test.py --keep DIR      # keep everything in DIR for inspection

Builds tiny RAVDESS / CREMA-D / TESS look-alikes (same file naming, same quirks such as
TESS's "OA_" typo) whose emotions differ in pitch, then runs: build_manifest ->
extract_prosody -> quality_checks -> train (prosody-only, full three-tower model with
tiny random encoders, and a held-out-corpus run) -> analyze -> export_onnx ->
annotate_sessions on a fake Jetson session. Nothing is downloaded. It fails loudly if
any stage fails or if the prosody model cannot learn the synthetic pitch cue, which
would mean labels and features are misaligned.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
PIPELINE = EXP_DIR.parent.parent
EMOTE = PIPELINE.parent
SCRIPTS = PIPELINE / "scripts"
TINY = ["model.audio.encoder=tiny-random-wavlm", "model.text.encoder=tiny-random-roberta", "train.epochs=4",
        "train.batch_size=8", "train.grad_accum=1", "train.eval_batch_size=8", "data.num_workers=2",
        "train.log_every=5", "train.patience=10", "data.max_train_seconds=2.0", "data.max_eval_seconds=3.0",
        "train.lr_head=3.0e-3"]
EMO_F0 = {"neutral": 110, "happy": 220, "sad": 90, "angry": 180, "fear": 260, "disgust": 140, "surprise": 300}


def tone(emotion: str, sr: int, seconds: float, rng: np.random.Generator, speaker_shift: float) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    f0 = EMO_F0[emotion] * speaker_shift * (1 + 0.05 * np.sin(2 * np.pi * 3 * t))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    y = sum(np.sin(k * phase) / k for k in range(1, 6))
    env = np.clip(np.sin(np.pi * t / seconds) * 3, 0, 1)
    return (0.2 * env * y / 2 + 0.003 * rng.standard_normal(t.size)).astype(np.float32)


def make_corpora(raw: Path) -> None:
    import soundfile as sf

    rng = np.random.default_rng(0)
    codes = {"neutral": "01", "happy": "03", "sad": "04", "angry": "05", "fear": "06", "disgust": "07", "surprise": "08"}
    for actor in range(1, 25):
        for emo, code in codes.items():
            path = raw / "ravdess" / f"Actor_{actor:02d}" / f"03-01-{code}-01-01-01-{actor:02d}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, tone(emo, 48000, 1.6, rng, 1 + 0.01 * actor), 48000)
    crema = raw / "crema_d"
    (crema / "AudioWAV").mkdir(parents=True, exist_ok=True)
    cemo = {"NEU": "neutral", "HAP": "happy", "SAD": "sad", "ANG": "angry", "FEA": "fear", "DIS": "disgust"}
    with open(crema / "VideoDemographics.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ActorID", "Age", "Sex", "Race", "Ethnicity"])
        for a in range(1001, 1021):
            w.writerow([a, 30, "Male" if a % 2 else "Female", "x", "x"])
    with open(crema / "summaryTable.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["", "FileName", "VoiceVote", "VoiceLevel"])
        for a in range(1001, 1021):
            for sent in ("DFA", "IEO"):
                for code, emo in cemo.items():
                    stem = f"{a}_{sent}_{code}_XX"
                    sf.write(crema / "AudioWAV" / f"{stem}.wav", tone(emo, 16000, 1.4, rng, 1 + 0.005 * (a - 1000)), 16000)
                    w.writerow([0, stem, code[0] if code != "NEU" else "N", 50])
    tess = raw / "tess"
    tess.mkdir(parents=True, exist_ok=True)
    temo = {"neutral": "neutral", "happy": "happy", "sad": "sad", "angry": "angry", "fear": "fear", "disgust": "disgust", "ps": "surprise"}
    for spk in ("OAF", "YAF"):
        for word in ("back", "bar", "base"):
            for lab, emo in temo.items():
                name = f"{'OA' if (spk, word, lab) == ('OAF', 'bar', 'neutral') else spk}_{word}_{lab}.wav"
                sf.write(tess / name, tone(emo, 24414, 1.2, rng, 0.98 if spk == "OAF" else 1.03), 24414)


def run(cmd: list[str], env_root: Path) -> None:
    print(f"\n$ {' '.join(Path(c).name if c.endswith('.py') else c for c in cmd)}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, env={**__import__("os").environ, "EMOTE_DATA": str(env_root)})
    if r.returncode:
        raise SystemExit(f"SMOKE TEST FAILED at: {' '.join(cmd)}")
    print(f"  ok ({time.time() - t0:.0f}s)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--keep", default=None, help="work in this directory and keep it")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    work = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="emote_smoke_"))
    root, results, figures = work / "data", work / "results", work / "figures"
    py = sys.executable
    dev = ["--device", args.device] if args.device else []
    t_all = time.time()
    try:
        make_corpora(root / "raw")
        run([py, str(SCRIPTS / "preprocessing/build_manifest.py"), "--workers", "4", "--force"], root)
        run([py, str(SCRIPTS / "preprocessing/extract_prosody.py"), "--workers", "4"], root)
        run([py, str(SCRIPTS / "preprocessing/quality_checks.py"), "--out", str(results / "quality_checks.json")], root)
        for exp, seeds in (("E1_prosody_only", (0, 1)), ("E5_audio_text_prosody", (0, 1)), ("X2_leave_out_crema", (0,))):
            for s in seeds:
                run([py, str(SCRIPTS / "training/train.py"), "--experiment", exp, "--seed", str(s),
                     "--results-dir", str(results), *dev, "--override", *TINY], root)
        run([py, str(SCRIPTS / "visualization/analyze.py"), "--results-dir", str(results), "--figures-dir", str(figures)], root)
        run([py, str(SCRIPTS / "export/export_onnx.py"), "--experiment", "E5_audio_text_prosody", "--seed", "0",
             "--out", str(work / "deploy"), "--check-clips", "8", "--min-agreement", "0.0"], root)
        session = work / "sessions" / "20260101_000000"
        (session / "clips").mkdir(parents=True)
        import soundfile as sf

        rng = np.random.default_rng(1)
        with open(session / "utterances.jsonl", "w") as fh:
            for i, emo in enumerate(("happy", "sad", "angry"), 1):
                sf.write(session / "clips" / f"utt_{i:04d}.wav", tone(emo, 16000, 1.5, rng, 1.0), 16000)
                fh.write(json.dumps({"index": i, "start_s": 2.0 * i, "end_s": 2.0 * i + 1.5, "text": f"I feel {emo}",
                                     "kept": True, "clip": f"clips/utt_{i:04d}.wav"}) + "\n")
        run([py, str(SCRIPTS / "export/annotate_sessions.py"), "--deploy", str(work / "deploy"),
             "--sessions-dir", str(work / "sessions")], root)

        summary = json.loads((results / "summary.json").read_text())
        e1 = summary["experiments"]["E1_prosody_only"]["best_test"]["mean_dataset_ua"]["mean"]
        chance = 1 / 7
        figs = sorted(f.name for f in figures.glob("*.png"))
        emotions = [json.loads(l)["emotion"] for l in (session / "emotions.jsonl").read_text().splitlines()]
        print(f"\nprosody-only test UA on the synthetic pitch cue: {e1:.3f} (chance ~{chance:.2f})")
        print(f"figures: {figs}")
        print(f"fake session annotated: {emotions}")
        if not e1 > 2 * chance:
            raise SystemExit("SMOKE TEST FAILED: the prosody model did not learn an obvious pitch cue -> labels/features misaligned")
        if len(figs) < 6 or len(emotions) != 3:
            raise SystemExit("SMOKE TEST FAILED: missing figures or annotations")
        print(f"\nSMOKE TEST PASSED in {(time.time() - t_all) / 60:.1f} min ({work})")
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
