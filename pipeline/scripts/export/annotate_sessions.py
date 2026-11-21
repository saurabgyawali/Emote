"""On the Jetson: label every kept utterance of every recorded session with an emotion.

    python annotate_sessions.py                         # all sessions, model in Emote/emotion_deploy/
    python annotate_sessions.py --session 20261010_190932 --force
    python annotate_sessions.py --latest                # only the newest session, as go.sh does after recording
    bash go.sh emotions                                 # the same, through the one user-facing script

Reads transcriber/sessions/<s>/utterances.jsonl (text) and clips/utt_NNNN.wav (16 kHz
audio), runs the exported ONNX model (int8 when it passed its parity check), and writes
sessions/<s>/emotions.jsonl -- one record per utterance with the emotion, its
confidence, all class probabilities, arousal/valence/dominance and the modality weights
-- plus emotions_summary.json. Needs only numpy, soundfile, scipy, onnxruntime and
tokenizers, which the transcriber's environment already has (faster-whisper brings the
last two); librosa only if the model uses the prosody tower.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
PIPELINE = EXP_DIR.parent.parent
EMOTE = PIPELINE.parent
sys.path.insert(0, str(EMOTE))

DEFAULT_DEPLOY = EMOTE / "emotion_deploy"
DEFAULT_SESSIONS = EMOTE / "transcriber" / "sessions"
CUE_NAMES = {"audio": "voice", "text": "words", "prosody": "pitch/energy"}
UNSURE_BELOW = 0.40
# Test UA of the deployed E5 per corpus (ACCRE run 2026-11-04): what "how often right" means here.
ACCURACY_NOTE = ("A model's guess, not a diagnosis: trained on actors and TV dialogue, it was right about 72-77% of "
                 "the time on acted speech and 44% on natural conversation.")


def find_deploy(path: Path) -> Path | None:
    """The folder holding deploy.json: the given one, or its single subfolder (a copy one level too deep)."""
    if (path / "deploy.json").exists():
        return path
    nested = sorted(p.parent for p in path.glob("*/deploy.json"))
    return nested[0] if len(nested) == 1 else None


def clock(seconds: float) -> str:
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}"


def report(session: Path, summary: dict, color: bool) -> None:
    """The end-of-session screen: one line per utterance, then the session's overall picture."""
    bold = (lambda t: f"\033[1m{t}\033[0m") if color else (lambda t: t)
    dim = (lambda t: f"\033[2m{t}\033[0m") if color else (lambda t: t)
    records = [json.loads(line) for line in (session / "emotions.jsonl").read_text(encoding="utf-8").splitlines()]
    print(f"\n{bold('How you sounded')}  session {session.name} - {len(records)} utterance(s), "
          f"labelled in {summary['seconds']}s with {summary['model']} ({summary['model_file']})\n")
    if not records:
        print("  Nothing to label: no kept utterances in this session.")
        return
    for r in records:
        ranked = sorted(r["probs"].items(), key=lambda kv: -kv[1])
        cues = ", ".join(f"{CUE_NAMES.get(m, m)} {w:.0%}" for m, w in
                         sorted(r["modality_weights"].items(), key=lambda kv: -kv[1]))
        unsure = dim(" (unsure)") if r["confidence"] < UNSURE_BELOW else ""
        text = r["text"] if len(r["text"]) <= 70 else r["text"][:67] + "..."
        print(f"  {clock(r['start_s'])}  {bold(r['emotion'].ljust(8))} {r['confidence']:4.0%}{unsure}  {text}")
        print(dim(f"                {'then ' + ranked[1][0] + f' {ranked[1][1]:.0%}' if len(ranked) > 1 else ''}"
                  f"  - judged from {cues}"))
    counts = summary["emotions"]
    top, n = next(iter(counts.items()))
    print(f"\n  {bold('Overall')}: mostly {bold(top)} ({n} of {len(records)} utterances)"
          + (f"; also {', '.join(f'{e} {c}' for e, c in list(counts.items())[1:])}" if len(counts) > 1 else ""))
    print(dim(f"  {ACCURACY_NOTE}"))
    print(dim(f"  Saved: {session / 'emotions.jsonl'}"))


class Model:
    def __init__(self, deploy: Path) -> None:
        import onnxruntime as ort

        self.info = json.loads((deploy / "deploy.json").read_text())
        path = deploy / ("model.int8.onnx" if self.info["int8_kept"] else "model.onnx")
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.inputs = self.info["inputs"]
        self.tokenizer = None
        if self.info["tokenizer"] == "tokenizers_json":
            from tokenizers import Tokenizer

            self.tokenizer = Tokenizer.from_file(str(deploy / "tokenizer.json"))
            self.tokenizer.enable_truncation(self.info["max_text_tokens"])
        self.path = path

    def text_inputs(self, text: str) -> dict:
        from emotion_model.text import normalize_text

        text = normalize_text(text)
        if self.info["tokenizer"] == "byte":
            ids = [0] + [b + 3 for b in text.encode("utf-8")][: self.info["max_text_tokens"] - 2] + [2]
        else:
            ids = self.tokenizer.encode(text).ids
        return {"input_ids": np.array([ids], dtype=np.int64), "text_mask": np.ones((1, len(ids)), dtype=np.int64),
                "has_text": np.array([1 if text.strip() else 0], dtype=np.int64)}

    def __call__(self, wave: np.ndarray, text: str) -> dict:
        from emotion_model.audio import crop, normalize_loudness

        feed: dict = {}
        y = crop(normalize_loudness(wave), int(self.info["sample_rate"] * self.info["max_seconds"]), None)
        if "wave" in self.inputs:
            feed["wave"] = y[None].astype(np.float32)
            feed["wave_mask"] = np.ones((1, y.size), dtype=np.int64)
        if "input_ids" in self.inputs:
            feed.update(self.text_inputs(text))
        if "prosody" in self.inputs:
            from emotion_model.prosody import prosody_vector, standardize

            feed["prosody"] = standardize(prosody_vector(wave)[None], self.info["prosody_stats"])
        probs, dims, alpha = self.session.run(None, feed)
        allowed = [self.info["emotions"].index(e) for e in self.info["inference_classes"]]
        p = probs[0][allowed] / probs[0][allowed].sum()
        k = int(np.argmax(p))
        return {
            "emotion": self.info["inference_classes"][k], "confidence": round(float(p[k]), 4),
            "probs": {e: round(float(v), 4) for e, v in zip(self.info["inference_classes"], p)},
            **{d: round(float(v), 4) for d, v in zip(self.info["dimensions"], dims[0])},
            "modality_weights": {m: round(float(a), 4) for m, a in zip(self.info["modalities"], alpha[0])},
        }


def annotate(session: Path, model: Model, force: bool) -> dict | None:
    from emotion_model.audio import load_audio

    src, out = session / "utterances.jsonl", session / "emotions.jsonl"
    if not src.exists() or (out.exists() and not force):
        return None
    records, t0 = [], time.monotonic()
    for line in src.read_text(encoding="utf-8").splitlines():
        u = json.loads(line)
        if not u.get("kept") or not u.get("clip") or not (session / u["clip"]).exists():
            continue
        res = model(load_audio(str(session / u["clip"])), u.get("text", ""))
        records.append({"index": u["index"], "start_s": u["start_s"], "end_s": u["end_s"], "text": u.get("text", ""), **res})
    tmp = out.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    tmp.replace(out)
    counts = Counter(r["emotion"] for r in records)
    summary = {"session": session.name, "utterances": len(records), "model": model.info["experiment"],
               "model_file": model.path.name, "emotions": dict(counts.most_common()),
               "mean": {d: round(float(np.mean([r[d] for r in records])), 4) for d in model.info["dimensions"]} if records else {},
               "seconds": round(time.monotonic() - t0, 1)}
    (session / "emotions_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--deploy", default=str(DEFAULT_DEPLOY))
    p.add_argument("--sessions-dir", default=str(DEFAULT_SESSIONS))
    p.add_argument("--session", default=None, help="one session folder name; default all")
    p.add_argument("--latest", action="store_true", help="only the newest session, with the full results screen")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    deploy = find_deploy(Path(args.deploy))
    if deploy is None:
        raise SystemExit(f"no exported model at {args.deploy}: copy the CONTENTS of the exported folder "
                         f"(deploy.json, model.int8.onnx, tokenizer.json, ...) into {DEFAULT_DEPLOY}/")
    if deploy != Path(args.deploy):
        print(f"  note: using the model one folder down, in {deploy}")
    if "prosody" in json.loads((deploy / "deploy.json").read_text())["modalities"]:
        try:
            import librosa  # noqa: F401
        except ImportError:
            raise SystemExit("This model uses the prosody tower, which needs librosa (pitch tracking). Install it once into "
                             f"this environment:\n  {sys.executable} -m pip install librosa")
    model = Model(deploy)
    sessions = sorted(p for p in Path(args.sessions_dir).glob(args.session or "*") if p.is_dir())
    if args.latest:
        sessions = sessions[-1:]
        if not sessions:
            raise SystemExit(f"no sessions in {args.sessions_dir}")
        summ = annotate(sessions[0], model, args.force)
        if summ is None:
            summ = json.loads((sessions[0] / "emotions_summary.json").read_text()) \
                if (sessions[0] / "emotions_summary.json").exists() else None
        if summ is None:
            raise SystemExit(f"{sessions[0].name}: no utterances.jsonl, nothing to label")
        report(sessions[0], summ, sys.stdout.isatty() and not os.environ.get("NO_COLOR"))
        return
    print(f"model {model.info['experiment']} ({model.path.name}); {len(sessions)} session(s)")
    for s in sessions:
        summ = annotate(s, model, args.force)
        if summ is None:
            continue
        print(f"  {s.name}: {summ['utterances']} utterance(s) in {summ['seconds']}s -> {summ['emotions']}")
        for r in [json.loads(l) for l in (s / "emotions.jsonl").read_text().splitlines()][:200]:
            print(f"    [{r['start_s']:7.2f}s] {r['emotion']:8} {r['confidence']:.2f}  {r['text'][:70]}")


if __name__ == "__main__":
    main()
