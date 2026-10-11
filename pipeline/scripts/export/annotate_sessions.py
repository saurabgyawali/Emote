"""On the Jetson: label every kept utterance of every recorded session with an emotion.

    python annotate_sessions.py                         # all sessions, model in Emote/emotion_deploy/
    python annotate_sessions.py --session 20261010_190932 --force
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
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    deploy = Path(args.deploy)
    if not (deploy / "deploy.json").exists():
        raise SystemExit(f"no exported model at {deploy}. Train on the GPU machine, run `bash pipeline/run.sh export`, "
                         f"and copy <data root>/deploy/<name>/ here as {DEFAULT_DEPLOY.name}/")
    if "prosody" in json.loads((deploy / "deploy.json").read_text())["modalities"]:
        try:
            import librosa  # noqa: F401
        except ImportError:
            raise SystemExit("This model uses the prosody tower, which needs librosa (pitch tracking). Install it once into "
                             f"this environment:\n  {sys.executable} -m pip install librosa")
    model = Model(deploy)
    sessions = sorted(Path(args.sessions_dir).glob(args.session or "*"))
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
