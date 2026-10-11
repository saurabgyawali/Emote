"""Export a trained model for the Jetson: ONNX fp32 + dynamic-int8, verified against PyTorch, timed.

    python export_onnx.py                                   # the best deployable experiment from results/summary.json, seed 0
    python export_onnx.py --experiment E5_audio_text_prosody --seed 1
    python export_onnx.py --checkpoint /path/best.pt --out DIR

Writes <data root>/deploy/<experiment>_seed<k>/:
  model.onnx, model.int8.onnx   inputs: wave, wave_mask, input_ids, text_mask, has_text, prosody (those the model uses)
                                outputs: probs (7 emotions), dims (arousal, valence, dominance), alpha (modality weights)
  deploy.json                   emotions, modalities, prosody standardiser, audio settings, tokenizer kind, parity + latency
  tokenizer.json                when the model has a text tower
Copy that folder to the Jetson as Emote/emotion_deploy/ and run: bash go.sh emotions
Int8 is kept only if it agrees with fp32 on >= --min-agreement of the check clips.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
PIPELINE = EXP_DIR.parent.parent
EMOTE = PIPELINE.parent
sys.path.insert(0, str(EMOTE))
sys.path.insert(0, str(PIPELINE / "scripts" / "downloads"))

from _fetch import data_root, use_data_cache  # noqa: E402

RESULTS = PIPELINE / "results"


def _real_feeds(root: Path, cfg: dict, ckpt: dict, names: list[str], tokenizer, n: int,  # noqa: ANN001
                rng: np.random.Generator) -> tuple[list[dict], str]:
    """Up to n validation clips of the model's own corpora, preprocessed exactly as on the Jetson."""
    import pandas as pd

    from emotion_model.audio import SAMPLE_RATE, crop, load_audio, normalize_loudness
    from emotion_model.prosody import PROSODY_FEATURES, standardize
    from emotion_model.text import normalize_text

    manifest_path = root / cfg["data"]["manifest"]
    if not manifest_path.exists():
        return [], ""
    m = pd.read_csv(manifest_path)
    rows = m[(m["split"] == "val") & ~m["dataset"].isin(cfg["data"]["heldout_datasets"])]
    if rows.empty:
        return [], ""
    rows = rows.sample(min(n, len(rows)), random_state=0)
    prosody = None
    if "prosody" in names:
        table = pd.read_parquet(root / cfg["data"]["prosody_table"])
        prosody = table.set_index("utt_id").loc[rows["utt_id"], list(PROSODY_FEATURES)].to_numpy(dtype=np.float32)
    feeds = []
    for i, r in enumerate(rows.itertuples(index=False)):
        f = {}
        if "wave" in names:
            y = crop(normalize_loudness(load_audio(str(root / r.path))), int(SAMPLE_RATE * cfg["data"]["max_eval_seconds"]), None)
            f["wave"], f["wave_mask"] = y[None].astype(np.float32), np.ones((1, y.size), dtype=np.int64)
        if "input_ids" in names:
            text = normalize_text(r.text if isinstance(r.text, str) else "")
            enc = tokenizer([text], padding=True, truncation=True, max_length=cfg["data"]["max_text_tokens"], return_tensors="pt")
            f["input_ids"], f["text_mask"] = enc["input_ids"].numpy(), enc["attention_mask"].numpy()
            f["has_text"] = np.array([1 if text.strip() else 0], dtype=np.int64)
        if prosody is not None:
            f["prosody"] = standardize(prosody[i : i + 1], ckpt["prosody_stats"])
        feeds.append(f)
    return feeds, f"real validation clips of {sorted(rows['dataset'].unique())}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--experiment", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--check-clips", type=int, default=64)
    p.add_argument("--min-agreement", type=float, default=0.97)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    root = data_root(args.data_root)
    use_data_cache(root)
    exp = args.experiment
    if not exp and not args.checkpoint:
        summary = RESULTS / "summary.json"
        if not summary.exists():
            raise SystemExit("no results/summary.json: run the analysis first or pass --experiment")
        exp = json.loads(summary.read_text())["best_deployable"]
    ckpt_path = Path(args.checkpoint) if args.checkpoint else root / "runs" / exp / f"seed{args.seed}" / "best.pt"
    import torch

    from emotion_model.audio import SAMPLE_RATE, TARGET_DBFS
    from emotion_model.labels import DIMENSIONS, EMOTIONS
    from emotion_model.model import ByteTokenizer, EmotionModel
    from emotion_model.prosody import PROSODY_FEATURES

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    name = f"{cfg['experiment']}_seed{cfg['seed']}"
    out = Path(args.out) if args.out else root / "deploy" / name
    if (out / "deploy.json").exists() and not args.force:
        print(f"already exported: {out} (use --force)")
        return
    if args.dry_run:
        print(f"would export {ckpt_path} -> {out}")
        return
    model = EmotionModel(cfg["model"], n_prosody=len(PROSODY_FEATURES))
    model.load_state_dict(ckpt["model"])
    model.eval()
    mods = model.modalities

    class Wrapped(torch.nn.Module):
        def __init__(self, m: EmotionModel) -> None:
            super().__init__()
            self.m = m

        def forward(self, *inputs):  # noqa: ANN002
            kw = dict(zip(names, inputs))
            if "has_text" in kw:
                kw["has_text"] = kw["has_text"].bool()
            res = self.m(**kw)
            return torch.softmax(res["logits"], dim=-1), res["dims"], res["alpha"]

    rng = np.random.default_rng(0)
    seconds = 3.0
    names, example, axes = [], [], {}
    if "audio" in mods:
        n = int(SAMPLE_RATE * seconds)
        names += ["wave", "wave_mask"]
        example += [torch.from_numpy((rng.standard_normal((1, n)) * 0.05).astype(np.float32)), torch.ones(1, n, dtype=torch.long)]
        axes.update({"wave": {1: "samples"}, "wave_mask": {1: "samples"}})
    tokenizer = model.tokenizer
    if "text" in mods:
        enc = tokenizer(["this is a test sentence"], padding=True, truncation=True, max_length=cfg["data"]["max_text_tokens"], return_tensors="pt")
        names += ["input_ids", "text_mask", "has_text"]
        example += [enc["input_ids"], enc["attention_mask"], torch.ones(1, dtype=torch.long)]
        axes.update({"input_ids": {1: "tokens"}, "text_mask": {1: "tokens"}})
    if "prosody" in mods:
        names.append("prosody")
        example.append(torch.zeros(1, len(PROSODY_FEATURES)))
    out.mkdir(parents=True, exist_ok=True)
    fp32 = out / "model.onnx"
    wrapped = Wrapped(model)
    with torch.no_grad():
        ref = [t.numpy() for t in wrapped(*example)]
        torch.onnx.export(wrapped, tuple(example), str(fp32), input_names=names, output_names=["probs", "dims", "alpha"],
                          dynamic_axes=axes, opset_version=args.opset, do_constant_folding=True)
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8 = out / "model.int8.onnx"
    # MatMul/Gemm only: they hold nearly all the weights of a transformer, and ONNX Runtime's
    # CPU backend has no kernel for the ConvInteger nodes that quantising the wav2vec-style
    # convolutional front end would produce.
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"],
                     per_channel=True)
    sess32 = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    try:
        sess8 = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
        int8_error = None
    except Exception as exc:  # noqa: BLE001 - a model that cannot be quantised still ships in fp32
        sess8, int8_error = None, f"{type(exc).__name__}: {exc}"
        print(f"  int8 model unusable, shipping fp32 only: {int8_error[:200]}")
    feed = {k: v.numpy() for k, v in zip(names, example)}
    got = sess32.run(None, feed)
    max_err = float(max(np.abs(a - b).max() for a, b in zip(ref, got)))
    if max_err > 1e-3:
        raise SystemExit(f"ONNX fp32 disagrees with PyTorch (max |diff| {max_err:.2e}); not exporting")

    # int8 agreement and latency on REAL validation clips when the data is here. Random noise
    # is a bad judge: the model is unsure on noise, near-tied classes flip under any rounding,
    # and a 1-epoch model scored 90.6% "agreement" on noise (2026-10-10).
    feeds, check_source = _real_feeds(root, cfg, ckpt, names, tokenizer, args.check_clips, rng)
    if not feeds:
        check_source = "random noise (no manifest found)"
        for _ in range(args.check_clips):
            f = dict(feed)
            if "wave" in f:
                n = int(SAMPLE_RATE * rng.uniform(1.0, 6.0))
                f["wave"] = (rng.standard_normal((1, n)) * rng.uniform(0.01, 0.2)).astype(np.float32)
                f["wave_mask"] = np.ones((1, n), dtype=np.int64)
            if "prosody" in f:
                f["prosody"] = rng.standard_normal((1, len(PROSODY_FEATURES))).astype(np.float32)
            feeds.append(f)
    print(f"  int8 check on {len(feeds)} clips: {check_source}")
    agree, lat32, lat8 = 0, [], []
    for f in feeds:
        t0 = time.perf_counter()
        a = sess32.run(None, f)[0]
        lat32.append(time.perf_counter() - t0)
        if sess8 is not None:
            t0 = time.perf_counter()
            b = sess8.run(None, f)[0]
            lat8.append(time.perf_counter() - t0)
            agree += int(a.argmax() == b.argmax())
    agreement = agree / len(feeds) if sess8 is not None else 0.0
    keep_int8 = sess8 is not None and agreement >= args.min_agreement
    if not keep_int8 and int8.exists():
        int8.unlink()
    tok_kind = "none"
    if "text" in mods:
        if isinstance(tokenizer, ByteTokenizer):
            tok_kind = "byte"
        else:
            tokenizer.save_pretrained(str(out))
            tok_kind = "tokenizers_json"
    info = {
        "experiment": cfg["experiment"], "seed": cfg["seed"], "checkpoint": str(ckpt_path), "epoch": ckpt.get("epoch"),
        "val_score": ckpt.get("val_score"), "modalities": mods, "inputs": names, "emotions": list(EMOTIONS),
        "inference_classes": cfg["inference"]["classes"], "dimensions": list(DIMENSIONS),
        "sample_rate": SAMPLE_RATE, "target_dbfs": TARGET_DBFS, "max_seconds": cfg["data"]["max_eval_seconds"],
        "max_text_tokens": cfg["data"]["max_text_tokens"], "tokenizer": tok_kind,
        "prosody_features": list(PROSODY_FEATURES), "prosody_stats": ckpt.get("prosody_stats"),
        "onnx_vs_torch_max_abs_diff": max_err, "int8_kept": keep_int8, "int8_argmax_agreement": agreement,
        "int8_error": int8_error, "int8_check_clips": f"{len(feeds)} x {check_source}",
        "latency_ms_this_machine": {"fp32_median": 1000 * float(np.median(lat32)),
                                    "int8_median": 1000 * float(np.median(lat8)) if lat8 else None},
        "size_mb": {"fp32": fp32.stat().st_size / 1e6, "int8": int8.stat().st_size / 1e6 if keep_int8 else None},
    }
    (out / "deploy.json").write_text(json.dumps(info, indent=2, default=float))
    print(f"exported {name} -> {out}\n  fp32 {info['size_mb']['fp32']:.0f} MB, int8 "
          f"{'kept' if keep_int8 else 'DROPPED'} (agreement {agreement:.2%}); latency here fp32 "
          f"{info['latency_ms_this_machine']['fp32_median']:.0f} ms, int8 {info['latency_ms_this_machine']['int8_median'] or float('nan'):.0f} ms")


if __name__ == "__main__":
    main()
