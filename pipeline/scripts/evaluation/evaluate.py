"""Score a trained checkpoint on any corpus and split, e.g. cross-corpus or after adding a new corpus.

    python evaluate.py --experiment E5_audio_text_prosody --seed 0 --datasets meld --splits test
    python evaluate.py --checkpoint /path/best.pt --datasets ravdess crema_d --splits val test

Writes pipeline/results/<experiment>/seed<k>/eval_<datasets>_<splits>.json (+ predictions csv).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
PIPELINE = EXP_DIR.parent.parent
EMOTE = PIPELINE.parent
sys.path.insert(0, str(EMOTE))
sys.path.insert(0, str(PIPELINE / "scripts" / "downloads"))

from _fetch import data_root, use_data_cache  # noqa: E402

RESULTS = PIPELINE / "results"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--experiment", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", default=None, help="default: <data root>/runs/<experiment>/seed<k>/best.pt")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--splits", nargs="+", default=["test"], choices=("train", "val", "test"))
    p.add_argument("--data-root", default=None)
    p.add_argument("--results-dir", default=str(RESULTS))
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    root = data_root(args.data_root)
    use_data_cache(root)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else root / "runs" / args.experiment / f"seed{args.seed}" / "best.pt"
    import torch

    from emotion_model.engine import load_prosody_table, make_loader, predict, predictions_frame, score
    from emotion_model.model import EmotionModel
    from emotion_model.prosody import PROSODY_FEATURES

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    name = args.experiment or cfg["experiment"]
    out_dir = Path(args.results_dir) / name / f"seed{cfg['seed']}"
    tag = f"eval_{'+'.join(args.datasets)}_{'+'.join(args.splits)}"
    if (out_dir / f"{tag}.json").exists() and not args.force:
        print(f"already evaluated: {out_dir / f'{tag}.json'}")
        return
    manifest = pd.read_csv(root / cfg["data"]["manifest"])
    rows = manifest[manifest["dataset"].isin(args.datasets) & manifest["split"].isin(args.splits)]
    rows = rows[(rows["emotion"].fillna("") != "") | rows["arousal"].notna()].reset_index(drop=True)
    if rows.empty:
        raise SystemExit("no labelled clips match those datasets and splits")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = EmotionModel(cfg["model"], n_prosody=len(PROSODY_FEATURES))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    prosody = load_prosody_table(root / cfg["data"]["prosody_table"], set(rows["utt_id"])) if cfg["model"]["prosody"]["enabled"] else None
    loader = make_loader(rows, root, cfg, False, model.tokenizer, prosody, ckpt["prosody_stats"], cfg["seed"])
    pred = predict(model, loader, device, cfg["train"]["precision"] if device.type == "cuda" else "fp32")
    result = {"checkpoint": str(ckpt_path), "datasets": args.datasets, "splits": args.splits, "metrics": score(pred, model.modalities)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2, default=float))
    predictions_frame(pred, model.modalities, tag).to_csv(out_dir / f"{tag}_predictions.csv", index=False)
    print(f"{tag}: mean dataset UA {result['metrics']['mean_dataset_ua']:.4f} -> {out_dir}")


if __name__ == "__main__":
    main()
