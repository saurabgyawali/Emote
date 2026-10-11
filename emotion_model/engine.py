"""Shared by training, evaluation and export: building data loaders, running the model, scoring."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import EmotionDataset, make_collate, sample_weights
from .labels import EMOTIONS
from .metrics import summarize
from .prosody import PROSODY_FEATURES


def load_prosody_table(path: Path, utt_ids: set[str]) -> dict[str, np.ndarray]:
    if not path.exists():
        raise SystemExit(f"prosody features missing: {path}. Run: bash pipeline/run.sh prepare")
    table = pd.read_parquet(path)
    table = table[table["utt_id"].isin(utt_ids)]
    missing = len(utt_ids) - len(table)
    if missing:
        raise SystemExit(f"prosody table lacks {missing} clip(s) of this experiment; re-run extract_prosody.py")
    values = table[list(PROSODY_FEATURES)].to_numpy(dtype=np.float32)
    return dict(zip(table["utt_id"], values))


def make_loader(rows: pd.DataFrame, root: Path, cfg: dict, train: bool, tokenizer, prosody, prosody_stats,  # noqa: ANN001
                seed: int) -> DataLoader:
    data_cfg = cfg["data"]
    ds = EmotionDataset(rows, root, train, data_cfg, prosody, prosody_stats, load_wave=cfg["model"]["audio"]["enabled"])
    g = torch.Generator()
    g.manual_seed(seed)
    collate = make_collate(tokenizer, data_cfg["max_text_tokens"])
    workers = data_cfg["num_workers"]
    common = {"collate_fn": collate, "num_workers": workers, "pin_memory": torch.cuda.is_available(),
              "persistent_workers": workers > 0, "generator": g}
    if train:
        w = sample_weights(rows.reset_index(drop=True), data_cfg["sampler"], data_cfg["sampler_power"])
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(rows),
                                        replacement=True, generator=g)
        return DataLoader(ds, batch_size=cfg["train"]["batch_size"], sampler=sampler, drop_last=True, **common)
    return DataLoader(ds, batch_size=cfg["train"]["eval_batch_size"], shuffle=False, **common)


def model_inputs(batch: dict, device: torch.device) -> dict:
    keys = ("wave", "wave_mask", "input_ids", "text_mask", "has_text", "prosody")
    return {k: batch[k].to(device, non_blocking=True) for k in keys if k in batch}


def autocast(device: torch.device, precision: str):  # noqa: ANN201
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


@torch.no_grad()
def predict(model, loader: DataLoader, device: torch.device, precision: str) -> dict[str, np.ndarray]:  # noqa: ANN001
    model.eval()
    out: dict[str, list] = {k: [] for k in ("utt_id", "dataset", "label", "probs", "dims_pred", "dims_true", "alpha")}
    for batch in loader:
        with autocast(device, precision):
            res = model(**model_inputs(batch, device))
        out["utt_id"] += batch["utt_id"]
        out["dataset"] += batch["dataset"]
        out["label"].append(batch["label"].numpy())
        out["probs"].append(torch.softmax(res["logits"].float(), dim=-1).cpu().numpy())
        out["dims_pred"].append(res["dims"].float().cpu().numpy())
        out["dims_true"].append(batch["dims"].numpy())
        out["alpha"].append(res["alpha"].float().cpu().numpy())
    if not out["utt_id"]:
        return {}
    return {
        "utt_id": np.array(out["utt_id"]),
        "dataset": np.array(out["dataset"]),
        "label": np.concatenate(out["label"]),
        "probs": np.concatenate(out["probs"]),
        "dims_pred": np.concatenate(out["dims_pred"]),
        "dims_true": np.concatenate(out["dims_true"]),
        "alpha": np.concatenate(out["alpha"]),
    }


def score(pred: dict[str, np.ndarray], modalities: list[str]) -> dict:
    if not pred:
        return {}
    s = summarize(pred["dataset"], pred["label"], pred["probs"], pred["dims_pred"], pred["dims_true"])
    s["mean_alpha"] = {m: float(pred["alpha"][:, i].mean()) for i, m in enumerate(modalities)}
    return s


def predictions_frame(pred: dict[str, np.ndarray], modalities: list[str], kind: str) -> pd.DataFrame:
    from .labels import DATASET_CLASSES, DIMENSIONS
    from .metrics import masked_argmax

    df = pd.DataFrame({"utt_id": pred["utt_id"], "dataset": pred["dataset"], "kind": kind,
                       "label": [EMOTIONS[i] if i >= 0 else "" for i in pred["label"]]})
    preds = [EMOTIONS[masked_argmax(p[None], list(DATASET_CLASSES[d]))[0]] for p, d in zip(pred["probs"], pred["dataset"])]
    df["pred"] = preds
    for j, e in enumerate(EMOTIONS):
        df[f"p_{e}"] = pred["probs"][:, j].round(5)
    for j, d in enumerate(DIMENSIONS):
        df[f"{d}_pred"] = pred["dims_pred"][:, j].round(4)
        df[f"{d}_true"] = pred["dims_true"][:, j]
    for i, m in enumerate(modalities):
        df[f"alpha_{m}"] = pred["alpha"][:, i].round(4)
    return df


def warmup_cosine(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
