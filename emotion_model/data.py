"""Manifest rows -> training batches.

A batch holds padded 16 kHz audio and its mask, tokenised transcripts, standardised
prosody features, the class target (-1 when a clip only has dimensional labels), the
clip's corpus label-set mask, and dimensional targets (NaN when absent).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio import SAMPLE_RATE, augment, crop, load_audio, normalize_loudness
from .labels import DIMENSIONS, EMOTION_INDEX, class_mask
from .prosody import PROSODY_FEATURES, standardize
from .text import normalize_text


class EmotionDataset(Dataset):
    def __init__(self, rows, data_root: str | Path, train: bool, data_cfg: dict,  # noqa: ANN001 - pandas DataFrame
                 prosody: dict[str, np.ndarray] | None = None, prosody_stats: dict | None = None,
                 load_wave: bool = True) -> None:
        self.rows = rows.reset_index(drop=True)
        self.root = Path(data_root)
        self.train = train
        self.cfg = data_cfg
        self.max_samples = int(SAMPLE_RATE * (data_cfg["max_train_seconds"] if train else data_cfg["max_eval_seconds"]))
        self.prosody = prosody
        self.prosody_stats = prosody_stats
        self.load_wave = load_wave
        self._masks = {ds: np.array(class_mask(ds), dtype=bool) for ds in self.rows["dataset"].unique()}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows.iloc[i]
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item())) if self.train else None
        if self.load_wave:
            y = normalize_loudness(load_audio(str(self.root / r["path"])))
            y = crop(y, self.max_samples, rng)
        else:  # prosody-only models never read audio here: their features are precomputed
            y = np.zeros(SAMPLE_RATE // 10, dtype=np.float32)
        if self.train and self.cfg["augment"] and self.load_wave:
            a = self.cfg["augmentation"]
            y = augment(y, rng, tuple(a["noise_snr_db"]), tuple(a["gain_db"]), a["p_noise"])
        emotion = r["emotion"] if isinstance(r["emotion"], str) else ""
        item = {
            "wave": y,
            "text": normalize_text(r["text"]) if isinstance(r["text"], str) else "",
            "label": EMOTION_INDEX[emotion] if emotion else -1,
            "class_mask": self._masks[r["dataset"]],
            "dims": np.array([r[d] for d in DIMENSIONS], dtype=np.float32),
            "dataset": r["dataset"],
            "utt_id": r["utt_id"],
        }
        if self.prosody is not None:
            raw = self.prosody.get(r["utt_id"])
            vec = raw if raw is not None else np.full(len(PROSODY_FEATURES), np.nan, dtype=np.float32)
            item["prosody"] = standardize(vec[None], self.prosody_stats)[0]
        return item


def make_collate(tokenizer, max_text_tokens: int):  # noqa: ANN001
    def collate(items: list[dict]) -> dict:
        n = len(items)
        width = max(max(len(it["wave"]) for it in items), SAMPLE_RATE // 10)
        wave = torch.zeros(n, width)
        mask = torch.zeros(n, width, dtype=torch.long)
        for i, it in enumerate(items):
            wave[i, : len(it["wave"])] = torch.from_numpy(it["wave"])
            mask[i, : len(it["wave"])] = 1
        batch = {
            "wave": wave,
            "wave_mask": mask,
            "label": torch.tensor([it["label"] for it in items], dtype=torch.long),
            "class_mask": torch.from_numpy(np.stack([it["class_mask"] for it in items])),
            "dims": torch.from_numpy(np.stack([it["dims"] for it in items])),
            "dataset": [it["dataset"] for it in items],
            "utt_id": [it["utt_id"] for it in items],
            "has_text": torch.tensor([bool(it["text"].strip()) for it in items]),
        }
        if tokenizer is not None:
            enc = tokenizer([it["text"] for it in items], padding=True, truncation=True,
                            max_length=max_text_tokens, return_tensors="pt")
            batch["input_ids"] = enc["input_ids"]
            batch["text_mask"] = enc["attention_mask"]
        if "prosody" in items[0]:
            batch["prosody"] = torch.from_numpy(np.stack([it["prosody"] for it in items]))
        return batch

    return collate


def sample_weights(rows, mode: str, power: float = 0.5) -> np.ndarray:  # noqa: ANN001
    """Per-clip sampling weights.

    dataset        every corpus contributes equally per epoch (MELD's 10 k clips would
                   otherwise drown RAVDESS's 1.4 k)
    dataset_class  additionally flattens each corpus's class distribution, softened by
                   `power` (0 = no flattening, 1 = fully balanced)
    none           uniform
    """
    if mode == "none":
        return np.ones(len(rows))
    ds_counts = rows["dataset"].map(rows["dataset"].value_counts())
    w = 1.0 / ds_counts.to_numpy(dtype=float)
    if mode == "dataset_class":
        key = rows["dataset"] + "|" + rows["emotion"].fillna("")
        cell = key.map(key.value_counts()).to_numpy(dtype=float)
        per_ds = rows.groupby("dataset")["emotion"].transform(lambda s: s.fillna("").nunique()).to_numpy(dtype=float)
        w = w * (ds_counts.to_numpy(dtype=float) / (cell * per_ds)) ** power
    elif mode != "dataset":
        raise ValueError(f"unknown sampler mode {mode!r}")
    return w / w.sum()


def class_weights(rows, power: float = 0.5) -> np.ndarray:  # noqa: ANN001
    """Inverse-frequency loss weights over the 7 classes, softened by `power`, mean 1 over present classes."""
    counts = np.zeros(len(EMOTION_INDEX))
    for e, n in rows["emotion"].fillna("").value_counts().items():
        if e:
            counts[EMOTION_INDEX[e]] = n
    w = np.zeros_like(counts)
    present = counts > 0
    w[present] = (counts[present].sum() / counts[present]) ** power
    w[present] /= w[present].mean()
    return w.astype(np.float32)
