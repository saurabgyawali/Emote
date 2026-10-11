"""Evaluation metrics, reported overall and per corpus.

UA (unweighted accuracy = mean per-class recall) is the headline number in speech
emotion recognition because corpora are imbalanced; WA (plain accuracy) and macro-F1
are reported beside it. BIG4 restricts to neutral/happy/sad/angry and argmaxes over
those four only, the convention that makes numbers comparable with the literature.
"""

from __future__ import annotations

import numpy as np

from .labels import BIG4, DATASET_CLASSES, DIMENSIONS, EMOTION_INDEX, EMOTIONS


def unweighted_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    classes = np.unique(y)
    return float(np.mean([np.mean(p[y == c] == c) for c in classes])) if classes.size else float("nan")


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    scores = []
    for c in np.unique(y):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def confusion(y: np.ndarray, p: np.ndarray, n: int = len(EMOTIONS)) -> list[list[int]]:
    m = np.zeros((n, n), dtype=int)
    np.add.at(m, (y, p), 1)
    return m.tolist()


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(ece)


def ccc_np(pred: np.ndarray, target: np.ndarray) -> float:
    ok = np.isfinite(target) & np.isfinite(pred)
    if ok.sum() < 2:
        return float("nan")
    x, y = pred[ok], target[ok]
    return float(2 * np.cov(x, y, bias=True)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2 + 1e-12))


def masked_argmax(probs: np.ndarray, allowed: list[str]) -> np.ndarray:
    idx = [EMOTION_INDEX[e] for e in allowed]
    return np.asarray(idx)[np.argmax(probs[:, idx], axis=1)]


def _categorical(y: np.ndarray, probs: np.ndarray, allowed: list[str]) -> dict:
    pred = masked_argmax(probs, allowed)
    sub = probs[:, [EMOTION_INDEX[e] for e in allowed]]
    conf = sub.max(axis=1) / np.clip(sub.sum(axis=1), 1e-12, None)
    return {
        "n": int(y.size),
        "ua": unweighted_accuracy(y, pred),
        "wa": float(np.mean(pred == y)),
        "macro_f1": macro_f1(y, pred),
        "ece": expected_calibration_error(conf, (pred == y).astype(float)),
        "confusion": confusion(y, pred),
        "per_class_recall": {EMOTIONS[c]: float(np.mean(pred[y == c] == c)) for c in np.unique(y)},
    }


def summarize(datasets: np.ndarray, labels: np.ndarray, probs: np.ndarray, dims_pred: np.ndarray,
              dims_true: np.ndarray) -> dict:
    """labels: class index or -1; probs: N x 7 softmax over all classes (masking is applied here)."""
    out: dict = {"per_dataset": {}}
    has = labels >= 0
    for ds in sorted(set(datasets.tolist())):
        sel = (datasets == ds) & has
        entry: dict = {}
        if sel.any():
            entry.update(_categorical(labels[sel], probs[sel], list(DATASET_CLASSES[ds])))
        d_sel = datasets == ds
        if np.isfinite(dims_true[d_sel]).any():
            entry["ccc"] = {d: ccc_np(dims_pred[d_sel, j], dims_true[d_sel, j]) for j, d in enumerate(DIMENSIONS)}
        out["per_dataset"][ds] = entry
    uas = [v["ua"] for v in out["per_dataset"].values() if "ua" in v]
    out["mean_dataset_ua"] = float(np.mean(uas)) if uas else float("nan")
    if has.any():
        # pooled over corpora, each clip argmaxed over its own corpus label set
        pooled = np.empty(int(has.sum()), dtype=int)
        for i, (ds, row) in enumerate(zip(datasets[has], probs[has])):
            pooled[i] = masked_argmax(row[None], list(DATASET_CLASSES[ds]))[0]
        y = labels[has]
        out["pooled"] = {"n": int(y.size), "ua": unweighted_accuracy(y, pooled), "wa": float(np.mean(pooled == y)),
                         "macro_f1": macro_f1(y, pooled), "confusion": confusion(y, pooled)}
        big = np.isin(y, [EMOTION_INDEX[e] for e in BIG4])
        if big.any():
            pb = masked_argmax(probs[has][big], list(BIG4))
            out["big4"] = {"n": int(big.sum()), "ua": unweighted_accuracy(y[big], pb), "wa": float(np.mean(pb == y[big]))}
    if np.isfinite(dims_true).any():
        out["ccc"] = {d: ccc_np(dims_pred[:, j], dims_true[:, j]) for j, d in enumerate(DIMENSIONS)}
    return out
