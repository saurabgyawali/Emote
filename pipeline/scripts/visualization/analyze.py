"""The analysis pipeline: every run's raw JSON/CSV -> summary tables, paired contrasts, figures, RESULTS.md.

    python analyze.py                       # pipeline/results/** -> results/summary.json, results/RESULTS.md, figures/
    python analyze.py --data-root DIR       # also draws the dataset overview from the manifest

Every headline number is recomputed here from per-seed test_metrics.json and
predictions.csv, never copied from a console log. Contrasts are paired by seed and
tested with an exact sign-flip permutation test; the smallest p that test can return
with n seeds (2 / 2^n) is printed beside every p, because with 3 seeds it is 0.25.

Figures (pipeline/figures/):
  fig01_dataset_overview      clips per corpus and emotion; clip durations
  fig02_training_curves       train loss and validation UA per epoch, every seed
  fig03_ablation              test UA per experiment, best vs last epoch, seeds as dots
  fig04_per_dataset           test UA per corpus per experiment
  fig05_confusion             row-normalised confusion of the best experiment, pooled and per corpus
  fig06_per_class_recall      per-emotion recall of the best experiment
  fig07_layer_weights         which encoder layers the audio tower relies on
  fig08_modality_weights      fusion gate weight per modality, per corpus
  fig09_calibration           reliability diagram of the best experiment
  fig10_cross_corpus          held-out-corpus UA against in-corpus UA
  fig11_dimensions            arousal / valence / dominance CCC (only with IEMOCAP or MSP-Podcast)
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
PIPELINE = EXP_DIR.parent.parent
EMOTE = PIPELINE.parent
sys.path.insert(0, str(EMOTE))
sys.path.insert(0, str(PIPELINE / "scripts" / "downloads"))

from _fetch import data_root  # noqa: E402
from emotion_model.labels import DATASET_CLASSES, DIMENSIONS, EMOTIONS  # noqa: E402

RESULTS = PIPELINE / "results"
FIGURES = PIPELINE / "figures"
JETSON_INFEASIBLE = {"E8_wavlm_large_full"}
CONTRASTS = [  # (question, experiment A, experiment B): is A better than B?
    ("Does fine-tuning the encoder help?", "E3_audio_finetune", "E2_audio_frozen"),
    ("Does the transcript help?", "E4_audio_text", "E3_audio_finetune"),
    ("Does hand-crafted prosody add to audio + text?", "E5_audio_text_prosody", "E4_audio_text"),
    ("Does the neural model beat hand-crafted features?", "E3_audio_finetune", "E1_prosody_only"),
    ("WavLM vs HuBERT", "E3_audio_finetune", "E6_hubert_finetune"),
    ("WavLM vs wav2vec 2.0", "E3_audio_finetune", "E7_wav2vec2_finetune"),
    ("Does WavLM-Large beat the deployable model?", "E8_wavlm_large_full", "E5_audio_text_prosody"),
]


# --- loading ---------------------------------------------------------------------
def load_runs(results: Path) -> list[dict]:
    runs = []
    for tm in sorted(results.glob("*/seed*/test_metrics.json")):
        d = tm.parent
        rec = {"experiment": d.parent.name, "seed": int(d.name[4:]), "dir": d,
               "metrics": json.loads(tm.read_text())}
        if (d / "train_summary.json").exists():
            rec["summary"] = json.loads((d / "train_summary.json").read_text())
        if (d / "epoch_metrics.jsonl").exists():
            rec["epochs"] = [json.loads(line) for line in (d / "epoch_metrics.jsonl").read_text().splitlines() if line]
        if (d / "predictions.csv").exists():
            rec["pred"] = pd.read_csv(d / "predictions.csv", keep_default_na=False)
        runs.append(rec)
    return runs


def _get(m: dict, *path, default=np.nan):  # noqa: ANN001
    for p in path:
        if not isinstance(m, dict) or p not in m:
            return default
        m = m[p]
    return m


def mean_sd(values: list[float]) -> dict:
    v = np.array([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    return {"mean": float(v.mean()) if v.size else np.nan, "sd": float(v.std(ddof=1)) if v.size > 1 else np.nan,
            "n": int(v.size), "values": v.round(4).tolist()}


def exact_paired_p(diffs: list[float]) -> tuple[float, float]:
    """Two-sided exact sign-flip permutation p and its floor (2 / 2^n)."""
    d = np.asarray(diffs, dtype=float)
    n = d.size
    if n == 0:
        return np.nan, np.nan
    obs = abs(d.mean())
    flips = np.array(list(itertools.product((1, -1), repeat=n)))
    null = np.abs((flips * d).mean(axis=1))
    return float(np.mean(null >= obs - 1e-12)), 2 / 2**n


# --- tables ----------------------------------------------------------------------
def summarize_experiments(runs: list[dict]) -> dict:
    out: dict = {}
    for exp in sorted({r["experiment"] for r in runs}):
        rs = sorted([r for r in runs if r["experiment"] == exp], key=lambda r: r["seed"])
        e: dict = {"seeds": [r["seed"] for r in rs]}
        for which in ("best", "last"):
            for split in ("val", "test", "heldout"):
                key = f"{which}_{split}"
                vals = [_get(r["metrics"], which, split, "mean_dataset_ua") for r in rs]
                if any(np.isfinite(v) for v in vals):
                    e[key] = {
                        "mean_dataset_ua": mean_sd(vals),
                        "pooled_ua": mean_sd([_get(r["metrics"], which, split, "pooled", "ua") for r in rs]),
                        "pooled_wa": mean_sd([_get(r["metrics"], which, split, "pooled", "wa") for r in rs]),
                        "pooled_macro_f1": mean_sd([_get(r["metrics"], which, split, "pooled", "macro_f1") for r in rs]),
                        "big4_ua": mean_sd([_get(r["metrics"], which, split, "big4", "ua") for r in rs]),
                        "per_dataset_ua": {ds: mean_sd([_get(r["metrics"], which, split, "per_dataset", ds, "ua") for r in rs])
                                           for ds in sorted({d for r in rs for d in _get(r["metrics"], which, split, "per_dataset", default={})})},
                    }
                    ccc = [_get(r["metrics"], which, split, "ccc", default=None) for r in rs]
                    if any(c for c in ccc):
                        e[key]["ccc"] = {d: mean_sd([c.get(d, np.nan) if c else np.nan for c in ccc]) for d in DIMENSIONS}
        sums = [r.get("summary", {}) for r in rs]
        e["parameters_m"] = _get(sums[0], "parameters", "total", default=0) / 1e6
        e["trainable_m"] = _get(sums[0], "parameters", "trainable", default=0) / 1e6
        e["train_minutes"] = mean_sd([s.get("wall_s", np.nan) / 60 for s in sums])
        e["peak_gpu_gb"] = mean_sd([s.get("peak_gpu_mem_gb", np.nan) for s in sums])
        e["best_epoch"] = [r["metrics"].get("best_epoch") for r in rs]
        e["modalities"] = sums[0].get("modalities", [])
        e["heldout_datasets"] = sums[0].get("heldout_datasets", [])
        out[exp] = e
    return out


def contrasts(runs: list[dict]) -> list[dict]:
    by = {(r["experiment"], r["seed"]): _get(r["metrics"], "best", "test", "mean_dataset_ua") for r in runs}
    out = []
    for question, a, b in CONTRASTS:
        seeds = sorted({s for (e, s) in by if e == a} & {s for (e, s) in by if e == b})
        if not seeds:
            continue
        diffs = [by[(a, s)] - by[(b, s)] for s in seeds]
        p, floor = exact_paired_p(diffs)
        out.append({"question": question, "a": a, "b": b, "seeds": seeds, "mean_diff": float(np.mean(diffs)),
                    "per_seed_diff": np.round(diffs, 4).tolist(), "p_exact": p, "p_floor": floor,
                    "a_wins_every_seed": bool(all(d > 0 for d in diffs))})
    return out


def pick_best(summary: dict, deployable: bool) -> str | None:
    cands = {e: v["best_test"]["mean_dataset_ua"]["mean"] for e, v in summary.items()
             if "best_test" in v and not v["heldout_datasets"] and (not deployable or e not in JETSON_INFEASIBLE)}
    cands = {e: s for e, s in cands.items() if np.isfinite(s)}
    return max(cands, key=cands.get) if cands else None


# --- figures ---------------------------------------------------------------------
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9})
    return plt


def fig_dataset_overview(manifest: pd.DataFrame, out: Path) -> None:
    plt = _plt()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
    counts = manifest[manifest["emotion"].fillna("") != ""].groupby(["dataset", "emotion"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=[e for e in EMOTIONS if e in counts.columns])
    counts.plot(kind="bar", stacked=True, ax=a, colormap="tab10", width=0.75)
    a.set_title("Clips per corpus and emotion")
    a.set_ylabel("clips")
    a.legend(fontsize=7, ncol=2)
    for ds, sub in manifest.groupby("dataset"):
        b.hist(sub["duration_s"].clip(upper=15), bins=40, histtype="step", label=ds)
    b.set_title("Clip duration (clipped at 15 s)")
    b.set_xlabel("seconds")
    b.legend(fontsize=7)
    fig.savefig(out)
    plt.close(fig)


def fig_training_curves(runs: list[dict], out: Path) -> None:
    plt = _plt()
    exps = sorted({r["experiment"] for r in runs if r.get("epochs")})
    if not exps:
        return
    fig, axes = plt.subplots(2, len(exps), figsize=(3.2 * len(exps), 5.2), squeeze=False)
    for j, exp in enumerate(exps):
        for r in [r for r in runs if r["experiment"] == exp and r.get("epochs")]:
            ep = [e["epoch"] for e in r["epochs"]]
            axes[0, j].plot(ep, [e["train_loss"] for e in r["epochs"]], lw=1, label=f"seed {r['seed']}")
            axes[1, j].plot(ep, [e["val"].get("mean_dataset_ua", np.nan) for e in r["epochs"]], lw=1)
            be = r["metrics"].get("best_epoch")
            if be and be > 0:
                axes[1, j].axvline(be, color="grey", lw=0.5, ls=":")
        axes[0, j].set_title(exp, fontsize=8)
        axes[1, j].set_xlabel("epoch")
    axes[0, 0].set_ylabel("train loss")
    axes[1, 0].set_ylabel("val mean-corpus UA")
    axes[0, 0].legend(fontsize=6)
    fig.savefig(out)
    plt.close(fig)


def fig_ablation(summary: dict, out: Path) -> None:
    plt = _plt()
    exps = [e for e in summary if "best_test" in summary[e]]
    if not exps:
        return
    fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(exps)), 3.8))
    x = np.arange(len(exps))
    for k, (which, off, color) in enumerate((("best_test", -0.18, "C0"), ("last_test", 0.18, "C7"))):
        m = [summary[e].get(which, {}).get("mean_dataset_ua", {}).get("mean", np.nan) for e in exps]
        s = [summary[e].get(which, {}).get("mean_dataset_ua", {}).get("sd", np.nan) for e in exps]
        ax.bar(x + off, m, 0.34, yerr=np.nan_to_num(s), color=color, alpha=0.8,
               label="best epoch (val-selected)" if k == 0 else "last epoch", capsize=2)
        for i, e in enumerate(exps):
            vals = summary[e].get(which, {}).get("mean_dataset_ua", {}).get("values", [])
            ax.scatter(np.full(len(vals), x[i] + off), vals, s=8, color="k", zorder=3)
    ax.set_xticks(x, exps, rotation=30, ha="right")
    ax.set_ylabel("test UA (mean over corpora)")
    ax.set_title("Ablation ladder (dots = seeds)")
    ax.legend(fontsize=7)
    fig.savefig(out)
    plt.close(fig)


def fig_per_dataset(summary: dict, out: Path) -> None:
    plt = _plt()
    exps = [e for e in summary if "best_test" in summary[e]]
    datasets = sorted({d for e in exps for d in summary[e]["best_test"]["per_dataset_ua"]})
    if not exps or not datasets:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(datasets) * max(1, len(exps)) / 3), 3.8))
    w = 0.8 / len(exps)
    for i, e in enumerate(exps):
        m = [summary[e]["best_test"]["per_dataset_ua"].get(d, {}).get("mean", np.nan) for d in datasets]
        s = [summary[e]["best_test"]["per_dataset_ua"].get(d, {}).get("sd", np.nan) for d in datasets]
        ax.bar(np.arange(len(datasets)) + i * w - 0.4 + w / 2, m, w, yerr=np.nan_to_num(s), label=e, capsize=1.5)
    for k, d in enumerate(datasets):
        ax.hlines(1 / len(DATASET_CLASSES[d]), k - 0.42, k + 0.42, color="red", lw=0.8, ls="--")
    ax.set_xticks(range(len(datasets)), datasets)
    ax.set_ylabel("test UA")
    ax.set_title("Per corpus (red dashes = chance)")
    ax.legend(fontsize=6, ncol=2)
    fig.savefig(out)
    plt.close(fig)


def _pred_for(runs: list[dict], exp: str, kind: str = "test") -> pd.DataFrame:
    frames = [r["pred"] for r in runs if r["experiment"] == exp and "pred" in r]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    return df[(df["kind"] == kind) & (df["label"] != "")]


def fig_confusion(runs: list[dict], exp: str, out: Path) -> None:
    plt = _plt()
    df = _pred_for(runs, exp)
    if df.empty:
        return
    panels = [("pooled", df)] + [(d, g) for d, g in df.groupby("dataset")]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.3), squeeze=False)
    for ax, (title, g) in zip(axes[0], panels):
        labels = [e for e in EMOTIONS if e in set(g["label"]) | set(g["pred"])]
        m = pd.crosstab(g["label"], g["pred"]).reindex(index=labels, columns=labels, fill_value=0).to_numpy(dtype=float)
        m = m / np.clip(m.sum(axis=1, keepdims=True), 1, None)
        ax.imshow(m, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(labels)), [l[:3] for l in labels], fontsize=7)
        ax.set_yticks(range(len(labels)), [l[:3] for l in labels], fontsize=7)
        for i, j in itertools.product(range(len(labels)), range(len(labels))):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="white" if m[i, j] > 0.5 else "black")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("predicted")
    axes[0, 0].set_ylabel("true")
    fig.suptitle(f"{exp}: test confusion, all seeds pooled", fontsize=9)
    fig.savefig(out)
    plt.close(fig)


def fig_per_class(runs: list[dict], exp: str, out: Path) -> None:
    plt = _plt()
    df = _pred_for(runs, exp)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for i, (ds, g) in enumerate(df.groupby("dataset")):
        rec = [np.mean(g.loc[g["label"] == e, "pred"] == e) if (g["label"] == e).any() else np.nan for e in EMOTIONS]
        ax.bar(np.arange(len(EMOTIONS)) + i * 0.8 / df["dataset"].nunique(), rec, 0.8 / df["dataset"].nunique(), label=ds)
    ax.set_xticks(np.arange(len(EMOTIONS)) + 0.3, EMOTIONS, rotation=20)
    ax.set_ylabel("recall")
    ax.set_title(f"{exp}: per-emotion recall (test)")
    ax.legend(fontsize=7)
    fig.savefig(out)
    plt.close(fig)


def fig_layer_weights(runs: list[dict], out: Path) -> None:
    plt = _plt()
    by: dict[str, list] = {}
    for r in runs:
        lw = r.get("summary", {}).get("layer_weights_best")
        if lw:
            by.setdefault(r["experiment"], []).append(lw)
    if not by:
        return
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for exp, ws in sorted(by.items()):
        n = min(len(w) for w in ws)
        arr = np.array([w[:n] for w in ws])
        ax.plot(range(n), arr.mean(0), marker="o", ms=3, label=exp)
    ax.set_xlabel("encoder layer (0 = CNN features)")
    ax.set_ylabel("learned weight")
    ax.set_title("Where the emotion information sits in the speech encoder")
    ax.legend(fontsize=6)
    fig.savefig(out)
    plt.close(fig)


def fig_modality(runs: list[dict], out: Path) -> None:
    plt = _plt()
    rows = []
    for r in runs:
        p = r.get("pred")
        if p is None:
            continue
        cols = [c for c in p.columns if c.startswith("alpha_")]
        if len(cols) < 2:
            continue
        for ds, g in p[p["kind"] == "test"].groupby("dataset"):
            rows.append({"experiment": r["experiment"], "dataset": ds, **{c[6:]: g[c].mean() for c in cols}})
    if not rows:
        return
    df = pd.DataFrame(rows).groupby(["experiment", "dataset"]).mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(df)), 3.2))
    df.plot(kind="bar", stacked=True, ax=ax, width=0.75)
    ax.set_ylabel("mean fusion weight")
    ax.set_title("How much each modality is used, per corpus (test)")
    ax.set_xticklabels([f"{e}\n{d}" for e, d in df.index], rotation=45, ha="right", fontsize=6)
    fig.savefig(out)
    plt.close(fig)


def fig_calibration(runs: list[dict], exp: str, out: Path) -> None:
    plt = _plt()
    df = _pred_for(runs, exp)
    if df.empty:
        return
    probs = df[[f"p_{e}" for e in EMOTIONS]].to_numpy(dtype=float)
    conf = np.array([probs[i, EMOTIONS.index(p)] / probs[i, [EMOTIONS.index(c) for c in DATASET_CLASSES[d]]].sum()
                     for i, (p, d) in enumerate(zip(df["pred"], df["dataset"]))])
    correct = (df["pred"] == df["label"]).to_numpy(dtype=float)
    edges = np.linspace(0, 1, 11)
    mids, accs, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.sum():
            mids.append(conf[sel].mean())
            accs.append(correct[sel].mean())
            ns.append(sel.sum())
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.scatter(mids, accs, s=np.array(ns) / max(ns) * 120 + 10)
    ax.plot(mids, accs, lw=1)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{exp}: calibration (test)", fontsize=8)
    fig.savefig(out)
    plt.close(fig)


def fig_cross_corpus(summary: dict, out: Path) -> None:
    plt = _plt()
    xs = {e: v for e, v in summary.items() if v["heldout_datasets"] and "best_heldout" in v}
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(5, 3.2))
    labels, inc, held = [], [], []
    for e, v in xs.items():
        labels.append(f"{e}\n(held out: {', '.join(v['heldout_datasets'])})")
        inc.append(v.get("best_test", {}).get("mean_dataset_ua", {}).get("mean", np.nan))
        held.append(v["best_heldout"]["mean_dataset_ua"]["mean"])
    x = np.arange(len(labels))
    ax.bar(x - 0.18, inc, 0.34, label="in-corpus test")
    ax.bar(x + 0.18, held, 0.34, label="unseen corpus")
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("UA")
    ax.set_title("Generalisation to a corpus never trained on")
    ax.legend(fontsize=7)
    fig.savefig(out)
    plt.close(fig)


def fig_dimensions(summary: dict, out: Path) -> None:
    plt = _plt()
    exps = [e for e, v in summary.items() if "ccc" in v.get("best_test", {})]
    if not exps:
        return
    fig, ax = plt.subplots(figsize=(5, 3.2))
    w = 0.8 / len(exps)
    for i, e in enumerate(exps):
        c = summary[e]["best_test"]["ccc"]
        ax.bar(np.arange(3) + i * w - 0.4 + w / 2, [c[d]["mean"] for d in DIMENSIONS], w, label=e)
    ax.set_xticks(range(3), DIMENSIONS)
    ax.set_ylabel("CCC")
    ax.set_title("Dimensional emotion (test)")
    ax.legend(fontsize=6)
    fig.savefig(out)
    plt.close(fig)


# --- report ----------------------------------------------------------------------
def fmt(ms: dict | None) -> str:
    if not ms or not np.isfinite(ms.get("mean", np.nan)):
        return "--"
    sd = ms.get("sd", np.nan)
    return f"{ms['mean']:.3f}" + (f" ± {sd:.3f}" if np.isfinite(sd) else "") + f" (n={ms['n']})"


def write_report(summary: dict, cons: list[dict], best: str | None, deploy: str | None, quality: dict | None, out: Path) -> None:
    lines = ["# Emote emotion model: results", "",
             "Generated by `pipeline/scripts/visualization/analyze.py` from the per-seed JSONs in `pipeline/results/`. "
             "UA = unweighted accuracy (mean per-class recall), averaged over corpora; ± is the SD over seeds. "
             "Test is scored at the validation-selected epoch (best) and at the final epoch (last).", ""]
    if quality:
        lines += ["## Data", "", f"{quality.get('clips', '?')} clips, {quality.get('hours', '?')} h.", "",
                  "| corpus | clips | speakers | splits | with transcript | with dimensions |", "|---|---|---|---|---|---|"]
        for ds, d in quality.get("datasets", {}).items():
            lines.append(f"| {ds} | {d['clips']} | {d['speakers']} | {d['splits']} | {d['with_text']} | {d['with_dimensions']} |")
        lines.append("")
    lines += ["## Experiments", "", "| experiment | test UA (best) | test UA (last) | pooled WA | macro-F1 | BIG4 UA | held-out UA | params (M) | minutes/run |",
              "|---|---|---|---|---|---|---|---|---|"]
    for e, v in summary.items():
        bt = v.get("best_test", {})
        lines.append(f"| {e} | {fmt(bt.get('mean_dataset_ua'))} | {fmt(v.get('last_test', {}).get('mean_dataset_ua'))} | "
                     f"{fmt(bt.get('pooled_wa'))} | {fmt(bt.get('pooled_macro_f1'))} | {fmt(bt.get('big4_ua'))} | "
                     f"{fmt(v.get('best_heldout', {}).get('mean_dataset_ua'))} | {v['parameters_m']:.1f} | {fmt(v['train_minutes'])} |")
    lines += ["", "## Paired contrasts (exact sign-flip permutation over seeds)", "",
              "| question | A - B (test UA) | per seed | p | floor | A wins every seed |", "|---|---|---|---|---|---|"]
    for c in cons:
        lines.append(f"| {c['question']} ({c['a']} vs {c['b']}) | {c['mean_diff']:+.4f} | {c['per_seed_diff']} | "
                     f"{c['p_exact']:.3f} | {c['p_floor']:.3f} | {'yes' if c['a_wins_every_seed'] else 'no'} |")
    lines += ["", f"Best experiment overall: **{best or '--'}**. Best deployable on the Jetson: **{deploy or '--'}** "
              "(export it with `bash pipeline/run.sh export`).", "", "Figures: `pipeline/figures/`."]
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-dir", default=str(RESULTS))
    p.add_argument("--figures-dir", default=str(FIGURES))
    p.add_argument("--data-root", default=None)
    args = p.parse_args()
    results, figures = Path(args.results_dir), Path(args.figures_dir)
    figures.mkdir(parents=True, exist_ok=True)
    runs = load_runs(results)
    quality_path = results / "quality_checks.json"
    quality = json.loads(quality_path.read_text()) if quality_path.exists() else None
    manifest_path = data_root(args.data_root) / "manifests" / "manifest.csv"
    if manifest_path.exists():
        fig_dataset_overview(pd.read_csv(manifest_path), figures / "fig01_dataset_overview.png")
    if not runs:
        print(f"no finished runs under {results}; nothing else to analyse")
        return
    summary = summarize_experiments(runs)
    cons = contrasts(runs)
    best, deploy = pick_best(summary, False), pick_best(summary, True)
    fig_training_curves(runs, figures / "fig02_training_curves.png")
    fig_ablation(summary, figures / "fig03_ablation.png")
    fig_per_dataset(summary, figures / "fig04_per_dataset.png")
    if best:
        fig_confusion(runs, best, figures / "fig05_confusion.png")
        fig_per_class(runs, best, figures / "fig06_per_class_recall.png")
        fig_calibration(runs, best, figures / "fig09_calibration.png")
    fig_layer_weights(runs, figures / "fig07_layer_weights.png")
    fig_modality(runs, figures / "fig08_modality_weights.png")
    fig_cross_corpus(summary, figures / "fig10_cross_corpus.png")
    fig_dimensions(summary, figures / "fig11_dimensions.png")
    (results / "summary.json").write_text(json.dumps({"experiments": summary, "contrasts": cons, "best": best,
                                                      "best_deployable": deploy}, indent=2, default=float))
    write_report(summary, cons, best, deploy, quality, results / "RESULTS.md")
    print(f"analysed {len(runs)} run(s) of {len(summary)} experiment(s); best {best}, deployable {deploy}")
    print(f"  {results / 'RESULTS.md'}\n  {results / 'summary.json'}\n  {figures}/")


if __name__ == "__main__":
    main()
