"""Train one experiment x one seed, select on validation, score test at the best AND the last epoch.

    python train.py --experiment E5_audio_text_prosody --seed 0
    python train.py --experiment E3_audio_finetune --seed 1 --override train.epochs=10 model.audio.finetune_layers=6
    python train.py --experiment E3_audio_finetune --dry-run          # data plan only, nothing trained

Writes the checkpoint to <data root>/runs/<experiment>/seed<k>/best.pt and the small,
analysable artifacts to pipeline/results/<experiment>/seed<k>/:
  config.yaml          the fully resolved config (base + experiment + overrides)
  step_metrics.jsonl   loss / lr / grad norm / throughput every `log_every` optimizer steps
  epoch_metrics.jsonl  training loss and the full validation scorecard per epoch
  test_metrics.json    {"best": {val, test, heldout}, "last": {...}} -- never subtract across splits
  predictions.csv      per-clip probabilities, dimensions and modality weights (best epoch; test + heldout)
  train_summary.json   data counts, parameter counts, layer weights, timing, versions, fingerprint
A finished run whose config fingerprint matches is skipped; --force retrains.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXP_DIR = Path(__file__).resolve().parent
PIPELINE = EXP_DIR.parent.parent
EMOTE = PIPELINE.parent
sys.path.insert(0, str(EMOTE))
sys.path.insert(0, str(PIPELINE / "scripts" / "downloads"))

from _fetch import data_root, use_data_cache  # noqa: E402
from emotion_model.config import fingerprint, load_config  # noqa: E402
from emotion_model.labels import validate_manifest  # noqa: E402

CONFIGS = PIPELINE / "configs"
RESULTS = PIPELINE / "results"


def resolve_datasets(cfg: dict, manifest: pd.DataFrame) -> tuple[list[str], list[str]]:
    present = sorted(manifest["dataset"].unique())
    heldout = list(cfg["data"]["heldout_datasets"])
    absent = [d for d in heldout if d not in present]
    if absent:
        raise SystemExit(f"heldout datasets {absent} are not in the manifest ({present})")
    wanted = cfg["data"]["train_datasets"]
    train = [d for d in present if d not in heldout] if wanted == "auto" else list(wanted)
    absent = [d for d in train if d not in present]
    if absent:
        raise SystemExit(f"train datasets {absent} are not in the manifest ({present})")
    return train, heldout


def select_rows(manifest: pd.DataFrame, train_ds: list[str], heldout: list[str], limit: int, seed: int) -> dict:
    has_target = (manifest["emotion"].fillna("") != "") | manifest["arousal"].notna()
    m = manifest[has_target]
    parts = {
        "train": m[m["dataset"].isin(train_ds) & (m["split"] == "train")],
        "val": m[m["dataset"].isin(train_ds) & (m["split"] == "val")],
        "test": m[m["dataset"].isin(train_ds) & (m["split"] == "test")],
        "heldout": m[m["dataset"].isin(heldout)],
    }
    if limit:
        parts = {k: v.groupby("dataset", group_keys=False).apply(lambda g: g.sample(min(len(g), limit), random_state=seed))
                 for k, v in parts.items()}
    return {k: v.reset_index(drop=True) for k, v in parts.items()}


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def jsonl(path: Path, record: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(record, default=float) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--experiment", required=True, help="name of a file in configs/experiments/ (or a path)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--override", nargs="*", default=[], help="dotted overrides, e.g. train.epochs=5")
    p.add_argument("--data-root", default=None)
    p.add_argument("--results-dir", default=str(RESULTS))
    p.add_argument("--device", default=None, help="cuda | cuda:1 | cpu (default: cuda if available)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    root = data_root(args.data_root)
    use_data_cache(root)
    exp_path = Path(args.experiment)
    if not exp_path.exists():
        exp_path = CONFIGS / "experiments" / f"{args.experiment}.yaml"
    cfg = load_config(exp_path, args.override)
    cfg["seed"] = args.seed
    name = cfg["experiment"]
    fp = fingerprint(cfg)
    res_dir = Path(args.results_dir) / name / f"seed{args.seed}"
    run_dir = root / "runs" / name / f"seed{args.seed}"
    summary_path = res_dir / "train_summary.json"
    if (res_dir / "test_metrics.json").exists() and summary_path.exists() and not args.force:
        if json.loads(summary_path.read_text()).get("fingerprint") == fp:
            print(f"{name} seed {args.seed}: already trained (fingerprint {fp}); --force to retrain")
            return
        print(f"{name} seed {args.seed}: config changed since the last run; retraining")

    manifest = pd.read_csv(root / cfg["data"]["manifest"])
    validate_manifest(manifest)
    train_ds, heldout = resolve_datasets(cfg, manifest)
    rows = select_rows(manifest, train_ds, heldout, cfg["train"]["limit_per_split"], args.seed)
    leak = set(rows["train"]["speaker"]) & (set(rows["val"]["speaker"]) | set(rows["test"]["speaker"]))
    leak = {s for s in leak if not s.startswith("meld_")}
    if leak:
        raise SystemExit(f"speaker leakage between train and val/test: {sorted(leak)[:10]}")
    plan = {k: v.groupby("dataset").size().to_dict() for k, v in rows.items()}
    print(f"{name} seed {args.seed} (fingerprint {fp})\n  train on {train_ds}, held out {heldout}\n  clips: {plan}")
    if args.dry_run:
        return

    import torch

    from emotion_model.engine import autocast, load_prosody_table, make_loader, model_inputs, predict, \
        predictions_frame, score, warmup_cosine
    from emotion_model.data import class_weights
    from emotion_model.losses import ccc_loss, masked_cross_entropy
    from emotion_model.model import EmotionModel, count_parameters
    from emotion_model.prosody import PROSODY_FEATURES, fit_standardizer

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cpu":
        print("  WARNING: training on CPU. Expect ~50-100x slower than a GPU; fine only for smoke tests.")
    tcfg = cfg["train"]
    precision = tcfg["precision"]
    if device.type == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
        print("  bf16 unsupported on this GPU; using fp16")
        precision = "fp16"

    prosody, stats = None, None
    if cfg["model"]["prosody"]["enabled"]:
        every = set().union(*(set(v["utt_id"]) for v in rows.values()))
        prosody = load_prosody_table(root / cfg["data"]["prosody_table"], every)
        stats = fit_standardizer(np.stack([prosody[u] for u in rows["train"]["utt_id"]]))

    model = EmotionModel(cfg["model"], n_prosody=len(PROSODY_FEATURES)).to(device)
    params = count_parameters(model)
    print(f"  model: {model.modalities}, {params['total'] / 1e6:.1f} M parameters, {params['trainable'] / 1e6:.1f} M trainable")
    tok = model.tokenizer
    loaders = {k: make_loader(v, root, cfg, k == "train", tok, prosody, stats, args.seed) for k, v in rows.items() if len(v)}
    cw = torch.as_tensor(class_weights(rows["train"], tcfg["class_weight_power"]), device=device)

    opt = torch.optim.AdamW(model.parameter_groups(tcfg["lr_encoder"], tcfg["lr_head"], tcfg["weight_decay"]))
    steps_per_epoch = max(1, len(loaders["train"]) // tcfg["grad_accum"])
    total = steps_per_epoch * tcfg["epochs"]
    if tcfg["max_steps"]:
        total = min(total, tcfg["max_steps"])
    warmup = int(total * tcfg["warmup_ratio"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, total, warmup))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")

    if res_dir.exists():
        shutil.rmtree(res_dir)
    res_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    best_score, best_epoch, bad_epochs, step = -np.inf, -1, 0, 0
    t_start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stopped_early = False
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        t0, seen, run_loss, n_loss = time.time(), 0, 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(loaders["train"], 1):
            with autocast(device, precision):
                out = model(**model_inputs(batch, device))
            label = batch["label"].to(device)
            ce = masked_cross_entropy(out["logits"], label, batch["class_mask"].to(device), cw, tcfg["label_smoothing"])
            cc = ccc_loss(out["dims"].float(), batch["dims"].to(device))
            parts = [x for x in (ce, None if cc is None else tcfg["dim_loss_weight"] * cc) if x is not None]
            if not parts:
                continue
            loss = sum(parts) / tcfg["grad_accum"]
            scaler.scale(loss).backward()
            run_loss += float(loss) * tcfg["grad_accum"]
            n_loss += 1
            seen += len(batch["utt_id"])
            if i % tcfg["grad_accum"]:
                continue
            scaler.unscale_(opt)
            gnorm = float(torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], tcfg["grad_clip"]))
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % tcfg["log_every"] == 0:
                lrs = [g["lr"] for g in opt.param_groups]
                jsonl(res_dir / "step_metrics.jsonl", {
                    "step": step, "epoch": epoch, "loss": run_loss / max(1, n_loss),
                    "ce": None if ce is None else float(ce), "ccc_loss": None if cc is None else float(cc),
                    "lr_head": lrs[0], "lr_encoder": lrs[1] if len(lrs) > 1 else None, "grad_norm": gnorm,
                    "clips_per_s": seen / max(1e-6, time.time() - t0)})
            if tcfg["max_steps"] and step >= tcfg["max_steps"]:
                break
        val_pred = predict(model, loaders["val"], device, precision) if "val" in loaders else {}
        val = score(val_pred, model.modalities)
        current = val.get(tcfg["selection_metric"], np.nan) if val else -run_loss
        rec = {"epoch": epoch, "step": step, "train_loss": run_loss / max(1, n_loss), "epoch_s": time.time() - t0,
               "val": {k: v for k, v in val.items() if k != "per_dataset"},
               "val_per_dataset": {d: {k: v[k] for k in ("ua", "wa", "macro_f1", "ccc") if k in v}
                                   for d, v in val.get("per_dataset", {}).items()}}
        if "audio" in model.towers:
            rec["layer_weights"] = model.towers["audio"].layer_weights().tolist()
        jsonl(res_dir / "epoch_metrics.jsonl", rec)
        print(f"  epoch {epoch:2d}: train loss {rec['train_loss']:.4f}  val {tcfg['selection_metric']} {current:.4f}"
              f"  ({rec['epoch_s']:.0f}s)", flush=True)
        if np.isfinite(current) and current > best_score:
            best_score, best_epoch, bad_epochs = current, epoch, 0
            torch.save({"model": model.state_dict(), "config": cfg, "prosody_stats": stats, "epoch": epoch,
                        "val_score": current, "modalities": model.modalities}, run_dir / "best.pt")
        else:
            bad_epochs += 1
        if bad_epochs >= tcfg["patience"]:
            print(f"  early stop: no validation gain for {bad_epochs} epochs")
            stopped_early = True
            break
        if tcfg["max_steps"] and step >= tcfg["max_steps"]:
            break

    def evaluate_all(tag: str) -> tuple[dict, list[pd.DataFrame]]:
        res, frames = {}, []
        for split in ("val", "test", "heldout"):
            if split in loaders:
                pred = predict(model, loaders[split], device, precision)
                res[split] = score(pred, model.modalities)
                if split != "val":
                    frames.append(predictions_frame(pred, model.modalities, split))
        print(f"  [{tag}] " + "  ".join(f"{k} UA {v.get('mean_dataset_ua', float('nan')):.4f}" for k, v in res.items()))
        return res, frames

    metrics = {"best_epoch": best_epoch, "selection_metric": tcfg["selection_metric"]}
    metrics["last"], _ = evaluate_all("last epoch")
    if tcfg["keep_last_checkpoint"]:
        torch.save({"model": model.state_dict(), "config": cfg, "prosody_stats": stats, "epoch": epoch,
                    "modalities": model.modalities}, run_dir / "last.pt")
    if (run_dir / "best.pt").exists():
        ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
    metrics["best"], frames = evaluate_all(f"best epoch {best_epoch}")
    (res_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    if frames:
        pd.concat(frames).to_csv(res_dir / "predictions.csv", index=False)

    import transformers

    summary = {
        "experiment": name, "seed": args.seed, "fingerprint": fp, "train_datasets": train_ds, "heldout_datasets": heldout,
        "clips": plan, "parameters": params, "modalities": model.modalities, "best_epoch": best_epoch,
        "epochs_run": epoch, "stopped_early": stopped_early, "optimizer_steps": step, "wall_s": time.time() - t_start,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu", "precision": precision,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0,
        "layer_weights_best": model.towers["audio"].layer_weights().tolist() if "audio" in model.towers else None,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "checkpoint": str(run_dir / "best.pt"),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"  done in {summary['wall_s'] / 60:.1f} min; results -> {res_dir}")


if __name__ == "__main__":
    main()
