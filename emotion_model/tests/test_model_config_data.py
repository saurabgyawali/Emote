import numpy as np
import pandas as pd
import pytest
import torch

from emotion_model.config import apply_overrides, fingerprint, load_config
from emotion_model.data import class_weights, make_collate, sample_weights
from emotion_model.model import TINY_AUDIO, TINY_TEXT, ByteTokenizer, EmotionModel
from emotion_model.prosody import PROSODY_FEATURES
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[2] / "pipeline" / "configs"


def tiny_cfg(audio=True, text=True, prosody=True, finetune=1):
    return {"embed_dim": 16, "dropout": 0.0, "modality_dropout": 0.5,
            "audio": {"enabled": audio, "encoder": TINY_AUDIO, "finetune_layers": finetune, "spec_augment": False},
            "text": {"enabled": text, "encoder": TINY_TEXT, "finetune_layers": 1},
            "prosody": {"enabled": prosody, "hidden": 8}}


def batch(n=3, seconds=1.0):
    t = int(16000 * seconds)
    items = [{"wave": np.random.randn(t - 1000 * i).astype(np.float32) * 0.1, "text": ["hi", "", "so happy"][i % 3],
              "label": i % 7, "class_mask": np.ones(7, bool), "dims": np.full(3, np.nan, np.float32),
              "dataset": "ravdess", "utt_id": f"u{i}", "prosody": np.zeros(len(PROSODY_FEATURES), np.float32)}
             for i in range(n)]
    return make_collate(ByteTokenizer(), 32)(items)


def test_experiment_configs_load_and_are_named_after_their_own_file():
    for path in sorted((CONFIGS / "experiments").glob("*.yaml")):
        cfg = load_config(path)
        assert cfg["experiment"] == path.stem
        assert cfg["model"]["audio"]["encoder"]
    e5 = load_config(CONFIGS / "experiments" / "E5_audio_text_prosody.yaml")
    assert e5["model"]["text"]["enabled"] and e5["model"]["prosody"]["enabled"]
    assert e5["train"]["epochs"] == load_config(CONFIGS / "base.yaml")["train"]["epochs"]


def test_overrides_are_typed_and_unknown_keys_fail():
    cfg = apply_overrides({"train": {"epochs": 1, "lr": 0.1}}, ["train.epochs=5", "train.lr=1e-3"])
    assert cfg["train"]["epochs"] == 5 and cfg["train"]["lr"] == pytest.approx(1e-3)
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["train.epoch=5"])
    a, b = dict(cfg, seed=0), dict(cfg, seed=1)
    assert fingerprint(a) == fingerprint(b)


def test_full_model_forward_backward_and_modality_gate():
    torch.manual_seed(0)
    model = EmotionModel(tiny_cfg())
    b = batch()
    model.train()
    out = model(wave=b["wave"], wave_mask=b["wave_mask"], input_ids=b["input_ids"], text_mask=b["text_mask"],
                has_text=b["has_text"], prosody=b["prosody"])
    assert out["logits"].shape == (3, 7) and out["dims"].shape == (3, 3)
    assert torch.allclose(out["alpha"].sum(1), torch.ones(3), atol=1e-5)
    out["logits"].sum().backward()
    assert model.towers["audio"].layer_logits.grad is not None
    model.eval()
    out = model(wave=b["wave"], wave_mask=b["wave_mask"], input_ids=b["input_ids"], text_mask=b["text_mask"],
                has_text=b["has_text"], prosody=b["prosody"])
    assert float(out["alpha"][1, model.modalities.index("text")]) < 1e-3   # the clip with no transcript


def test_frozen_encoder_has_no_trainable_encoder_weights():
    model = EmotionModel(tiny_cfg(text=False, prosody=False, finetune=0))
    enc = [p for n, p in model.named_parameters() if n.startswith("towers.audio.encoder.")]
    assert enc and not any(p.requires_grad for p in enc)
    groups = model.parameter_groups(1e-5, 1e-3, 0.0)
    assert len(groups) == 1   # only the new head parameters


def test_samplers_balance_corpora_and_classes():
    rows = pd.DataFrame({"dataset": ["a"] * 90 + ["b"] * 10, "emotion": ["happy"] * 80 + ["sad"] * 10 + ["happy"] * 10})
    w = sample_weights(rows, "dataset")
    assert w[:90].sum() == pytest.approx(w[90:].sum())
    w2 = sample_weights(rows, "dataset_class", power=1.0)
    a = rows["dataset"] == "a"
    assert w2[a & (rows["emotion"] == "happy")].sum() == pytest.approx(w2[a & (rows["emotion"] == "sad")].sum())
    cw = class_weights(rows)
    assert cw[3] == 0 and cw[2] > cw[1] > 0       # angry absent; sad rarer than happy
