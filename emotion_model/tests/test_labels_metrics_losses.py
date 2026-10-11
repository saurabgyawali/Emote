import numpy as np
import pandas as pd
import pytest
import torch

from emotion_model.labels import DATASET_CLASSES, EMOTION_INDEX, EMOTIONS, MANIFEST_SCHEMA, class_mask, validate_manifest
from emotion_model.losses import ccc_loss, masked_cross_entropy
from emotion_model.metrics import macro_f1, masked_argmax, summarize, unweighted_accuracy


def manifest(**over):
    base = {"utt_id": "ravdess/a", "dataset": "ravdess", "path": "audio16k/ravdess/a.flac", "speaker": "ravdess_01",
            "gender": "M", "split": "train", "emotion": "happy", "arousal": np.nan, "valence": np.nan,
            "dominance": np.nan, "text": "", "duration_s": 2.0, "source_label": "happy"}
    base.update(over)
    return pd.DataFrame([base])[list(MANIFEST_SCHEMA)]


def test_every_dataset_label_set_is_inside_the_unified_space():
    for ds, classes in DATASET_CLASSES.items():
        assert set(classes) <= set(EMOTIONS)
        assert sum(class_mask(ds)) == len(classes)


def test_manifest_validation_catches_schema_breaks():
    validate_manifest(manifest())
    with pytest.raises(ValueError, match="outside its label set"):
        validate_manifest(manifest(dataset="crema_d", speaker="crema_d_1", emotion="surprise"))
    with pytest.raises(ValueError, match="0..1"):
        validate_manifest(manifest(arousal=3.0))
    with pytest.raises(ValueError, match="neither an emotion"):
        validate_manifest(manifest(emotion=""))
    two = pd.concat([manifest(), manifest(utt_id="tess/b", dataset="tess", speaker="ravdess_01")])
    with pytest.raises(ValueError, match="shared across datasets"):
        validate_manifest(two)


def test_masked_cross_entropy_ignores_classes_a_corpus_cannot_express():
    logits = torch.zeros(1, 7)
    logits[0, EMOTION_INDEX["surprise"]] = 50.0          # huge logit on a class CREMA-D does not have
    target = torch.tensor([EMOTION_INDEX["happy"]])
    mask = torch.tensor([class_mask("crema_d")])
    masked = masked_cross_entropy(logits, target, mask)
    unmasked = masked_cross_entropy(logits, target, torch.ones(1, 7, dtype=torch.bool))
    assert masked < 2.0 < unmasked
    assert masked_cross_entropy(logits, torch.tensor([-1]), mask) is None


def test_ccc_loss_is_zero_for_perfect_and_skips_missing_dimensions():
    y = torch.tensor([[0.1, np.nan, 0.5], [0.4, np.nan, 0.2], [0.8, np.nan, 0.9]], dtype=torch.float32)
    pred = torch.nan_to_num(y.clone(), nan=0.3)
    assert float(ccc_loss(pred, y)) == pytest.approx(0.0, abs=1e-5)
    assert ccc_loss(pred, torch.full_like(y, float("nan"))) is None


def test_unweighted_accuracy_is_mean_recall():
    y = np.array([0, 0, 0, 0, 1])
    p = np.array([0, 0, 0, 0, 0])
    assert unweighted_accuracy(y, p) == pytest.approx(0.5)
    assert macro_f1(y, y) == 1.0


def test_summarize_argmaxes_within_each_corpus_label_set():
    probs = np.zeros((2, 7))
    probs[:, EMOTION_INDEX["surprise"]] = 0.9
    probs[:, EMOTION_INDEX["happy"]] = 0.1
    labels = np.array([EMOTION_INDEX["happy"]] * 2)
    s = summarize(np.array(["crema_d", "crema_d"]), labels, probs, np.zeros((2, 3)), np.full((2, 3), np.nan))
    assert s["per_dataset"]["crema_d"]["ua"] == 1.0      # surprise is not a CREMA-D answer
    assert masked_argmax(probs, list(EMOTIONS))[0] == EMOTION_INDEX["surprise"]
