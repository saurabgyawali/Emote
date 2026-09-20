import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_baseline as tb  # noqa: E402


def _fake_features(tmp_path, n_per_cluster=10):
    rng = np.random.default_rng(0)
    rows = []
    centers = [{"f0_mean": 120, "rms_mean": 0.05}, {"f0_mean": 260, "rms_mean": 0.25}]
    for c, center in enumerate(centers):
        for i in range(n_per_cluster):
            rows.append({
                "session": "s1", "index": len(rows) + 1, "kept": True, "text": f"u{len(rows)}",
                "f0_mean": center["f0_mean"] + rng.normal(0, 5),
                "f0_std": 10 + rng.normal(0, 1),
                "rms_mean": center["rms_mean"] + rng.normal(0, 0.01),
                "speech_rate_wps": 2.0 + rng.normal(0, 0.1),
            })
    df = pd.DataFrame(rows)
    path = tmp_path / "features.csv"
    df.to_csv(path, index=False)
    return path


def test_insufficient_data_is_handled_gracefully(tmp_path):
    path = tmp_path / "features.csv"
    pd.DataFrame([{"session": "s1", "index": 1, "kept": True, "text": "hi", "f0_mean": 150}]).to_csv(path, index=False)
    rc = tb.main(["--features", str(path), "--out", str(tmp_path / "model"), "--min-rows", "8"])
    assert rc == 0
    status = json.loads((tmp_path / "model" / "STATUS.json").read_text())
    assert status["state"] == "insufficient_data"
    assert not (tmp_path / "model" / "model.joblib").exists()


def test_missing_features_file_fails_clearly(tmp_path):
    rc = tb.main(["--features", str(tmp_path / "nope.csv"), "--out", str(tmp_path / "model")])
    assert rc == 1


def test_clusters_separable_groups(tmp_path):
    features = _fake_features(tmp_path)
    out = tmp_path / "model"
    rc = tb.main(["--features", str(features), "--out", str(out), "--clusters", "2", "--min-rows", "5"])
    assert rc == 0
    assert (out / "model.joblib").exists()
    assignments = pd.read_csv(out / "cluster_assignments.csv")
    assert len(assignments) == 20
    summary = json.loads((out / "summary.json").read_text())
    assert summary["clusters"] == 2
    assert summary["silhouette_score"] > 0.3  # two well-separated synthetic clusters


def test_idempotent_skips_without_force(tmp_path, capsys):
    features = _fake_features(tmp_path)
    out = tmp_path / "model"
    assert tb.main(["--features", str(features), "--out", str(out), "--clusters", "2", "--min-rows", "5"]) == 0
    mtime1 = (out / "model.joblib").stat().st_mtime
    assert tb.main(["--features", str(features), "--out", str(out), "--clusters", "2", "--min-rows", "5"]) == 0
    assert (out / "model.joblib").stat().st_mtime == mtime1
    assert tb.main(["--features", str(features), "--out", str(out), "--clusters", "2", "--min-rows", "5", "--force"]) == 0


def test_feature_columns_excludes_metadata():
    df = pd.DataFrame([{"session": "s1", "index": 1, "start_s": 0.0, "end_s": 1.0, "kept": True, "text": "hi", "clip": "c.wav", "f0_mean": 150.0}])
    assert tb.feature_columns(df) == ["f0_mean"]
