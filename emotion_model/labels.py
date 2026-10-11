"""One label space for every corpus, and the manifest schema every preprocessing step must produce.

Corpora disagree on which emotions they contain (CREMA-D has no surprise, IEMOCAP is
used with the standard four classes). Each dataset therefore declares its label set,
and the loss only normalises over that set: a CREMA-D clip is never punished for not
being "surprise", a class CREMA-D cannot express. Mapping decisions, all documented
in pipeline/datasets.yaml:

  RAVDESS  calm -> neutral (the usual merge; calm is a low-arousal neutral)
  TESS     ps (pleasant surprise) -> surprise
  MELD     joy -> happy, sadness -> sad, anger -> angry
  IEMOCAP  excited -> happy, then the standard four classes only
  MSP-Pod  contempt, other and no-agreement are dropped from the categorical task
"""

from __future__ import annotations

EMOTIONS: tuple[str, ...] = ("neutral", "happy", "sad", "angry", "fear", "disgust", "surprise")
BIG4: tuple[str, ...] = ("neutral", "happy", "sad", "angry")
DIMENSIONS: tuple[str, ...] = ("arousal", "valence", "dominance")
EMOTION_INDEX = {e: i for i, e in enumerate(EMOTIONS)}

DATASET_CLASSES: dict[str, tuple[str, ...]] = {
    "ravdess": EMOTIONS,
    "crema_d": ("neutral", "happy", "sad", "angry", "fear", "disgust"),
    "tess": EMOTIONS,
    "meld": EMOTIONS,
    "iemocap": BIG4,
    "msp_podcast": EMOTIONS,
}

SPLITS = ("train", "val", "test")

# column -> (dtype kind, required non-empty)
MANIFEST_SCHEMA: dict[str, tuple[str, bool]] = {
    "utt_id": ("str", True),        # "<dataset>/<original stem>", unique
    "dataset": ("str", True),       # a key of DATASET_CLASSES
    "path": ("str", True),          # 16 kHz mono audio, relative to the data root
    "speaker": ("str", True),       # "<dataset>_<speaker>", never shared between datasets
    "gender": ("str", True),        # M | F | U
    "split": ("str", True),         # train | val | test, speaker-independent where the corpus allows
    "emotion": ("str", False),      # one of EMOTIONS, or "" when the clip has only dimensional labels
    "arousal": ("float", False),    # 0..1 or NaN
    "valence": ("float", False),
    "dominance": ("float", False),
    "text": ("str", False),         # transcript, "" when unknown
    "duration_s": ("float", True),
    "source_label": ("str", True),  # the corpus's own label, kept for audits
}


def class_mask(dataset: str) -> list[bool]:
    allowed = set(DATASET_CLASSES[dataset])
    return [e in allowed for e in EMOTIONS]


def validate_manifest(df) -> None:  # noqa: ANN001 - pandas DataFrame
    """Fail loudly, with every problem listed, if a manifest breaks the schema."""
    import numpy as np

    problems: list[str] = []
    missing = [c for c in MANIFEST_SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"manifest is missing columns {missing}; has {list(df.columns)}")
    for col, (kind, required) in MANIFEST_SCHEMA.items():
        if required and kind == "str" and (df[col].isna() | (df[col].astype(str) == "")).any():
            problems.append(f"{col}: {int((df[col].isna() | (df[col].astype(str) == '')).sum())} empty value(s)")
    if df["utt_id"].duplicated().any():
        problems.append(f"utt_id: {int(df['utt_id'].duplicated().sum())} duplicate(s), e.g. {df.loc[df['utt_id'].duplicated(), 'utt_id'].iloc[0]}")
    bad_ds = sorted(set(df["dataset"]) - set(DATASET_CLASSES))
    if bad_ds:
        problems.append(f"dataset: unknown {bad_ds}")
    bad_split = sorted(set(df["split"]) - set(SPLITS))
    if bad_split:
        problems.append(f"split: unknown {bad_split}")
    bad_gender = sorted(set(df["gender"]) - {"M", "F", "U"})
    if bad_gender:
        problems.append(f"gender: unknown {bad_gender}")
    emo = df["emotion"].fillna("")
    bad_emo = sorted(set(emo) - set(EMOTIONS) - {""})
    if bad_emo:
        problems.append(f"emotion: unknown {bad_emo}")
    for ds, allowed in DATASET_CLASSES.items():
        sub = emo[(df["dataset"] == ds) & (emo != "")]
        outside = sorted(set(sub) - set(allowed))
        if outside:
            problems.append(f"emotion: {ds} uses {outside}, outside its label set {allowed}")
    dims = df[list(DIMENSIONS)].to_numpy(dtype=float)
    finite = dims[np.isfinite(dims)]
    if finite.size and (finite.min() < 0 or finite.max() > 1):
        problems.append(f"dimensions must be scaled to 0..1, found {finite.min():.3f}..{finite.max():.3f}")
    no_target = (emo == "") & ~np.isfinite(dims).any(axis=1)
    if no_target.any():
        problems.append(f"{int(no_target.sum())} row(s) have neither an emotion nor a dimension")
    if (df["duration_s"].astype(float) <= 0).any():
        problems.append(f"duration_s: {int((df['duration_s'].astype(float) <= 0).sum())} non-positive")
    speakers = df.groupby("speaker")["dataset"].nunique()
    if (speakers > 1).any():
        problems.append(f"speaker ids shared across datasets: {list(speakers[speakers > 1].index[:5])}")
    if problems:
        raise ValueError("manifest failed validation:\n  - " + "\n  - ".join(problems))
