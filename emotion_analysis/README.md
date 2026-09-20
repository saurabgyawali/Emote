# emotion_analysis

Feature extraction over transcriber sessions. Input: `../transcriber/sessions/*/`
(`utterances.jsonl` + `clips/*.wav`). Output: `results/features.csv` (+ `.parquet`),
one row per kept utterance.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/extract_features.py            # idempotent: appends only new utterances
.venv/bin/python scripts/extract_features.py --force    # recompute all
.venv/bin/python scripts/summarize_sessions.py          # per-session table
.venv/bin/python -m pytest -q tests
```

## Feature groups (per utterance)

| Group    | Columns | Why it matters for emotion |
|----------|---------|----------------------------|
| prosody  | `f0_mean/std/min/max/range/slope`, `voiced_fraction` | Arousal: higher, more variable pitch in anger/joy/fear; flat, low pitch in sadness. |
| energy   | `rms_mean/std/max`, `dynamic_range_db` | Loudness and its variability track arousal. |
| spectral | `spectral_centroid/bandwidth/rolloff/flatness`, `zcr` | Voice brightness and tension (harsh vs breathy voice). |
| timbre   | `mfcc1..13_mean/std` | Standard input for classical speech-emotion classifiers. |
| timing   | `speech_rate_wps`, `articulation_rate_wps`, `syllable_rate_proxy`, `pause_mean/max/fraction`, `duration_s` | Fast, few pauses = excited/angry; slow, many pauses = sad/tired. |
| asr      | `avg_logprob`, `no_speech_prob`, `word_prob_mean`, `word_count`, `text` | Quality control plus text for lexical emotion cues. |

Pitch uses `librosa.pyin` (60-500 Hz); everything is computed on the 16 kHz
utterance clip that Whisper heard. The native-rate `session.wav` and the JSONL
timestamps are kept so features can be recomputed from the original audio.

## Baseline model (`scripts/train_baseline.py`)

No emotion labels exist yet, so there is no classifier to train. What exists
is an **unsupervised baseline**: it standardises the feature table (per
session, when there is more than one) and runs k-means to see whether
utterances separate into distinguishable vocal-style clusters at all, before
any label is collected. Outputs land in `results/model/`: `model.joblib`,
`cluster_assignments.csv`, a 2-D PCA scatter plot (`clusters.png`), and
`summary.json` with a silhouette score (near 0 = clusters overlap, near 1 =
well separated). Run it in the background with `run_pipeline.sh` (default) or
foreground with `--foreground-train`; `STATUS.json` in the same folder reports
whether it is still running.

This script is intentionally the seam where a real classifier will go once
labels exist: same input table, same output location, swap k-means for a
supervised model.

## Recommended next steps (not built yet)

1. Collect labelled sessions (self-report per session or per utterance) --
   this is what `train_baseline.py` is standing in for.
2. Supervised baseline: logistic regression / gradient boosting on the
   per-speaker-normalised feature table, replacing the k-means step above.
3. Stronger acoustic model: a pretrained speech-emotion encoder
   (e.g. wav2vec2 / HuBERT-based SER checkpoints) on the clips, fused with a
   text emotion classifier on `text`.
4. Report categories as probabilities so overlapping states are visible rather
   than forced into one label.
