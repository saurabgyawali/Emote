# Emote stage 3 — emotion model training pipeline

Trains the model that turns what the Jetson records (an utterance's audio plus Whisper's
transcript) into an emotion estimate: one of seven categories (neutral, happy, sad,
angry, fear, disgust, surprise) and continuous arousal / valence / dominance. Training
runs on a GPU machine; the trained model is exported to ONNX and runs on the Jetson after
each session (`bash go.sh emotions`).

**Status (2026-10-10): validated on real data, not yet trained to convergence.** The setup
script ran end to end on a GPU machine (RAVDESS, CREMA-D, TESS: 11,682 clips; a real MELD
slice for the video path), every model type trained one epoch (full model 63 s/epoch,
5.3 GB peak; after one epoch the audio model reaches 0.64 test UA over 7 emotions), the
analysis, ONNX export and Jetson-side labelling ran on real recordings. `bash
pipeline/run.sh smoke` re-checks every stage on synthetic data in ~1 minute. Full
training needs the GPU time in [NEXT_STEPS.md](../NEXT_STEPS.md).

## The model

```
                     ┌─ audio tower ─── WavLM-Base-Plus ─ learned mix of all 13 layers ─ attentive stats pool ─┐
 16 kHz waveform ────┤                                                                                         │
 (loudness -23 dBFS) └─ prosody tower ─ 42 stage-2 features (pitch, energy, spectral, MFCC) ─ MLP ─────────────┤─ gated fusion ─┬─ 7 emotions (masked per corpus)
 transcript ──────────── text tower ─── emotion-pretrained DistilRoBERTa ─ mean pool ──────────────────────────┘ (+ modality     └─ arousal / valence / dominance
                                                                                                                    dropout)
```

| Choice | Why |
|---|---|
| WavLM-Base-Plus audio encoder | Best base-size self-supervised encoder on SUPERB emotion recognition; 95 M parameters is small enough to quantise for the Jetson. E6/E7 test HuBERT and wav2vec 2.0, E8 WavLM-Large. |
| Weighted sum of all layers | Emotion information sits in the middle layers of speech encoders, not the last one; the learned weights are plotted (fig07). |
| Fine-tune only the top 4 layers | Most of the gain of full fine-tuning at a fraction of the memory; the CNN front end stays frozen. E2 measures the frozen baseline. |
| Text tower on Whisper's transcript | What is said carries emotion; the encoder starts from a model already trained on text emotion. Only natural corpora (MELD, MSP-Podcast, IEMOCAP) have informative text. |
| Prosody tower | Your existing stage-2 features: cheap, interpretable, robust to an unfamiliar microphone. E1 alone is the floor every neural model must beat. |
| Gated fusion + modality dropout | Per-clip weights over the modalities present (logged, fig08); the model keeps working when Whisper returns nothing. |
| Label-set-masked loss | Corpora disagree on which emotions exist (CREMA-D has no surprise); each clip is only scored over its own corpus's classes. |
| Loudness normalisation | Corpora (and the Jetson mic) differ by 20+ dB; without it the model learns "loud recording = this corpus". |

## The experiments (`configs/experiments/`, 3 seeds each)

| | Experiment | Question |
|---|---|---|
| E1 | prosody only | How far do hand-crafted features get? |
| E2 | audio, frozen encoder | What does the pretrained representation already know? |
| E3 | audio, top 4 layers fine-tuned | The main audio-only model |
| E4 | + transcript | Does what is said help? |
| E5 | + transcript + prosody | The full model; Jetson candidate |
| E6, E7 | HuBERT / wav2vec 2.0 instead of WavLM | Does the encoder choice matter? |
| E8 | WavLM-Large full model | Upper bound; distillation teacher (needs >= 24 GB GPU) |
| X1 | train on acted corpora, test on MELD | Generalisation from actors to natural speech (the Jetson's real situation) |
| X2 | CREMA-D held out entirely | Generalisation to unseen speakers and recording setup |

`configs/suite.yaml` groups them: `core` (E1–E5, X1), `extended` (E6, E7, X2), `large` (E8).

## Layout

```
pipeline/
  run.sh                      the one entry point: setup | download | prepare | train | analyze | export | all | status | test | smoke
  setup_training_machine.sh   one-time machine setup (venv, PyTorch, downloads, preparation, tests)
  datasets.yaml               every corpus: source, licence, size, labels, splits
  requirements.txt
  configs/                    base.yaml, experiments/*.yaml, suite.yaml
  notebook/pipeline.ipynb     thin wrapper: CONFIG cell + DO_RETRAIN master switch, one section per stage
  scripts/downloads/          one resumable downloader per corpus + prepare_all_external.py
  scripts/preprocessing/      build_manifest.py (parsers, 16 kHz cache, splits), extract_prosody.py, quality_checks.py
  scripts/training/           train.py (one experiment x seed), train_suite.py, smoke_test.py
  scripts/evaluation/         evaluate.py (any checkpoint on any corpus/split)
  scripts/visualization/      analyze.py -> results/summary.json, results/RESULTS.md, figures/fig01..fig11
  scripts/export/             export_onnx.py (fp32 + int8, parity-checked), annotate_sessions.py (runs on the Jetson)
  results/                    small per-run JSON/CSV (tracked)
  figures/                    analysis figures (tracked)
../emotion_model/             the importable model code + unit tests
```

Data, caches and checkpoints live in `$EMOTE_DATA` (default `~/emote_data`):
`raw/`, `audio16k/`, `manifests/`, `features/`, `pretrained/`, `runs/`, `deploy/`, `logs/`.

## Per-run outputs (`results/<experiment>/seed<k>/`)

`config.yaml` (fully resolved), `step_metrics.jsonl`, `epoch_metrics.jsonl`,
`test_metrics.json` (`best` = validation-selected epoch and `last` epoch, each with val,
test and held-out scorecards), `predictions.csv` (per-clip probabilities, dimensions,
modality weights), `train_summary.json` (data counts, parameters, layer weights, timing,
GPU memory, versions, config fingerprint). Runs whose fingerprint matches are skipped.

## Metrics

UA (unweighted accuracy = mean per-class recall) averaged over corpora is the selection
and headline metric; WA, macro-F1, BIG4 UA (neutral/happy/sad/angry, the literature's
convention), per-class recall, confusion, ECE calibration and CCC for the dimensions are
reported beside it. Paired contrasts across seeds use an exact sign-flip permutation
test with its floor (2/2^n) printed next to every p.
