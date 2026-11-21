#!/usr/bin/env bash
# The emotion-model pipeline on the training (GPU) machine. One entry point, one verb per stage:
#
#   bash pipeline/run.sh setup [opts]    # once: venv, PyTorch, every download, data preparation, smoke test
#   bash pipeline/run.sh download        # open corpora + pretrained encoders (resumable)
#   bash pipeline/run.sh prepare         # manifest + 16 kHz cache, prosody features, quality checks
#   bash pipeline/run.sh train [opts]    # the experiment suite (configs/suite.yaml); --groups core extended large
#   bash pipeline/run.sh analyze         # results/summary.json, results/RESULTS.md, figures/*.png
#   bash pipeline/run.sh export [opts]   # best deployable model -> ONNX fp32 + int8 for the Jetson
#   bash pipeline/run.sh all             # download -> prepare -> train -> analyze -> export
#   bash pipeline/run.sh status          # what is downloaded, prepared and trained
#   bash pipeline/run.sh test            # unit tests;  smoke = end-to-end on synthetic data
#   bash pipeline/run.sh notebook        # executes notebook/pipeline.ipynb headless
#
# Data lives outside the repo in $EMOTE_DATA (default ~/emote_data; setup records the choice).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMOTE="$(dirname "$HERE")"
S="$HERE/scripts"

[ -z "${EMOTE_DATA:-}" ] && [ -f "$HERE/.data_root" ] && EMOTE_DATA="$(cat "$HERE/.data_root")"
export EMOTE_DATA="${EMOTE_DATA:-$HOME/emote_data}"
# Our own model cache, always: a machine-wide HF_HOME belongs to someone else.
export HF_HOME="$EMOTE_DATA/pretrained" HF_HUB_CACHE="$EMOTE_DATA/pretrained/hub" HUGGINGFACE_HUB_CACHE="$EMOTE_DATA/pretrained/hub"
VENV="${EMOTE_TRAIN_VENV:-}"
[ -z "$VENV" ] && [ -f "$HERE/.venv_path" ] && VENV="$(cat "$HERE/.venv_path")"
PY="${VENV:+$VENV/bin/python}"
PY="${PY:-python3}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
need_py() { "$PY" -c "import torch, transformers" 2>/dev/null || { echo "No training environment ($PY). Run: bash pipeline/run.sh setup" >&2; exit 1; }; }

STEP="${1:-help}"; shift || true
case "$STEP" in
  setup)    exec bash "$HERE/setup_training_machine.sh" "$@" ;;
  download) say "Downloading corpora and encoders into $EMOTE_DATA"; "$PY" "$S/downloads/prepare_all_external.py" "$@" ;;
  prepare)
    say "Preparing data in $EMOTE_DATA"
    "$PY" "$S/preprocessing/build_manifest.py" "$@" && \
    "$PY" "$S/preprocessing/extract_prosody.py" --workers "$(nproc)" && \
    "$PY" "$S/preprocessing/quality_checks.py" ;;
  train)    need_py; say "Training"; "$PY" "$S/training/train_suite.py" "$@" ;;
  evaluate) need_py; "$PY" "$S/evaluation/evaluate.py" "$@" ;;
  analyze)  say "Analysing"; "$PY" "$S/visualization/analyze.py" "$@" ;;
  export)   need_py; say "Exporting for the Jetson"; "$PY" "$S/export/export_onnx.py" "$@" ;;
  all)
    "$0" download && "$0" prepare && "$0" train "$@" && "$0" analyze && "$0" export ;;
  test)     need_py; "$PY" -m pytest -q "$EMOTE/emotion_model/tests" ;;
  smoke)    need_py; "$PY" "$S/training/smoke_test.py" "$@" ;;
  notebook) need_py; "$PY" -m jupyter nbconvert --to notebook --execute --inplace "$HERE/notebook/pipeline.ipynb" ;;
  status)
    echo "data root : $EMOTE_DATA"
    echo "python    : $PY"
    for d in ravdess crema_d tess meld esd iemocap msp_podcast; do
      if [ -f "$EMOTE_DATA/raw/$d/DOWNLOAD_OK.json" ]; then s="downloaded"; elif [ -d "$EMOTE_DATA/raw/$d" ]; then s="present (licensed or partial)"; else s="-"; fi
      printf '  %-12s %s\n' "$d" "$s"
    done
    [ -f "$EMOTE_DATA/manifests/manifest.csv" ] && echo "manifest  : $(($(wc -l < "$EMOTE_DATA/manifests/manifest.csv") - 1)) clips" || echo "manifest  : not built"
    [ -f "$EMOTE_DATA/features/prosody.parquet" ] && echo "prosody   : extracted" || echo "prosody   : not extracted"
    echo "finished runs:"; find "$HERE/results" -name test_metrics.json 2>/dev/null | sed "s#$HERE/results/##; s#/test_metrics.json##" | sort | sed 's/^/  /'
    ;;
  *) sed -n '2,15p' "$0" ;;
esac
