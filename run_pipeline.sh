#!/usr/bin/env bash
# End-to-end Emote pipeline: transcriber -> feature extraction -> baseline model.
#
#   bash run_pipeline.sh                       # live microphone session (Jetson), then features, then background training
#   bash run_pipeline.sh --wav clip.wav         # run the same pipeline over a file instead of the mic
#   bash run_pipeline.sh --name interview1      # label the session
#   bash run_pipeline.sh --skip-train           # stop after feature extraction
#   bash run_pipeline.sh --no-record            # skip recording; extract from the sessions already on disk
#   bash run_pipeline.sh --foreground-train     # wait for training instead of backgrounding it
#
# Each stage is idempotent: extract_features.py only processes new utterances,
# and train_baseline.py skips retraining unless --force is passed. Stopping this
# script after stage 1 or 2 loses nothing already written.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSCRIBER="$HERE/transcriber"
ANALYSIS="$HERE/emotion_analysis"

SKIP_TRAIN="no"
FOREGROUND_TRAIN="no"
NO_RECORD="no"
TRANSCRIBER_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-train) SKIP_TRAIN="yes"; shift ;;
    --foreground-train) FOREGROUND_TRAIN="yes"; shift ;;
    --no-record) NO_RECORD="yes"; shift ;;
    *) TRANSCRIBER_ARGS+=("$1"); shift ;;
  esac
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }

# Reuse an existing interpreter if one is known (ANALYSIS_PYTHON env var, or a
# path recorded in .venv_path from a previous run) before creating a new venv --
# some filesystems (notably network shares) reject the symlinks `venv` needs.
APY="${ANALYSIS_PYTHON:-}"
if [ -z "$APY" ] && [ -f "$ANALYSIS/.venv_path" ]; then
  CANDIDATE="$(cat "$ANALYSIS/.venv_path")/bin/python"
  [ -x "$CANDIDATE" ] && APY="$CANDIDATE"
fi
if [ -z "$APY" ] && [ -x "$ANALYSIS/.venv/bin/python" ]; then
  APY="$ANALYSIS/.venv/bin/python"
fi
if [ -z "$APY" ]; then
  say "emotion_analysis has no Python environment yet -- creating .venv"
  if python3 -m venv "$ANALYSIS/.venv" 2>"$ANALYSIS/.venv_create_error.log"; then
    rm -f "$ANALYSIS/.venv_create_error.log"
    APY="$ANALYSIS/.venv/bin/python"
    "$APY" -m pip install -q --upgrade pip
    "$APY" -m pip install -q -r "$ANALYSIS/requirements.txt"
    echo "$ANALYSIS/.venv" > "$ANALYSIS/.venv_path"
  else
    echo "venv creation failed (see $ANALYSIS/.venv_create_error.log)." >&2
    echo "This can happen on filesystems that reject symlinks (e.g. some network shares)." >&2
    echo "Fix: create a venv elsewhere and point at it, e.g.:" >&2
    echo "    python3 -m venv ~/emote-analysis-venv && ~/emote-analysis-venv/bin/pip install -r $ANALYSIS/requirements.txt" >&2
    echo "    ANALYSIS_PYTHON=~/emote-analysis-venv/bin/python bash run_pipeline.sh ..." >&2
    exit 1
  fi
fi
info "Using interpreter: $APY"

if [ "$NO_RECORD" = "yes" ]; then
  say "1) Transcriber: skipped (--no-record); using the sessions already in $TRANSCRIBER/sessions"
elif ! bash "$TRANSCRIBER/run.sh" "${TRANSCRIBER_ARGS[@]}"; then
  echo "Transcriber stage failed; stopping (feature extraction needs a session)." >&2
  exit 1
else
  say "1) Transcriber: audio -> session folder"
fi

say "2) Feature extraction: session folders -> features.csv"
if ! "$APY" "$ANALYSIS/scripts/extract_features.py" --sessions-dir "$TRANSCRIBER/sessions"; then
  echo "Feature extraction failed; see emotion_analysis/results/logs/ for details." >&2
  exit 1
fi

if [ "$SKIP_TRAIN" = "yes" ]; then
  info "Skipping training stage (--skip-train)."
  exit 0
fi

say "3) Baseline model: features.csv -> unsupervised clusters"
if [ "$FOREGROUND_TRAIN" = "yes" ]; then
  "$APY" "$ANALYSIS/scripts/train_baseline.py"
else
  mkdir -p "$ANALYSIS/results/logs"
  TRAIN_LOG="$ANALYSIS/results/logs/train_pipeline_$(date +%Y%m%d_%H%M%S).log"
  nohup "$APY" "$ANALYSIS/scripts/train_baseline.py" > "$TRAIN_LOG" 2>&1 &
  TRAIN_PID=$!
  info "Training started in the background (PID $TRAIN_PID)."
  info "Log:    $TRAIN_LOG"
  info "Status: $ANALYSIS/results/model/STATUS.json"
  info "Check progress any time with: cat $ANALYSIS/results/model/STATUS.json"
fi

say "Pipeline finished (training may still be running in the background)."
