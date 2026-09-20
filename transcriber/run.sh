#!/usr/bin/env bash
# Run the transcriber with config.toml. Extra arguments pass through:
#   bash run.sh                       # microphone session (shows the INFO log)
#   bash run.sh --quiet               # ...transcript only, nothing else on screen
#   bash run.sh --verbose             # ...full DEBUG log on screen
#   bash run.sh --list-devices        # find your microphone
#   bash run.sh --doctor              # environment health report
#   bash run.sh --wav file.wav        # transcribe a file through the same pipeline
#   bash run.sh --name interview1     # label the session folder
#   bash run.sh --config other.toml   # use a different config file
set -uo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
export HF_HOME="$PROJECT_DIR/.hf_cache" HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

VENV_DIR="${VENV_DIR:-}"
[ -z "$VENV_DIR" ] && [ -f "$PROJECT_DIR/.venv_path" ] && VENV_DIR="$(cat "$PROJECT_DIR/.venv_path")"
[ -z "$VENV_DIR" ] && VENV_DIR="$PROJECT_DIR/.venv"
if [ -x "$VENV_DIR/bin/python" ]; then
  PY="$VENV_DIR/bin/python"
else
  echo "No virtualenv at $VENV_DIR -- run: bash install.sh" >&2
  exit 1
fi
[ -f "$PROJECT_DIR/config.toml" ] || { echo "config.toml missing in $PROJECT_DIR" >&2; exit 1; }
exec "$PY" -m transcriber "$@"
