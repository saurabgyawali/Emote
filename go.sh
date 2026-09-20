#!/usr/bin/env bash
# The one command for a recording session. Run it on the Jetson:
#
#   bash go.sh                 # update the code, check the machine, record, bundle the logs
#   bash go.sh --name demo1    # ...and label the session folder
#   bash go.sh --wav clip.wav  # ...over an audio file instead of the microphone
#
# Individual steps, when you want just one:
#
#   bash go.sh update     # git pull (or tell you how to copy the code across)
#   bash go.sh check      # health report only
#   bash go.sh record     # recording session only
#   bash go.sh logs       # diagnostics tarball only
#   bash go.sh features   # emotion_analysis stage: sessions -> features.csv
#   bash go.sh copy       # build a tarball to carry to a machine with no network
#
# Every step is a thin wrapper around the script that already does the work
# (transcriber/run.sh, scripts/collect_diagnostics.sh, run_pipeline.sh, pack.sh).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSCRIBER="$HERE/transcriber"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '    \033[33m%s\033[0m\n' "$*"; }
die()  { printf '\n\033[31mStopped: %s\033[0m\n' "$*" >&2; exit 1; }

STEP="session"
case "${1:-}" in
  update|check|record|logs|features|copy) STEP="$1"; shift ;;
  -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
esac
RUN_ARGS=("$@")

step_update() {
  say "Updating the code"
  if [ ! -d "$HERE/.git" ]; then
    warn "Not a git checkout. Copy the Emote folder here, or clone it:"
    info "  git clone <your repo url> Emote"
    return 0
  fi
  if ! git -C "$HERE" remote get-url origin >/dev/null 2>&1; then
    warn "This checkout has no 'origin' remote, so there is nothing to pull."
    info "Push it to GitHub once, then here:  git remote add origin <url>"
    return 0
  fi
  if git -C "$HERE" pull --ff-only; then
    info "now at: $(git -C "$HERE" log --oneline -1)"
  else
    warn "git pull failed (no network, or local edits). Continuing with the code already here."
  fi
}

step_check() {
  say "Checking this machine"
  bash "$TRANSCRIBER/run.sh" --doctor
}

step_record() {
  say "Recording a session"
  info "Speak when it says it is listening. Press Enter or Ctrl+C to stop and save."
  bash "$TRANSCRIBER/run.sh" "${RUN_ARGS[@]}"
}

step_logs() {
  say "Bundling the logs"
  bash "$TRANSCRIBER/scripts/collect_diagnostics.sh"
  info "Copy that tarball to whoever is debugging this."
}

step_features() {
  say "Extracting features from the sessions already recorded"
  bash "$HERE/run_pipeline.sh" --no-record "${RUN_ARGS[@]}"
}

step_copy() {
  say "Building a tarball to carry across"
  bash "$HERE/pack.sh" --with-model
  info "Copy that .tar.gz to the other machine and extract it over the existing Emote folder."
}

case "$STEP" in
  update)   step_update ;;
  check)    step_check ;;
  record)   step_record ;;
  logs)     step_logs ;;
  features) step_features ;;
  copy)     step_copy ;;
  session)
    step_update
    if ! step_check; then
      warn "The health report found problems (see above)."
      printf '    Record anyway? [y/N]: '
      read -r answer
      case "$answer" in [Yy]*) ;; *) die "fix the problems above, then run bash go.sh again" ;; esac
    fi
    step_record || warn "The session ended with an error; bundling the logs anyway."
    step_logs
    say "Done"
    ;;
esac
