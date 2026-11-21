#!/usr/bin/env bash
# The one command for a recording session. Run it on the Jetson:
#
#   bash go.sh                 # update the code, check the machine, record, bundle the logs
#   bash go.sh --quiet         # ...showing only the transcript while recording
#   bash go.sh --log           # ...showing the INFO log instead of the mascot screen
#   bash go.sh --name demo1    # ...and label the session folder
#   bash go.sh --wav clip.wav  # ...over an audio file instead of the microphone
#
# Individual steps, when you want just one:
#
#   bash go.sh update     # git pull (or tell you how to copy the code across)
#   bash go.sh install    # create/repair the virtualenv, CUDA CTranslate2 and the model
#   bash go.sh check      # health report only
#   bash go.sh mic        # 10s microphone level meter (run this if nothing gets transcribed)
#   bash go.sh record     # recording session only
#   bash go.sh logs       # diagnostics tarball only
#   bash go.sh vocab      # suggest vocabulary words from your sessions (--write adds them)
#   bash go.sh emotions   # label recorded utterances with the trained emotion model (emotion_deploy/)
#   bash go.sh features   # emotion_analysis stage: sessions -> features.csv
#   bash go.sh copy       # build a clean folder to drag to the Jetson (--tar for a tarball)
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
  update|install|check|mic|record|logs|vocab|emotions|features|copy) STEP="$1"; shift ;;
  -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
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

# The venv holds the CUDA ctranslate2 build; without it nothing runs. It is absent
# after a fresh clone, and after an Emote folder is replaced rather than merged.
venv_missing() {
  local venv="$TRANSCRIBER/.venv"
  [ -f "$TRANSCRIBER/.venv_path" ] && venv="$(cat "$TRANSCRIBER/.venv_path")"
  [ ! -x "$venv/bin/python" ]
}

step_install() {
  say "Installing (venv, requirements, CUDA CTranslate2, model)"
  info "First time on this machine this takes a while. Re-runs skip what is already done."
  bash "$TRANSCRIBER/install.sh" ${RUN_ARGS[@]+"${RUN_ARGS[@]}"}
}

step_check() {
  say "Checking this machine"
  bash "$TRANSCRIBER/run.sh" --doctor
}

step_mic() {
  say "Microphone level check"
  bash "$TRANSCRIBER/run.sh" --mic-check ${RUN_ARGS[@]+"${RUN_ARGS[@]}"}
}

step_record() {
  say "Recording a session"
  info "Speak when it says it is listening."
  info "Press Esc (or Enter) to stop: the session is saved and the logs are bundled"
  info "up to that point -- stopping early never loses what was already transcribed."
  info "Add --quiet for transcript-only output, --log for the INFO log, --verbose for DEBUG."
  # Anything the recorder prints to stderr is also kept in a file, so a failure that
  # happens before Python's own run log exists still reaches the diagnostics bundle.
  # Twice on 2026-10-10 the record step left no trace at all.
  local console_log rc
  console_log="$TRANSCRIBER/logs/$(date +%Y-%m-%d)/$(date +%H)/record_console_$(date +%Y%m%d_%H%M%S).log"
  mkdir -p "$(dirname "$console_log")"
  # stdout stays on the terminal (fd 3) so the mascot screen still sees a TTY; only stderr
  # goes through tee, and tee -i survives Ctrl+C so the shutdown summary is not lost.
  { bash "$TRANSCRIBER/run.sh" ${RUN_ARGS[@]+"${RUN_ARGS[@]}"} 2>&1 1>&3 3>&- | tee -i "$console_log" >&2; } 3>&1
  rc=${PIPESTATUS[0]}
  echo "[record exit code $rc]" >> "$console_log"
  if [ "$rc" -ne 0 ]; then
    warn "The recorder exited with code $rc. Its error output is saved in:"
    info "  $console_log"
  fi
  return "$rc"
}

step_logs() {
  say "Bundling the logs"
  bash "$TRANSCRIBER/scripts/collect_diagnostics.sh" "$@"
  local bundle
  bundle="$(find "$TRANSCRIBER/logs/diagnostics" -name 'diagnostics_*.tar.gz' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)"
  if [ -n "$bundle" ]; then
    printf '\n\033[1m==> Bring this ONE file back\033[0m\n'
    info "$bundle"
    info ""
    info "Drag it off this machine into the same place on the other side:"
    info "  Emote/transcriber/logs/diagnostics/<day>/<hour>/"
    info "It holds the logs, health report, transcripts and system facts -- but no audio."
    info "Need the recordings too? Re-run: bash go.sh logs --with-audio"
  else
    warn "No diagnostics tarball was produced; see the output above."
  fi
}

step_vocab() {
  say "Vocabulary suggestions from your sessions"
  local venv="$TRANSCRIBER/.venv"
  [ -f "$TRANSCRIBER/.venv_path" ] && venv="$(cat "$TRANSCRIBER/.venv_path")"
  [ -n "${VENV_DIR:-}" ] && venv="$VENV_DIR"
  "$venv/bin/python" "$TRANSCRIBER/scripts/suggest_vocabulary.py" ${RUN_ARGS[@]+"${RUN_ARGS[@]}"}
}

step_emotions() {
  say "Emotion labels for recorded sessions"
  local venv="$TRANSCRIBER/.venv"
  [ -f "$TRANSCRIBER/.venv_path" ] && venv="$(cat "$TRANSCRIBER/.venv_path")"
  [ -n "${VENV_DIR:-}" ] && venv="$VENV_DIR"
  "$venv/bin/python" "$HERE/pipeline/scripts/export/annotate_sessions.py" "$@"
}

# True when an exported model is in emotion_deploy/, or one folder too deep inside it.
has_emotion_model() {
  [ -f "$HERE/emotion_deploy/deploy.json" ] || compgen -G "$HERE/emotion_deploy/*/deploy.json" >/dev/null
}

step_features() {
  say "Extracting features from the sessions already recorded"
  bash "$HERE/run_pipeline.sh" --no-record ${RUN_ARGS[@]+"${RUN_ARGS[@]}"}
}

step_copy() {
  for arg in ${RUN_ARGS[@]+"${RUN_ARGS[@]}"}; do
    if [ "$arg" = "--tar" ]; then
      say "Building a tarball to carry across"
      bash "$HERE/pack.sh" --with-model
      info "Extract it OVER the existing Emote folder on the Jetson. Do not delete that folder first."
      return 0
    fi
  done

  local dest="$(dirname "$HERE")/Emote-to-copy"
  say "Building a clean folder to drag across"
  rm -rf "$dest"
  mkdir -p "$dest"
  # Code and config only: no 11 GB ct2 tree, no venv, no logs, no recordings.
  # The Jetson keeps its own models, venv, logs and sessions when you copy over the top.
  tar -C "$HERE" -cf - \
      --exclude="ct2/CTranslate2" --exclude="transcriber/models" --exclude="transcriber/sessions" \
      --exclude="transcriber/logs" --exclude="logs copy" --exclude=".git" \
      --exclude=".venv" --exclude=".venv.broken" --exclude=".venv_path" --exclude=".hf_cache" \
      --exclude="__pycache__" --exclude=".pytest_cache" --exclude="*.pyc" \
      --exclude="emotion_analysis/results" --exclude="emotion_deploy" \
      --exclude="pipeline/.data_root" --exclude="pipeline/.venv_path" \
      . | tar -C "$dest" -xf -
  info "Folder : $dest  ($(du -sh --apparent-size "$dest" | cut -f1) of code in $(find "$dest" -type f | wc -l) files)"
  info ""
  info "Drag that folder onto the Jetson and drop it ON TOP of /home/jetson/Developer/RASL/Emote"
  info "(choose Merge / Replace when asked). Do NOT delete the Emote folder there first --"
  info "its models, venv, CUDA build and past logs are not in this copy and would be lost."
  info ""
  info "Then on the Jetson:  cd ~/Developer/RASL/Emote && bash go.sh"
}

case "$STEP" in
  update)   step_update ;;
  check)    step_check ;;
  record)   step_record ;;
  logs)     step_logs ${RUN_ARGS[@]+"${RUN_ARGS[@]}"} ;;
  vocab)    step_vocab ;;
  emotions) step_emotions ${RUN_ARGS[@]+"${RUN_ARGS[@]}"} ;;
  features) step_features ;;
  copy)     step_copy ;;
  install)  step_install ;;
  mic)      step_mic ;;
  session)
    step_update
    if venv_missing; then
      warn "No virtualenv on this machine, so nothing can run yet."
      info "That is normal on a fresh clone, or after the Emote folder was replaced"
      info "instead of merged. install.sh will rebuild it."
      printf '    Run the installer now? [Y/n]: '
      read -r answer
      case "$answer" in [Nn]*) die "run: bash go.sh install" ;; *) step_install ;; esac
    fi
    if ! step_check; then
      warn "The health report found problems (see above)."
      printf '    Record anyway? [y/N]: '
      read -r answer
      case "$answer" in [Yy]*) ;; *) die "fix the problems above, then run bash go.sh again" ;; esac
    fi
    step_record || warn "The session ended with an error; bundling the logs anyway."
    # Emotion labels only after the whole talk is over: Whisper has exited, so the model
    # gets the memory and CPU to itself instead of competing with live transcription.
    if has_emotion_model; then
      step_emotions --latest || warn "Emotion labelling failed (see above); the transcript is unaffected."
    else
      # Skipping silently hid a missing model on 2026-11-08: the session ran and nobody knew why
      # no emotions appeared.
      warn "No emotion model on this machine, so the session was not labelled."
      info "Copy the CONTENTS of the exported folder (deploy.json, model.int8.onnx, tokenizer.json ...)"
      info "into $HERE/emotion_deploy/ ; then label this session with: bash go.sh emotions --latest"
    fi
    step_logs
    say "Done"
    ;;
esac
