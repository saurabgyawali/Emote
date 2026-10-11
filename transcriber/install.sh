#!/usr/bin/env bash
# One-shot, idempotent install for the Emote transcriber. Safe to re-run.
#
#   bash install.sh                 # non-interactive: everything from config.toml
#   bash install.sh --interactive   # ...then a short wizard edits config.toml
#
# Steps (each skipped when already satisfied):
#   1 system packages (apt)   2 virtualenv (reuse or create)   3 pip requirements
#   4 CUDA CTranslate2 on Jetson (reuses a previous build)     5 Whisper model
#   6 unit tests              7 health report (--doctor)
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMOTE_DIR="$(dirname "$PROJECT_DIR")"
CT2_BUILD_ROOT="${CT2_BUILD_ROOT:-$EMOTE_DIR/ct2}"
cd "$PROJECT_DIR"
export HF_HOME="$PROJECT_DIR/.hf_cache"
# Same module path run.sh uses; without it `python -m transcriber` cannot be found (Jetson log, step 7).
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# Everything this script prints (and every command it runs) is also written to
# logs/install_<timestamp>.log so a failed install can be diagnosed afterwards.
LOG_DAY_DIR="$PROJECT_DIR/logs/$(date +%Y-%m-%d)/$(date +%H)"
mkdir -p "$LOG_DAY_DIR"
INSTALL_LOG="$LOG_DAY_DIR/install_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$INSTALL_LOG") 2>&1
ln -sfn "${INSTALL_LOG#"$PROJECT_DIR/logs/"}" "$PROJECT_DIR/logs/install_latest.log" 2>/dev/null || printf "%s\n" "$INSTALL_LOG" > "$PROJECT_DIR/logs/install_latest.log"
echo "install.sh started $(date -Is); log: $INSTALL_LOG; args: $*"

VENV_DIR="${VENV_DIR:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INTERACTIVE="no"; SKIP_APT="no"; SKIP_CUDA="no"; SKIP_MODEL="no"; SKIP_TESTS="no"; ONLY=""
CT2_BUILD_ARGS=()

usage() {
  cat <<USAGE
Usage: bash install.sh [options]
  --interactive        run the config wizard at the end (mic / device / model)
  --venv <dir>         virtualenv to use or create (default: existing .venv_path, else ./.venv)
  --python <path>      interpreter used to CREATE the venv (default: python3)
  --skip-apt           do not install system packages
  --skip-cuda-build    do not build CUDA CTranslate2 (CPU only)
  --skip-model         do not download the Whisper model
  --skip-tests         do not run unit tests
  --only-model         only ensure the model from config.toml is downloaded
  --only-cuda          only ensure CUDA CTranslate2 is installed
  --ct2-jobs <N>       parallel jobs for the CTranslate2 compile (default 4)
  --ct2-clean          force a clean CTranslate2 recompile
  --debug              trace every shell command (set -x) into the log
USAGE
}
while [ $# -gt 0 ]; do
  case "$1" in
    --interactive) INTERACTIVE="yes"; shift ;;
    --venv) VENV_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --skip-apt) SKIP_APT="yes"; shift ;;
    --skip-cuda-build) SKIP_CUDA="yes"; shift ;;
    --skip-model) SKIP_MODEL="yes"; shift ;;
    --skip-tests) SKIP_TESTS="yes"; shift ;;
    --only-model) ONLY="model"; shift ;;
    --only-cuda) ONLY="cuda"; shift ;;
    --ct2-jobs) CT2_BUILD_ARGS+=(--jobs "$2"); shift 2 ;;
    --ct2-clean) CT2_BUILD_ARGS+=(--clean); shift ;;
    --debug) set -x; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '    \033[33mWARNING:\033[0m %s\n' "$*"; }
FAILED=()

# ---- platform ------------------------------------------------------------------------
IS_JETSON="no"
[ "$(uname -m)" = "aarch64" ] && { [ -f /etc/nv_tegra_release ] || grep -qi "jetson\|tegra" /proc/device-tree/model 2>/dev/null; } && IS_JETSON="yes"
say "Platform: $(uname -s) $(uname -m), Jetson=$IS_JETSON"
info "kernel      : $(uname -r)"
[ -f /etc/os-release ] && info "os          : $(. /etc/os-release && echo "$PRETTY_NAME")"
[ -f /etc/nv_tegra_release ] && info "L4T         : $(head -1 /etc/nv_tegra_release)"
[ -f /proc/device-tree/model ] && info "board       : $(tr -d '\0' < /proc/device-tree/model)"
info "python3     : $(command -v python3 || echo none) $(python3 -V 2>&1)"
info "nvcc        : $( (command -v nvcc || ls /usr/local/cuda*/bin/nvcc 2>/dev/null | head -1) || echo 'not found')"
[ -f /usr/local/cuda/version.json ] && info "cuda        : $(grep -m1 -oE '"version" *: *"[0-9.]+"' /usr/local/cuda/version.json)"
info "memory      : $(free -m | awk '/Mem:/ {print $2 " MB total, " $7 " MB available"}')"
info "disk free   : $(df -h "$PROJECT_DIR" | awk 'NR==2 {print $4 " on " $6}')"
info "existing venv marker: $(cat "$PROJECT_DIR/.venv_path" 2>/dev/null || echo none)"
info "ct2 build tree      : $([ -f "$CT2_BUILD_ROOT/CTranslate2/build/libctranslate2.so" ] && echo "present at $CT2_BUILD_ROOT" || echo absent)"
info "ct2 wheel cache     : $(ls -t "$CT2_BUILD_ROOT"/wheels/ctranslate2-*.whl 2>/dev/null | head -1 | grep . || echo empty)"
info "sudo        : $(sudo -n true 2>/dev/null && echo passwordless || echo 'will prompt / unavailable')"
info "network     : $(timeout 5 python3 -c 'import urllib.request;urllib.request.urlopen("https://pypi.org/simple/",timeout=4);print("pypi reachable")' 2>/dev/null || echo 'pypi NOT reachable')"

# ---- 1 apt ------------------------------------------------------------------------------
if [ "$SKIP_APT" = "no" ] && [ -z "$ONLY" ] && command -v apt-get >/dev/null 2>&1; then
  say "1) System packages"
  PKGS=(python3-venv python3-dev portaudio19-dev libportaudio2 libsndfile1 ffmpeg build-essential)
  MISSING=()
  for p in "${PKGS[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || MISSING+=("$p"); done
  if [ ${#MISSING[@]} -eq 0 ]; then
    info "all present"
  elif sudo -n true 2>/dev/null || [ -t 0 ]; then
    info "installing: ${MISSING[*]}"
    sudo apt-get update -qq && sudo apt-get install -y -qq "${MISSING[@]}" || { warn "apt install failed"; FAILED+=("apt: ${MISSING[*]}"); }
  else
    warn "no sudo available; install manually: sudo apt-get install -y ${MISSING[*]}"
    FAILED+=("apt: ${MISSING[*]}")
  fi
fi

# ---- 2 venv ------------------------------------------------------------------------------
say "2) Virtual environment"
if [ -z "$VENV_DIR" ]; then
  if [ -f "$PROJECT_DIR/.venv_path" ] && [ -x "$(cat "$PROJECT_DIR/.venv_path")/bin/python" ]; then
    VENV_DIR="$(cat "$PROJECT_DIR/.venv_path")"
  else
    VENV_DIR="$PROJECT_DIR/.venv"
  fi
fi
if [ -x "$VENV_DIR/bin/python" ] && "$VENV_DIR/bin/python" -c 'import sys' 2>/dev/null; then
  info "reusing $VENV_DIR ($("$VENV_DIR/bin/python" -V 2>&1))"
else
  if [ -d "$VENV_DIR" ]; then
    warn "$VENV_DIR exists but its python does not run on this machine (copied from another architecture?)"
    warn "moving it aside to $VENV_DIR.broken"
    rm -rf "$VENV_DIR.broken"; mv "$VENV_DIR" "$VENV_DIR.broken"
  fi
  info "creating $VENV_DIR with $PYTHON_BIN"
  "$PYTHON_BIN" -m venv "$VENV_DIR" || { echo "venv creation failed (apt install python3-venv)"; exit 1; }
fi
printf '%s\n' "$VENV_DIR" > "$PROJECT_DIR/.venv_path"
PY="$VENV_DIR/bin/python"
info "interpreter : $PY ($("$PY" -V 2>&1)), pip $("$PY" -m pip --version 2>&1 | awk '{print $2}')"
"$PY" -m pip install -q --upgrade pip 2>/dev/null || true

# ---- 3 pip ----------------------------------------------------------------------------------
if [ -z "$ONLY" ]; then
  say "3) Python requirements"
  # Never let pip replace a source-built CUDA ctranslate2 with the CPU wheel:
  # pin whatever CUDA-capable version is already installed as a constraint.
  CONSTRAINT_ARGS=()
  CT2_CUDA_VER="$("$PY" -c 'import ctranslate2 as c; print(c.__version__ if c.get_cuda_device_count() > 0 else "")' 2>/dev/null || true)"
  if [ -n "$CT2_CUDA_VER" ]; then
    printf 'ctranslate2==%s\n' "$CT2_CUDA_VER" > "$LOG_DAY_DIR/ct2_constraint.txt"
    CONSTRAINT_ARGS=(-c "$LOG_DAY_DIR/ct2_constraint.txt")
    info "keeping CUDA ctranslate2 $CT2_CUDA_VER pinned during pip install"
  fi
  if "$PY" -m pip install -q "${CONSTRAINT_ARGS[@]}" -r "$PROJECT_DIR/requirements.txt"; then
    info "ok"
    info "installed   : $("$PY" -m pip list 2>/dev/null | grep -iE '^(numpy|scipy|sounddevice|webrtcvad|faster-whisper|ctranslate2|soundfile|onnxruntime|tomli) ' | awk '{printf "%s=%s ", $1, $2}')"
    "$PY" -c 'import sounddevice' 2>/dev/null && info "portaudio   : ok" || warn "sounddevice cannot load PortAudio (apt: libportaudio2 portaudio19-dev)"
  else
    warn "pip install failed (no network?)"; FAILED+=("pip requirements")
  fi
fi

# ---- 4 CUDA ct2 ------------------------------------------------------------------------------
cuda_count() { "$PY" -c 'import ctranslate2 as c; print(c.get_cuda_device_count())' 2>/dev/null || echo 0; }
if [ "$SKIP_CUDA" = "no" ] && { [ -z "$ONLY" ] || [ "$ONLY" = "cuda" ]; }; then
  say "4) CUDA-enabled CTranslate2"
  N="$(cuda_count)"
  info "ctranslate2 : $("$PY" -c 'import ctranslate2 as c,sys;print(c.__version__, "cuda devices", c.get_cuda_device_count())' 2>&1 | tail -1)"
  if [ "$N" -ge 1 ] 2>/dev/null; then
    info "already CUDA-enabled ($N device(s))"
  elif [ "$IS_JETSON" = "yes" ]; then
    info "building/installing from source (reuses $CT2_BUILD_ROOT if present)"
    bash "$PROJECT_DIR/scripts/build_ctranslate2_cuda.sh" --python "$PY" --build-root "$CT2_BUILD_ROOT" --yes "${CT2_BUILD_ARGS[@]}"
    N="$(cuda_count)"
    if [ "$N" -ge 1 ] 2>/dev/null; then
      info "CUDA CTranslate2 ready ($N device(s), checked independently of the build script)"
    else
      warn "CUDA build failed; transcription will run on CPU. Re-run: bash install.sh --only-cuda"
      FAILED+=("cuda build")
    fi
  else
    info "not a Jetson: keeping the CPU wheel (set model.device = cpu or auto)"
  fi
fi

# ---- 5 model -----------------------------------------------------------------------------------
if [ "$SKIP_MODEL" = "no" ] && { [ -z "$ONLY" ] || [ "$ONLY" = "model" ]; }; then
  say "5) Whisper model"
  "$PY" "$PROJECT_DIR/scripts/download_model.py" --yes || { warn "model download failed"; FAILED+=("model download"); }
  FALLBACK="$("$PY" -m transcriber.config get model.cpu_fallback_model || true)"
  if [ -n "$FALLBACK" ] && [ "$(cuda_count)" = "0" ]; then
    info "no CUDA: also fetching the CPU fallback model ($FALLBACK)"
    "$PY" "$PROJECT_DIR/scripts/download_model.py" --model "$FALLBACK" --yes || warn "fallback model download failed"
  fi
fi

# ---- 6 tests ----------------------------------------------------------------------------------------
if [ "$SKIP_TESTS" = "no" ] && [ -z "$ONLY" ]; then
  say "6) Unit tests"
  (cd "$PROJECT_DIR" && "$PY" -m pytest -q tests) || { warn "tests failed"; FAILED+=("tests"); }
fi

# ---- 7 doctor ------------------------------------------------------------------------------------------
if [ -z "$ONLY" ]; then
  say "7) Health report"
  (cd "$PROJECT_DIR" && "$PY" -m transcriber --doctor) || warn "doctor found problems (see above)"
fi

if [ "$INTERACTIVE" = "yes" ]; then
  say "Configuration wizard"
  "$PY" -m transcriber.configure "$PROJECT_DIR/config.toml"
fi

say "Install finished $(date -Is)"
info "full log: $INSTALL_LOG"
if [ ${#FAILED[@]} -gt 0 ]; then
  printf '    \033[31mIncomplete steps:\033[0m %s\n' "${FAILED[*]}"
  info "Fix the issue and re-run bash install.sh (completed steps are skipped)."
  exit 1
fi
info "Edit config.toml if needed, then:  bash run.sh"
