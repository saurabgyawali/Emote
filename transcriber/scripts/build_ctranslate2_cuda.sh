#!/usr/bin/env bash
# Build + install CUDA-enabled CTranslate2 on a Jetson (Orin = SM 8.7), idempotently.
#
# The PyPI / Jetson AI Lab ctranslate2 wheels for aarch64 are CPU-only, so GPU
# transcription needs a source build with -DWITH_CUDA=ON. This script:
#   1. reuses an existing compiled tree in $BUILD_ROOT (skips the 30-45 min make)
#   2. reuses a previously built wheel in $BUILD_ROOT/wheels
#   3. only compiles when neither exists
#   4. installs into the Python you pass with --python and verifies CUDA.
#
# $BUILD_ROOT defaults to Emote/ct2, so the source tree and the wheel cache travel
# with the project folder instead of living in $HOME.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"          # transcriber/
EMOTE_DIR="$(dirname "$PROJECT_DIR")"           # Emote/

CT2_VERSION="${CT2_VERSION:-v4.5.0}"
CUDA_ARCH="${CUDA_ARCH:-87}"
PYTHON="${PYTHON:-python3}"
# The source tree and wheel cache belong to the project, not to $HOME: a copy of
# Emote carries its own ct2/, and nothing outside the folder has to be present.
BUILD_ROOT="${BUILD_ROOT:-$EMOTE_DIR/ct2}"
NPROC="$(nproc)"
JOBS="${JOBS:-$(( NPROC > 4 ? 4 : NPROC ))}"
ASSUME_YES="no"
CLEAN="no"
SKIP_APT="no"

usage() {
  cat <<USAGE
Usage: bash scripts/build_ctranslate2_cuda.sh [options]
  --python <path>      interpreter to install into (default: $PYTHON)
  --version <tag>      CTranslate2 git tag (default: $CT2_VERSION)
  --cuda-arch <NN>     Orin=87, Xavier=72 (default: $CUDA_ARCH)
  --jobs <N>           parallel jobs (default: $JOBS; lower if OOM-killed)
  --build-root <dir>   where to clone/build (default: $BUILD_ROOT)
  --clean              discard an existing build tree and recompile
  --skip-apt           do not apt-get build dependencies
  -y, --yes            no prompts
USAGE
}
while [ $# -gt 0 ]; do
  case "$1" in
    --python) PYTHON="$2"; shift 2 ;;
    --version) CT2_VERSION="$2"; shift 2 ;;
    --cuda-arch) CUDA_ARCH="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --build-root) BUILD_ROOT="$2"; shift 2 ;;
    --clean) CLEAN="yes"; shift ;;
    --skip-apt) SKIP_APT="yes"; shift ;;
    -y|--yes) ASSUME_YES="yes"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

BUILD_LOG="${BUILD_LOG:-$PROJECT_DIR/logs/ct2_build_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$BUILD_LOG")"
exec > >(tee -a "$BUILD_LOG") 2>&1
echo "[ct2-build] started $(date -Is); log: $BUILD_LOG; args: $*"
say()  { printf '\n\033[1m[ct2-build] %s\033[0m\n' "$*"; }
info() { printf '  %s\n' "$*"; }
confirm() { [ "$ASSUME_YES" = "yes" ] && return 0; local a; read -r -p "$1 [y/N]: " a; case "$a" in [Yy]*) return 0 ;; *) return 1 ;; esac; }

# Runs from / on purpose: inside $SRC/python the CTranslate2 *source* package
# shadows the installed one, has no compiled extension, and makes a working
# install look broken (AttributeError: no attribute 'get_cuda_device_count').
verify_cuda() {
  (cd / && "$PYTHON" - <<'PY'
import sys
try:
    import ctranslate2 as c
    import ctranslate2._ext  # surfaces the real error (missing .so, ABI) instead of a silent skip
except Exception as exc:
    print("  ctranslate2 import failed:", type(exc).__name__, exc); sys.exit(1)
n = c.get_cuda_device_count()
print(f"  ctranslate2 {c.__version__} ({c.__file__}): CUDA device count = {n}")
sys.exit(0 if n >= 1 else 1)
PY
  )
}

say "Checking whether $PYTHON already has a CUDA ctranslate2"
if verify_cuda; then say "Already CUDA-enabled -- nothing to do."; exit 0; fi

# ---- CUDA toolkit --------------------------------------------------------------
find_cuda_home() {
  if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then echo "$CUDA_HOME"; return; fi
  if command -v nvcc >/dev/null 2>&1; then dirname "$(dirname "$(command -v nvcc)")"; return; fi
  local d; for d in $(ls -d /usr/local/cuda* 2>/dev/null | sort -r); do [ -x "$d/bin/nvcc" ] && { echo "$d"; return; }; done
  echo ""
}
CUDA_HOME="$(find_cuda_home)"
[ -n "$CUDA_HOME" ] || { echo "ERROR: nvcc not found (install JetPack CUDA or set CUDA_HOME)" >&2; exit 1; }
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

WHEEL_DIR="$BUILD_ROOT/wheels"
SRC="$BUILD_ROOT/CTranslate2"
PYTAG="$("$PYTHON" -c 'import sys;print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')"
EXISTING_WHEEL="$(ls -t "$WHEEL_DIR"/ctranslate2-*-"$PYTAG"-*.whl 2>/dev/null | head -1 || true)"

say "Plan"
info "nvcc        : $(nvcc --version 2>/dev/null | grep -m1 release || echo unknown)"
info "cmake       : $(cmake --version 2>/dev/null | head -1 || echo 'not installed yet')"
info "gcc         : $(gcc --version 2>/dev/null | head -1 || echo 'not installed yet')"
info "cudnn.h     : $(find /usr -name cudnn.h 2>/dev/null | head -1 || echo 'not found')"
info "memory      : $(free -m | awk '/Mem:/ {print $7 " MB available"}')   disk free: $(df -h "$BUILD_ROOT" 2>/dev/null | awk 'NR==2 {print $4}' || df -h "$HOME" | awk 'NR==2 {print $4}')"
info "CTranslate2 : $CT2_VERSION   CUDA: $CUDA_HOME   arch: $CUDA_ARCH"
info "Python      : $PYTHON ($("$PYTHON" -V 2>&1))"
info "Build root  : $BUILD_ROOT"
if [ "$CLEAN" = "no" ] && [ -n "$EXISTING_WHEEL" ]; then
  info "Found built wheel: $EXISTING_WHEEL  (will install it, no compile)"
elif [ "$CLEAN" = "no" ] && [ -f "$SRC/build/libctranslate2.so" ]; then
  info "Found compiled library in $SRC/build (will install it, no compile)"
else
  info "Full compile needed (~30-45 min on Orin Nano, -j$JOBS)"
fi
confirm "Proceed?" || { echo "Aborted -- nothing changed."; exit 0; }

# ---- fast path: wheel already built -------------------------------------------------
if [ "$CLEAN" = "no" ] && [ -n "$EXISTING_WHEEL" ]; then
  say "Installing cached wheel"
  if [ ! -f /usr/local/lib/libctranslate2.so ] && [ -f "$SRC/build/libctranslate2.so" ]; then
    (cd "$SRC/build" && make -j"$JOBS" && sudo make install >/dev/null && sudo ldconfig)
  fi
  "$PYTHON" -m pip install --force-reinstall --no-deps "$EXISTING_WHEEL"
  if verify_cuda; then say "SUCCESS -- CUDA CTranslate2 installed from cached wheel."; exit 0; fi
  info "Cached wheel did not give CUDA; rebuilding."
fi

# ---- deps -----------------------------------------------------------------------------
if [ "$SKIP_APT" = "no" ]; then
  say "1) Build dependencies (apt)"
  sudo apt-get update -qq || info "apt update had issues; continuing."
  sudo apt-get install -y -qq cmake build-essential git pkg-config libopenblas-dev || info "some apt packages failed; continuing."
  if ! find /usr -name 'cudnn.h' 2>/dev/null | grep -q cudnn.h; then
    sudo apt-get install -y -qq libcudnn9-dev-cuda-12 2>/dev/null || sudo apt-get install -y -qq cudnn 2>/dev/null || info "cuDNN dev headers not auto-installed; build may fail on cudnn.h"
  fi
fi

# ---- source -----------------------------------------------------------------------------
say "2) Source $CT2_VERSION"
mkdir -p "$BUILD_ROOT" "$WHEEL_DIR"
if [ "$CLEAN" = "yes" ]; then rm -rf "$SRC"; fi
if [ ! -d "$SRC" ]; then
  git clone --depth 1 --branch "$CT2_VERSION" --recursive https://github.com/OpenNMT/CTranslate2.git "$SRC"
else
  info "Reusing $SRC"
fi

# ---- compile ------------------------------------------------------------------------------
if [ "$CLEAN" = "yes" ] || [ ! -f "$SRC/build/libctranslate2.so" ]; then
  say "3) Configure + compile (-j$JOBS)"
  rm -rf "$SRC/build"; mkdir -p "$SRC/build"; cd "$SRC/build"
  cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_MKL=OFF \
           -DWITH_OPENBLAS=ON -DOPENMP_RUNTIME=COMP -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
  T0=$(date +%s)
  make -j"$JOBS"
  info "compile took $(( ($(date +%s) - T0) / 60 )) min"
else
  # Incremental as the normal user: a no-op when the tree is current, parallel when
  # files are stale. Leaving stale files to `sudo make install` rebuilds them
  # single-threaded and as root (root-owned objects in your build tree).
  say "3) Incremental build (no-op if the existing tree is up to date, -j$JOBS)"
  T0=$(date +%s)
  if ! (cd "$SRC/build" && make -j"$JOBS"); then
    info "make failed. If the messages say 'Permission denied', an earlier 'sudo make install'"
    info "left root-owned files in $SRC; fix with: sudo chown -R \"\$USER\" \"$SRC\""
    exit 1
  fi
  info "incremental build took $(( ($(date +%s) - T0) )) s"
fi

say "4) Install C++ library"
(cd "$SRC/build" && sudo make install >/dev/null && sudo ldconfig)

say "5) Build + install Python binding"
cd "$SRC/python"
"$PYTHON" -m pip install -q -r install_requirements.txt wheel
export CTRANSLATE2_ROOT=/usr/local
rm -rf build
"$PYTHON" -m pip wheel --no-build-isolation --no-deps -w "$WHEEL_DIR" .
NEW_WHEEL="$(ls -t "$WHEEL_DIR"/ctranslate2-*-"$PYTAG"-*.whl | head -1)"
"$PYTHON" -m pip install --force-reinstall --no-deps "$NEW_WHEEL"

say "6) Verify"
if verify_cuda; then
  say "SUCCESS -- CUDA CTranslate2 installed into $PYTHON (wheel cached at $NEW_WHEEL)."
  exit 0
fi
say "FAILED -- CUDA device count is still 0 in $PYTHON."
info "Check: nvcc --version; ls /usr/local/lib/libctranslate2*; that --python is the venv interpreter."
exit 1
