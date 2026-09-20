#!/usr/bin/env bash
# Bundle everything needed to diagnose a run into one tar.gz you can copy off the Jetson.
#
#   bash scripts/collect_diagnostics.sh               # logs, session metadata, config, system/audio/CUDA facts
#   bash scripts/collect_diagnostics.sh --with-audio  # also session.wav and clips (large; a recording of your voice)
#   bash scripts/collect_diagnostics.sh --out DIR     # where to write the tarball (default: the project folder)
#   bash scripts/collect_diagnostics.sh --dry-run     # list what would be collected; run and write nothing
#
# Read-only apart from the tarball itself. It runs `run.sh --doctor` and `--list-devices`
# first so their logs are included. Logs and transcripts contain the words that were
# spoken, so look before sharing.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$HERE")"
CT2_BUILD_ROOT="${CT2_BUILD_ROOT:-$(dirname "$PROJECT_DIR")/ct2}"
OUT_DIR="$PROJECT_DIR"; WITH_AUDIO="no"; DRY="no"
while [ $# -gt 0 ]; do
  case "$1" in
    --with-audio) WITH_AUDIO="yes"; shift ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --dry-run) DRY="yes"; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ "$DRY" = "yes" ]; then
  echo "Would collect from $PROJECT_DIR:"
  echo "  system.txt   OS, L4T/JetPack, CUDA, memory, disk, top memory users, power mode, tegrastats sample"
  echo "  audio.txt    lsusb, arecord -l/-L, /dev/snd holders, sound-server processes, PortAudio device list, capture gain (amixer/pactl)"
  echo "  python.txt   interpreter, pip list, ctranslate2 version/CUDA/compute types/library resolution"
  echo "  kernel.txt   dmesg / journal tail, OOM-kill and USB/audio lines"
  echo "  build.txt    $CT2_BUILD_ROOT contents, installed libctranslate2, ldconfig entries"
  echo "  doctor.txt   output of run.sh --doctor and run.sh --list-devices"
  echo "  project/     config.toml, .venv_path, logs/, and per session: session.json, utterances.jsonl, transcript.txt, run.log"
  [ "$WITH_AUDIO" = "yes" ] && echo "  project/     plus session.wav and clips/ (--with-audio)"
  echo "Output: $OUT_DIR/diagnostics_<host>_<timestamp>.tar.gz"
  exit 0
fi

VENV_DIR="${VENV_DIR:-}"
[ -z "$VENV_DIR" ] && [ -f "$PROJECT_DIR/.venv_path" ] && VENV_DIR="$(cat "$PROJECT_DIR/.venv_path")"
[ -z "$VENV_DIR" ] && VENV_DIR="$PROJECT_DIR/.venv"
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY="$(command -v python3 || echo python3)"
export PROJECT_DIR CT2_BUILD_ROOT PY VENV_DIR PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

STAMP="$(date +%Y%m%d_%H%M%S)"
NAME="diagnostics_$(hostname)_$STAMP"
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
W="$STAGE/$NAME"; mkdir -p "$W/project"; export STAGE W

# sh_run <file> <shell snippet>: append the snippet's output (stdout+stderr) under a header; never fails the script.
sh_run() { local out="$W/$1"; shift; { echo "\$ $*"; timeout 30 bash -c "$*" 2>&1; echo "[exit $?]"; echo; } >> "$out" 2>&1 || true; }

echo "Collecting diagnostics (about 30 seconds)..."

sh_run doctor.txt 'bash "$PROJECT_DIR/run.sh" --doctor'
sh_run doctor.txt 'bash "$PROJECT_DIR/run.sh" --list-devices'

sh_run system.txt 'date -Is; uptime; uname -a; nproc'
sh_run system.txt 'head -4 /etc/os-release; head -2 /etc/nv_tegra_release; tr -d "\0" < /proc/device-tree/model; echo'
sh_run system.txt 'nvcc --version || /usr/local/cuda/bin/nvcc --version'
sh_run system.txt 'free -m; df -h "$PROJECT_DIR" /'
sh_run system.txt 'ps -eo pid,rss,comm --sort=-rss | head -12'
sh_run system.txt 'nvpmodel -q'
sh_run system.txt 'timeout 4 tegrastats --interval 1000 | head -3'

sh_run audio.txt 'id; ls -l /dev/snd'
sh_run audio.txt 'lsusb'
sh_run audio.txt 'arecord -l; arecord -L | head -60'
sh_run audio.txt 'fuser -v /dev/snd/*'
sh_run audio.txt 'pgrep -a "pulseaudio|pipewire|wireplumber|jackd"; pactl info | head -6; pactl list short sources'
sh_run audio.txt '"$PY" -c "import sounddevice as sd; print(sd.__version__, sd.get_portaudio_version()); print(sd.query_devices()); print(sd.query_hostapis())"'

cat > "$STAGE/ct2_probe.py" <<'EOF'
import subprocess, traceback
try:
    import ctranslate2 as c
    import ctranslate2._ext as ext
    print("ctranslate2", c.__version__, c.__file__)
    print("cuda devices", c.get_cuda_device_count())
    for dev in ("cpu", "cuda"):
        try:
            print(dev, "compute types", sorted(c.get_supported_compute_types(dev)))
        except Exception as exc:
            print(dev, "compute types: error", exc)
    print(subprocess.run(["ldd", ext.__file__], capture_output=True, text=True).stdout)
except Exception:
    traceback.print_exc()
EOF
sh_run python.txt '"$PY" -V; echo "$PY"; "$PY" -m pip --version'
sh_run python.txt 'cd / && "$PY" "$STAGE/ct2_probe.py"'
sh_run python.txt '"$PY" -c "import transcriber, faster_whisper, numpy, scipy, webrtcvad; print(\"transcriber\", transcriber.__version__, \"faster_whisper\", faster_whisper.__version__, \"numpy\", numpy.__version__, \"scipy\", scipy.__version__)"'
sh_run python.txt '"$PY" -m pip list'

sh_run audio.txt 'for c in $(cat /proc/asound/cards 2>/dev/null | grep -oE "^ *[0-9]+" ); do echo "--- card $c ---"; amixer -c "$c" contents 2>&1 | head -60; done'
sh_run audio.txt 'amixer -c 0 2>&1 | head -40'
sh_run audio.txt 'pactl list sources 2>/dev/null | grep -E "Name:|Volume:|Mute:|Description:"'

sh_run kernel.txt 'dmesg | tail -80'
sh_run kernel.txt 'journalctl -k -n 80 --no-pager'
sh_run kernel.txt 'dmesg | grep -iE "oom|out of memory|killed process|usb.*(disconnect|reset|error)|snd|audio" | tail -40'

sh_run build.txt 'ls -la "$CT2_BUILD_ROOT" "$CT2_BUILD_ROOT/wheels"'
sh_run build.txt 'ls -la /usr/local/lib | grep -i ctranslate2'
sh_run build.txt 'ldconfig -p | grep -Ei "ctranslate2|libcudnn|libcublas"'

cp "$PROJECT_DIR/config.toml" "$PROJECT_DIR/.venv_path" "$W/project/" 2>/dev/null
[ -d "$PROJECT_DIR/logs" ] && cp -r "$PROJECT_DIR/logs" "$W/project/logs"
for s in "$PROJECT_DIR"/sessions/*/; do
  [ -d "$s" ] || continue
  d="$W/project/sessions/$(basename "$s")"; mkdir -p "$d"
  for f in session.json utterances.jsonl transcript.txt run.log; do [ -f "$s$f" ] && cp "$s$f" "$d/"; done
  [ "$WITH_AUDIO" = "yes" ] && cp -r "$s"clips "$s"session.wav "$d/" 2>/dev/null
done

mkdir -p "$OUT_DIR"
tar -C "$STAGE" -czf "$OUT_DIR/$NAME.tar.gz" "$NAME"
echo "Wrote $OUT_DIR/$NAME.tar.gz ($(du -h "$OUT_DIR/$NAME.tar.gz" | cut -f1))"
[ "$WITH_AUDIO" = "yes" ] || echo "Audio not included (add --with-audio if the recordings are needed)."
echo "It contains logs and transcripts, which include the words that were spoken."
