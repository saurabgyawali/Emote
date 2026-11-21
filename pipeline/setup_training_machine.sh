#!/usr/bin/env bash
# One-time setup of a GPU training machine for the Emote emotion model. Run it after copying
# the Emote folder over; it is idempotent, so re-running resumes where it stopped.
#
#   bash pipeline/setup_training_machine.sh                       # everything, data in ~/emote_data
#   bash pipeline/setup_training_machine.sh --data-root /big/disk/emote_data --skip-meld
#   bash pipeline/setup_training_machine.sh --dry-run             # print the plan, change nothing
#
# Steps: checks -> virtualenv -> PyTorch (CUDA build matched to the driver) -> requirements
#        -> pretrained encoders -> RAVDESS, CREMA-D, TESS, MELD -> manifest + 16 kHz cache
#        -> prosody features -> quality checks -> unit tests -> end-to-end smoke test on the GPU
# Options:
#   --data-root DIR     where data, checkpoints and caches go (~35 GB with MELD, ~6 GB without)
#   --venv DIR          virtualenv location (default ~/emote-train-venv)
#   --cuda TAG          cu124 (default) | cu126 | cu118 | cpu -- the PyTorch wheel index
#   --skip-meld         skip the 10.9 GB MELD download (the only open corpus with real transcripts)
#   --extended-models   also HuBERT and wav2vec 2.0 (experiments E6, E7)
#   --large-model       also WavLM-Large (experiment E8; needs a >= 24 GB GPU)
#   --esd               also ESD, 10 English speakers (only after its licence form was emailed)
#   --crema-label L     acted (default) | voice_vote (what listeners heard)
#   --no-smoke          skip the final smoke test
#   --dry-run
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMOTE="$(dirname "$HERE")"
DATA="${EMOTE_DATA:-$HOME/emote_data}"
VENV="$HOME/emote-train-venv"
CUDA_TAG="cu124"
SKIP_MELD=0; ESD=0; EXT=0; LARGE=0; CREMA="acted"; SMOKE=1; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --data-root) DATA="$2"; shift 2 ;;
    --venv) VENV="$2"; shift 2 ;;
    --cuda) CUDA_TAG="$2"; shift 2 ;;
    --skip-meld) SKIP_MELD=1; shift ;;
    --esd) ESD=1; shift ;;
    --extended-models) EXT=1; shift ;;
    --large-model) LARGE=1; shift ;;
    --crema-label) CREMA="$2"; shift 2 ;;
    --no-smoke) SMOKE=0; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
DATA="$(mkdir -p "$DATA" 2>/dev/null; cd "$DATA" 2>/dev/null && pwd || echo "$DATA")"
mkdir -p "$DATA/logs"
LOG="$DATA/logs/setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[31mSetup stopped: %s\033[0m\n    log: %s\n' "$*" "$LOG"; exit 1; }
run()  { info "\$ $*"; [ "$DRY" = 1 ] || "$@" || die "failed: $*"; }

say "Emote training-machine setup  ($(date))"
info "Emote folder : $EMOTE"
info "data root    : $DATA"
info "virtualenv   : $VENV"
info "log          : $LOG"

say "1) Checks"
PY3="$(command -v python3 || true)"
[ -n "$PY3" ] || die "python3 not found"
"$PY3" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' || die "Python >= 3.9 needed, found $("$PY3" -V)"
info "python : $("$PY3" -V) ($PY3)"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/    gpu    : /'
else
  info "gpu    : none visible (nvidia-smi missing) -- training would run on CPU, which is only good for smoke tests"
  [ "$CUDA_TAG" = "cpu" ] || info "         consider --cuda cpu if this machine really has no GPU"
fi
need=$([ "$SKIP_MELD" = 1 ] && echo 8 || echo 40)
free=$(df -BG --output=avail "$DATA" | tail -1 | tr -dc 0-9)
info "disk   : ${free} GB free under $DATA (need ~${need} GB)"
[ "$DRY" = 1 ] || [ "$free" -ge "$need" ] || die "not enough disk under $DATA"
info "nproc  : $(nproc), RAM $(free -g | awk '/Mem:/{print $2}') GB"

say "2) Virtual environment"
if [ ! -x "$VENV/bin/python" ]; then
  run "$PY3" -m venv "$VENV"
else
  info "reusing $VENV"
fi
PY="$VENV/bin/python"
run "$PY" -m pip install -q --upgrade pip wheel

say "3) PyTorch ($CUDA_TAG)"
if [ "$DRY" = 0 ] && "$PY" -c 'import torch, sys; v=tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]); sys.exit(0 if v >= (2, 6) else 1)' 2>/dev/null; then
  info "torch $("$PY" -c 'import torch; print(torch.__version__)') already installed"
else
  if [ "$CUDA_TAG" = "cpu" ]; then IDX="https://download.pytorch.org/whl/cpu"; else IDX="https://download.pytorch.org/whl/$CUDA_TAG"; fi
  run "$PY" -m pip install -q "torch>=2.6" "torchaudio>=2.6" --index-url "$IDX"
fi
[ "$DRY" = 1 ] || "$PY" -c 'import torch; print("    torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")'

say "4) Python requirements"
run "$PY" -m pip install -q -r "$HERE/requirements.txt"
export EMOTE_DATA="$DATA" HF_HOME="$DATA/pretrained" HF_HUB_CACHE="$DATA/pretrained/hub" HUGGINGFACE_HUB_CACHE="$DATA/pretrained/hub"
if [ "$DRY" = 0 ]; then
  echo "$VENV" > "$HERE/.venv_path"
  echo "$DATA" > "$HERE/.data_root"
  printf '# source this in a new shell to use the pipeline by hand\nexport EMOTE_DATA="%s"\nexport HF_HOME="%s/pretrained" HF_HUB_CACHE="%s/pretrained/hub" HUGGINGFACE_HUB_CACHE="%s/pretrained/hub"\nsource "%s/bin/activate"\n' \
    "$DATA" "$DATA" "$DATA" "$DATA" "$VENV" > "$DATA/emote_env.sh"
  info "wrote $DATA/emote_env.sh"
fi
# The download scripts need only the standard library, so a dry run can use the system python.
DLPY="$PY"; [ "$DRY" = 1 ] && [ ! -x "$PY" ] && DLPY="$PY3"

say "5) Downloads (resumable; re-run this script if a network error stops it)"
DL=(--data-root "$DATA")
[ "$SKIP_MELD" = 1 ] && DL+=(--skip-meld)
[ "$ESD" = 1 ] && DL+=(--esd)
[ "$EXT" = 1 ] && DL+=(--extended-models)
[ "$LARGE" = 1 ] && DL+=(--large-model)
[ "$DRY" = 1 ] && DL+=(--dry-run)
"$DLPY" "$HERE/scripts/downloads/prepare_all_external.py" "${DL[@]}" || die "downloads incomplete -- re-run to resume"

say "6) Data preparation"
run "$PY" "$HERE/scripts/preprocessing/build_manifest.py" --data-root "$DATA" --crema-label "$CREMA" --workers "$(nproc)"
run "$PY" "$HERE/scripts/preprocessing/extract_prosody.py" --data-root "$DATA" --workers "$(nproc)"
run "$PY" "$HERE/scripts/preprocessing/quality_checks.py" --data-root "$DATA"

say "7) Tests"
run "$PY" -m pytest -q "$EMOTE/emotion_model/tests"
if [ "$SMOKE" = 1 ]; then
  run "$PY" "$HERE/scripts/training/smoke_test.py"
fi

say "Setup complete"
info "Everything is in $DATA; the pipeline remembers it (pipeline/.data_root)."
info "Next:  bash pipeline/run.sh status"
info "       bash pipeline/run.sh train                 # core suite: 6 experiments x 3 seeds"
info "       bash pipeline/run.sh analyze && bash pipeline/run.sh export"
info "Full guide: NEXT_STEPS.md"
