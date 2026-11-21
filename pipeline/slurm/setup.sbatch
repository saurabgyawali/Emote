#!/bin/bash
# One-time ACCRE setup on a GPU node: builds the Miniforge env `emote`, then runs
# pipeline/setup_training_machine.sh (PyTorch, requirements, every open-corpus and encoder
# download, manifest + 16 kHz cache, prosody features, quality checks, unit tests, GPU
# smoke test). Compute nodes have https. Idempotent: resubmit after any failure to resume.
#
# Submit through submit.sh from the Emote folder:
#   bash pipeline/slurm/submit.sh setup                       # core: open corpora + WavLM/RoBERTa
#   bash pipeline/slurm/submit.sh setup --all-models          # + HuBERT, wav2vec2, WavLM-Large
#   bash pipeline/slurm/submit.sh prepare                     # after adding IEMOCAP / MSP-Podcast
#SBATCH --job-name=emote_setup
#SBATCH --account=dsi_dgx_iacc
#SBATCH --partition=interactive_gpu
#SBATCH --qos=dgx_iacc
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=pipeline/logs/slurm/setup_%j.out
#SBATCH --error=pipeline/logs/slurm/setup_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
MODE=${MODE:-setup}
CUDA_TAG=${CUDA_TAG:-cu124}
SETUP_ARGS=${SETUP_ARGS:-}
echo "=== emote $MODE  host=$(hostname)  $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

ENV_DIR="${EMOTE_ENV:-$HOME/miniforge3/envs/emote}"
if [ ! -x "$ENV_DIR/bin/python" ]; then
  echo "--- creating conda env $ENV_DIR (conda-forge only)"
  set +u
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  # The compiled scientific stack comes from conda-forge (its own libs, no GLIBC clash);
  # setup_training_machine.sh then pip-installs torch and the rest from PyPI.
  conda create -y -p "$ENV_DIR" -c conda-forge --override-channels \
    python=3.11 pip numpy scipy pandas pyarrow matplotlib scikit-learn librosa pysoundfile pyyaml ffmpeg
  set -u
fi
source pipeline/slurm/env.sh
# Pin the combination validated on real data (2026-10-10: torch 2.6.0+cu124); the setup
# script then sees torch >= 2.6 and leaves it alone instead of taking the newest release.
if ! python -c 'import torch' 2>/dev/null; then
  python -m pip install -q "torch==2.6.0" "torchaudio==2.6.0" --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
fi

case $MODE in
  setup)
    # shellcheck disable=SC2086
    bash pipeline/setup_training_machine.sh --data-root "$EMOTE_DATA" --venv "$EMOTE_ENV" --cuda "$CUDA_TAG" $SETUP_ARGS ;;
  prepare)
    # --force rebuilds the manifest only; the audio cache and prosody rows are reused.
    python pipeline/scripts/preprocessing/build_manifest.py --data-root "$EMOTE_DATA" --workers "$SLURM_CPUS_PER_TASK" --force
    python pipeline/scripts/preprocessing/extract_prosody.py --data-root "$EMOTE_DATA" --workers "$SLURM_CPUS_PER_TASK"
    python pipeline/scripts/preprocessing/quality_checks.py --data-root "$EMOTE_DATA" ;;
  *) echo "MODE must be setup or prepare, not $MODE" >&2; exit 2 ;;
esac
echo "=== done $(date) ==="
bash pipeline/run.sh status
