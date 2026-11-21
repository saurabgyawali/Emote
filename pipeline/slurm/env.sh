# Emote on ACCRE: the ONE file every *.sbatch in this folder sources first.
# Modelled on DALE's clip_clip/slurm/env.sh, which hit every pitfall handled here.
#
# Check on a login node, from the Emote folder:
#   source pipeline/slurm/env.sh && python -c "import torch; print(torch.__version__)"

# ---- 1. Python environment -------------------------------------------------
# A Miniforge env, not a venv: conda-forge brings its own libs, while the /cvmfs
# Compute Canada wheels need GLIBC 2.35 and ACCRE's Rocky 9 has 2.34 (matplotlib
# broke that way in the DALE env, 2026-10-30).
export EMOTE_ENV="${EMOTE_ENV:-$HOME/miniforge3/envs/emote}"
# Lmod's module function and conda's activate scripts read unset variables, and
# the sbatch files run under `set -u` (DALE job 14542587 died in 1 s from this).
case $- in *u*) _emote_had_u=1 ;; *) _emote_had_u= ;; esac
set +u
module purge >/dev/null 2>&1 || true
source "$HOME/miniforge3/etc/profile.d/conda.sh"
# sbatch copies the login shell's env; if `emote` is already active, `conda activate`
# is a no-op and the system python that `module purge` exposed wins. Deactivate first.
for _i in 1 2 3 4 5; do [ -n "${CONDA_PREFIX:-}" ] || break; conda deactivate || break; done
if [ -x "$EMOTE_ENV/bin/python" ]; then conda activate "$EMOTE_ENV"; fi
if [ -n "$_emote_had_u" ]; then set -u; fi
unset _emote_had_u _i
if [ "${EMOTE_ALLOW_NO_ENV:-0}" != 1 ] && [ "$(command -v python)" != "$EMOTE_ENV/bin/python" ]; then
  echo "env.sh: python is '$(command -v python)', not $EMOTE_ENV -- run: bash pipeline/slurm/submit.sh setup" >&2
  return 1
fi

# pip must never pick up the cluster's /cvmfs wheelhouse (GLIBC mismatch, see above).
export PIP_CONFIG_FILE=/dev/null PIP_NO_CACHE_DIR=1
unset PIP_FIND_LINKS PIP_EXTRA_INDEX_URL PIP_INDEX_URL PYTHONPATH

# ---- 2. Where things live --------------------------------------------------
# Home, not /nobackup: the dgx GPU nodes cannot write /nobackup/userspace by any path
# (DALE jobs 14809208, 14809319). Home quota is 186 GB and panfs charges ~1.16x the bytes.
export EMOTE_DATA="${EMOTE_DATA:-$HOME/emote_data}"
export EMOTE_TRAIN_VENV="$EMOTE_ENV"
# Our own model cache, always: a machine-wide HF_HOME belongs to someone else.
export HF_HOME="$EMOTE_DATA/pretrained" HF_HUB_CACHE="$EMOTE_DATA/pretrained/hub" HUGGINGFACE_HUB_CACHE="$EMOTE_DATA/pretrained/hub"
mkdir -p "$EMOTE_DATA" pipeline/logs/slurm

# ---- 3. Nothing to edit below ----------------------------------------------
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "emote env: python=$(command -v python)  data=$EMOTE_DATA  host=$(hostname)"
