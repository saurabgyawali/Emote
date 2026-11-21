#!/bin/bash
# Training: one array task = one (experiment, seed) run, mapped by task_list.py from
# configs/suite.yaml. train.py skips a run whose config fingerprint already finished, so
# resubmitting the same array only redoes what failed or never ran.
#
# Submit through submit.sh (it sizes --array and chains post.sbatch):
#   bash pipeline/slurm/submit.sh train                              # core: 6 experiments x 3 seeds = 18 tasks
#   bash pipeline/slurm/submit.sh train --groups core extended large
#   bash pipeline/slurm/submit.sh train --experiments E5_audio_text_prosody --seeds 0 --override train.epochs=2
#
# Peak GPU memory measured 2026-10-10: 5.3 GB (E5, batch 16), 7.1 GB (WavLM-Large, batch 8),
# so a 40 GB A100 is plenty. The header asks for an 80 GB one because dgx_iacc wants a typed
# GRES and those were free on 2026-11-04; when they are busy: submit.sh train --gres=gpu:nvidia_a100-sxm4-40gb:1
#SBATCH --job-name=emote_train
#SBATCH --account=dsi_dgx_iacc
#SBATCH --partition=interactive_gpu
#SBATCH --qos=dgx_iacc
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=pipeline/logs/slurm/train_%A_%a.out
#SBATCH --error=pipeline/logs/slurm/train_%A_%a.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source pipeline/slurm/env.sh
# Every encoder was cached by setup; offline mode stops many tasks hitting the hub at once.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# shellcheck disable=SC2086
read -r EXP SEED < <(python pipeline/slurm/task_list.py ${EMOTE_SELECT:-} --index "$SLURM_ARRAY_TASK_ID")
LOG="pipeline/results/$EXP/seed$SEED.log"
mkdir -p "$(dirname "$LOG")"

echo "=== $EXP seed $SEED  task=$SLURM_ARRAY_TASK_ID  host=$(hostname)  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=== $(date '+%F %T') slurm $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID on $(hostname)" >> "$LOG"
# shellcheck disable=SC2086
python pipeline/scripts/training/train.py --experiment "$EXP" --seed "$SEED" --data-root "$EMOTE_DATA" ${TRAIN_ARGS:-} 2>&1 | tee -a "$LOG"
echo "=== done $(date) ==="
