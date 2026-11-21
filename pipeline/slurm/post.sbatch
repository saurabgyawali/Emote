#!/bin/bash
# After a training array: analysis (results/summary.json, RESULTS.md, figures/) and the ONNX
# export of the best deployable model for the Jetson (CPU work). submit.sh chains it with
# --dependency=afterany, so it runs even if some tasks failed (the analysis reports what exists).
#
#   bash pipeline/slurm/submit.sh post                 # analyze + export now
#   sbatch --export=ALL,STEP=analyze pipeline/slurm/post.sbatch
#
# dgx_iacc rejects jobs without a GPU ("Missing GRES flag", 2026-11-04), so this CPU work
# holds one GPU too.
#SBATCH --job-name=emote_post
#SBATCH --account=dsi_dgx_iacc
#SBATCH --partition=interactive_gpu
#SBATCH --qos=dgx_iacc
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=pipeline/logs/slurm/post_%j.out
#SBATCH --error=pipeline/logs/slurm/post_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source pipeline/slurm/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
STEP=${STEP:-all}
echo "=== emote post ($STEP)  host=$(hostname)  $(date) ==="

if [ "$STEP" = all ] || [ "$STEP" = analyze ]; then
  python pipeline/scripts/visualization/analyze.py --data-root "$EMOTE_DATA"
fi
if [ "$STEP" = all ] || [ "$STEP" = export ]; then
  # shellcheck disable=SC2086
  python pipeline/scripts/export/export_onnx.py --data-root "$EMOTE_DATA" ${EXPORT_ARGS:-}
fi
echo "=== done $(date) ==="
bash pipeline/run.sh status
