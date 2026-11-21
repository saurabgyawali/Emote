#!/bin/bash
# One command per stage on ACCRE (run on a login node, from anywhere). It checks that the
# stage can start (env built, data prepared, encoders cached, room in home, nothing of the
# same name already queued), submits, and records it in pipeline/logs/submissions.log.
#
#   bash pipeline/slurm/submit.sh setup [--all-models] [--esd]   # env + downloads + data prep + GPU smoke test
#   bash pipeline/slurm/submit.sh prepare                # rebuild manifest after adding IEMOCAP / MSP-Podcast
#   bash pipeline/slurm/submit.sh list [selection]       # print the array index -> run table, submit nothing
#   bash pipeline/slurm/submit.sh train [selection]      # GPU array, one task per run, + post.sbatch afterany
#   bash pipeline/slurm/submit.sh post                   # analysis + ONNX export now
#   bash pipeline/slurm/submit.sh status                 # queue, finished runs, quota
#
# selection: --groups core extended large | --experiments E5_audio_text_prosody ... | --seeds 0 1 2
#            --override train.epochs=2 ...   (passed to train.py)
# Anything else that starts with -- goes to sbatch in its = form (e.g. --time=02:00:00, --array=3,7).
# --force skips the space and already-queued checks.
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE=${1:-}; [ $# -gt 0 ] && shift

FORCE=0; ALL_MODELS=0; ESD=0; SBATCH_EXTRA=(); SELECT=(); OVERRIDES=(); mode=""
for a in "$@"; do
  case $a in
    --force) FORCE=1; mode="" ;;
    --all-models) ALL_MODELS=1; mode="" ;;
    --esd) ESD=1; mode="" ;;
    --groups|--experiments|--seeds) SELECT+=("$a"); mode=select ;;
    --override) mode=override ;;
    --*) SBATCH_EXTRA+=("$a"); mode="" ;;
    *)
      case $mode in
        select) SELECT+=("$a") ;;
        override) OVERRIDES+=("$a") ;;
        *) echo "unexpected argument '$a' (sbatch options need the --opt=value form)" >&2; exit 2 ;;
      esac ;;
  esac
done

case $STAGE in
  setup|prepare) JOB=setup.sbatch; NAME=emote_setup ;;
  train)         JOB=train.sbatch; NAME=emote_train ;;
  post)          JOB=post.sbatch;  NAME=emote_post ;;
  list|status)   ;;
  *) sed -n '2,16p' "$0"; exit 2 ;;
esac

EMOTE_ALLOW_NO_ENV=1 source pipeline/slurm/env.sh
PY="$EMOTE_ENV/bin/python"

if [ "$STAGE" = status ]; then
  squeue -u "$USER" -o "%.12i %.14j %.8T %.10M %.12l %R" || true
  [ -f pipeline/logs/submissions.log ] && { echo; echo "last submissions:"; tail -n 5 pipeline/logs/submissions.log; }
  echo; bash pipeline/run.sh status
  echo; accre_storage 2>/dev/null || true
  exit 0
fi

if [ "$STAGE" != setup ] && [ ! -x "$PY" ]; then
  echo "STOP: no env at $EMOTE_ENV. Run: bash pipeline/slurm/submit.sh setup" >&2; exit 1
fi
# task_list.py needs only PyYAML; before setup the env may not exist, so fall back to python3.
TLPY="$PY"; [ -x "$TLPY" ] || TLPY=python3
export EMOTE_SELECT="${SELECT[*]:-}"

if [ "$STAGE" = list ]; then
  # shellcheck disable=SC2086
  "$TLPY" pipeline/slurm/task_list.py $EMOTE_SELECT; exit 0
fi

NEED=0
case $STAGE in
  setup)
    NEED=50
    SETUP_ARGS=""
    [ "$ALL_MODELS" = 1 ] && SETUP_ARGS="--extended-models --large-model"
    [ "$ESD" = 1 ] && SETUP_ARGS="$SETUP_ARGS --esd"
    export SETUP_ARGS
    export MODE=setup ;;
  prepare)
    export MODE=prepare ;;
  train)
    [ -f "$EMOTE_DATA/manifests/manifest.csv" ] || { echo "STOP: no manifest in $EMOTE_DATA. Finish setup first (submit.sh status)." >&2; exit 1; }
    # shellcheck disable=SC2086
    N=$("$TLPY" pipeline/slurm/task_list.py $EMOTE_SELECT --count)
    # shellcheck disable=SC2086
    EXPS=$("$TLPY" pipeline/slurm/task_list.py $EMOTE_SELECT | awk '{print $2}' | sort -u)
    # Train jobs run offline, so every encoder they name must already be in the cache.
    for e in $EXPS; do
      enc=$(grep -o 'encoder: *[^,}]*' "pipeline/configs/experiments/$e.yaml" | head -1 | sed 's/encoder: *//; s/ *$//' || true)
      enc=${enc:-microsoft/wavlm-base-plus}
      if [ ! -d "$HF_HUB_CACHE/models--${enc//\//--}" ]; then
        echo "STOP: $e needs $enc, which is not in the encoder cache. Run setup first (add --all-models for E6-E8)" >&2; exit 1
      fi
    done
    # ~0.7 GB best.pt per run (1.3 GB for WavLM-Large) x panfs 1.16 -> plan 1.5 G each.
    NEED=$(( (N * 15 + 9) / 10 ))
    export TRAIN_ARGS="${OVERRIDES[*]:+--override ${OVERRIDES[*]}}"
    HAS_ARRAY=0
    for o in "${SBATCH_EXTRA[@]+"${SBATCH_EXTRA[@]}"}"; do case $o in --array=*) HAS_ARRAY=1 ;; esac; done
    [ "$HAS_ARRAY" = 1 ] || SBATCH_EXTRA+=("--array=0-$((N - 1))")
    "$TLPY" pipeline/slurm/task_list.py $EMOTE_SELECT ;;
esac

if [ "$FORCE" -eq 0 ]; then
  if [ -n "$(squeue -h -u "$USER" -n "$NAME" -o %i)" ]; then
    echo "STOP: $NAME jobs are already queued or running (squeue -u $USER). A second copy would write into the same runs." >&2
    exit 1
  fi
  free=$(accre_storage 2>/dev/null | awk '
    function g(x,  m, v) { m = substr(x, length(x)); v = substr(x, 1, length(x) - 1) + 0
      return m == "T" ? v * 1024 : m == "G" ? v : m == "M" ? v / 1024 : m == "K" ? v / 1048576 : x + 0 }
    /^Home \(user\)/ { printf "%d", g($4) - g($3) }')
  if [ "$NEED" -eq 0 ]; then
    :
  elif [ -z "$free" ]; then
    echo "  space: could not read accre_storage; check home yourself (this stage needs up to ~${NEED} G)"
  elif [ "$free" -lt "$NEED" ]; then
    echo "STOP: home has ${free} G free, this stage needs up to ~${NEED} G. Free space first, or add --force." >&2
    exit 1
  else
    echo "  space: home ${free} G free, this stage needs up to ~${NEED} G"
  fi
fi

mkdir -p pipeline/logs/slurm
# Job e-mails go to EMOTE_MAIL (e.g. in ~/.bashrc) so no address lives in the repo;
# unset, Slurm mails the submitting account.
MAIL=(); if [ -n "${EMOTE_MAIL:-}" ]; then MAIL=(--mail-user="$EMOTE_MAIL"); fi
jid=$(sbatch --parsable "${MAIL[@]+"${MAIL[@]}"}" "${SBATCH_EXTRA[@]+"${SBATCH_EXTRA[@]}"}" "pipeline/slurm/$JOB"); jid=${jid%%;*}
line="$(date '+%F %T')  $STAGE  job=$jid  ${EMOTE_SELECT} ${TRAIN_ARGS:-} ${SBATCH_EXTRA[*]:-}"
echo
echo "  $JOB -> job $jid"
if [ "$STAGE" = train ]; then
  aid=$(sbatch --parsable "${MAIL[@]+"${MAIL[@]}"}" --dependency=afterany:"$jid" pipeline/slurm/post.sbatch); aid=${aid%%;*}
  line="$line  post=$aid"
  echo "  post.sbatch -> job $aid (analysis + export, starts when every task of $jid has ended)"
  echo "  logs: pipeline/logs/slurm/train_${jid}_<task>.out   per run: pipeline/results/<experiment>/seed<k>.log"
else
  echo "  logs: pipeline/logs/slurm/${JOB%.sbatch}_${jid}.out"
fi
echo "$line" >> pipeline/logs/submissions.log
echo
squeue -u "$USER"
