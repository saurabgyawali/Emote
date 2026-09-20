#!/usr/bin/env bash
# Build a clean tarball of Emote for copying to the Jetson (no venvs, caches, sessions).
#   bash pack.sh [--with-model] [--with-sessions] [--with-ct2-src] [--out FILE]
#
# ct2/CTranslate2 (the ~11 GB CTranslate2 source + build tree) is excluded by default:
# it is aarch64 build output that install.sh can re-clone and recompile. ct2/wheels IS
# packed -- a cached aarch64 wheel there turns the 45 min CUDA build into seconds.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(dirname "$HERE")"; NAME="$(basename "$HERE")"
OUT="$PARENT/Emote-$(date +%Y%m%d).tar.gz"
EXCL=(--exclude='*/.venv' --exclude='*/.venv.broken' --exclude='*/.venv_path' --exclude='*/.hf_cache' --exclude='*/.cache'
      --exclude='*/__pycache__' --exclude='*/.pytest_cache' --exclude='*/logs' --exclude='*/results' --exclude='*.pyc')
WITH_MODEL="no"; WITH_SESSIONS="no"; WITH_CT2_SRC="no"
while [ $# -gt 0 ]; do
  case "$1" in
    --with-model) WITH_MODEL="yes"; shift ;;
    --with-sessions) WITH_SESSIONS="yes"; shift ;;
    --with-ct2-src) WITH_CT2_SRC="yes"; shift ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
[ "$WITH_MODEL" = "yes" ] || EXCL+=(--exclude="$NAME/transcriber/models")
[ "$WITH_SESSIONS" = "yes" ] || EXCL+=(--exclude="$NAME/transcriber/sessions")
[ "$WITH_CT2_SRC" = "yes" ] || EXCL+=(--exclude="$NAME/ct2/CTranslate2")
tar -C "$PARENT" -czf "$OUT" "${EXCL[@]}" "$NAME"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
