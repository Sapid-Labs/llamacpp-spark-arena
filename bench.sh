#!/usr/bin/env bash
# Thin wrapper over harness/arena.py so the contributor loop is two commands.
#
#   ./bench.sh --target laguna-xs-2-1-q4-k-m --baseline   # once, before editing
#   ./bench.sh --target laguna-xs-2-1-q4-k-m              # paired, after editing
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE=bench
ARGS=()
for a in "$@"; do
  case "$a" in
    --baseline) MODE=baseline ;;
    *) ARGS+=("$a") ;;
  esac
done

exec python3 harness/arena.py "$MODE" "${ARGS[@]}"
