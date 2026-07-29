#!/usr/bin/env bash
# Package a passing run as a submission.
#
# Produces submissions/<handle>-<slug>/ containing the three things a referee
# needs and nothing else: your diff, the record that says it passed, and a note
# saying what you did. Open a PR with that directory.
#
#   ./submit.sh --handle yourname --slug wider-q4k-tile --note note.md
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HANDLE=""; SLUG=""; NOTE=""; MODEL=""; TARGET=""
while [ $# -gt 0 ]; do
  case "$1" in
    --handle) HANDLE="$2"; shift 2;;
    --slug)   SLUG="$2";   shift 2;;
    --note)   NOTE="$2";   shift 2;;
    --model)  MODEL="$2";  shift 2;;
    --target) TARGET="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 1;;
  esac
done
[ -n "$HANDLE" ] && [ -n "$SLUG" ] || { echo "usage: ./submit.sh --handle <gh-handle> --slug <short-name> [--note note.md] [--model \"Claude Opus 5\"] [--target <t>]" >&2; exit 1; }

PIN=$(python3 -c "import json;print(json.load(open('benchmark.json'))['vendor']['commit'])")
OUT="submissions/$HANDLE-$SLUG"
mkdir -p "$OUT"

# The diff IS the submission. Taken against the pin, so it is exactly "what
# changed vs the engine everyone else is measuring".
git -C vendor/llama.cpp diff "$PIN" > "$OUT/changes.patch"
if [ ! -s "$OUT/changes.patch" ]; then
  echo "!! vendor/llama.cpp is identical to the pin — there is nothing to submit." >&2
  exit 1
fi

# Newest passing record, unless a target was named.
REC=$(ls -t results/${TARGET:+$TARGET-}*.json 2>/dev/null | head -1 || true)
[ -n "$REC" ] || { echo "!! no results/*.json — run ./bench.sh first" >&2; exit 1; }
python3 - "$REC" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
if not r.get("promotable"):
    failed = [k for k, v in r.get("gates", {}).items() if not v]
    raise SystemExit(f"!! {sys.argv[1]} did not pass its gates ({', '.join(failed) or 'score <= 1.0'}). "
                     "Submitting it wastes a referee's node and your own credibility.")
print(f"    record: {sys.argv[1]}  decode x{r['decodeSpeedup']:.4f}  score {r['score']:.4f}")
PY
cp "$REC" "$OUT/record.json"

if [ -n "$NOTE" ] && [ -f "$NOTE" ]; then cp "$NOTE" "$OUT/note.md"; else
  cat > "$OUT/note.md" <<NOTE
# $SLUG

**What changed:** (one or two sentences — "wider Q4_K tile, 2 waves per SM" is a
note; "optimized the matmul" is not)

**Why it is faster:** 

**Targets claimed:** ${TARGET:-<which targets>}

**Scope:** engine-general | model-specific

**Model used:** ${MODEL:-<none / e.g. Claude Opus 5>}
NOTE
fi

echo "==> $OUT"
ls -la "$OUT" | tail -3
echo
echo "    Open a PR containing this directory."
echo "    A referee re-runs it and runs gate 3 (held-out prompts) on their own node."
