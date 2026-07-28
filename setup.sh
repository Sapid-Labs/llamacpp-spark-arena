#!/usr/bin/env bash
# Prepare the arena on a DGX Spark: fetch the pinned engine, lay down a
# pristine base tree and an editable candidate tree, build the base.
#
# Run this on the node. It is idempotent.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# DGX OS puts nvcc in /usr/local/cuda/bin, which a non-interactive shell (i.e.
# every `ssh spark ./setup.sh`) does not have on PATH. Without this, cmake dies
# with "No CMAKE_CUDA_COMPILER could be found" on a box that has a perfectly
# good CUDA 13 toolkit.
if ! command -v nvcc >/dev/null && [ -x /usr/local/cuda/bin/nvcc ]; then
  export PATH="/usr/local/cuda/bin:$PATH"
fi

jq_get() { python3 -c "import json,sys;d=json.load(open('benchmark.json'));print(eval(sys.argv[1],{'d':d}))" "$1"; }

REPO=$(jq_get "d['vendor']['repo']")
REF=$(jq_get "d['vendor']['ref']")
PIN=$(jq_get "d['vendor']['commit']")
UPSTREAM=$(jq_get "d['vendor']['upstream']['repo']")
UPSTREAM_PIN=$(jq_get "d['vendor']['upstream']['commit']")
CAND="$ROOT/vendor/llama.cpp"
BASE="$ROOT/vendor/llama.cpp-base"

echo "==> arena setup"
echo "    engine  : $REPO @ $REF ($PIN)"
echo "    upstream: $UPSTREAM @ ${UPSTREAM_PIN:0:12}"

# --- sanity: this only measures anything on the right hardware ---------------
if ! command -v nvidia-smi >/dev/null; then
  echo "!!  no nvidia-smi. This arena measures GB10 (sm_121); run it on a Spark." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,temperature.gpu --format=csv,noheader | sed 's/^/    gpu     : /'

# --- candidate tree (yours, editable) ---------------------------------------
mkdir -p vendor
if [ ! -d "$CAND/.git" ]; then
  echo "==> cloning engine (full history -- the pin must be resolvable, and a"
  echo "    shallow clone is exactly how 'what changed vs stock' becomes unanswerable)"
  git clone "$REPO" "$CAND"
  git -C "$CAND" remote add upstream "$UPSTREAM" 2>/dev/null || true
fi
git -C "$CAND" fetch --all --tags --quiet
if ! git -C "$CAND" cat-file -e "${PIN}^{commit}" 2>/dev/null; then
  echo "!!  pinned commit $PIN not found in $REPO" >&2
  exit 1
fi

if ! git -C "$CAND" diff --quiet || [ -n "$(git -C "$CAND" status --porcelain)" ]; then
  echo "==> candidate tree has local changes -- leaving it exactly as it is"
else
  git -C "$CAND" checkout --quiet -B arena-work "$PIN"
  echo "==> candidate tree at $PIN (branch arena-work)"
fi

# --- base tree (pristine, never edit) ---------------------------------------
if [ ! -d "$BASE" ]; then
  echo "==> adding pristine base worktree at $PIN"
  git -C "$CAND" worktree add --detach "$BASE" "$PIN"
fi
BASE_HEAD=$(git -C "$BASE" rev-parse HEAD)
if [ "${BASE_HEAD:0:${#PIN}}" != "$PIN" ]; then
  echo "!!  base tree is at $BASE_HEAD, expected $PIN. Re-pin or delete vendor/llama.cpp-base." >&2
  exit 1
fi

# --- targets: check the weights are here before a 15-minute build ------------
echo "==> checking target weights"
python3 - <<'PY'
import json, os, pathlib
for t in sorted(pathlib.Path("targets").glob("*/target.json")):
    d = json.loads(t.read_text())
    integ = d.get("modelIntegrity")
    if not integ:
        print(f"    {t.parent.name}: no modelIntegrity declared")
        continue
    p = pathlib.Path(os.path.expanduser(integ["path"]))
    if not p.exists():
        print(f"    {t.parent.name}: MISSING {p}")
        print(f"      -> {d.get('weightsHint','see targets/%s/README.md' % t.parent.name)}")
    else:
        size = p.stat().st_size
        flag = "ok" if size == integ["bytes"] else f"SIZE MISMATCH ({size} != {integ['bytes']})"
        print(f"    {t.parent.name}: {flag}")
PY

# --- build the base once ----------------------------------------------------
echo "==> building base tree (~15 min cold)"
FLAGS=$(python3 -c "import json;print(' '.join(json.load(open('benchmark.json'))['vendor']['buildFlags']))")
cmake -B "$BASE/build" -S "$BASE" $FLAGS >/dev/null
cmake --build "$BASE/build" --config Release -j
"$BASE/build/bin/llama-server" --version 2>&1 | head -2 | sed 's/^/    /'

cat <<EOF

==> ready.

    edit    : vendor/llama.cpp/ggml/src/ggml-cuda/*.cu   (or the model graph)
    baseline: ./bench.sh --target <target> --baseline
    measure : ./bench.sh --target <target>

    Editable surface is enforced from benchmark.json. Check it any time with:
      python3 harness/arena.py gate --target <target>
EOF
