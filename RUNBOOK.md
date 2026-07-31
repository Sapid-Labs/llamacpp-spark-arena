# Runbook — running the arena

Two loops. The **solo loop** is you making a change and promoting it. The
**referee loop** is you verifying someone else's. They share every step except
who runs which half.

Everything below runs on a Spark. The repo lives at `~/Dev/llamacpp-spark-arena`
on Spark-1.

---

## Once per node

```bash
./setup.sh          # clones the pinned engine, builds the base tree (~15 min)
```

Builds `vendor/llama.cpp-base` (pristine, never edit) and `vendor/llama.cpp`
(yours). Both at the pin. Re-running is safe — it leaves a dirty candidate tree
alone.

## Once per target

```bash
./bench.sh --target laguna-xs-2-1-q4-k-m --baseline
```

Records `goldens.json` (the hashes gate 2 compares against) and `baseline.json`.
Runs two arms and **fails if they disagree** — a baseline that cannot reproduce
itself has no business defining a gate.

Already done for `laguna-xs-2-1-q4-k-m`: 90.09 tok/s decode, 1384.6 prefill.

---

## The solo loop

```bash
# 1. change something
vim vendor/llama.cpp/ggml/src/ggml-cuda/mmq.cu

# 2. check the surface before spending 20 minutes measuring — instant
python3 harness/arena.py gate --target laguna-xs-2-1-q4-k-m

# 3. measure: rebuild if needed, thermal gate, paired arms, gates 1/2/4
./bench.sh --target laguna-xs-2-1-q4-k-m

# 4. gate 3 — prompts generated fresh from a random seed, arms timed.
#    --claimed-speedup is the decodeSpeedup step 3 printed; without it the
#    speed half is skipped and only identity is checked.
python3 harness/arena.py heldout --target laguna-xs-2-1-q4-k-m \
  --claimed-speedup <decodeSpeedup from step 3> --referee joe

# 5. promote (needs BOTH records)
python3 harness/arena.py promote \
  --target laguna-xs-2-1-q4-k-m \
  --record results/laguna-xs-2-1-q4-k-m-<stamp>.json \
  --held-out-record results/laguna-xs-2-1-q4-k-m/heldout-<stamp>.json \
  --author <your-gh-handle> --model "Claude Opus 5" \
  --note "mmq: wider Q4_K tile, 2 waves per SM" \
  --scope model-specific --referee joe

# 6. see the frontier
python3 harness/arena.py leaderboard --target laguna-xs-2-1-q4-k-m
```

Step 3 exits non-zero and prints `REJECTED` unless it passes. Step 5 refuses a
failing gate-3 record, a record from a *different diff*, a held-out speedup that
did not reproduce, a record measured on **another node**, or none at all. `--force`
does not override a failure — it only waives the missing-record and foreign-node
checks, and it says so in the output.

## The referee loop

A submission arrives as a PR containing `submissions/<handle>-<slug>/` with
`changes.patch`, `record.json` and `note.md`.

```bash
# 1. apply their diff to YOUR candidate tree
git -C vendor/llama.cpp checkout . && git -C vendor/llama.cpp clean -fd
git -C vendor/llama.cpp apply submissions/<handle>-<slug>/changes.patch

# 2. surface check first — cheapest rejection
python3 harness/arena.py gate --target <target>

# 3. re-measure on YOUR node. Their number is predictive, not authoritative,
#    and `promote` will refuse their record — it publishes yours.
./bench.sh --target <target>

# 4. gate 3 — this is the half they could not run. Feed it THEIR claim so the
#    held-out arms have to reproduce it.
python3 harness/arena.py heldout --target <target> \
  --claimed-speedup <decodeSpeedup from their record.json> --referee joe

# 5. promote if it holds, then publish (below)
```

Their local number should land close to yours: the paired ratio cancels the
host, which is the whole reason submissions are ratios rather than tok/s. A
large gap is itself a finding — say so in the PR.

Two guardrails make that a gate rather than a habit. `promote` compares the bench
record's `node` against this machine and refuses a record measured elsewhere, so
the ratio on the frontier is always one you measured. And gate 3 now fails when
the claimed decode gain does not reproduce on the held-out prompts, to at least
50% of the claimed excess over 1.0 — the tolerance lives in `benchmark.json`.
Pass `--claimed-speedup` or that half is skipped, and both `heldout` and
`promote` warn you that it was.

---

## Publishing to the site

From `~/Dev/sapid/howtospark` on the Mac:

```bash
npm run arena:pull       # copies leaderboard.json from the arena checkouts
git diff data/arena/     # read it — this is the review step
npm run arena:sync       # ingests ONLY the arena tables
```

`arena:sync` deliberately does not touch the other seed-backed tables. A
promotion should not rewrite 55 recipes to get published.

The result appears in three places automatically: the target's frontier chart on
`/recipes/<slug>#arena`, a row on `/arena`, and a **revision-history entry** on
the recipe — arena wins become changelog entries without anyone editing a recipe
file.

Retraction works the same way: remove the promotion from `leaderboard.json`,
pull, sync. The row and its revision entry both disappear.

---

## Things that will cost you a run

- **Never compare against yesterday's number.** A baseline on this fleet drifted
  24.05 → 20.09 tok/s overnight. The harness pairs the arms for you; do not
  work around it.
- **Batch 1 only.** Any batching breaks token identity here (`n=2` already
  splits). That is why the ranked track is serial.
- **The thermal gate costs ~1 min per arm** and is not optional: an arm heats
  this part 46 → 56 °C, repeatably.
- **Do not touch `tools/server`.** It is what is being measured. Gate 1 rejects
  it, and gate 1 also rejects anything the harness itself leaves in the tree —
  that bug has already been found and fixed once (build stamps now live in
  `results/_build/`).
- **`pkill -f` over ssh kills your own session** if the pattern appears in your
  command line. Guard *every* pattern (`[l]lama-server`), not just one.

## When a measurement looks too good

Two false passes have already happened in this repo's own history, so the habit
is worth keeping: if a result is surprising, check the harness can still detect
a difference at all. Perturb something and confirm the number moves. An
equality result is only worth as much as the test's ability to report
inequality.
