# Contributing

Make llama.cpp faster on a DGX Spark without changing what it outputs.

## What counts

An **optimization**: CUDA kernels or the model graph, producing byte-identical
tokens and a better paired ratio. That is what this repo ranks.

A change that alters output — different precision, quantization, sampling — is a
**recipe**, not a submission. Those are legitimate and often better, but they
need eval evidence, and they belong on
[howtospark.com](https://howtospark.com) as a new recipe rather than here as a
score. Holding that line is what keeps verification here cheap enough to happen
at all.

## The loop

```bash
./setup.sh                                            # ~15 min, builds the pinned engine
./bench.sh --target <target> --baseline               # your node's baseline

# edit vendor/llama.cpp/ggml/src/ggml-cuda/*.cu, or the model graph

./bench.sh --target <target>                          # paired ratio + gates 1, 2, 4
```

`bench.sh` writes a machine-readable record to `results/`. If it prints a score
and `promotable`, you have something worth submitting.

Run `python3 harness/arena.py gate --target <target>` any time to check your
diff stays inside `editablePaths` — it is instant and catches the most common
rejection before you spend 20 minutes measuring.

## Submitting

Open a PR containing:

1. Your `vendor/llama.cpp` diff, as a patch file under `submissions/`
   (`git -C vendor/llama.cpp diff <pin> > submissions/<handle>-<slug>.patch`).
2. The `results/*.json` record from your run.
3. A short note: what you changed and why it is faster. "Wider tile, 2 waves per
   SM" is a note. "Optimized the matmul" is not.

Say which **targets** you claim. Engine-general changes score across all of them;
model-specific ones score on one. If you claim a Q4_K MoE matmul win and a dense
target also moves, the claim is incoherent and will not promote — the target
matrix is a free blast-radius check on your own explanation.

## What happens next

A referee re-runs your submission on their own node and runs **gate 3**:
token identity on prompts generated fresh from a random seed, which neither you
nor they could know in advance. Promotion requires that record.

Verification is **not** run in CI. Self-hosted runners must not execute
untrusted fork code on hardware someone owns, and there is no version of that
caveat that gets safer with volume.

## Things that will save you a run

- **Discard the warmup.** The first request after a server start differs, always.
- **Batch 1 only.** Any batching breaks token identity on this hardware
  (`n=2` already splits), which is why the ranked track is serial.
- **Never compare against yesterday's number.** A baseline on this fleet has
  drifted 24.05 → 20.09 tok/s overnight. The harness pairs the arms for you;
  don't work around it.
- **Don't touch `tools/server`.** It is the thing being measured. Gate 1 rejects
  it.
