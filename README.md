# llama.cpp Spark arena

Crowd-optimized llama.cpp kernels for the NVIDIA DGX Spark (GB10, **sm_121**).

Make the engine faster on this part without changing what it outputs. A
submission is a diff against a pinned llama.cpp, measured as a paired ratio on
a real Spark, and gated on producing byte-identical tokens.

## Why this exists

Nothing is built for sm_121. `torch.cuda.get_arch_list()` on a GB10 returns
`sm_80, sm_90, sm_100, sm_110, sm_120` — the chip is 12.1. The documented
consequences are not subtle: `FLASHINFER_CUTLASS` for NVFP4 MoE was removed on
sm_121 *"and it costs 13%"*, `HUMMING` cannot initialize, and `VLLM_CUTLASS` /
`FLASHINFER_TRTLLM` fail outright with *"kernel does not support current
device"*. Every one of those is unclaimed performance sitting on hardware
people already own.

The corpus of recipes this arena optimizes against lives at
[howtospark.com](https://howtospark.com) — 55 measured deployment recipes on
this hardware. Wins here flow back there.

## The loop

```bash
git clone https://github.com/Sapid-Labs/llamacpp-spark-arena && cd $_
./setup.sh                                            # fetch + build the pinned engine (~15 min)
./bench.sh --target laguna-xs-2-1-q4-k-m --baseline   # record your node's baseline

# edit vendor/llama.cpp/ggml/src/ggml-cuda/*.cu, or the model graph

./bench.sh --target laguna-xs-2-1-q4-k-m              # paired ratio + gates
```

The last command rebuilds only if you changed something, waits for the thermal
gate, then runs **baseline → candidate → baseline → candidate**, alternating,
in one session on one node. It prints a score and writes a machine-readable
record to `results/`.

## Scoring

```
score = decode_speedup^0.75 * prefill_speedup^0.25      # both floor at 0.95, hard
```

Speedups are **paired ratios**, never absolute numbers. This is not pedantry:
a published Laguna baseline on this fleet failed to reproduce a day later
(24.05 → 20.09 tok/s), and Spark-1 has been measured at 84 °C under sustained
load while Spark-2 idled at 43 °C. The ratio cancels the room; a leaderboard of
absolute tok/s would mostly rank thermostats.

**Gates, cheapest first.** A submission fails at the first one it trips:

| # | Gate | Runs where |
|---|------|-----------|
| 1 | **Editable surface** — the diff touches only `editablePaths`; the weights are unchanged | your node |
| 2 | **Token identity** — output is byte-identical to the target's goldens, batch 1, warm | your node |
| 3 | **Held-out token identity** — same, on prompts you have never seen | referee's node |
| 4 | **Speedup floors** — ≥ 0.95 on both axes, so you cannot trade prefill for decode | your node |
| 5 | **Beat the incumbent** | referee's node |

Gate 3 is the one that matters most and the one you cannot run: it catches a
kernel that reassociated its way into a different argmax on everything except
the prompts it was tuned against.

## Why a token diff is enough

At **batch size 1, warm**, greedy output on this engine is deterministic —
measured 2026-07-28 on a Spark, 24/24 runs byte-identical. So "did this change
alter the output?" is answerable for free, with no eval.

The boundary is hard. Sweeping only concurrency on vLLM 0.26.0 on the same
hardware: `n=1 → 1 distinct output, n=2 → 2, n=6 → 5`. **Any batching breaks
token identity**, which is why the ranked track is serial and every target
serves `--parallel 1`.

And the first request after a server start differs on both engines, always. The
harness discards it. If you build your own tooling around this, discard it too,
or you will chase a phantom mismatch for an afternoon.

## Optimization vs. recipe

**Optimizations preserve output and get the cheap gate. Changes that alter
output are a recipe, not a submission.**

Different precision, different quantization, different sampling — those are
legitimate and often better, but they need eval evidence, and they belong on
[howtospark.com](https://howtospark.com) as a new recipe rather than here as a
score. Holding that line is what keeps this arena's verification cheap enough
to actually happen.

## Targets

A submission names the targets it claims to improve. **Engine-general changes
score across all targets; model-specific ones score on one.**

| Target | Shape | Exercises |
|---|---|---|
| `laguna-xs-2-1-q4-k-m` | 33B MoE, ~3B active, Q4_K_M | MoE routing, Q4_K matmul, flash attention |

Which targets move is a free blast-radius check on the claim:

| Kernel | Should reach |
|---|---|
| `rope.cu`, `norm.cu`, `softmax.cu`, `cpy.cu` | ~every target |
| `mmq.cu`, `mmvq.cu` | quantized targets, per quant type (a Q4_K change won't move Q8_0) |
| `fattn*.cu` | only targets served `-fa on` |
| `mmid.cu`, `topk-moe.cu`, `add-id.cu` | only MoE models |

A claim of "faster Q4_K MoE matmul" where a dense Q8_0 target also moves is
incoherent and does not promote.

## The pin

`vendor/llama.cpp` is fetched, never committed, at
[`Sapid-Labs/llama.cpp`](https://github.com/Sapid-Labs/llama.cpp) `7ad9bd2`
(branch `laguna-support`) — which is upstream `ggml-org/llama.cpp`
`ee445f93` plus 5 commits carrying the Laguna architecture port (upstreamed as
[ggml-org/llama.cpp#25595](https://github.com/ggml-org/llama.cpp/pull/25595),
unmerged at pin time) and its DFlash draft support.

Full history is cloned deliberately. A shallow clone is exactly how *"what
changed vs stock"* — the one question an arena has to answer — becomes
unanswerable.

Everything above is machine-readable in [`benchmark.json`](benchmark.json),
which is the contract. Scores are only comparable within one
`contractVersion`.

## Submitting

Not open yet — see `docs/` as it fills in. When it opens, submission is a PR
carrying your `vendor/` diff plus the `results/` record, and verification runs
on a referee's node, **not** on CI. Self-hosted GitHub runners must not execute
untrusted fork PRs on hardware you own, and there is no version of that caveat
that gets safer with volume.

## Status

Early. Chunks 01–03 of the build plan: engine pinned, harness written, first
target defined. The baseline record for `laguna-xs-2-1-q4-k-m` lands next.
