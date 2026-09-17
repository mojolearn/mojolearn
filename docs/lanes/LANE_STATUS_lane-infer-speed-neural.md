# lane/infer-speed-neural: faster neural inference in IDENTICAL mode, no bit moved

Written 2026-09-17 for a session with no memory. Everything measured here ran
on one rented RunPod RTX 4090 (driver 580.126.20, CUDA column `sm_89`, Mojo
from the repository's pixi lock) and on that pod's AMD EPYC 7532 for the CPU
column. Evidence outside the repository:
`~/mojolearn-evidence/infer-speed-neural/` (pod state, A/B records and logs,
identity JSONs, sweep and gate reports). The small A/B records are committed
under `bench/results/infer_speed_neural_2026-09-17/`.

## What changed, and why no output bit can move

Three changes, each a change of LIFETIME or of host bookkeeping. None respells
an arithmetic seam; every kernel, its launch order and its operands are the
ones the per-call paths already ran.

### 1. `TransformerBlock.decode_session(state)` (DEVIATION 2940)

`bindings/_mojolearn_transformer.mojo` gains a `TransformerDecodeSession`
Python object holding one `DeviceContext`, the nine weights as
`LlamaDeviceWeights`, the caller's cache as `LlamaKVCache`, the rotary table
at `(theta, head_dim, max_tokens)` and the `LlamaDeviceStages` at
`(B, 1, max_tokens, window)`. `step` uploads one token per row, calls
`llama_decoder_layer_forward` at `l = 1` and `pos0 = cached_tokens`, the SAME
certified entry point the per-call `transformer_decode_step` calls, and
downloads the block output, one completion wait per token. `forward` runs `L`
tokens on the resident weights and cache with stages built for that call.

The invariant: the per-call entry builds exactly these structs from exactly
these bytes on every token and destroys them; the session builds them once.
Reusing a stage struct across calls is `training/byte_lm.mojo::ByteTrainer`'s
own pattern (its per-layer `forward` stages serve every training step), and
every stage the block reads it writes first in the same call.

Ownership and refresh, the rule `docs/NEURAL_METAL_DECODE.md` asks for: the
session COPIES the weights and the cache at open. While it is open the
state's `k_cache`/`v_cache` are stale and `TransformerBlock.forward`/`step`
REFUSE that state by name (`_refuse_resident`); `cached_tokens` is kept
current. `sync_state()` and `close()` copy the cache back; `load_state()`
re-uploads the caller's cache (the state half of refresh); the weight half is
`close()` and a new `decode_session`. On the CPU host route
(`TransformerBlockInference`) the binding exports no session and
`decode_session` refuses by name.

### 2. `Mamba1Block.decode_session(state)` (DEVIATION 2941)

The same object on `bindings/_mojolearn_mamba.mojo`: `MambaDeviceWeights`,
`MambaDeviceState` (conv window and `h`), `MambaDeviceStages` at `(B, 1)` and
the resident x buffer; `step` is `mamba_step`, the certified decode entry (the
block at `L = 1`). Same ownership rule, same refusal in `Mamba1Block._call`.

Mamba-2, Mamba-3 and Samba have no session yet (see "Not done").

### 3. Byte LM logits: one wait and a retained scratch (DEVIATION 2942)

`training/byte_lm_logits.mojo::_logits_forward` enqueued five completion waits
between the ids upload, the embedding, each block and the head GEMM, all on one
in-order context; they ordered nothing and are gone, leaving the download's
wait. The call-shaped buffers (ids, the embedding output, the KV cache reset
per block, one stage struct per block, the logits and the head GEMM workspace)
are a `ByteLogitsScratch` that the resident `ByteLMSession` keeps across
`logits` calls of the same `[batch, length]` and rebuilds when the shape
changes; the stateless entry builds one per call and destroys it with its
context, as before. Every buffer is written whole before it is read (the ids by
the upload, `x` by the embedding, the cache from `s = 0`, the stages by the
block, the logits by the head GEMM, whose workspace is scratch the GEMM writes
before reading, as the trainer's own retained workspaces are).

Python: `_byte_lm_host._refuse_ids` settles the common case with two C-speed
scans of the view (`min`, `max`) and keeps its loop only to name the first
offender with the same message. It serves both the CPU class and the GPU
trainer's `logits`.

## Numbers (RTX 4090, IDENTICAL, one process per A/B, arms interleaved)

`tools/bench_neural_decode.py --resident-ab`: a fresh state prefilled to 1024
positions through the per-call `forward`, then 64 decode tokens through the
per-call `step` (arm `percall`) or a `decode_session` (arm `resident`), five
rounds after a warmup with the arm order reversed every round. Every decoded
output and both final state pieces are bytewise equal across arms and rounds
and equal to the fresh full forward over the 1088 positions (the `stepfull`
property), asserted per round. Medians are per token.

| model | d_model | B | per-call ms | resident ms | spread (per-call, resident) | paired ratio |
|---|---|---|---|---|---|---|
| Transformer (16 heads, 4 kv, hd 64, ff 2816) | 1024 | 1 | 9.440 | 0.817 | 1.002, 1.057 | 11.54 |
| Transformer | 1024 | 8 | 12.528 | 0.987 | 1.005, 1.014 | 12.68 |
| Mamba-1 (d_inner 2048) | 1024 | 1 | 31.801 | 0.946 | 1.015, 1.006 | 33.61 |
| Mamba-1 | 1024 | 8 | 26.930 | 1.234 | 1.002, 1.005 | 21.82 |

Records: `bench/results/infer_speed_neural_2026-09-17/*_d1024_b*_p1024_t64.json`.

Byte LM logits, the shipped profile, `~/mojolearn-evidence/infer-speed-neural/
bytelm_ab/probe_{base,cand}.log`, four alternating processes (base, cand, cand,
base), 15 calls each after a warmup, medians:

| call | before ms | after ms | ratio |
|---|---|---|---|
| resident, batch 8, length 32 | 3.05 / 3.04 | 1.63 / 1.66 | 1.85 |
| resident, batch 256, length 32 | 44.94 / 44.90 | 35.40 / 35.89 | 1.26 |
| stateless, batch 8, length 32 | 6.61 / 6.71 | 6.67 / 6.60 | 1.00 |
| stateless, batch 256, length 32 | 48.27 / 48.66 | 47.76 / 47.98 | 1.01 |

The stateless call is unchanged because its time is the per-call context and
the weight uploads through the trainer's own helpers, which this lane does not
touch. At batch 256 the remaining time is `attn.core`, about 14 ms per block
under `MOJOLEARN_TRANSFORMER_TIMING=1`, a kernel matter outside this lane.

## Identity

IDENTITY_PENDING

## Sabotage controls

SABOTAGE_PENDING

## Commands

Pod, from a checkout at `/root/mojolearn` (the runner is `tools/trees_leg.sh`
with `TREES_LEG_STATE=$HOME/mojolearn-evidence/infer-speed-neural/pod`):

```
INFER_PHASE=setup    ROOT=/root/mojolearn sh tools/infer_speed_neural_body.sh
INFER_PHASE=identity ROOT=<tree> LABEL=<before|after> sh tools/infer_speed_neural_body.sh
INFER_PHASE=bytelm   ROOT=<tree> LABEL=<before|after> sh tools/infer_speed_neural_body.sh
INFER_PHASE=diff     ROOT=<tree> sh tools/infer_speed_neural_body.sh
INFER_PHASE=sabotage ROOT=<tree> sh tools/infer_speed_neural_body.sh
pixi run python tools/bench_neural_decode.py --resident-ab --kind transformer --dm 1024 --batch 8 --prefill 1024 --tokens 64 --rounds 5 --out ab.json
pixi run python tools/bench_neural_decode.py --resident-ab --kind mamba1 --dm 1024 --batch 8 --prefill 1024 --tokens 64 --rounds 5 --out ab.json
```

## Not done, and why

- Mamba-2, Mamba-3 and Samba resident sessions. Mamba-3's state is ten
  pieces with a chunk buffer and a pending flag, Samba is a stack of blocks;
  both fit the same object shape and are the next step, not this one.
- The stateless byte LM call keeps its per-call context (DEVIATION 2520's
  design) and its weight uploads through `training/byte_lm.mojo`'s helpers,
  which are training code this lane does not edit.
- The per-call `TransformerBlock.step` and `Mamba1Block.step` are unchanged.
  A cache upload of the used prefix only would move no arithmetic but it
  changes which bytes of the caller's buffer are written back, which the
  identity records hash, so it was not built.
- Apple and AMD columns for the session paths and for the byte LM changes:
  owed at the next release record, as every column is.

## False claims found in owned docs

None edited away. `docs/NEURAL_METAL_DECODE.md` says a resident decode API
"is a separate improvement"; it is now this lane's, recorded in the CUDA
section added there.
