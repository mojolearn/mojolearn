# afn-attn: the transformer block forward on Apple FAST

Lane afn-attn, branch lane/apple-fast-neural-attn, 2026-10-03. Board lane
transformer-forward (B 1, L 2048, d_model 384, 6 heads, 6 kv heads, head_dim
64, ffn 1024), binding `_mojolearn_transformer`. Code:
transformer/impl/llama/afn_apple_fast.mojo (kernels and launchers),
transformer/impl/llama/modeling_llama.mojo (`_afn_*` helpers and the four
`comptime if AFN_ATTN_ANY` arms of `llama_decoder_layer_forward_planted`),
bindings/_mojolearn_transformer.mojo (ARENA). Nothing here was run or timed.

## Guard

Every candidate is `GLOBAL_NUMERIC_MODE == NUMERIC_FAST and
has_apple_gpu_accelerator()` plus its `-D MOJOLEARN_AFN_ATTN_<NAME>` (or
`_ALL`), as `comptime` aliases at the top of afn_apple_fast.mojo. The forward
arms are `comptime if AFN_ATTN_ANY`; their else arms are main's lines verbatim,
so IDENTICAL (and FAST without a define) compiles main's code. The helpers
also refuse at runtime (main's launches run) unless the call is the default
options record, forward only, no plant, no trace, no int15 and
`d_model % 4 == 0`.

## Main's forward on Apple, per block call (the session's fresh prefill)

| stage | launches | waits / host |
|---|---|---|
| binding: 30 stage zero fills (`stages.reset`) | 30 fills | none |
| binding: x upload (+ cache fill/upload) | 1-3 copies | 1 synchronize |
| norm1 (`llama_rms_norm_kernel`: one THREAD per row, serial 384 fold; 16 blocks at M 2048) | 1 | none |
| q, k, v projections (`llama_proj`, identical GEMM) | 3 | none |
| RoPE q, RoPE k | 2 | none |
| cache append (`kv_append2_kernel`) + 2 device copies into the cache | 1 + 2 copies | 1 synchronize |
| attention (Apple eager forward: fresh [B, nh, L, S] score stash, ~100 MB at the board shape; scalar-chain kernel; regime scan; corner flag readback) | ~4 | ~3 round trips, 1 allocation |
| o_proj | 1 | none |
| residual1 + norm2 (fused on Apple at m <= 2048) | 1 | none |
| gate, up, SwiGLU, down | 4 | none |
| residual2 add | 1 | none |
| binding: residual2 + cache downloads | 3 copies | 1 synchronize |

About 20 kernel launches plus 30 fills and 5 copies, 5-6 host waits, and ~35
separately allocated live buffers (cache, rope, ~30 stages, x) that every
Metal launch makes resident (~0.25 us per live buffer per launch).
M3 stage timing (Oct 1): attention 4.6 ms, norm1 1.24 ms, rope_and_cache
1.19 ms, o_proj 0.59 ms: the small stages are launch and wait bound.

## Candidates (each its own define, default off)

1. `MOJOLEARN_AFN_ATTN_NORM_SG`: RMSNorm with one simdgroup per row (8 rows
   per 256-thread block, vector-4 loads, shuffle_xor butterfly for the sum of
   squares): 256 blocks instead of 16 serial-row blocks. Also the
   residual1+norm2 and residual2+next-norm kernels. f32, fold order free.
2. `MOJOLEARN_AFN_ATTN_ROPE_CACHE`: q RoPE, k RoPE, the cache append and both
   cache copies in ONE launch, no synchronize (fresh full-causal prefill,
   `kv.s == 0`, `window == 0`; otherwise main's launches).
3. `MOJOLEARN_AFN_ATTN_FLASH`: one launch per (batch, head, 32 query rows):
   online softmax (running max and sum per row), QK^T and PV on the 8x8 f32
   simdgroup matrix unit, K/V tiles in threadgroup memory (27 KB page, fits
   gate), causal key-block skipping, no stash, no scan, no flag readback.
   head_dim 64 only (else main's attention). Writes ctxv, amax, denom.
4. `MOJOLEARN_AFN_ATTN_GQA_TILE`: FLASH with the n_rep heads of one KV group
   stacked on the tile's row axis (GROUP 2 or 4), so one staged K/V tile serves
   the whole group. At n_kv == n_heads (the board) it is FLASH at GROUP 1:
   same kernel, same cost; implies FLASH.
5. `MOJOLEARN_AFN_ATTN_FUSE_PRE`: norm1 folded into ONE q+k+v projection
   launch (row rstd and norm weight applied while staging A, so norm1_out is
   never written) with RoPE and the cache append in its epilogue; norm2
   folded the same way into the gate/up projections. The norm kernel writes
   only the row sums of squares. Fresh full-causal prefill, head_dim 64,
   whole 64x64x32 GEMM tiles (`afn_gemm_ok`); else main's launches.
6. `MOJOLEARN_AFN_ATTN_FUSE_MLP`: one gate+up launch with `silu(g) * u` in
   the epilogue (two weight tiles, two accumulators), the residual adds in
   the epilogues of o_proj (`residual1 = x + ctx . Wo^T`) and down_proj
   (`residual2 = residual1 + gated . Wd^T`).
7. `MOJOLEARN_AFN_ATTN_ARENA` (binding only): the workspace's cache, rope,
   stages and x are sub-buffer views of one device arena
   (core/device_arena.mojo `arena_begin/arena_end/arena_release`; the
   `_zeros`/`_upload` helpers already take views while an arena is active),
   and the session forward drops the synchronize between the uploads and the
   forward. The thirty reset fills stay (they keep stale scratch out of a
   reused workspace; `MOJOLEARN_TRANSFORMER_STAGE_RESET=0` already exists as
   the A/B for dropping them).
8. `MOJOLEARN_AFN_ATTN_ALL`: all of the above together (they compose: the
   helpers check each shape condition and fall back stage by stage).

The projection GEMM (FUSE_PRE, FUSE_MLP) is a 64x64 output tile per block,
K windows of 32, 8 simdgroups each owning 8x8 f32 fragments, f32 accumulate.

## Quality

f32 storage and accumulation everywhere; only the fold order changes
(simdgroup butterfly, MMA fragment order, online softmax rescaling). No
approximate exp/rsqrt, no lower precision, no caps. The judge is
tools/neural_fast_quality.py.

## Build (compile only)

`.afn-scratch/b.sh <tag> fast|identical <defines>` in the worktree (not
committed): `MOJOLEARN_NUMERIC_MODE=<mode> MOJOLEARN_COMPILE_JOBS=1
MOJOLEARN_MOJO_BUILD_FLAGS="<defines>" bash bindings/build_transformer.sh`
through compile_slot.sh. The script exports MOJOLEARN_SKIP_BUILD_GATE=1, so
its Python smoke never runs; only the strings/otool binary checks do.
Results are in the table below.

## Compile table (2026-10-03)

The orchestrator stopped local compiles mid-lane (the M3 manager peer
compiles everything); only one build ran here.

| build | defines | rc | state |
|---|---|---|---|
| FAST | `-D MOJOLEARN_AFN_ATTN_ALL` | 1 | compiled here at 1316a219c..ARENA head; the only errors were the three instantiations (GROUP 1, 2, 4) of `afn_flash_forward_kernel` at afn_apple_fast.mojo:609-610 (`SIMD[...](negmax)` / `(False)` splats need `fill=`). Fixed in the next commit, NOT recompiled. The failure was in the offload (GPU) pass, so kernels after it may not have been fully elaborated. |
| FAST | `ALL` (after the fill= fix) | - | UNCOMPILED (peer compiles) |
| FAST | `NORM_SG` | - | UNCOMPILED (peer compiles) |
| FAST | `ROPE_CACHE` | - | UNCOMPILED (peer compiles) |
| FAST | `FLASH` | - | UNCOMPILED (peer compiles) |
| FAST | `GQA_TILE` | - | UNCOMPILED (peer compiles) |
| FAST | `FUSE_PRE` | - | UNCOMPILED (peer compiles) |
| FAST | `FUSE_MLP` | - | UNCOMPILED (peer compiles) |
| FAST | `ARENA` | - | UNCOMPILED (peer compiles) |
| FAST | none | - | UNCOMPILED (peer compiles) |
| IDENTICAL | none | - | UNCOMPILED (peer compiles) |

IDENTICAL note: every kernel change is behind `comptime if` on the FAST +
Apple aliases. The binding gains one always-present field
(`TransformerWorkspace.arena_id`, -1 and never read off ARENA) and a
`__deinit__` whose body is empty off ARENA; no kernel, launch or byte moves.
