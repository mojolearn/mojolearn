# Lane w2-gemm2: second Apple FAST GEMM round (gemm, gemm-bf16)

Branch `lane/apple-fast-neural-w2-gemm2`, 2026-10-03, base `675a74a28`.
Code: `gemm/afn_apple_fast2.mojo` (new) plus one comptime branch and two define
lines in `gemm/afn_apple_fast.mojo`. Every changed line is inside the FAST +
Apple + not-CPU-column guard and a `MOJOLEARN_AFN_GEMM2_*` define; IDENTICAL
compiles main's code unchanged (`AFN_GEMM2_ON` is False there, so the branch
and every kernel of the new file are never instantiated).

## Profile of the wave-1 path (read from the code, not measured)

Per product at 4096^3 under `MOJOLEARN_AFN_GEMM_SIMDGROUP` (fp32) or
`_BF16_MMA` (bf16 bits): one launch of `afn_gemm_mma_kernel[2,2,4,4]`, 4096
blocks of 128 threads, 64x64 tile, KB = 32, one threadgroup page of 17,920
bytes (two do not fit 32 KB) so two barriers per window, 128 windows. Each
window stages 64 x 32 A words and 32 x 64 B words; each simdgroup issues 16
MMAs per 8 steps from 4 A + 4 B fragment loads. No host copies, no waits, no
allocations inside the route (the binding's own lease, upload and download
are outside it). Row-major block order: block `bid` is tile `(bid / nbn,
bid % nbn)`.

## Candidates (each its own define, default OFF; all exact f32 accumulate)

| define | mechanism | KB / pages |
|---|---|---|
| `MOJOLEARN_AFN_GEMM2_BIGTILE` | 128x64 tiles, 8 simdgroups (4x2), each a 32x32 corner of 4x4 fragments (16 accumulators); used when the 128x64 grid has >= 2 x `AFN_GEMM_CORES` (160) tiles, else 64x64 | 32 / 1 page (26,112 B) |
| `MOJOLEARN_AFN_GEMM2_DBUF` | KB chosen so TWO pages fit: next window stages into the other page while this one multiplies, one barrier per window | 64x64: 24 / 2; 128x64: 16 / 2 |
| `MOJOLEARN_AFN_GEMM2_DIRECT_B` | B fragments built per lane from device memory (lane cells `(frow, fcol + e)`, the FAST k-NN idiom at neighbors/impl/detail/fast_mma_knn.mojo:204-215); only A is staged, so the A panel deepens | 64x64: 64 / 1; 128x64: 56 / 1; with DBUF 56 / 2 and 24 / 2 |
| `MOJOLEARN_AFN_GEMM2_SWIZZLE` | grouped block order: 8 tile rows (`MOJOLEARN_AFN_GEMM2_SWZ_G`) walked column by column for L2 reuse | unchanged |
| `MOJOLEARN_AFN_GEMM2_ALL` | all four | 128x64: 24 / 2 |

KB is computed at comptime by `afn2_kb` (deepest multiple of 8, cap 32 with a
staged B page and 64 without, pages fit `column_shared_limit(COLUMN_APPLE)`,
whole 4-wide slots per thread).

Dispatch: `_afn_dispatch` (wave-1) hands the product to `afn2_gemm_dispatch`
when any GEMM2 define is on, the epilogue is NONE, the wave-1 tile is the
square one and wave-1 SPLITK would not split; otherwise the wave-1 kernel
runs as before. Any GEMM2 define also turns on the wave-1 SIMDGROUP and
BF16_MMA entry points (`afn_apple_fast.mojo` define table), so the fp32
entries of `gemm_identical.mojo`, `gemm_lowbit.identical_gemm_bf16w_into` and
the binding's bf16-bits call all reach it. The A/B lines therefore compare
against the wave-1 kernel (arm A = SIMDGROUP + BF16_MMA).

Risk notes: DIRECT_B depends on writing a fragment's lane cells `v[0], v[1]`
before the MMA (the k-NN kernel's idiom; the GEMM kernels so far only READ
them); if that misbehaves on a target, the judge shows it on the gemm lanes
first. DIRECT_B re-reads B per simdgroup row (SGM times) through the cache
instead of once into threadgroup memory.

## Compile results

UNCOMPILED (wave-2 order: the peer compiles). Builds owed: FAST linalg with
each of the five defines alone, FAST linalg with none, IDENTICAL linalg, FAST
transformer with `MOJOLEARN_AFN_GEMM2_ALL`. The linalg build gate RUNS
matmuls; the wave-1 lane set `MOJOLEARN_SKIP_BUILD_GATE=1`.

## Additions for the w2-lmgrad lane (gemm/afn_apple_fast.mojo, wave-1 kernel)

No new define: the low-level launchers need only the FAST Apple tier; the
entry points ride the wave-1 EPILOGUE and SPLITK switches. The wave-1 kernel
gained one pointer argument `aux` (second output); `_afn_launch_tile` and
`_afn_strides` keep their names and signatures (`aux` = `c`).

| symbol | signature | gate |
|---|---|---|
| `afn_strides` | `(op, m, n, k) -> (a_si, a_sp, b_sp, b_sj)` | none (host arithmetic) |
| `afn_launch_tile[AT, BT, SPLIT, EPI]` | `(ctx, tile, c, a, b, bias, resid, m, n, k, st, splits, k_split) raises` | FAST Apple (raises otherwise) |
| `afn_launch_tile_aux[AT, BT, SPLIT, EPI]` | `(ctx, tile, c, a, b, bias, resid, aux, m, n, k, st, splits, k_split) raises` | none (callers' guards) |
| `AFN_EPI_RESID` (5) | `C = A.B + resid[i, j]`, no bias; `c == resid` safe | kind |
| `AFN_EPI_SWIGLU_BWD` (6) | `bias` = gate, `resid` = up, `c` = d_gate, `aux` = d_up | kind |
| `afn_gemm_resid_ptr_into` | `(ctx, c, a, b, resid, m, n, k, op) raises -> Bool` | AFN_GEMM_EPILOGUE |
| `afn_gemm_swiglu_bwd_ptr_into` | `(ctx, d_gate, d_up, a, b, gate, up, m, n, k, op) raises -> Bool` | AFN_GEMM_EPILOGUE |
| `afn_gemm_accum_ptr_into` | `(ctx, c, a, b, m, n, k, op, k_split) raises -> Bool`: `C += A.B`, no zero launch, atomics onto the existing C; `k_split <= 0` takes the policy, no split = one atomic split | AFN_GEMM_SPLITK |
| `afn_gemm_fused_into` | now also accepts `epi = AFN_EPI_RESID` | AFN_GEMM_EPILOGUE |

SwiGLU backward formula (f32, `a` = the GEMM's d_gated cell, `g` = gate,
`u` = up): `d = exp(-g) + 1; sg = 1/d; s = g/d; d_up = a s; dsi = a u;
d_gate = dsi (sg (1 + g (1 - sg)))`, the S21 products plus the S20 SiLU VJP of
transformer/checks/transformer_backward.mojo (sigmoid recomputed from `g`),
in the same operation order, without its ftz and pinned-rounding helpers
(FAST).
