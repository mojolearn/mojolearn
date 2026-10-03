# Lane afn-gemm: the Apple FAST GEMM (gemm, gemm-bf16, gemm-int8)

Branch `lane/apple-fast-neural-gemm`, 2026-10-03. Binding `linalg`. Code in
`gemm/afn_apple_fast.mojo` (new) plus hooks in `gemm/checks/gemm_identical.mojo`,
`gemm/checks/gemm_lowbit.mojo`, `bindings/_mojolearn_linalg.mojo`, and the
Python face `python/mojolearn/linalg_fast.py`. Every changed line is behind
`AFN_GEMM_APPLE` (FAST tier, Apple GPU build) and one `MOJOLEARN_AFN_GEMM_*`
define; IDENTICAL compiles main's code unchanged.

## What the FAST Apple path does today (read from the code, not measured)

### fp32, `mojolearn.linalg.matmul` (board lane `gemm`, 4096 x 4096 x 4096)

`gemm_binding` -> `identical_gemm_host` -> `identical_gemm`:

| step | today |
|---|---|
| buffers | pooled lease (A, B, C; `gemm/host_transport.mojo`), staged uploads of A and B |
| GEMM | `_fast_vendor_gemm`: MAX's `matmul[target="gpu"]` (its tile, its launches); with `MOJOLEARN_APPLE_FAST_GEMM_PINNED=1` the pinned `PLAN_TUNED_128_8X8` scalar-FMA kernel instead (one launch, a workspace alloc) |
| waits | `identical_gemm` synchronizes; the staged download waits; the binding synchronizes again |

The in-tree neural callers (`transformer/impl/llama/modeling_llama.mojo` through
`GemmWorkspace.run`, `mamba/impl/modules/mamba2.mojo` and `mamba3.mojo` through
`identical_gemm`, `transformer/checks/transformer_backward.mojo`) reach the same
two routes through `identical_gemm_into` / `identical_gemm`; callers that pass
`allow_vendor=False` (mamba's tolerance contract, the bf16 widen plan) get the
scalar tuned kernel on Apple today, never a matrix unit.

`PLAN_APPLE_MMA` (the IDENTICAL simdgroup kernel) is never picked under FAST:
`choose_gemm_plan`'s Apple branches are `GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL`
only, and `APPLE_MMA_FAST` serves only callers that NAME the plan (x_cnn).

### bf16, `mojolearn.linalg.matmul_bf16` (board lane `gemm-bf16`, both operands bf16 bits)

`gemm_bf16_binding`: upload B bits (u16), upload A bits, lease a float32 image of
A, `bf16_widen` launch (m k words), `_lowbit_run` -> `identical_gemm_bf16w_into`
-> (m n > 16384) the widen plan: `LowbitWorkspace.ensure` (alloc, and a wait
when it grows), `bf16_widen` launch for B (n k words), `identical_gemm_into[False]`
-> vendor route closed -> `PLAN_TUNED_128_8X8` scalar kernel; wait; download; wait.
Per call: 2 widen launches, 1 scalar GEMM launch, a float32 image of both
operands (2 x 64 MB at the board shape), 2 to 3 waits.

### int8, `mojolearn.linalg.matmul_int8` (board lane `gemm-int8`, OP_NT)

`gemm_int8_binding`: 4 fresh device buffers + copies, `identical_gemm_int8_into`
-> Apple has no int8 matrix row (`lib_int8_matrix_unit_for` False) -> the flat
one-thread-per-cell kernel, sliced at `LOWBIT_APPLE_SLICE_MAC = 2^30` MACs per
launch with a `ctx.synchronize()` between slices: at 4096^3 (2^36 MACs) that is
64 launches and 63 waits. `gemm/checks/gemm_int8_apple_chunk.mojo` (lane
lowbit-units) already holds an exact matrix-unit kernel (codes staged as float32,
chunks of 1024 steps so every partial sum is below 2^24 and exact, Int32 between
chunks, the flat kernel's epilogue) that nothing dispatches to.

## The candidates (each its own define, default off)

| define | mechanism | touches |
|---|---|---|
| `MOJOLEARN_AFN_GEMM_SIMDGROUP` | fp32 on `afn_gemm_mma_kernel` (64x64 tile, 2x2 simdgroups of 4x4 8x8 fragments, 32-step windows, register prefetch of the next window, two staged pages where 32 KB allows, no leaf fold, no admission reductions, no flush, no rtf seam) ahead of the vendor route, from `identical_gemm` and `identical_gemm_into` on both `allow_vendor` arms | gemm; every in-tree fp32 GEMM caller (transformer, mamba, LM, MLP); the bf16 widen plan |
| `MOJOLEARN_AFN_GEMM_SPLITK` | the same kernel; outputs with fewer than `2 x 80` tiles split `k` over `grid.y` (whole windows, >= 256 steps per split, <= 16 splits) and add into a zeroed C with global f32 atomics: one 4-wide zero launch + one GEMM launch, no workspace, no fold launch | gemm (no effect at 4096^3: 4096 tiles); the LM's weight gradients and small projections |
| `MOJOLEARN_AFN_GEMM_TILESHAPE` | host tile selection from (m, n), no readback: 128x32 for n <= 32, 32x128 for m <= 32, 32x32 when the 64x64 grid has fewer than 160 tiles, else 64x64 | gemm (no effect at 4096^3); skinny and under-filled shapes |
| `MOJOLEARN_AFN_GEMM_BF16_MMA` | both bf16 entry points run the kernel from the bits (`AT`/`BT` = uint16, widened `bits << 16` on the way into threadgroup memory, exact): no widen launches, no float32 images, half the global bytes | gemm-bf16; `identical_gemm_bf16w_into` callers (transformer bf16w) |
| `MOJOLEARN_AFN_GEMM_INT8_MMA` | `identical_gemm_int8_into` dispatches to `identical_gemm_int8_apple_chunk_into`: one launch on the matrix unit, zero intermediate waits, the same Int32 per cell | gemm-int8 |
| `MOJOLEARN_AFN_GEMM_EPILOGUE` | `afn_gemm_fused_into` (bias, bias + residual, bias + SiLU, bias + GELU with exact erf, in the store) and the binding's `gemm_fused` + `linalg_fast.matmul_fused` | nothing on the board; an API for the attention, mamba and MLP lanes |
| `MOJOLEARN_AFN_GEMM_ALL` | every define above | all three lanes |

Composition: SPLITK and TILESHAPE are policies of the SIMDGROUP kernel, so each
of the three fp32 defines alone turns the kernel on; A/B of SPLITK or TILESHAPE
against SIMDGROUP isolates the policy. Under ALL the small tile is chosen first
and a split is added only if the grid is still under 160 blocks.

Scheduling knobs (defines, not env): `MOJOLEARN_AFN_GEMM_KB` (window, 32; 16
gives two pages and one barrier per window), `MOJOLEARN_AFN_GEMM_CORES` (80).

## Entry points and which define each gains

| entry point | define |
|---|---|
| `gemm_identical.identical_gemm`, `identical_gemm_into` (any `allow_vendor`), `GemmWorkspace.run` | SIMDGROUP / SPLITK / TILESHAPE |
| `gemm_lowbit.identical_gemm_bf16w_into`, `identical_gemm_bf16w` | BF16_MMA (float32 A, bf16 B); SIMDGROUP through the widen plan |
| `bindings gemm_bf16` with `a_bf16=1` | BF16_MMA (both from bits) |
| `gemm_lowbit.identical_gemm_int8_into`, `identical_gemm_int8_from_f32` | INT8_MMA |
| `afn_apple_fast.afn_gemm_fused_into`, binding `gemm_fused`, `linalg_fast.matmul_fused` | EPILOGUE |

`afn_gemm_fused_into(ctx, c, a, b, bias, resid, m, n, k, op, epi)`: `epi` in
`AFN_EPI_NONE, AFN_EPI_BIAS, AFN_EPI_BIAS_RESID, AFN_EPI_BIAS_SILU,
AFN_EPI_BIAS_GELU`; `bias` holds n floats, `resid` m n floats (read by
BIAS_RESID only); asynchronous, every buffer the caller's, returns False when
it declines (the caller runs its own epilogue).

## Quality

fp32 products are exact on the matrix unit and the accumulate is f32 (the
Apple probe `gemm/checks/apple_simdgroup_probe.mojo` showed the unit's chain
equals the ascending FMA chain on the M4: no tf32-like rounding). bf16 x bf16
products are exact in f32. The int8 route is exact by construction. Only the
fold order moves against IDENTICAL (f32 reassociation noise); SPLITK adds a
nondeterministic order across splits. No approximation anywhere.

## Build notes

* `bindings/build_linalg.sh`'s FAST smoke gate imports the package and RUNS
  matmuls on the GPU; the lane set `MOJOLEARN_SKIP_BUILD_GATE=1` for every
  build (the AIR-blob floor and the minos check still run). The manager's
  builds should let it run.
* The board's `OursGEMM` calls `mojolearn.linalg.matmul(a, b)` with its default
  `identical=True`, which a FAST linalg binding REFUSES (`require_identical`).
  The FAST A/B worker must call `matmul(a, b, identical=False)`, or select the
  tier it loaded; this lane did not change that default (a public contract).
* The `gemm` AIR-blob floor (>= 8 `gemm`-prefixed kernels) still counts the
  IDENTICAL plans, which stay compiled under FAST.

## Compile results (2026-10-03, linalg binding, compile only, nothing run)

| build | defines | rc |
|---|---|---|
| FAST | `MOJOLEARN_AFN_GEMM_ALL` | 0 |
| FAST | `MOJOLEARN_AFN_GEMM_SIMDGROUP` | 0 |
| FAST | `MOJOLEARN_AFN_GEMM_SPLITK` | 0 |
| FAST | `MOJOLEARN_AFN_GEMM_TILESHAPE` | 0 |
| FAST | `MOJOLEARN_AFN_GEMM_BF16_MMA` | 0 |
| FAST | `MOJOLEARN_AFN_GEMM_INT8_MMA` | 0 |
| FAST | `MOJOLEARN_AFN_GEMM_EPILOGUE` | 0 |
| FAST | none | 0 |
| IDENTICAL | none | 0 |

The first ALL build failed on `DeviceBuffer.unsafe_ptr()` origins at the
entry points (gemm/afn_apple_fast.mojo ~643-723); fixed with
`unsafe_origin_cast[MutAnyOrigin]()`.
