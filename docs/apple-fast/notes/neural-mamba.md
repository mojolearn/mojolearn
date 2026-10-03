# afn-mamba: Apple FAST candidates for the Mamba-1/2/3 block forwards

Lane afn-mamba, 2026-10-03, branch lane/apple-fast-neural-mamba. Board lanes
mamba1-forward, mamba2-forward, mamba3-forward (shape full: batch 1, length
2048, d_model 384, so d_inner 768). Binding: mamba (bindings/build_mamba.sh).
Nothing here was run or measured; every candidate was compiled only.

## Profile of main's forwards (static count, per block call)

Counted in main's source (merge base), host side of one block call through
bindings/_mojolearn_mamba.mojo:

| piece | launches | synchronizes | allocations | host copies |
|---|---|---|---|---|
| binding run (all three blocks) | 0 | 27 sites | weights, state, x per call | 23 sites (uploads + y/state readback) |
| Mamba-1 (modeling_mamba.mojo + selective_scan_interface.mojo) | about 9 (gemms not counted) | 18 + 2 sites (one per stage under `mamba_zeros[wait]` and the stage waits) | about 30 separate buffers (10 weights, state, 19 stages, x) | refusal: 13 device-to-host downloads (x, 10 weights, state), about 7 MB at the board shape |
| Mamba-2 (mamba2.mojo + ssd_minimal.mojo) | 12 + 9 | about 30 (each stage `mamba_zeros[wait=True]`) | about 40 (9 weights, state, 28 stages, x) | refusal: host download per name |
| Mamba-3 (mamba3.mojo + mamba3_siso.mojo) | 9 + 25 | 8 + 2 sites plus the stage waits | about 45 (9 weights, state, 33 stages, x) | refusal (mamba3_refusal.mojo): one device reduction but one readback + wait per name |

Hot spots by mechanism (Apple costs from AFN-COMMON: ~20 us per launch growing
~0.25 us per live buffer, ~180 us per launch + readback + sync):

- Mamba-1 selective scan: ONE launch, but one thread per (batch, channel)
  walking all 2048 steps of 16 states: 768 threads, most of the GPU idle.
- Mamba-1 conv1d: one thread per (batch, channel) walking l, plus a window
  update launch; x_proj split and A = -exp(A_log) are two more launches with
  waits.
- Mamba-2 chunked SSD: S12 C.B^T, S14 (G o L).X_d and S16 X_d^T.(B o decay)
  are small per-chunk matmuls spelled one thread per output cell, serial k.
- Mamba-3: nine small elementwise launches around the SISO core with five
  waits; launch bound on the M3.
- All three: dozens of separately allocated buffers live per call (each
  launch pays for all of them) and a wait per stage allocation.
- Mamba-1/2 refusal: a host download and scan per named input every call.

## Candidates (each its own define, FAST + Apple only, default OFF)

All switches live in mamba/impl/modules/afn_defines.mojo:
`AFN_APPLE_FAST = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and
has_apple_gpu_accelerator()`, and each switch is `AFN_APPLE_FAST and
is_defined["MOJOLEARN_AFN_<NAME>"]()` or `MOJOLEARN_AFN_MAMBA_ALL`. Every use
site is a `comptime if` whose other arm is main's spelling, or (ssd_forward) a
runtime flag that is constant False unless the switch is on, with the afn
launch wrappers themselves comptime-guarded so no other build instantiates an
afn kernel.

1. MOJOLEARN_AFN_MAMBA1_CHUNKSCAN (mamba/impl/ops/afn_selective_scan.mojo,
   dispatch in selective_scan_interface.mojo). One simdgroup of 32 lanes per
   (batch, dim); the sequence is cut into 32 chunks. Pass 1 computes each
   chunk's local end state and decay product, lane 0 folds the 32 summaries in
   threadgroup memory, pass 2 re-walks each chunk from its carried start with
   main's per-step arithmetic. Still one launch, 32x the parallelism, no
   scratch. Fold order of the recurrence across chunk boundaries changes (f32
   reassociation); exp(dt A) recomputed, never stored.
2. MOJOLEARN_AFN_MAMBA1_FUSE_IN (mamba/impl/modeling/afn_mamba1_fused.mojo,
   dispatch in modeling_mamba.mojo). Conv1d + SiLU + window update as one
   token-parallel launch (one thread per (token, channel)); x_proj split and
   A = -exp(A_log) as one launch. Per-element arithmetic is main's; no float
   crosses a thread differently.
3. MOJOLEARN_AFN_MAMBA2_SSD_MMA (mamba/impl/modules/afn_ssd_mma.mojo, dispatch
   in ssd_minimal.mojo ssd_forward). S12, S13+S14 and S15+S16 as one launch
   each of 256-thread blocks owning a 64-row tile, operands staged 32 deep in
   threadgroup memory, `air.simdgroup_matrix_8x8_multiply_accumulate` f32
   (no lower precision). Per-cell result is a serial ascending fma chain;
   main folds two leaves then adds, so only fold order moves. Causal Y_diag
   skips structurally zero windows. Applies only when Q % 64 == 0 (board
   shape: yes); otherwise main's kernels run.
4. MOJOLEARN_AFN_MAMBA3_SISO_FUSED (mamba/impl/ops/afn_mamba3_fused.mojo,
   dispatch in mamba3.mojo and mamba3_siso.mojo). Nine elementwise launches
   become four (prep, scale+angle increments, rot+kscale+reports,
   dacs+state decay); per-stage waits dropped (a traced run keeps them). The
   serial angle chain stays as is.
5. MOJOLEARN_AFN_MAMBA_ARENA (mamba/impl/modules/afn_arena.mojo, used in the
   binding and the three block files). Every weight, state piece, stage and
   x of one block call is a `create_sub_buffer` view of ONE allocation,
   zero-filled by one launch; uploads go straight from the caller's arrays
   into views with no per-buffer wait; no per-stage waits. Bytes read and
   written are identical; only storage placement moves. Arena kept alive past
   the last struct built from it.
6. MOJOLEARN_AFN_MAMBA_DEVICE_REFUSAL (mamba/impl/modules/afn_refusal.mojo,
   used in modeling_mamba.mojo, mamba2.mojo, mamba3.mojo). The non-finite
   refusal of every named input as one block-strided reduction launch per
   name plus one fold launch, then ONE readback and ONE wait per call. Code
   scheme and raised text are the host refusal's; Int32 codes (Metal has no
   64-bit atomics), longer buffers refused by name.
7. MOJOLEARN_AFN_MAMBA_ALL: all six (they compose: different stages, and the
   arena views are plain DeviceBuffers to every other candidate).

## Quality

No candidate lowers precision: f32 everywhere, accumulation f32, no
approximate exp/softplus/rsqrt, no caps. What changes is fold order only
(CHUNKSCAN, SSD_MMA) or launch shape and buffer placement (FUSE_IN,
SISO_FUSED, ARENA, DEVICE_REFUSAL: bit-identical to FAST-off by
construction). The judge is tools/neural_fast_quality.py through afn_ab.sh.

## Build notes

- Build: `MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_MOJO_BUILD_FLAGS="-D
  MOJOLEARN_AFN_<NAME>" bash bindings/build_mamba.sh`. The script exports
  MOJOLEARN_SKIP_BUILD_GATE=1 itself, so no build here imported or ran the
  binding; the AIR-blob and otool checks (strings/otool only) still run.
- Compile table (2026-10-03). Andrew's change of plan: lanes stop compiling,
  the M3 manager peer compiles everything.

| build | status here |
|---|---|
| FAST + `-D MOJOLEARN_AFN_MAMBA_ALL` | compiled once, rc=1 at the front end with 9 errors in two causes, both FIXED in source and not recompiled: `MAMBA_GUARD` not imported in mamba2.mojo; the SSD MMA launches passed one buffer's pointer mutably twice (now `unsafe_origin_cast[MutAnyOrigin]`). No other front-end error was reported; Metal kernel codegen was never reached. UNCOMPILED after the fixes (peer compiles). |
| FAST + MAMBA1_CHUNKSCAN | UNCOMPILED (peer compiles) |
| FAST + MAMBA1_FUSE_IN | UNCOMPILED (peer compiles) |
| FAST + MAMBA2_SSD_MMA | UNCOMPILED (peer compiles) |
| FAST + MAMBA3_SISO_FUSED | UNCOMPILED (peer compiles) |
| FAST + MAMBA_ARENA | UNCOMPILED (peer compiles) |
| FAST + MAMBA_DEVICE_REFUSAL | UNCOMPILED (peer compiles) |
| FAST, no afn define | UNCOMPILED (peer compiles) |
| IDENTICAL | UNCOMPILED (peer compiles) |

  Watch for, in the peer's builds: single-define builds take arms the ALL
  build did not (for example mamba3.mojo's `SISO_FUSED or ARENA` wait
  helper with only one of the two on); and the IDENTICAL/NVIDIA/AMD builds
  must not instantiate any afn kernel (every afn launch wrapper is reached
  only through a `comptime if` on its switch; afn_ssd_mma.mojo's wrappers
  guard themselves because ssd_forward's call site is a runtime flag).
- Not touched: the *_backward files (afn-samba owns them), training/**,
  gemm/**. modeling_mamba.mojo (mamba/impl/modeling, the Mamba-1 forward)
  is edited only behind the switches.
