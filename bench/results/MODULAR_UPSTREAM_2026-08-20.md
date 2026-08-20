# DRAFT upstream report to Modular, 2026-08-20 — NOT YET FILED

STATUS: draft for Andrew's review. Filing (GitHub issue / forum post) is an
external action and waits for his explicit word. Every finding below is a
transcript of a probe that ran on this box; nothing is speculated.

## Environment (pin)

- MAX/Mojo `max-26.5.0` (conda-meta; kernel sources read at repo tag
  `max/v26.5.0`), via the pixi env at `../mojotrees/pixi.toml`
- Apple M4 (base, 10-core GPU), macOS (Darwin 25.5.0), Metal backend
- Reproduce harness: this repo's `vendor_main.mojo` —
  `tools/with_build_lock.sh pixi run --manifest-path ../mojotrees/pixi.toml
  mojo build -I . vendor_main.mojo -o /tmp/vendor_probe && /tmp/vendor_probe`
  prints the full verdict table; `/tmp/vendor_probe --transpose` runs the
  probe that aborts the process. Probe code:
  `mojo_only/vendor_correctness_check.mojo` (each finding cites its function).

Findings are ordered by severity: silent wrong answers first, then a process
abort, then wrong-at-scale, then a default that cannot launch, then the
performance gap.

## 1. SILENT WRONG: `linalg.matmul.matmul[transpose_b=True]` at `n = 1` never writes the output

`matmul[transpose_b=True, target="gpu"]` at m=64, n=1, k=32 leaves the output
buffer **untouched** — not written zero: pre-poisoned cells survive, so the
caller reads whatever was in memory. The same product with
`transpose_b=False` is CORRECT at the same shape, so the defect belongs to
the `transpose_b=True` arm, not to n=1 degeneracy generally.
(`vendor_correctness_check.mojo::check_matmul`, ~:423-585.)

Severity: any GEMV-degenerate call spelled as a matmul silently returns
stale memory. Simple linear regression on one predictor is exactly this
shape. Our workaround: route n=1 to `linalg.gemv.gemv_gpu`.

## 2. SILENT WRONG, ARM-DEPENDENT: `linalg.matmul` with a `col_major` view as operand A

Zero-copy T-N spelling (col-major view of a row-major buffer as A): the
dispatcher honors the view's strides on some internal arms and silently
IGNORES them on others, producing plausible wrong numbers. Measured
boundary: CORRECT at 32x32x100003, 33x17x255, 8x8x8, 129x127x513; **every
cell wrong** across m=n in {4..64} x k in {64..2048} (8x8x512, 32x32x2048,
64x64x64, ...); at 8x1x33 the output is WRITTEN wrong (distinct from
finding 1's unwritten). The ok/wrong boundary zigzags with shape and matches
no predicate we could form, so the feature is unusable defensively: there is
no shape guard a caller can write. Full sweep:
`bench/results/LANE_covariance-unblock_2026-08-19.md`.

## 3. SILENT WRONG: `linalg.gemv.gemv_gpu` with a `col_major` view as A

Wrong at 8 of 8 outputs at k=4001: the GEMV kernel indexes A as raw
row-major memory, ignoring the view's layout entirely.
(`vendor_correctness_check.mojo::check_gemv`.) Together with finding 2, no
zero-copy transposed operand works on this backend; every T-op needs a
materialized transpose.

## 4. PROCESS ABORT: `linalg.transpose.transpose` on device buffers

257x129 float32 device buffers: aborts the process (exit 133), not a
catchable raise, inside `linalg::transpose::_copy_with_strides rank=2
dtype=f32` — "enqueue_cpu_range is only supported on CPU DeviceContexts".
Re-verified 2026-08-19 evening, unchanged. Reproducer:
`/tmp/vendor_probe --transpose`
(`vendor_correctness_check.mojo::check_transpose_aborts`). Consequence of
2+3+4 combined: the materialized transpose that findings 2 and 3 force must
be hand-written too.

## 5. WRONG AT SCALE: `nn.argsort.argsort` not monotone past 256

Correct at n = 1, 2, 255, 256; **not monotone at 257** and every larger size
tried, with the first inversion always at output index 256. Reads as a
single-block sort with no cross-block merge on this backend.
(`vendor_correctness_check.mojo::check_argsort`.)

## 6. CANNOT LAUNCH: `nn.toppminp_gpu.run_radix_sort_pairs_gpu` default parameterization

At `BLOCK_SIZE=256` (the default), raises on Apple at every size including
n=1: "Threadgroup memory size (32900) exceeds the maximum threadgroup memory
allowed (32768)". The default cannot run on this device at all.
(`vendor_correctness_check.mojo::check_radix_sort_pairs`.)

## 7. PERFORMANCE: fp32 matmul launches ONE threadgroup on tall-skinny Gram shapes; all split-K arms are comptime-gated off Apple

The measured problem: `matmul[transpose_b=True]` on 32 x 32 x 4,000,000
(X^T X, the covariance/normal-equations shape) runs at ~25 GFLOP/s — 322.9
ms where the bandwidth floor (512 MB read once at ~120 GB/s measured) is
~10-15 ms. Square shapes on the same box measure ~248 GFLOP/s, so this is
shape-specific starvation, not a general ceiling.

Root cause, read in source at `max/v26.5.0`:

- The Apple fp32 arm for M1-M4 (`compute_capability != 5`) is
  `gemm_kernel_apple_8x8` with BLOCK_M=64, BLOCK_N=64, launched
  `grid_dim = (ceildiv(n, 64), ceildiv(m, 64))`
  (`max/kernels/src/linalg/matmul/gpu/__init__.mojo:663-688`). At 32x32
  output that is grid (1,1): one threadgroup on a 10-core GPU, with the
  4M-deep reduction serialized inside it.
- Every arm that reads `MatmulConfig.num_k_partitions` is comptime-gated
  `not has_apple_gpu_accelerator()` (`matmul/gpu/__init__.mojo:725`,
  `:1368`); the AMD split-K family (`amd_4wave_split_k_matmul`,
  `SplitKWorkspace`) is inside `has_amd_gpu_accelerator()`. The public
  `matmul` entry exposes no k-partitioning parameter. So no spelling
  reaches the k axis on Apple.
- No syrk/rank-k primitive exists as an alternative (grep over
  `max/kernels/src/linalg` is empty for syrk|rank_k|herk).
- `linalg.bmm.batched_matmul`'s only Apple arm is
  `naive_batched_matmul_kernel` (`bmm.mojo:899-925`; the tiled arm requires
  compute >= A100 at `:820-823`), so batching over k-chunks is not a route
  either.

Evidence the gap is the launch geometry and nothing exotic: a hand-written
split-K kernel in this repo (`core/gram_splitk.mojo` — 240 k-chunk blocks,
16 KB shared staging, deterministic ascending fold, no atomics, single read
of X) computes the same product in **49 ms first-shot vs 345 ms** for the
matmul route at the same shape on the same device (the 345 includes the two
materialized transposes findings 2-4 force; the matmul alone is 322.9 ms).

The ask: split-K (or any k-axis grid) for the Apple backend when
`ceildiv(m,64) * ceildiv(n,64)` is far below the core count. Even a naive
fixed-chunk split with a separate reduce pass recovers ~7x on this shape.

## What we are NOT reporting

- `linalg.matmul` at n=1 returning zeros with `transpose_b=False` spellings
  in other codepaths — superseded by finding 1's sharper attribution.
- The M5 simdgroup arm's fp32 gating (`__init__.mojo:616-630`) — observed in
  source, not exercised (no M5 here).
- Anything about warp/block primitives: `std.gpu.primitives.warp` and
  `block` both WORK on Metal; earlier internal reports of their absence were
  our own wrong import paths.
