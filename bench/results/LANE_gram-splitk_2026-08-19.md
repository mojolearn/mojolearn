# LANE gram-splitk, 2026-08-19: the Gram shape gets a split-K kernel; `gemm_tn` dispatches; every MAX route was exhausted in source first

## (a) The MAX-route probe, and why the verdict is hand-write

The measured problem (orchestrator postscript, LANE_covariance-unblock):
`linalg.matmul[transpose_b=True]` delivers ~25 GFLOP/s on 32 x 32 x 4M
(322.9 ms, `PHASE pca.gemm_nt_core`) against ~248 on square shapes, and the
product's bandwidth floor (512 MB read once) is ~10-15 ms.

This lane read the MAX kernels SOURCE at the installed version rather than
probing blind: the pixi env ships only compiled `.mojoc`, so the modular
repo was sparse-cloned at tag `max/v26.5.0` (matching conda-meta
`max-26.5.0`), path `max/kernels/src/linalg/`. Findings, each with file:line:

1. **No split-K is reachable on Apple, by construction.** The public
   `matmul` entry (`matmul/__init__.mojo`) exposes `transpose_a`
   (refused), `transpose_b`, `b_packed`, epilogue lambdas, `use_tf32` —
   and NO k-partitioning parameter. `num_k_partitions` lives only in the
   internal `MatmulConfig` consumed by `multistage_gemm`, and every arm
   that reads it is comptime-gated `not has_apple_gpu_accelerator()`
   (`matmul/gpu/__init__.mojo:725` and `:1368`); the AMD split-K family
   (`amd_4wave_split_k_matmul`, `SplitKWorkspace`) is inside
   `has_amd_gpu_accelerator()` dispatch. There is no parameter to pass and
   no gate we can open.
2. **The starvation is confirmed in source, not just measured.** The Apple
   fp32 arm on M1-M4 (`compute_capability != 5`) is
   `gemm_kernel_apple_8x8` with `BLOCK_M=64, BLOCK_N=64`, launched
   `grid_dim=(ceildiv(n, 64), ceildiv(m, 64))`
   (`matmul/gpu/__init__.mojo:663-688`). At 32 x 32 that is grid (1, 1):
   ONE threadgroup on a 10-core GPU, the 4M-deep reduction serialized
   inside it. The M5 simdgroup arm above it is `compute_capability == 5`
   only and fp32 is additionally gated lossy (`:616-630`).
3. **No syrk / rank-k update exists.** `grep -ri syrk|rank_k|herk` over
   `max/kernels/src/linalg` is empty; the module index (matmul, bmm, gemv,
   grouped_matmul, qr_factorization, transpose, ...) has no candidate.
4. **`linalg.bmm.batched_matmul` exists and is CORRECT on this device**
   (standing vendor-table row: batch 1..7, m/n/k straddling 255/256/257
   and 1025, poisoned output, Float64 oracle — re-run green on this
   lane's final tree). It could EXPRESS k-chunked partials
   (batch = chunks). It is rejected on structure, read from source:
   - Its only Apple GPU arm is `naive_batched_matmul_kernel`
     (`bmm.mojo:899-925`, "Dispatching Batched Matmul via Naive
     Kernels"): grid `(n/16, m/16, batch)`, block 16x16, a SCALAR
     per-thread k-loop with no shared-memory tiling; under
     `transpose_b=True` the 16 j-threads read B at stride k floats,
     uncoalesced. The tiled arm requires `compute >= A100` (`:820-823`),
     the other requires AMD (`:853`).
   - The route also REQUIRES a materialized chunk-major transpose of X —
     twice, since one buffer cannot be two matmul operands (PORTING.md
     24) — plus a reduce pass. Minimum extra traffic ~2.5 GB against the
     hand-written kernel's single 512 MB read, before the naive kernel's
     own re-reads.
   - This lane cannot time (standing rule 5: the orchestrator times), so
     the "within ~2x of bandwidth" gate is decided by structure: a
     scalar-loop, no-reuse kernel plus 5x the mandatory traffic cannot
     reach a floor the tuned matmul misses by 13x. The deleted
     register-tiled contraction — a STRONGER kernel than this naive one —
     measured ~15 GFLOP/s.

So order-of-work (a) terminates at: their route is cuBLAS (CLOSED —
`raft/stats/detail/cov.cuh:65-66` gemm `CUBLAS_OP_T`;
`raft/linalg/detail/lstsq.cuh:293-309`), the MAX equivalent is measured 13x
off bandwidth at this shape, and no other MAX spelling reaches the k axis
on Apple. That is the hand-write exception, and it is written into
`core/gram_splitk.mojo`'s header with these citations.

## (b) The kernel: `core/gram_splitk.mojo`

Two kernels, one host wrapper:

- `gram_splitk_partial_kernel[CELLS]`: grid = `gram_splitk_chunk_count()`
  blocks (one per k-chunk), 256 threads. Each block streams its contiguous
  row-slice of X through a 16 KB shared staging tile (32 rows per
  `barrier()`; the load is a linear copy of a row-major span, coalesced)
  and accumulates the full m x m partial Gram in registers, CELLS cells
  per thread (`SIMD[float32, CELLS]`, comptime-unrolled). Cell
  coordinates are hoisted out of the row loop. X is read from DRAM exactly
  once across the grid, and NO transposes are needed — the two
  `transpose_kernel` passes of the fallback route (~22 ms at the bench
  shape) disappear on this arm too.
- `gram_splitk_reduce_kernel`: one thread per output cell folds the
  partials chunk 0..N-1 ascending, serial.
- `gemm_tn_splitk`: picks CELLS in {4, 16, 64} (smallest that covers
  m*m — 32 cols rides the 4-cell build, 128 the 64-cell one; the choice
  moves which thread owns which cell, never any cell's accumulation
  order), allocates the partials workspace per call (240 * m^2 floats;
  240 KB at m=32 — the callers' `xt` scratch is `k*m` and does NOT cover
  it at small k, so reuse was not an option), launches both kernels,
  synchronizes.

**Determinism.** The chunk count is a fixed function of the pinned
hardware constants — `APPLE_M4_GPU_CORES * max_active_blocks_per_core(256,
16 KB) * GRAM_OVERSUBSCRIBE = 10 * 12 * 2 = 240` — imported from
`neighbors/gbdt/distance/detail/pairwise_distance_base.mojo`, not
restated. Same (m, k) therefore always yields the same partition; each
cell is one serial fp32 chain per chunk in row order; the fold is serial
ascending. No device-wide float atomics anywhere. Run-to-run bit-identical.

**Bitwise symmetry.** Cells (i, j) and (j, i) multiply the same two staged
loads (IEEE multiply is exactly commutative) over the same rows in the same
order, and the reduce folds both in the same chunk order — bit-identical
across the diagonal by construction. Asserted with `!=` and no tolerance at
every check shape, and `check_covariance_is_symmetric` stays green through
the wrapper.

**Accuracy.** The split sum (chunk partials of ~k/240 terms, then a
240-term fold) is better conditioned than one k-long serial fp32 chain.
Proven, not argued: per-cell Float64 host oracle at 8 hazard shapes
(below), tolerance mag-relative 1e-5 + 1e-6 (the same budget, for the same
same-sign-diagonal reason, as the existing `gemm_tn` vendor row).

## (c) The dispatch threshold, computed not chosen

`gram_splitk_applies(m, n, k)` in `core/gram_splitk.mojo`:

    vendor_tiles = ceildiv(m, 64) * ceildiv(n, 64)     # 64 = BLOCK_M/N of
                                                       # gemm_kernel_apple_8x8,
                                                       # cited from v26.5.0 source
    block_slots  = APPLE_M4_GPU_CORES
                 * max_active_blocks_per_core(GRAM_TPB=256, 16 KB)   # = 120
    split-K iff  m == n  AND  m <= 128 (staging-tile capacity)
             AND m*n <= 256 * 64 (register-cell capacity)
             AND vendor_tiles < block_slots

The predicate is exactly the starvation condition: fewer vendor output
tiles than resident-block slots means the vendor kernel cannot fill the
device at ANY k. `k` is accepted and deliberately unused — starvation is a
property of the output shape, and at small k both routes are microseconds.
`gemm_tn` keeps its signature, so PCA (`pca.mojo:202`), tSVD (`tsvd.mojo:89`)
and OLS (`lstsq.mojo:120`) got the new path with zero caller changes; the
transpose+matmul route survives as `gemm_tn_via_transpose` for outputs big
enough to fill the device.

## (d)/(e) Checks, all green at the final tree

New: `mojo_only/gram_splitk_check.mojo`, run from `pca_main` (both sides of
the switch by name, PORTING_RULES 8). Final-tree outputs:

    === pca_main === (exit 0)
    check_gram_splitk_oracle OK: split-K arm matches the Float64 oracle per cell and is bitwise symmetric at 8 shapes (m 1..128 covering all three CELLS widths; k odd, prime, below/above the 240-chunk grid, and never a chunk multiple)
    check_gram_vendor_arm OK: transpose+matmul arm matches the Float64 oracle per cell and is bitwise symmetric at 33x33x257
    check_gram_dispatch OK: predicate routes 32x32x4M/1x1x7/128x128 to split-K and 129x129/768x768/m!=n to the fallback; wrapper verified per cell on arm 'split-K' at 32x32x100003 and arm 'transpose+matmul' at 768x768x257
    check_covariance_is_symmetric OK: all 6 off-diagonal pairs bitwise equal
    check_pca_fit OK: 4/4 eigenvalues within 6% of planted [100, 50, 20, 5], 4/4 components aligned with the planted rotation and orthogonal to the others, sign convention applied, ratios sum to 1.0
    check_pca_invariants OK: x3 scaling moved variances by exactly 9 and moved no direction; a +1000 column shift moved nothing at all, which is the reach evidence for the centering path
    check_input_restored OK: worst element moved 9.536743e-07 after a full fit
    check_tsvd_against_pca OK: identical directions on centered data, and a +1000 shift moved tsvd's first component to |dot| = 0.8276962458795801 while PCA's was unmoved

    === pca_wide_main === (exit 0)
    check_pca_wide OK at 64 and 128 features
    check_pca_truncation OK at 128 features: top 10 variances and ratios identical to the full fit, ...

    === ols_main === (exit 0)
    check_ols_exact OK: all 8 coefficients recovered within 1% from a noiseless planted model
    check_ols_scale_invariant OK: y x5 scaled every coefficient by exactly 5, which is the reach evidence for xty_kernel
    check_ols_beats_truth_on_noise OK: fitted residual 85.00028895726304 against the true model's 85.21995181198805
    check_ols_dispatch_guard OK: n_cols>n_rows and n_cols==1 both refused, 256x4 accepted

    === jacobi_main === (exit 0)
    check_jacobi_device_sizes OK: 16, 32, 33, 64, 128, 256 ...
    check_jacobi_reaches_past_32 OK ...
    check_jacobi_scale_invariance OK ...

    === vendor_main === (exit 0, "no WIRED primitive tested WRONG")
    core/gemm.mojo::gemm_tn (arm: split-K) ok
    core/gemm.mojo::gemm_tn_via_transpose (vendor arm) ok
    (table rows) core/gemm.mojo::gemm_tn (WIRED PATH) ... GPU, CORRECT ... 32x32x10007 through the real wrapper, arm: split-K
                 core/gemm.mojo::gemm_tn_via_transpose (vendor arm) ... GPU, CORRECT ... 32x32x10007 called by name, aliases pre-poisoned

The 8 split-K oracle shapes: (1, 7), (8, 33), (8, 239), (8, 241),
(33, 257), (32, 100003), (64, 4001), (128, 1025) — m = n = 1 with odd k, k
below/straddling the 240-chunk grid, k prime (never a chunk multiple), m
not dividing the 256-thread block, and all three CELLS instantiations.

**Reach by sabotage, three separate defects planted and каждый caught on the
first run (then reverted):**

1. Reduce kernel drops chunk 0 -> `check_gram_splitk_oracle FAILED at m=1
   k=7: cell (0, 0) device 1.4332 host 1.6172`.
2. Partial kernel drops each chunk's last row -> `... device 0.0 host
   1.6172` (at kc=1 the whole chunk empties, as predicted).
3. Wrapper dispatch passes k-1 to the split-K arm -> the DIRECT-arm checks
   stayed green and `check_gram_dispatch FAILED through the wrapper (arm
   split-K) at 32x32x100003: cell (0, 1) device 150.5615 host 150.1989` —
   proof the wrapper actually routes through the split-K arm, which no
   both-arms-correct run can show (reach is per-branch).

## Docs falsified by this result, fixed in the same commit

- `core/gemm.mojo` header and trailer ("a tuned matmul is not something to
  reimplement when one ships" was a square-shape truth; replaced with the
  dispatch story and the 25 GFLOP/s measurement).
- `core/column_stats.mojo` header ("they call a tuned BLAS, so we call
  ours" as an unconditional rule).
- `VENDOR_LIBRARIES.md`: gemm_tn wired row (now names the arm), new
  `gemm_tn_via_transpose` row, the "RAFT distance / stats::cov" mapping
  row, the RESOLVED paragraph, C8, and the `transpose_a` limits paragraph.
- `VENDOR_LIBS.md` MADE table (covariance_kernel row's "now calls").
- `PORTING.md` col-major hazard tail.
- `glm/.../lstsq.mojo`, `decomposition/.../pca.mojo`, `.../tsvd.mojo` call-site
  comments ("steps 1, 5, 6 all on MAX's tuned kernels" etc.).

## What the orchestrator should time

- The PCA and OLS bench arms at 4M x 32 (SCOREBOARD rows PCA 0.25x / OLS
  0.38x were taken on the starved vendor route; both fits now take split-K
  for their Gram step).
- The `PHASE pca.gemm_tn_alone` / `PHASE pca.gemm_nt_core` brackets: the
  322.9 ms matmul bracket becomes the split-K pair, and the
  `pca.one_transpose` bracket (10.8 ms x2) should vanish from the fit path
  entirely. Expected order: tens of ms for the Gram step (bandwidth floor
  ~10-15 ms for the 512 MB read + reduce noise), i.e. roughly 5-10x on the
  step and, per the postscript's phase table, most of PCA's 459 ms
  `compute_covariance` and OLS's 362 ms `lstsq_eig_total`.
- If the split-K step measures far off the ~15 ms floor, the first knob is
  in-kernel: hoisting the `tile[base + jj]` load out of the CELLS unroll
  when `256 % m == 0` (jj is constant across a thread's cells there) halves
  shared-memory reads; the second is `GRAM_ROWS_TILE`. Both are scheduling,
  neither moves a bit of the answer... except GRAM_OVERSUBSCRIBE/chunk
  count, which is NUMERIC (it fixes the summation split) and pinned.
- The workspace allocation (`enqueue_create_buffer` per `gemm_tn_splitk`
  call) is on the timed path; if it shows up in the brackets, promoting it
  to a caller-owned scratch is a signature change this lane deliberately
  did not make ("no caller changes").

## Commit

One add+commit; `git log -1 --format='%h parent %p'` reported in the lane
message. Files: `core/gram_splitk.mojo` (new), `core/gemm.mojo`,
`core/column_stats.mojo`, `mojo_only/kernel_matrix.mojo` (K_LIB_GRAM_SPLITK),
`mojo_only/gram_splitk_check.mojo` (new), `mojo_only/vendor_correctness_check.mojo`,
`decomposition/pca_main.mojo`, `decomposition/gbdt/linalg/detail/{pca,tsvd}.mojo`,
`glm/gbdt/linalg/detail/lstsq.mojo`, `VENDOR_LIBRARIES.md`, `VENDOR_LIBS.md`,
`PORTING.md`, this report. `neighbors/`, `dbscan/`, the scoreboard and
`bench/bench_main.mojo` untouched.
