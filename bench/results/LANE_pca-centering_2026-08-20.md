# LANE_pca-centering, 2026-08-20

Two commits. No timings were run (bench lock respected); every number below
is a correctness/bit-identity check, milliseconds of runtime each.

## Task A: split-K floor knobs (commit `a4a7ed0 parent d72bd97`)

Both named knobs from SCOREBOARD_2026-08-19 item 5, in
`core/gram_splitk.mojo` + `core/gemm.mojo`:

1. **Hoisted jj tile read.** The "redundant load" exists and is this: in the
   inner unroll, cell `c` of a thread reads `tile[base + jj[c]]` with
   `jj[c] = (tid + c * GRAM_TPB) % m`. When `GRAM_TPB % m == 0` (every
   power-of-two width up to 128, including the bench's m = 32), that index
   is `tid % m` for EVERY c, so the unroll was loading the same shared
   address CELLS times per row. Now one load per row
   (`var xj = tile[base + jm]`), branch block-uniform, guards and the
   `ii[c]` read untouched. The non-multiple column (e.g. m = 33) keeps the
   original loop.
2. **Workspace off the per-call path.** `gemm_tn` now feeds its `xt` alias
   scratch (>= k*m floats, pure scratch on both arms) to a new
   `gemm_tn_splitk_into`, which lands the 240-chunk partials there whenever
   `gram_splitk_scratch_covers(m, k)` (`k >= 240 * m`; k >= 7,680 at
   m = 32, so every shipped tall-skinny fit) and allocates only below that.
   This mirrors RAFT/cuML passing workspace down the chain instead of
   allocating inside a GEMM. Chunk count, fold order, dispatch predicate:
   unchanged by construction and pinned by the unchanged checks.

Bit-identity PROVEN, not argued: an FNV-1a hash of the exact fp32 bit
patterns of z, on hashed data, at 8 shape/arm combos (direct m=1/8/32/33/
64/128 covering all three CELLS widths, hoisted and non-hoisted columns,
alloc and scratch-reuse paths; wrapper m=32 k=10007 and m=8 k=241) is
IDENTICAL before and after the change:

    bitdump direct m=1 k=7 hash=8799171291094083541
    bitdump direct m=8 k=241 hash=5069481695786553245
    bitdump direct m=32 k=10007 hash=16103696918470406760
    bitdump direct m=33 k=257 hash=11426734749910421101
    bitdump direct m=64 k=1025 hash=14255954601037525424
    bitdump direct m=128 k=1025 hash=8730320934637824066
    bitdump wrapper m=32 k=10007 hash=16103696918470406760
    bitdump wrapper m=8 k=241 hash=5069481695786553245

(The dump harness was a temporary untracked main, deleted after use; the
hashes above are the record. It was re-run after Task B as well: the plain
arm is still bit-identical through the kernel-body refactor.)

## Task B: fused mean-centering -- RAFT's `stable=false` @todo (this commit)

**Upstream basis** (read, not recalled):
`~/CascadeProjects/upstream/raft/cpp/include/raft/stats/detail/cov.cuh` --
`stable=true` arm at :58-66 (`meanCenter` in place :61, cuBLAS GEMM
:65-66; the header note :44-45 says the input comes back centered, and
cuML's `pca.cuh:138` meanAdd restores it); `stable=false` arm at :67-69 is
`///@todo: implement this using cutlass + customized epilogue!` over
`ASSERT(false, "cov: Implement stable=false case!")`. The fusion is THEIR
declared design, unshippable on cuBLAS (no epilogue hook), shippable on our
hand-written kernel. DEVIATION 42 (PORTING.md) records it.

**What changed:**

- `core/gram_splitk.mojo`: partial kernel refactored into
  `_gram_splitk_partial_body[CELLS, CENTERED]` (shared, `@always_inline`)
  with two entries -- the plain `gram_splitk_partial_kernel` (passes a dead
  mu slot the `comptime if` eliminates) and
  `gram_splitk_partial_centered_kernel`, whose staging-tile load reads
  `x[t*m+i] - mu[i % m]` so BOTH operand reads of every product see
  centered values in registers and X is never written. Host entries
  `gram_centered_splitk` / `gram_centered_splitk_into` mirror the plain
  pair (same width dispatch, same reduce, same workspace rule). OPT-IN
  ONLY: `gemm_tn`'s dispatch never takes the centered entry.
- `decomposition/gbdt/linalg/detail/pca.mojo::compute_covariance`: arms
  split on the SAME `gram_splitk_applies(n_cols, n_cols, n_rows)` that
  `gemm_tn` asks (one predicate, both readers, no drift; no new target
  test). Fused arm: mean -> `gram_centered_splitk_into` (x_alias as
  workspace) -> scale; NO center, NO restore, x untouched. Fallback arm:
  their shipped stable path verbatim (center -> `gemm_tn` -> scale ->
  restore under `restore_input`). The old "`meanAdd`, pca.cuh:138. Do not
  drop this." comment was falsified on the fused arm and is rewritten per
  rule 17: the restore is dropped there precisely because x is never
  centered in place, and running it would CORRUPT pristine data; on the
  fallback arm it stays, guarded `restore_input and not fused`.
- Docs the change falsified, fixed in the same commit (rule 17):
  `pca.mojo` module docstring (step-6 paragraph), `pca_check.mojo` module
  docstring + `check_input_restored` docstring (both named
  `shift_columns_kernel` as the only centering mechanism),
  `core/column_stats.mojo` semantics paragraph (also carried an invented
  `detail/pca.cuh:186` citation; now `pca.cuh:138`),
  `decomposition/README.md` invariant-2 passage, `UNWIRED.md` reach row.

**New checks (both wired into `decomposition/pca_main.mojo`):**

- `check_gram_centered_fused` (`mojo_only/gram_splitk_check.mojo`): the
  shipped pipeline (`column_mean_kernel` -> `shift_columns_kernel(-1)` ->
  `gemm_tn_splitk`) vs the fused arm, SAME device mu, hashed data with a
  deliberately nonzero per-column mean (`+ col * 0.25`, so a fused arm
  reading raw X or a zero mu cannot pass), z poisoned, every cell compared
  with `!=`, and X asserted bit-identical after the fused call. Shapes
  m=4/32/33 x k=8192/100003/257, both workspace paths, covering the
  hoisted and non-hoisted accumulation columns.
- `check_covariance_fused_and_fallback_restore`
  (`decomposition/mojo_only/pca_check.mojo`): the WIRED arms, split by the
  same predicate the code uses (and raising if the fixture shapes ever stop
  landing on their intended arms). Fused arm (4 cols): x bit-identical
  under both `restore_input` values. Fallback arm (129 cols, one past
  `GRAM_MAX_COLS`): with `restore_input=False` the center MUST move x
  (sentinel -- proves the center pass has reach, so "restored" cannot be
  two no-ops), with `restore_input=True` x returns within rounding.

**Check output (full `pca_main` run at head):**

    check_hardware_matrix OK: apple column = the old constants bit-for-bit (10 cores, 3072 threads/core, 32 KB wall, occupancy 12); nvidia resolves 108/2048/48KB wall/164K partition -> 8 blocks; amd resolves 110/2048/64K partition -> 3 blocks (LDS term binding); readers (TARGET_GPU_CORES, max_active_blocks_per_core, gram chunk count = 240, gram dispatch, fused_l2_knn_grid) all agree with the table; build column apple
    check_gram_splitk_oracle OK: split-K arm matches the Float64 oracle per cell and is bitwise symmetric at 8 shapes (m 1..128 covering all three CELLS widths; k odd, prime, below/above the 240-chunk grid, and never a chunk multiple)
    check_gram_vendor_arm OK: transpose+matmul arm matches the Float64 oracle per cell and is bitwise symmetric at 33x33x257
    check_gram_dispatch OK: predicate routes 32x32x4M/1x1x7/128x128 to split-K and 129x129/768x768/m!=n to the fallback; wrapper verified per cell on arm 'split-K' at 32x32x100003 and arm 'transpose+matmul' at 768x768x257
    check_gram_centered_fused OK: fused centered read is bitwise equal to center-then-split-K at every cell (m=4/32/33, k=8192/100003/257, both workspace paths), and x is bit-identical afterwards
    check_covariance_is_symmetric OK: all 6 off-diagonal pairs bitwise equal
    check_covariance_fused_and_fallback_restore OK: fused arm left x bit-identical under both restore_input values; fallback arm's center moved 66048/66048 elements and its restore brought the worst back to 2.3841858e-07
    check_pca_fit OK: 4/4 eigenvalues within 6% of planted [100, 50, 20, 5], 4/4 components aligned with the planted rotation and orthogonal to the others, sign convention applied, ratios sum to 1.0
    check_pca_invariants OK: x3 scaling moved variances by exactly 9 and moved no direction; a +1000 column shift moved nothing at all, which is the reach evidence for the centering path
    check_input_restored OK: worst element moved 0.0 after a full fit
    check_tsvd_against_pca OK: identical directions on centered data, and a +1000 shift moved tsvd's first component to |dot| = 0.8276962458795801 while PCA's was unmoved

Note `check_input_restored`: worst move is now EXACTLY 0.0 (was 9.5e-07) --
the fused arm's x is bit-identical, not restored-within-rounding.

`glm/ols_main.mojo`: 4/4 OK, unchanged (fusion is opt-in; `gemm_tn`
untouched). `neighbors/knn_main.mojo`: all 13 checks OK, unchanged.

## NOT done, and why

- **The 1/(n-1) scale is NOT folded into the kernel.** The brief allowed it
  only if bit-identity is trivial. It is not: the scale kernel computes
  `cov[i] * (1/(n-1))` on the FOLDED total, and folding the multiply into
  the reduce's per-cell tail is the same multiply -- but folding it into
  the partial kernel (the tempting version, saving a pass) would multiply
  partials before summation, which reassociates. The launch it would save
  is one 256-thread-block pass over n_cols^2 elements, microseconds; left
  alone.
- **`pca_transform`'s center/restore passes are untouched.** Its product is
  `gemm_nt` (rows x components), not the Gram shape; fusing centering there
  is a different kernel's epilogue and was not in scope.
- **No timings.** The ~100 ms claim for center+restore at the bench shape
  is the scoreboard's number, not re-measured here; the orchestrator owns
  the bench lock.

## Suggested SCOREBOARD sentence (orchestrator's to place)

PCA next-step 2 LANDED: the split-K arm now fuses mean-centering into the
Gram read (RAFT's own cov.cuh stable=false @todo) and drops the center +
restore passes entirely -- x is bit-identical after a fit -- and item 5's
two floor knobs (hoisted uniform-jj tile read, xt-scratch partials
workspace) are in, all bit-identical by check; awaiting a quiet-box re-time
of pca/ols.

## Commits

- Task A: `a4a7ed0 parent d72bd97`
- Task B: this commit (hash in the lane's closing report; parent will be
  whatever the shared checkout holds at commit time -- report format
  `%h parent %p`).
