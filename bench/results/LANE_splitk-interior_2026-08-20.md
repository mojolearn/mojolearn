# LANE_splitk-interior, 2026-08-20

One code commit. No timings were run (bench lock respected); every number
below is a correctness/bit-identity check.

## What the staging copy looked like before

Scalar per thread, the lead suspect confirmed: the global->shared copy in
`_gram_splitk_partial_body` was

    var i = tid
    while i < span:
        tile[i] = x.unsafe_load(t * m + i)        # (or - mu[i % m], centered)
        i += GRAM_TPB

one 4-byte global load per element, stride GRAM_TPB. At the steady-state
shape (32x32x4M) that is 128M scalar global loads per call -- the exact
pattern LANE_kmeans-kernel measured at ~3x cost on this device versus
`SIMD[float32, 4]` vector loads (assignment kernel 63 -> 21 ms; upstream's
scalar reads lean on NVIDIA warp-coalescing Apple does not replicate).

## What changed (commit `2c2b46c parent e7ba52e`)

- `core/gram_splitk.mojo`: the staging copy now has a VECTOR arm --
  `SIMD[float32, GRAM_STAGE_W]` (= 4 floats, 16 bytes) per global load and
  per shared store -- taken whenever `m % GRAM_STAGE_W == 0`, and the
  original scalar loop otherwise. The predicate is ONE shared symbol,
  `gram_splitk_stage_vectorized(m)` (`@always_inline`), branched on by the
  kernel body and asserted by the check file, so kernel and check cannot
  drift. The change lives in the SHARED `_gram_splitk_partial_body`, so the
  plain and centered (DEVIATION 42) variants get it structurally
  identically; on the centered arm the vector load of mu is legal because
  `e % m` is a multiple of GRAM_STAGE_W and never wraps a row
  (`e % m + GRAM_STAGE_W <= m` when both are multiples of 4).
- Alignment is decidable, per the brief: with `m % 4 == 0`, every chunk's
  span start `t * m`, every vector offset, and the span length are
  multiples of 4 elements, so every access is 16-byte aligned off the
  buffer base and no vector splits a row remainder. Every other width
  (m = 1, 3, 33, ...) takes the scalar arm whole -- the same
  alignment-by-selection discipline as the kmeans veclen port.
- `mojo_only/gram_splitk_check.mojo`: two new hazard shapes in
  `check_gram_splitk_oracle` -- m = 12 k = 1021 (vector arm, fresh width)
  and m = 3 k = 1021 (scalar arm, below the vector width, 256 % 3 != 0 so
  the non-hoisted accumulation column runs too) -- and
  `check_gram_dispatch` now asserts `gram_splitk_stage_vectorized` routes
  m = 32/4/128 to the vector arm and m = 33/1 to the scalar arm.
- Header prose updated in the same commit (rule 17): the "linear copy,
  coalesced" sentence now states the vector/scalar width split.

Chunk count, fold order, dispatch predicate, accumulation arithmetic and
order: UNTOUCHED.

## Why bit-identity is structural

The change is pure data movement. Whichever arm runs, the SAME fp32 values
land in the SAME shared-tile slots between the SAME barriers: a vector load
of 4 elements is 4 lane-wise IEEE reads of the same addresses, and on the
centered arm the per-lane fp32 subtract is the same IEEE op the scalar arm
performs lane by lane. The accumulation loops read the tile only after the
same `barrier()`, in unchanged order, so every cell's summation chain is
term-for-term identical. Proven, not argued:

FNV-1a over the exact fp32 bit patterns of z, hashed fixtures, IDENTICAL
before and after (harness was a temporary untracked main, deleted after
use, same protocol as LANE_pca-centering's; `diff` of the two dumps is
empty):

    bitdump direct m=1 k=7 hash=14091677597459411453
    bitdump direct m=8 k=241 hash=10424335584382362688
    bitdump direct m=32 k=10007 hash=11913496509763373480
    bitdump direct m=33 k=257 hash=13264787460106532682
    bitdump direct m=64 k=1025 hash=10184750221815004814
    bitdump direct m=128 k=1025 hash=3313393310186696935
    bitdump wrapper m=32 k=10007 hash=11913496509763373480
    bitdump wrapper m=8 k=241 hash=10424335584382362688
    bitdump centered m=4 k=8192 hash=9834973765613295896
    bitdump centered m=32 k=10007 hash=10607074516402292040
    bitdump centered m=33 k=257 hash=10422875596304972686

(These hashes are NOT comparable to LANE_pca-centering's -- different
fixture, and the centered rows here use a hashed mu rather than the real
producer. The before/after identity on one harness is the proof.)

## Reach, probed destructively (verify reach, not output)

A digest cannot distinguish a working arm from a no-op, so both arms were
sabotaged once (+1.0 planted in the load), run, and reverted:

- Sabotage in the VECTOR arm: `check_gram_splitk_oracle` PASSED m=1 and
  m=3 (scalar shapes) and FAILED at the first m % 4 == 0 shape --
  "FAILED at m=8 k=33: cell (0, 0) device 31.204... host 9.043...". The
  vector arm is TAKEN at every shipped width.
- Sabotage in the SCALAR arm: FAILED at the first scalar shape --
  "FAILED at m=1 k=7: cell (0, 0) device 9.406... host 1.617...". The
  scalar arm is taken-and-checked at ragged widths.

The permanent record of the split is the shared predicate asserted in
`check_gram_dispatch` plus per-cell oracle coverage of both arms
(m = 12 vector / m = 3, 33, 1 scalar) on hashed scattered data.

## Gates, all green after the change

`decomposition/pca_main.mojo`:

    check_hardware_matrix OK: apple column = the old constants bit-for-bit (10 cores, 3072 threads/core, 32 KB wall, occupancy 12); nvidia resolves 108/2048/48KB wall/164K partition -> 8 blocks; amd resolves 110/2048/64K partition -> 3 blocks (LDS term binding); readers (TARGET_GPU_CORES, max_active_blocks_per_core, gram chunk count = 240, gram dispatch, fused_l2_knn_grid) all agree with the table; build column apple
    check_gram_splitk_oracle OK: split-K arm matches the Float64 oracle per cell and is bitwise symmetric at 10 shapes (m 1..128 covering all three CELLS widths and both staging-copy arms; k odd, prime, below/above the 240-chunk grid, and never a chunk multiple)
    check_gram_vendor_arm OK: transpose+matmul arm matches the Float64 oracle per cell and is bitwise symmetric at 33x33x257
    check_gram_dispatch OK: predicate routes 32x32x4M/1x1x7/128x128 to split-K and 129x129/768x768/m!=n to the fallback; staging copy vectorizes m=32/4/128 and falls back scalar at m=33/1; wrapper verified per cell on arm 'split-K' at 32x32x100003 and arm 'transpose+matmul' at 768x768x257
    check_gram_centered_fused OK: fused centered read is bitwise equal to center-then-split-K at every cell (m=4/32/33, k=8192/100003/257, both workspace paths), and x is bit-identical afterwards
    check_covariance_is_symmetric OK: all 6 off-diagonal pairs bitwise equal
    check_covariance_fused_and_fallback_restore OK: fused arm left x bit-identical under both restore_input values; fallback arm's center moved 66048/66048 elements and its restore brought the worst back to 2.3841858e-07
    check_pca_fit OK: 4/4 eigenvalues within 6% of planted [100, 50, 20, 5], 4/4 components aligned with the planted rotation and orthogonal to the others, sign convention applied, ratios sum to 1.0
    check_pca_invariants OK: x3 scaling moved variances by exactly 9 and moved no direction; a +1000 column shift moved nothing at all, which is the reach evidence for the centering path
    check_input_restored OK: worst element moved 0.0 after a full fit
    check_tsvd_against_pca OK: identical directions on centered data, and a +1000 shift moved tsvd's first component to |dot| = 0.8276962458795801 while PCA's was unmoved

`glm/ols_main.mojo`:

    check_ols_exact OK: all 8 coefficients recovered within 1% from a noiseless planted model
    check_ols_scale_invariant OK: y x5 scaled every coefficient by exactly 5, which is the reach evidence for xty_kernel
    check_ols_beats_truth_on_noise OK: fitted residual 85.00028895726304 against the true model's 85.21995181198805
    check_ols_dispatch_guard OK: n_cols>n_rows and n_cols==1 both refused, 256x4 accepted

Note the oracle now says 10 shapes and the dispatch check now prints the
staging-arm verdicts.

## Deliberately NOT done, and why

- **Staging depth (more rows per barrier).** `GRAM_STAGE_FLOATS * 4` feeds
  `max_active_blocks_per_core`, which feeds `gram_splitk_chunk_count()`:
  deepening the tile changes occupancy and therefore the 240-chunk
  partition -- the summation split the determinism claims and the checks
  pin. Structurally forbidden by the brief, not merely untried.
- **Vector shared reads in the accumulation inner loop.** The layout does
  not permit it without remapping cell ownership: a thread's cells are
  `tid + c * GRAM_TPB`, so their tile addresses are strided, not
  contiguous (`ii` steps by GRAM_TPB/m rows). A contiguous-cell remap
  (`cell = tid * CELLS + c`) would keep per-cell order and enable a vector
  lds, but it rewrites ownership, guards, the uniform-jj hoist, and the
  partials store -- a larger scheduling change with its own risk, and the
  loads it widens are SHARED, not the global loads the ~3x device lesson
  is about. Left for a profiled round, not tuned blindly.
- **Two accumulator streams per thread.** Cannot provably preserve
  per-cell summation order; forbidden.
- **Reduce-kernel touches.** Fold order is pinned by the determinism
  guarantee, and at the steady-state shape the fold is microseconds of a
  42.5 ms call.
- **No timing runs.** Bench lock is the orchestrator's; the measured
  claim stays theirs.

If the orchestrator's timing shows the gap did NOT close materially, the
next suspect is the accumulation loop's shared-read traffic (5 scalar
shared loads per row per thread at m = 32, ~40x the staging op count),
which wants Apple Instruments device-time attribution, not structure
guessing.

## Suggested SCOREBOARD sentence (orchestrator's to place)

Split-K Gram staging copy vectorized to 16-byte loads on both variants
(scalar arm kept for ragged widths, one predicate both readers,
FNV-bit-hash-proven identical, both arms sabotage-probed for reach);
steady-state effect on the 42.5 ms / ~15 ms-floor gap awaits the
orchestrator's bench.

## Commits

- `2c2b46c parent e7ba52e` -- kernel + checks (this change)
- lane report commit below
