# LANE_gram-tile 2026-08-20: register-tile cell ownership for the split-K Gram

The change GRAM_PROFILE_2026-08-20.md funded: the accumulation loop's
cell-ownership remap from strided singles to per-thread T x T register
rectangles, in `_gram_splitk_partial_body` (the SHARED body -- the plain
and the centered/DEVIATION-42 kernels changed identically). No timing was
run in this lane (orchestrator owns the bench lock); the deliverable is a
bit-identical, reach-proven remap awaiting measurement.

## The ownership map, before and after (m = 4 drawn; m = 32/64/128 same shape)

BEFORE -- strided singles, thread number in each output cell (thread t owns
cells {t + c*256}; at m = 4 that is one single each for threads 0..15):

    j:     0   1   2   3
    i=0:   0   1   2   3
    i=1:   4   5   6   7
    i=2:   8   9  10  11
    i=3:  12  13  14  15

AFTER -- 2x2 rectangles (T = 2, mt = m/T = 2; thread t owns rows
(t//mt)*T..+T-1, cols (t%mt)*T..+T-1):

    j:     0   1   2   3
    i=0:   0   0   1   1
    i=1:   0   0   1   1
    i=2:   2   2   3   3
    i=3:   2   2   3   3

At the bench m = 32 a thread previously owned 4 singles in one column
scattered across rows {t//32, 8+t//32, 16+t//32, 24+t//32}; it now owns a
2x2 block at (2*(t//16), 2*(t%16)). T is the square root of the existing
CELLS width (2/4/8 for CELLS 4/16/64), so T*T == CELLS and the accumulator
register count per thread is UNCHANGED -- only the read pattern moved. The
tiled path also drops the 2*CELLS hoisted int32 index lanes (ii/jj), which
now fill only on the strided arms.

## FMA per shared read, per CELLS width (per staged row)

| CELLS | shape reached  | strided singles | uniform_jj hoist (pre-change at these m) | register tile (this lane) |
|-------|----------------|-----------------|-------------------------------------------|---------------------------|
| 4     | m<=32 (bench)  | 4/8 = 0.50      | 4/5 = 0.80                                | 4/4 = **1.00**            |
| 16    | 33<=m<=64      | 16/32 = 0.50    | 16/17 = 0.94 (m=64 only)                  | 16/8 = **2.00**           |
| 64    | 65<=m<=128     | 64/128 = 0.50   | 64/65 = 0.98 (m=128 only)                 | 64/16 = **4.00**          |

The register tile reads T column values + T row values (2T reads) and does
T*T FMAs: ratio T/2. Honest note: at the shipped power-of-two widths the
pre-change arm was the uniform_jj hoist (LANE_pca-centering_2026-08-20),
whose ratio is CELLS/(CELLS+1), not the 0.5 the profile quotes for strided
singles -- but the hoist's CELLS reads per row are DISTINCT addresses (one
per owned row index), whereas the tile's 2T reads amortize over T*T cells,
so the read count per FMA still falls 1.25x/2.1x/4.1x at m = 32/64/128.

## Why per-cell accumulation order is provably unchanged

- Each cell (i, j) is still ONE serial fp32 chain: staged rows r ascending
  within a tile, tiles t ascending within the chunk, chunks folded 0..239
  ascending by the untouched `gram_splitk_reduce_kernel`. The remap moves
  WHICH thread runs a cell's chain, never the chain itself.
- The product for cell (i, j) is the same `tile[base+i] * tile[base+j]`,
  i-operand first, staged through registers (a load cannot change a value).
- The writeback still lands cell (i, j) at `partials[chunk*mn + i*m + j]`
  -- cell ADDRESSES did not move, so writer and fold reader agree without
  either changing.
- Staging copy, barriers, chunk count (240), `gram_splitk_applies`, fold
  order, and all public signatures: untouched.

Proof, not argument -- FNV-1a dump over every output cell's fp32 bit
pattern at 11 shapes (8 plain: m=1/3/8/12/32/33/64/128 with odd/prime k
including k=100003; 3 centered: m=4/32/33), hashed-scattered fixtures,
poisoned outputs, before vs after:

    all 11 lines IDENTICAL (bits_before.txt == bits_final.txt)

## Reach, per branch (sentinel sabotage, hazard shapes, hashed data)

- SABOTAGE-A (+1.0 planted in the register-tile FMA): the FNV moved at
  EXACTLY the tiled shapes -- plain m=8/12/32/64/128 and centered m=4/32 --
  and the strided shapes m=1/3/33 (plain) + m=33 (centered) held their
  baseline bits. Both variants proven through the shared body.
- SABOTAGE-B (+1.0 planted in BOTH strided sub-arms): exact inversion --
  only m=1 (the uniform_jj sub-arm), m=3/33 plain and m=33 centered moved;
  every tiled shape held baseline bits.
- Both sabotages reverted; final rebuild re-matched the baseline dump.

Standing coverage: `check_gram_dispatch` now asserts the ownership
predicate (`gram_splitk_reg_tiled`, the symbol the body branches on) and
pins the CELLS width per m via `gram_splitk_cells_for` -- the same
one-predicate-both-readers discipline as the staging copy's split. The
oracle's existing shapes hold both arms per cell: m=8/12/32/64/128 tiled,
m=1/3/33 strided.

## Check output (required binaries, both ALL green)

decomposition/pca_main:

check_hardware_matrix OK: apple column = the old constants bit-for-bit (10 cores, 3072 threads/core, 32 KB wall, occupancy 12); nvidia resolves 108/2048/48KB wall/164K partition -> 8 blocks; amd resolves 110/2048/64K partition -> 3 blocks (LDS term binding); readers (TARGET_GPU_CORES, max_active_blocks_per_core, gram chunk count = 240, gram dispatch, fused_l2_knn_grid) all agree with the table; build column apple
check_gram_splitk_oracle OK: split-K arm matches the Float64 oracle per cell and is bitwise symmetric at 10 shapes (m 1..128 covering all three CELLS widths and both staging-copy arms; k odd, prime, below/above the 240-chunk grid, and never a chunk multiple)
check_gram_vendor_arm OK: transpose+matmul arm matches the Float64 oracle per cell and is bitwise symmetric at 33x33x257
check_gram_dispatch OK: predicate routes 32x32x4M/1x1x7/128x128 to split-K and 129x129/768x768/m!=n to the fallback; staging copy vectorizes m=32/4/128 and falls back scalar at m=33/1; register tile owns m=32/64/128 (2x2/4x4/8x8) and declines m=1/3/33 to the strided arm; wrapper verified per cell on arm 'split-K' at 32x32x100003 and arm 'transpose+matmul' at 768x768x257
check_gram_centered_fused OK: fused centered read is bitwise equal to center-then-split-K at every cell (m=4/32/33, k=8192/100003/257, both workspace paths), and x is bit-identical afterwards
check_covariance_is_symmetric OK: all 6 off-diagonal pairs bitwise equal
check_covariance_fused_and_fallback_restore OK: fused arm left x bit-identical under both restore_input values; fallback arm's center moved 66048/66048 elements and its restore brought the worst back to 2.3841858e-07
check_pca_fit OK: 4/4 eigenvalues within 6% of planted [100, 50, 20, 5], 4/4 components aligned with the planted rotation and orthogonal to the others, sign convention applied, ratios sum to 1.0
check_pca_invariants OK: x3 scaling moved variances by exactly 9 and moved no direction; a +1000 column shift moved nothing at all, which is the reach evidence for the centering path
check_input_restored OK: worst element moved 0.0 after a full fit
check_tsvd_against_pca OK: identical directions on centered data, and a +1000 shift moved tsvd's first component to |dot| = 0.8276962458795801 while PCA's was unmoved

glm/ols_main:

check_ols_exact OK: all 8 coefficients recovered within 1% from a noiseless planted model
check_ols_scale_invariant OK: y x5 scaled every coefficient by exactly 5, which is the reach evidence for xty_kernel
check_ols_beats_truth_on_noise OK: fitted residual 85.00028895726304 against the true model's 85.21995181198805
check_ols_dispatch_guard OK: n_cols>n_rows and n_cols==1 both refused, 256x4 accepted

## What this lane did NOT do

- NO timing runs. The win is priced by the profile's elimination argument,
  not measured; the orchestrator's next quiet-box window prices it.
- Did not vectorize the tile's column read into one wide shared load
  (would cut 2T reads to T+1; pure data movement, bit-safe, but unpriced --
  left as the obvious follow-up if the tile arm measures short of the
  floor).
- Did not widen tiles beyond the square factorization of CELLS: register
  count would grow and the M4 spills silently (mojolearn-hardware-limits).
- Did not remove the uniform_jj FLOOR KNOB arm: it now serves only m = 1
  among GRAM_TPB-dividing widths (comment corrected in place, rule 17),
  but it remains the better read pattern where it applies and its
  bit-identity proof stands.
- Did not add partial-rectangle edge handling for ragged m: ragged widths
  (m=1/3/33 among the checked shapes) keep the existing strided arms
  WHOLE, per the brief's sanctioned dispatch-split option. Every ragged m
  is far from the bench shapes; if a ragged width ever ships hot, the
  edge-thread variant is the follow-up.
- Did not touch neighbors/, gbdt/methods/ (peer in-progress work),
  SCOREBOARD, or archive/reference/PORTING.md (no caller-visible behavior change; the three
  new symbols -- `gram_splitk_cells_for`, `gram_splitk_reg_tile_side`,
  `gram_splitk_reg_tiled` -- are additive, no existing signature moved).

## Suggested SCOREBOARD sentence

Split-K Gram accumulation is now register-tiled (each thread owns a
sqrt(CELLS)-square rectangle; FMA/shared-read 1.0/2.0/4.0 at m=32/64/128
vs 0.8/0.94/0.98 before), bit-identical at 11 FNV shapes across both
variants with per-branch sabotage reach proofs, ragged m unchanged on the
strided arm; UNMEASURED, awaiting the bench window.

## Commits

- `50451a9 parent c2aa6a0` -- the remap + checks + this report (parent is
  a PEER commit that landed in the shared checkout after this lane's last
  log read; mojotrees-shared-checkout-parents rule).
- The report-finalization commit below records this section itself.
