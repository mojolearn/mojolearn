# LANE_kmeans-kernel 2026-08-20: line-by-line policy diff of the fused L2-NN port, and the fixes

Assignment: diff our fused SIMT kernel (`cluster/ported/distance/
fused_distance_nn/simt_kernel.mojo`) against upstream
(`cuvs/cpp/src/distance/detail/fused_distance_nn/simt_kernel.cuh`,
`raft/cpp/include/raft/linalg/detail/contractions.cuh`,
`raft/cpp/include/raft/distance/detail/pairwise_distance_base.cuh`,
`cuvs/cpp/src/distance/fused_distance_nn-inl.cuh`) at every policy-level
decision, fix what differs, leave what is faithful. NO timings were run; the
orchestrator re-verdicts.

**Verdict up front: the port was NOT faithful.** Three port errors (not
decisions) were found and fixed, all on the brief's suspect list, plus one
dropped epilog guard. The accumulate side (`reduce_by_key.mojo`) came back
CLEAN on the veclen question. Two deliberate differences are newly priced as
PORTING.md 42 and 43.

## The diff table

| # | Policy item | Theirs (file:line) | Ours BEFORE | Ours AFTER |
|---|---|---|---|---|
| 1 | Global loads | `ldg` of `Veclen` floats (`raft::TxN_t`), `detail/contractions.cuh:189-259`; zero-fill on the same guard | ONE float at a time, flat `e = tid; e += threads` loop over tile cells | **FIXED**: `LdgPerThX/Y` vector loads of `veclen` floats per thread, `srowid = tid / LdgThRow`, `scolid = (tid % LdgThRow) * Veclen` (`:96-102`), zero-filled exactly where theirs is |
| 1b | veclen SELECTION | `fused_distance_nn-inl.cuh:107-110` (16 B: veclen 4), `:158` (8 B: veclen 2), `:210` (scalar), from `4k % {16,8}` AND both base-pointer alignments | none -- one scalar instantiation | **FIXED**: `fused_veclen_for(k, px, py)`, the computation not the constant; launcher and checks dispatch on the same function; scalar arm is the fallback exactly where theirs is |
| 2 | Tile/thread policy | `Policy4x4<float>` = KernelPolicy<f32, v, 32, 4, 4, 16, 16> (`contractions.cuh:160-166`); `Policy4x4Skinny` = <f32, v, 8, 4, 4, 8, 8> at `k < 32` (`:183-196`, selected at `-inl.cuh:105`) | Policy4x4 constants correct, but hardwired; NO skinny arm, so `k < 32` ran the k=32 tiles their comment calls redundant | **FIXED**: kernel parameterized `[veclen, kblk, tr, tc]` (rpt=cpt=4, both their float policies); `fused_is_skinny(k) = k < 32` routes to Skinny. Not a hardware-derived number -- it is their k-shape selection, ported as a computation |
| 2b | Row/col OWNERSHIP | STRIDED: `accrowid + i * AccThRows`, `acccolid + j * AccThCols` (`contractions.cuh:100-102`, lds/epilog/write indexing throughout) | BLOCKED: `tr * 4 + i`, `tc * 4 + j` | **FIXED**: strided, theirs. Argmin result unchanged (total order is partition-independent; per-cell dot order unchanged), smem/lane access pattern now theirs |
| 3 | Launch grid | `launchConfigGenerator<P>(m, n, shmemSize, kernel)` (`pairwise_distance_base.cuh:295-322`, called at `fused_l2_nn.cuh:135-138`); kernel grid-strides BOTH axes (`run()`, `:131-186`) | hardcoded `grid = (1, ceil(m/64))`, kernel had NO m grid-stride | **FIXED**: `_launch_fused` calls the existing `launch_config_generator` port (M4 inputs via `hardware_matrix`, reused from `neighbors/ported/distance/detail/pairwise_distance_base.mojo`, not duplicated); kernel grid-strides m exactly as `run()` does, with the `val` reset per row tile (`simt_kernel.cuh:144-147`) that the stride requires. `grid.x` stays PINNED to 1 -- that is the pre-existing `updateReducedVal` mutex replacement, and their generator returns grid.x=1 at every shipped k-means shape anyway |
| 4 | Smem staging | DOUBLE-buffered (`P::SmemSize = 2 * SmemPage`, `contractions.cuh:104`; `pageWr/pageRd`), one `__syncthreads` per k-tile; this IS their pre-Ampere/SIMT arm (no cp.async in it) | single page, two barriers per k-tile | **KEPT, now priced**: Policy4x4's double buffer is 36,864 B + norms and Apple caps a threadgroup at 32,768 B. Costs overlap only when `k > Kblk` (=32); every shipped shape has `k <= Kblk`, where their main loop body runs zero times. **DEVIATION 42** |
| 4b | Vector smem traffic | `sts`/`lds` move `Veclen` floats (`detail/contractions.cuh:262-299`); `SmemStride = Kblk + Veclen` padding | scalar smem reads/writes; stride padding was already theirs | **FIXED**: vector sts and lds, register tile FMA per `accumulate_reg_tile` (`pairwise_distance_base.cuh:207-221`), `v` ascending inside ascending chunks so every accumulator cell keeps the identical k-ascending sum order |
| 5 | Epilogue norms | staged through smem, `load_norms` (`pairwise_distance_base.cuh:243-274`), X row only on the first column tile | `xn`/`yn` read from GLOBAL per cell/column-tile (xn re-read from global for every column tile) | **FIXED**: ported `load_norms` including its first-tile guard; +512 B smem |
| 5b | Epilogue op | `l2_exp_distance_op::epilog` (`l2_exp.cuh:117-136`): positivity clamp AND self-neighbor round-off guard `!((val*val < 1e-6) * (xn == yn))` | sign clamp only; the guard was dropped (recorded in the old PORTED_MAP row) | **FIXED**: full guard ported, `FUSED_CLAMP_PRECISION = 1e-6` = their `get_clamp_precision<float>` (`l2_exp.cuh:36`) |
| 5c | `sqrt` placement | per accumulator cell before the min reduce (`l2_exp.cuh:137-145`) | once per row at the final write | **KEPT, now priced**: monotone + injective on `[0, inf)`, so argmin, tie set, and written value are bit-identical; 16 sqrts/thread/tile -> 4/row. **DEVIATION 43** |
| 5d | Tie-break | fused: `KVPMinReduceImpl` value-only (`helper_structs.cuh:39-44`, shuffle-shape dependent); unfused: `raft::argmin_op` total order | argmin_op total order in both arms | UNCHANGED -- pre-existing documented deviation, PORTING.md 14 |
| 6 | Accumulate side | `reduce_rows_by_key.cuh` reads `d_A` ONE element per thread, no veclen anywhere (rowmajor `:280-287`, colmajor `:213-227`; `reduce_cols_by_key.cuh` likewise); privatization structure = their cached/colmajor arms | scalar reads, same flat cell indexing, Int32 fixed-point (documented) | **CLEAN. No change.** The veclen suspicion does not apply: upstream's own accumulate is scalar |
| 6b | Cross-thread merge | `raft::shfl` rotate in width-`AccThCols` subgroups (`simt_kernel.cuh:119-130`) | `shuffle_xor` butterfly, identical result over the total order | UNCHANGED -- already ported and argued in the module docstring |

## What was fixed vs what was already faithful

Fixed (all cited upstream file:line in code comments): vectorized global
loads + their thread-to-load partition; the veclen selection computation
with scalar fallback; the skinny policy and its `k < 32` selection; strided
row/column ownership; vector smem sts/lds with the register-tile FMA; the m
grid-stride + per-tile `val` reset; the launch grid from their
`launchConfigGenerator` (existing port reused); smem-staged norms; the
self-neighbor epilog guard.

Already faithful: the fused-arm dispatch itself (sentinel-proven before this
lane); the `KVPair val[]`-in-registers design; the SmemStride padding; the
intra-warp merge; the flat-scalar accumulate side (`reduce_by_key.mojo`) --
upstream is scalar there too, so that half of the 55 ms is NOT a load-width
story.

Deliberate differences, all priced: single smem page (DEVIATION 42, forced
by Apple's 32 KB threadgroup cap, free at `k <= 32`), sqrt at the write
(DEVIATION 43, provably bit-identical), grid.x pinned 1 + no mutex
(pre-existing `replaced` row, updated), argmin_op total order (PORTING.md
14, pre-existing).

## Summation-order caveat

None to accept: every accumulator cell still sums its k terms in strictly
ascending k order at every veclen (the `v` loop ascends inside ascending
chunks), zeros padding contributes `+0.0`, and the min-merge is over a total
order, so no reduction shape or ownership change can move a value or a key.
The one arithmetic change is the self-neighbor guard (5b): it zeroes a cell
only when `val^2 < 1e-6` AND the two norms are bit-equal -- their behavior,
absent from ours until now. All checks stayed green with it.

## Check output (build + run under the build lock, `build/kmeans_main`)

```
check_reach_by_sabotage OK: centroid_norm moved 384/512 labels; x_norm moved 512 distances and 0 labels, which is the predicted shape
check_kmeans_fit OK: 4/4 centroids matched as a permutation, 0/512 rows misassigned, inertia 170.703125 vs expected 171.19361649473004 (rel 0.0028651272446548032), 2 iterations
check_device_inclusive_scan OK: 20000 entries, worst relative error 0.0, past one block's worth
check_kmeans_plus_plus_init OK: 4/4 centroids recovered as a permutation through the k-means++ path, inertia 170.703125, 2 iterations
check_fused_reduction_across_lanes OK: 512 rows x 40 clusters match a host argmin, winners spread over 16 owner lanes
check_assignment_arm_dispatch OK: fused arm proved (0/32768 tile cells written; unfused sabotage overwrote 32768); arms agree on all labels, min_dist worst rel 0.0
  veclen=1 arm at d=33: 8256 labels correct (grid.y 120 of 129 tiles, 576 rows past the resident grid), fused == unfused
  veclen=2 arm at d=34: 8256 labels correct (grid.y 120 of 129 tiles, 576 rows past the resident grid), fused == unfused
check_fused_policy_dispatch OK: selection pinned (4/2/1 by k and alignment, skinny at k<32), bench alignment k=32 takes veclen 4 on real buffers, scalar and 2-wide arms correct through the launcher with the m grid-stride exercised
check_privatized_accumulate OK: 2048 sum cells + 64 weight cells bit-identical direct vs privatized, run-twice bitwise equal, dropped flush moved 128 cells
```

Reach, per the standing rule: the new `check_fused_policy_dispatch` pins the
selection VALUES against hand-transcribed upstream arms, proves k=32 on real
device buffers takes veclen 4 (raises if the allocator ever stops 16-byte
aligning), and runs the scalar (k=33) and 2-wide (k=34) fallback arms for
real through the launcher on hashed-jitter planted fixtures at 8,256 rows --
129 row tiles against a 120-block resident grid, so 576 rows are reachable
ONLY through the new m grid-stride loop, and their labels are checked. The
skinny arm is exercised through the launcher by `check_kmeans_fit` (d=4).
Fused and unfused labels agree bitwise on every fixture.

Note for the fit checks: `check_kmeans_fit`'s d=4 now routes to
`Policy4x4Skinny` per their selection, so the fit path itself changed
instantiation and stayed green; `check_fused_reduction_across_lanes`
instantiates the NORMAL 16-wide policy directly (its subject is the 16-lane
merge), with its fixture math updated for strided ownership.

## Suggested SCOREBOARD sentence (I do not edit the scoreboard)

> k-means assignment kernel re-ported faithful to cuVS's SIMT policy
> (veclen-4 vector loads via their alignment selection, skinny policy at
> k<32, strided ownership, occupancy-computed grid with m grid-stride,
> smem-staged norms); accumulate side confirmed faithful-scalar like
> upstream's; awaiting re-timing -- the 0.81x figure predates this.

## Commits

```
d62ab0e parent 10ae918   (git log -1 --format='%h parent %p')
```
plus the report commit that follows it.

Files: `cluster/ported/distance/fused_distance_nn/simt_kernel.mojo`,
`cluster/ported/cluster/detail/min_cluster_distance_compute.mojo`,
`cluster/mojo_only/kmeans_check.mojo`, `cluster/kmeans_main.mojo`,
`cluster/PORTED_MAP.tsv`, `PORTING.md` (deviations 42, 43), this report.
