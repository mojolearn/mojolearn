# lane/apple-fast-yetirank: YetiRank tree_search (score grid) and sym.hist (8-bit unroll)

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check.
Every switch is compiled under FAST + Apple only and defaults OFF; IDENTICAL compiles main's code.
The branch carries lane/apple-fast-trees-yeti (merged 2026-10-02: `MOJOLEARN_YETI_SEARCH_TASK16K`,
`MOJOLEARN_SYM_HIST_FAST`) and its own `MOJOLEARN_YETI_FAST_SORT` (A/B aft-ab-ysort1, m3 #517, queued
earlier; not re-requested). The sym.hist switch below is layered on `MOJOLEARN_SYM_HIST_FAST`'s 512
block: both arms of its pair carry that define.

| switch | kind | site | what it changes under FAST on Apple |
|---|---|---|---|
| `-D MOJOLEARN_YETI_TREE_SEARCH_SCORE_GRID=1` | build define | `gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo` `sym_score_grid_fast_for`, `sym_argmax_blocks_for` (the workspace's `out_score`/`out_bin` allocation and the level loop's `argmax_blocks`) | the `sym.score` launch (`compute_optimal_splits_kernel`, the per-level split scoring inside `tree_search`) on `ceil(binFeatures / 128)` blocks (cap 1024), one thread per bin feature, instead of CatBoost's `min(64, ceil(binFeatures / 256))` grid-strided blocks (8192 threads for Istella's ~56k bin features on 80 cores). Same bits: one thread scores each bin feature in the same order on either grid; the block argmax and both resolve kernels (`resolve_and_pack_kernel`, `sym_resolve_split_count_kernel`, each a loop over `argmax_blocks`) apply one total order (gain, then the smaller bin feature), so the winner is partition-independent. Corner noted in the docstring: an all-skipped or all-`-inf` block's sentinel; present on the reference grid too. |
| `-D MOJOLEARN_YETI_SYM_HIST_UNROLL8=1` | build define | `gbdt/methods/greedy_subsets_searcher/kernel/hist_2_one_byte_8bit.mojo` `_h8_unroll_for` (`H8_UNROLL`, and through it `H8_POINTS`, `H8_MIN_DOCS`, `ALIGN_SIZE`) | the fused 8-bit kernel's main loop (every greedy one-byte width's `sym.hist` under FAST) at 8 aligned warp loads (32 points) per trip instead of 4: twice the bytes in flight before the trip's atomics on the one 32 KB block a core holds, half the trips. Same bits: Int32 position-dithered addends, Int32 atomic sums; every point read once at the same position. |

Causes.
- YetiRank Istella FAST fit 9.56 s at 46f0cf09f (aft-st-gbdt1): `tree_search` 6.0 s, `est.approx` 2.7 s,
  `sym.hist` 1.75 s; 5.6 s after the estimation reuse (aft-ab-yreuse1). The task kernel's share of
  `tree_search` is trees-yeti's (`TASK16K`) and this branch's sort (`FAST_SORT`); the remaining
  `tree_search` stages are the symmetric level's (`sym.score`, `sym.winner`, `sym.split.*`,
  `sym.pstats`). `sym.score` is the one whose grid is data-sized and capped at 64 blocks regardless
  of the device, so it is the first non-task target.
- `sym.hist`: trees-yeti's 512 block fixes the per-core occupancy; with one block per core what is
  left is per-thread memory-level parallelism, which the unroll sets.

Keep rule: a switch becomes the FAST default when its arm is faster on the M3 with the same model
hash (`aft_ab.sh` prints the hashes) and held-out quality within FAST's run-to-run spread; then
the define goes and the arm is the code.

Compile and launch risks for the M3 build to watch: `comptime if SYM_SCORE_GRID_FAST:` with
`return` inside a `def` that also returns below it (`sym_argmax_blocks_for`); the multi-line
`comptime if (...)` in `_h8_unroll_for`; under `UNROLL8` the three `InlineArray[.., H8_POINTS]`
register arrays double, so the 512 block may exceed the pipeline's `maxTotalThreadsPerThreadgroup`
on a part without Dynamic Caching (the M3 has it; the cap's note in `checks/kernel_matrix.mojo`).
