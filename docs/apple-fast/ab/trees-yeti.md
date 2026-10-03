# lane/apple-fast-trees-yeti: YetiRank search and the symmetric histogram (PLAN-trees next-experiments 2)

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check.
Every switch is compiled under FAST + Apple only and defaults OFF; IDENTICAL compiles the old code.
The estimation-side reuse (`MOJOLEARN_YETI_EST_REUSE_SEARCH`) lives on lane/apple-fast-yetirank;
nothing here touches `pointwise_oracle.mojo`, so the branches build together.

| switch | kind | site | what it changes under FAST on Apple |
|---|---|---|---|
| `-D MOJOLEARN_YETI_SEARCH_TASK16K=1` | build define | `gbdt/targets/kernel/yeti_rank.mojo` `yeti_task_16k_for`, `yeti_rank_task_block16k_kernel`, `launch_yeti_rank_with` | the task kernel (the whole pairwise gradient pass: draws, 1024-key sort, pairs) on 16 KiB of threadgroup memory instead of 32 KiB: in-place rank merge (reads, barrier, writes), accumulators gathered by each document's owner thread into registers in the scatter's (lane, phase) order, the hoisted pair decay tabled once per call (`yeti_rank_decay_table_kernel`, 1024 cells of `y.s_exp`). Same bits: same draws, same unique sort order, same expressions, same per-document accumulation order. |
| `-D MOJOLEARN_SYM_HIST_FAST=1` | build define | `gbdt/methods/greedy_subsets_searcher/kernel/hist_2_one_byte_8bit.mojo` `_h8_block_for` (the fused 8-bit kernel every greedy one-byte width takes under FAST); `checks/kernel_matrix.mojo` `hist2_block_size_for` (the hist2 shared-Int32 ladder) | the 512-thread block the shared-Int32 arm was measured at (1.94x, `hist_2_one_byte_base.mojo` docstring) instead of the 256 M2 Pro dispatch cap (`APPLE_HIST2_SHARED_I32_BLOCK_CAP`). Same bits: Int32 fixed-point sums, the block changes the grid only (the cap's own note). |

Causes.
- YetiRank Istella FAST fit 9.56 s (LightGBM 6.8): `tree_search` 6.0 s, `est.approx` 2.7 s, `sym.hist` 1.75 s
  (aft-st-gbdt1). The leaf estimation runs ONE evaluation per tree (`leaf_estimation_iterations=1`, the
  YetiRank default and the lane's setting), so `est.approx` is one task-kernel call, not a Newton relaunch
  loop; the search gradient is another. The task kernel (`yeti_rank_task_block_kernel`, commit 46f0cf09f)
  fills Apple's 32 KiB threadgroup limit exactly, so one task runs per GPU core (8 simdgroups), with 19
  barriers a round and the pair phases' read-modify-writes scattered into threadgroup memory. At 16 KiB
  two tasks share a core; the pair step has no shared read-modify-write and no barrier of its own.
- `sym.hist`: under FAST on Apple the greedy one-byte histograms run on `hist2_one_byte_8bit_kernel` at
  `H8_BLOCK = 256` since the M2 Pro cap of 2026-09-29; the M3 ran the 512 block correctly (the cap's
  note) and every histogram launch holds one 32 KB block per core either way, so the cap halves the
  per-core occupancy. The same cap bounds the hist2 5/6/7-bit ladder (`hist2_block_size_for`).
- Not changed: `scan_histograms_kernel` is one thread per (feature, leaf, stat) over at most 255 folds
  (a NUMERIC row, bits would move); the level winner reduce is one small block over the argmax partials.

Keep rule: a switch becomes the FAST default when its arm is faster on the M3 with the same model hash
(both arms are same-bits by construction; `aft_ab.sh` prints the hashes) and held-out quality within
FAST's run-to-run spread; then the define goes and the arm is the code. `yeti-both` measures the two
together on the YetiRank lane; the `gbdt-symmetric` and `gbdt-ordered` rows measure the histogram
define on the other symmetric-tree lanes it reaches.

Compile risks for the local session to watch: `_sort4_keys(mut v: SIMD[DType.uint64, 4])` and the
SIMD register arrays indexed by a runtime loop variable in `yeti_rank_task_block16k_kernel`; the
multi-line `comptime if (...)` in `hist2_block_size_for` and `_h8_block_for`.
