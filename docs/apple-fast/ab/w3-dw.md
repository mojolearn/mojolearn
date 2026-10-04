# lane/apple-fast-w3-dw: depthwise grids sized by the work (`MOJOLEARN_GBDT_DW_FLAT_GRID`)

Base: origin/main `5a5fc6395b0bf736162623294e1b27c5e41ee2f4`. Code only. No local build, test or timing.
Row: gbdt-depthwise taxi, FAST main 10,812 ms against xgboost-cpu 10,435 ms. AUC must not drop.

## Why (found by reading the code)

- Profile: depthwise taxi took 14.3 ms per tree at 1M rows (trees-apple3 run 6856b5f8f) and about
  21.6 ms per tree at 5.25M rows on the board (10,812 ms / 500). So a large part of each tree's
  cost does not scale with rows. Earlier attempts to remove waits did not move it: one wait per
  tree (`DW_TREE_SYNC`) was -0.5%, noise.
- What does not scale with rows is grid width. In the DW2 split chain (`_launch_fused_split_chain`),
  K1 (flags + count), K3 (place + scatter) and K4 (copy back) launch
  `(min(max_chunks(n_rows), 2 * sm_count), n_split)` threadgroups of 512 threads. That is 160 x 128
  = 20,480 threadgroups at a depth-8 level on an 80-core M3 Ultra, while each leaf has about 20
  chunks of work. About 88% of the threadgroups load a leaf header and return. Summed over the
  levels and the three kernels, that is about 120k threadgroups per tree, nearly all of them empty.
- The quantize pass of the quantized histogram build (`quantize_pair_kernel`) also launches
  `min(n_rows / 512, 4 * sm_count) = 320` blocks per built leaf. At level 7 that is
  320 x 64 = 20,480 blocks for about 36 blocks of rows per leaf.

## Change

| file:line | what |
|---|---|
| `gbdt/methods/greedy_subsets_searcher/kernel/dw3_flat.mojo` (new) | K1 / K3 / K4 bodies unchanged, run on a 1D grid over the level's (slot, chunk) work list. Each block takes a block prefix sum of the per-slot chunk counts into threadgroup memory, then binary-searches each work item's slot. Grid `min(chunk upper bound, 4 * sm_count)`. GUARD trims the slots to the device split count. |
| `greedy_search_helper_depthwise.mojo` `DW_FLAT_GRID` (after `DW2_SCAN_SMEM`) | The define: FAST + Apple + DW2_PART_VEC4 + `-D MOJOLEARN_GBDT_DW_FLAT_GRID`. |
| `greedy_search_helper_depthwise.mojo` `_launch_fused_split_chain`, DW2 arm | The flat launches when `n_split <= 512`. K2 is unchanged. Above 512 slots the DW2 2D kernels run. |
| `greedy_search_helper_depthwise.mojo`, the `launch_quantized_histograms` call sites | Depthwise only: `qh_max_rows` = the largest built leaf's host size, from the last level's readback, the sizes `non_zero` was chosen by. |
| `quantized_hist_launcher.mojo` `launch_quantized_histograms` | New `max_live_rows = -1` argument. When it is set, `qx <= ceil(max_live_rows / 512)`. |

Bits: same work items, same bodies, integer moves only. The quantize pass writes each position once
with the same expression whatever the grid. So the change is bit-identical by construction.
Lossguide never takes the fused chain and keeps the full quantize grid. IDENTICAL never compiles
any of this.

## Build (manager, M2)

Build both arms from this branch with `bindings/build_gbdt.sh`, `MOJOLEARN_NUMERIC_MODE=fast`,
`MOJOLEARN_COMPILE_JOBS=1`. Arm A gets `MOJOLEARN_EXTRA_DEFINES=""` and arm B gets
`MOJOLEARN_EXTRA_DEFINES="-D MOJOLEARN_GBDT_DW_FLAT_GRID"`.

**build_gbdt.sh reads `MOJOLEARN_EXTRA_DEFINES`, not `MOJOLEARN_MOJO_BUILD_FLAGS`.** That means
`tools/afc_ab_def.sh` (which sets `MOJOLEARN_MOJO_BUILD_FLAGS`) builds two identical gbdt arms.
Its `afc_ab.sh` also has no `trees` family on origin/main. Stage the arms at
`~/mq/verified-arms/<SHA>/gbdt/` with manifest `defines_A=""` and
`defines_B="-D MOJOLEARN_GBDT_DW_FLAT_GRID"`.

Bindings needed on the M3: the FAST base (`_mojolearn.so`) and **gbdt** (`_mojolearn_gbdt.so`, the
staged arms). No other tree binding is needed.

## Quality gate (`tools/aft_w3dw.py`, fixed before results)

The gate uses the held-out `predict_proba` of the 500-tree depth-8 Depthwise board config:

- PASS-EXACT when the A and B prediction hashes are equal.
- Otherwise PASS when AUC_B >= AUC_A - 4e-4 and logloss_B <= logloss_A + 1e-4. These bounds are
  the largest spread recorded between main and bit-identical arms on depthwise taxi (AUC .632211 to
  .632554, logloss .527920 to .528002).
- FAIL otherwise.

The quality job runs on the first 1,000,000 taxi rows. The timing job refuses to start without its
PASS.json, and it applies the same gate to the full-data metrics of the board row.
