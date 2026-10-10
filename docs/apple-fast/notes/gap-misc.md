# lane apple-fast-gap-misc: FAST Apple rows slower than the best opponent

Written from reading the code. Nothing in this lane was run or timed locally; the M3 A/Bs settle each point.
Every candidate is FAST + Apple only and default OFF. IDENTICAL compiles the old code.

## moe synthetic: 72.7 ms vs torch-eager-bf16 37.7 ms (1.93x)

Shape (tools/bench_board_algos.py:716): x (8192, 1024), D 1024, F 2816, E 8, top-2, forward only.
Route: `MoEBlock.forward` (python/mojolearn/_x_sequence_moe.py:90) -> binding `moe_forward` with the weight
handle, so the weights stay on the device -> `moe_forward_run` (sequence/pyapi.mojo:1562) -> `_moe_devgroup_rest`
(sequence/pyapi.mojo:1502) -> `DeviceExec.launch` (sequence/exec_device.mojo:597, :617).

Time breakdown hypothesis, largest first:
1. The expert products. 16,384 pairs x (2F x D + D x F) is 283 GFLOP, counting an fma as 2. The
   register-tiled kernels `moe_hidden_reg_kernel` (sequence/moe_reg.mojo:93) and `moe_out_reg_kernel` (:182)
   are scalar fma: each slab step reads 12 (hidden) or 8 (out) threadgroup words for every 32 or 16 fmas.
   That is about 5 TFLOP/s if the products take ~55 ms. The M3 Ultra's simdgroup matrix unit is unused.
2. Transfers inside the timed call. x goes up (32 MB, sequence/pyapi.mojo:1562) and y comes down (32 MB,
   :1546). The memory note puts this at ~20 ms per 64 MB for raw host pointers. Torch's x and y stay on
   MPS. This is the API contract (numpy in, numpy out), so it is not attacked here.
3. Zero fills. `DeviceExec._alloc` (sequence/exec_device.mojo:394) fills every buffer: H is 184 MB, S is
   64 MB, and the kernels overwrite both completely. That is ~250 MB of writes, about 0.5 ms.
4. Routing and grouping. 7 launches; the router logits are 67 MFLOP. This is negligible.

Candidates (sequence/moe_mma.mojo, wired at sequence/exec_device.mojo:597 and :617, DEVGROUP route only; the
pyapi flag is `c.i6 = 1` at sequence/pyapi.mojo:~1540):
- `-D MOJOLEARN_MOE_FAST_MMA`: both products as 8 x 8 simdgroup fragment products
  (`air.simdgroup_matrix_8x8_multiply_accumulate`, the AIR spelling core/gemm.mojo uses).
  - Block shape: 64 pairs x 32 (hidden) or 64 (out) outputs, 4 simdgroups, 16-word slabs.
  - Kernels: `moe_hidden_mma_kernel` :119, `moe_out_mma_kernel` :219.
  - Each cell is still one fma chain in ascending reduction order, without the per-step ftz. Bits may move
    in subnormal cases, so quality is judged on the board's rel_fro / max_rel_diff.
- `-D MOJOLEARN_MOE_FAST_MMA_KB32`: 32-word slabs, half the barriers (deleted 2026-10-09, docs/TOMBSTONES.md).
- `-D MOJOLEARN_MOE_FAST_MMA_WIDE`: hidden block 64 x 64, 32 accumulator fragments per simdgroup (deleted 2026-10-10, docs/TOMBSTONES.md).
- `-D MOJOLEARN_MOE_FAST_MMA_PF`: the next slab's global words are loaded into registers before the current
  slab's fragment products (deleted 2026-10-10, docs/TOMBSTONES.md).

## gbdt-depthwise taxi: 11,123 ms vs xgboost-cpu 10,435 ms (1.07x)

Shape (tools/speed_gbdt_arm.py:2016-2058): Depthwise, Logloss, 16 features, 500 trees, depth 8, 254 borders,
Cosine, Newton with 1 iteration, no bootstrap or weights, random_strength 0. Already default on FAST + Apple:
DW_FUSED_CHAIN, DW_NO_LEVEL_SYNC, QH_MODE_SKIP, DW2_PART_VEC4, DW2_SCAN_SMEM, NS_INHERIT_PARTITION.
DW_TREE_SYNC (one wait per tree) measured neutral, so most of the ~21 ms per tree is GPU row work.

Time breakdown hypothesis, largest first:
1. The histogram gather. `qh_hist_skip_kernel[True]`
   (gbdt/methods/greedy_subsets_searcher/kernel/hist_quantized_shared.mojo:550) reads 4 column-major cindex
   words per row through `row_index` (4 random lines per row), then does up to 32 threadgroup atomics.
2. `quantize_pair_kernel` (same file :152, launched at quantized_hist_launcher.mojo:231). Every level it
   gathers both stat planes per row, writes an 8-byte pair, and the histogram kernel reads the pair back.
3. Leaf estimation (doc_parallel_boosting.mojo ~2843). It re-gathers target and cursor and re-derives
   der/der2 that the tree-start gradient kernel already computed: 4 row passes and 2 waits per tree.
4. The end-of-tree partition-stats sweep (greedy_search_helper_depthwise.mojo:4712) plus its wait. It feeds
   only the searcher's leaf weights and values, and the caller re-estimates the values.
5. The per-tree magnitudes drain (doc_parallel_boosting.mojo:2646, a sync for two floats for the host
   `choose_scale`).
6. The full `row_index` copy-back every level (`dw2_copy_back_kernel`), and the snap-gradients pass plus
   two memsets per tree.

Candidates (all keep the same splits and leaf values):
- `-D MOJOLEARN_GBDT_QH_FAST_FUSED_Q` (hypothesis 2).
  - Sites: quantized_hist_launcher.mojo:83 and :217; kernel `fq` arm at hist_quantized_shared.mojo:584
    and :648.
  - When the level has ONE feature group (taxi's 16 features), the skip build quantizes each row's pair
    itself with `quantize_pair_kernel`'s expression (same dither keyed on position, same gathered value),
    and the quantize launch is dropped. Same pairs, same bits. Istella (14 groups) keeps the old route.
- `-D MOJOLEARN_GBDT_DW_FAST_SKIP_FINAL_STATS` (hypothesis 4).
  - Sites: greedy_search_helper_depthwise.mojo:373 and :4678; caller flag
    `TTreeStructureSearcherOptions.unit_weight_plane` set at doc_parallel_boosting.mojo:2607.
  - Taken only when plane 0 is the constant weight 1 (no weights, no bootstrap, not second-order, a
    pointwise single-target loss), and not for Lossguide, tree sync or the identity trace.
  - The leaf weights are the row counts, exact as integer sums below 2^24. The searcher's leaf values
    become 0 and are replaced by the caller's estimate as before.
- `-D MOJOLEARN_GBDT_DW_FAST_DEV_SCALE` (hypothesis 5).
  - Sites: greedy_search_helper_depthwise.mojo:383 and :2272; doc_parallel_boosting.mojo:2637.
  - The magnitudes buffer goes to the searcher, and `choose_scale_kernel` (DEVIATION 95, the host function
    bit for bit) writes the scale on the device. No per-tree drain. Depthwise only.

Not done (larger, noted): a row-major cindex copy for the gather (hypothesis 1); reusing the gradient
kernel's der/der2 in leaf estimation (hypothesis 3); ping-pong row_index instead of the copy-back (hypothesis 6).
