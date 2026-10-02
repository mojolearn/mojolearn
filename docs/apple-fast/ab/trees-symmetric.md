# trees-symmetric: SymmetricTree (CatBoost-style oblivious) FAST

Switch: `-D MOJOLEARN_SYM_DEVICE_LEVEL` (comptime alias `SYM_DEVICE_LEVEL` in
`gbdt/methods/kernel/pointwise_scores.mojo`), FAST + Apple only, default OFF.

What main already does on the device per level (`oblivious_tree_doc_parallel_structure_searcher.mojo`,
DEVIATION 207): the histogram covers all `2^d` parts in one launch (`compute_hist2_non_binary`,
`numBlocks.y = partCount`), the winner is folded and packed on the device, the split is applied to
the bins on the device, and the tree drains ONCE (no per-level host wait). What is not parallel:
the level score, `find_optimal_split_single_fold_kernel`, runs at most 32 blocks x 128 threads
(`PolicyScoreHelper.result_blocks`, CatBoost's `blockCount = 32`), and every thread loops the
level's nodes serially with strided histogram loads. istella (~56k bin features) gets 32 blocks
where 438 would fit; the cross-block winner fold is one thread.

What the switch changes (plain boosting, `fold_count == 1`, direct histogram layout):

- `pointwise_scores.mojo`: `pw_sym_score_partial_kernel`, grid (ceil(binFeatures/128) x
  ceil(p_count/4)) -- one thread per (bin feature, node tile) runs the same `add_leaf` body over its
  tile's nodes and stores the calcer's two running sums; `pw_sym_score_fold_kernel`, grid
  ceil(binFeatures/128), uncapped -- combines the tiles in order (`ScoreCalcer.combine`, new),
  `get_score`, the same fused gain, the unchanged `_block_argmin_and_store`. Dispatcher
  `find_optimal_split_sym` mirrors `find_optimal_split_plain`'s five arms.
- `pointwise_split_resolve.mojo`: `pw_fold_winner_block_kernel` (256 threads, strided sequential
  fold from the sentinel per thread, then a shared tree with the same `_record_less`), replacing the
  one-thread fold over the now-uncapped record count.
- `pointwise_scores_calcer.mojo`: under the alias, `result_blocks` is uncapped, the helper owns the
  partials buffer (`d_sym_partials`, sized for `1 << (max_depth - 1)` parts), `compute_optimal_split_dev`
  routes to the tiled pass, `resolve_optimal_split` to the block fold. Under other builds the only
  difference is a 1-float pool buffer never launched against.

Bits: the per-candidate score is the same sum in a different order (tile partials added instead
of one left-to-right chain over nodes), so FAST only. The cosine `DenumSqr` seed rides with tile 0.

Risky compile sites: struct field assignment inside kernels (`calcer.score = ...`,
`calcer.denum_sqr = ...`); the `comptime if ... else` initialising `self.d_sym_partials`;
`block_idx.y` in `pw_sym_score_partial_kernel`; shared `stack_allocation` of `Scalar[DType.uint32]`
in the block fold.

Request: `tsym-level-istella` (gbdt, gbdt-symmetric, istella, 2 pairs). taxi only after istella wins.
