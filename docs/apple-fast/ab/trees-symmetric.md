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
where 438 would fit; the cross-block winner fold (<= 32 records) is cheap and stays as main has it.

What the switch changes (plain boosting, `fold_count == 1`, direct histogram layout):

- `pointwise_scores.mojo`: `pw_sym_score_partial_kernel`, grid (ceil(binFeatures/128) x
  ceil(p_count/4)), uncapped -- one thread per (bin feature, node tile) runs the same `add_leaf`
  body over its tile's nodes and stores the calcer's two running sums; `pw_sym_score_fold_kernel`,
  the one-launch kernel's grid (`result_blocks`, <= 32) and grid-stride loop -- combines the tiles
  in order (`ScoreCalcer.combine`, new), `get_score`, the same fused gain, the unchanged
  `_block_argmin_and_store`. Dispatcher `find_optimal_split_sym` mirrors `find_optimal_split_plain`'s
  five arms.
- `pointwise_scores_calcer.mojo`: under the alias the helper owns the partials buffer
  (`d_sym_partials`, sized for `1 << (max_depth - 1)` parts) and `compute_optimal_split_dev` routes
  to the tiled pass. The record count stays `result_blocks`, so main's fused
  `pw_resolve_pack_bins_kernel` (DEVIATION 3111, which re-runs the record fold in every thread of
  the bin update) is untouched. Under other builds the only difference is a 1-float pool buffer
  never launched against.

Bits: the per-candidate score is the same sum in a different order (tile partials added instead
of one left-to-right chain over nodes), so FAST only. The cosine `DenumSqr` seed rides with tile 0.

Risky compile sites: struct field assignment inside kernels (`calcer.score = ...`,
`calcer.denum_sqr = ...`); the `comptime if ... else` initialising `self.d_sym_partials`;
`block_idx.y` in `pw_sym_score_partial_kernel`.

Request: `tsym-level-istella` (gbdt, gbdt-symmetric, istella, 2 pairs). taxi only after istella wins.

## Pass 2: per-tree fixed cost (SymmetricTree-1000: CatBoost 55 s / 121 s)

Per-tree sequence on the pointwise SymmetricTree arm (`doc_parallel_boosting.mojo`, Plain, one
permutation), host waits in CAPS:
1. gradient launch + `fv` copy to host (enqueued, no wait); mags DRAIN only when the fixed-point
   histogram rows are live (`_needs_magnitudes`, IDENTICAL columns);
2. structure search: one DRAIN per tree (DEVIATION 207; level loop blind, hist over all parts in
   one launch, winner/pack/bin-update fused on main as DEVIATION 3111);
3. `compute_bins_for_model`: ten per-tree buffers + one DRAIN to keep them alive;
4. `partition_from_bins`: bins DOWNLOADED (DRAIN), two host passes over n_rows (counting sort),
   row order UPLOADED (DRAIN), three per-tree allocations;
5. `make_bin_optimized_oracle`: per-tree host staging (h_leaves/h_shift/h_fv/h_part_stats) when no
   host scratch is passed; a DRAIN only with sample weights;
6. Newton walker (`descent_helpers.mojo`): its own derivative/value DRAINS per iteration (not
   touched here);
7. `_estimate_and_apply` tail: `h_est` upload, `add_model_value_kernel`, one DRAIN to keep the
   oracle's temporaries alive.

Switches (FAST + Apple only, default OFF):

- `-D MOJOLEARN_SYM_DEVICE_PARTITION` (`SYM_DEVICE_PARTITION`, doc_parallel_boosting.mojo): steps 3-4
  become pooled level tables (`ObliviousBinsTables`, new, doc_parallel_leaves_estimator.mojo) +
  `DeviceLeafPartitioner.partition` (main's DEVIATION 2551 stable radix sort, same integers): one
  DRAIN of 2*n_leaves+1 words, no host row passes, nothing allocated per tree.
- `-D MOJOLEARN_SYM_NO_TAIL_DRAIN` (`SYM_NO_TAIL_DRAIN`): `_estimate_and_apply(tail_drain=False)` from
  the symmetric call sites; the oracle's host staging comes from `est_ws.arena_host` (pooled, built
  with the buffer-taking `OracleHostScratch` init), and the tail drain is skipped when the device
  scratch is pooled, there are no sample weights and the method is not Exact. Risk: any temporary
  the Newton walker still has in flight at return; the gate is conservative but unverified here.
- Both plus `SYM_DEVICE_LEVEL` on `gbdt-symmetric-1000` as `tsym-all-1000-istella`.

Not done (next): leaf values resident on the device with one readback per fit (the Newton walker's
per-iteration drains, step 6), and reusing the searcher's own `subsets` partition for the learn
permutation instead of recomputing bins.
