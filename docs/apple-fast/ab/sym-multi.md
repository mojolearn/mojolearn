# lane/apple-fast-sym-multi: the next layer under MultiClass, PairLogit and YetiRank (Apple FAST)

Board lanes gbdt-multiclass (istellamc, taximc), gbdt-rank-pairlogit (istellarank), gbdt-rank-yetirank
(istellarank). Everything is compiled under FAST + Apple + its own define only and defaults OFF; IDENTICAL compiles
main's code unchanged. `-D MOJOLEARN_SYM_MULTI_ALL` turns every define below on at once. The per-iteration
profile the defines come from is `docs/apple-fast/notes/sym-multi.md`. Request lines: `sym-multi.txt` (the
deciding dataset first; one tag per lane x dataset x define, plus the compositions). Quality to hold: multiclass
mlogloss and accuracy, ranking ndcg10 and map, within run-to-run spread of the A arm (FSPEED-ACC lines).

## `-D MOJOLEARN_MC_CLASS_BATCH_EST` (gbdt-multiclass)
Mechanism: the Newton leaf oracle's evaluation is two launches on main (value + first derivatives, then the K-row
Hessian), each followed by its own partition reduce, device-to-host copy and `synchronize`; the walker asks for both
at the SAME point (at `leaf_estimation_iterations` 1 that is the whole walk). `multilogit_est_fused_kernel`
(`gbdt/targets/kernel/multilogit.mojo`) writes the K-1 der planes and the K(K+1)/2 lower-triangle der2 planes in
one launch; the oracle (`pointwise_oracle.mojo`, `_write_multiclass_fused_evaluation`) reduces all
(K-1) + K(K+1)/2 columns in one `compute_partition_stats`, copies once, waits once, and serves the Hessian to
`write_second_derivatives` from `cached_der2` (no launch). Per tree: 3 launches and 1 of the oracle's 2 drains
removed. Expected: a few percent per tree (drains cost ~0.2 ms plus the queue bubble on Apple). Risk: low; the
element values are main's kernels' (recomputed per use unless `..._DERIV` is also on), the wider reduce folds the
same per-row values per bin. Needs `MULTICLASS_HESSIAN_BATCH` (the FAST + Apple default) and K <= 8.

## `-D MOJOLEARN_MC_CLASS_BATCH_DERIV` (gbdt-multiclass)
Mechanism: main's multilogit kernels read the K-1 cursor planes three times per row (max, sum of exps, the der
loop) and the Hessian kernel recomputes `exp(approx_k - max) / sum` for every (row, column) of the lower triangle
(K(K+1)/2 + 2K exps per row: 15 at K=4, 25 at K=5). The `_reg` kernels load the K-1 approxes once into registers,
take the K-1 exps once, and reuse `p_k = exps[k] / se`; `routed_exp` on the same operands gives the same float,
so every element value is bit for bit main's. Applies to the search gradient (`doc_parallel_boosting.mojo`), the
estimation value/der and the Hessian pass (and, when both defines are on, inside the fused kernel). Expected:
smaller than EST (the kernels are a modest share of a tree; the win is the K^2 -> K exps in the Hessian pass).
Risk: none to quality (same values); register pressure at K=8 (36 unrolled triangle stores) is the only
performance risk, and the board's K is 4 and 5.

## `-D MOJOLEARN_PL_PAIRS_ONCE` (gbdt-rank-pairlogit)
Mechanism: the merged group kernel gives each document a thread that loops over every other document of its
query, so each unordered pair's exp, divide, clamp (and the winner's log) run twice, once from each endpoint. For
a group that fits one block, the once path runs floor((size-1)/2) rounds; in round r thread i pairs with
(i + r) mod size, a permutation of the threads, so every unordered pair is evaluated by exactly one thread (plus
the size/2 half round for even sizes). The computing thread folds its own side in registers and hands the other
side's der/der2 term to j through a shared slot only it writes that round; after a barrier j folds it in. Halves
the transcendental work of the fit's dominant kernel (~1e8 pairs per call on Istella). Groups wider than the block
keep the chunked loop. Expected: the largest win of the three PairLogit arms if the pair kernel dominates the
3.1 s. Risk: the per-document fold order changes (FAST bits move; fixed and repeatable, same terms), so ndcg/map
must be checked; 2 barriers per round (~50 rounds for a 103-document query).

## `-D MOJOLEARN_PL_GROUP_NARROW` (gbdt-rank-pairlogit)
Mechanism: the group kernel's block is 128 threads instead of 256. Istella's queries hold ~103 documents, so a
256-thread block leaves four of its eight SIMD groups idle through every tile loop and barrier; at 128 the idle
half is gone and twice the blocks fit a core. A group wider than 128 takes two chunks of the width-agnostic loop
(its O(size^2) work is split over fewer threads, so the define loses on wide groups). Same terms, same
per-document fold order (the j order), so bits do not move. Expected: a modest occupancy win on Istella; the
`pl-both` line tests it on top of `PL_PAIRS_ONCE`. Risk: a dataset with wide queries gets slower; nothing to
quality.

## `-D MOJOLEARN_YR_TASK_FUSED` (gbdt-rank-yetirank)
Mechanism: `launch_yeti_rank_with` is five launches per call: `compute_group_ids` (the same row -> query map every
call), the gather (estimation only), `compute_group_means`, the centering, the task kernel, the row scatter
(planes, zero value partials, magnitudes). `yeti_rank_task_fused_kernel` (`gbdt/targets/kernel/yeti_rank.mojo`)
reads the point itself (through the inverse order in estimation), takes each query's unit-weight mean inside the
task block (a segmented sum over the task's 1024 positions in the sort-key scratch, ten steps; every task holds
whole queries), centers, runs the block kernel's draws, sort and pairs unchanged, and stores the planes, the
reuse accumulators, the zero value partials and one magnitude pair per task; the group ids are computed once per
fit. One launch per call instead of five; the task kernel itself (the dominant cost) is unchanged. Expected: a
few percent per tree (the removed launches are small kernels; Apple pays ~0.2 ms per launch plus per-live-buffer
cost). Risk: the query mean's fold order and the magnitude partition change (FAST bits move; the pairs are the
same arithmetic on the same centered values up to the mean's last bit), so ndcg/map must be checked. A fit with
more tasks than 256-row blocks (never on the board data) keeps the five launches.

## `-D MOJOLEARN_SYM_MULTI_ALL`
Every define above; the `all-*` lines measure the composition per lane.

## Not done: MC_HIST_MULTI (the brief's candidate 3)
The class dimension is already one launch (grid z = stat planes), so there is no host loop to remove; a hist pass
accumulating all K stat planes per compressed-index read would need K shared-memory accumulators where the
one-byte 8-bit kernel's pair already fills Apple's 32 KiB threadgroup budget at 254 borders
(`pointwise_hist2_one_byte_5bit.mojo:107`). Left for a lane that owns the histogram kernels.

## Risky compile sites (first M3 build is the compile check for the Metal side; the laptop compiled the .so)
- `multilogit.mojo`: `comptime for k in range(max_k): if k < eff:` loops over `InlineArray` registers; `var pk:
  Float32` assigned in both arms of a `comptime if reg`.
- `pointwise_oracle.mojo`: `return` from inside `comptime if MC_EST_ACTIVE` in the multi-dim arm; `MC_EST_ACTIVE`
  defined below its first use (module scope, like `MULTICLASS_HESSIAN_BATCH`).
- `pair_logit_group.mojo`: `stack_allocation` of the two hand-over slots inside the once branch; `once_done`
  breaks the chunk loop uniformly.
- `yeti_rank.mojo`: `sh_keys.unsafe_bitcast[Float32]()` / `[UInt32]()` views of the shared sort-key scratch;
  one `index_map` pointer serves both the load and the scatter (two mutable arguments from one allocation are
  refused as aliasing).
