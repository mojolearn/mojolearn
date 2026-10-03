# lane/apple-fast-pairlogit: PairLogit and MultiClass under FAST on Apple (NEXT_PASS_TREES lanes 5 and 6)

The plan's families `trees-pairlogit` (lane gbdt-rank-pairlogit, istellarank) and `trees-multiclass` (lane
gbdt-multiclass, taximc) share this branch under separate defines. Written without a Mojo toolchain (cloud peer); the
first M3 build of `gbdt` (`bindings/build_gbdt.sh`, FAST) is the compile check. Every change is compiled under FAST +
Apple only and defaults OFF; IDENTICAL compiles main's code unchanged.

Why the board had no PairLogit number: the lane exists in every driver (`tools/bench_board.py` TREE_TASK_LANES,
`bench/speed/forest_speed_arm.py`, `tools/speed_gbdt_arm.py`), nothing refuses it; it was never queued on the M3
(`docs/apple-fast/m3/results.txt` has no `gbdt-rank-pairlogit` line). The first line below is also its first FAST time.

| switch | kind | site | what it changes under FAST on Apple |
|---|---|---|---|
| `-D MOJOLEARN_PAIRLOGIT_GROUP_FUSED=1` | define, `PAIRLOGIT_GROUP_FUSED` comptime in `gbdt/targets/kernel/pair_logit_group.mojo` | `pair_logit_group_kernel` (new), `make_pairwise_group_buffers` and the dispatch at the top of `launch_pair_logit_with` (`gbdt/targets/kernel/pair_logit.mojo`), the pair-buffer and partial-count sites of `doc_parallel_boosting.mojo`, the pair generation and weight fill of `gbdt/train.mojo` | main generates the pair list on the host once per fit (every `first < second` of a query with different grades, ~1e8 pairs on Istella-S), uploads it with per-row endpoint lists, and every target call (the search call and each estimation evaluation, per tree) streams it twice: `pair_logit_pair_kernel` stores two floats per pair, `pair_logit_row_kernel` gathers them per endpoint. The define rebuilds the pairs from the grades on the device every call: one block per query group, a thread per document, the group's point and grades in shared-memory tiles, every pair's logistic term (the same arithmetic, exp/log shims, clamps and `ftz` sites) folded into the document's der/der2 in increasing j. No pair list, endpoint lists or per-pair scratch; the value partials are one per group and the magnitudes two per group (`part_blocks`, `fv_blocks`, `mag_blocks` sized to the group count). The host pair generation, ordering and `prepare_pairs` are skipped for generated pairs; a once-per-fit setup kernel writes the grades, group weights (the first row's weight) and per-row pair weights (`w * count`) and one readback over the groups applies the reference's refusals (per-group cap, constant target, total weight) and sums `PairsTotalWeight` in Float64. A caller's own pair list keeps main's path even under the define (`n_pairs >= 0`). |
| `-D MOJOLEARN_PAIRLOGIT_EST_REUSE=1` (with the first) | define, `PAIRLOGIT_EST_REUSE` comptime, same file | `pair_logit_group_reuse_kernel`, `launch_pair_logit_estimation_from_search` (`pair_logit.mojo`), the PairLogit arm of the oracle's evaluation (`pointwise_oracle.mojo`, `elif self.pairs`) | the YetiRank `YETI_EST_REUSE_SEARCH` model: the search call (no inverse) also keeps its per-row der/der2 and per-group value partials in the group layout's accumulators; the leaf estimation's first evaluation, at the same point (the oracle's search-point flag, cleared by any move that shifts a leaf), scatters them to bin order in one launch instead of re-enumerating the pairs. PairLogit draws nothing, so the sample is the same too; only the fold can move FAST bits. |
| `-D MOJOLEARN_MULTICLASS_HESSIAN_BATCH=1` | define, `MULTICLASS_HESSIAN_BATCH` comptime in `gbdt/methods/leaves_estimation/pointwise_oracle.mojo` | `multilogit_second_der_all_rows_kernel` (new, `gbdt/targets/kernel/multilogit.mojo`), `_oracle_multi_planes`, the pool's `take`, the factory's `multi_planes`, the batched branch of `_write_blocked_second_derivatives` | main's DEVIATION 75 loop runs K rows per estimation iteration, each a second-derivative launch, a `compute_partition_stats`, a device-to-host copy and a `synchronize()` (K = 4 on taxi, 5 on istella). The define writes every row of the lower triangle in one launch into K (K + 1) / 2 planes of `d_multi_der` (the reference's `reducedHessianGpu` slices), one reduce over all planes, one copy, one wait, then the same per-row mirror from each row's slot. The scratch width (`multi_planes`: pool key, factory, host staging) is the triangle for MultiClass up to 8 classes; wider fits keep main's loop. Element values are the row kernel's bit for bit; the wider reduce can move FAST bits. The multiclass value/der pass is already one launch over classes (`launch_multilogit_value_and_der`), so nothing else was batched. |

Bits: FAST only (IDENTICAL compiles none of it). PairLogit's per-row fold takes the j order where main takes increasing
pair index, and the per-row pair weight is `w * count` where main folds `count` copies of `w`; both are fixed
sequential orders, so FAST runs are repeatable. A FAST-only edge: a target that varies across groups but is constant
within every group raises the constant-target sentence where main raises the total-weight sentence (both refusals).

Request lines (`pairlogit.txt`): the group kernel on istellarank first, the reuse on top of it, the multiclass batch on
taximc (istella multiclass is capped/unknown on the M3; its line follows a taxi win).

## Risky compile sites

- `gbdt/targets/kernel/pair_logit_group.mojo`: `stack_allocation[..., address_space = AddressSpace.SHARED]()` and
  `barrier()` as in `yeti_rank.mojo`; `comptime if store_acc:` inside `if tid == 0:`; `pinned_block_sum` called by every
  thread outside the row branches; `acc.unsafe_ptr() + offset` passed to `MutPointer[Float32, MutAnyOrigin]` kernel
  parameters (the idiom of `doc_parallel_boosting.mojo:3366`); no `MutAnyOrigin` casts added.
- `gbdt/targets/kernel/pair_logit.mojo`: `make_pairwise_group_buffers` returns inside `comptime if` and raises in the
  else; `n_pairs` negative as the layout's mark (`blocks()` branch); `String(Int(pairs_q))` from a Float32;
  `launch_pair_logit_group[estimation, second_order, PAIRLOGIT_EST_REUSE]` takes a comptime Bool as its third parameter.
- `gbdt/train.mojo`: `pass` as the body of a `comptime if` arm (the generated-pairs branch); the host weight fill
  duplicated verbatim in the else arm.
- `gbdt/methods/leaves_estimation/pointwise_oracle.mojo`: `dims = (dims[0], _oracle_multi_planes(...))` reassigns a
  `Tuple[Int, Int]`; `return` from inside the `comptime if MULTICLASS_HESSIAN_BATCH:` branch of
  `_write_blocked_second_derivatives`; `self.pairs.value()` passed as `mut` twice in sibling branches (main's idiom).
- `gbdt/targets/kernel/multilogit.mojo`: `var p_row: Float32` assigned in both branches (the row kernel's idiom);
  `slot` accumulated across the row loop.
