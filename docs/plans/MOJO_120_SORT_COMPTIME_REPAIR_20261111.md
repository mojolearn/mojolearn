# Mojo 1.2 sort and compile-time control repair

Source-only repair after frozen integration `6ebb6c98ca` failed compilation.
Local merge `38afaafbd5` adopts that exact tree and is not a producer.
No compilation, tests, timing, identity or box execution ran in this lane.

The official Mojo 1.1 and 1.2 standard-library sources both take a sort
comparator as a runtime argument (`sort(span, comparator)`). The label-key
comparator now explicitly borrows its existing keys; the archive comparator
uses only its arguments. Both comparison expressions, the spans, default
sort stability, parallel partitions and subsequent merge/fold bodies remain
unchanged. The obsolete comparator decorators and parameter-form calls are
removed. The matching AGENTS rule now describes the supported interface.

There are 277 syntax-only control replacements across 37 present tracked
Mojo files: 117 `comptime if` and 160 `comptime for`. A source comparison
confirms every changed control file differs only by replacing its deprecated
decorator plus statement with the corresponding `comptime` statement. Conditions,
bounds and bodies are unchanged, including fixed folds and unrolling. The six
sparse-absent tracked files are historical benchmark output, recorded in the
inventory and untouched. No matching generator-template control sites remain.

Official references:

- [Mojo 1.2 sort source](https://github.com/modular/modular/blob/mojo/v1.2.0/Mojo/stdlib/std/builtin/sort.mojo)
- [Mojo 1.1 sort source](https://github.com/modular/modular/blob/mojo/v1.1.0/Mojo/stdlib/std/builtin/sort.mojo)
- [Compile-time evaluation](https://mojolang.static.modular.com/1.1.0/docs/manual/metaprogramming/comptime-evaluation/)
- [Mojo 1.2 closure captures](https://mojolang.org/docs/manual/functions/closures/)

Full before inventory, downloaded official sources with hashes, exact changed-path
proof and the separate OneClass numeric-refusal follow-up are in
`~/mojolearn-evidence/python-mojo-migration-20261110/native-upgrade-120/`:
`sort-comptime-before.json` and `sort-comptime-repair.json`.
Integration owns compile/runtime acceptance. No numerical result or performance
claim follows from the source comparison.

Changed control files:

- `bindings/class_weight_boundary.mojo`
- `bindings/gp_mapping_boundary.mojo`
- `bindings/label_weights_boundary.mojo`
- `checks/hist2_check.mojo`
- `checks/replicated_half_byte_check.mojo`
- `core/forest_inference.mojo`
- `gbdt/gpu_data/kernel/binarize.mojo`
- `gbdt/gpu_util/kernel/bootstrap.mojo`
- `gbdt/gpu_util/kernel/reorder_one_bit.mojo`
- `gbdt/methods/doc_parallel_boosting.mojo`
- `gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo`
- `gbdt/methods/greedy_subsets_searcher/kernel/hist_2_one_byte_5bit.mojo`
- `gbdt/methods/greedy_subsets_searcher/kernel/hist_2_one_byte_6bit.mojo`
- `gbdt/methods/greedy_subsets_searcher/kernel/hist_2_one_byte_7bit.mojo`
- `gbdt/methods/greedy_subsets_searcher/kernel/hist_2_one_byte_8bit.mojo`
- `gbdt/methods/greedy_subsets_searcher/kernel/hist_2_one_byte_base.mojo`
- `gbdt/methods/greedy_subsets_searcher/kernel/hist_binary.mojo`
- `gbdt/methods/greedy_subsets_searcher/kernel/hist_half_byte.mojo`
- `gbdt/methods/greedy_subsets_searcher/kernel/hist_one_byte.mojo`
- `gbdt/methods/greedy_subsets_searcher/kernel/hist_quantized_shared.mojo`
- `gbdt/methods/greedy_subsets_searcher/kernel/histogram_utils.mojo`
- `gbdt/methods/greedy_subsets_searcher/kernel/lane_sync.mojo`
- `gbdt/methods/greedy_subsets_searcher/kernel/point_hist_half_byte_template.mojo`
- `gbdt/methods/greedy_subsets_searcher/kernel/split_points.mojo`
- `gbdt/methods/leaves_estimation/pointwise_oracle.mojo`
- `gbdt/models/cuda/evaluator.mojo`
- `gbdt/targets/kernel/multilogit.mojo`
- `gbdt/targets/kernel/pointwise_targets.mojo`
- `neighbors/checks/ball_cover_canonical_order.mojo`
- `neighbors/checks/ball_cover_check.mojo`
- `neighbors/checks/ball_cover_knn_check.mojo`
- `neighbors/checks/radius_check.mojo`
- `neighbors/impl/ball_cover/registers.mojo`
- `neighbors/impl/detail/fused_l2_knn.mojo`
- `neighbors/impl/matrix/detail/select_warpsort.mojo`
- `xtrees/dart_device.mojo`
- `xtrees/shap_device.mojo`
