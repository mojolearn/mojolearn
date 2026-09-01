# Built here, not yet reached

Rule 3: **a ported file that no caller reaches is not done.** `mojotrees`
accumulated four fully implemented, documented, tested stages that no default
fit could reach, and it took a day to find them. This file exists so this lane
cannot repeat that quietly. **Audited by grep, not by memory.**

The root `UNWIRED.md` covers `gbdt/`; this one covers `extratrees/` only.

## Wired and driving

| piece | reached by | guard |
|---|---|---|
| `NodeQueue`, `max_nodes` | `train_classification` / `train_regression` | `builder_check`, `tree_check` |
| `node_split_random_gini` / `_mse` | both `train_*` | `host_splitter_check`, `tree_check` |
| `node_feature_min_max`, `node_feature_is_constant`, `draw_threshold` | `node_split_random_*` | `range_draw_check` |
| `partition_samples` | both `train_*` | `partition_check` |
| `sample_features` and its three-arm dispatch | both `train_*` | `feature_sampler_check`, and `tree_check`'s `max_features` case |
| `set_leaf_predictions_*` | both `train_*` | `leaf_check` |
| Gini and MSE objectives | `node_split_random_*` | `objectives_check` |
| `Split.update`, `split_not_valid` | `NodeQueue.push`, `node_split_random_*` | `split_check` |
| `SparseTreeNode`, `TreeMetaDataNode`, predict traversal | the forest | `flatnode_check`, `forest_check` |
| `PCGenerator`, `key_for`, `uniform_float` | `draw_threshold`, `sample_features` | `pcg_rng_check` vs a C++ oracle |
| `DecisionTreeParams`, `validity_check` | both `train_*`, both `fit_*` | `params_check` |
| `fit_classification` / `fit_regression` / the vote | `forest_check`, `quality_band_check` | both |
| `phase_setup_a_kernel` / `phase_setup_b_kernel` (DEVIATION 470), `leaf_kernel[zero_fill=True]` (471), `_stage_upload_if_changed` (472) | both `train_forest_*_device` twins, every search cycle / leaf tail / `stage_batch` call | `device_batched_check` A/B plus the per-kernel checks -- gates OWED, none run yet (see DEVIATIONS 470-472) |

Note on the three standalone init LAUNCHES after DEVIATION 470:
`node_feature_range_init_kernel`, `node_feature_score_init_kernel` and
`split_reduce_init_kernel` are no longer enqueued by any fit -- the fused
`phase_setup_a_kernel` / `_b_kernel` pair seeds instead (two kernels, not
one: Metal's 31-entry binding cap). This is NOT a Rule 3 trap: the fit still
reaches their BODIES (`range_init_seed_at`, `score_init_seed_cell_at` /
`_acc_at`, `split_reduce_seed_at` are imported by the fused kernel, one
spelling of each seed), and the standalone launches stay deliberately as the
isolated seeders of `range_kernel_check`, `score_kernel_check`,
`regression_score_check` and `split_reduce_check`.

## NOT REACHED, and what it would take

| piece | why it exists | what would reach it |
|---|---|---|
| ~~`node_feature_range_kernel`~~ | | **WIRED 2026-08-21** by `train_classification_device`, along with the score pass, the finalize pass and the split reduction. `device_tree_check` grows a tree on the GPU and it is bit-identical to the host's. |
| ~~`build_workload_info`~~ | | **WIRED** by the same. It still sits in `builder_kernels_impl.mojo` rather than `builder.mojo` for lane-ownership reasons; moving it is a merge-time action |
| ~~`fixed_point.mojo`~~ | DEVIATION 135's ruling | **WIRED** by the device regression score pass (deviation 206) and by the estimator's `quantize_labels` (deviation 188's closure); `fixed_point_check` and `device_regression_check` guard it |
| ~~`ProxyImpurityExact` / `CompareProxyExact`~~ | DEVIATION 144-145 | **WIRED** on the host by `node_split_random_gini`; the device reduction this row was held open for exists (deviations 189-190) and `split_reduce_check` guards it |
| cuML's `max_leaves` budget | `builder.cuh:89` (`IsExpandable`) and `:106` (the `break` in `Push`), transcribed at `builder.mojo:292-296` and `:341-345` | **HALF WIRED 2026-09-01.** It was FULLY UNREACHED until then and this file did not say so: `builder_check` set `params.max_leaves` directly, but `estimator.mojo`'s `resolve` pinned `params.max_leaves = -1`, so no fit on any arm could reach the budget. It now rides from `ExtraTreesConfig.max_leaves`, gated by `estimator_check`'s REACH assert on `plan.params.max_leaves` plus a `max_leaves=0` refusal arm. **The PYTHON half is still unreached** and deliberately so: the 22-slot params list is decoded in `bindings/_mojolearn_trees.mojo`, which this lane does not own, so `ExtraTreesClassifier` / `ExtraTreesRegressor` expose NO keyword for it rather than accepting one that would do nothing. What would reach it: one slot appended to that list, `config.max_leaves = Int32(Int(py=params[22]))` beside the `max_leaf_nodes` line at `:160`, and the matching kwarg in `python/mojolearn/extratrees.py`. |

## Deliberately not built

Tracked in `NOT_IMPLEMENTED.tsv` rather than here: `quantiles.cuh` (the file this
formulation exists to delete), the bootstrap MASK behind `oob_score`,
`sample_weight`, NaN/`missing_go_to_left`, three unported regression
criteria, and cuML's dead `adaptive_sample_kernel`. (The bootstrap row
sampler and Entropy left this list on 2026-08-23, DEVIATIONS 460 / 459.)
sklearn's `max_leaf_nodes` joined that file on 2026-09-01 as a `not yet` and
LEFT IT THE SAME DAY as `IMPLEMENTED`. It is best-first growth, a second
growth mode (DEVIATION 466-469), fully wired from the Python keyword through
`bindings/_mojolearn_trees.mojo` slot 19 -- which already carried it -- to
`DecisionTreeParams.max_leaf_nodes`, and gated by `bestfirst_check`. It is
NOT the same thing as cuML's `max_leaves` in the row above and the two are
never aliased; note the asymmetry that row leaves standing, which is that
sklearn's budget reaches Python and cuML's does not, because the params list
carries a slot for one and not the other.
