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

## NOT REACHED, and what it would take

| piece | why it exists | what would reach it |
|---|---|---|
| ~~`node_feature_range_kernel`~~ | | **WIRED 2026-08-21** by `train_classification_device`, along with the score pass, the finalize pass and the split reduction. `device_tree_check` grows a tree on the GPU and it is bit-identical to the host's. |
| ~~`build_workload_info`~~ | | **WIRED** by the same. It still sits in `builder_kernels_impl.mojo` rather than `builder.mojo` for lane-ownership reasons; moving it is a merge-time action |
| `fixed_point.mojo` (`choose_scale`, `quantize`, `mse_proxy_exact`) | DEVIATION 135's ruling | the device regression score pass. The host oracle runs at `Float64` and does not need it, so today only `fixed_point_check` calls it |
| `ProxyImpurityExact` / `CompareProxyExact` | DEVIATION 144-145 | reached by `node_split_random_gini`, so this row is WIRED — kept here to record that the DEVICE reduction must also honour it, and does not exist yet |

## Deliberately not built

Tracked in `UNPORTED.tsv` rather than here: `quantiles.cuh` (the file this
formulation exists to delete), the bootstrap row sampler, `sample_weight`,
NaN/`missing_go_to_left`, four unported criteria, and cuML's dead
`adaptive_sample_kernel`.
