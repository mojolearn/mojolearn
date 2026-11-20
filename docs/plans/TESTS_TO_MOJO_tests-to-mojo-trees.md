# Tests to Mojo, lane tests-to-mojo-trees (2026-11-16)

The 15 `L-trees` rows of `TESTS_TO_MOJO_2026-11-15.md`, read function by function. Nothing was compiled or run
(lane rule). Verdicts as in `TESTS_TO_MOJO_tests-to-mojo-classical.md`: KEEP-BOUNDARY, KEEP-NOPORT, COVERED,
SOURCE-TEXT, PORT.

Totals: 15 files; 0 deleted, 5 shrunk, 10 kept unchanged; 66 Python lines removed, 17 added (docstrings and
comments). Two new checks and two new units in an existing one; both new checks are registered in
`checks/native_tests/registry.mojo` in one block marked `lane tests-to-mojo-trees`, and
`tools/ops/native_checks_report.py plan --column cpu` lists them (166 cpu checks).

| file | lines | result | removed | added | function verdicts |
|---|---:|---|---:|---:|---|
| `test_cpu_training_e3_gbdt_trees.py` | 161 | kept | 0 | 0 | KEEP-NOPORT 1 (rng(0) inputs; the save/load round trip is boundary; its repeat already moved to `gbdt_public_fits`) |
| `test_cpu_training_forest_variants.py` | 187 | kept | 0 | 0 | KEEP-NOPORT 1 |
| `test_cpu_training_gbdt_ordered.py` | 246 | shrunk (orphan source-text comments) | 15 | 0 | KEEP-NOPORT 1, SOURCE-TEXT 3 comment blocks |
| `test_dart_buffer_lifetime.py` | 17 | kept | 0 | 0 | KEEP-NOPORT 1 |
| `test_feature_freq_experimental_surface.py` | 86 | moved to `gbdt_feature_freq_experimental`; shrunk | 3 | 7 | KEEP-BOUNDARY 5 (two shrunk; their numeric arms ported) |
| `test_forest_inference_engine.py` | 191 | kept | 0 | 0 | KEEP-BOUNDARY 6, KEEP-NOPORT 2 |
| `test_forest_list_label_admission_cost.py` | 39 | kept | 0 | 0 | KEEP-BOUNDARY 1 |
| `test_gbdt_host_modes.py` | 193 | moved to `gbdt_host_archive_text`; shrunk | 15 | 4 | KEEP-BOUNDARY 1, TOOLING 1, KEEP-NOPORT 1, PORT 1 |
| `test_gbdt_pair_logit.py` | 99 | moved to `gbdt_public_fits` (unit 24); shrunk | 5 | 2 | KEEP-BOUNDARY 2 (one shrunk) |
| `test_gbdt_tree_metadata.py` | 52 | kept | 0 | 0 | KEEP-BOUNDARY 3 (numerics already `gbdt_tree_metadata`) |
| `test_gbdt_yeti_rank.py` | 89 | moved to `gbdt_public_fits` (unit 25); shrunk | 28 | 4 | KEEP-BOUNDARY 2, PORT 1 |
| `test_host_tree_native.py` | 182 | kept | 0 | 0 | KEEP-BOUNDARY 10 |
| `test_native_tree_lifecycle.py` | 181 | kept | 0 | 0 | KEEP-BOUNDARY 9 |
| `test_rf_class_weight.py` | 60 | kept | 0 | 0 | KEEP-BOUNDARY 2, KEEP-NOPORT 3 |
| `test_trees_repeat.py` | 153 | kept | 0 | 0 | KEEP-NOPORT 1 |

## New checks and units

| check | moved from | calls |
|---|---|---|
| `gbdt_feature_freq_experimental` (new; cpu 1 unit, gpu 2) | `test_feature_freq_experimental_surface.py::test_experimental_feature_freq_validates_and_forwards` (non-empty text, two fits same text, learning_rate 0.5 differs) and `::test_experimental_feature_freq_forwards_weights` (unit weights equal unweighted, [1,2,3,4] differ; device column only, as the Python test skipped the CPU install) | `gbdt_fit_two_level_feature_freq_run` (`gbdt/mojolearn_gbdt_host_run.mojo` cpu; `gbdt/mojolearn_gbdt_run.mojo` gpu under `_gbdt_device_plan()`), X column-major as the door admits it |
| `gbdt_host_archive_text` (new; host code on both columns) | `test_gbdt_host_modes.py::test_ctr_and_tensor_ctr_records_without_tables_refuse` (whole function) | `core.host_archive_gbdt.host_archive_gbdt_text`, the entry of the `host_tree_parse` door operation |
| `gbdt_public_fits` unit 24 | `test_gbdt_pair_logit.py::test_pairs_validation`, its "no pairs_weight is unit weights" arm | `pair_logit[I](3)`: two trees, fixture pairs, no weights vs 1894 unit weights, same predict bytes |
| `gbdt_public_fits` unit 25 | `test_gbdt_yeti_rank.py::test_learns_on_uneven_queries` (whole function, with `_ndcg`) | `yeti_learns[I]`: 240 finite predictions, every learn-curve value 0, NDCG above the constant's (the NDCG restated in Mojo, stable tie order as Python's `sorted`) |

`GBDT_COMMON_UNITS` goes from 24 to 26 (both columns).

## Kept, and what blocks the move

| file | function | verdict | blocker |
|---|---|---|---|
| `test_cpu_training_e3_gbdt_trees.py` | `test_non_symmetric_save_load_on_the_host_when_built` | KEEP-NOPORT | rng(0) inputs, no fixture; the property is the public save/load archive round trip (boundary) |
| `test_cpu_training_forest_variants.py` | `test_variants_fit_on_the_host_when_built` | KEEP-NOPORT | rng(1) inputs, no forest fixture generator; the forest fit has no typed check entry used by a registered check (`trees_entry_forest`, `host_entry_forest` are `main()` suites, not in `registry.mojo`) |
| `test_cpu_training_gbdt_ordered.py` | `test_ordered_save_load_and_ctr_predictions_on_the_host_when_built` | KEEP-NOPORT | save/load (boundary); the CTR predictions and the eval-curve-vs-predict loss need the host CTR predict door, which the check's `predict` helper does not reach |
| `test_dart_buffer_lifetime.py` | `test_binary_probability_rows` | KEEP-NOPORT | rng(0) inputs, no DART fixture; DART fit entry not used by a registered check (`trees_entry_dart` is an unregistered suite) |
| `test_feature_freq_experimental_surface.py` | `..._validates_and_forwards` (shrunk), `..._forwards_weights` (shrunk) | KEEP-BOUNDARY | what stays is the door: fit returns the estimator, `n_features_in_`, and that the door forwards `learning_rate` and `sample_weight` (the check calls the entry with them directly) |
| `test_feature_freq_experimental_surface.py` | `..._refusals`, `..._weight_refusals`, `..._refuses_constant_numeric_column` | KEEP-BOUNDARY | refusals by name raised in the door (`bindings/trees_public_gbdt_door.mojo` `feature_freq_fit`) on admitted Python values; no typed entry |
| `test_forest_inference_engine.py` | 8 functions | KEEP-BOUNDARY / KEEP-NOPORT | params, archive format tags, pickle, `set_params` unfitting, monkeypatched routing; the numeric ones (`test_classifier_labels_are_the_argmax_of_the_vote`, `test_identical_auto_keeps_sequential_bits_and_archive`) draw rng(0) inputs |
| `test_forest_list_label_admission_cost.py` | `test_list_labels_fit_within_twice_the_numpy_fit` | KEEP-BOUNDARY | Python-list label admission cost; rng inputs |
| `test_gbdt_host_modes.py` | `test_archive_names_and_losses_refuse_before_any_binding_loads` | KEEP-BOUNDARY | archive members and the estimator-name check of `HostGBDT.from_file` |
| `test_gbdt_host_modes.py` | `test_gate_coded_transform_is_the_documented_rule` | TOOLING | tests `tools/forest_host_gate.py` |
| `test_gbdt_host_modes.py` | `test_four_modes_predict_through_host_model_when_built` | KEEP-NOPORT | rng(0) inputs; `host_model` on saved archives (boundary) |
| `test_gbdt_pair_logit.py` | `test_refusals_by_name`, `test_pairs_validation` (shrunk) | KEEP-BOUNDARY | door refusals and the pairs formats (tuples, lists, int array) admitted by the door |
| `test_gbdt_tree_metadata.py` | 3 functions | KEEP-BOUNDARY | dtypes through `np.asarray`, no GPU binding loaded, `ValueError` type, unfitted refusal; the values are already `gbdt_tree_metadata` |
| `test_gbdt_yeti_rank.py` | `test_l2_default_is_zero_for_yeti_rank_only` | KEEP-BOUNDARY | the default `l2_leaf_reg=None` is resolved to 0 by the fit door; the check's fit request restates that rule itself, so a Mojo port would test the check, not the door |
| `test_gbdt_yeti_rank.py` | `test_refusals_by_name` | KEEP-BOUNDARY | door refusals by name |
| `test_host_tree_native.py` | 10 functions | KEEP-BOUNDARY | every assert is `audit_native_call` (no Python frames) and compares five Python input layouts (list, Fortran, strided, big-endian) against the hand-written `LegacyHostForest`/`LegacyHostGBDT`; a float32-only Mojo port would delete nothing |
| `test_native_tree_lifecycle.py` | 9 functions | KEEP-BOUNDARY | constructor binding, `ClassLabels` hook refusal, `get_params`/`set_params`, not-fitted refusal, all under `audit_native_call`; the regressor repeat (3 asserts) runs through the resident Python owner |
| `test_rf_class_weight.py` | `test_unit_path`, `test_balanced_and_partial_mapping`, `test_fit_dispatch` | KEEP-NOPORT | rng(0) inputs; class_weight to sample_weight resolution is the door's (`class_weight_plan` is an unregistered suite) |
| `test_rf_class_weight.py` | `test_invalid_weights`, `test_invalid_option` | KEEP-BOUNDARY | `ValueError` refusals |
| `test_trees_repeat.py` | `test_every_entry_point_twice_on_gpu_and_cpu` | KEEP-NOPORT | rng(0) inputs over 30 public estimators in two child processes (`trees_door_run` is an unregistered suite) |

## Blockers to fix so more can move

1. Fixtures: the rng-drawn inputs of the forest, DART, class-weight and repeat tests need a committed generator
   (as `tools/gbdt_native_fixtures.py` does for GBDT) before a check can use the same bytes.
2. Registration: `host_entry_forest`, `trees_entry_forest`, `trees_entry_dart`, `trees_entry_gbdt`,
   `trees_entry_glue`, `trees_door_run`, `class_weight_plan`, `gp_label_owner` are `main()` suites never imported in
   `registry.mojo`, so they do not run in `native-checks.yml` and are no evidence for deleting a Python test. They
   need a `register()` and a registry line (their owner lane, not this one).
3. Door-resolved defaults and option forwarding (YetiRank `l2_leaf_reg`, FeatureFreq options, class_weight) have no
   typed entry below the Python fields; they stay Python boundary tests until the door's UNPACK hands a typed
   request to a RUN function a check can call.

## Removed

Python: the moved functions and asserts above, the `_ndcg` helper, the `parse_model_text` import of
`test_gbdt_host_modes.py`, and three orphan comment blocks of `test_cpu_training_gbdt_ordered.py` (pointers to
source-text checks deleted earlier). No product code changed. Verify and everything it imports are untouched.

## Product defects seen

None asserted. `test_trees_repeat.py` still prints the GPU and CPU digests side by side and says "never required to
agree"; under the current bits rule the host column must match the GPU columns within a version. The release
verification is what enforces that, so the test was left as it is.
