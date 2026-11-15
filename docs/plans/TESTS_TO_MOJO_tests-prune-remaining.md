# Tests to Mojo, lane tests-prune-remaining (2026-11-15)

The 131 files of the L-neural, L-parallel, L-spatial, L-models and L-trees rows of `TESTS_TO_MOJO_2026-11-15.md`
(`lanes.tsv`), read function by function by five Opus sub-lanes on disjoint file sets, following "How to move a
test" of the classical lane. Nothing was compiled or run. Evidence per family:
`~/mojolearn-evidence/briefs-20261115/tests-prune-remaining/<family>/result.md`.

Totals: 26 test files shrunk, 105 unchanged, 0 deleted; 1,183 Python lines removed, 154 added; one frozen-digest
file deleted (`tests/autograd_smoke_digests.json`, digests from 7939b679a2, replaced by a same-build twice-same-digest
assert). Three new checks registered in `checks/native_tests/registry.mojo`: `gbdt_tree_metadata`,
`mamba_decode_n_tokens`, `tokenizer_pretokenize_cuts`.

| family | files | shrunk | lines removed | added | checks |
|---|---:|---:|---:|---:|---|
| neural | 35 | 6 | 73 | 25 | none |
| parallel | 32 | 4 | 94 | 21 | none |
| spatial | 32 | 8 | 522 | 45 | none |
| models | 17 | 2 | 35 | 21 | mamba_decode_n_tokens, tokenizer_pretokenize_cuts |
| trees | 15 | 6 | 459 | 42 | gbdt_tree_metadata |

## neural

| file | lines | result | removed | added | function verdicts |
|---|---|---|---|---|---|
| python/mojolearn/tests/test_autograd_model.py | 275 | unchanged | 0 | 0 | all KEEP (boundary: admission, refusals, audit_native_call, ownership; numerics already in the named check or on random inputs) |
| python/mojolearn/tests/test_autograd_public.py | 324 | unchanged | 0 | 0 | all KEEP (guide blocks and docstring examples run, same-build twice equality) |
| python/mojolearn/tests/test_autograd_usability.py | 41 | unchanged | 0 | 0 | all KEEP (boundary: admission, refusals, audit_native_call, ownership; numerics already in the named check or on random inputs) |
| python/mojolearn/tests/test_checkpoint_native_loading.py | 64 | unchanged | 0 | 0 | all KEEP (boundary: admission, refusals, audit_native_call, ownership; numerics already in the named check or on random inputs) |
| python/mojolearn/tests/test_generator_draw_native.py | 230 | unchanged | 0 | 0 | all KEEP (independent Python Philox oracle + audits) |
| python/mojolearn/tests/test_generator_state_native.py | 45 | unchanged | 0 | 0 | all KEEP (boundary: admission, refusals, audit_native_call, ownership; numerics already in the named check or on random inputs) |
| python/mojolearn/tests/test_import_onnx_native.py | 281 | unchanged | 0 | 0 | all KEEP (Mojo reader vs Python oracle reader; inventory "source-text" was a false hit) |
| python/mojolearn/tests/test_layernorm_public_native.py | 36 | unchanged | 0 | 0 | all KEEP (boundary: admission, refusals, audit_native_call, ownership; numerics already in the named check or on random inputs) |
| python/mojolearn/tests/test_linalg_fast_bf16.py | 173 | unchanged | 0 | 0 | all KEEP (Apple fast arm, paused programme) |
| python/mojolearn/tests/test_linalg_lowbit.py | 138 | unchanged | 0 | 0 | all KEEP (boundary: admission, refusals, audit_native_call, ownership; numerics already in the named check or on random inputs) |
| python/mojolearn/tests/test_lowbit_weights.py | 190 | unchanged | 0 | 0 | all KEEP (boundary: admission, refusals, audit_native_call, ownership; numerics already in the named check or on random inputs) |
| python/mojolearn/tests/test_model_config_native.py | 65 | unchanged | 0 | 0 | all KEEP (boundary: admission, refusals, audit_native_call, ownership; numerics already in the named check or on random inputs) |
| python/mojolearn/tests/test_neural_backward_native.py | 111 | shrunk (assert not hasattr(Mamba3Block, "_prefill_backward")) | 3 | 1 | private-name assert removed from test_mamba3_backward_is_native_and_matches_raw_abi; all 4 KEEP (raw-ABI oracle, refusals) |
| python/mojolearn/tests/test_neural_block_native.py | 70 | unchanged | 0 | 0 | all KEEP (boundary: admission, refusals, audit_native_call, ownership; numerics already in the named check or on random inputs) |
| python/mojolearn/tests/test_neural_inference.py | 378 | shrunk (test_binding_exports_no_training_entry deleted) | 28 | 0 | test_binding_exports_no_training_entry: SOURCE-TEXT (exact dir() manifest) deleted; 13 others KEEP (boundary, rows-alone/ragged/decode same-build relations on random inputs) |
| python/mojolearn/tests/test_neural_public_coverage_p1.py | 331 | unchanged | 0 | 0 | all KEEP (boundary: admission, refusals, audit_native_call, ownership; numerics already in the named check or on random inputs) |
| python/mojolearn/tests/test_neural_repeat.py | 220 | unchanged | 0 | 0 | KEEP (in-process second-call continuity across columns) |
| python/mojolearn/tests/test_neural_repeat_calls.py | 84 | unchanged | 0 | 0 | all KEEP (boundary: admission, refusals, audit_native_call, ownership; numerics already in the named check or on random inputs) |
| python/mojolearn/tests/test_ops_graph.py | 19 | unchanged | 0 | 0 | KEEP (drives _verify_graph; verify is not ours) |
| python/mojolearn/tests/test_ops_program.py | 308 | shrunk (test_program_guide_block deleted: duplicate of test_autograd_public.test_guide_program) | 17 | 0 | test_program_guide_block: duplicate deleted; 10 others KEEP (eager==program relations on random data, refusals) |
| python/mojolearn/tests/test_ops_session.py | 83 | unchanged | 0 | 0 | all KEEP (boundary: admission, refusals, audit_native_call, ownership; numerics already in the named check or on random inputs) |
| python/mojolearn/tests/test_optimizer_native_moment_assignment.py | 132 | unchanged | 0 | 0 | all KEEP (boundary: admission, refusals, audit_native_call, ownership; numerics already in the named check or on random inputs) |
| python/mojolearn/tests/test_optimizer_native_owner_copy.py | 63 | shrunk (two retired private-field-name absence asserts) | 2 | 0 | 3 tests KEEP (copy/pickle/restore behaviour) |
| python/mojolearn/tests/test_optimizer_native_scalar_constructors.py | 118 | unchanged | 0 | 0 | all KEEP (boundary: admission, refusals, audit_native_call, ownership; numerics already in the named check or on random inputs) |
| python/mojolearn/tests/test_program_constructor_native.py | 106 | unchanged | 0 | 0 | all KEEP (boundary: admission, refusals, audit_native_call, ownership; numerics already in the named check or on random inputs) |
| python/mojolearn/tests/test_sequence_optimizer_public_native.py | 192 | unchanged | 0 | 0 | all KEEP (boundary: admission, refusals, audit_native_call, ownership; numerics already in the named check or on random inputs) |
| python/mojolearn/tests/test_sequential_native_checkpoint.py | 258 | shrunk (find_spec package-layout assert) | 3 | 0 | 8 tests KEEP (checkpoint round trip, poisoned-helper audit, refusals) |
| python/mojolearn/tests/test_sequential_native_surface.py | 119 | unchanged | 0 | 0 | all KEEP; _program/_table literal encodings are the only oracle of the native description builder (see missing entries) |
| python/mojolearn/tests/test_small_mlp_surface.py | 410 | unchanged | 0 | 0 | all KEEP (inventory "source-text" was checkpoint-file bytes: public file contract) |
| python/mojolearn/tests/test_training_array_metadata_contract.py | 73 | unchanged | 0 | 0 | all KEEP (boundary: admission, refusals, audit_native_call, ownership; numerics already in the named check or on random inputs) |
| python/mojolearn/tests/test_training_scalar_buffer_native.py | 293 | unchanged | 0 | 0 | all KEEP (boundary: admission, refusals, audit_native_call, ownership; numerics already in the named check or on random inputs) |
| python/mojolearn/tests/test_tsa_resident_state.py | 147 | unchanged | 0 | 0 | all KEEP (boundary: admission, refusals, audit_native_call, ownership; numerics already in the named check or on random inputs) |
| python/mojolearn/tests/test_vjp_rank_native.py | 29 | unchanged | 0 | 0 | all KEEP (boundary: admission, refusals, audit_native_call, ownership; numerics already in the named check or on random inputs) |
| python/mojolearn/tests/test_weight_import.py | 294 | unchanged | 0 | 0 | all KEEP (boundary: admission, refusals, audit_native_call, ownership; numerics already in the named check or on random inputs) |
| tests/test_autograd_smoke.py | 77 | shrunk (frozen 7939b679a2 digest assert -> same-build two-fit equality; digests json deleted) | 20 | 24 | test_mlp_fit_default_path, test_lstm_fit_default_path: FROZEN-OLD assert replaced by twice-same-bytes; rest KEEP |
| tests/autograd_smoke_digests.json (data, not in list) | 12 | deleted | 12 | 0 | frozen digests with no reader left |

#### New checks

None. No test in this set has a typed run entry plus a literal fixture that is not already in a registered check: the numeric parts named in docstrings (training_neural_inference, sequence_optimizer_*, core_ops_session_*, core_ops_program_*, core_vjp_rank_*, theta_state_*, bf16_conversion/bf16_linalg/int8_linalg, training_autograd_model, training_autograd_usability) moved earlier; what remains is boundary or uses np.random inputs.

| check | moved from | entries called |
|---|---|---|
| (none) | | |

#### Missing typed entries

- Sequential description builder host run entry taking (layer table, n_features, sample shape) -> program words and option table (test_sequential_native_surface `_program`/`_table`/`_options` literals).
- Generator draw host run entry taking (seed, stream, offset, n, kind, low/high or mean/std, p) -> f32 out ptr and stream advance (test_generator_draw_native Philox oracle).
- ONNX reader host run entry taking (file bytes ptr, n) -> typed GraphIR owner, and resolve taking that owner (test_import_onnx_native, now compared against the Python oracle reader).
- Sequential weight-import host run entry taking (layer table, named f32/f16/bf16 buffers + shapes) -> params ptr (test_weight_import word checks).

#### Product defects seen

None.

## parallel

| file | lines | result | removed | added | function verdicts |
|---|---|---|---|---|---|
| test_autoarima_search_ownership.py | 58 | unchanged | 0 | 0 | 2 KEEP: public subprocess lifetime checks, random inputs |
| test_cache_release_native.py | 221 | unchanged | 0 | 0 | 13 KEEP: DeviceCache/resident Python-object lifecycle, audit one-call, refusals |
| test_cache_stats_native.py | 320 | unchanged | 0 | 0 | 12 KEEP: hook refusal, builtin poisoning, TLS, audit one-call |
| test_cache_upload_native.py | 417 | unchanged | 0 | 0 | 12 KEEP: id_of key/counter admission, hook refusal, real upload ownership |
| test_classical_partition_native.py | 60 | unchanged | 0 | 0 | 3 KEEP: opt-in parallel-vs-plain relations, random inputs, process pool |
| test_cpu_parallel_rbf_sampler.py | 74 | shrunk (unused host_surface import) | 1 | 1 | 2 KEEP (shard trace/bytes relation, fault negative control); random inputs |
| test_cross_vendor_coordinator_native.py | 175 | unchanged | 0 | 0 | 3 KEEP: socket peers, refcount ownership, subprocess peer |
| test_cross_vendor_native.py | 56 | unchanged | 0 | 0 | 3 KEEP: hashlib oracle of state_hash field order and view admission (boundary per resample_cross_vendor_fold docstring), refusals |
| test_cross_vendor_worker_native.py | 204 | unchanged | 0 | 0 | 4 KEEP: BatchFeed/LearningRateTable ownership, borrowed exporter, subprocess coordinator |
| test_distributed_program.py | 158 | unchanged | 0 | 0 | 3 KEEP: resume/topology/refusals with np.random inputs; numerics already in training_distributed_program |
| test_distributed_program_boundaries.py | 63 | unchanged | 0 | 0 | 1 KEEP: audit one-call lifecycle |
| test_meta_members_native.py | 256 | unchanged | 0 | 0 | 13 KEEP: public meta-estimator relations between public routes, random inputs |
| test_native_cv_gbdt_nb.py | 209 | unchanged | 0 | 0 | 7 KEEP: public CV vs explicit fits, hook refusal, cold arena context |
| test_native_cv_lda.py | 224 | unchanged | 0 | 0 | 9 KEEP: constructor/params, CV relation, hook refusal, dataframe names, label ancestry |
| test_native_cv_outputs.py | 56 | unchanged | 0 | 0 | 2 KEEP: string label placement, canonical split descriptor (hashlib oracle, same build) |
| test_native_cv_row_order.py | 80 | unchanged | 0 | 0 | 1 KEEP: independent SplitMix oracle for group permutation (row orders already in resample_cv_row_order) |
| test_native_stratified_connection.py | 66 | unchanged | 0 | 0 | 2 KEEP: loader poisoning, repeat-seed oracle |
| test_parallel_arima_serialization.py | 39 | unchanged | 0 | 0 | 1 KEEP: parallel fit saves the plain fit bytes (same build), random input |
| test_parallel_byte_borrowed_inputs.py | 181 | unchanged | 0 | 0 | 6 KEEP: buffer forms admitted in place, refusals, corpus manifest (GPU) |
| test_parallel_causal_lm.py | 460 | shrunk (uniform-sampling test: tokens frozen from an older build -> same-build generate/stream/EOS relations; unused import) | 7 | 18 | 1 FROZEN-OLD rewritten; 8 KEEP (worker-process integration, refusals, restore, Apple fast parent) |
| test_parallel_forecasting.py | 59 | unchanged | 0 | 0 | 4 KEEP: driver vs estimator bytes (same build), refusals, random input |
| test_parallel_mlp_native.py | 76 | unchanged | 0 | 0 | 3 KEEP: snapshot ownership/publication, target refusal; numerics already in training_small_mlp; model_fixture imported by 2 files |
| test_parallel_model_selection.py | 235 | unchanged | 0 | 0 | 13 KEEP: worker-pool trace, refusals, completed-capture archives |
| test_parallel_neural_native.py | 325 | unchanged | 0 | 0 | 4 KEEP: whole training via child audits, rollback, Samba replay; independent_gradient imported elsewhere |
| test_parallel_pool_numpy_parity.py | 65 | unchanged | 0 | 0 | 2 KEEP: GPU driver vs plain fit/predict bytes |
| test_parallel_shard_payloads.py | 124 | unchanged | 0 | 0 | 4 KEEP: forest/radius drivers vs plain bytes, random input |
| test_parallel_trainers_coverage.py | 224 | unchanged | 0 | 0 | 10 KEEP: split/device fold bits vs train_step, refusals, checkpoints |
| test_parallel_worker_raw_results.py | 91 | shrunk (deleted test of tests-tree helper public_method_result, which no product code calls, plus unused fakes) | 42 | 1 | 1 deleted (tests test code); 1 KEEP (GPU user-vs-plain predict relation) |
| test_partitioned_program.py | 128 | unchanged | 0 | 0 | 3 KEEP: checkpoint resume, physical pipeline, admission; numerics already in core_partitioned_program |
| test_public_model_selection_verifier_gaps.py | 129 | unchanged | 0 | 0 | 6 KEEP: public model-selection relations vs NumPy oracles, random inputs |
| test_runtime_classical_partition.py | 99 | shrunk (deleted 2 manifest/owner-layout tests: HOST_IMPORT_ALIASES spec count/ABI, binding attribute identity; dropped legacy-owner identity assert) | 44 | 1 | 2 SOURCE-TEXT deleted; 3 KEEP (public fit/predict/archive audit, tree residency stats, input-protocol refusal) |
| test_x_cnn_pool.py | 120 | unchanged | 0 | 0 | 2 KEEP: ORACLE, NumPy reference vs torch (outside library) |
| **total** | | 4 changed, 28 unchanged | 94 | 21 | |

#### New checks

| check | moved from | entries called |
|---|---|---|
| (none) | | no test in this set has a typed run entry with a literal fixture that is not already in a registered check |

#### Missing typed entries

- parallel forecast partition driver: host run entry (model owner, start, end, series_per_shard) -> out ptr for ARIMA predict/forecast and Holt-Winters forecast/predict(index), without the native process pool
- parallel shard driver: host run entry (x ptr, n, d, rows_per_shard, RBFSampler owner) -> out ptr for `rbf_rows`, without the pool
- parallel forest fit by tree range: host run entry (x ptr, n, d, label codes ptr, class count, trees_per_shard, forest options) -> forest owner, without the pool
- ParallelQueries radius_neighbors: host run entry (model owner, q ptr, n, rows_per_shard, return_distance, sort) -> CSR (indptr, indices, distances) ptrs
- cross_vendor state_hash: host run entry (field ptrs, byte lengths, field count) -> sha256 words, plus a Mojo sha256 oracle (today the field order is checked against hashlib)
- parallel causal LM layer owner: host run entry (layer range, ids ptr, b, t, state owner) -> logits ptr, without worker processes
- msel parallel cross_val_score: host run entry (estimator descriptor, x ptr, y ptr, fold index ptrs, devices) -> score ptr, without worker processes
- meta-estimator vote/OVR/MultiOutput predict and score: host run entry over (member output ptrs, weights, classes) -> out ptr

#### Product defects seen

- none

#### Leftovers (outside this file set)

- `python/mojolearn/tests/_parallel_pool.py` `public_method_result` has no caller left (only comments in tools/public_runtime_inventory.py); not in this lane's file list, so not deleted.
- `python/mojolearn/host_surface.py` names `test_runtime_classical_partition.py` as `gate=` of the HOST_IMPORT_ALIASES specs; the spec-identity test it pointed at is deleted (the import-time ABI check in `_host_import_aliases.connect` remains). Product file, not touched.

## spatial


Branch `lane/tests-prune-remaining-spatial` (local, not pushed), base 7dcd11184c, head 13dda78078. 32 files read, 8 changed.
Python lines: 522 removed, 45 added (`git diff --numstat 7dcd11184c HEAD`). No compile, no tests run.

| file | lines | result | removed | added | function verdicts |
|---|---|---|---|---|---|
| test_arima_exog.py | 322 | shrunk (7 source-text/manifest tests) | 109 | 16 | DELETE validate_order_takes_a_regressor_count, one_file_permutes_the_exog_layout, the_device_reaches_the_exog_arms, the_host_oracle_restates_them, the_manifest_declares_the_lanes, the_unimplemented_row_is_gone, the_saved_format_is_versioned (source text of .mojo/.py, host_surface, NOT_IMPLEMENTED.tsv); ADDED to exog_fit_forecast_and_save: public fit with 18 regressors refused naming EXOG_MAX (the behaviour behind the deleted validator text; negative n_exog has no public form); KEEP the 3 runtime tests (refusals, save/load round trip, output-store save) |
| test_arima_parameter_lifetime.py | 75 | unchanged | 0 | 0 | KEEP (owner lifetime in a child; expected bits are the same fit's params_, not an older hash) |
| test_centroid_fit_native_boundary.py | 92 | unchanged | 0 | 0 | KEEP all 5 (refusal order, owner preservation, pickling, separate storage) |
| test_cnn_max_gradient_ownership.py | 35 | unchanged | 0 | 0 | KEEP (public SAGEConv max fwd/bwd ownership against an independent tie scatter; a public-path lifetime regression, the op-level relation alone would drop the layer wiring) |
| test_cpu_training_embedding_ivf.py | 389 | shrunk (9 source-text/manifest/workflow tests) | 212 | 17 | DELETE manifest_declares_both_families, bindings_register_the_gpu_names, host_modules_are_host_only, ivf_select_limit_is_the_device_literal, embedding_plan_codes_are_the_device_literals, sabotage_reaches_both_and_reads_back, workflow_checks_out_the_records, manifest_declares_the_byte_lm_trainer_lanes, adapter_is_an_import_time_connection_only; KEEP 3 host runtime tests (they also read each binding's sabotage flag back) |
| test_holtwinters_inference.py | 153 | shrunk (3 manifest/source/private-registry tests) | 73 | 6 | DELETE forecast_family_serves_holtwinters, three_bindings_register_predict_from_one_source, host_class_is_registered (private _FORMATS; the runtime test asserts host_model returns HostExponentialSmoothing); KEEP saved_model_forecasts_and_predicts |
| test_host_model_kmeans.py | 344 | shrunk (2 private-registry/manifest tests) | 31 | 3 | DELETE kmeans_format_is_a_host_model_format (private _FORMATS / constant), kmeans_is_a_declared_inference_lane_and_no_longer_owed (host_surface); KEEP round trip, same-bytes save, one-ULP fail-first, 9 refusals |
| test_host_radius_bindings.py | 43 | shrunk (private _bind routing test) | 15 | 0 | DELETE radius_sort_helper_routes_only_to_the_host (monkeypatched private `_bind`); KEEP saved_radius_sorted_ties_and_empty_rows (host_model route through the native sort) |
| test_ivf_embedding_cpu_inference.py | 288 | shrunk (4 binding-source/manifest tests) | 77 | 3 | DELETE inference_families_ship_and_carry_no_training, gpu_and_reference_bindings_register_build_and_search, extend_is_registered_on_the_gpu_reference_and_shipped_bindings, extend_uses_the_build_assignment_and_one_layout_rule; KEEP save/load, refusals, clone, 4 host runtime tests |
| test_kmeans_native_boundary.py | 258 | unchanged | 0 | 0 | KEEP all (provider one-call, hook refusal, admission order, copy/pickle) |
| test_mixture_reload_state.py | 32 | unchanged | 0 | 0 | KEEP (save/load round trip, same build) |
| test_neighbors_ann_native.py | 202 | unchanged | 0 | 0 | KEEP (audit_native_call, refusal class/wording, state vs the in-tree former glue in the same build; see doubts) |
| test_neighbors_graph_native.py | 206 | unchanged | 0 | 0 | KEEP (same as above, `_legacy_neighbors_graph`) |
| test_optics_native_boundary.py | 55 | unchanged | 0 | 0 | KEEP all 4 (admission, refusal order, owner, pickling) |
| test_prophet_native_ownership.py | 42 | unchanged | 0 | 0 | KEEP (repeat forecast vs the same run's hashes; input owners) |
| test_public_cnn_verifier_methods.py | 53 | unchanged | 0 | 0 | KEEP (public fit contract; BN running stats are natively held in cnn_batchnorm.mojo) |
| test_x_ann_repeat.py | 97 | shrunk (dead PKG constant) | 2 | 0 | KEEP (second call same bits, same build) |
| test_x_cluster_twice.py | 99 | unchanged | 0 | 0 | KEEP |
| test_x_cnn_batchnorm.py | 88 | unchanged | 0 | 0 | KEEP (ORACLE: reference vs torch) |
| test_x_cnn_blocks.py | 62 | unchanged | 0 | 0 | KEEP (eval-mode boundary; ORACLE vs torch) |
| test_x_cnn_conv.py | 172 | unchanged | 0 | 0 | KEEP (ORACLE vs torch) |
| test_x_cnn_gnn.py | 252 | unchanged | 0 | 0 | KEEP-NOPORT (np.random inputs; cache-key repeat is public) |
| test_x_cnn_repeat.py | 157 | unchanged | 0 | 0 | KEEP (repeat, same build) |
| test_x_cnn_trainer.py | 21 | unchanged | 0 | 0 | KEEP (public learning sanity, rng data) |
| test_x_decomp_kernel_table.py | 65 | unchanged | 0 | 0 | KEEP (capsule export, refusal, operand not written, repeat) |
| test_x_linear_repeat.py | 128 | unchanged | 0 | 0 | KEEP (repeat, same build) |
| test_x_metrics_repeat.py | 94 | unchanged | 0 | 0 | KEEP (repeat, same build) |
| test_x_neighbors_repeat.py | 116 | shrunk (dead PKG constant, unused _backend import) | 3 | 0 | KEEP (repeat, same build) |
| test_x_sequence_garch.py | 31 | unchanged | 0 | 0 | KEEP (ORACLE: arch) |
| test_x_sequence_prophet.py | 32 | unchanged | 0 | 0 | KEEP (rng data; batch == single relation is public) |
| test_x_sequence_repeat.py | 120 | unchanged | 0 | 0 | KEEP (repeat, same build) |
| test_x_sequence_theta.py | 28 | unchanged | 0 | 0 | KEEP (ORACLE: statsforecast) |

#### New checks

None. No test had both a literal fixture and a typed entry where moving would not drop public-path coverage.

| check | moved from | entries called |
|---|---|---|
| (none) | | |

#### Missing typed entries

- SAGEConv layer one-call host run entry taking (x ptr, edge_index ptr, n, nnz, F, lin_l weight ptr, root/bias flags) -> (y ptr, dx ptr). Today the layer is reached only through `GraphCalls` + door-built CSR (`x_cnn/graph_layer_run.mojo`). With it, test_cnn_max_gradient_ownership's literal arange fixture could move.

#### Product defects seen

None.

#### Doubts

- test_neighbors_ann_native / test_neighbors_graph_native compare native output bit for bit with `_legacy_neighbors_*`, the former Python glue frozen in tests/. It runs in the same build on the same kernels, so I kept it as a same-build oracle and not FROZEN-OLD. If the orchestrator counts frozen old glue as an "older version", these asserts become owed deletions and need same-build replacements (refit equals fit, repeat equals first).
- test_x_ann_repeat and test_x_neighbors_repeat still assert GPU column == CPU column inside one test. That is a same-version check, but test_x_metrics_repeat dropped the same comparison as "the release's one verification". I left them as they were.
- The new EXOG_MAX assertion expects the native refusal text (arima_common.mojo:131) to come up through the public fit. If the Python boundary refuses first with other wording, that assert needs loosening.

## models


| file | lines | result | removed | added | function verdicts |
|---|---|---|---|---|---|
| test_byte_lm_checkpoint_binary.py | 236 | unchanged | 0 | 0 | 11 KEEP: binary archive round trip, hostile bits, same-build two saves, 7 refusals (boundary); digests are same-build file hashes |
| test_byte_lm_host_native_provider.py | 213 | unchanged | 0 | 0 | 6 KEEP: audit_native_call, hook refusal, transactional outputs, checkpoint errors; rng(91) parameters (KEEP-NOPORT) |
| test_byte_lm_host_ragged_lifetime.py | 29 | unchanged | 0 | 0 | 1 KEEP: borrowed padded-input lifetime through the public door (boundary) |
| test_byte_lm_session.py | 247 | unchanged | 0 | 0 | 12 KEEP: resident session lifecycle, result keys, export copies, refusals; numerics already in byte_lm_trainer_session |
| test_byte_lm_surface.py | 327 | unchanged | 0 | 0 | 17 KEEP: registry packing (public return), copies, cursor/state/checkpoint admission and refusals, lock, metadata; rng(23) inputs |
| test_decode_n_tokens.py | 188 | moved to mamba_decode_n_tokens; shrunk (Mamba decode==steps relation) | 19 | 17 | test_mamba_decode_equals_the_steps -> PORT (replaced by test_mamba_decode_returns_every_token: shape, dtype, finiteness, sync); refusals, transformer, samba KEEP (fixture helpers imported by test_neural_repeat_calls/test_neural_coverage kept) |
| test_decode_sessions_cpu_route.py | 244 | unchanged | 0 | 0 | 4 KEEP: transformer session vs per-call, ownership/stale/load_state, weight copy at open (boundary) |
| test_mamba3_constructor_native.py | 35 | unchanged | 0 | 0 | 2 KEEP: audit_native_call constructor, dt_bias refusal |
| test_mamba_native_public.py | 221 | unchanged | 0 | 0 | 6 KEEP: audit_native_call doors vs retained driver, pickling, owned weights, session doors, unfilled outputs |
| test_models_loader.py | 664 | moved to tokenizer_pretokenize_cuts; shrunk (16 of 19 pretokenize cases) | 16 | 4 | test_pretokenize_cuts_as_the_pattern_says -> PORT (one case per pattern kept for the door's names); rest KEEP: safetensors reader, plan_for, refusals, loading, tokenizer encode/decode, GPT-2 oracle |
| test_resident_sampling_policy.py | 150 | unchanged | 0 | 0 | 6 KEEP: GPU-only two-build comparisons and stream lease/resume boundary |
| test_samba_attention_wiring.py | 94 | unchanged | 0 | 0 | 3 KEEP: explicit ABI vs public backward, missing-entry refusal, independent gradient oracle (audit) |
| test_samba_cpu_inference_native.py | 150 | unchanged | 0 | 0 | 4 KEEP: audit_native_call vs independent host primitives; rng(412) weights |
| test_samba_decode_session.py | 94 | unchanged | 0 | 0 | 2 KEEP: session vs per-call step on Generator(1)-initialized stack (Python-side generated weights), refusals |
| test_samba_inference_native.py | 269 | unchanged | 0 | 0 | 9 KEEP: audit vs primitive recipe, hook refusal, upload counters; rng(428) weights |
| test_samba_step_native.py | 209 | unchanged | 0 | 0 | 5 KEEP: step vs independent recipe, misalignment refusal, custom-schedule refusal, owner preservation |
| test_tokenizer_fingerprint.py | 34 | unchanged | 0 | 0 | 2 KEEP: the fingerprint's settings tail is built from Python fields in the door (see missing entries); expected digest is an independent formula, not frozen |

No SOURCE-TEXT tests and no FROZEN-OLD digests were found in these 17 files.

#### New checks

| check | moved from | entries called |
|---|---|---|
| checks/native_tests/tokenizer_pretokenize_cuts.mojo (cpu) | test_models_loader.py::test_pretokenize_cuts_as_the_pattern_says (19 literal cases) | tokenizer/impl/pretokenize.mojo `pretokenize_pattern(data, builtin_unicode_classes(), PATTERN_*)` |
| checks/native_tests/mamba_decode_n_tokens.mojo (cpu + gpu) | test_decode_n_tokens.py::test_mamba_decode_equals_the_steps | host: `MambaDecodeSessionRun.decode` vs `.step` (mamba/mamba_decode_session_run.mojo); device: `MambaBlockProgram.decode` vs `.step`, `export_state` (mamba/impl/mamba_block_program.mojo); fixtures `_Case` of mamba_decode_sessions.mojo |

Both registered in registry.mojo (one import, one append each); `tools/ops/native_checks_report.py plan --column cpu|gpu` lists them. Not compiled.

#### Missing typed entries

- tokenizer fingerprint: typed entry taking (VocabTokenizer, pattern code, normalizer, unicode version string, bos ids, eos ids) -> hex; the door `tokenizer_vocab_fingerprint` builds the settings tail with `builtins.str` over PythonObjects (bindings/tokenizer_public_native_door.mojo:1201-1236), so the tail rule cannot be checked natively.

#### Product defects seen

- none

## trees


Branch `lane/tests-prune-remaining-trees` (local, not pushed), base 7dcd11184c, head 775858fbfe. No compile, no tests run.
Source-only plan check: `python3 tools/ops/native_checks_report.py plan --column cpu` rc=0, lists `gbdt_tree_metadata` (shard 5); output in `plan_cpu.txt` here.

| file | lines | result | removed | added | function verdicts |
|---|---|---|---|---|---|
| test_cpu_training_e3_gbdt_trees.py | 161 | shrunk (4 source-text/manifest tests, constants, `_read`) | 109 | 7 | manifest_covers_both_lanes, oracles_import_no_gpu_module, driver_spells_..., sabotage_reaches_both_policies: SOURCE-TEXT deleted; non_symmetric_save_load: KEEP (archive round trip, rng) |
| test_cpu_training_forest_variants.py | 187 | shrunk (5 source-text/manifest tests, constants, `_read`) | 121 | 11 | manifest_declares_..., bindings_register_..., groves_module_..., rf_oracle_carries_..., extratrees_best_first_...: SOURCE-TEXT deleted; variants_fit_on_the_host: KEEP (public twice-same + engine reached, rng) |
| test_cpu_training_gbdt_ordered.py | 246 | shrunk (9 source-text/manifest/workflow tests) | 152 | 8 | manifest, binding_registers, oracles_import, ordered/feature_freq/pointwise/one_hot oracle spells, sabotage_reaches, workflow_triggers: SOURCE-TEXT deleted; ordered_save_load_and_ctr: KEEP (archive, eval curve, rng) |
| test_dart_buffer_lifetime.py | 17 | unchanged | 0 | 0 | binary_probability_rows: KEEP (public shape, rng) |
| test_feature_freq_experimental_surface.py | 86 | unchanged | 0 | 0 | validates_and_forwards: KEEP (door forwarding of lr/l2/random_state, n_features_in_; twice-same already in gbdt_public_fits); refusals, weight_refusals, constant_column: KEEP (pytest.raises); forwards_weights: KEEP (GPU-only door forwarding) |
| test_forest_inference_engine.py | 191 | unchanged | 0 | 0 | all 8 KEEP: engine routing, archives/format tags, pickle, monkeypatched Python surface, admission refusals; numeric relations use rng inputs |
| test_forest_list_label_admission_cost.py | 39 | unchanged | 0 | 0 | KEEP (list-label admission cost, boundary) |
| test_gbdt_host_modes.py | 193 | shrunk (3 tests, `_read`, `KINDS`) | 42 | 6 | gbdt_estimators_are_the_three_saving_classes: deleted (private `GBDT_ESTIMATORS` table; refusal behaviour kept below); forest_family_and_gate_declare_the_four_kinds, identity_break_host_infer_is_opt_in: SOURCE-TEXT deleted; archive_names_and_losses, ctr_records_refuse, gate_coded_transform, four_modes_predict_through_host_model: KEEP |
| test_gbdt_pair_logit.py | 99 | unchanged | 0 | 0 | refusals_by_name, pairs_validation: KEEP (pytest.raises, pairs admission, rng) |
| test_gbdt_tree_metadata.py | 52 | moved to gbdt_tree_metadata; shrunk | 19 | 10 | tree_values_recover_bits (both records): bits+counts MOVED, dtype kept as one public case; tree_offsets_and_depth_zero: MOVED; incomplete_metadata_refused (3 texts): MOVED, one ValueError mapping case kept; unfitted_inspection_refused: KEEP |
| test_gbdt_yeti_rank.py | 89 | unchanged | 0 | 0 | learns_on_uneven_queries, l2_default, refusals: KEEP (rng fixture, public defaults, pytest.raises) |
| test_host_tree_native.py | 182 | shrunk (ast one-call test, `ast`/`Path` imports) | 16 | 0 | tree_prediction_and_state_methods_are_one_native_call: SOURCE-TEXT (ast) deleted; the rest KEEP (audit_native_call, input layouts, label types, callback refusal, archives vs the test-side legacy oracle) |
| test_native_tree_lifecycle.py | 181 | unchanged | 0 | 0 | all 9 KEEP (constructor binding, audit_native_call, ClassLabels hook refusal, set_params, NotFittedError) |
| test_rf_class_weight.py | 60 | unchanged | 0 | 0 | all KEEP (class_weight admission and public dispatch, rng) |
| test_trees_repeat.py | 153 | unchanged | 0 | 0 | KEEP (same-build twice-same subprocess run per column; no frozen digest) |

Totals: Python 459 removed, 42 added; Mojo +92 (new check 90, registry 2). No frozen-old digests found in these 15 files.
Removed (rule 5): the 22 source-text/manifest/private-table test functions above and their constants/`_read` helpers; nothing else identified, because every remaining function tests public behaviour or the boundary.

#### New checks

| check | moved from | entries called |
|---|---|---|
| checks/native_tests/gbdt_tree_metadata.mojo (6 cases, cpu+gpu, parallel) | test_gbdt_tree_metadata.py: tree_values_recover_bits (tree/ntree), tree_offsets_and_depth_zero, incomplete_metadata_refused | core.trees_gbdt_metadata.gbdt_tree_metadata(text) -> GBDTTreeMetadata(counts, values) |

#### Missing typed entries

None. Every kept native-looking test stays for one of: inputs drawn from `np.random.default_rng` (no committed fixture), door forwarding of estimator fields (`feature_freq_fit` reads `learning_rate`/`l2_leaf_reg`/`random_state` from the Python model), or archive/pickle/audit boundary. Typed run entries exist for the FeatureFreq fit (host `gbdt/mojolearn_gbdt_host_run.mojo:1574`, device `gbdt/mojolearn_gbdt_run.mojo:474`).

#### Product defects seen

- bindings/trees_public_gbdt_door.mojo:730-822 `feature_freq_fit`: validation (source range, integer codes, density, constant numeric columns) and the reductions run on PythonObjects and through `py_call` inside the door, not in a typed RUN; this is open "Python objects only at the public boundary" migration work (AGENTS.md part 2), not a test defect.

#### Doubts

- lanes.tsv named target checks for several of these files that are NOT imported in registry.mojo and so never run in CI: class_weight_plan, host_entry_forest, trees_door_run, trees_entry_dart, trees_entry_forest, gp_label_owner (and trees_entry_gbdt, trees_entry_glue). Nothing was joined to them; whether they should be registered is a separate decision.
- The new check is uncompiled; written to the shape of core_all_finite and with idioms found in sibling checks (`except:`, `.cast[DType.float32]()`).

