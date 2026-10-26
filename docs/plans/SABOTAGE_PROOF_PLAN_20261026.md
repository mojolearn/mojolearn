# Sabotage proof-run plan (2026-10-26)

Every arm in `tools/identity_lanes/coverage_sabotage.tsv` not yet in `coverage_sabotage_proven.tsv` (41 arms: the 11 left unproven before lane precise-coverage-0859 and the 30 it added), grouped into 28 batches by `python3 tools/coverage_sabotage.py batch-plan` over the whole-surface audit at `b4ed03f68`. One sabotaged build per batch proves every arm in it: clean records, every patch of the batch applied and its bindings rebuilt (sabotaged records: each arm's lanes must read DIVERGENT), patches reverted and the saved clean native files restored (restored records: IDENTICAL to clean), then `coverage_sabotage.py check --batch-bindings` per arm and column.

Two arms share a batch only when neither one's patch can reach the other's lanes (the audit's closure of every registration the patch edits, with that registration's lanes) and their patches apply together; arms that use the same patch are one fault and always share a build. So a lane that moves is attributable to its own arm.

## Builds

20 distinct binding builds over the 28 batches (each batch rebuilds only its own; a `_host` stem is a CPU-column build, every other stem a device build): `build`, `build_core_host`, `build_estimators`, `build_estimators_host`, `build_mamba`, `build_rf`, `build_runtime_classical_host`, `build_svm_host`, `build_tokenizer_host`, `build_training_host`, `build_x_cnn_host`, `build_x_decomp`, `build_x_decomp_host`, `build_x_linear_host`, `build_x_metrics`, `build_x_metrics_host`, `build_x_neighbors`, `build_x_neighbors_host`, `build_x_sequence`, `build_x_sequence_host`.

Batches whose columns include `device` need a GPU box (the 2026-10-24 arms ran on the Mac's Metal column); `cpu`-only batches run on any box with the host set built.

## Commands

Each line runs one batch in the branch tree (`lq` builds the branch's bindings first; the batch then rebuilds its own bindings with the patches applied). `OUT` defaults to `$HOME/mojolearn-sabotage/<batch>`; the job prints one `SABOTAGE batch=... arm=... column=... verdict=CAUGHT|MISSED|BUILD-FAIL` line per arm and column.

```sh
lq add apple CMD lane/precise-coverage-0859 sab-core-1 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json core-1'
lq add apple CMD lane/precise-coverage-0859 sab-core-2 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json core-2'
lq add apple CMD lane/precise-coverage-0859 sab-core-3 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json core-3'
lq add apple CMD lane/precise-coverage-0859 sab-estimators-1 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json estimators-1'
lq add apple CMD lane/precise-coverage-0859 sab-mamba-1 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json mamba-1'
lq add apple CMD lane/precise-coverage-0859 sab-mamba-2 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json mamba-2'
lq add apple CMD lane/precise-coverage-0859 sab-mamba-3 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json mamba-3'
lq add apple CMD lane/precise-coverage-0859 sab-mamba-4 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json mamba-4'
lq add apple CMD lane/precise-coverage-0859 sab-rf-1 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json rf-1'
lq add apple CMD lane/precise-coverage-0859 sab-rf-2 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json rf-2'
lq add apple CMD lane/precise-coverage-0859 sab-runtime_classical-1 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json runtime_classical-1'
lq add apple CMD lane/precise-coverage-0859 sab-runtime_classical-2 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json runtime_classical-2'
lq add apple CMD lane/precise-coverage-0859 sab-runtime_classical-3 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json runtime_classical-3'
lq add apple CMD lane/precise-coverage-0859 sab-svm-1 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json svm-1'
lq add apple CMD lane/precise-coverage-0859 sab-tokenizer-1 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json tokenizer-1'
lq add apple CMD lane/precise-coverage-0859 sab-tokenizer-2 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json tokenizer-2'
lq add apple CMD lane/precise-coverage-0859 sab-tokenizer-3 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json tokenizer-3'
lq add apple CMD lane/precise-coverage-0859 sab-tokenizer-4 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json tokenizer-4'
lq add apple CMD lane/precise-coverage-0859 sab-training-1 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json training-1'
lq add apple CMD lane/precise-coverage-0859 sab-x_cnn-1 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json x_cnn-1'
lq add apple CMD lane/precise-coverage-0859 sab-x_decomp-1 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json x_decomp-1'
lq add apple CMD lane/precise-coverage-0859 sab-x_linear-1 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json x_linear-1'
lq add apple CMD lane/precise-coverage-0859 sab-x_metrics-1 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json x_metrics-1'
lq add apple CMD lane/precise-coverage-0859 sab-x_neighbors-1 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json x_neighbors-1'
lq add apple CMD lane/precise-coverage-0859 sab-x_neighbors-2 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json x_neighbors-2'
lq add apple CMD lane/precise-coverage-0859 sab-x_neighbors-3 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json x_neighbors-3'
lq add apple CMD lane/precise-coverage-0859 sab-x_neighbors-4 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json x_neighbors-4'
lq add apple CMD lane/precise-coverage-0859 sab-x_sequence-1 'bash tools/coverage_sabotage_batch.sh docs/plans/sabotage_batches_20261026.json x_sequence-1'
```

Read results with `lq results apple SABOTAGE`. A CAUGHT arm gets its row in `coverage_sabotage_proven.tsv`; a MISSED arm gets a better patch (or its lanes corrected), never a rerun of the same thing.

## Arms caught as REFUSED

These arms' faults trip a product check inside the lane rather than moving a digest; `check` counts a REFUSED cell only with a hash-bound `--evidence` review naming the check reached (place it at `$OUT/evidence_<arm>_<column>.json` before the check step, or rerun the check by hand): `coverage-precise-core-msel_rebase_device`, `coverage-precise-mamba-m3_session_copy_in`, `coverage-precise-mamba-m3_session_export`, `coverage-precise-x_neighbors-p2m_const_cols`, `coverage-precise-x_neighbors-row_all_zero`.

## Batches

| batch | arms | builds | columns | lanes |
|---|---|---|---|---|
| `core-1` | coverage-precise-core-msel_host_put, coverage-precise-core-msel_rebase_device, coverage-precise-core-shard_topk_merge_device, coverage-training-optimizer_public_common | build_core_host, build | cpu, device | x-metrics-search, par-forest, par-forest-reg, par-forest-et-clf, par-reference-knn, par-reference-knn-reg, optim-sgd, optim-adam-clip, optim-maximize |
| `core-2` | coverage-precise-core-msel_host_read | build_core_host | cpu | x-metrics-search |
| `core-3` | coverage-precise-core-shard_topk_merge_host | build_core_host | cpu | par-reference-knn |
| `estimators-1` | coverage-precise-decomposition-pca_fit_host | build_estimators | device | pca, pca-whiten |
| `mamba-1` | coverage-precise-mamba-m3_load_weights | build_mamba | device | mamba3-decode-session |
| `mamba-2` | coverage-precise-mamba-m3_session_copy_in | build_mamba | device | mamba3-decode-session |
| `mamba-3` | coverage-precise-mamba-m3_session_export | build_mamba | device | mamba3-decode-session |
| `mamba-4` | coverage-precise-mamba-rms_out | build_mamba | device | mamba1-decode-session, mamba2-decode-session, mamba3-decode-session |
| `rf-1` | coverage-precise-bindings-rf_aux_oob_score | build_rf | device | trees-oob-cv-link |
| `rf-2` | coverage-precise-core-rf_export_colmajor, coverage-precise-core-rf_export_shard, coverage-precise-core-rf_export_weighted | build_rf | device | trees-dt-clf, trees-dt-reg, par-forest, par-forest-reg, trees-rf-weighted, trees-adaboost-clf, rf-clf-balanced-parallel |
| `runtime_classical-1` | coverage-core-scaler_fit_impl | build_runtime_classical_host | cpu | classical-float16-sequences |
| `runtime_classical-2` | coverage-core-sequential_inputs_impl | build_runtime_classical_host | cpu | classical-float16-sequences |
| `runtime_classical-3` | coverage-neighbors-knn_door | build_runtime_classical_host | cpu | knn-clf, knn-reg, knn-clf-distance |
| `svm-1` | coverage-precise-svm-svc_ovo_layout_host | build_svm_host | cpu | x-neighbors-svc-multiclass |
| `tokenizer-1` | coverage-bindings-native_params, coverage-core-causal_checkpoint_impl, coverage-core-small_mlp_state_impl, coverage-precise-bindings-cross_vendor_fold, coverage-precise-tokenizer-ranks_text | build_tokenizer_host | cpu | public-set-params, x-huber-outliers, hf-checkpoint, hf-model-surface, mlp, cross-vendor-fold, bpe-vocabulary |
| `tokenizer-2` | coverage-core-native_json_impl | build_tokenizer_host | cpu | hf-checkpoint, hf-model-surface, hf-causal-lm |
| `tokenizer-3` | coverage-precise-bindings-parallel_mlp_apply, coverage-precise-tokenizer-vocab_fingerprint | build_tokenizer_host | cpu | par-mlp, hf-causal-lm-text |
| `tokenizer-4` | coverage-transformer-causal_lm_construct_impl | build_tokenizer_host | cpu | hf-model-surface |
| `training-1` | coverage-training-optimizer_public_host | build_training_host | cpu | optim-sgd |
| `x_cnn-1` | coverage-precise-x_cnn-csr_build_host | build_x_cnn_host | cpu | x-cnn-gcn, x-cnn-sage |
| `x_decomp-1` | coverage-precise-x_decomp-pca_mle_pick_dev, coverage-precise-x_decomp-pca_mle_pick_host | build_x_decomp, build_estimators, build_x_decomp_host, build_estimators_host | cpu, device | x-decomp-pca-randomized |
| `x_linear-1` | coverage-precise-x_linear-spearman_sign_host | build_x_linear_host | cpu | x-isotonic |
| `x_metrics-1` | coverage-precise-x_metrics-first_rows | build_x_metrics, build_x_metrics_host | cpu, device | x-metrics-splitters |
| `x_neighbors-1` | coverage-precise-x_neighbors-log_softmax_dev, coverage-precise-x_neighbors-nan_cells_host, coverage-precise-x_neighbors-p2m_transpose_i | build_x_neighbors, build_x_neighbors_host | cpu, device | x-neighbors-nearest-centroid, x-neighbors-knn-imputer, knn-clf |
| `x_neighbors-2` | coverage-precise-x_neighbors-p2m_const_cols | build_x_neighbors_host | cpu | x-neighbors-nearest-centroid |
| `x_neighbors-3` | coverage-precise-x_neighbors-row_all_zero | build_x_neighbors_host | cpu | x-cluster-agglo-linkages |
| `x_neighbors-4` | coverage-precise-x_neighbors-row_argmax | build_x_neighbors_host | cpu | x-neighbors-label-propagation, x-neighbors-label-spreading |
| `x_sequence-1` | coverage-autograd-imported_run | build_x_sequence, build_x_sequence_host | cpu, device | sequence-onnx-import |
