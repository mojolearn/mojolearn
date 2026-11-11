# Python package audit (2026-11-11)

What every function in `python/mojolearn/` (not `tests/`, plus `python/mojolearn_diagnostics.py`) does, by the six classes of the lane brief. Static: AST scan of every file plus a repository-wide name search (`git grep` over HEAD: Python, Mojo, shell, TOML, YAML, JSON; `bench/results` and `experiments/removed` excluded). Nothing was compiled, imported or run.

Package size (`python/mojolearn` outside `tests/`, `wc -l` of `*.py`): **118,552 lines before, 116,933 after** this lane.

The per-function table, one row per function and method (nested functions are counted in their parent), is [`PYTHON_PACKAGE_AUDIT_2026-11-11.tsv`](PYTHON_PACKAGE_AUDIT_2026-11-11.tsv): file, line, qualified name, lines, class, reason, loop/comprehension/arithmetic/numpy-math/value-function counts, and for (e) the public operations that reach it.

## Classes and how a function was put in one

| class | meaning | rule applied | functions | lines |
|---|---|---|---:|---:|
| (a) | public API shell / admission | default: no loop with arithmetic, no numpy math, fewer than 4 arithmetic ops | 3,150 | 33,274 |
| (b) | import-time wiring | in a wiring module (`_surface_*`, `_version`, `_native_loaded_modules`, `_merged`, `_host_import_aliases`, `_sklearn_modules`, `_runtime_fallbacks`) or a connect/load/install function called at import, or loader, binary-set and packaging metadata (`_backend`, `_rocm_runtime`, `gpu_plugins`, `host_surface`, `_native_*` metadata modules) that reads no user data | 270 | 8,006 |
| (c) | testing / verification / benchmark tool | module reached only from `mojolearn.__main__` (the `verify`/`conformance` CLI) and not from any public module, Mojo door or wiring table; or a definition in a product module whose only callers are tests or tools | 581 | 15,620 |
| (d) | sklearn tags | `__sklearn_tags__`, `_more_tags` and tag helpers | 14 | 112 |
| (e) | PRODUCT WORK IN PYTHON | product-reachable and (numpy/pandas/torch math, or a loop with arithmetic, or 4+ arithmetic ops, or a loop calling sorted/sum/min/max/math/tolist, or 3+ such calls) | 108 | 2,145 |
| (f) | dead code (deleted) | no transitive caller from public names, import-time code, other package modules, tests, tools or Mojo | 128 | 1,306 |

Function lines total 60,463; the rest of the package is module-level code: docstrings, constant tables (`host_surface.py`, `_verification_evidence_data.py`), class bodies and imports.

Limits: the (e) test is a static feature test (good to about +-15 %, as the 2026-11-09 import audit). A loop over parameters with no arithmetic is (a); a function composing device primitives without arithmetic (for example `_expansion_decomp._Kit` callers) is (a) unless it also computes. The reached-by column follows names defined at most twice (same module first), so an inherited public method shows as its mixin.

## Modules

| module | lines | functions | a | b | c | d | e | f | e lines |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `host_surface.py` | 5,225 | 45 | 37 | 8 | 0 | 0 | 0 | 0 | 0 |
| `_expansion_decomp.py` | 4,823 | 231 | 208 | 1 | 0 | 1 | 19 | 2 | 248 |
| `_verify_all.py` | 4,495 | 91 | 0 | 0 | 91 | 0 | 0 | 0 | 0 |
| `_expansion_prep.py` | 3,676 | 205 | 173 | 0 | 24 | 3 | 5 | 0 | 52 |
| `_verification_evidence_data.py` | 3,375 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_expansion_trees.py` | 2,842 | 114 | 100 | 1 | 3 | 0 | 0 | 10 | 0 |
| `_backend.py` | 2,754 | 76 | 66 | 10 | 0 | 0 | 0 | 0 | 0 |
| `_expansion_linear.py` | 2,544 | 59 | 57 | 0 | 0 | 1 | 0 | 1 | 0 |
| `_expansion_neighbors.py` | 2,313 | 102 | 80 | 1 | 15 | 0 | 0 | 6 | 0 |
| `model_selection.py` | 2,152 | 107 | 96 | 0 | 4 | 0 | 2 | 5 | 57 |
| `_verify_graph.py` | 1,841 | 47 | 0 | 0 | 47 | 0 | 0 | 0 | 0 |
| `_verify.py` | 1,681 | 42 | 0 | 0 | 42 | 0 | 0 | 0 | 0 |
| `_metrics_impl.py` | 1,672 | 45 | 39 | 0 | 6 | 0 | 0 | 0 | 0 |
| `ops.py` | 1,656 | 156 | 141 | 8 | 2 | 0 | 0 | 5 | 0 |
| `_verification_api_reference.py` | 1,625 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_training_impl.py` | 1,565 | 79 | 67 | 3 | 4 | 0 | 1 | 4 | 13 |
| `_classical_host.py` | 1,550 | 73 | 63 | 2 | 1 | 0 | 2 | 5 | 34 |
| `linear_model.py` | 1,515 | 67 | 54 | 3 | 8 | 2 | 0 | 0 | 0 |
| `neighbors.py` | 1,498 | 41 | 35 | 4 | 2 | 0 | 0 | 0 | 0 |
| `_byte_lm_impl.py` | 1,481 | 76 | 49 | 3 | 7 | 0 | 8 | 9 | 230 |
| `_verification_catalog.py` | 1,386 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `ensemble.py` | 1,295 | 19 | 15 | 1 | 2 | 1 | 0 | 0 | 0 |
| `_mamba_impl.py` | 1,294 | 69 | 65 | 4 | 0 | 0 | 0 | 0 | 0 |
| `_expansion_cnn.py` | 1,272 | 175 | 150 | 0 | 3 | 0 | 6 | 16 | 98 |
| `_svm_impl.py` | 1,258 | 23 | 21 | 2 | 0 | 0 | 0 | 0 | 0 |
| `_expansion_cluster.py` | 1,193 | 36 | 27 | 6 | 1 | 0 | 1 | 1 | 15 |
| `_expansion_metrics.py` | 1,172 | 100 | 89 | 0 | 0 | 0 | 7 | 4 | 133 |
| `_expansion_ann.py` | 1,170 | 44 | 37 | 1 | 1 | 0 | 1 | 4 | 7 |
| `_verification_endpoint_data.py` | 1,152 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_buffer.py` | 1,149 | 55 | 51 | 0 | 0 | 0 | 4 | 0 | 183 |
| `_conformance.py` | 1,055 | 20 | 0 | 0 | 20 | 0 | 0 | 0 | 0 |
| `_verify_reference.py` | 1,035 | 23 | 0 | 0 | 23 | 0 | 0 | 0 | 0 |
| `_native_msel_metric_plans.py` | 941 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_linalg_impl.py` | 924 | 34 | 27 | 2 | 4 | 0 | 1 | 0 | 69 |
| `randomforest.py` | 888 | 18 | 16 | 1 | 1 | 0 | 0 | 0 | 0 |
| `_gp_impl.py` | 885 | 54 | 47 | 1 | 2 | 0 | 0 | 4 | 0 |
| `_transformer_impl.py` | 879 | 39 | 31 | 1 | 4 | 0 | 1 | 2 | 21 |
| `_tsa_impl.py` | 790 | 21 | 14 | 2 | 1 | 0 | 1 | 3 | 11 |
| `__init__.py` | 752 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_spectral_impl.py` | 740 | 16 | 13 | 1 | 2 | 0 | 0 | 0 | 0 |
| `_mode.py` | 734 | 33 | 26 | 4 | 0 | 2 | 1 | 0 | 12 |
| `_x_sequence_rnn.py` | 732 | 13 | 11 | 1 | 1 | 0 | 0 | 0 | 0 |
| `_arima_impl.py` | 719 | 15 | 14 | 1 | 0 | 0 | 0 | 0 | 0 |
| `kernel_methods.py` | 707 | 22 | 18 | 3 | 0 | 0 | 0 | 1 | 0 |
| `_array.py` | 684 | 43 | 37 | 1 | 1 | 0 | 4 | 0 | 23 |
| `extratrees.py` | 656 | 17 | 15 | 1 | 0 | 0 | 0 | 1 | 0 |
| `tokenizer.py` | 654 | 54 | 36 | 1 | 11 | 0 | 1 | 5 | 12 |
| `__main__.py` | 634 | 9 | 0 | 0 | 8 | 0 | 0 | 1 | 0 |
| `_sk_validate.py` | 634 | 33 | 32 | 0 | 0 | 0 | 0 | 1 | 0 |
| `_x_sequence_model.py` | 627 | 45 | 31 | 3 | 3 | 0 | 0 | 8 | 0 |
| `preprocessing.py` | 606 | 36 | 30 | 2 | 2 | 2 | 0 | 0 | 0 |
| `decomposition.py` | 603 | 17 | 12 | 2 | 3 | 0 | 0 | 0 | 0 |
| `density.py` | 587 | 19 | 15 | 2 | 1 | 0 | 0 | 1 | 0 |
| `_samba_impl.py` | 578 | 37 | 30 | 3 | 1 | 0 | 0 | 3 | 0 |
| `python/mojolearn_diagnostics.py` | 538 | 15 | 11 | 0 | 4 | 0 | 0 | 0 | 0 |
| `parallel_classical.py` | 537 | 22 | 22 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_ivf_impl.py` | 529 | 19 | 16 | 1 | 1 | 0 | 0 | 1 | 0 |
| `_portable_math.py` | 514 | 31 | 21 | 1 | 0 | 0 | 9 | 0 | 223 |
| `models/config.py` | 512 | 32 | 28 | 0 | 0 | 0 | 4 | 0 | 134 |
| `_numeric_profile.py` | 507 | 17 | 16 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_gpc_impl.py` | 501 | 14 | 13 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_labels.py` | 492 | 17 | 16 | 0 | 0 | 0 | 1 | 0 | 33 |
| `_mlp_impl.py` | 484 | 31 | 23 | 3 | 3 | 0 | 1 | 1 | 35 |
| `resample.py` | 470 | 10 | 10 | 0 | 0 | 0 | 0 | 0 | 0 |
| `hdbscan.py` | 454 | 9 | 8 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_byte_lm_host.py` | 448 | 32 | 23 | 2 | 3 | 0 | 2 | 2 | 46 |
| `gpu_plugins.py` | 448 | 30 | 27 | 3 | 0 | 0 | 0 | 0 | 0 |
| `cluster.py` | 444 | 10 | 9 | 1 | 0 | 0 | 0 | 0 | 0 |
| `neural_inference.py` | 443 | 22 | 21 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_solver_impl.py` | 440 | 7 | 6 | 1 | 0 | 0 | 0 | 0 | 0 |
| `models/causal_lm.py` | 432 | 41 | 34 | 2 | 2 | 0 | 2 | 1 | 30 |
| `mixture.py` | 423 | 13 | 12 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_native_full_reset_params.py` | 422 | 7 | 2 | 5 | 0 | 0 | 0 | 0 | 0 |
| `_hierarchy_impl.py` | 421 | 8 | 7 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_identity.py` | 410 | 13 | 0 | 0 | 13 | 0 | 0 | 0 | 0 |
| `_x_sequence_optim.py` | 409 | 25 | 20 | 1 | 0 | 0 | 0 | 4 | 0 |
| `svm.py` | 403 | 10 | 10 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_classical_bridge.py` | 402 | 16 | 11 | 5 | 0 | 0 | 0 | 0 | 0 |
| `_serialize.py` | 401 | 18 | 10 | 0 | 0 | 0 | 7 | 1 | 136 |
| `_verify_worker.py` | 399 | 17 | 0 | 0 | 17 | 0 | 0 | 0 | 0 |
| `_verify_distributed.py` | 385 | 9 | 0 | 0 | 9 | 0 | 0 | 0 | 0 |
| `lm_corpus.py` | 385 | 21 | 19 | 0 | 2 | 0 | 0 | 0 | 0 |
| `_native_msel_prep.py` | 382 | 3 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| `_x_sequence_theta.py` | 375 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_trees_bridge.py` | 374 | 15 | 10 | 5 | 0 | 0 | 0 | 0 | 0 |
| `_tokenizer_synthetic.py` | 354 | 18 | 7 | 0 | 0 | 0 | 11 | 0 | 207 |
| `_umap_impl.py` | 353 | 7 | 6 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_gbdt_adapters.py` | 339 | 20 | 16 | 2 | 1 | 1 | 0 | 0 | 0 |
| `_x_sequence_mlp.py` | 333 | 8 | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_verify_model.py` | 328 | 17 | 0 | 0 | 17 | 0 | 0 | 0 | 0 |
| `_native_public_attributes.py` | 306 | 11 | 7 | 4 | 0 | 0 | 0 | 0 | 0 |
| `_forest_host.py` | 302 | 17 | 14 | 1 | 2 | 0 | 0 | 0 | 0 |
| `_forest_protocol.py` | 292 | 30 | 23 | 0 | 3 | 1 | 0 | 3 | 0 |
| `_verify_light.py` | 288 | 12 | 0 | 0 | 12 | 0 | 0 | 0 | 0 |
| `_verify_parallel_cv.py` | 285 | 6 | 0 | 0 | 6 | 0 | 0 | 0 | 0 |
| `embedding.py` | 284 | 14 | 13 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_native_optimizer_bridge.py` | 281 | 5 | 0 | 5 | 0 | 0 | 0 | 0 | 0 |
| `_iforest_impl.py` | 278 | 9 | 8 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_verification_api_targets.py` | 276 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `parallel_training.py` | 274 | 36 | 32 | 2 | 1 | 0 | 0 | 1 | 0 |
| `parallel_ensemble.py` | 270 | 12 | 12 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_verification_coverage.py` | 262 | 4 | 0 | 0 | 4 | 0 | 0 | 0 | 0 |
| `_x_sequence_norm.py` | 252 | 8 | 7 | 0 | 0 | 0 | 0 | 1 | 0 |
| `_verify_native_runtime.py` | 248 | 12 | 0 | 0 | 12 | 0 | 0 | 0 | 0 |
| `_cholesky_impl.py` | 237 | 11 | 9 | 1 | 1 | 0 | 0 | 0 | 0 |
| `_gpu_witness.py` | 237 | 8 | 0 | 0 | 8 | 0 | 0 | 0 | 0 |
| `_x_sequence_sched.py` | 228 | 9 | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_verify_causal_lm.py` | 221 | 6 | 0 | 0 | 6 | 0 | 0 | 0 | 0 |
| `_x_sequence_prophet.py` | 220 | 4 | 3 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_x_sequence_autoarima.py` | 218 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_verify_api_runner.py` | 215 | 9 | 0 | 0 | 9 | 0 | 0 | 0 | 0 |
| `_x_sequence_croston.py` | 215 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_x_sequence_moe.py` | 213 | 14 | 8 | 2 | 0 | 0 | 0 | 4 | 0 |
| `_native_xcnn_bridge.py` | 209 | 5 | 2 | 3 | 0 | 0 | 0 | 0 | 0 |
| `_x_sequence_ets.py` | 208 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_gbdt_host.py` | 203 | 10 | 10 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_causal_lm_fixtures.py` | 202 | 11 | 0 | 0 | 11 | 0 | 0 | 0 | 0 |
| `_verify_api_estimators.py` | 202 | 3 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| `_verify_api_metadata.py` | 200 | 3 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| `_native_declared_constructors.py` | 196 | 3 | 1 | 2 | 0 | 0 | 0 | 0 | 0 |
| `_bufcheck.py` | 195 | 14 | 12 | 0 | 0 | 0 | 1 | 1 | 5 |
| `_verify_generation.py` | 190 | 8 | 0 | 0 | 8 | 0 | 0 | 0 | 0 |
| `_native_parallel_public.py` | 189 | 5 | 4 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_rocm_runtime.py` | 187 | 8 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| `lowbit.py` | 186 | 11 | 11 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_verify_api_neural.py` | 183 | 3 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| `_native_samba_bridge.py` | 182 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| `_verify_partitioned_program.py` | 180 | 4 | 0 | 0 | 4 | 0 | 0 | 0 | 0 |
| `_ragged.py` | 177 | 8 | 6 | 0 | 0 | 0 | 2 | 0 | 32 |
| `cross_vendor.py` | 172 | 27 | 21 | 1 | 4 | 0 | 0 | 1 | 0 |
| `_surface_sequence.py` | 170 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_byte_lm_config.py` | 167 | 18 | 17 | 1 | 0 | 0 | 0 | 0 | 0 |
| `importers/_graph.py` | 167 | 14 | 12 | 1 | 0 | 0 | 0 | 1 | 0 |
| `models/safetensors.py` | 164 | 28 | 27 | 0 | 0 | 0 | 0 | 1 | 0 |
| `_arena_io.py` | 162 | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_surface_prep.py` | 162 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_surface_metrics.py` | 161 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_surface_neighbors.py` | 160 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_host_archive_predict.py` | 157 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_verify_api_training.py` | 156 | 4 | 0 | 0 | 4 | 0 | 0 | 0 | 0 |
| `_surface_trees.py` | 155 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_verify_distributed_program.py` | 155 | 4 | 0 | 0 | 4 | 0 | 0 | 0 | 0 |
| `_native_arena_rows.py` | 152 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_native_params_metadata.py` | 152 | 6 | 2 | 4 | 0 | 0 | 0 | 0 | 0 |
| `models/parallel_causal_lm.py` | 148 | 24 | 21 | 1 | 2 | 0 | 0 | 0 | 0 |
| `_native_mode_reconstruction.py` | 143 | 3 | 1 | 2 | 0 | 0 | 0 | 0 | 0 |
| `_native_logistic_bridge.py` | 139 | 5 | 2 | 3 | 0 | 0 | 0 | 0 | 0 |
| `models/tokenizer.py` | 139 | 15 | 13 | 0 | 2 | 0 | 0 | 0 | 0 |
| `_surface_linear.py` | 138 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_parallel_cv_witness.py` | 136 | 5 | 0 | 0 | 5 | 0 | 0 | 0 | 0 |
| `_x_sequence_var.py` | 134 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_x_sequence_garch.py` | 133 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_msel_dispatch.py` | 132 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_verify_api_utilities.py` | 131 | 3 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| `_native_loaded_modules.py` | 127 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| `_native_archive_bridge.py` | 126 | 6 | 5 | 1 | 0 | 0 | 0 | 0 | 0 |
| `importers/_program.py` | 126 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| `parallel_neighbors.py` | 126 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| `linalg_fast.py` | 125 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_merged.py` | 124 | 10 | 0 | 10 | 0 | 0 | 0 | 0 | 0 |
| `_x_sequence_stl.py` | 122 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_host_archive_bridge.py` | 120 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| `_native_labels_bridge.py` | 120 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_xseq_bridge.py` | 120 | 3 | 2 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_expansion_sequence.py` | 116 | 3 | 2 | 1 | 0 | 0 | 0 | 0 | 0 |
| `adapters/max.py` | 115 | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| `parallel_neighbors_reference.py` | 113 | 7 | 7 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_x_linear_bridge.py` | 111 | 4 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| `_verify_api_misc.py` | 110 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| `_native_family_bridge.py` | 108 | 6 | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `_native_spectral_umap_bridge.py` | 108 | 4 | 3 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_verify_small.py` | 108 | 4 | 0 | 0 | 4 | 0 | 0 | 0 | 0 |
| `adapters/torch.py` | 108 | 6 | 5 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_x_sequence_device.py` | 105 | 16 | 14 | 0 | 0 | 0 | 0 | 2 | 0 |
| `linalg.py` | 104 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_scaler_bridge.py` | 103 | 4 | 1 | 3 | 0 | 0 | 0 | 0 | 0 |
| `_native_mode_metadata.py` | 101 | 4 | 0 | 4 | 0 | 0 | 0 | 0 | 0 |
| `importers/_table.py` | 98 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_tsa_bridge.py` | 97 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_native_neighbor_queries.py` | 92 | 4 | 1 | 2 | 0 | 0 | 1 | 0 | 16 |
| `_native_backend_public.py` | 91 | 6 | 5 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_native_parallel_shard.py` | 91 | 3 | 1 | 2 | 0 | 0 | 0 | 0 | 0 |
| `_verify_light_command.py` | 91 | 4 | 0 | 0 | 4 | 0 | 0 | 0 | 0 |
| `_native_pca_bridge.py` | 90 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| `parallel_preprocessing.py` | 89 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_decomp_bridge.py` | 88 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| `_native_decomp_kit_bridge.py` | 87 | 3 | 0 | 2 | 0 | 0 | 1 | 0 | 18 |
| `_surface_cnn.py` | 87 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `training.py` | 85 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_classical_warnings.py` | 83 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_verify_api_faults.py` | 82 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| `importers/_native_reader_bridge.py` | 82 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_native_arena_prog.py` | 81 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| `_native_prep_encode_bridge.py` | 81 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_native_prep_stateless_bridge.py` | 80 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_native_qn_bridge.py` | 79 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_surface_decomp.py` | 79 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_minmax_bridge.py` | 78 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| `_verify_api.py` | 78 | 5 | 0 | 0 | 5 | 0 | 0 | 0 | 0 |
| `_verify_api_models.py` | 78 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| `importers/_native_import_bridge.py` | 78 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_native_linear_constructors.py` | 76 | 3 | 1 | 2 | 0 | 0 | 0 | 0 | 0 |
| `adapters/_shipped.py` | 76 | 5 | 4 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_user_objects.py` | 75 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_sample_weight.py` | 72 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `parallel_graph.py` | 72 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_params_setter.py` | 71 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_native_msel_more.py` | 70 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_native_parallel_cv.py` | 70 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_distributed_program.py` | 68 | 10 | 9 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_native_robust_bridge.py` | 68 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_native_msel_extra.py` | 67 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_native_msel_inputs.py` | 67 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_native_neural_bridge.py` | 67 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_surface_cluster.py` | 67 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_declared_params.py` | 65 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_native_prep_classify_bridge.py` | 61 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `language_model.py` | 61 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_tsvd_bridge.py` | 60 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| `_verify_resources.py` | 60 | 3 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| `_native_runtime_manifold_bridge.py` | 59 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_verify_light_faults.py` | 59 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| `models/__init__.py` | 59 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `parallel_model_selection.py` | 59 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_verify_api_optimizers.py` | 58 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| `_verify_api_programs.py` | 58 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| `_native_maxabs_bridge.py` | 56 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_sklearn_modules.py` | 56 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `mamba.py` | 54 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_surface_ann.py` | 53 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `parallel_forecasting.py` | 53 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_family_layouts.py` | 50 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_partitioned_program.py` | 50 | 10 | 10 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_verification_profiles.py` | 50 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| `parallel_ivf.py` | 50 | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_iforest_depth.py` | 49 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_native_linear_score_bridge.py` | 49 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `exceptions.py` | 49 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `models/_catalog.py` | 49 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_host_import_aliases.py` | 48 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_native_numeric_profile.py` | 48 | 5 | 4 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_arrays.py` | 47 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_ipca_bridge.py` | 47 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| `parallel_gaussian_process.py` | 47 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_owner_lifetime.py` | 46 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `model_pool_training.py` | 46 | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_msel_scorers.py` | 44 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_native_program_save_bridge.py` | 42 | 4 | 1 | 3 | 0 | 0 | 0 | 0 | 0 |
| `importers/onnx.py` | 42 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_parallel_neural_admission.py` | 40 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `importers/__init__.py` | 40 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_verification_ctr_models.py` | 39 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| `offload_training.py` | 39 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_msel_search.py` | 38 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_native_array_storage.py` | 37 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_operation_lock.py` | 37 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_lazy_out.py` | 36 | 2 | 0 | 0 | 0 | 0 | 1 | 1 | 12 |
| `transformer.py` | 35 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_verification_sources.py` | 34 | 3 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| `_native_ops.py` | 33 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_verify_api_availability.py` | 32 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| `_byte_lm_trainer_host.py` | 28 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_cpu_reference.py` | 28 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| `_native_array_protocol_bridge.py` | 28 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_native_model_selection_bridge.py` | 28 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_native_parallel_shard_worker.py` | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_residual_dropout_impl.py` | 27 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_mode_properties.py` | 26 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_native_jl_bridge.py` | 25 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `adapters/__init__.py` | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_runtime_fallbacks.py` | 23 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| `_byte_lm_checkpoint.py` | 22 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `_parallel_mlp.py` | 22 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_verify_public.py` | 21 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| `_optional_numpy.py` | 20 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_parallel_forecast_worker.py` | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_parallel_neural_worker.py` | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `neural_network.py` | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_parallel_neural.py` | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_parallel_cv_worker.py` | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_runtime_sequential_bridge.py` | 15 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `manifold.py` | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_process.py` | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `adapters/_torch_compile.py` | 12 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `umap.py` | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_class_weights.py` | 8 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `gaussian_process.py` | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `impute.py` | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `kernel_approximation.py` | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_native_parallel_causal_worker.py` | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `calibration.py` | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `covariance.py` | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `cross_decomposition.py` | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `discriminant_analysis.py` | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `feature_selection.py` | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `isotonic.py` | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `kernel_ridge.py` | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `multiclass.py` | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `multioutput.py` | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `naive_bayes.py` | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `random_projection.py` | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `semi_supervised.py` | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `tree.py` | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_version.py` | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `_verification_light_reference.py` | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## (f) Dead code deleted by this lane

128 definitions, 1,306 lines (plus attached comments and imports left unused). One commit per file on `lane/python-package-audit`.

| file:line (before) | definition | lines |
|---|---|---:|
| `__main__.py:111` | `_ships_card` | 6 |
| `_bufcheck.py:177` | `ctypes_view` | 5 |
| `_byte_lm_host.py:45` | `_parameters_sha256` | 4 |
| `_byte_lm_host.py:374` | `LanguageModelHostTrainer._moment` | 9 |
| `_byte_lm_impl.py:531` | `_gpu_logits_ids` | 18 |
| `_byte_lm_impl.py:541` | `_require_written` | 3 |
| `_byte_lm_impl.py:907` | `SmallByteLanguageModelTrainer._export_gradients_impl` | 15 |
| `_byte_lm_impl.py:1307` | `SmallByteLanguageModelTrainer._logits_call` | 23 |
| `_byte_lm_impl.py:1367` | `SmallByteLanguageModelTrainer._next_bytes_resident` | 3 |
| `_byte_lm_impl.py:1371` | `SmallByteLanguageModelTrainer._logits_stateless` | 15 |
| `_byte_lm_impl.py:1387` | `SmallByteLanguageModelTrainer._logits_resident` | 22 |
| `_byte_lm_impl.py:1530` | `_decode_checkpoint` | 40 |
| `_byte_lm_impl.py:1572` | `_unique_object` | 7 |
| `_classical_host.py:159` | `_HostBound._host_refusals` | 3 |
| `_classical_host.py:334` | `HostLogisticRegression._host_refusals` | 13 |
| `_classical_host.py:971` | `HostGaussianProcessClassifier._host_refusals` | 3 |
| `_classical_host.py:1000` | `HostCholesky._host_refusals` | 3 |
| `_classical_host.py:1150` | `_HostKernelMethod._host_refusals` | 7 |
| `_expansion_ann.py:42` | `_ann_mask` | 17 |
| `_expansion_ann.py:114` | `_AnnResident._resident_search` | 16 |
| `_expansion_ann.py:739` | `CagraIndex._search_k` | 32 |
| `_expansion_ann.py:772` | `CagraIndex._search_filtered` | 22 |
| `_expansion_cluster.py:108` | `_XCluster._input_like_fit` | 6 |
| `_expansion_cnn.py:48` | `_pair` | 7 |
| `_expansion_cnn.py:63` | `_uniform` | 6 |
| `_expansion_cnn.py:71` | `_kaiming_uniform` | 7 |
| `_expansion_cnn.py:424` | `_ones_dev` | 8 |
| `_expansion_cnn.py:481` | `Conv2d._pad_params` | 3 |
| `_expansion_cnn.py:539` | `Conv2d._pad_t` | 9 |
| `_expansion_cnn.py:549` | `Conv2d._group_prm` | 4 |
| `_expansion_cnn.py:554` | `Conv2d._check_x` | 5 |
| `_expansion_cnn.py:563` | `Conv2d._group_dev` | 7 |
| `_expansion_cnn.py:570` | `Conv2d._backward_t` | 60 |
| `_expansion_cnn.py:634` | `Conv2d._bind_input` | 13 |
| `_expansion_cnn.py:675` | `_t4` | 5 |
| `_expansion_cnn.py:682` | `_t3` | 2 |
| `_expansion_cnn.py:866` | `BatchNorm2d._sync_running` | 11 |
| `_expansion_cnn.py:894` | `BatchNorm2d._grads_host` | 15 |
| `_expansion_cnn.py:1140` | `_AdaptivePool._divides` | 2 |
| `_expansion_decomp.py:737` | `_Kit._idn_flags` | 11 |
| `_expansion_decomp.py:1407` | `_Base._kit` | 2 |
| `_expansion_linear.py:56` | `_coef_block` | 14 |
| `_expansion_metrics.py:293` | `_pair` | 16 |
| `_expansion_metrics.py:707` | `_assemble` | 9 |
| `_expansion_metrics.py:811` | `_scores` | 15 |
| `_expansion_metrics.py:1136` | `_mix64` | 6 |
| `_expansion_neighbors.py:253` | `_XNeighbors._rowsum` | 4 |
| `_expansion_neighbors.py:258` | `_XNeighbors._colsum` | 4 |
| `_expansion_neighbors.py:263` | `_XNeighbors._scale_div` | 4 |
| `_expansion_neighbors.py:280` | `_XNeighbors._variance` | 4 |
| `_expansion_neighbors.py:307` | `_p2m_sort_rows` | 18 |
| `_expansion_neighbors.py:451` | `LocalOutlierFactor._novelty` | 3 |
| `_expansion_trees.py:484` | `_trees_fast_tier` | 7 |
| `_expansion_trees.py:487` | `_TreesEnsembleBase._arange` | 8 |
| `_expansion_trees.py:493` | `_trees_switch` | 7 |
| `_expansion_trees.py:496` | `_TreesEnsembleBase._class_major` | 10 |
| `_expansion_trees.py:502` | `_trees_build_switch` | 6 |
| `_expansion_trees.py:510` | `_trees_py2mojo` | 8 |
| `_expansion_trees.py:543` | `_TreesEnsembleBase._acc_votes` | 3 |
| `_expansion_trees.py:550` | `_TreesEnsembleBase._argmax` | 4 |
| `_expansion_trees.py:592` | `_TreesEnsembleBase._codes_of` | 4 |
| `_expansion_trees.py:642` | `_BaggingBase._slice_members` | 3 |
| `_forest_protocol.py:101` | `_frozen_copy` | 3 |
| `_forest_protocol.py:229` | `ForestProtocol._resident_ordered_flag` | 18 |
| `_forest_protocol.py:272` | `ForestProtocol._prepare_resident_forest` | 40 |
| `_gp_impl.py:100` | `_as_length_scale` | 15 |
| `_gp_impl.py:122` | `_as_bounds` | 24 |
| `_gp_impl.py:169` | `Kernel._with_free_values` | 5 |
| `_gp_impl.py:196` | `Kernel._free_bounds` | 2 |
| `_ivf_impl.py:351` | `IVFIndex._search_filter` | 11 |
| `_lazy_out.py:25` | `_mapped_out` | 15 |
| `_mlp_impl.py:376` | `_unique_object` | 7 |
| `_samba_impl.py:352` | `SambaStack._refuse_no_backward` | 9 |
| `_samba_impl.py:412` | `SambaStack._experiment_profile` | 7 |
| `_samba_impl.py:471` | `SambaStack._read_legacy_checkpoint` | 4 |
| `_serialize.py:291` | `read_npy` | 2 |
| `_sk_validate.py:319` | `_integral` | 8 |
| `_training_impl.py:829` | `_f32_bits` | 2 |
| `_training_impl.py:851` | `_lr_table_wanted` | 3 |
| `_training_impl.py:861` | `_lr_buffers` | 3 |
| `_training_impl.py:1177` | `Generator._next_stream` | 2 |
| `_transformer_impl.py:202` | `_block_options` | 126 |
| `_transformer_impl.py:377` | `_want_shape` | 10 |
| `_tsa_impl.py:537` | `ExponentialSmoothing._check_dims` | 44 |
| `_tsa_impl.py:611` | `ExponentialSmoothing._set_fitted` | 49 |
| `_tsa_impl.py:679` | `ExponentialSmoothing._shaped` | 10 |
| `_x_sequence_device.py:37` | `resident_binding` | 3 |
| `_x_sequence_device.py:105` | `on_binding` | 3 |
| `_x_sequence_model.py:38` | `_x_2d` | 7 |
| `_x_sequence_model.py:39` | `_positive` | 4 |
| `_x_sequence_model.py:47` | `_x_nd` | 9 |
| `_x_sequence_model.py:58` | `_shape_arg` | 10 |
| `_x_sequence_model.py:143` | `Sequential._append` | 4 |
| `_x_sequence_model.py:151` | `Sequential._layer_norm` | 2 |
| `_x_sequence_model.py:185` | `Sequential._dropout` | 2 |
| `_x_sequence_model.py:453` | `Sequential._import_shape` | 26 |
| `_x_sequence_moe.py:131` | `MoEBlock._routing_out` | 2 |
| `_x_sequence_moe.py:152` | `MoEBlock._weights_handle` | 4 |
| `_x_sequence_moe.py:157` | `MoEBlock._free_weights` | 2 |
| `_x_sequence_moe.py:165` | `MoEBlock._check_weights` | 3 |
| `_x_sequence_norm.py:187` | `LayerNorm._dev_binding` | 3 |
| `_x_sequence_optim.py:37` | `_ResidentState._state_binding` | 2 |
| `_x_sequence_optim.py:40` | `_ResidentState._slot_addrs` | 2 |
| `_x_sequence_optim.py:64` | `_ResidentState._lr_for_step` | 2 |
| `_x_sequence_optim.py:397` | `Adafactor._af_res` | 4 |
| `cross_vendor.py:62` | `_recv_exact` | 2 |
| `density.py:30` | `_check_queries` | 19 |
| `extratrees.py:170` | `_max_features_slots` | 18 |
| `importers/_graph.py:51` | `_message` | 2 |
| `kernel_methods.py:101` | `_real` | 7 |
| `model_selection.py:520` | `_as_index` | 15 |
| `model_selection.py:537` | `_arange_rows` | 6 |
| `model_selection.py:563` | `_msel_native` | 2 |
| `model_selection.py:569` | `_row_means` | 13 |
| `model_selection.py:604` | `_check_splits` | 7 |
| `models/causal_lm.py:208` | `_prompt_then_new` | 20 |
| `models/safetensors.py:72` | `_parse_header` | 8 |
| `ops.py:123` | `_reduction` | 4 |
| `ops.py:129` | `_axis` | 4 |
| `ops.py:140` | `_dims` | 7 |
| `ops.py:146` | `_keepdims` | 4 |
| `ops.py:330` | `Session._is_handle` | 3 |
| `parallel_training.py:108` | `ParallelByteLanguageModelTrainer._require_split_step` | 2 |
| `tokenizer.py:121` | `_blob` | 9 |
| `tokenizer.py:132` | `_ro` | 2 |
| `tokenizer.py:225` | `BpeTokenizer._from_handle` | 3 |
| `tokenizer.py:413` | `BpeTokenizer._id_list` | 8 |
| `tokenizer.py:504` | `TrainedBpeVocabulary._render` | 2 |

## (c) Tool code in the wheel

The 50 (c) modules (`_verify*`, `_verification_*`, `_conformance`, `_identity`, `_gpu_witness`, `_parallel_cv_witness`, `_causal_lm_fixtures`, `_cpu_reference`, `_runtime_fallbacks`, `__main__`) stay in the wheel: `mojolearn.__main__` imports them for `python -m mojolearn verify` and `conformance`, which the release's one verification runs from the installed wheel. Moving them out would remove a shipped command, so no (c) file was moved.

Test-only definitions moved out of product modules into the test that reads them:

| from | to | definition | lines |
|---|---|---|---:|
| `python/mojolearn/_expansion_trees.py` | `python/mojolearn/tests/test_x_trees_sanity.py` | `_FoldRows + _trees_native_folds` | 39 |
| `python/mojolearn/_cast_in.py` | `python/mojolearn/tests/_cast_in.py` | `module (describe kept; as_input, matrix_view dropped)` | 56 |

Definitions in product modules whose only callers are tests or tools (left in place; move with their tests when those are migrated to Mojo per AGENTS.md):

| file:line | definition | lines | callers |
|---|---|---:|---|
| `_array.py:521` | `Array._row_view` | 14 | `python/mojolearn/tests/_array_reference.py`, `python/mojolearn/tests/test_native_array_view_lifetime.py` +1 |
| `_byte_lm_host.py:60` | `_native_shape` | 6 | `python/mojolearn/tests/test_byte_lm_host_trainer.py` |
| `_byte_lm_host.py:136` | `_thread_count` | 9 | `tools/identity_break.py` |
| `_byte_lm_host.py:147` | `_flatten_parameters` | 23 | `tools/test_framework_native_provider.py` |
| `_byte_lm_impl.py:154` | `_require_schedule_vocabulary` | 15 | `tools/lm_byte_path_main_copy.py` |
| `_byte_lm_impl.py:213` | `_round_f32` | 4 | `bench/speed/python_hotpath_other_lanes.py`, `python/mojolearn/tests/test_generator_draw_native.py` |
| `_byte_lm_impl.py:219` | `_float32` | 10 |  |
| `_byte_lm_impl.py:231` | `_configuration` | 13 | `tools/bench_board_neural.py` |
| `_byte_lm_impl.py:351` | `_runtime_profile` | 4 | `python/mojolearn/tests/test_parallel_byte_native.py` |
| `_byte_lm_impl.py:373` | `_validate_state` | 41 | `python/mojolearn/tests/test_byte_lm_checkpoint_binary.py`, `tools/bench_board_neural.py` |
| `_byte_lm_impl.py:416` | `_sha256_hex` | 5 | `python/mojolearn/tests/test_native_steady_state.py` |
| `_cholesky_impl.py:152` | `Cholesky._door_binding` | 2 | `python/mojolearn/tests/test_cholesky_cpu_inference.py` |
| `_classical_host.py:1081` | `_HostCD._solver` | 2 | `bench/bench_sklearn.py`, `packaging/macos/smoke.py` +3 |
| `_expansion_ann.py:653` | `CagraIndex._rs` | 10 | `tools/gemm_remote_leg.sh` |
| `_expansion_cluster.py:42` | `_f32` | 9 | `python/mojolearn/tests/_legacy_neighbors_graph.py`, `python/mojolearn/tests/_legacy_neighbors_loc.py` +12 |
| `_expansion_cnn.py:487` | `Conv2d._out_shape` | 3 | `python/mojolearn/tests/test_x_cnn_repeat.py` |
| `_expansion_cnn.py:566` | `_Pool2d._out_shape` | 3 | `python/mojolearn/tests/test_x_cnn_repeat.py` |
| `_expansion_cnn.py:1009` | `BasicBlock._bns` | 2 | `tools/identity_lanes/cnn.py` |
| `_expansion_neighbors.py:79` | `_accuracy` | 6 | `python/mojolearn/tests/_legacy_neighbors_loc.py`, `python/mojolearn/tests/_legacy_neighbors_semi.py` |
| `_expansion_neighbors.py:87` | `_f32_1d` | 3 | `python/mojolearn/tests/_legacy_neighbors_graph.py` |
| `_expansion_neighbors.py:107` | `_argmax_labels` | 8 | `python/mojolearn/tests/_legacy_neighbors_loc.py`, `python/mojolearn/tests/_legacy_neighbors_semi.py` |
| `_expansion_neighbors.py:117` | `_class_array` | 10 | `python/mojolearn/tests/_legacy_neighbors_loc.py` |
| `_expansion_neighbors.py:153` | `_kfeat_flags` | 10 | `python/mojolearn/tests/_legacy_neighbors_semi.py`, `tools/kfeat_quality.py` |
| `_expansion_neighbors.py:165` | `_classical_graph_normalization` | 7 | `python/mojolearn/tests/_legacy_neighbors_semi.py` |
| `_expansion_neighbors.py:178` | `_p2m_relabel` | 13 | `python/mojolearn/tests/_legacy_neighbors_graph.py` |
| `_expansion_neighbors.py:211` | `_XNeighbors._l1dist` | 6 | `python/mojolearn/tests/_legacy_neighbors_loc.py` |
| `_expansion_neighbors.py:218` | `_XNeighbors._fast_tier` | 7 | `python/mojolearn/tests/_suite_env.py` |
| `_expansion_neighbors.py:234` | `_XNeighbors._matmul` | 6 | `python/mojolearn/tests/_legacy_neighbors_graph.py`, `python/mojolearn/tests/_legacy_neighbors_loc.py` +1 |
| `_expansion_neighbors.py:259` | `_XNeighbors._take_cols` | 5 | `python/mojolearn/tests/_legacy_neighbors_loc.py` |
| `_expansion_neighbors.py:601` | `NearestCentroid._decision_matrix` | 12 | `python/mojolearn/tests/_legacy_neighbors_loc.py` |
| `_expansion_neighbors.py:1030` | `_seed_word` | 19 | `python/mojolearn/tests/_legacy_neighbors_semi.py` |
| `_expansion_neighbors.py:1775` | `KNNImputer._nan_cells` | 16 | `python/mojolearn/tests/_legacy_neighbors_loc.py`, `tools/knn_lean_gpu_quality.py` |
| `_expansion_neighbors.py:1792` | `KNNImputer._masked` | 8 | `python/mojolearn/tests/_legacy_neighbors_loc.py`, `tools/knn_lean_gpu_quality.py` |
| `_expansion_prep.py:132` | `_native_helper` | 4 | `python/mojolearn/tests/_former_prep_labels.py` |
| `_expansion_prep.py:151` | `_p2m_chunks` | 5 |  |
| `_expansion_prep.py:213` | `_switch` | 2 |  |
| `_expansion_prep.py:217` | `_blocked` | 2 | `python/mojolearn/tests/test_native_prep_scores_door.py`, `python/mojolearn/tests/test_samba_config_native.py` +1 |
| `_expansion_prep.py:221` | `_col_stats` | 15 | `tools/afc_shape_data.py` |
| `_expansion_prep.py:249` | `_ptimpute_flags` | 10 | `tools/identical_gmm_power_bit_diagnostic.py`, `tools/pt_score_quality.py` |
| `_expansion_prep.py:506` | `_fast_on` | 15 |  |
| `_expansion_prep.py:539` | `_x2d` | 16 | `python/mojolearn/tests/_former_prep_labels.py`, `python/mojolearn/tests/test_native_prep_iterimp_door.py` +2 |
| `_expansion_prep.py:788` | `_cls2_cat` | 12 |  |
| `_expansion_prep.py:802` | `_fit_categories_cls2` | 50 |  |
| `_expansion_prep.py:854` | `_fit_categories` | 37 | `python/mojolearn/tests/_former_prep_labels.py`, `python/mojolearn/tests/test_native_prep_select_door.py` |
| `_expansion_prep.py:1109` | `_stage_folds` | 29 |  |
| `_expansion_prep.py:1140` | `_refuse_folds` | 11 |  |
| `_expansion_prep.py:1153` | `_native_folds` | 12 | `python/mojolearn/tests/test_x_prep_target_encoder.py`, `python/mojolearn/tests/test_x_trees_sanity.py` |
| `_expansion_prep.py:1170` | `_target_scratch` | 9 | `tools/target_scratch_quality.py` |
| `_expansion_prep.py:1584` | `_seed_words` | 4 | `python/mojolearn/tests/test_native_prep_iterimp_door.py` |
| `_expansion_prep.py:1599` | `_idn_fam` | 17 | `python/mojolearn/tests/test_native_prep_scores_door.py`, `tools/identical_gmm_power_bit_diagnostic.py` |
| `_expansion_prep.py:1618` | `_cs` | 9 | `python/mojolearn/tests/test_native_prep_iterimp_door.py`, `python/mojolearn/tests/test_native_prep_scores_door.py` +1 |
| `_expansion_prep.py:1629` | `_cls` | 10 | `packaging/check_ext_lists.py`, `packaging/linux/build_sets.sh` +15 |
| `_expansion_prep.py:1645` | `_class_stats` | 29 |  |
| `_expansion_prep.py:1764` | `_DiscreteNB._nb_csr_ready` | 24 | `tools/bench_board_algos.py` |
| `_expansion_prep.py:2625` | `_label_buffer` | 29 | `python/mojolearn/tests/_former_prep_labels.py`, `python/mojolearn/tests/test_native_prep_select_door.py` |
| `_expansion_prep.py:2656` | `_label_chunk` | 11 | `python/mojolearn/tests/_former_prep_labels.py` |
| `_expansion_prep.py:2716` | `_class_words` | 7 | `python/mojolearn/tests/_former_prep_labels.py` |
| `_expansion_trees.py:503` | `_TreesEnsembleBase._gather_codes` | 5 | `tools/test_native_metadata_snapshot.py` |
| `_expansion_trees.py:509` | `_TreesEnsembleBase._gather_vec` | 4 | `tools/test_native_metadata_snapshot.py` |
| `_expansion_trees.py:514` | `_TreesEnsembleBase._acc_cols` | 4 | `tools/test_native_metadata_snapshot.py` |
| `_forest_host.py:168` | `HostForest._groves_handle` | 4 | `python/mojolearn/tests/_legacy_host_trees.py`, `python/mojolearn/tests/test_host_tree_native.py` |
| `_forest_host.py:186` | `HostForest._vote` | 5 | `python/mojolearn/tests/_legacy_host_trees.py`, `python/mojolearn/tests/test_host_tree_native.py` |
| `_forest_protocol.py:126` | `ForestProtocol._refresh_config` | 12 | `python/mojolearn/tests/test_forest_protocol.py` |
| `_forest_protocol.py:148` | `ForestProtocol._capture_fit_mode` | 5 | `python/mojolearn/tests/test_forest_inference_engine.py` |
| `_forest_protocol.py:223` | `ForestProtocol._forest_auxiliary` | 3 | `python/mojolearn/tests/test_native_forest_auxiliary.py`, `tools/test_native_forest_auxiliary_connection.py` |
| `_gbdt_adapters.py:304` | `GradientBoostingRegressor._regression_target` | 11 | `python/mojolearn/tests/test_list_input_float32.py` |
| `_gp_impl.py:145` | `Kernel._free_flags` | 2 | `python/mojolearn/tests/test_gp_optimizer.py` |
| `_gp_impl.py:148` | `Kernel._free_values` | 2 | `tools/gp_optimize_oracle_check.py`, `tools/identity_break.py` |
| `_identity.py:160` | `_judge` | 23 | `tools/checks_that_cannot_fail_probe.py` |
| `_ivf_impl.py:95` | `_int_param` | 4 | `python/mojolearn/tests/_legacy_neighbors_ann.py` |
| `_linalg_impl.py:375` | `_lowbit_binding` | 16 | `tools/bench_board.py`, `tools/int8_profile.py` |
| `_linalg_impl.py:461` | `_int8_operand` | 11 | `tools/int8_profile.py` |
| `_linalg_impl.py:520` | `_out_or_new` | 11 | `tools/int8_profile.py` |
| `_linalg_impl.py:582` | `_door` | 13 | `python/mojolearn/tests/test_cholesky_cpu_inference.py`, `python/mojolearn/tests/test_linalg_decompositions.py` +1 |
| `_metrics_impl.py:184` | `_vector_shape` | 12 |  |
| `_metrics_impl.py:208` | `_as_i32_1d` | 23 | `bench/speed/python_hotpath_cells.py`, `python/mojolearn/tests/test_hotpath_native.py` |
| `_metrics_impl.py:266` | `_prepare_cluster_labels` | 29 | `bench/speed/python_hotpath_cells.py`, `python/mojolearn/tests/test_hotpath_native.py` |
| `_metrics_impl.py:303` | `_native_union_codes` | 24 |  |
| `_metrics_impl.py:1226` | `_classification_pair` | 12 | `bench/speed/python_hotpath_cells.py`, `python/mojolearn/tests/test_hotpath_native.py` |
| `_metrics_impl.py:1423` | `_encode_classification` | 4 | `bench/speed/python_hotpath_cells.py`, `python/mojolearn/tests/test_hotpath_native.py` |
| `_mlp_impl.py:149` | `_batch` | 15 | `python/mojolearn/tests/test_arima_surface.py`, `python/mojolearn/tests/test_neural_surface_consistency.py` |
| `_mlp_impl.py:376` | `_encode_array` | 13 |  |
| `_mlp_impl.py:391` | `_encode_state` | 5 | `python/mojolearn/tests/test_neural_inference.py`, `python/mojolearn/tests/test_small_mlp_surface.py` |
| `_samba_impl.py:57` | `_canonical` | 3 | `python/mojolearn/tests/_samba_checkpoint.py`, `python/mojolearn/tests/test_byte_lm_session.py` +8 |
| `_spectral_impl.py:416` | `SpectralClustering._predict_embedding` | 4 | `python/mojolearn/tests/test_spectral_predict.py`, `tools/test_spectral_umap_native_routes.py` |
| `_spectral_impl.py:660` | `SpectralEmbedding._precomputed_knn_affinity` | 7 | `python/mojolearn/tests/test_native_dense_coo.py`, `tools/py_consolidated/kern_bench.py` +1 |
| `_training_impl.py:219` | `_is_buffer` | 6 | `python/mojolearn/tests/test_training_cross_entropy_public_native.py` |
| `_training_impl.py:844` | `_lr_native` | 3 | `tools/sched_table_check.py`, `tools/verify_coverage_audit.py` |
| `_training_impl.py:1095` | `_rng_call` | 14 | `python/mojolearn/tests/test_cpu_training_samba.py`, `python/mojolearn/tests/test_generator_draw_native.py` +1 |
| `_training_impl.py:1269` | `_c32` | 5 | `python/mojolearn/tests/test_generator_draw_native.py`, `python/mojolearn/tests/test_training_embedding_public_native.py` +2 |
| `_transformer_impl.py:183` | `_f32_bits` | 17 | `tools/lm_cpu_witness.py`, `tools/lm_segment.py` +2 |
| `_transformer_impl.py:202` | `_default_tail` | 10 | `python/mojolearn/tests/test_transformer_options.py` |
| `_transformer_impl.py:214` | `_f32_strict` | 33 | `python/mojolearn/tests/_mamba_block_reference.py` |
| `_transformer_impl.py:526` | `TransformerBlock._with_tails` | 7 | `python/mojolearn/tests/_transformer_block_reference.py`, `python/mojolearn/tests/test_transformer_options.py` |
| `_tsa_impl.py:712` | `ExponentialSmoothing._named` | 10 | `tools/test_verify_coverage_rules.py`, `tools/verify_coverage_audit.py` |
| `_x_sequence_model.py:278` | `Sequential._sample_of` | 3 | `python/mojolearn/tests/test_sequential_native_surface.py`, `tools/tests/sequential_admission_fixture.py` |
| `_x_sequence_model.py:339` | `Sequential._param_layers` | 3 | `tools/tests/sequential_admission_fixture.py` |
| `_x_sequence_model.py:406` | `Sequential._save` | 2 | `tools/lm_run_driver.py`, `tools/neural_fixtures.py` |
| `_x_sequence_rnn.py:64` | `_RecurrentBase._cell` | 6 | `docs-site/tools/gen_models.py`, `python/mojolearn/tests/test_x_sequence_rnn.py` +11 |
| `cross_vendor.py:29` | `_f32` | 2 | `python/mojolearn/tests/_legacy_neighbors_graph.py`, `python/mojolearn/tests/_legacy_neighbors_loc.py` +12 |
| `cross_vendor.py:54` | `_send` | 2 | `python/mojolearn/tests/test_cpu_training_par_classical.py`, `python/mojolearn/tests/test_cross_vendor.py` +3 |
| `cross_vendor.py:58` | `_recv` | 2 | `python/mojolearn/tests/test_cross_vendor.py` |
| `cross_vendor.py:102` | `Coordinator._drive` | 2 | `python/mojolearn/tests/test_cross_vendor_coordinator_native.py` |
| `decomposition.py:23` | `_saved_mode` | 9 | `python/mojolearn/tests/_legacy_density_archive.py`, `python/mojolearn/tests/test_host_model_kmeans.py` +1 |
| `decomposition.py:34` | `_restore_mode` | 5 | `tools/lane_select.py` |
| `decomposition.py:41` | `_check_saved_by` | 8 | `tools/lane_select.py` |
| `density.py:442` | `KernelDensity._release_resident_fit` | 5 | `python/mojolearn/tests/_legacy_host_inference.py`, `python/mojolearn/tests/test_host_kde_fit_native.py` +1 |
| `ensemble.py:920` | `GradientBoosting._resolved_boosting_type` | 15 | `python/mojolearn/tests/test_gbdt_catboost_defaults.py` |
| `ensemble.py:993` | `GradientBoosting._resolved_learning_rate` | 4 | `python/mojolearn/tests/test_gbdt_catboost_defaults.py` |
| `linear_model.py:51` | `_saved_mode` | 9 | `python/mojolearn/tests/_legacy_density_archive.py`, `python/mojolearn/tests/test_host_model_kmeans.py` +1 |
| `linear_model.py:68` | `_round_f32` | 6 | `bench/speed/python_hotpath_other_lanes.py`, `python/mojolearn/tests/test_generator_draw_native.py` |
| `linear_model.py:81` | `_labels_1d` | 9 | `bench/speed/python_hotpath_other_lanes.py` |
| `linear_model.py:145` | `_accuracy_host` | 11 | `bench/speed/python_hotpath_other_lanes.py` |
| `linear_model.py:245` | `_vector_mean` | 13 | `bench/speed/python_hotpath_other_lanes.py`, `tools/kmeans_linear_profile.py` |
| `linear_model.py:290` | `_check_sample_weight` | 15 | `bench/speed/python_hotpath_other_lanes.py`, `python/mojolearn/tests/_legacy_neighbors_kde.py` +1 |
| `linear_model.py:1306` | `_qn_weight_fields` | 4 |  |
| `linear_model.py:1312` | `_qn_fit_one_target` | 22 | `python/mojolearn/tests/test_qn_objectives_surface.py` |
| `lm_corpus.py:291` | `TokenBatches._batches` | 2 | `python/mojolearn/tests/test_byte_lm_train_steps.py`, `tools/lm_shakedown_resume.py` |
| `lm_corpus.py:313` | `_expected_vocabulary` | 4 | `python/mojolearn/tests/test_lm_corpus_native_boundary.py`, `tools/test_lm_corpus_public_native_source.py` |
| `model_selection.py:96` | `_overlap` | 5 | `bench/speed/python_hotpath_cells.py`, `python/mojolearn/tests/test_hotpath_native.py` |
| `model_selection.py:168` | `_clone` | 24 | `bench/x_msel_speed.py`, `python/mojolearn/tests/test_classical_pip_smoke_sep22.py` +6 |
| `model_selection.py:1855` | `GridSearchCV._candidates` | 2 | `tools/declared_constructor_schemas.py` |
| `model_selection.py:1922` | `RandomizedSearchCV._candidates` | 2 | `tools/declared_constructor_schemas.py` |
| `models/causal_lm.py:187` | `_float_weight` | 5 | `python/mojolearn/tests/test_causal_model_native_loading.py` |
| `models/causal_lm.py:194` | `_argmax_last` | 12 | `python/mojolearn/tests/test_models_loader.py`, `tools/identity_break.py` |
| `models/parallel_causal_lm.py:38` | `_admit_route` | 6 | `python/mojolearn/tests/test_parallel_causal_lm.py` |
| `models/parallel_causal_lm.py:123` | `ParallelCausalLM._rpc` | 2 | `python/mojolearn/tests/test_parallel_causal_lm.py` |
| `models/tokenizer.py:73` | `Tokenizer._from_spelled` | 4 | `tools/identity_lanes/coverage.py`, `tools/verify_coverage_audit.py` |
| `models/tokenizer.py:85` | `Tokenizer._from_json` | 3 | `python/mojolearn/tests/test_models_loader.py`, `python/mojolearn/tests/test_tokenizer_native_loading.py` +2 |
| `neighbors.py:170` | `_check_metric_input` | 7 | `python/mojolearn/tests/_legacy_neighbors_knn.py` |
| `neighbors.py:576` | `NearestNeighbors._dist_params` | 6 | `python/mojolearn/tests/_legacy_neighbors_knn.py` |
| `ops.py:207` | `Session._sid` | 4 | `tools/test_native_import_boundary.py` |
| `ops.py:616` | `Session._compare` | 2 | `python/mojolearn/tests/test_native_prep_labels_door.py`, `tools/byte_lm_gradient_oracle.py` +4 |
| `parallel_training.py:124` | `ParallelByteLanguageModelTrainer._require_fold` | 2 | `tools/lm_segment.py`, `tools/tests/test_lm_segment_controls.py` |
| `preprocessing.py:39` | `_seen` | 9 | `tools/bench_board_params.py`, `tools/runpod_cpu_leg.sh` |
| `preprocessing.py:51` | `_ScalerProtocol._input` | 26 | `python/mojolearn/tests/test_classical_archive_reader_native.py`, `python/mojolearn/tests/test_host_iforest_native.py` +9 |
| `randomforest.py:270` | `_max_features_fraction` | 32 | `packaging/portable_math/test_host_math.py` |
| `tokenizer.py:90` | `_identity_from_tokens_text` | 3 | `tools/test_tokenizer_public_native_source.py` |
| `tokenizer.py:95` | `_identity_of_ranks_file` | 5 | `tools/test_tokenizer_public_native_source.py` |
| `tokenizer.py:102` | `_identity_of_handle` | 2 | `tools/test_tokenizer_public_native_source.py` |
| `tokenizer.py:270` | `BpeTokenizer._synthetic` | 7 | `python/mojolearn/tests/test_public_aliases.py`, `python/mojolearn/tests/test_tokenizer_surface.py` +1 |
| `tokenizer.py:318` | `BpeTokenizer._flag` | 7 | `tools/identity_break.py`, `tools/test_estimator_provider_frontdoors.py` |
| `tokenizer.py:384` | `BpeTokenizer._encode_corpus` | 6 | `tools/test_lm_corpus_public_native_source.py`, `tools/test_tokenizer_public_native_source.py` |
| `tokenizer.py:590` | `_native_trainer` | 4 | `tools/test_tokenizer_public_native_source.py` |
| `tokenizer.py:596` | `_train_native` | 5 | `tools/test_tokenizer_public_native_source.py` |
| `tokenizer.py:603` | `_read_trained` | 2 | `tools/test_tokenizer_public_native_source.py` |
| `tokenizer.py:607` | `_train_corpus` | 5 | `tools/test_tokenizer_public_native_source.py` |
| `tokenizer.py:614` | `_gather_rows` | 4 | `tools/test_tokenizer_public_native_source.py` |

## (e) Product work in Python, by family

Do not patch these in Python. Each family is one lane: move the work into the named Mojo door, delete the Python, and keep the public method one binding call.

| family | Mojo door that should own it | functions | Python lines | est. Mojo lines |
|---|---|---:|---:|---:|
| neural-lm | bindings/_mojolearn_byte_lm*.mojo, _mojolearn_transformer*.mojo, _mojolearn_training*.mojo, _mojolearn_tokenizer_host.mojo | 31 | 728 | ~1092 |
| runtime-boundary | bindings/backend_public_native.mojo, array_native.mojo, native_array_*.mojo, native_buffer_*.mojo | 32 | 693 | ~1039 |
| decomp-linalg | bindings/decomp_native.mojo, decomp_kit_native.mojo, linalg_decomp_native.mojo | 21 | 335 | ~502 |
| metrics | bindings/metrics_*_native.mojo | 7 | 133 | ~199 |
| cnn | bindings/_mojolearn_x_cnn*.mojo | 6 | 98 | ~147 |
| model-selection | bindings/model_selection_native.mojo, msel_*_native.mojo | 2 | 57 | ~85 |
| prep | bindings/prep_*_native.mojo | 5 | 52 | ~78 |
| neighbors-ann-cluster-gp | bindings/neighbors_*_native.mojo, cluster_*_native.mojo, gp_*_native.mojo | 3 | 38 | ~57 |
| sequence-tsa | bindings/_mojolearn_runtime_xseq_host.mojo, xseq_*_native.mojo, _mojolearn_tsa*.mojo | 1 | 11 | ~16 |
| **total** | | 108 | 2145 | |

Estimated Mojo lines are 1.5x the Python lines (typed structs and explicit loops); the table entries already in place cut that for linear and trees.


### neural-lm (728 lines)

Door: bindings/_mojolearn_byte_lm*.mojo, _mojolearn_transformer*.mojo, _mojolearn_training*.mojo, _mojolearn_tokenizer_host.mojo

| file:line | function | lines | why | reached by (public operation) |
|---|---|---:|---|---|
| `_byte_lm_host.py:73` | `_refuse_ids` | 20 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_byte_lm_host.py:108` | `_greedy_next_bytes` | 26 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_byte_lm_impl.py:171` | `_schedule` | 34 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_byte_lm_impl.py:246` | `_array` | 26 | arithmetic on values | import-time or internal helper (no public root found statically) |
| `_byte_lm_impl.py:274` | `_parameters` | 13 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_byte_lm_impl.py:430` | `_binding_metadata` | 36 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_byte_lm_impl.py:476` | `_logits_ids` | 24 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_byte_lm_impl.py:917` | `SmallByteLanguageModelTrainer._session_entries` | 10 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_byte_lm_impl.py:1051` | `SmallByteLanguageModelTrainer._run_stateless` | 35 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_byte_lm_impl.py:1087` | `SmallByteLanguageModelTrainer._run_resident` | 52 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_mlp_impl.py:45` | `_schedule` | 35 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_tokenizer_synthetic.py:58` | `_lcg` | 5 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_tokenizer_synthetic.py:65` | `corpus_pieces` | 23 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_tokenizer_synthetic.py:91` | `_vocabulary` | 32 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_tokenizer_synthetic.py:131` | `byte_to_char` | 15 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_tokenizer_synthetic.py:152` | `_decode_cp` | 25 | arithmetic on values | import-time or internal helper (no public root found statically) |
| `_tokenizer_synthetic.py:194` | `_run_end` | 8 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_tokenizer_synthetic.py:204` | `_pretoken_end` | 25 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_tokenizer_synthetic.py:240` | `_bpe` | 17 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_tokenizer_synthetic.py:283` | `cases` | 29 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_tokenizer_synthetic.py:314` | `fixture` | 12 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_tokenizer_synthetic.py:335` | `main` | 16 | loop with value functions (sorted/sum/min/max/math/tolist) | import-time or internal helper (no public root found statically) |
| `_training_impl.py:1161` | `Generator._shape` | 13 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_transformer_impl.py:249` | `_take` | 21 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `models/causal_lm.py:49` | `_accepted_options` | 12 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `models/causal_lm.py:63` | `_block_kwargs` | 18 | loop with value functions (sorted/sum/min/max/math/tolist) | import-time or internal helper (no public root found statically) |
| `models/config.py:254` | `_transformer_common` | 57 | arithmetic on values | import-time or internal helper (no public root found statically) |
| `models/config.py:349` | `_plan_phi3` | 22 | arithmetic on values | import-time or internal helper (no public root found statically) |
| `models/config.py:416` | `_plan_mamba` | 24 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `models/config.py:442` | `_plan_mamba2` | 31 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `tokenizer.py:106` | `_native` | 12 | loop with arithmetic | import-time or internal helper (no public root found statically) |

### runtime-boundary (693 lines)

Door: bindings/backend_public_native.mojo, array_native.mojo, native_array_*.mojo, native_buffer_*.mojo

| file:line | function | lines | why | reached by (public operation) |
|---|---|---:|---|---|
| `_array.py:143` | `_c_strides` | 7 | loop with arithmetic | GaussianMixture.__init__,gpu_plugins.required_sets,host_surface.inference_lanes |
| `_array.py:152` | `_f_strides` | 7 | loop with arithmetic | GaussianMixture.__init__,gpu_plugins.required_sets,host_surface.inference_lanes |
| `_array.py:161` | `_prod` | 5 | loop with arithmetic | GaussianMixture.__init__,gpu_plugins.required_sets,host_surface.inference_lanes |
| `_array.py:409` | `Array._both_orders` | 4 | loop with value functions (sorted/sum/min/max/math/tolist) | parallel_classical.bootstrap,resample.bootstrap,Session.run |
| `_bufcheck.py:127` | `nelems` | 5 | loop with arithmetic | linalg_fast.matmul_fused |
| `_buffer.py:172` | `Buf._fill` | 31 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_buffer.py:252` | `_contiguity` | 28 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_buffer.py:414` | `_materialize` | 116 | loop with value functions (sorted/sum/min/max/math/tolist) | linalg_fast.matmul_fused,parallel_classical.bootstrap,resample.bootstrap |
| `_buffer.py:1142` | `InitStream.child_seed` | 8 | arithmetic on values | import-time or internal helper (no public root found statically) |
| `_classical_host.py:162` | `_HostBound.model_sha256` | 9 | loop with value functions (sorted/sum/min/max/math/tolist) | import-time or internal helper (no public root found statically) |
| `_classical_host.py:1334` | `_host_archive_forest_metadata` | 25 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_labels.py:423` | `finite_integer_codes` | 33 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_lazy_out.py:25` | `_empty_out` | 12 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_mode.py:554` | `ParamsMixin._get_param_names` | 12 | loop with value functions (sorted/sum/min/max/math/tolist) | import-time or internal helper (no public root found statically) |
| `_portable_math.py:86` | `isclose` | 11 | value functions (sorted/sum/min/max/math) | import-time or internal helper (no public root found statically) |
| `_portable_math.py:99` | `comb` | 11 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_portable_math.py:175` | `_scaled_integer` | 27 | arithmetic on values | import-time or internal helper (no public root found statically) |
| `_portable_math.py:277` | `_pow_bound` | 13 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_portable_math.py:294` | `powi` | 36 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_portable_math.py:381` | `_horner` | 6 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_portable_math.py:389` | `erfc` | 44 | arithmetic on values | import-time or internal helper (no public root found statically) |
| `_portable_math.py:438` | `normal_cdf` | 6 | arithmetic on values | import-time or internal helper (no public root found statically) |
| `_portable_math.py:446` | `normal_inv_cdf` | 69 | arithmetic on values | import-time or internal helper (no public root found statically) |
| `_ragged.py:77` | `_lengths_array` | 27 | loop with value functions (sorted/sum/min/max/math/tolist) | import-time or internal helper (no public root found statically) |
| `_ragged.py:112` | `_row_bytes` | 5 | arithmetic on values | import-time or internal helper (no public root found statically) |
| `_serialize.py:73` | `_header` | 14 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_serialize.py:89` | `_parse_header` | 29 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_serialize.py:123` | `_utf32` | 6 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_serialize.py:183` | `_describe_list` | 18 | loop with value functions (sorted/sum/min/max/math/tolist) | import-time or internal helper (no public root found statically) |
| `_serialize.py:233` | `decode_npy` | 43 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_serialize.py:278` | `_nest_text` | 11 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_serialize.py:294` | `write_npz` | 15 | loop with value functions (sorted/sum/min/max/math/tolist) | import-time or internal helper (no public root found statically) |

### decomp-linalg (335 lines)

Door: bindings/decomp_native.mojo, decomp_kit_native.mojo, linalg_decomp_native.mojo

| file:line | function | lines | why | reached by (public operation) |
|---|---|---:|---|---|
| `_expansion_decomp.py:108` | `_pool_give` | 8 | arithmetic on values | import-time or internal helper (no public root found statically) |
| `_expansion_decomp.py:202` | `_M.zeros` | 3 | arithmetic on values | ARIMA.__init__,cross_vendor.fold_pair,ParallelByteLanguageModelTrainer.apply_gradient |
| `_expansion_decomp.py:282` | `_M.rows` | 7 | arithmetic on values | gpu_plugins.validate_baseline_manifest,host_surface.markdown_table,gpu_plugins.validate_bundled_ptx |
| `_expansion_decomp.py:290` | `_M.take_rows` | 18 | arithmetic on values | import-time or internal helper (no public root found statically) |
| `_expansion_decomp.py:312` | `_M.take_cols` | 13 | loop with arithmetic | linalg_fast.matmul_fused,parallel_classical.bootstrap,resample.bootstrap |
| `_expansion_decomp.py:438` | `_vstack` | 5 | loop with value functions (sorted/sum/min/max/math/tolist) | import-time or internal helper (no public root found statically) |
| `_expansion_decomp.py:445` | `_hstack` | 12 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_expansion_decomp.py:557` | `_Kit._use` | 17 | loop with arithmetic | ARIMA.__init__,cross_vendor.fold_pair,ParallelByteLanguageModelTrainer.apply_gradient |
| `_expansion_decomp.py:609` | `_Kit.ew` | 28 | arithmetic on values | import-time or internal helper (no public root found statically) |
| `_expansion_decomp.py:641` | `_Kit.diag_mask` | 15 | arithmetic on values | import-time or internal helper (no public root found statically) |
| `_expansion_decomp.py:658` | `_Kit.mm` | 18 | arithmetic on values | import-time or internal helper (no public root found statically) |
| `_expansion_decomp.py:677` | `_Kit.vstack_dev` | 16 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_expansion_decomp.py:737` | `_Kit.rand` | 12 | arithmetic on values | import-time or internal helper (no public root found statically) |
| `_expansion_decomp.py:806` | `_Kit.take_rows` | 16 | arithmetic on values | import-time or internal helper (no public root found statically) |
| `_expansion_decomp.py:932` | `_Kit.take_cols_m` | 12 | arithmetic on values | import-time or internal helper (no public root found statically) |
| `_expansion_decomp.py:951` | `_Kit.vstack` | 5 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_expansion_decomp.py:957` | `_Kit.hstack` | 15 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_expansion_decomp.py:1133` | `_Kit.lle_local` | 10 | arithmetic on values | import-time or internal helper (no public root found statically) |
| `_expansion_decomp.py:1313` | `_Kit.pad_zero_row` | 18 | arithmetic on values | import-time or internal helper (no public root found statically) |
| `_linalg_impl.py:798` | `_svd_tall` | 69 | arithmetic on values | import-time or internal helper (no public root found statically) |
| `_native_decomp_kit_bridge.py:22` | `_entries` | 18 | loop with arithmetic | import-time or internal helper (no public root found statically) |

### metrics (133 lines)

Door: bindings/metrics_*_native.mojo

| file:line | function | lines | why | reached by (public operation) |
|---|---|---:|---|---|
| `_expansion_metrics.py:160` | `_Prog.stage` | 5 | loop with arithmetic | parallel_classical.bootstrap,resample.bootstrap,Session.run |
| `_expansion_metrics.py:187` | `_Prog._check` | 15 | loop with arithmetic | mojolearn_diagnostics.collect_diagnostics,mojolearn_diagnostics.command_bug_report,mojolearn_diagnostics.command_doctor |
| `_expansion_metrics.py:222` | `_Prog.get` | 7 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_expansion_metrics.py:237` | `_group` | 10 | arithmetic on values | Session.run,Program.run |
| `_expansion_metrics.py:472` | `_Reg.__init__` | 39 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_expansion_metrics.py:901` | `_proba` | 43 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_expansion_metrics.py:1118` | `_fold_rows_run` | 14 | loop with arithmetic | import-time or internal helper (no public root found statically) |

### cnn (98 lines)

Door: bindings/_mojolearn_x_cnn*.mojo

| file:line | function | lines | why | reached by (public operation) |
|---|---|---:|---|---|
| `_expansion_cnn.py:226` | `_size` | 6 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_expansion_cnn.py:481` | `Conv2d._padded_shape` | 5 | arithmetic on values | import-time or internal helper (no public root found statically) |
| `_expansion_cnn.py:731` | `CNNClassifier._build` | 23 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_expansion_cnn.py:846` | `BatchNorm2d._backward_dev` | 15 | arithmetic on values | import-time or internal helper (no public root found statically) |
| `_expansion_cnn.py:1029` | `BasicBlock._conv_back_dev` | 15 | numpy/pandas/torch math | import-time or internal helper (no public root found statically) |
| `_expansion_cnn.py:1045` | `BasicBlock._backward_chain` | 34 | numpy/pandas/torch math | import-time or internal helper (no public root found statically) |

### model-selection (57 lines)

Door: bindings/model_selection_native.mojo, msel_*_native.mojo

| file:line | function | lines | why | reached by (public operation) |
|---|---|---:|---|---|
| `model_selection.py:233` | `_native_default_folds` | 41 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `model_selection.py:276` | `_select_folds` | 16 | loop with arithmetic | import-time or internal helper (no public root found statically) |

### prep (52 lines)

Door: bindings/prep_*_native.mojo

| file:line | function | lines | why | reached by (public operation) |
|---|---|---:|---|---|
| `_expansion_prep.py:421` | `_Prog.stage` | 5 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_expansion_prep.py:444` | `_Prog._check` | 9 | loop with arithmetic | gpu_plugins.distribution_rows,gpu_plugins.meta_requirements,gpu_plugins.payload_for |
| `_expansion_prep.py:454` | `_Prog._read` | 16 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_expansion_prep.py:893` | `_category_block` | 7 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_expansion_prep.py:902` | `_codes` | 15 | loop with arithmetic | import-time or internal helper (no public root found statically) |

### neighbors-ann-cluster-gp (38 lines)

Door: bindings/neighbors_*_native.mojo, cluster_*_native.mojo, gp_*_native.mojo

| file:line | function | lines | why | reached by (public operation) |
|---|---|---:|---|---|
| `_expansion_ann.py:289` | `IVFPQIndex._resident_arrays` | 7 | arithmetic on values | import-time or internal helper (no public root found statically) |
| `_expansion_cluster.py:53` | `_aux` | 15 | loop with arithmetic | import-time or internal helper (no public root found statically) |
| `_native_neighbor_queries.py:32` | `_routes` | 16 | loop with arithmetic | import-time or internal helper (no public root found statically) |

### sequence-tsa (11 lines)

Door: bindings/_mojolearn_runtime_xseq_host.mojo, xseq_*_native.mojo, _mojolearn_tsa*.mojo

| file:line | function | lines | why | reached by (public operation) |
|---|---|---:|---|---|
| `_tsa_impl.py:78` | `_one_series` | 11 | arithmetic on values | import-time or internal helper (no public root found statically) |

## Linear and trees: nothing left to move

The (e) items found in the two families whose doors already take typed requests were Python copies of work the doors already do, with no Python caller left: `_expansion_linear._coef_block` (native in `bindings/x_linear_native.mojo:371`), `_expansion_trees._TreesEnsembleBase._class_major` / `_arange` / `_argmax` / `_acc_votes` (`bindings/trees_public_boost.mojo:214`, `trees_public_meta.mojo`), `_forest_protocol.ForestProtocol._prepare_resident_forest` (`bindings/trees_public_forest.mojo:1316`) and `extratrees._max_features_slots`. They were deleted as (f), not moved. `randomforest._max_features_fraction` has its native twin at `bindings/trees_public_forest.mojo:641` and is kept only because `packaging/portable_math/test_host_math.py` reads it (listed as tool-only). No Mojo was written by this lane.


## Left for other lanes

- (a) functions that loop without arithmetic (admission over parameters, or Python-object work inside UNPACK/PACK): 172 functions, 2,740 lines. They are boundary-migration work (AGENTS.md, Python objects only at the public boundary), not cross-module product work; the TSV marks them `shell/admission (loops=N)`.
- Native dispatcher arms whose only Python caller was deleted here (remove in the sequence lane): `seq_handles`, `seq_lr`, `seq_slots`, `seq_state_binding`, `seqdev_on`, `seqdev_resident`, `seqln_device_binding`, `seqmoe_weights` (the `_SEQ_NATIVE` actions of `bindings/xseq_*_native.mojo`).
- The (c) verification modules stay in the wheel while `python -m mojolearn verify` ships; moving them out needs the verify command to move with them (an orchestrator decision).
- Method: `~/mojolearn-evidence/python-package-audit/` holds the scanner (`scan.py`), import graph (`graph.py`), classifier (`classify.py`), transitive liveness (`liveness.py`), deleter and generator; rerun them in that order to refresh this file.
