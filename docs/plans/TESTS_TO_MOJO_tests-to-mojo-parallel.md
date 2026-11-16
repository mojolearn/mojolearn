# Tests to Mojo, lane tests-to-mojo-parallel (2026-11-16)

The `L-parallel` rows of `TESTS_TO_MOJO_2026-11-15.md` (32 files, 4,694 lines), read function by function. Nothing
was compiled or run (lane rule). `test_meta_members_native.py` is in the row set but belongs to the crash lane
(segfaulting file): not read for changes, not touched.

Verdicts as in `TESTS_TO_MOJO_tests-to-mojo-classical.md`: KEEP-BOUNDARY (only Python can test it: admission,
`pytest.raises`, returned objects and types, hook/callback refusal, `audit_native_call`, subprocess/child audits,
pickling, archives, sockets, outside oracles); KEEP-NOPORT (native numerics whose inputs come from
`np.random.default_rng`, or whose entry is reachable only through a `PythonObject` driver); COVERED (an existing check
asserts the same, assert by assert); SOURCE-TEXT; PORT.

Totals: 31 files read; 0 moved, 0 deleted, 0 shrunk; 0 Python test lines removed. No new Mojo check; nothing added to
`checks/native_tests/registry.mojo`. Four docstrings corrected to name the check files that exist
(`core_partitioned_program.mojo`, `resample_cv_row_order.mojo`, `training_small_mlp.mojo`,
`training_primitives.mojo`; they named `*_cpu/_gpu`/`*_cases` files that were merged away).

Why nothing moved: every native numeric in these files was already ported by earlier lanes (tmig-training,
tmig-training-2, tmig-sequence-1, tmig-msel-metrics-tsa-1), and their docstrings say so; what is left is the
boundary they deliberately kept, or whole-operation driver tests whose inputs are NumPy random draws or whose native
owner is a `PythonObject` driver with no typed entry a check could call. No SOURCE-TEXT function was found: the
`read_text()` hits of the inventory are child-process logs and JSON audit reports, not source.

## Per file

| file | lines | result | removed | function verdicts | what blocks a move |
|---|---:|---|---:|---|---|
| `test_autoarima_search_ownership.py` | 57 | kept | 0 | KEEP-NOPORT 1, KEEP-BOUNDARY 1 | subprocess lifetime tests; search inputs from `default_rng(518)`; the literal single-d case needs a typed AutoARIMA search entry (only the Python estimator drives it) |
| `test_cache_release_native.py` | 220 | kept | 0 | KEEP-BOUNDARY 13 | `DeviceCache` state is a Python dict (`_entries`), every assert is on it or on an audit; no typed native cache owner exists |
| `test_cache_stats_native.py` | 321 | kept | 0 | KEEP-BOUNDARY 12 | hook refusal, thread-local Python scopes, NumPy string prefixes: boundary only |
| `test_cache_upload_native.py` | 416 | kept | 0 | KEEP-BOUNDARY 12 | `id_of` key/owner publication on the Python dict, refusals, real upload aliasing through the binding |
| `test_classical_partition_native.py` | 55 | kept | 0 | KEEP-NOPORT 3 | driver = plain public method on fitted models; ARIMA/GPC inputs from `default_rng`; the Holt-Winters literal case needs a typed `parallel_forecast_call` entry (driver takes the Python model) |
| `test_cpu_parallel_rbf_sampler.py` | 70 | kept | 0 | KEEP-NOPORT 1, KEEP-BOUNDARY 1 | `default_rng` inputs; shard trace and fault arming are `PythonObject` driver state |
| `test_cross_vendor_coordinator_native.py` | 180 | kept | 0 | KEEP-BOUNDARY 3 | independent Python socket peers and wire format; the fold words themselves are `resample_cross_vendor_fold` fixtures 0-2 |
| `test_cross_vendor_native.py` | 55 | kept | 0 | KEEP-BOUNDARY 3 | `state_hash` field order vs `hashlib`, buffer-view admission, audits; `cross_vendor_state_hash` takes a Python dict (no typed entry) |
| `test_cross_vendor_worker_native.py` | 203 | kept | 0 | KEEP-BOUNDARY 4 | `BatchFeed`/`LearningRateTable` are `PythonObject` owners (`bindings/cross_vendor_native_door.mojo` `cross_vendor_feed_*`, `cross_vendor_rate*`); the source-mutation and audit asserts are boundary; the complete worker is a child process vs a protocol oracle |
| `test_distributed_program.py` | 157 | kept | 0 | KEEP-NOPORT 2, KEEP-BOUNDARY 1 | Adam/dropout resume and physical topology use `default_rng` inputs and weights; `training_distributed_program` units 0-12 already hold the analytic cases and say these stay |
| `test_distributed_program_boundaries.py` | 62 | kept | 0 | KEEP-BOUNDARY 1 | one-native-call audits only (values in `training_distributed_program` units 3-12) |
| `test_native_cv_gbdt_nb.py` | 208 | kept | 0 | KEEP-NOPORT 3, KEEP-BOUNDARY 4 | `default_rng(37)` data; CV result vs explicit public fits; hook refusals |
| `test_native_cv_lda.py` | 223 | kept | 0 | KEEP-NOPORT 2, KEEP-BOUNDARY 7 | `default_rng(43)` data; constructor metadata, callbacks, dataframe names, label output ancestry |
| `test_native_cv_outputs.py` | 55 | kept | 0 | KEEP-BOUNDARY 2 | string label placement and the canonical JSON/unicode descriptor hash: Python objects |
| `test_native_cv_row_order.py` | 79 | kept | 0 | KEEP-NOPORT 1 | group permutation on `default_rng(31/37)` data scored by public fits; the row-order oracle itself is already `resample_cv_row_order` |
| `test_native_stratified_connection.py` | 65 | kept | 0 | KEEP-BOUNDARY 2 | loader poisoning (monkeypatch) and audits; second test on `default_rng(43)` data |
| `test_parallel_arima_serialization.py` | 38 | kept | 0 | KEEP-NOPORT 1 | `default_rng(7)` data; asserts the saved archive file bytes (Python archive boundary) |
| `test_parallel_byte_borrowed_inputs.py` | 180 | kept | 0 | KEEP-NOPORT 2, KEEP-BOUNDARY 4 | byte-LM state from `default_rng(0)`; Fortran/strided/Array admission and refusal messages |
| `test_parallel_causal_lm.py` | 469 | kept | 0 | KEEP-NOPORT 3, KEEP-BOUNDARY 6 | checkpoints from the Python `_causal_lm_fixtures`; worker child audits; owners behind `_native_parallel_causal` (`PythonObject`) |
| `test_parallel_forecasting.py` | 58 | kept | 0 | KEEP-NOPORT 2, KEEP-BOUNDARY 2 | `default_rng(3/4)` data; partition refusals |
| `test_parallel_mlp_native.py` | 75 | kept | 0 | KEEP-BOUNDARY 3 | snapshot dict publication, rejected-snapshot atomicity, uint64 target refusal; numerics already in `training_small_mlp` fixtures 3-4 and `training_primitives` |
| `test_parallel_model_selection.py` | 234 | kept | 0 | KEEP-NOPORT 2, KEEP-BOUNDARY 11 | `default_rng(37)` data; worker/pool failures, callback refusal, Metal device rule, capture bytes |
| `test_parallel_neural_native.py` | 326 | kept | 0 | KEEP-NOPORT 2, KEEP-BOUNDARY 2 | multi-process training with child audits; the literal-input MLP case needs a typed `ParallelNeuralTrainer` entry (driver is `_native_parallel_neural`, `PythonObject`); the Samba case uses the Python `_reference` oracle of another file; `independent_gradient` is imported by `test_parallel_neural_admission.py` |
| `test_parallel_pool_numpy_parity.py` | 64 | kept | 0 | KEEP-BOUNDARY 2 | returned type/dtype/flags of the in-process driver vs plain fit; `default_rng(0)` data |
| `test_parallel_shard_payloads.py` | 123 | kept | 0 | KEEP-NOPORT 2, KEEP-BOUNDARY 2 | `default_rng` data; string/bool labels and call-token storage are Python-object payloads |
| `test_parallel_trainers_coverage.py` | 223 | kept | 0 | KEEP-NOPORT 4, KEEP-BOUNDARY 6 | byte-LM state from `default_rng(0)`, many-shard fold from `default_rng(11)`; refusals, context managers, checkpoints |
| `test_parallel_worker_raw_results.py` | 49 | kept | 0 | KEEP-BOUNDARY 1 | user vs package result type (`_mode._public_result`): Python boundary |
| `test_partitioned_program.py` | 127 | kept | 0 | KEEP-NOPORT 2, KEEP-BOUNDARY 1 | `default_rng(26/18)` inputs; physical multi-GPU profile; admission and damaged checkpoint; analytic cases are `core_partitioned_program` |
| `test_public_model_selection_verifier_gaps.py` | 128 | kept | 0 | KEEP-NOPORT 6 | `default_rng(407)` data; public search/curve objects vs independent public fits |
| `test_runtime_classical_partition.py` | 55 | kept | 0 | KEEP-BOUNDARY 3 | audits of public fit/predict/save/load, resident-table stats through the bridge, input-protocol refusal |
| `test_x_cnn_pool.py` | 119 | kept | 0 | ORACLE 2 | Python pooling reference vs torch (outside library); the layers vs that reference are already `cnn_pool` units 0-16 |
| `test_meta_members_native.py` | 255 | not touched | 0 | - | owned by the crash lane |

## Native entries missing (what would let the KEEP-NOPORT driver tests move)

Each driver below takes and returns Python objects in its door, so a Mojo check has nothing typed to call. Listed for
the Python-at-the-boundary work, not added here (no product changes in this lane):

- `bindings/cross_vendor_native_door.mojo`: `cross_vendor_feed_init/feed_batch`, `cross_vendor_rates_init/rate`,
  `cross_vendor_state_hash` (Python dict of buffers), the coordinator `cross_vendor_drive`.
- `_native_parallel_neural` (ParallelNeuralTrainer init/train_step/checkpoint/from_checkpoint) and
  `_native_parallel_causal`: no typed owner entry.
- `parallel_forecast_call` (predict/forecast ARIMA, Holt-Winters) and `fit_arima`: take the fitted Python model.
- `_native_parallel_shard` (`rbf_rows` shards, trace, fault arming).
- `_arena_io.DeviceCache`: state is a Python dict; no typed cache owner.
- AutoARIMA `search`: no typed entry below the estimator.

Inputs from `np.random.default_rng`: about 30 functions; they move only with committed fixture bytes from a generator
(plan rule 3), which was not done here because each also asserts boundary properties that keep the Python test.

## Product defects seen

None found by reading. Not run.

Removed by this lane: nothing, because every remaining function is boundary or blocked as listed above; no
source-text function and no assert-by-assert duplicate of a registered check was found.
