# Verify smoke triage 3, 2026-11-15 (lane smoke-triage-3)

The 18 non-crash test files that fail in the CPU-wheel verify smoke of `9685746f95` (GitHub runner, Linux, CPU-only,
Python 3.12.16; imports 76/76, self-test pass, lanes 510/510, 24 of 630 test files failing). Evidence:
`~/mojolearn-evidence/briefs-20261115/verify-smoke-9685/`; first errors per file:
`~/mojolearn-evidence/briefs-20261115/smoke-triage-3/first_errors.txt`. Not this lane's: test_parallel_causal_lm,
test_parallel_neural_native (transport timeout) and the four crashing files (crash lane). Nothing was compiled or run;
every fix is owed the next smoke, which decides completion. Branch `lane/smoke-triage-3` (base
`preinteg/wave2-20261113` at `0f86dbea48`).

Classes (per file): product defect 5; product defect (probable) 1; stale test 11; stale test (open) 1.

| file | class | cause file:line | introducing commit | fix (files) |
|---|---|---|---|---|
| test_agglo_native_boundary.py | stale test | the replaced-factory refusal reads "native provider requires compiled entry <name>" (providers/python/neural_descriptor_pyapi_door.mojo:293); the test matched "compiled extension entry" | abcb713f2b | test regex |
| test_byte_lm_native_state_restore.py | stale test | per-layer statuses and their `_counts` come together (bindings/byte_lm_trainer_native_door.mojo:1797); the CPU host session reports none, the test read the counts unconditionally | 98075d970b | test |
| test_c3_c5_no_copies.py | stale test | the test module is `mojolearn.tests.*`, so estimator_outside (providers/python/estimator_lifecycle_pyapi_unpack.mojo:16) counts its calls internal and IsolationForest returns its internal float32 scores; the widening (bindings/iforest_public_lifecycle_pack.mojo:190) is correct | c2c551b0b7 (widening) | `__mojolearn_user_boundary__ = True` in the test |
| test_cache_get_native.py, test_cache_stats_native.py | stale test | child cases imported bare `_arena_io`; `_ROWS_CALL` is connected when model_selection loads (python/mojolearn/_native_arena_rows.py:147) | b6aa87e41d | children import model_selection |
| test_cpu_training_par_classical.py, test_samba_gradients_native.py | product defect | training/native_samba_execution.mojo:287 gradients: `backward(Int(dh.unsafe_ptr()))` then `dh = next^`; the reassignment is not a use, so dh was freed before the layer read it ("grad_output must be finite") | 9411991aa3 | `_ = len(dh)` after the call |
| test_samba_checkpoint_native.py | product defect | same line: the layer's own dx reused the freed block, so it read a zeroed dy (total_norm 1.19 vs 3.10); checkpoint bytes, digest, state and loss already matched | 9411991aa3 | same |
| test_cross_vendor_coordinator_native.py | stale test | 10 == 9 after eight refusals is one retained reference, not one per refusal | (test) | count after one warm-up refusal |
| test_full_reset_numpy_free_import.py | stale test | `_CONTEXT['ready']` is set by the full-reset connect at the first family load (python/mojolearn/_families.py:266), not at `import mojolearn` | b6aa87e41d | drop the import-time assert |
| test_host_inference_native.py | stale test (open) | the hook record covered the whole controller (class build, patch install), not only the product call | (test) | record cleared before audit_native_call; if the next smoke still shows ['metaclass'], the product reads the category before its export refusal |
| test_linear_svr_native_fit_boundary.py | stale test | the analytic values (0.72, 27/35, 9/140, 243/1400) are the l2 optimum; LinearSVR's default penalty is l1 (python/mojolearn/svm.py:325), whose optimum 0.65 and 11/15 the product returns | 81ca435715 | test sets penalty='l2' |
| test_native_forest_auxiliary.py | product defect | bindings/host_tree_native_door.mojo:60 host_tree_cast refused rank with a bare "X must be 2-D" | fed1b4e384 (file) | the one wording of bindings/admitted_family_door.mojo:167 with the reshape hint |
| test_native_idna.py | product defect | bindings/native_idna_data.mojo held CPython's older B.3 table (650 rows); CPython 3.12.16's stringprep pins B.3 to Unicode 3.2 with 684 identity rows (U+023A, Georgian, Cherokee, ...), so the door's str.lower fallback lowered code points Unicode 3.2 does not | dd14a3b568 | table regenerated (1,333 rows) |
| test_neighbors_graph_native.py, test_neighbors_loc_native.py | stale test | legacy oracles called the model helpers `_take_rows`/`_take_cols` deleted with no product caller; the LOF metric_params refusal carries the `LocalOutlierFactor:` prefix like its siblings (neighbors/neighbors_local_run.mojo:54) | ec78e70324, d6c22d3a57 | oracle copies of the helpers; test wording |
| test_ocsvm_native_transport_boundary.py | stale test | same refusal wording as agglo; the resident factory is read only on the device route (bindings/svc_ocsvm_native_door.mojo:187), so on the CPU column the fit completes without calling the hook | abcb713f2b | test |
| test_trees_repeat.py | product defect (probable) | the two digests agree; the child aborted at exit (-6, "PyImport_GetModuleDict: interpreter has no modules dictionary" while garbage-collecting). Probable route: every stamped model's records death callback (bindings/py_model_records.mojo `_records_finalize_c`) drops pins whose owners read loaded modules; the resident capsule destructor calls a binding's free entry | not isolated | both return before any release once the interpreter is finalizing (destructor paths only) |

Also in this lane: `runtime_report()` gains `code_kind` (ptx|native) and `bindings[*]["expected_path"]`, read by
tools/identity_break.py (the PTX column read `code_mode`, the numeric mode; the forest guard keyed
`_mojolearn_forest_host`, which never matched `mojolearn._host.*`); `feature_freq_fit` admits sources and weights on
typed values; `_parallel_pool.public_method_result` and `mojolearn._cpu_reference` deleted; the smoke plan
`tests.tsv` regenerated from this smoke.
