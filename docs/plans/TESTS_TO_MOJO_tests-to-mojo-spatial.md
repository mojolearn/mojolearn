# Tests to Mojo, lane tests-to-mojo-spatial (2026-11-16)

The 32 files of the `L-spatial` rows of `TESTS_TO_MOJO_2026-11-15.md`, read function by function. Nothing was
compiled or run (lane rule). Verdicts as in `TESTS_TO_MOJO_tests-to-mojo-classical.md`: KEEP-BOUNDARY (only Python
can test it), KEEP-ORACLE (outside library: torch, arch, statsforecast), KEEP-NOPORT (native numerics that cannot
move yet; the blocker is named), PORT (moved to a Mojo check, Python deleted), SOURCE-TEXT (none found in these
files).

Totals: 32 files; 1 deleted (35 lines), 31 kept unchanged. One new check, `xcnn_sage_max_adjoint`, registered in
`checks/native_tests/registry.mojo` under a `# lane tests-to-mojo-spatial` comment
(`python3 tools/ops/native_checks_report.py plan --column cpu` lists it: 166 cpu checks).

| file | lines | result | removed | function verdicts and blockers |
|---|---:|---|---:|---|
| `test_arima_exog.py` | 322 | kept | 0 | KEEP-BOUNDARY 3: save/load archive formats (`mojolearn-arima-1/2`), named refusals, a native output store (`_typed_input.output_store`); inputs from `default_rng` |
| `test_arima_parameter_lifetime.py` | 75 | kept | 0 | KEEP-NOPORT 1: child process, public parameter npz and owner lifetime; inputs `default_rng(123)` |
| `test_centroid_fit_native_boundary.py` | 92 | kept | 0 | KEEP-BOUNDARY 5: refusal order, field identity after refit, pickle and teardown of public arrays |
| `test_cnn_max_gradient_ownership.py` | 35 | moved to `xcnn_sage_max_adjoint`; deleted | 35 | PORT 4 (2 node counts x ties): literal arange inputs, exact equality kept |
| `test_cpu_training_embedding_ivf.py` | 389 | kept | 0 | KEEP-NOPORT 3: CPU-only install smokes over whole public estimators (classical precedent: they test the install; kernels are in `ann_ivf_surface`, `embedding_surface`, `byte_lm_*`) |
| `test_holtwinters_inference.py` | 153 | kept | 0 | KEEP-BOUNDARY 1: its numerics already live in `decomp_holtwinters_inference` (diffed: NaN warm-up, level update, in-sample prediction); what remains is save/load bytes, loaded answers, the straddling public predict, the `ValueError` and the host-model route |
| `test_host_model_kmeans.py` | 344 | kept | 0 | KEEP-BOUNDARY 16: archive format, refusals, host-model route, one-ulp file perturbation; inputs `default_rng` |
| `test_host_radius_bindings.py` | 43 | kept | 0 | KEEP-BOUNDARY 1: the saved model's host binding (aux route `_mojolearn_x_neighbors`) answers the public model's bytes; a route property, not a kernel value |
| `test_ivf_embedding_cpu_inference.py` | 288 | kept | 0 | KEEP-BOUNDARY 10: save/load, cast/shape/metric refusals, clone aliasing, shipped-binding agreement; inputs `default_rng` |
| `test_kmeans_native_boundary.py` | 258 | kept | 0 | KEEP-BOUNDARY 11: provider envelopes (monkeypatch), refusals before Python hooks, serialization refusal, scalar admission |
| `test_mixture_reload_state.py` | 32 | kept | 0 | KEEP-BOUNDARY 1: public save/load equality; inputs `default_rng(7)` |
| `test_neighbors_ann_native.py` | 202 | kept | 0 | KEEP-NOPORT 7: compared with the former Python path (`_legacy_*` oracle) on `default_rng(7)` data, plus `audit_native_call` and refusal wording |
| `test_neighbors_graph_native.py` | 206 | kept | 0 | KEEP-NOPORT 8: `_legacy_neighbors_graph` oracle on random graphs (scipy CSR), `audit_native_call`, refusal parity |
| `test_optics_native_boundary.py` | 55 | kept | 0 | KEEP-BOUNDARY 4: scalar admission (int, float fraction, NumPy scalars), refusal order, hierarchy owner identity, pickle. The six-point labels/core distances are already `optics_native` unit 0 |
| `test_prophet_native_ownership.py` | 42 | kept | 0 | KEEP-NOPORT 1: child process over the `_identity_break` harness lane closure, heap-reuse ownership |
| `test_public_cnn_verifier_methods.py` | 53 | kept | 0 | KEEP-BOUNDARY 3: `fit` returns self and leaves weights, one-batch `fit` restores the `training` flag, predict gathers the class words (the BN running update itself is `cnn_batchnorm`) |
| `test_x_ann_repeat.py` | 97 | kept | 0 | KEEP-NOPORT 1: every ann entry twice in a child process per column (hang timeout), `default_rng` |
| `test_x_cluster_twice.py` | 99 | kept | 0 | KEEP-NOPORT 1: child process per column, `default_rng(7)` |
| `test_x_cnn_batchnorm.py` | 88 | kept | 0 | KEEP-ORACLE 2: the NumPy reference against torch (the layers are already in `cnn_batchnorm`) |
| `test_x_cnn_blocks.py` | 62 | kept | 0 | KEEP-ORACLE 1 (adaptive pool reference vs torch), KEEP-BOUNDARY 1 (public Dropout2d eval returns x) |
| `test_x_cnn_conv.py` | 172 | kept | 0 | KEEP-ORACLE 2: the conv references against torch (the layers are already in `cnn_conv`) |
| `test_x_cnn_gnn.py` | 252 | kept | 0 | KEEP-ORACLE 1 (max reference vs torch); KEEP-NOPORT 6: GCN/SAGE mean/sum/max-normalize/project against NumPy references on `default_rng` inputs, cached-graph re-forward through the public layer's `_gcache` |
| `test_x_cnn_repeat.py` | 157 | kept | 0 | KEEP-NOPORT 3: every public layer and the trainer repeated in one process with saved arrays, `default_rng` |
| `test_x_cnn_trainer.py` | 21 | kept | 0 | KEEP-NOPORT 1: public CNNClassifier learns (`default_rng(1)`; Linear numerics already in `cnn_trainer`) |
| `test_x_decomp_kernel_table.py` | 65 | kept | 0 | KEEP-BOUNDARY 4: capsule export, public NaN refusal, linalg reads an admitted NumPy operand without writing it, repeat bytes (the manifold numerics are already `mani_decomp_repeat`) |
| `test_x_linear_repeat.py` | 128 | kept | 0 | KEEP-NOPORT 1: 21 public estimators twice on host and GPU bindings |
| `test_x_metrics_repeat.py` | 94 | kept | 0 | KEEP-NOPORT 1: every metrics entry family twice via the public API, `default_rng(7)` |
| `test_x_neighbors_repeat.py` | 116 | kept | 0 | KEEP-NOPORT 1: child process per column |
| `test_x_sequence_garch.py` | 31 | kept | 0 | KEEP-ORACLE 1: against `arch` (recovery is already `sequence_garch`) |
| `test_x_sequence_prophet.py` | 32 | kept | 0 | KEEP-NOPORT 1: public fit on `default_rng(0)` noise; batch row equals single fit |
| `test_x_sequence_repeat.py` | 120 | kept | 0 | KEEP-NOPORT 1: child process per column, `default_rng(3)` |
| `test_x_sequence_theta.py` | 28 | kept | 0 | KEEP-ORACLE 1: against statsforecast (identities already in `sequence_theta`, `sequence_theta_variants`) |

## New check

| check | moved from | calls |
|---|---|---|
| `xcnn_sage_max_adjoint` (cpu column, 4 parallel units) | `test_cnn_max_gradient_ownership.py::test_max_backward_matches_independent_tie_scatter[37,53 x ties]` | `x_cnn/graph_layer_run.mojo` `GraphTableCalls` + `graph_views_run`, `x_cnn/graph_run.mojo` `graph_run_graph_op` (kind 0, the door's max forward) and `sage_backward_host_run` (kind 2, norm/root/project off), on the host image's x_cnn entry |

Assert by assert: inputs are the test's literals (x = arange * 0.125 - 1, x[1] = x[0] for ties, the 8 edges, upstream
(arange + 1) / 8, lin_l weight identity, no bias, no root); the expected maxima and the tie-split float32 adjoint are
rebuilt in the test's order (targets ascending, sources in edge order); both compared exactly. One difference: the
Python test asserted the layer output, which with lin_l the identity equals the aggregate; the check asserts the
aggregate the door feeds lin_l (lin_l's forward is covered by `cnn_trainer`), and the backward runs the whole host
tape, lin_l's backward included.

## What would let the KEEP-NOPORT groups move

- Seeded NumPy inputs (`default_rng`) in every remaining numeric test: they move once a committed generator stores
  the input words (or the relation is restated over a Mojo-generated input, as `cnn_cases.normals` does), and the
  Python `_legacy_*` oracle outputs are stored as fixture words (`test_neighbors_ann_native.py`,
  `test_neighbors_graph_native.py`).
- `*_repeat` / `*_twice` files: they test a whole public estimator twice in one child process per column (hangs,
  context lifetime). The kernels are in the family checks; the process-level property stays Python.
- `test_x_cnn_gnn.py` cached-graph tests: the cache key and `_gcache` live in the door on PythonObjects
  (`bindings/xcnn_graph_native_door.mojo` `_graph_key`, `_keys_equal`); no typed RUN owns the cache.
- `test_x_cnn_gnn.py` GCN/SAGE reference tests: portable over `GraphTableCalls` like the new check, but their inputs
  are `default_rng` draws; restating them needs a generator plus the float64 GCN/SAGE references in Mojo.

## Native entries missing

- No typed whole-layer SAGEConv/GCNConv forward RUN: the forward (edges, views, graph cache, op or spmm, lin_l, root
  add, normalization) is composed in `bindings/xcnn_graph_native_door.mojo` `xcnn_sage_forward` / `xcnn_gcn_forward`
  on PythonObjects. The new check composes the same typed pieces itself.

## Removed

`python/mojolearn/tests/test_cnn_max_gradient_ownership.py` (35 lines). Product code: nothing, by rule. Stale
references left to their owners: `tools/ops/wheel_verify_smoke_plan/tests.tsv` (the smoke's memory baseline row;
a verify tool, not this lane's) and the inventory docs, which are history.

## Product defects seen

None.
