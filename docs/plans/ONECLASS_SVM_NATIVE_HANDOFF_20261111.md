# OneClassSVM typed state checkpoint

Source only: compilation, product execution, tests, timing and identity were not
run by this lane. Root collects this with the Mojo 1.2 / MAX 26.7 migration into
the one combined pre-integration branch. No new family work follows this freeze.

Producers: `2e9df61c52` before-role inventory, `f4e90945ee` additive plans/API,
`6a96587b8ebb33dd243ce5b543d85a8d80c4025e` public source switch, and this fixture
commit. The preceding SVR chain and its canonical shared dependencies are listed
in `SVR_NATIVE_HANDOFF_20261111.md`. Dependency adoptions are not new producers.

Seven inventoried family roles now use native storage: numerical request lists,
positive-weight compaction, private Gram matrices, alpha/result scratch, resident
solver requests, private query scores and private sign-label codes. Public fit,
score, decision, predict and fit_predict consume these states. Public support,
support-vector, dual, intercept and requested output Arrays retain their original
publication stages. Prefix and row/column copies remain in the same routes.

The existing `XnDriver` transports the same operations. Its v1 schema is
unchanged. The additive `mojolearn.ocsvm.resident.v1` has mode, compiled-flag and
resident-solve operations. It preserves both existing resident exports and calls
the unchanged `op_ocsvm_x`; the candidate route keeps its original x_neighbors
descriptor. Existing `OCSVM_CLS2_RES`, `OCSVM_IDN_RES`, fused labels and old-items
choices remain. No numerical default, tuned implementation or experiment was
removed. New core files use Mojo 1.2 `Array`; root owns the remaining mechanical
SDK updates on collection, with no compatibility alias or workaround.

Removed private mechanisms: `_op`, `_dims`, `_take_rows`, `_take_cols`, `_kernel`,
`_i32_of`, `_ocsvm_decision_of`, copied operation-argument lists, Python request
lists, private Array scratch and the `tolist`/int-map/array.array index path.
Gamma names now use admitted native C comparisons; the operator module lookup
and callable transport are gone. Native class identities are cached at the
boundary, and scalar object-array owner chains are admitted under the GIL before
comparison. Array diagnostics use a fixed container description instead of
calling NumPy's Python formatter. Namespace checks precede mode-table lookup.

Retained and owed: the existing OneClass coefficient weak-cache tuple/layout
record in `gp_svm_records.mojo`, shared lifecycle/public storage admission,
serialization and mandatory boundary pins. That tuple was outside these seven
inventoried roles and is not claimed eliminated. Shared `XnDriver` schema reuse
also means seven family roles must not automatically become seven new aggregate
roles; the independent inventory audit decides the aggregate addition. No whole
public family is claimed Python-free.

Native suites:

- `checks.native_tests.ocsvm_native_cpu_suite.run_checks()`: 12 independent cases
  through `host_parallelize`.
- `checks.native_tests.ocsvm_native_gpu_suite.run_checks()`: 12 ordered base/Gram
  cases and 10 ordered resident cases, 22 total at the existing default. With
  `MOJOLEARN_IDN_OCSVM_RES_OFF` or `IDN_ALL_OFF`, it reports 12 run and an explicit
  skip of the 10 unavailable resident cases. GPU concurrency stays parked under
  the recorded upstream shared-context defect; kernels are not serialized anew.

The cases cover all five kernels, positive/zero weights, alpha-sum and KKT
conditions, public-equivalent support mapping, Gram/fused scores, sign labels,
resident solving and all-zero refusal. References are closed-form two-point
duals/rho and independent Float64 kernel equations, with parallel reference rows.
The original 150x6 training and 40x6 held-out inputs from
`test_x_neighbors_sanity._data` are frozen as native constants. Input preparation
only was executed: system Python lacked NumPy; the existing Pixi Python prepared
the exact bytes. Both logs are retained. No estimator, oracle or test executed.

Fourteen Python boundary cases (nine definitions) cover scalar conversions,
public storage/pickle lifetime, original weighted row indices, coefficient
replacement, native support-index conversion, factory refusals, callback traps,
formatter refusal and early/late publication stages. All existing Python
numerical coverage remains unchanged and retained.

Affected roots: `_mojolearn_x_neighbors`, `_mojolearn_x_neighbors_host`,
`_mojolearn_native_classical_family_host`, `_mojolearn_runtime_family_host`.
The additive exports are also recorded in `x_neighbors/gen.py`; it was not
regenerated. New scalar-type metadata uses only its own V1 shared slot; no
existing slot layout changed and no Python-exported owner type was added.

Evidence directory:
`~/mojolearn-evidence/python-mojo-migration-20261110/svm-native/ocsvm/`.
`switch-source-conservation.json` records 11 unchanged numerical/shared sources,
three retained Python coverage files, and zero Python imports in the four new
native files. This evidence predates root's mechanical namespace migration and
proves source conservation only. Full input provenance, original snapshots and
commit/push logs remain there. Independent source review is separate from
compilation and execution.

RUN OWED: combined SDK-1.2 product compilation, 12 CPU / 22 default GPU native
cases (or the stated switch-off coverage), and 14 retained boundary cases.
Other SVM work still owed after the combined freeze: SVC's eight inventoried
roles, SVR/LinearSVR score composition, shared weighted/precomputed temporary
storage, and the explicitly retained OneClass weak-cache record. No box job was
queued by this lane; no performance or identity result is claimed.
