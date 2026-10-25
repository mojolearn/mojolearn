# Native public parallel boundary — 2026-10-25

The remaining classical, graph and ensemble public wrappers performed Python
admission, binding lookup and output conversion before/after their native
operation. `ParallelForestPredictor` also copied and locked the model, checked
state and released resident handles in Python. Each now hands raw arguments to
one `family_par_public` entry. This is a boundary migration, not new arithmetic.

The import-time `_native_parallel_public` registry connects exact supported
classes, owner entry/context pairs, binding namespaces containing compiled
exports, argument recipes and class/storage identities. The native boundary
admits buffers and scalar configurations before reading protocols, rejects
custom classes/storage hooks, validates devices/modes and invokes the same owner
operation. Existing `family_par_run`, cooperative device-plan cleanup and shard
ABIs are preserved. Existing native forecasting aliases and `fit_forest` remain
on their existing entries.

The 27 migrated functions include the three resampling wrappers; argument recipes
preserve optional sample weights, validation data, supplied permutations,
posterior-standard-deviation requests and resampling configuration. Constructors,
queries, close and context-manager methods add six predictor operations. The
predictor takes its own shallow model snapshot with a separate resident holder,
uses a native `_thread.RLock`, and drops compiled resident owners on close.

`bindings/parallel_public_result.mojo` owns external-caller result conversion.
Internal callers and callers without NumPy keep native results. External callers
receive NumPy arrays; top-level regressor predictions and outlier scores preserve
the prior float64 widening. Tuple/list-of-array results keep the original nested
conversion semantics. The helper is shared with other parallel boundary lanes.

## Validation and integration owed

Source-only checks passed: four `tools/test_native_parallel_public.py` tests
cover all 33 one-call shells, raw argument/device positions, nondefault optional
argument recipes and compiled-only binding namespaces. Fourteen existing driver
and kernel-method wire checks passed without importing the package. The strict
inventory at this lane's base plus changes still fails with five other public
operations; this is not a zero-runtime-Python or release qualification claim.

No build or native runtime test was run in this lane. The changed native source
is reached by `classical_native` / the runtime-classical host binding. The
integrator must compile the consolidated freeze and run:

- `test_native_parallel_public_boundary.py`: protocol rejection before hooks,
  zero Python preparation frames on admission refusal, exact forest-class
  admission, snapshot isolation and repeatable close (seven runtime cases).
- Existing `test_parallel_public_drivers.py`,
  `test_parallel_gaussian_process.py`, `test_cpu_parallel_rbf_sampler.py`,
  `test_parallel_pool_numpy_parity.py` and `test_parallel_worker_raw_results.py`:
  native caller/output parity, installed buffers, optional NumPy, supported
  parameter combinations and preserved refusal behavior.
- Existing predictor/forest driver probes: repeated prediction, class
  probabilities, source refit independence, closed-pool refusal and handle
  release on both success and failure.
- Required affected identical lanes and two-physical-GPU NVIDIA/AMD negative
  controls on the frozen installed artifacts. Existing one-device/two-device
  numerical owners and sabotage mechanisms remain the same.

This source inventory and these local wire tests do not establish transitive
compliance of every existing owner. Native integration, supported input coverage,
whole-operation traces, platform identity and installed-wheel qualification stay
pending until their retained evidence passes.
