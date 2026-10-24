# Native multi-GPU drivers: one native call per driver (2026-10-24, target 0.8.57)

Goal (Andrew, 2026-10-24): the `parallel_*` drivers stop orchestrating in Python. Each public
driver becomes argument checks plus one `_NATIVE(...)` call; the device fan-out, the shard
boundaries, the per-device work and the join run in Mojo. The 33 driver entry points that
`tools/public_runtime_inventory.py` reads as `python` are the scope
(`~/mojolearn-evidence/public-runtime-20261024-nonnative.md`, sections `## multi_gpu`,
`## ensemble` (the `parallel_*` rows) and `## manifold`).

## 1. What exists today

Two kinds of Python driver sit on `_parallel_pool.DevicePool`:

| kind | pool | who shards | drivers |
|---|---|---|---|
| A. cooperative | `DevicePool(devices, cooperative=True)`: ONE worker process, the visibility mask narrowed to the selected devices, every `MOJOLEARN_<FAMILY>_DEVICE_COUNT` set to `len(devices)` | the estimator's own GPU binding: `DeviceContext(device_id=rank)` per owner, `peer_clone`/`transfer_bytes` copies, `host_parallelize` over owners, join on the root (e.g. `svm/impl/distance/kernel_matrices.mojo::_kernel_rows`, `cholesky/multi_gpu.mojo::chol_trailing_rows`) | most of `parallel_classical`, `parallel_graph`, the cooperative half of `parallel_ensemble` |
| B. Python-sharded | `DevicePool(devices)`: one worker process per device | Python: `rows_per_shard` / `trees_per_shard` / `series_per_shard` slices, waves of `len(devices)` (shard `i` on rank `i % len(devices)`), the join by input order in Python (`parallel_preprocessing.py:143`, `parallel_gaussian_process.py:98`, `parallel_classical.py` RBF sampler, forest, ARIMA/Holt-Winters, `ParallelQueries`, IVF) | `parallel_preprocessing`, `parallel_gaussian_process`, `parallel_neighbors`, `parallel_ivf`, `fit_forest`, `transform_rbf_sampler`, `fit_arima`, `fit_exponential_smoothing` |

Already native: `parallel_model_selection.cross_val_score` (`bindings/parallel_cv_native.mojo`),
`parallel_forecasting`, `parallel_training` (byte LM, neural trainer).

### How `cross_val_score` became one native call

`family_msel_parallel_score` (in the classical owner, `bindings/parallel_cv_native.mojo`) validates
every argument and every fold before anything starts, then owns a native process pool
(`bindings/native_process_pool.mojo`: posix spawn, authenticated socket transport, the native codec,
no pickle). Each child runs `_native_parallel_cv_worker.py`, which is two lines: import the package
and enter `family_msel_parallel_worker`, a compiled request loop. The parent sends whole-fold waves,
reads replies in fold order and joins in Mojo. The worker processes are kept, because a fold is an
arbitrary estimator's whole fit; the Python orchestration is gone.

That pattern generalizes to every driver (kind A and B alike) with no new toolchain capability. It is
the fallback. It is not the first choice for kind A, because kind A's binding already does its own
fan-out inside one process: the worker process exists only to narrow device visibility and set
environment variables.

## 2. Decision: one process, a thread-scoped device plan

The native driver runs in the caller's process. A **device plan** (`core/device_plan.mojo`) is the
ordered list of physical device indices one native call may use, bound to the calling OS thread
for the length of that call:

- `device_plan_enter(ids)` / `device_plan_exit()`: claim and release the calling thread's plan slot.
  The table is a process-lifetime `_Global` of fixed slots keyed by `pthread_self()`, claimed by
  compare-exchange, so two Python threads running two drivers at once (the bindings release the GIL
  around their kernels) never see each other's plan. A nested plan on one thread is refused by name.
- `device_plan_setting(name)`: the calling thread's plan size as text while a plan is bound, else
  `getenv(name)`. It replaces the `getenv("MOJOLEARN_<FAMILY>_DEVICE_COUNT")` reads, so the count
  every owner loop already parses comes from the plan. With no plan bound nothing changes: the env
  variable path (the check tools and the remaining Python pools) reads exactly as before.
- `device_plan_device(rank)`: `ids[rank]` while a plan is bound, else `rank`. It replaces
  `DeviceContext(device_id=rank)` in the owner loops.
- `device_plan_root()`: `ids[0]` or -1. Every binding's root context (`core/neural_context.mojo::process_ctx`, which `neural_ctx` and the
  family contexts call) follows it: device 0 keeps the binding's process slot, any other root device gets
  a process-lifetime context of its own in the same slot, so no context is created and destroyed per
  call on the root.

A plan also carries a **family mask** (`DEVICE_PLAN_*` bits): the old cooperative worker set every
`DEVICE_COUNT_VARIABLES` count, and `MOJOLEARN_CHOLESKY_DEVICE_COUNT` only for the kernel-method and
Cholesky operations, so `device_plan_setting(name)` answers the plan size only for the families the
driver's worker enabled. The driver binds the plan in every binding its plain owner may reach (correct
whether a `_Global` table is per DSO or shared by name across the process: entering the same plan
again on one thread adds a level) (`_backend.plan_bindings(names)`), each binding registering `device_plan_enter` /
`device_plan_exit` / `device_plan_log` (`bindings/device_plan_binding.mojo`).

**Kind A (cooperative) doors.** `family_par_run` (`bindings/family_native.mojo`) takes (driver,
devices, owner operation, owner args, owner entry, owner context, plan bindings). The driver's
Python shell makes the type and mode refusals and resolves the plain method's entry, context and
bindings (binding acquisition, no Python frame in the door); the door admits the devices as the old
pool did (`bindings/parallel_plan_door.mojo` `plan_devices`, the CPU route refused by name), checks
the owner operation against the driver's spec (`bindings/parallel_plan_specs.mojo`: allowed owners,
families, the old rebuild refusal), binds the plan in every binding, runs the plain owner (a family
operation in place, any other through its own compiled entry) and releases the plans in a `finally`.
The kernel-method and Cholesky drivers have their own doors (`family_km_par_*`, `family_chol_par_*`).

**Kind B (Python-sharded) doors.** Running shards on host threads in one process was rejected: the
bindings' process caches (resident registries, context-keyed workspace caches, `process_ctx` slots)
are not written for concurrent plain owners. Kind B keeps one worker process per device, as the old
pool did, on the native local pool (`bindings/native_process_pool.mojo`, the `cross_val_score`
pattern): `parallel_shard_call` (`bindings/parallel_shard_native.mojo`) and the forecasting owner
(`bindings/parallel_forecast_native.mojo`) compute the old shard boundaries, send each shard's state
record and data through the native codec in waves of `len(devices)` (shard i on worker i % D, the
old `DevicePool.map` order), run the plain native owner in the worker (a model rebuilt from its
fields without a constructor) and place every block at its offset. Every pool's workers first answer
an identity request and are admitted by `native_require_worker_inventory` (distinct processes and,
on CUDA and HIP, distinct physical GPUs, one per worker); the pool and its served requests are
recorded for the verifier (`shard_open`, `shard_map`). `ReferenceShardedNeighbors`,
`ParallelQueries` and `DistributedIVFIndex` keep one such pool for their lifetime; their per-shard
work is the plain neighbor and IVF query methods (`python/mojolearn/_parallel_shard_ops.py`), which
are not one native call each yet: that is the neighbors family's own migration (inventory section
`neighbors`), not this one.

### What the worker processes were for, and what replaces each

| need | worker process | native driver |
|---|---|---|
| device selection | `CUDA_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` narrowed per worker; rank r saw device `devices[r]` as index r | the plan maps rank r to `devices[r]` explicitly; indices are the caller's own enumeration, which already honors the caller's mask, so `devices=(1, 3)` names the same physical devices as before |
| device counts | `MOJOLEARN_<FAMILY>_DEVICE_COUNT` set in the worker environment | `device_plan_setting` returns the plan size on the calling thread only; the process environment is never mutated |
| separate CUDA/HIP/Metal contexts | a fresh process, so fresh contexts | owners already create `DeviceContext(device_id=...)` per call inside one process today (the cooperative worker runs them); the root uses one process-lifetime context per device |
| process-global state | env variables, binding process slots, resident caches (`krr_resident`), GEMM phase counters | plan is per thread; resident device caches are bypassed by the parallel doors (a resident copy minted on a plan root would outlive the call on that device); `STEP_PHASE_TIMERS` builds still refuse more than one owner |
| memory isolation | everything freed at worker exit | shard structs free their buffers before their context at scope exit (`SVMKernelShard.__deinit__`, `CholTrailingShard`); the root's buffers are the plain method's |
| fault isolation | a crashed worker raised `RuntimeError: GPU worker failed` | an owner failure raises the plain method's error in the caller; a device fault that kills a process now kills the caller, which is the plain method's behavior on one device |
| CPU quota | `worker_cpu_threads` split `MOJOLEARN_CPU_THREADS` across workers | one process, one budget: the owner threads are `host_parallelize` over at most `len(devices)` owners |
| NumPy parity (`MOJOLEARN_POOL_IMPORT_NUMPY`) and raw-vs-public results (`__mojolearn_internal__`) | the worker imported NumPy when the parent had it; the parent re-applied the public conversion | gone: the owner runs in the caller with the caller's connected types; the driver applies `_parallel_pool.public_method_result` (the conversion every plain method's wrapper applies) |
| Metal | one child, `devices=(0,)` | `devices=(0,)` only, in process |
| CPU-only installs | logical per-device CPU workers for `CPU_OPERATIONS`; cooperative drivers refused | kind B: one CPU worker process per device index on the native pool, the host bindings' plain owners; kind A: refused by name, as before |

### Capability check (Mojo/MAX)

Every primitive the design needs is shipped and in use: several `DeviceContext(device_id=...)` in one
process (`svm/impl/distance/kernel_matrices.mojo`, `cholesky/multi_gpu.mojo`, `training/byte_lm_*`
with an explicit device list), device-to-device copies (`peer_clone`, `core/multi_gpu.mojo::
transfer_bytes`, host-staged on AMD by design), host threads (`core/host_parallel.mojo`), host
atomics (`std.atomic.Atomic`, `core/device_mutex.mojo`). No Mojo or Modular gap was found.

## 3. Bits

- **Kind A keeps every bit.** The owner loops, the partition (`m * rank // active` rows,
  `active = min(count, m)`), the per-cell contraction and the join order are the same code; only where
  the count and the device index come from changes. The root device does not enter any value. The
  par-* lanes hash model arrays and outputs (for example `par-kernel-ridge`: `dual`, `info`,
  `predict`; `par-nystroem`: `components`, `indices`, `normalization`, `eigenvalues`, `eigenvectors`,
  `sweeps`, `transform`), never process or shard traces, so no lane code or record changes.
- **Kind B keeps every bit**: the Mojo shard plans transcribe the Python ones (the same `*_per_shard`
  boundaries, wave assignment `rank = index % len(devices)`, join by input order), and each shard runs
  the plain native owner the old worker ran. The Python driver negative control
  (`MOJOLEARN_PAR_DRIVER_SABOTAGE` with `MOJOLEARN_HOST_ALLOW_SABOTAGE`, `driver_read_shift`) is native
  with the same rank rule (`bindings/parallel_shard_common.mojo` `shard_read_shift`: shift only above
  rank 0, merge at the unshifted offset). Two deliberate differences: the Holt-Winters and ARIMA
  merges store the fit through the plain native setters, so the merged model is a plain fit's (the
  lanes' arrays are unchanged); `fit_kmeans` refuses a non-default `metric` or `oversampling_factor`
  where the old worker silently fitted the defaults.
- No par-* lane is declared moving. If a later migration finds a kind B driver whose shard plan
  cannot be transcribed, its lane moves on every vendor in the same commit, recorded with the change.

Negative controls: kind A drivers keep their compile-time arms (`-D MOJOLEARN_SVM_PARALLEL_SABOTAGE=1`
for the kernel-row seam: `par-svm`, `par-svm-svr`, `par-kernel-ridge`, `par-nystroem`;
`-D MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE=1` for factor rows and solve columns). Both read the rank,
not the device id, so they stay inert at one device.

## 4. Migration record (all on `lane/native-parallel-driver-20261024`)

1. Kernel methods (pilot): `fit_kernel_method`, `apply_kernel_method`.
2. Cholesky, SVM and GP regressor: `fit_cholesky`, `solve_cholesky`, `fit_svm`, `predict_svm`,
   `fit_gaussian_process`, `predict_gaussian_process`.
3. The other cooperative drivers through `family_par_run`: `fit_logistic`, `fit_coordinate_descent`,
   `fit_gram_estimator`, `fit_kmeans`, `fit_dbscan`, `fit_gaussian_mixture`,
   `predict_gaussian_mixture`, `fit_hdbscan`, `bootstrap`, `permutation_test`,
   `monte_carlo_integrate`, `parallel_graph.fit_graph` / `transform_umap`,
   `parallel_ensemble.fit_boosting` / `fit_feature_freq` / `fit_ordered_rmse` /
   `fit_isolation_forest` / `score_isolation_forest`, `ParallelForestPredictor` (an in-process snapshot
   whose first predict builds its parallel_groves pool on the plan's devices).
4. Python-sharded drivers: `fit_arima`, `fit_exponential_smoothing`, `transform_rbf_sampler`,
   `fit_scaler`, `transform_scaler`, the GPC fit and predict, `fit_forest`, `ReferenceShardedNeighbors`,
   `ParallelQueries`, `DistributedIVFIndex`.
5. `DevicePool` and `_parallel_worker.py` removed (`_parallel_pool.py` keeps `public_method_result`);
   the verifier's two-device witness (`_verify_par.PoolWitness`) reads the plan logs and the pool trace;
   `_verify_distributed` reads the trace and drives the native transport negative control
   (`MOJOLEARN_PAR_TRANSPORT_FAULT` with `MOJOLEARN_HOST_ALLOW_SABOTAGE`: the first numerical batch of
   more than one result dropped or reversed, once).

## 5. Bindings to build

Every GPU binding that compiles an owner loop changed (counts and device ids from the plan) and
registers the plan entries: `_mojolearn` (base), `_mojolearn_estimators`, `_mojolearn_gbdt`,
`_mojolearn_hdbscan`, `_mojolearn_ivf`, `_mojolearn_linalg`, `_mojolearn_metrics`,
`_mojolearn_mixture`, `_mojolearn_resample`, `_mojolearn_rf`, `_mojolearn_solver`, `_mojolearn_trees`,
`_mojolearn_x_ann`, `_mojolearn_x_cluster`, `_mojolearn_x_decomp`, `_mojolearn_x_linear`,
`_mojolearn_x_neighbors`, `_mojolearn_kernel_methods`, `_mojolearn_gp`, `_mojolearn_svm`.
`core/neural_context.mojo` changed (the root follows the plan): every binding that takes a process
context rebuilds (the rebuild fence lists 31). The tokenizer host binding (`_mojolearn_tokenizer_host`,
the classical owner: `family_native`, `kernel_methods_native`, `linalg_native`, `gp_classifier_native`,
`parallel_forecast_native`, the new `parallel_shard_native` and its `parallel_shard_call` entry) and
`_mojolearn_runtime_classical_host`. Host twins compile the changed shared modules too. No bit moves
with no plan bound.

## 6. Runs owed

Nothing here has been compiled or run. On the integrated, compiled branch:

- **One-device identity, every vendor (Apple, nv, amd)**: the par-* lanes' one-device columns and the
  plain lanes must hash as recorded (no plan bound: nothing moves). The table admit covers it.
- **Two-device columns, nv and amd** (`python -m mojolearn verify --par --par-devices 0,1`, then the
  default scope on a two-GPU box): every par-* lane below must hash equal to its one-device column, and
  the witness must read a placed native plan (cooperative) or an admitted native pool (Python-sharded)
  on both devices.
  - cooperative (device plan): par-kernel-ridge, par-nystroem, par-cholesky, par-svm, par-svm-svr,
    par-gp, par-logistic, par-cd, par-cd-elasticnet, par-gram, par-gram-ols, par-gram-pca,
    par-gram-tsvd, par-kmeans, par-dbscan, par-gmm, par-hdbscan, par-resample,
    par-graph-agglomerative, par-graph-spectral, par-graph-umap, par-boosting, par-boosting-clf,
    par-boosting-reg, par-boosting-pointwise, par-border-types, par-feature-freq, par-ordered,
    par-ordered-rmse, par-iforest, par-forest-pool.
  - Python-sharded (native pool): par-arima, par-holtwinters, par-forecast-arima,
    par-forecast-holtwinters, par-rbf-sampler, par-scaler, par-scaler-minmax, par-gpc-fit,
    par-gpc-predict, par-forest, par-forest-reg, par-forest-et, par-forest-et-clf, par-ivf,
    par-queries-knn, par-queries-nn, par-queries-radius, par-queries-kde, par-reference-knn,
    par-reference-knn-reg.
  - unchanged native drivers, as a control: par-byte-lm, par-byte-lm-model-pool, par-byte-lm-offload,
    par-causal-lm, par-cross-val, par-mlp, par-samba, par-samba-clip.
- **Negative controls, nv and amd, one- and two-device columns** (each must move only at two devices):
  the compile-time arms `-D MOJOLEARN_<X>_PARALLEL_SABOTAGE=1` for GBDT, GRAM, QR, SVM, SOLVER,
  NEIGHBORS, KMEANS, GLM, DBSCAN, GP, ISOLATION_FOREST, FOREST_POOL, CHOLESKY, GMM, HIERARCHY and
  RESAMPLE against their lanes (docs/multi_gpu/PAR_SABOTAGE_ARMS.md); the driver arm
  `MOJOLEARN_PAR_DRIVER_SABOTAGE=1 MOJOLEARN_HOST_ALLOW_SABOTAGE=1` against the Python-sharded lanes;
  `python -m mojolearn._verify_distributed --devices 0,1` (its transport controls now drive the native
  `MOJOLEARN_PAR_TRANSPORT_FAULT` switch).
- **Checks**: `tools/parallel_kernel_methods_check.py`, `parallel_cholesky_check.py`,
  `parallel_graph_check.py`, `parallel_neighbors_check.py`, `parallel_forest_pool_check.py` with
  `--devices 0,1`, including two driver calls in one process (per-call owner contexts are now created
  in a long-lived process), and the CPU-only route tests (`test_cpu_training_par_classical`,
  `test_cpu_parallel_rbf_sampler`, `test_parallel_gaussian_process`, `test_distributed_ivf`,
  `test_numpy_free_features`) on a CPU-only install.
