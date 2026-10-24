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
- `device_plan_root()`: `ids[0]` or -1. A binding's root context (`_family_ctx()` and its peers)
  follows it: device 0 keeps the binding's process slot, any other root device gets a process-lifetime
  context of its own (`core/device_plan_context.mojo::process_ctx_on`), so no context is created and
  destroyed per call on the root.

A kind A driver's door: validate, `device_plan_enter(devices)`, run the plain estimator owner (the
same `family_*` code the plain method runs), `device_plan_exit()` in a `finally`, return. A kind B
driver's door lives in the estimator's binding DSO (its typed run functions, no Python call inside a
thread): it computes the same shard boundaries in Mojo, runs `host_parallelize` over the devices,
each thread entering a one-device plan `[d]` and running the plain kernels on its shards in wave
order, and copies results into the output in input order on the host or the root device.

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
| CPU-only installs | logical per-device CPU workers for `CPU_OPERATIONS`; cooperative drivers refused | kind B: the host binding runs the same shards on host threads; kind A: refused by name, as now |

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
- **Kind B keeps every bit if the Mojo shard plan transcribes the Python one exactly**: the same
  `*_per_shard` boundaries, wave assignment `rank = index % len(devices)`, and join by input order.
  The Python driver negative control (`MOJOLEARN_PAR_DRIVER_SABOTAGE` with
  `MOJOLEARN_HOST_ALLOW_SABOTAGE`, `driver_read_shift`) moves into the native shard plan with the
  same rank rule (shift only above rank 0, merge at the unshifted offset).
- No par-* lane is declared moving. If a later migration finds a kind B driver whose shard plan
  cannot be transcribed, its lane moves on every vendor in the same commit, recorded with the change.

Negative controls: kind A drivers keep their compile-time arms (`-D MOJOLEARN_SVM_PARALLEL_SABOTAGE=1`
for the kernel-row seam: `par-svm`, `par-svm-svr`, `par-kernel-ridge`, `par-nystroem`;
`-D MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE=1` for factor rows and solve columns). Both read the rank,
not the device id, so they stay inert at one device.

## 4. Migration order

1. **Pilot: kernel methods** (`fit_kernel_method`, `apply_kernel_method`). Kind A, two owner loops
   (SVM kernel rows, Cholesky factor rows and solve columns), one binding. Done on
   `lane/native-parallel-driver-20261024`.
2. The other kind A drivers in `parallel_classical` that share those loops: `fit_cholesky`,
   `solve_cholesky`, `fit_svm`, `predict_svm`, `fit_gaussian_process`, `predict_gaussian_process`.
3. The remaining kind A drivers, one owner family per lane (each replaces its family's
   `getenv("MOJOLEARN_<FAMILY>_DEVICE_COUNT")` and `device_id=rank` sites): logistic and GLM, Gram
   estimators, KMeans, coordinate descent, DBSCAN, HDBSCAN, Gaussian mixture, resampling,
   `parallel_graph`, cooperative `parallel_ensemble` (boosting, isolation forest).
4. Kind B drivers, with their shard plans in Mojo: `transform_rbf_sampler`, `parallel_preprocessing`
   scalers, `parallel_gaussian_process`, `fit_forest`, `fit_arima`, `fit_exponential_smoothing`,
   `ParallelQueries`, `parallel_ivf`, `parallel_neighbors_reference`.
5. Remove `_parallel_pool.DevicePool`, `_parallel_worker.py` and their tests once no driver uses them.

Each migrated family's bindings must be rebuilt: the shared owner loops (`svm/impl/distance/
kernel_matrices.mojo`, `cholesky/multi_gpu.mojo`, `cholesky/checks/trsm.mojo`) are compiled into every
binding that imports them, with no change in behavior while no plan is bound.

## 5. Pilot record

- `core/device_plan.mojo`: the thread-scoped plan.
- `core/device_plan_context.mojo::process_ctx_on`: one process-lifetime context per root device.
- `svm/impl/distance/kernel_matrices.mojo`, `cholesky/multi_gpu.mojo`, `cholesky/checks/trsm.mojo`,
  `kernel_methods/checks/kernel_matrix.mojo`: counts from `device_plan_setting`, device ids from
  `device_plan_device`.
- `kernel_methods/estimator.mojo::_family_ctx`: the root follows `device_plan_root`.
- `bindings/_mojolearn_kernel_methods.mojo`: `kernel_methods_device_plan_enter` / `_exit` entries.
- `bindings/kernel_methods_native.mojo`: `family_km_par_fit` / `family_km_par_apply` doors (admission,
  device validation, plan scope, the plain owners with the resident predict bypassed).
- `python/mojolearn/parallel_classical.py`: the two drivers are one `_NATIVE` call each.
- Identity lanes `par-kernel-ridge` / `par-nystroem`: parts unchanged, bits unchanged.
- RUN OWED: an nv and amd two-device column (`MOJOLEARN_PAR_DEVICES=0,1`) of `par-kernel-ridge` and
  `par-nystroem` against the one-device column, with the default build and the
  `MOJOLEARN_SVM_PARALLEL_SABOTAGE` / `MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE` arms (they must move only at
  two devices), plus `tools/parallel_kernel_methods_check.py --devices 0,1`, including two driver calls
  in one process (the per-call owner contexts are created and destroyed in a long-lived process now).
