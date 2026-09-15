# LANE STATUS: lane/cpu-training-par-classical (2026-09-15)

The CPU verification column for the two-device `par-*` lanes, wave 1: the
shared routing and the classical lanes. This serves the occasional full CPU
verification (`FULL_CPU_VERIFY`), not routine push testing, and adds no
public CPU training API.

## The route

`python/mojolearn/_parallel_pool.py` gains a CPU branch. On a CPU-only
install (`_backend._CPU_ONLY` set) a pool admits only `CPU_OPERATIONS`:
`scaler_fit`, `scaler_transform`, `arima_fit` and `holtwinters_fit`. Those are
the operations whose driver splits the work in Python (column ranges for the
scalers, series ranges for ARIMA and Holt-Winters), sends one request per
shard from a non-cooperative pool, and merges the shard results in shard
order. On CPU each device index is one worker process with no device
selection. Each shard runs the host binding's plain fit of that shard, and
the driver's split and merge code runs unchanged, so the CPU column checks
the same sharding logic the GPU columns check.

The shard fits run only inside `_cpu_reference.reference_training()`. The
pool wraps each request as `('cpu_reference', None, request)` only when its
caller is inside that scope, and `_parallel_worker.execute` re-enters the
scope for that request only. It is a pipe message from the parent, not an
environment variable. Outside the scope the worker's fit behaves exactly as
the plain CPU fit does.

Everything else refuses by name before any worker starts:

- a cooperative pool: "no CPU implementation of the cooperative multi-GPU
  driver <op> yet: its shards are device row tiles, chunks or ranges inside
  the GPU binding, which no host binding restates";
- any other non-cooperative operation: "no CPU implementation of the
  parallel worker operation <op> yet".

## Done (covered, `host_surface.covered_lanes()` 117 to 120)

| lane | family | shards in the lane | M4 CPU column, one core |
|---|---|---|---|
| par-scaler | preprocessing | 4 column shards (fit and transform) | IDENTICAL x4 |
| par-arima | arima | 2 series per shard | IDENTICAL x4 |
| par-holtwinters | tsa | 2 series per shard | IDENTICAL x4 |

The evidence was taken on the M4, one core, shared machine, on fixtures
base, odd and denormal, two repeats, at ee13e0d4b plus this change. Host
bindings were built from that tree with one compile job.

- **Four-column diff.** Diffed against the 166-lane record with
  `--require-columns 4`, filtered to the three fixtures: train IDENTICAL=18,
  infer/model IDENTICAL=18 with N/A=18, batch IDENTICAL=18, exit 0. The 18
  cells cover the three par lanes and their plain partners standard-scaler,
  arima and holtwinters.
- **Par against plain.** Each par lane holds itself to the plain CPU fit
  through `_same_bytes` inside the lane, and every cell read STABLE, not
  REFUSED.
- **The shards are real.** A probe counting the pool's requests saw four
  `(64, 4)` column shards for `fit_scaler`, four for `transform_scaler`, and
  two `(2, 64)` series shards for `fit_arima`. `fit_logistic` sent nothing
  and refused by name.
- **Host sabotage.** The sabotage set (`-D MOJOLEARN_HOST_SABOTAGE=1`, the
  four families rebuilt) read train DIVERGENT=9, infer DIVERGENT=9 and batch
  DIVERGENT=9, and the diff exited 1.
- **Shard-merge sabotage.** A scratch edit merged the shard results in
  reversed order (`parts[::-1]` in `parallel_preprocessing.py` and in the
  two series merges of `parallel_classical.py`), then was restored with
  `git checkout`. All 9 cells read REFUSED. Eight named the pair, for example
  "fit_arima ar and plain ar differ: 16 bytes of 16" and
  "fit_exponential_smoothing level_ and plain level_ differ: 6276 bytes of
  8000"; par-scaler/odd failed inside a worker on the mismatched widths.
- **No other lane moved.** All 39 par lanes ran on base: the three covered
  lanes STABLE, the other 36 REFUSED by name. `tools/cpu_identity_gate_check.py
  column --covered par-scaler,par-arima,par-holtwinters` reported 0 failures.
  The change touches only the pool and the worker, which only the par drivers
  use.

A two-device CPU run (`MOJOLEARN_PAR_DEVICES=0,1`, two worker processes) was
not taken locally, because of the one-core rule. It is the cheap next check
on a hosted runner.

## Refused, with reasons

### Cooperative drivers: the shards live in the device binding

These lanes are par-kmeans, par-gram, par-logistic, par-cd, par-svm, par-gp,
par-gmm, par-resample, par-cholesky, par-kernel-ridge and par-nystroem. Each
Python driver hands the whole fit to ONE worker. The partition is done
inside the GPU binding's multi-device code:

- `cluster/multi_gpu.mojo`, `glm/impl/qn/multi_gpu.mojo`, `solver/multi_gpu.mojo`
- `svm/impl/distance/kernel_matrices.mojo`, `cholesky/multi_gpu.mojo`,
  `mixture/multi_gpu.mojo`
- `resample/estimator.mojo`, `core/gram_multi_gpu.mojo`

That code reads `MOJOLEARN_<X>_DEVICE_COUNT` and moves row tiles, chunks or
replicate ranges between `DeviceContext(device_id=rank)` buffers. At a count
of 1 it takes the plain single-device path; `assignment_device_count()`
returns 1 when the variable is empty. No host binding restates the tile
split, and the host bindings deliberately omit the `*_parallel_available`
probes. Routing these lanes to the plain host fit would hash the plain fit
under a par label, which is the fake this lane must not do.

### par-rbf-sampler: no four-column record

`transform_rbf_sampler` shards in Python (rows per shard) and would take the
route unchanged. But the only committed columns carrying the lane are
`bench/results/identity_break/2026-09-15_par-lanes-new/{nvidia-2xh100-new8,amd-2xmi300x-new8}`.
They have no Apple column, and TRAINING_GPU_COLUMNS (the 166-lane record)
lacks the lane, so `--require-columns 4` would fail the gate. `rbf_sampler_rows`
stays out of `CPU_OPERATIONS` until the next release record carries the lane
on three columns.

## Proposed wave 2

1. **queries/knn/kde/radius.** `par-queries-{knn,radius,kde}` and
   `par-reference-knn{,-reg}` are non-cooperative and shard rows in Python
   (`ParallelQueries`, `ReferenceShardedNeighbors`). The core and estimators
   host bindings already serve the plain searches. Admit `neighbor_query`,
   `neighbor_reference` and `neighbor_vote`, and check that the lanes' GPU
   cells exist in the 166-lane record.
2. **forest.** `par-forest` and `par-forest-et` shard tree ranges in Python
   (`forest_fit`, `_fit_with_tree_start`). They need the rf and trees host
   bindings to serve a tree-start fit. `par-forest-pool`, `par-boosting*`,
   `par-ordered-rmse`, `par-feature-freq` and `par-iforest` are cooperative
   and refuse, as in the section above.
3. **dbscan/hdbscan/graph.** All are cooperative (the neighborhood,
   hierarchy and graph row drivers). Covering them needs a host restatement
   of the row tile split in each host oracle, keyed by a logical tile count.
   That is a Mojo change per family, and it checks the host restatement, not
   the device code.
4. **mlp/samba/byte-lm.** `par-mlp` and `par-samba*` send `mlp_gradient` and
   `samba_gradient` per logical shard from a non-cooperative pool, and fold
   the ordered gradient sum in Python. The training host binding serves the
   MLP step, so `par-mlp` is the nearest. `par-samba*` waits for the Samba
   host lane, and `par-byte-lm*` for a CPU route through `model_pool_training`
   and `offload_training`, which do not use `DevicePool`.
5. **The cooperative classical families** from the refused section, if the
   sharding claim on CPU is wanted there. That needs one host tile
   restatement per family, as in item 3.

## Found on the way

`StandardScaler.fit` and `ExponentialSmoothing.fit` do not refuse outside
`reference_training()` on a CPU-only install, while `ARIMA.fit` and
`LogisticRegression.fit` do. The first two classes do not reach
`_mode._guard_cpu_training`. This predates this lane (ee13e0d4b) and is
outside its scope, so it is reported, not changed.
