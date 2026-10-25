# Parallel neighbor and IVF native boundary

Source lane based on `integ/0857` at
`75b6b0acba0c0f0361186195f06998dabb50bd16`.

`ReferenceShardedNeighbors`, `ParallelQueries`, `DistributedIVFIndex.from_index`
and the scaler transform hand arguments directly to `parallel_shard_call`.
Constructors, close/context-manager operations, input admission, binding choice
and public output handling are native. The import-time bridge retains plain
module namespaces containing verified compiled functions, including optional
resident-index entries. It keeps the same merge helper route formerly selected
by `_buffer._native` during each reference query.

`bindings/parallel_neighbor_worker.mojo` replaces the Python operation registry:

- Reference shards use the existing native KNN fit/admission and search entries;
  reference votes call `knn_classify_neighbors` / `knn_regress_neighbors`.
- Query snapshots use native KNN classifier/regressor, RBC/brute nearest-neighbor,
  radius count/fill/sort and KDE owners. The worker owns resident handles and
  releases them without calling Python estimator methods.
- IVF snapshots use the original resident prepare/search or partial-search
  entries. Admission materializes a resident source index natively when needed,
  preserves its handle and splits its original stored rows without retraining.

The driver retains its partition order, worker mapping, candidate merge, input
read-shift negative control and result placement. Radius results retain ragged
lists and do not acquire the ordinary public-output conversion. Ordinary outputs
use the shared `parallel_public_result` native helper (dependency:
`lane/native-parallel-wrappers-20261025`, helper added in `a8e3d528a`). That helper
retains internal raw outputs and external NumPy conversion/regressor widening.
Single-output named labels use the existing native decoder; joining string/bool
label lists concatenates query rows instead of treating them as output columns.

The obsolete `_parallel_shard_ops.py` callback module is removed. Its two
distributed-verifier inventory/import references and the coverage audit's three
descriptions are updated. No verifier lane or arithmetic registration is added.

## Validation and required runtime evidence

Completed locally, without importing the product or compiling:

- Nine source/hostile-object tests check one-call public handoff, untouched
  invalid inputs and keywords, compiled-entry snapshots, callback-registry
  removal and retained verifier source-file inventory.
- Fourteen existing parallel-driver/kernel-method source checks pass, including
  shard placement, negative-control and plan-cleanup checks.

Complete logs and exit codes are in
`~/mojolearn-evidence/parallel-neighbor-output-final-20261025/`.

This lane has **no Mojo compile or runtime qualification**. The orchestrator must
merge the shared output helper first, compile the integrated source once, then
run the affected installed APIs. Existing focused suites:

```sh
MOJOLEARN_PAR_DEVICES=0 python -m pytest -q \
  mojolearn/tests/test_parallel_public_drivers.py \
  -k 'scaler or queries or reference'
python -m pytest -q mojolearn/tests/test_distributed_ivf.py
```

Repeat the affected physical multi-device checks with `0,1` and reversed device
order on the owned NVIDIA/AMD boxes, retaining the existing native read-shift and
transport negative controls. The exact configurations requiring explicit probes
are KNN uniform/distance weights, multi-output labels/probabilities/regression,
string/bool label joins, RBC nearest neighbors, sorted/unsorted radius results
including zero-neighbor rows, KDE, scaler inverse transforms, repeated resident
queries and both IVF squared/rooted distances. Internal raw/external NumPy dtype
boundaries must be checked. These are pending, not implied by source tests.

Standalone estimator methods are outside this change. Source-only inspection
must be reconciled with import-time installation and actual symbols before
claiming their runtime status; no whole-library compliance claim follows here.
