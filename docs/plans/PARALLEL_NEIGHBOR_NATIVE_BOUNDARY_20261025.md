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

## Standalone query follow-up

The selected-operation inventory omitted `kneighbors`, `radius_neighbors`, and
`search`; its native counts did not qualify these methods. Source inspection of
class installation, mixin wrapping and archive lifetime setup confirmed the
standalone methods remained live Python operations. The broader endpoint gate
is being repaired separately.

`NearestNeighbors.kneighbors`, `RadiusNeighbors.radius_neighbors`, and
`IVFIndex.search` now hand off once to `neighbor_query_call`. Import-time method
markers prevent the old output/validation wrappers from wrapping them again.
A native attribute descriptor replaces `PublicOutputsMixin.__getattribute__`
for these connected classes and the inherited classifier/regressor query
surface; its native cases preserve query diagnostics, resident state, classifier
class labels and IVF's lazy `list_data_` materialization. This does not migrate
or qualify every other method or constructor on these classes.

The native owner reuses the worker search helpers. It retains brute-force/RBC
metrics, the optional resident path, ragged radius lists and sorting, IVF
squared/rooted metrics, boolean per-row filters and the per-call IVF fallback.
The existing methods' raw/NumPy output boundary remains: k-neighbors publishes
NumPy results to external callers; radius and IVF retain their original raw
Array results. Ordinary NumPy scalar integer/boolean/radius inputs are admitted
natively. Python-defined input protocols are refused before their hooks run.

Fixed GPU-tier and explicit saved-model CPU routes are captured as plain modules
of verified compiled functions during import. The original resident binding
identity is retained in a private map to its compiled function table, so
materialization/release never invokes a lazy Python binding proxy. Native route
selection checks the requested numeric mode and the compiled mode readback.
`HostNearestNeighbors`, `HostKNeighborsClassifier`, `HostKNeighborsRegressor`,
`HostRadiusNeighbors` and `HostIVFIndex` remain connected to explicit CPU binaries.
No new Apple-fast work or evidence campaign is authorized by this shared repair.

Seventeen boundary tests pass (eight new standalone tests plus the previous
nine worker tests), with logs in `standalone-boundary.log` and its `.exit` file.
They check untouched hostile-input forwarding, suppression of mixin wrappers,
builtin-only capture, proxy resolution only during setup, explicit CPU routing
and removal of IVF's Python lazy getter. They do not execute Mojo.

After the integrated build, run the installed existing neighbor/IVF suites,
including `test_knn_metrics.py`, `test_neighbors_sparse_input.py`,
`test_host_radius_bindings.py`, `test_ivf_surface.py`,
`test_ivf_embedding_cpu_inference.py` and the affected tests in
`test_neighbors_knn_native.py`/`test_neighbors_ann_native.py`. Keep the runtime
matrix above, adding filtered/unfiltered and strided boolean IVF filters,
ordinary bool lists/NumPy booleans, lazy resident data reads, repeated search,
save/load CPU queries, optional resident-entry absence, binding mode mismatch,
and callback/descriptor negative controls. These checks are pending; this
source change is not runtime qualification or whole-library compliance.

## Integrated compiler diagnostics repaired in this lane

The orchestrator's first frozen compile at `4f7bd6049` identified a reserved
`out` parameter in `prep_labels_native._class_words`, ambiguous Mojo attribute
syntax for the native list slicing in KNN class validation, incorrect getpid
FFI generic argument order, and a missing `raises` on shard trace initialization.
Commits `fa963b3e6` and `6b4d24858` repair these source errors without changing
arithmetic. Full original failure logs remain in
`~/mojolearn-evidence/takeover-20261025/0857-mac-freeze-4f7bd6049/`.
No rebuild or runtime success is claimed by this lane.
