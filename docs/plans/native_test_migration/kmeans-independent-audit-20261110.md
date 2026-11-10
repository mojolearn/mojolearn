# Independent KMeans migration source audit

Reviewed production `3a5b05c487` against its parent
`ddbc4fc751434ce9a7a8c489cb1529a8e5ed7ec5`, then reviewed repairs
`d00701687d` and `7b599a4b7d`. The final reviewed source is `7b599a4b7d`.
This is source review only: no compilation, product execution, installed
loading, timing or identity run occurred in this lane.

## Findings and repairs

1. **Missing native owner type registration, found by this audit.**
   `kmeans_owner_prepare` constructs `PythonObject(alloc=KMeansResidentOwner)`
   but the initial migration registered no Python type for that struct.
   The existing native owner pattern requires a canonical type registration.
   Repair `d00701687d` registers it in the sole allocating operation's owner,
   `bindings/_mojolearn_runtime_family_host.mojo`, before exporting
   `family_native_call`. It reuses the established guarded
   `lookup_py_type_object` / `py_type_absent` / `py_type_attach` pattern so a
   later package namespace attaches the existing type. The family dispatch
   goes through this initializer in split and merged layouts. Compilation
   and actual installed allocation still need execution evidence.
2. **Resident provider identity, found by the producing lane during review.**
   Matching originally covered model, centers, binding module and dimensions.
   Repair `7b599a4b7d` also matches the admitted native connection index, so
   any admitted replacement provider cannot receive the previous driver's
   resident handle. Ordinary cross-module factory reassignment was already
   refused by `neural_descriptor_entry`'s exact factory-module ownership
   check; this repair should not be described as fixing unrestricted foreign
   module callbacks. The native owner fixture now covers a connection change.

No remaining concrete source defect was identified in the reviewed scope.
That is not a claim that the code builds or runs.

## Call-path and ownership review

- The six-operation descriptor retains explicit lengths and operation slots.
  Fit carries six addresses, thirteen integer fields and two Float64 fields;
  query carries four addresses and ten integer fields. The shared result
  words retain inertia, iterations, sum/weight scales, finite status and fold
  range status. Fixed request envelopes avoid reconstructing Python argument
  lists between compiled functions.
- Admission uses an exact module namespace and an exact compiled factory
  owned by that module, then an ABI-named capsule and native descriptor entry.
  Connection metadata and provider pins are obtained while the GIL is held.
  RUN uses a copied native driver; a local provider reference remains live
  until after the GIL-released call completes. A connection-table resize
  cannot invalidate that driver or image pin.
- Query retains its resident owner, original centers, converted centers,
  admitted input and destination through synchronous execution. The owner
  carries the minting driver and a provider/image pin. Dropping the model's
  reference retires the handle only after an in-flight query drops its own
  reference. `take_handle` clears the native handle before quiet teardown.
- Owner preparation initially owns the returned handle locally; constructing
  the movable native owner transfers cleanup to that owner's destructor before
  Python allocation. The provider and centroid storage remain pinned on both
  paths. Allocation-failure cleanup remains an owed runtime check, not a
  source-only proof about exception unwinding.
- Resident reuse is restricted to the same freezable, frozen native Array,
  model, module, connection and dimensions. Writable/rebound caller buffers
  retain their existing per-query admission/upload behavior. An ordinary
  NumPy read-only view is not newly treated as an immutable native owner.
- `getstate` copies the exact instance dictionary and excludes the resident
  owner. `setstate` accepts the class's native dictionary format with exact
  string field names and clears residency. Copy/pickle therefore remint an
  independent handle; the retired Python lifetime registration is removed.
  The generic neighbors finish hook only creates dictionary holders for its
  own `family_xn_knn_` operations and does not overwrite the KMeans owner.

## Algorithm, admission and copying review

- GPU fit still calls the same `cluster.estimator.kmeans_fit`, with metric,
  seed, restarts, initialization, tolerance, oversampling and weight count in
  the same positions, `lazy_shift=False` and the same zero fixed argument.
  Absent `transform_out` remains an empty Optional; fused fit-transform passes
  the actual destination pointer and keeps the single-upload numerical path.
- Resident predict/transform still call `cluster.resident_centers` operations.
  DeviceRows fit/query still use the existing store context, gather kernels,
  finite/range words and unit-weight fold fit. A weighted fold or unavailable
  fold capability retains the previous host materialization branch; this
  audit does not claim that old branch became a GPU-only operation.
- Host fit/predict/transform still call the existing `core.core_host_impl`
  functions. Deferred finite scanning and output-pointer admission remain in
  the native path. Host integer fields retain `operator.index` semantics and
  Boolean refusal via `PyNumber_Index`; GPU integer conversion keeps its
  former `Int` semantics. Explicit conversions of tolerance/seed and their
  order remain in UNPACK. Unsupported custom scalar/protocol objects are
  refused before a conversion hook can execute.
- No numerical kernels or data-sized loops were rewritten. Tensor-sized
  input/initial-center/output copies and resident/fold routes are preserved.
  Metadata allocations changed: typed bounded request/result storage replaces
  Python lists/boxed scalar results. This does not establish a speedup or
  prove the absence of a performance regression.

## Evidence and remaining checks

Raw old/new sources, repair snapshots, full patch, source hashes and the
independent review script are preserved under:

`~/mojolearn-evidence/python-mojo-migration-20261110/kmeans-independent-audit/`

`independent-source-audit.json` records a conservative repository import walk
of 135 sources, including the owner state and local package prefixes, with no
Python imports. This is a manual review aid, not an installed source guard or
an execution proof. Five direct numerical files have identical source bytes
before/after the migration: `cluster/estimator.mojo`,
`cluster/resident_centers.mojo`, `cluster/kmeans_fold.mojo`,
`core/core_host_impl.mojo` and `cluster/host/kmeans_oracle.mojo`.

The producing lane's `e18ebd3562` and `7b599a4b7d` supply 26 descriptor/owner,
6 CPU kernel and 9 GPU kernel fixtures plus seven Python boundary cases.
They were not compiled or run by this audit. Actual registered-owner
allocation, readonly/rebound storage behavior, copy/getstate independence,
in-flight retirement and allocation-failure cleanup remain runtime evidence
owed. These checks do not add an installed-wheel gate to development merges.

Deletions reviewed: Python parameter/result connections, optional-door probing
cache, private resident dictionary/tuple state, and the KMeans Python lifetime
registration. No experiment, tuned kernel or historical evidence was deleted;
no A/B define or release action was introduced.
