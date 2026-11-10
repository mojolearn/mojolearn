# Mojolearn linear boundary pilot

This is a development handoff, not a runtime, performance or identity verdict.
Root owns compilation and measurement. The lane has run no builds or tests.

## Operation inventory

- Ridge fit: typed requests, scratch buffers, weighted target packing, solver
  selection and intermediate results in `core/linear_fit_run.mojo`. Existing
  weighted/one-feature, Gram, resident and general kernels are retained.
- LinearRegression fit: typed preparation and solver selection in
  `core/linear_ols_run.mojo`; native matrix ownership and the existing TSQR/SVD
  operation sequence in `core/linear_tsqr_run.mojo`. Existing width limits,
  rank decisions, weighted preparation and resident switches are retained.
- Predict: typed request and model-state validation in `core/linear_*`; existing
  narrow and wide prediction kernels remain the operations of the native table.
- Score: native R² program, arena, input/output ranges and scalar decode in
  `core/linear_score_*`. Prediction writes directly into that arena. The shared
  `classical_r2_score` helper uses the same native execution; its QN/x_linear
  callers changed only their imports and descriptor-factory connection.
- Constructor, get/set params and fitted-state metadata: native parameter names,
  validation and key decisions in `core/linear_metadata.mojo`. Explicit boundary
  code transfers exact external parameter owners using CPython dictionary APIs.
- Mode normalization and connected-provider mode/vendor checks use native
  metadata. Unicode whitespace spellings retain the prior accepted behavior
  without calling Python string methods. Public mode/vendor reads use the same
  descriptor table, and fitted admission no longer boxes a Boolean only to
  immediately unbox it.
- Save/hash: the separate archive lane's native writer/hash batch is incorporated;
  public dispatch uses its native byte ownership and serialization. Archive read
  completion remains with that lane. The complete pilot is not yet accepted.
- Shared float32 admission: `core/admitted_f32_plan.mojo` owns rank, shape,
  byte-order, borrow/layout/cast, target and weight decisions. Existing parallel
  cast, finite and weight kernels run with the GIL released. The classical X
  shape verdict is reused, and pilot callers consume the admitted native address
  and dimensions. Exact exporter/protocol admission and owner pins remain in
  the boundary.

## Ownership contract

Public `coef_` and `_x_mean` arrays remain writable and rebindable. They do not
become read-only views merely because an older design document proposed that.
Before prediction, the boundary admits the current coefficient owner and reads
its current buffer descriptor, count, feature count and intercept. Public scalar
and coefficient owners are captured before data preparation releases the GIL,
so concurrent refits cannot mix scalar owners from different snapshots. Native code
receives those values, so external changes remain visible.

`LinearModelState` contains native addresses, dimensions, scalars and flags.
`LinearStateOwner` lives only at the Python boundary and retains the public
coefficient and mean owners. The existing weak model-record registry retains
this envelope without a strong reference back to the model. `set_params` drops
the state record when it resets fitted attributes. Each released-GIL operation
uses its own native snapshot and retains its admitted arrays and extension
providers until completion; no native RUN function receives a Python object.

TSQR host storage uses native owners and reference counts for matrix views.
Uploading a view releases its host ownership when no view still needs it.
Device buffers belong to the existing pool; one allocation ledger owns each
successful allocation, including exceptions. Cleanup attempts every release
and then reports the first release error. Final resident coefficients download
straight into the public output buffer.

## Removed mechanisms

Removed Python intermediate arrays, boxed parameter lists and dynamic per-step
calls from Ridge/OLS fitting, the Python-backed TSQR matrix manager, Python R²
arena ownership and result arithmetic, parameter-name partition/sort calls and
fitted-key string-method calls. The old classical Ridge/OLS/TSQR/score/metadata
binding implementations are deleted. Existing low-level address APIs remain
for their other callers; no tuned kernel or experiment was deleted.

## Validation and measurement owed

Affected binding groups include estimators, x_linear, x_decomp, x_metrics and
their host images, plus the classical runtime's shared dependencies. Build
closures determine any additional groups. The root integration repair fixes
import ordering and native result moves; lane commit history retains it.

Native contract tests are in a separate lane. Runtime coverage must include
weighted/unweighted fits, intercept on/off, one-feature Ridge, TSQR full-rank
and deficient-rank inputs, the generic OLS route, prediction output dtype,
score range I/O, parameter resets, mutable/rebound fitted attributes and archive
roundtrips. Compilation alone does not establish any of those behaviors.

Shared registry reuse now snapshots native owner identities, parent links,
addresses, dimensions and finite flags. Native code traverses ancestry and scans
buffers while boundary owners stay pinned. After reacquiring the GIL, verdicts
are applied by owner identity and descriptor agreement, so another thread's
registry compaction cannot make a saved index point at a different buffer.
DeviceRows shape/store checks and gather-status decisions likewise run on native
values; the existing gather is unchanged. Exact exporter adapters still require
final call-graph review, and none of these changes is proof from an import scan. Public attribute and
exception-class operations are Python protocol admission/packing, and remain
subject to exact-type/hook refusal review. Archive read remains outstanding.
The pandas exporter now transfers block descriptors into
`core/pandas_input_plan.mojo`: block dimensions, dtype promotion, placements,
nullable-mask validation and scatter decisions use Mojo values. Numeric rows
still use the existing parallel scatter kernels, now with the GIL released and
each source owner pinned. Exact object cells require GIL-held C-API decoding;
they write through a native conversion/finite-policy writer directly into the
final result, preserving the previous conversion order without a staging copy.
This decoding cannot be parallelized under the supported GIL contract. Old
Python owner/placement lists and duplicate metadata rereads were removed.
Compatibility metadata wrappers for other families share the new native plan;
those families are not claimed as otherwise migrated. Python sequence
preparation is owned by the root integration lane and remains unaccepted here.

No arithmetic, fold order or tuned-kernel change is intended. No development
identity run is requested. No speed result is claimed. End-to-end NVIDIA/AMD
measurement of the affected Ridge and LinearRegression recipes remains owed;
the native plumbing has no new experimental kernel define.
