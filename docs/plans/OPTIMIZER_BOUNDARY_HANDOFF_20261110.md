# Optimizer boundary migration

## First batch: OpsSession configuration and checkpoint values

`Session.optimizer`, `optimizer_state` and `load_optimizer_state` now decode
into typed `OpsOptimizerRequest` / `OpsOptimizerRestoreRequest` and pack an
`OpsOptimizerStateValue` from the native session. Name-to-code resolution runs
on native strings. Validation and executor state ownership remain in
`autograd/ops_api.mojo`, using the existing `OptimizerConfig`, `OptimizerState`
and update kernels. State buffer snapshots/copies retain their original
lifetime: returned handles remain valid until reset, and restoration copies
all admitted state buffers on the session executor after validating them.

Removed: the redundant Python `_opt_kind` dictionary and its unused private
`_kind_of` implementation; integer/float staging lists in these Python
boundary calls; bound `state.__contains__` calls; and the builtins module
lookup used only to construct an output dictionary. Public state dictionaries
are constructed only at PACK. Exact supported argument admission now precedes
scalar/checkpoint conversion, refusing unsupported hooks without executing them.

The existing Mojo list and tuple APIs remain as small native adapters for
supported callers. They call the same canonical typed implementation. No
optimizer formulas, update kernels, schedules or float conversion ordering
inside native execution changed. No A/B switch is added and no speed claim
is made.

This is an intermediate batch. Public `SGD` / `Adam` / `AdamW` still route
through `optimizer_native_router` and `optimizer_public_door`, whose Python
configuration/state and compiled-Python-entry dispatch remain to migrate.
Those paths already call typed `PublicStep` / `OptKernels` operations; the
migration must preserve those tuned operations. Other Session operations also
still cross a request that retains Python boundary objects. Session registry
and call ownership retain their existing GIL discipline in this batch;
releasing it independently would require coordinating concurrent close/reset
and other operations on the same native registry. This batch does not claim
the whole optimizer family or Session API is migrated.

## Integration owed

No lane compilation or tests were run. Root owns affected-closure planning and
compilation: direct roots include `_mojolearn_x_sequence` and its host peer;
`autograd/ops_api.mojo` is shared, so all transitive reaching groups must be
included by the existing development controller. Native optimizer/session
fixtures and Python public admission, state mapping, mutation and lifetime
checks are owed. The native tests lane retains existing list/tuple API tests
and can add direct typed request/snapshot checks. End-to-end GPU performance
measurement is still owed; no development identity run is requested.

## Native public-optimizer operation interface

The `OptKernels` interface and its GPU/host implementations now live in
`training/optimizer_kernels.mojo`, `training/optimizer_kernels_gpu.mojo`, and
`training/host/optimizer_kernels.mojo`. These modules carry only typed native
requests/results and import no Python support. They call the same existing
public optimizer operations. The Python entry files retain only their legacy
boundary exports and import these typed implementations.

GIL management moved to `optimizer_public_door.mojo`, immediately around the
six typed RUN calls. Every release/retention scope remains the same: host state
upload/download and host IO query keep their former GIL discipline; step,
clip, accumulation and GPU transfers release it. The accumulated microbatch
count is captured before release, so no Python container is read during RUN.
This removes the old Python-aware kernel adapters and their duplicate context
key declaration. It does not yet remove the outer router's compiled Python
entry calls or its persistent Python config/state fields. Those are next.

Additional direct build roots are `_mojolearn_training` and its host peer.
Compilation and focused public optimizer/state/ownership checks remain owed to
root. No native optimizer update function or measured kernel was edited.

## Typed optimizer descriptor producer

The training images now export `optimizer_native_api_v1` using the existing
native descriptor ABI. `OptimizerNativeDriver` sends typed `PublicStep`, clip,
accumulation and resident-transfer values to the same host/GPU operations.
Writable requests/results remain heap-owned and pinned until the native call
returns; each call owns its scratch, including its error sink. No Python
objects cross this interface. Native open/close use the existing slot pools.

This producer batch is incremental: public routing activation follows in a
separate batch, so no old dispatch is removed yet. It adds no kernel switch,
changes no optimizer equations, and makes no performance claim. Root must
compile both training binding groups and run native descriptor refusal,
state/ownership and public optimizer checks after activation.

## Public typed descriptor activation

Public step, resident IO, clipping, accumulation and state transfers now enter
shared UNPACK/PACK functions directly, then use `OptimizerNativeDriver` for
RUN. Legacy compiled exports call those same boundary functions. Removed:
router calls through compiled Python optimizer entries; the intermediate step
argument tuple and exception relay; the Python mode-check dictionary; runtime
Python mode strip/lower calls; and foreign-owner state transfer calls through
Python entry objects. The import-time bridge now retains the single optimizer
descriptor factory instead of the redundant per-operation entry table.

The explicit descriptor boundary keeps admitted provider/image pins and caches
native host/mode metadata while holding the GIL. Native work receives only the
function pointer, typed fields and native buffers. Host registry transfers
retain their existing GIL discipline, GPU work releases it, and each descriptor
request owns its heap result/error storage. Public arrays and rebound fields
remain admitted on every operation. Exception prefixes cross the native error
sink unchanged and are translated once at the outer Python boundary.

Still incomplete: constructor/get/set-params preparation, public persistent
state snapshots, some boundary helpers' Python staging containers, resident
finalization and allocation-failure cleanup. Those must migrate before calling
the public optimizer family complete. No update kernel changed. Root owes
both training/core closure compilation, focused public mutation/ownership/error
checks and native descriptor tests; end-to-end GPU measurement remains owed.

## Native optimizer resource release

Resident close during parameter replacement/finalization and device-handle
cleanup after failed publication now call the typed descriptor. Removed:
mutable builtins lookups, bound object attribute calls and compiled Python close
calls from the optimizer finalizer, plus the obsolete `resident_entry` helper.
Exact owner fields are read by the existing native dictionary API; module and
capsule owners remain pinned by the descriptor boundary. Host close retains the
GIL, GPU close/free releases it, and the established best-effort finalizer keeps
ownership fields when cleanup fails. Native descriptor operation count is 11
(the added operation only frees an unpublished device allocation).

Configuration/state preparation and staging cleanup remain outstanding. Root
owes compilation and resource-failure/finalization checks. No math changed.

## Typed constructor configuration

Constructor scalar values and beta pairs now decode into
`OptimizerConstructorConfig`; accumulation/Nesterov decisions use that native
value. Public fields are packed from it. `get_params` uses native fixed name
lists. Removed: internal Python beta tuples, the Python parameter-name list,
constructor calls through the compiled Python registry initializer and its
unused import-time connection. Exact admitted scalar conversion now uses
cached CPython number/truth functions; no mutable builtin float/int/bool call
is involved, and the established text/number conversion behavior is retained.

Transactional `set_params` still stages its old dictionary/envelope and will
be migrated next. Public field/schedule owner pins remain necessary at PACK.
Root owes constructor/get-params/error checks and native config validation;
this lane has not compiled or run tests.

## Typed parameter replacement

`set_params` now validates native parameter names, decodes a typed constructor
configuration and shares the same publication function as construction.
Removed: the internal Python configuration dictionary, reconstructed argument
tuple, Python keyword/owner-field lists, bound string partition, and bound
clear/update calls. Exact admitted dictionaries are unpacked through cached
CPython iteration. The existing CPython table gained only Clear/Update for
publication and rollback; no new symbol loader or runtime import was added.

A shallow Python dictionary remains solely as a boundary rollback owner for
public attributes, buffers and the old extension pin while replacement is
published. Algorithm decisions use the native configuration. Failed replacement
restores those fields; successful replacement closes the old slot natively.
This preserves writable/rebound public attributes and custom-field clearing.
Registry/checkpoint preparation and persistent state planning still remain.
Root owes get/set-params mutation/rollback/refusal checks and affected closure
compilation. No lane tests or compilation were run.

## Native tensor plans and boundary owner vectors

Parameter/microbatch staging now uses native vectors. Python lists are created
only for the public `params` and accumulated-gradient results. Removed: private
Python list construction/append calls, an unused sequence adapter, redundant
subclass/isinstance paths after exact type admission, and Python mixed-precision
message lookup from the tensor plan. Layout, dtype, gradient shape/count and
microbatch-shape decisions now operate on native `TensorView` values in
`training/optimizer_input_plan.mojo` (its shared TensorView module has no Python
imports). The existing update kernels and reduction order are unchanged.

UNPACK retains both exact object/storage pins and the existing buffer exports
through native execution. Each Py_buffer lease reuses its admission allocation
and captures the existing cached release function; native RUN receives neither
Python owners nor leases. Error unwinding releases them after the GIL scope
returns. Array storage memoryviews are retained from the same slot admission,
without rereading or copying the buffer. This also prevents ordinary exporter
resize while native code still borrows its storage.

Remaining: strided preparation still calls PyBuffer_ToContiguous; registry and
checkpoint/output planning plus persistent optimizer state still need migration.
No family-complete or performance claim is made. Root owes compilation, direct
layout/owner fixtures and public mutation/concurrency/refusal checks.
