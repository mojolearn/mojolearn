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

## Native strided gradient preparation

Strided FP32 gradients now gather raw words with the existing native host
parallel worker policy. Contiguous spans remain borrowed. The plan consumes
only the admitted address, shape and signed strides, retaining the original
bit patterns (including signed zero and nonfinite payloads) without numeric
conversion. Boundary owners/exports remain pinned; gather runs with the GIL
released and synchronously joins before the optimizer kernel executes.

Removed: PyBuffer_ToContiguous, the second buffer export, fabricated Py_buffer
records for Array storage, their release helper, and the redundant microbatch
owner-vector collection after the single admission-pin vector became sufficient.
No optimizer update/reduction kernel changed. Root owes signed/reversed/F-order
strided fixtures, public gradient mutation/error/lifetime checks and compile.
End-to-end GPU measurements remain owed; this is no performance claim.

## Shared output allocation and native keyword policy

Optimizer arrays now use the existing admitted native storage/layout boundary,
including its checked dimensions/strides and exact Array owner publication.
Removed: the duplicate size/stride allocator, temporary Python bytes/list/tuple
storage construction, buffer_info call and fresh memoryview symbol lookup.
Fixed keyword decisions/text are native, so the remaining keyword references
to the former `_not_covered` helper are resolved and its Python refusal caches
are removed. The door connects its already-loaded module without importing it.

The alignment query now calls its existing native contract predicate directly;
its Python parameter list and redundant compiled-entry connection are gone.
Remaining scalar readback uses the shared admitted native conversions. These
are plumbing/metadata changes; kernels and profile constants are unchanged.
Root owes full affected-closure compilation and public output/constructor/
keyword/metadata checks. State/checkpoint planning is still outstanding.

## Typed state snapshots and bounded checkpoint spans

`OptimizerStateSnapshot` now supplies native handle/residency/pristine/moment
addresses to the existing PublicStep, and native plans decide host freshness,
lazy mirror allocation and checkpoint lengths. Public moment/state arrays are
unpacked once into retained TensorViews; load validation reads native dimensions
rather than Python size properties.

A concrete unsafe path is fixed: moment assignment previously passed any writable
buffer address to an upload of n_total floats. Native span validation now refuses
wrong dtype, extent, layout, readonly/null storage and shape overflow before a
read/upload. A valid assignment uploads first and publishes its public field only
after success. Valid writable/rebound FP32 state still aliases the caller's buffer;
checkpoint loading still owns its converted/copied storage. No algorithm/state
pool kernel changes. Native resident pools remain the actual buffer owners;
public fields are compatibility mirrors/UNPACK inputs, not a new backend.

Root owes direct span/refusal fixtures and public assignment/rollback/ownership
coverage, plus compile. Remaining cleanup includes checkpoint input preparation,
registry metadata and private legacy Python helpers; the family is not complete.

## Parallel checkpoint preparation

Checkpoint and beta-array admission now captures native TensorViews and export
pins once. Rank/count and endian/stride planning use those values; converted
checkpoint storage is prepared through the existing shared allocator. Removed:
repeated Python ndim/size/strides/itemsize/address reads and dtype.isnative checks
from these paths. Native checkpoint conversion runs after the GIL is released.

`optimizer_state_cast` now splits independent output elements through the
existing host worker policy. Its conversion expressions and input bytes are
unchanged. Worker refusal records select the earliest invalid input index, so
exception type/text does not depend on scheduling. Converted scratch is private
and published only on success; disjoint later scratch chunks may complete when
another chunk refuses. The native fixture lane is updating its incidental serial
scratch-tail assumption while retaining source, guard, error and public atomicity
coverage. Empty conversion remains valid; nonempty null/out-of-range spans refuse.

Root owes the existing 35 native conversion fixtures, parallel/error cases,
public checkpoint admission/ownership/rollback tests and compile. No update
kernel changed and no performance/identity claim is made.

## Import-time owner declaration

Removing the door's runtime import exposes its sole package-owner dependency:
`mojolearn._training_impl._OPT_CONTEXT`. That completed context is now declared
through the existing import-time connect_module mechanism after the optimizer
bridge returns. The connection checker records exactly this export. This fixes
the normal push refusal without a guard exemption, runtime import or new loader.
Root owes the existing connection-tooling and fresh-import/public-constructor
checks; product/native logic is unchanged from the prior freeze.

## Audit repair: constructor refusal precedence

Native early checks now preserve accumulation before maximize/beta/scalar
conversion, and SGD Nesterov checks before lr conversion, including zero
momentum refusing before dampening is decoded. Final scalar decoding retains
lr then weight_decay then family-specific fields. The typed full validation
method shares these native checks; no Python staging is restored. New focused
public cases exercise conflicting invalid arguments and unchanged parameter
storage. Registry-versus-scalar precedence follows in the next required batch;
root must include both repairs before its consolidated build.

## Audit repair: registry preparation before final scalar decoding

Constructor preparation now has typed early configuration, native registry
preparation, then final scalar decoding, matching the established refusal order.
The registry uses the same offsets operation and allocates each public output
buffer once; the prepared record holds only publication owners plus a native
OptimizerRegistryPlan. It introduces no data copy or second registry pass.
The same prepared registry publishes in constructors, set_params and the legacy
registry export. Live owners remain untouched until preparation succeeds, and
failed replacement retains its prior fields/storage.

Public precedence coverage now includes empty, wrong-dtype and readonly registries
against malformed scalar options, plus set_params rollback/owner preservation.
Root must include this commit together with the early-validation repair for the
consolidated build. No lane compilation/tests were run, and no tuned kernel or
valid-input numeric operation changed. Parent owns the separate TSQR final-copy
and strided-gather coordinate optimizations; this batch does not touch them.

## Obsolete Python optimizer implementations removed

Caller review found no production callers of the old _Optimizer initializer,
SGD/Adam _config methods, dtype/keyword/maximize refusal helpers or their Python
policy dictionaries. The public constructors/methods already enter the native
boundary, so those nine definitions/tables are deleted (259 original source
lines), along with unused buffer-helper imports. The cross-entropy seam test no
longer poisons two deleted optimizer helpers; its active native-boundary checks
remain. Existing schedule/generator/Samba helpers with real external callers are
retained and are outside this optimizer deletion batch.

This removes duplicate runtime-looking Python implementations that could mislead
future edits. Public constructor signatures, output types and native kernels are
unchanged. Root owes import/public-method tests after the coherent constructor
repairs; this lane has run no tests or compilation. Source review is not timing
or correctness evidence.

## Remaining optimizer admission containers

The end-to-end read found Python key-list staging in external checkpoint/instance
metadata admission. Exact dictionaries now use the cached native PyDict_Next;
exact OrderedDict inputs retain their iterator order (including move_to_end),
without a copied Python list. That iterator is solely an external-input lifetime
pin under the GIL; no iterator or Python value reaches RUN. Native fixed string
and tuple identities replace mutable builtins lookups on those paths.

The existing OptDoor connection now retains one typed PyObject_Repr function
pointer for already-admitted scalar failure text, replacing its builtins module
owner and repr call. Its SharedSlot layout is versioned V2; the global CPython
table is unchanged. Error type names use the setup-retained native type.__name__
descriptor. Removed: temporary Python key lists, builtins owner/lookup helpers
and dead imports. No tensor data traversal or kernel changed.

Two public OrderedDict refusal-order/rollback cases are added; the lane has not
run them. Root owes compilation of the training/classical closure and focused
optimizer admission tests. Native RUN remains PublicStep/OptHyper/LrSpec and
native address/count vectors; public API fields, output checkpoint dictionaries,
module/capsule owners and exported-buffer leases stay in UNPACK/PACK. Schedule
preparation and inter-helper IO/load boxing remain the next concrete cleanup.

## Typed built-in schedule preparation

`training/optimizer_schedule_plan.mojo` now owns the scalar warmup/decay/cycle
validation, configuration-name mapping, native evaluation request and status
refusals. It calls the existing `lr_spec_fill` operation unchanged. Constructors
retain their prior scalar conversion/refusal order and Float32 warmup rounding;
readback of mutable decay fields checks finite values before decoding `_step`.
Public schedule fields, configuration dictionaries and `_lr_spec` list results
are still packed at the boundary and may be rebound as before.

The schedule boundary uses cached PyNumber_Index and optimizer scalar decoding
instead of mutable operator.index/int/bool calls. from_config admits four fields
directly, constructs one typed LrSpec and allocates the guarded native owner;
there is no internal Python name table or argument tuple. Native series/scalar
fills run with the GIL released, retaining their native output storage. Array
output uses a direct native buffer address instead of buffer_info dispatch.
No schedule math, exact fallback, block size or kernel was replaced.

Removed: the dead import-time schedule_entries Python dictionary and duplicate
refusal implementation in lr_table_helpers (it imports the same pure native
refusal now). The tests lane owns the obsolete schedule_entries poison-case
repair. Existing direct native schedule math fixtures remain applicable; root
owes compilation/public mutation, conversion and config tests for this batch.
No lane compilation, test execution or performance claim is made.

## Typed IO/checkpoint control words

Optimizer IO now returns its native integer result to the public router. Query
and ownership decisions no longer box that result to a Python float and decode
it again. The documented raw/legacy result is boxed only by its PACK helper;
unpublished device allocations retain native free-on-publication-failure.

Checkpoint load keeps the decoded step native until final public-field packing,
and directly passes its final owned state outputs to the shared upload boundary.
It removes the internal True/True/True ownership envelope while preserving the
legacy compiled entry's external flags and conversion order. The public router
still re-admits the owner after GIL-released state conversion, before reading
possibly rebound fields. Numerical kernels/state upload order are unchanged.
The stale shared training device-upload caller was independently repaired in
14ab3618f9 using the existing native gather and retained export/Array owners.
Root owes the existing IO/checkpoint/lifetime cases; no lane runs occurred.
