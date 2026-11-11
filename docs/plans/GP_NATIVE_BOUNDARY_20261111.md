# GP boundary migration

The first chunk is additive. Public GPR/GPC calls still use their current
boundary until the following switch; this chunk alone does not remove their
Python containers or claim the family is migrated.

`core/gp_native_api.mojo` defines ABI `mojolearn.gp.native.v1`, 17 operations,
12 named requests, `GPKernelView`, `GPQueryResult`, `GPNativeKernels` and
`GPDriver`. `gp_native_dispatch[K]` implements the existing synchronous
fixed-span descriptor protocol. All address owners must survive the call.
Two query status words use heap storage, following the proven descriptor ABI.

The provider types are:

| Provider | Existing execution routes |
| --- | --- |
| `GPDeviceKernels` | List, tuned pointer and resident GPR; binary and fused all-member GPC; covariance, sampling, gradient and optimization |
| `GPHostKernels` | Full native host training and prediction, covariance, sampling, gradient and optimization |
| `GPHostInferenceKernels` | GPR mean/std, GPC binary prediction and OVR combine only |

Inference-only host code does not import the training/optimization provider.
Capability bits 1/2/4 preserve the existing OVR/pointer/resident choices; new
bits describe fit/covariance/sample/gradient/optimizer/theta availability.
Vendor codes reuse the existing native provider convention (CPU = 1).

The existing `gaussian_process` numerical bodies are unchanged. The adapters
retain their input-copy lengths, float32 narrowing, seed halves, call order,
mean-only allocation choices, optional binary target construction and output
packing calls. The all-member APIs borrow native address tables; they do not
copy fitted tensors. Ordinary and resident GPR remain distinct routes. Query
fold IDs stay explicit, and resident leases belong to the provider image's
existing shared store. Its metadata already uses the existing store lock;
GP's context resolves through synchronized `neural_ctx` / `process_ctx`.

`core/gp_native_storage.mojo` provides owned postfix tables, regression fit and
gradient scalar scratch, optimizer scratch, and per-member address/scalar
storage. These are independent of public mutable arrays and public estimator
member objects. Kernel-table `finish()` runs once after boundary unpacking;
it retains the unused 1.0 word for a kernel with no length scales. Seed
admission and arbitrary-size integer masking remain boundary work.

The following provider chunk exports `gp_native_api_v1` from all three GP
images through native `GPProviderAPI` storage. Host export inventories include
the factory; this is manifest/source coherence, not tested loading.

Pending next chunk: connection pins, exact admitted kernel
unpacking and cached state, public fit/query/optimizer switches, canonical
field-owner pin inspection where needed, and deletion of newly dead request
envelopes. Public kernel objects, fitted output arrays, mutable GPC member
objects, warning/output packing and mandatory storage/image pins remain
boundary responsibilities. No public feature is deprecated.

Validation for this additive chunk is source review only. No lane compilation,
test execution, performance measurement or identity run occurred. Root owns
the combined compile and native fixture run. The native test lane owns
`gp_contract_*`, `gp_state_*` and `gp_native_*`; its original reference inputs
are frozen at `b469feccbe`. Existing Python public/external-oracle tests remain.

Affected future build roots: `_mojolearn_gp`, `_mojolearn_gp_host`,
`_mojolearn_gp_infer_host`, and the GP public lifecycle/runtime host groups.
The provider export chunk changes the three GP binding roots above. No A/B define
or default changes; no intended numerical changes. No kernels or old paths
deleted yet because public switching is pending. The separate Modular shared
context submission problem is neither fixed nor worked around here.

GPR request switch (staged public chunk): fit, mean/std query, covariance,
sampling, optimizer, likelihood-gradient and theta-map execution now use
`GPConnection` / `GPDriver`. GIL-free calls retain the admitted image and all
buffer owners. Fit and gradient scalars, predictive variance, optimizer
theta/value/run scratch, free flags/bounds and likelihood return metadata use
native owners. Python gradient lists and optimizer records are created only
while packing public results. Post-RUN namespace keys are re-admitted before
publication. A fit snapshots the normalization setting and owns its ridge
retry scalar; the same tenfold retry rule and warning remain.

This is not a completed whole-family switch. Kernel-array packing/cache
records, fitted resident lease records, StandardScaler normalization, GPC and
cross-interpreter GP workers are still staged dependencies. Existing missing
legacy-entry checks temporarily preserve boundary refusal positions; they
do not execute those numerical entry callables. They will be replaced by
native capability admission with the complete provider switch. Source-only
review is not runtime, performance or identity evidence.

GPC request switch (next staged public chunk): binary and fused all-member
fit/query and OVR combine use typed requests and native scalar/address tables.
Member output references and latent/combine results have typed boundary pin
envelopes. The OVR likelihood still uses the same `reduce_stat` operation and
class order, backed by a native scalar list. Worker fit/query and parent combine
share these native calls; cross-interpreter transport, worker counts, stored
request reuse and scheduler behavior remain unchanged. Fused route/default
switches remain in place. GPC scalar scratch Arrays and address/parameter/result
Python envelopes were removed. Public estimator/member objects and output
Arrays are retained; selected latent scratch and label-code wrappers still await native storage
completion. Public and interpreter-transport outputs retain the supported
exporter-backed Array PACK; no ownerless memoryview is introduced.

This chunk additionally affects `_mojolearn_runtime_parallel_host` and the
runtime family dispatcher. Its request constructors and unchanged numerical
source were audited, but no lane builds or runtime tests were run.

The subsequent metadata cleanup removes the temporary legacy numerical-entry
existence checks. Capability admission now belongs to the actual native GP
descriptor, including inference-only refusals. Native mode/capability rows
replace the two boxed scalar cache payloads; rows retain the exact provider
pin and follow the existing descriptor cache's 64-row bound. A replaced
factory obtains fresh metadata. These native rows are UNPACK state protected
by the GIL, not Python objects needing class registration.

The GP worker import prerequisite now requires `gp_native_api_v1`. Seven
unreferenced GPC Python helpers/private method bodies (209 lines) were removed,
including their buffer-helper/Array imports. Public methods and archive field
classes remain. `numbers` stays as an import-time archive class-table dependency.
Private `_extension` remains used by resident/benchmark controllers; loader
mode/vendor probes and raw numerical exports still have compatibility consumers.
The saved inventory distinguishes those references from ordinary product calls.
No tests or numerical source were deleted by this cleanup.

GP resident preparation now uses `GPLeaseState` for the handle, provider
identity, model generation, ordered storage identities and take-once cleanup.
`GPPreparedLease` retains only the provider and mirrored storage pins around
that native state. Its sole runtime-family registrar does not add it to model
field release admission: this is a weak-keyed prepared-registry owner. GPR and
fused GPC keep the existing freeze eligibility, slot order, invalidation and
non-weakref per-call upload behavior. Native descriptor lease calls replace
the opener callable and five-item Python record. Registry metadata operations
retain their original GIL boundary; no tensor copies or numerical calls change.
The shared legacy resident helper remains used by SVM compatibility paths.

The lease addition is source-reviewed only. Compilation and native lifetime
fixtures are queued to the integration owner; no lane execution occurred.

GP normalization now runs `gp_normalize_targets` with a typed request/result
and native column, statistics, program, missing-value and output scratch.
The public normalized target Array is retained as `y_train_` storage. Two
separate descriptors pin the original preprocessing and x_prep images. The
plan uses the existing direct switch, separate fit/transform kernels, exact
264/248 validation stages, 1/101/102 missing-fit stages, 1/24 missing-fill
stages and 10/103 restoration stages. Input-to-column and output-to-flat
copies remain; the temporary Python StandardScaler, fitted-field dictionary,
private Arrays and one-use prepared-row cache are removed. The cache's unused
12-word retained-row copy is removed with it. No numerical source changes.

Additive factories: `scaler_native_api_v1` in preprocessing device/full-host,
`x_prep_native_api_v1` in x_prep device/host. Existing scaler callers and raw
exports remain intact. Import-time scaler context now retains their owning
modules; native connection admits actual factories and verifies mode before
RUN. Required groups include preprocessing, x_prep, runtime family and its
parallel consumers; affected image roots are listed by the export inventory.
Source audit only: the orchestrator still owes compilation, normalization
fixtures (including missing/refusal paths) and installed behavior. No timing
or cross-vendor identity claim is made.

Selected GPC query outputs now use `GPCQueryScratch` native Float32/Float64
lists whenever they are private. The caller selects only actual public or
interpreter-transport outputs for Array packing: public latent mean/variance,
worker probability or mean, and final binary outputs. Nonfused OVR retains
native probability columns and supplies their native address table to the
same combine request. Cross-interpreter columns still unpack through their
own mandatory exporter pins. No tensor copies, numerical kernels or fused
route defaults changed. This removes private latent scratch Arrays and
nonfused probability-column Arrays; native storage stays alive through RUN.
The query owner addition is uncompiled/unrun in this lane.

Packed kernel tables now live in `GPKernelStorage` behind the prepared-only
`GPPackedKernel` owner. Ordinary GP calls pass its native postfix spans;
worker transport explicitly packs real Arrays. The four intermediate Python
value lists and four internal Array stores are removed from ordinary calls.
The shared strict node capture still owns the exact boundary validation and
archive representation; its snapshot owner/state lists are now native lists
of mandatory public-object pins.

`GPPreparedKernel` replaces the five-item packed-cache tuple, Python key
list/count boxes and separate stamped payload tuple. It retains native key
entries, content/generation words, and native lists of object/class pins.
Identity, exact builtin integer-value matching, four-level container capture,
class watchers, mutable parameter invalidation and generation/content fast
reuse remain. The unused old pack-stamp mechanism is removed; the separate
fitted-state admission record remains because it covers other public fields.
Two prepared-only types are registered once in runtime_family_host; neither
is a model field owner. Cache UNPACK still uses exact builtin dictionary/item
views transiently and the shared archive-compatible postfix capture; these
are not passed to RUN. No numerical source changes or lane executions.

GPC fit encoding now returns a typed envelope with the observable class-list
pin and `NativeLabelCodes` Int32 storage. Numeric preparation calls the v2
label descriptor on the originally supplied base image; core host now exposes
the same v2 table over its existing LabelsHostKernels. Numeric class scratch
and strided floating staging are native lists. Text encoding shares its
existing hash/stable-sort implementation with a storage parameter: ordinary
GP calls own native codes, while legacy/shared callers and worker transport
write directly into exporter-backed Arrays. This introduces no transport
staging copy. The old fast text code-copy loop becomes a bulk copy for the
legacy Array output; native output takes the existing code list directly.
The general class-order path reuses the existing native dictionary code
writer, removing Python map/list code reconstruction. Class comparison,
first-label retention, admitted inputs and sorting routes remain unchanged.
Decode table/state cleanup is the next bounded follow-up; no full-family
completion or runtime result is claimed by this encoding chunk.

GP decoding now uses `GPLabelTable` (native int64/float64 gather storage and
availability state) and the prepared-only `GPPreparedClassTable` (native list
of public class pins). Numeric gather calls the explicitly supplied base
image's v2 descriptor. Arbitrary-width integer classes still return the
original objects when an int64 gather table cannot represent them. The cache
retains exact prior label matching, including float bit patterns and fresh
NumPy-scalar misses. The shared old class-table helpers remain for SVC.

Ordinary binary/fused prediction codes and nonfused combine/widening scratch
are native lists. Only actual public probabilities or worker-transport outputs
are Array-backed; the same fused/pointer/resident and combine calls receive
their addresses. Unused combine probabilities use native scratch. The
homogeneous list label admission now reuses `labels_list_scalar_kind`, removing
the Python set/map pair while retaining empty/mixed/bool/out-of-int64 behavior.

Remaining GP boundary debt is explicit: exact input list staging through
`array_type`/`buffer_info`, the general primitive-label dict/sorted/index table,
the shared archive-compatible postfix capture, and the broader GP fitted-state
admission tuple. These are not claimed removed by the compiled RUN migration.
Public classes, fitted model fields, estimator objects, exporter-backed result
Arrays and interpreter transport pins remain intentionally observable boundary
storage. All current chunks still await root compilation/runtime evidence.

### Frozen tranche accounting (source through `de371e5d37`)

The bounded before/after audit records **51 distinct migrated private roles**:
19 state, 18 scratch, 12 boundary-container and 2 shared dispatch-envelope
roles. This is not a runtime allocation count, a struct count, or a family
completion percentage. Repeated calls and alternative prediction-code routes
are counted once. Mandatory public/provider/transport pins are excluded;
boundary-container rows count only the replaced container schema. Every row
has existing before/after source anchors, with no missing anchors.

Baseline: `eb2682df505911ce01dd154d07a0d9be110ea61f` (before the GP migration).
Producer freeze: `de371e5d37`; root compilation snapshot:
`d6090df58b3293250c9c377b1a057f9bb9f7ba39`. The audit's explicit native RUN
roots close over 207 local sources with zero Python imports. The entire
`gaussian_process/`, `preprocessing/` and `x_prep/` numerical source trees
are unchanged against that baseline. Neither fact is runtime, speed or
cross-vendor identity evidence.

Four reviewed boundary mechanism groups remain: exact-list numeric staging,
general primitive-label class/index metadata, shared archive-compatible
postfix capture, and broader GP fitted-state admission metadata. Each has a
current source anchor in the JSON audit; this is not an exhaustive remaining
role denominator. They continue in the next source tranche while root builds
the immutable snapshot. No migration-complete claim is made for GP.

Complete evidence, including all role rows and remaining anchors:
`~/mojolearn-evidence/python-mojo-migration-20261110/python-object-typed-gp-family.json`.
The native source inventory is `gp-native-owner/final-source-closure.json`;
complete conservative build roots are `gp-native-owner/affected-build-roots.json`
in that same evidence directory. The reverse import audit identifies 19
binding roots: GP device/full-host/inference-host; preprocessing device/host;
x_prep device/host; core host; and runtime byte-trainer, classical, decomp,
family, linear, manifold, model-selection, parallel, prep, trees and xseq host
roots. The root build planner owns merged-group selection and reuse.

Live boxed compatibility consumers remain recorded: portable host math
`gp_theta_params`, historical GP predict benchmark `gpr_predict`, and resident
lifetime/upload controller `gp_resident_stats`. Those exports are retained;
retired source-name probes are not treated as reasons to preserve dispatch.
No lane compilation, test, timing, identity run or box job was launched.


### Next tranche: exact-list numeric input storage

This source follows the immutable `de371e5d37`/root `d6090df58b` freeze and
must not be included in its active compilation. `GPLabelInput` now owns
Float64/int64 input words natively. The two GP homogeneous-list paths use
public CPython scalar slots after exact per-item type admission, replacing
private `array_type` owners and `buffer_info()` calls. The existing shared
kind scan, numeric prepare kernel, class ordering and bool publication stay
unchanged. A bigint outside int64 still returns to the existing general label
path. This is scalar UNPACK under the GIL, not a numerical kernel replacement;
the original single input conversion is retained without an extra data copy.
The 51-role frozen audit remains immutable; this pending follow-up is separate.
No lane build, runtime test or timing was performed.

### Following tranche: native postfix capture

Ordinary GP kernel packing now consumes `gp_kernel_native_nodes` directly. The
shared capture keeps postfix records, scale words and its traversal stack in
Mojo containers. It no longer creates a synthetic Python context dictionary,
node tuples or an intermediate Python postfix list. Exact integer kind objects
remain temporary UNPACK pins so an overflowing saved kind cannot overtake a
later node's admission error; all kinds still convert before parameters,
lengths and scale words. Public kernel/storage references remain lifetime pins.

The archive entry packs its existing tuple/list schema from that same capture.
Layout validation retains archive destination admission first. No numerical
kernel, provider call, default, transfer or public archive schema changes.
This source is for the batch after the frozen `d6090df58b` tranche. Source
review and `git diff --check` only; compilation, runtime seams and performance
are pending with the integrator. The frozen 51-role audit is unchanged; the
following-tranche role audit records additions separately. General primitive
label ordering/index metadata and broader fitted-state admission remain open.

The following tranche also reuses the existing parallel `ScoreKeyIndex` for
admitted exact builtin int/bool labels, including arbitrary-size integers.
Class PACK selects the first original object; normalized integer keys do not
replace observable bool/int identities. Exact NumPy string labels reuse the
existing native text plan, including its surrogate path. The original full
admission, NaN and mixed-type refusals precede either plan. Numeric buffer/list
fast routes are unchanged. Exact NumPy numeric and mixed real labels retain
their current scalar comparison path: no new Float64 comparison approximation
is introduced. This is a partial removal of general ordering metadata, not a
claim that all GP label containers or all Python imports are gone. No new
numerical kernel or replacement sort/hash algorithm was added.
