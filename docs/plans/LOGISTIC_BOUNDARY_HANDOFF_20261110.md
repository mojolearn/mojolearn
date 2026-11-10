# LogisticRegression native boundary migration

This is an intermediate migration. The family is not yet entirely typed.
No lane compile, runtime test, identity run or timing was performed.

## Typed numerical calls

`core/logistic_native_api.mojo` defines QnFitRequest, QnQueryRequest,
QnLinkRequest and QnNativeDriver. Dense operations use the existing native
descriptor ABI: 0 fit, 1 decision, 2 binary codes, 3 multiclass codes, 4 fused
probabilities/log probabilities, 5 sigmoid, 6 softmax, 7 row logarithm, 8 mode,
9 host capability and 10 multiclass-code capability. The descriptor has one
dispatch pointer; its operation count is 11.

The pure implementations are QnDeviceKernels in glm/logistic_native_device.mojo
and QnHostKernels in glm/host/logistic_native_host.mojo. They call the same
existing QN optimizer, decision, code, link and row kernels. The measured device
probability path still keeps scores on the device. CPU scratch and GPU uploads,
downloads and synchronization retain the original operations. Legacy compiled
QN entry points now decode their external arguments and call these same typed
implementations; their public raw signatures remain available to other families.

Resident folds use QnFoldStore, QnFoldDriver and QnFoldResult, with a separate
descriptor in glm/logistic_fold_native.mojo. Fit and query still use the actual
model-selection store/context and existing qn_fit_fold/qn_infer_fold kernels.
Finite/range status travels in native Int32 words and a typed result, without
Python ID/parameter/result lists. Existing legacy fold exports remain adapters.

`bindings/logistic_descriptor_boundary.mojo` owns exact descriptor admission and
extension lifetime pins. Its native records cache mode/capabilities. RUN receives
typed values with the GIL released; the boundary holds the provider and every
input/output owner through return. The old lr_entries_admitted Python dictionary
and per-call compiled-Python QN dispatch are removed from LogisticRegression.
Explicit host inference uses the same native descriptor and retains its original
host code/decision/link algorithms and warning sequence.

The import-time connections retain the required descriptor factory. Removed:
LogisticRegression's legacy per-entry connection table, QN Python parameter
lists, fold lists and boxed iteration results, duplicate numerical bodies in
legacy QN adapters, and unreachable missing-fused-export fallbacks. Fused QN
probabilities are exported unconditionally by both current estimator bindings;
the native multiclass capability preserves the existing QN_DEV_ARGMAX switch
and CPU decision/argmax route. No tuned kernel or experiment was removed.

## Outstanding work

Regularization/tolerance preparation now uses `core.logistic_config` typed
scalar operations. Builtin, NumPy half/single/double, integer and bool promotion
remain distinct. The shared native promotion metadata reader replaces the
corpus-specific copy; it is consulted once only when half/single parameters
make the promotion policy relevant. Native arithmetic produces ordered error
events consumed by the existing native NumPy error-state/warning boundary;
no NumPy numeric slot, Python arithmetic, builtin conversion lookup or Unicode
mode fallback remains in this configuration path. Scalar conversions and
public `_l1_ratio` publication remain boundary operations. Constructor refusal
order and the later strength/tolerance conversion order are retained.

The new native scalar/status plan and public scalar/error-state seams still
owe compilation/runtime qualification. Reference sources retained in the lane
evidence directory are NumPy 2.2 `scalarmath.c.src` and `arraytypes.c.src`;
promotion follows [NEP 50](https://numpy.org/neps/nep-0050-scalar-promotion.html)
and the [2.2 promotion-state removal](https://numpy.org/doc/2.3/release/2.2.0-notes.html#nep-50-promotion-state-option-removed).

Fitted-query width/scalar validation and shared class labels still need native
plan completion; public coef/intercept/classes/config mutation remains supported.
Weighted-fit class encoding, class-weight mapping and scalar preparation still
need native-container completion. Score's shared label/union/code containers
also remain outstanding. Those transitive paths
are not claimed complete. Metadata/get_params/set_params, ownership and admitted
buffer preparation must also receive the final end-to-end review.

The fitted coefficient output layout is now typed in
`core.logistic_model_state`: target count, checked word counts and QN-column to
public-row mapping. `bindings.logistic_state_boundary` retains the input/output
owners and uses the existing native allocator with native shape lists. The
old nested coefficient-copy loops in the public door are removed. Independent
raw-word copies use the existing parallel host work policy and one initial
coordinate calculation per worker; kernels, downloads, output allocation count,
public writable coefficient arrays and private `_w` semantics are unchanged.
The fresh `_w` stays private through GIL-released PACK; publication happens
after the join, with the previous failed-allocation `_w` state retained on the
error path. This output-layout step does not complete fitted-query snapshots.

The public query door no longer copies a Python dictionary of model fields.
`LogisticQueryOwners` is an explicit UNPACK/PACK owner record containing only
the five actual public field references consumed by a query, plus native
presence flags. Its `finally` pin covers the entire query, including all GIL
releases and failures. These are actual public values retained for rebinding
safety, not Python-owned private algorithm state; the numerical interface is
still typed QN requests. The no-longer-used `logistic_fields(copy_state)` branch
and the score method's temporary `(X, y)` Python operation envelope are removed.
Class labels and width comparisons remain explicitly listed above as unfinished
transitive work; retaining owner pins is not proof that those decisions migrated.

The explicit host binary prediction now gives the unchanged QN binary-code
operation the final owned native `List[Int]` consumed by label decoding. The
temporary Python int64 Array and serial per-row append/copy are removed. The
LP64 descriptor word width, class-code values, decoder, source/weight/provider
pins and all numerical kernels remain unchanged. This removes one output
allocation/copy without introducing a new output path or a borrowed list.

Weighted Logistic calls now use the fixed `mojolearn.label_weights.v1`
descriptor for the existing xtrees code-count, class-row gather and float32
product operations. The exact host/device kernel choice remains fixed by
`TARGET_COLUMN`; no kernel, count algorithm, gather, product, allocation or
data copy changed. Requests use the shared fixed-span descriptor helper, and
the boundary pins the actual compiled provider through GIL-released RUN.
Logistic's old three function-name connections and Python parameter-list calls
are removed. The shared class-weight helper selects this path at compile time
for Logistic; its old public entries remain because tree/SVC and other
unmigrated classifier callers actually still use them. The shared mapping and
class-label Python containers are separate outstanding work, not claimed native
by this numerical dispatch change.

This batch additionally affects both xtrees groups and the transitive shared
label-owner closures. Root owes direct descriptor arity/span/error propagation
fixtures plus balanced/dictionary/sample-weight public Logistic cases. Parent
dependency `22d5da1c5f` is the lane cherry-pick of fixed-span batch `63f40d36ca`.

The new GIL-released calls exposed the xtrees context's old unguarded lazy
`Optional` initialization. `_ctx` now reuses `core.neural_context.process_ctx`
with an explicit root-zero plan, preserving the same device and process-lifetime
context. The shared owner provides locked first construction and atomic
publication, so later calls do not take a lock. The private `_OpsContext` and
its duplicate initializer are deleted; the slot names advance to V4 because
the shared owner's layout replaces the old one. All transitive xtrees context
consumers belong to this build closure. No device-plan reading, kernel change
or additional per-operation data allocation was introduced. Concurrent first
use of the actual descriptor remains required runtime coverage.

Explicit host score now reuses the host Logistic admission/inference function
directly. `host_inference_state` contains the existing watched class-kind and
layout admission; `host_logistic_query_boundary` contains the former Logistic
branch's exact field/class/fitted/mode admission order. Public host inference
and Logistic score share both helpers. The old `host_predict` callable/tuple
connection and score's compiled-Python-entry calls through the classical/tree
doors are removed. Caller-frame/output-format rules, mutable public fields,
warnings, input preparation and QN kernels remain unchanged. The shared
accuracy implementation is still being migrated separately; this does not
claim its class/union containers are native. Root owes public host Logistic
score/refusal/output-format tests and regression coverage for the other host
families that reuse the extracted class-kind admission.

Query feature equality now uses `core.logistic_query_state.LogisticQueryWidth`.
Ordinary fitted integers use a direct integer comparison; primitive scalar
metadata uses the existing exact native scalar decoder. The plan preserves
the separate Int-conversion contract of binary prediction and explicit host
inference. Exact native Array fields retain exact representability; NumPy
primitive fields retain weak floating conversion when that policy is active.
Single-element arrays are unpacked without a comparison result array. Object
array metadata uses bounded exact-type unboxing before any scalar conversion;
unknown equality hooks are refused. The original feature-mismatch messages
remain in the pure native checks.

`native_scalar_buffer_bytes` extracts the pre-existing fixed 32-byte public
buffer UNPACK for reuse by numeric and temporal metadata. No scalar decoder
algorithm changes. `native_torch_scalar` retains native TensorBase item/numel
descriptors in the existing admission cache (slot V2) and applies the same
thread-local mode refusals before scalar extraction. It avoids the former
comparison tensor allocation, with any existing device-scalar synchronization
confined to UNPACK. The query plan receives only native values. All original
model/QN/fused kernels and their data buffers are unchanged.

Removed: the three public-field Python inequality operations and the duplicate
binary/host width-error branches; no legacy comparison fallback remains.
Root owes native query-plan fixtures and public rebound-field coverage:
NumPy legacy/weak scalar and single-element array promotion, empty-array
warning/error versions, complex and temporal fields, native Array values,
object-array callback refusal, and Torch primitive scalar dtype/ambiguity/mode
behavior. Exotic dtype refusal and older NumPy weak-integer promotion remain
specific compatibility qualification points; source inspection is not runtime
evidence. There is no timing or identity result for this batch.

The fit's two-word solver info and one-word finite status, and inference's
one-word finite status, now have native owned List storage. Their addresses
and initial zeros are unchanged; owners live through the synchronous QN
operation and the result/refusal reads. They no longer allocate private
Python Array wrappers. Public outputs still use the same zero-filled native
Array allocator and writable owners, now through its existing typed-shape
entry instead of constructing and immediately unpacking Python shape tuples.
Removed the superseded scratch address lookups, the unused float pointer
alias/weight-presence local, and the obsolete probability fallback docstring.
No row traversal, reduction, upload or tuned kernel changed. The shared label
preparation/weight buffers remain part of the separately recorded label
boundary dependency; this fixed-size scratch cleanup does not claim those
row buffers or score class containers have been migrated.

Root owes the affected estimator GPU/host, base GPU and classical/tokenizer owner
closures, all dependent groups selected by the development controller, direct
native requests plus public LogisticRegression/host/DeviceRows tests, and later
GPU performance measurement. The tests lane owns independent numerical/state
fixtures and retains Python admission, writable-state, label publication, archive
and provider-lifetime seams. There is no A/B define or performance claim.
