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

Fitted state/admission and class labels still contain Python snapshots and
containers; public coef/intercept/classes/config mutation must remain supported
when those internals become typed. Weighted-fit class encoding and row weighting
still enter their existing compiled Python APIs. Explicit host score still calls
its connected host owner through a compiled Python entry. Those transitive paths
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

Root owes the affected estimator GPU/host, base GPU and classical/tokenizer owner
closures, all dependent groups selected by the development controller, direct
native requests plus public LogisticRegression/host/DeviceRows tests, and later
GPU performance measurement. The tests lane owns independent numerical/state
fixtures and retains Python admission, writable-state, label publication, archive
and provider-lifetime seams. There is no A/B define or performance claim.
