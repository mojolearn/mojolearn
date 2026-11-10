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

Regularization/tolerance preparation still contains Python scalar arithmetic.
Fitted state/admission and class labels still contain Python snapshots and
containers; public coef/intercept/classes/config mutation must remain supported
when those internals become typed. Weighted-fit class encoding and row weighting
still enter their existing compiled Python APIs. Explicit host score still calls
its connected host owner through a compiled Python entry. Those transitive paths
are not claimed complete. Metadata/get_params/set_params, ownership and admitted
buffer preparation must also receive the final end-to-end review.

Root owes the affected estimator GPU/host, base GPU and classical/tokenizer owner
closures, all dependent groups selected by the development controller, direct
native requests plus public LogisticRegression/host/DeviceRows tests, and later
GPU performance measurement. The tests lane owns independent numerical/state
fixtures and retains Python admission, writable-state, label publication, archive
and provider-lifetime seams. There is no A/B define or performance claim.
