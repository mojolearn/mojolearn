# TruncatedSVD native requests and resident ownership

Selective dependencies: `7928f536d0` adds the native API and providers; the
boundary follow-up implements the public call path. The separate shared-context
repair `4c76f65dd5` belongs to the integration concurrency batch. Preserve the
orchestrator's subsequent allocation-free root-zero context adjustment and
resident/device-pool repairs when combining these commits. Do not cherry-pick
the lane's earlier base merge.

The public TSVD owner now passes typed `TSVDFitRequest` / `TSVDQueryRequest`
values to the native descriptor. Fit status is `TSVDFitResult`; finite/range
words, dimensions, options, route, fold handles and lease IDs remain Mojo
values. Fit-transform retains its `AdmittedX` with native dimensions/address
through projection, so the converted input is neither converted again nor
re-read through Python shape objects. The fixed descriptor ABI has eight
operations: mode, vendor, capabilities, qfix, fit, query, open and close.

| Operation | Existing execution retained |
| --- | --- |
| Exact Gram/fused fit | Estimator image, existing fused one-upload fit preferred |
| Randomized fit | Existing `tsvd_randomized_fit_run`, original x_decomp executor/image |
| TSQR fit | Existing `TSVD_QFIX` gate and dimensions, then estimator explained variance |
| Resident fold fit | Base image and `msel_fold_entry_context`, original device gather |
| Transform/inverse | Existing plain/resident estimator kernels and finite checks |
| CPU estimator fit/query | Existing host kernels, workers and inverse component copy |

No numerical algorithm, tuned kernel, experiment default, transfer count or
supported public method was changed. TSQR's optional-driver refusal and the
Gram/explained compatibility route remain. Fit-only folds remain fit-only;
fit-transform keeps the ordinary converted input/projection route. Explicit
host fits use the existing import-connected host estimator image directly,
without calling Python `_HostBound._bind`. Their randomized decomposition
provider remains the previously selected decomposition image; this patch does
not silently change that inherited route.

Publication order remains explicit: dropping the previous resident owner comes
before tolerance/algorithm refusal; `input_copied_` appears before minimum-size
or component-count refusal; exact variance outputs publish before component
allocation; Gram component/singular outputs publish before native fit; TSQR
publishes them after successful TSQR and before explained variance; randomized
outputs publish only after success. Fitted dimensions publish only after fit
success. Query output allocation and component admission keep their prior order.

Private resident state is now `TSVDLeaseState`: typed connection identity,
generation, component identity, lease and driver. `TSVDResidentOwner` carries
that state with separate provider/storage lifetime pins and closes the native
lease once. The weak model registry retains the owner without retaining the
model. The sole type registrar is `_mojolearn_runtime_decomp_host`, using
lookup/add/attach before the allocating public entry is exported. Queries pin
the lease owner through GIL-free execution. Refit, rebinding, non-owning storage,
conversion copies and non-weak-referenceable models keep their existing lease
invalidation/fallback behavior. Public writable arrays still use the existing
per-call route; immutable owned components still reuse the resident copy.

Scalar admission now precedes the first numeric conversion or invalid-algorithm
repr: unsupported conversion, repr and `__class__` hooks do not execute.
Existing exact builtin/NumPy scalar conversions remain, including admitted
numeric zero-dimensional arrays; random seeds retain the builtin-int-only
contract and native counter-based modulo-2^32 seed semantics. The exact host
mode refusal stays at the former binding-selection stage.

Removed: per-call Python parameter/result lists between compiled entries,
compiled callable dictionaries and vendor/mode getter dispatch, the prepared
resident five-tuple/boxed lease/mirror list, and Python host `_bind` dispatch.
Legacy exported functions remain for direct callers; the public TSVD owner no
longer calls them. No fallback or source guard was added.

Staging still present: Python import-time factory connection, public lifecycle
and dictionary publication, exact scalar/buffer admission and error formatting,
output shape/Array packing, and provider/buffer pins. These are boundary work;
this is not a claim that the entire family or package has no Python imports.
The six native API/provider/state roots have a 201-file transitive source
closure with zero Python/binding imports. Eight directly reached numerical and
route files are byte-identical to frozen integration `e53128fe7f`. The shared
`x_decomp/device.mojo` context/stage repair is separate plumbing work.

Compile roots for the integration batch:

- Identical `core`: `_mojolearn`, through `msel_device` fold descriptor.
- Identical `estimators`: `_mojolearn_estimators`.
- Identical `x_decomp`: `_mojolearn_x_decomp`.
- Host `estimators`: `_mojolearn_estimators_host`.
- Host `x_decomp`: `_mojolearn_x_decomp_host`.
- Host `runtime_decomp`: `_mojolearn_runtime_decomp_host` and owner registration.

Validation authored: 11 Python boundary definitions / 40 expanded cases in
`test_tsvd_native_boundary.py`; AST parsing only. The native-test lane separately
owns TSVD contract/execution fixtures and the direct numerical test migration.
Source diff checks and a saved closure/byte audit completed; no compilation,
installed runtime tests, benchmarks, boxes or identity runs were performed by
this lane. Native and boundary execution remain owed to the orchestrator's
combined build. There is no A/B define or intended numerical bit change; speed
and device identity are not claimed.

Evidence: `~/mojolearn-evidence/python-mojo-migration-20261110/tsvd-native-owner/`.
