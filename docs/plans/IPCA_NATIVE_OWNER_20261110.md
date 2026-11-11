# IncrementalPCA native requests and resident owners

IncrementalPCA's public fit, partial_fit, fit_transform, transform and
inverse_transform now pass typed Mojo requests through one fixed descriptor to
the existing host/device owners. The six output buffers, dimensions, options,
count/noise result and five resident handles no longer travel through Python
argument lists or result tuples. No numerical body, SVD route, batch rule,
transfer, copy or default changed.

## Native API and ownership

`core/ipca_native_api.mojo` contains `IPCAOutputs`, `IPCAPrevious`,
`IPCAStateIds`, the three request types, `IPCAFitResult` and `IPCANativeDriver`.
`core/ipca_native_provider.mojo` specializes `IPCAProviderKernels` and
`IPCAProviderAPI` as `[HostExec, NoRDev, False, NoIpcaDev]` or
`[DevExec, DevRDev, True, DevIpcaDev]`. Fit and partial results with `kept=True`
transfer five pool handles to their caller. Each must be released or adopted
once. The descriptor's prepare/free operations use that same provider's pool.

`core/ipca_native_state.mojo` owns the six zero-filled scratch rooms and typed
resident identity, shape, generation and take-once release state. Five opaque
registered `IPCAResidentOwner` values replace the former dictionary of nine-word
Python tuples and Python callback capsules. Each owner's only Python fields pin
its admitted source array and minting provider. The runtime-decomp host module
registers this type once, before exposing the public operation.

A fit drops prior resident owners before input admission. New handles are
adopted only after scalar and six-array publication in the old order. Pending
adoptions receive their generation on the first query. A partial update matches
source/provider/read-only storage without requiring the generation that its
own lifecycle has already changed. Query operands additionally match generation
and dimensions. Rebound, writable, borrowed and converted arrays retain the old
per-call path; only exact owned Arrays are frozen and cached. Old resident pins
outlive the new state's publication and adoption, and each released-GIL call
copies its native driver and pins the provider image.

The original native `ipca_fit_run`, `ipca_partial_step`, `ipca_transform_run` and
`ipca_inverse_run` still implement every operation. Their existing Gram/SVD,
resident eigensolver, batching, previous-state and whitening routes are intact.
IPCA does not substitute PCA's different query/whitening kernels. The six final
host copies and the same kept state transfers remain.

## Boundary retained and removed

The boundary still admits public objects, converts supported exact scalar
values with the existing builtin int/truth operations, pins input/output
storage, selects an import-connected descriptor and publishes Python-visible
attributes. Shared `decomp_scalar_admit` preserves exact numeric and text 0-D
arrays while rejecting object arrays and conversion hooks before invocation.
Those staged public metadata operations and provider/storage pins remain Python
objects; this is not a claim that every file in the family is Python-free.

Removed from the public path: `_entries`, the seven-function Python dispatch
map, `_room_addresses`, Python argument/result envelopes, Python output
shape/size lists and tuples, state-ID boxing, and IPCA's import/use of the broad
`decomp_kit_native` resident tuple/callback mechanism. Legacy direct-call exports
remain for their supported callers, but the public IPCA owner no longer reaches
them. No measured kernel, switch or algorithm was deleted.

## Evidence and queued work

The source audit under
`~/mojolearn-evidence/python-mojo-migration-20261110/ipca-native-owner/`
records 158 native transitive source files, including concrete host/device
executors, with zero Python/binding imports. Seven numerical/executor files are
byte-identical to root's `0eb10b0aeb`. This is source evidence, not execution.

`python/mojolearn/tests/test_ipca_native_boundary.py` has 11 definitions / 57
expanded cases for removed legacy dispatch, descriptor/scalar hook refusal,
exact native text scalar support, refusal/publication order, partial batching
and seen count, writable/rebound public arrays, and weak model lifetime. AST
parsing passed; these cases were not run. The native test lane owns standalone
`ipca_contract_*` / `ipca_native_*` correctness and state fixtures.

Root must compile the affected GPU `_mojolearn_x_decomp`, host
`_mojolearn_x_decomp_host` and `_mojolearn_runtime_decomp_host` groups, including
merged initialization ordering. No compile, installed runtime, measurement or
cross-vendor identity result is claimed here. A/B define: none. Numerical source is unchanged;
output identity remains untested. No box job was queued by this lane.

## Namespace admission follow-up

The shared lifecycle key-admission repair is reused after GIL-releasing input,
fit and resident preparation steps, before later public-field probes or writes.
Unchanged exact-key dictionaries use the existing native mutation stamp.
Adoption now reads/admit-checks the source before transferring its result handle,
so a refused or missing field cannot strand that handle. Numerical bodies and
copy/transfer routes are unchanged. Runtime checks remain owed to root.
