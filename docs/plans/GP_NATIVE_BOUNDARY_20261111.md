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
