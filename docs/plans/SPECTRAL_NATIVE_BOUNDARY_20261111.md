# Spectral native boundary migration

The additive `core/spectral_native_api.mojo` defines typed input, fit/embedding
options, prediction shape, retained-state spans and five synchronous requests.
`SpectralDriver` uses the existing fixed native descriptor ABI; the nine
operations cover mode/capabilities, fit, embedding, query, resident prepare/query/
release and resident count. Dense and resident capabilities remain distinct.
The image table in `spectral_native_provider.mojo` is generic over the actual
provider. No public method is switched by this additive chunk, so no private
Python roles are claimed removed yet.

The current public routes were traced through `cfb_spectral.mojo`: clustering
has dense, COO graph and nearest-neighbor dataset fits; embedding has the same
three routes with its existing drop-first/laplacian/tolerance defaults. RBF
uses the existing x_decomp squared-distance, scale and exp cells, including
resident cells and their exact upload/download sequence. Precomputed kNN uses
the existing dense/sparse affinity entry. Prediction retains the current
Nyström state, freezability/readonly switch, first-use upload and generation/
object/shape matching. Discretize and cluster-QR retain the existing x_cluster
entry and settings. These routes are not interchangeable algorithms.

The native spectral resident registry already takes `ArcPointer` owners under
metadata locks and uses the established synchronized metrics context. No
registry repair has been identified or introduced. The shared boundary table
is the integrator's V5 variants snapshot; the adoption commit is not a new
spectral producer. Provider image pins and public mutable output Arrays must
remain at UNPACK/PACK. The next chunks will connect the existing numerical
bodies and migrate their argument/state envelopes without changing kernels,
copy counts, defaults, refusal or publication order.

Source-only status: ABI/provider table authored, no numerical bodies or public
bindings changed, no lane compilation/tests/benchmarks/identity. The integrator
will compile the coherent batch. Native tests retain the original numerical
workloads; the test lane owns `spectral_contract_*`, `spectral_state_*`,
`spectral_native_*` and `spectral_dense_*` fixtures. Full pre-edit source and
route inventory are retained under
`~/mojolearn-evidence/python-mojo-migration-20261110/spectral-native-owner/`.

The additive backend chunk connects `SpectralHostKernels` and
`SpectralDeviceKernels` through shared native marshalling/result owners in
`spectral_native_execution.mojo`. It calls the existing dataset/COO/dense
spectral functions, and the existing Nyström/resident functions. The ordinary
non-keep fit wrappers already delegate to the same `_keep` functions with
`False`; the adapter passes that same flag and does not replace their body.
Native result owners permit retained legacy wrappers to share execution while
keeping their UNPACK/PACK stages. Public exports and callers are not switched
by this chunk.

The source mapping retains all input copies, Float32 tolerance rounding,
output guards and output/state copies. It also retains the existing provider
differences: CPU has no dense/resident capability, GPU clamps negative dataset
copy counts before the existing numerical refusal, and the device query keeps
its output-count guard. Host and device graph/query counts retain their prior
clamping. These are compatibility plumbing facts, not performance claims.
Dense negative refusal codes still return before any output write. The native
resident prepare/query/release entries retain the original metrics context and
locked owning registry. No `spectral/` numerical source was edited.

API consumers can use `spectral_native_v1[SpectralHostKernels]` or the device
counterpart with `SpectralDriver` for native fixtures. Original graph/dataset/
state workloads and tolerances remain the test lane's oracle input. Compile,
execution, public boundary replacement and final role accounting remain owed.

## Public fit and embedding descriptor switch

`bindings/cfb_spectral.mojo::_sc_fit` and `_se_fit` now pass typed options,
input/state views and output addresses to the production descriptor. The
former private parameter tails, per-route Python parameter/address lists,
state-address builder and private four-Array tuple are removed. Four explicit
boundary pins retain the same observable prediction-state Arrays. Dense
availability comes from the provider capability, so the retired raw entry's
presence no longer controls this route. Provider/image pins survive GIL-free
execution; the host provider continues its dataset/COO route.

The output Array allocations, fitted borrowed-input copy, four-state copy
sequence, dense refusal decoding, shape/finite/neighbor validation and final
publication order remain. The numerical provider retains the existing
Float32 default embedding tolerance. Import-time capture and the host manifest
now include `spectral_native_api_v1` from the real metrics image.

Affected roots: `_mojolearn_metrics`, `_mojolearn_metrics_host`, and the family
runtime/dispatcher roots importing `cfb_spectral`; compile the existing metrics
and family runtime groups with their normal dependency closure. No new build
scheme. No lane compilation or execution; public seam/native correctness,
performance and cross-vendor identity are not claimed by this source audit.

Still staged: RBF/COO/kNN preparation, alternative assignment, query and resident
cache transport, validated private scalar fields, routed argument tuples and
shared admission/lifecycle metadata. Raw boxed exports still have direct
compatibility/test consumers and remain; their public fit callers have moved.
The family is not yet fully migrated. Shared resident transport already counted
for KMeans variants will not be counted again when Spectral reuses it.

## Native query and resident ownership

`_sc_predict` now constructs `SpectralQueryShape` and typed direct/resident
requests. `_sc_resident` retains a `SpectralResidentLease`, whose native Arc
owns `SpectralResidentState(driver, handle)`. The five public fitted buffers
and actual descriptor provider remain explicit boundary pins. Query eviction
cannot retire an active Arc; final release occurs after metadata removal,
outside the Dict operation. Ordinary fit-state freezing, reuse, resident upload
and query-copy routes are unchanged. No kernel or registry numerical body changed.

The shared resident table is V6 with a Spectral branch beside the unchanged
V5 variants branch. Limits, LRU accounting and generation/dimension/object
matching remain; provider identity additionally prevents a replacement factory
from reusing a different minting driver's handle. The new storage is internal
native state, never a Python model-field owner, so no canonical field-release
registration is needed. Native tests can own an Arc of `SpectralResidentState`
and exercise release through a mock native descriptor; actual provider fixtures
use the frozen SpectralDriver prepare/query_resident/release interface.

Removed from Spectral execution: Python query dimension/address lists, fitted
object identity tuple, `(query_callable, handle)` envelope, compiled-callable
resident prepare/query/release dispatch. Public `(labels, embedding)` is still
the documented output of the embedding prediction seam. Legacy table routines
remain only because this lane's adopted older `cluster_native` still consumes
them; the parent already has c47be7762d removing those consumers. After both
changes collect, delete the now-unreferenced legacy table APIs and callable
fields in that combined source. No aggregate role claim for shared table schema.

Source evidence: `E/spectral-native-owner/public-query-source-audit.json` records
five anchors and nine lifetime invariants; neither is a runtime verdict.
Affected roots additionally include every current `cluster_resident_table`
consumer through the existing runtime family build closure. Native ownership,
public mutation/cache/lifetime tests and all target compiles remain root-owned.
Affinity preparation and alternative label assignment remain staged next.
