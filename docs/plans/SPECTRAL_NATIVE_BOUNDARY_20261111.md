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

## Alternative assignment descriptor

`core/spectral_assignment_api.mojo` defines a separate one-operation ABI
`mojolearn.spectral.assign.v1`; the nine-operation fit/query ABI is unchanged.
`SpectralAssignmentRequest(source, count, rows, components, method, seed,
restarts, iterations)` is primitive native state. `SpectralAssignmentOutput`
keeps one heap-owned result (`labels: List[Int32]`, `iterations: Int`). Its
host/device adapters call the same `cluster_*_execute(ENTRY_SPECTRAL_ASSIGN)`
with the existing 30/20 settings and `[0.0]` native floating envelope. There
is no duplicate discretization/QR implementation or algorithm switch.

The public fit path removes its Python option lists and nested ClusterOut
serialization/readback. The provider transfers the label list with pop, and
PACK copies it once into the same owning array exporter before wrapping it.
Iteration publication still precedes label packing. Import-time Spectral
capture replaces its unused x_cluster_call entry with the new compiled factory;
other families and direct compatibility callers retain the raw export.

Tests may instantiate `SpectralAssignmentDriver(spectral_assign_native_v1[
SpectralAssignmentHost/Device])` and inspect output.storage[0].labels and
iterations. Preserve original assignment workloads/options; no tests executed
by this lane. Additional roots: `_mojolearn_x_cluster` and host, plus the
family runtime closure. Source audit: `E/spectral-native-owner/assignment-source-audit.json`.
RBF/COO/kNN preparation remains the next staged execution transport.

## Native RBF cell plan

`core/spectral_affinity_plan.mojo::spectral_rbf_affinity(entry, request)` now
owns the existing RBF driver chain. `SpectralRBFRequest` fields are source,
destination, rows, features, gamma and resident. Reuse the existing x_decomp
native descriptor: its image capability preserves the default resident route.
Host/device numerical cells and switches are unchanged. The boundary pins the
actual provider and both input/output Arrays across the complete native call.

Removed: Python squared/scaled/zero scratch Arrays, parameter lists, the handle
list and `_cells_free`, and compiled callable dispatch for each cell. Native
Lists own host scratch; a native operation owner returns the five device buffers
in the same ascending allocation order, also on errors. The plan performs the
same one input upload, one constant upload and one final download. It adds no
numerical loop, fused replacement or copy. All scalar rounding remains inside
the existing descriptor operations.

Pure-native tests can call the plan with the existing x_decomp host/device API
entry or a native recording entry to inspect allocation/error cleanup. Original
RBF numerical workloads remain required; no execution by this lane. Additional
build closure is the existing x_decomp descriptor already imported by the
family runtime, not a new provider export. Source evidence:
`E/spectral-native-owner/rbf-source-audit.json`. COO and precomputed-kNN
preparation, shared public metadata/admission and routed tuples remain staged.

## Additive COO and precomputed-kNN API

`core/spectral_preparation_api.mojo` exposes the independent five-operation ABI
`mojolearn.spectral.preparation.v1`: F32 count0, F64 count1, F32 fill2, F64 fill3,
kNN affinity4. `SpectralCOOCountRequest(source,count,wide)` and
`SpectralCOOFillRequest(source,row_output,column_output,value_output,rows,columns,
capacity,wide)` preserve the two-pass count/fill path and dtype narrowing.
`SpectralKNNRequest(dense,row_indices,column_indices,values,affinity,status,
rows,neighbors,nonzeros,sparse)` carries the original address/options contract.
`SpectralAffinityStatus.words` is native three-word heap storage.

`SpectralPreparationDriver` offers count/fill/knn. Its entry is
`spectral_preparation_native_v1[SpectralPreparationHost/Device]` with the backend
imported from the corresponding `core.spectral_preparation_*` module. Each
backend retains its original F64 helper and exact null/overflow error behavior;
the device kNN uses the original core image's synchronized context slot and
existing kernel. Existing F32 helpers are shared directly, unchanged. This is
an additive API chunk; public callers switch in its following boundary chunk.
No lane compile/test and no algorithm, defaults, work size or copy change.

## Public COO and kNN descriptor switch

Both base images export the new preparation factory. Spectral import-time
capture now keeps that factory instead of five retired raw entries; raw
exports remain for other compatibility consumers. COO count/fill use typed
requests and one retained connection. The precomputed-kNN path uses native
three-word status storage and typed addresses/options. The original two-pass
scan, output allocation, finite/shape checks, return counts and refusal
publication order remain. No numerical implementation changed.

Four authored Python boundary cases (two definitions) replace factory entries
with Python hooks and require refusal without invocation, through the actual
metrics/preparation descriptor seam. They have not run. Source evidence is
`E/spectral-native-owner/preparation-public-source-audit.json`. Build roots add
`_mojolearn` and `_mojolearn_core_host`; the host manifest includes the factory.

Remaining private work is explicit: dense-COO intermediate Arrays also serve
its public tuple API, routed argument tuples, normalized scalar instance
metadata, and shared admission/packing metadata. This tranche does not claim
that all Spectral Python support is gone. Public buffer/class/provider pins
remain necessary and are excluded from removal counts.

## Native dense-COO fit storage

`SpectralCOOStorage(count)` owns named rows/columns Int32 vectors and values
Float32 vector. Private dense fit conversion fills those buffers directly and
keeps their owner through the graph descriptor call. The boundary `_Coo` also
retains admitted sparse exporters when needed. The explicit public triples
operation selects its existing Array allocations directly, so neither route
adds a tensor copy. The numerical count/fill helpers and finite/minimum
reductions are unchanged. F32 dense conversion retains the square check then
its former empty rows-vector refusal via the existing shared refusal helper.

Three private buffer roles move; public returned Arrays and sparse input pins
are excluded. Source evidence: `E/spectral-native-owner/coo-storage-source-audit.json`.
The ABI/constructors already frozen remain unchanged; native tests may use the
new `SpectralCOOStorage` as the production output owner. No lane execution.
Remaining work is boundary/lifecycle metadata and internal routing tuples,
not numerical Python execution in these fitted descriptor plans.

## Typed internal route/output pin envelopes

`SpectralRoutePins` holds the four import-time provider pins in a Mojo struct;
fit/embedding no longer construct or index an internal routed Python tuple.
Prepared input passes as a separate admitted source parameter. Archive predict
uses a native source index instead of rebuilding its public argument tuple.
`SpectralPredictionPins` retains labels/embedding until the operation packs the
actual public result; plain predict does not create an intermediate Python pair.
The public embedding pair and COO tuple remain unchanged at PACK.

No numerical source or copy changed. Source audit:
`E/spectral-native-owner/route-pins-source-audit.json`. Normalized scalar instance
metadata, the standalone convenience operation's temporary model and shared
admission metadata remain pending; this does not claim all Python imports gone.

## Standalone admission repair

Source audit found that `spectral_embedding(...)` bypassed the estimator's
scalar admission and could reach conversion/class or input shape hooks. It
now reuses `spectral_admit` before the existing validator, and admits the
standalone input after the option/route checks before any shape inspection.
It retains the existing exact sparse-input path instead of imposing the
estimator door's dense-only contract. Invalid Boolean diagnostics obtain the
canonical type name getset, so a custom metaclass name property cannot run.

Seven new hook-refusal cases bring the boundary file to five definitions and
11 expanded cases. All are authored, unrun. This repairs inherited boundary
behavior; it changes no supported numerical option, kernel, route or copy.
Evidence: `E/spectral-native-owner/standalone-admission-source-audit.json`.

## Bounded tranche accounting at ad7259567e

The before/after audit records **18 distinct private roles** for the shared
SpectralClustering/SpectralEmbedding implementation: two options, four requests,
four boundary containers, one native handle collection, and seven scratch
buffers. Every row has both source anchors. This is a bounded minimum of
schemas/roles, not runtime allocations, a percentage, or a struct count.
The 36 new core struct declarations are recorded separately. Shared Cres and
ClusterOut transport roles already counted for variants add zero here.

The 13 native RUN roots have a union of **275 local source files and zero
Python-support imports**. Numerical source directories/helpers compared against
the pre-switch caller snapshot have no changes. These are source findings only:
all target compilation, installed public runtime behavior, performance and
cross-vendor identity remain separate root-owned evidence.

Required Python pins remain for actual public Arrays/options, borrowed/native
input owners, compiled provider images, public tuple results and archive fields.
The audit explicitly leaves the standalone temporary model, normalized scalar
instance metadata consumed by the archive schema, fixed validation/admission
collections, and the unreviewed shared admission/sparse metadata tail pending.
It does not claim the complete family is Python-free.

Complete audit: `E/python-object-typed-spectral-family.json`; native closure and
hashes: `E/spectral-native-owner/native-run-closure.json`. Baseline caller source
764e2ad158, producer ad7259567e. No compile or test was run by this lane.

## Four further boundary collection roles at 56b35b89b8

The unsupported-option value list, replacement-field names, lifecycle scalar
schema and lifecycle buffer schema now use native Lists. Value pins and
one exact Python key per lookup stay at the boundary. Their order and refusal
sequence are unchanged. Four additional before/after rows are recorded in
`E/python-object-typed-spectral-boundary-tail.json`, with zero missing anchors;
the immutable preceding18-role audit is preserved. This yields22 audited
Spectral roles across the two disjoint tranches, subject to the same bounded
minimum and shared-schema exclusions. No kernel/RUN source changed, and no
compile, runtime or performance result is claimed. Remaining temporary-model,
archive scalar and shared metadata work is still explicit.
