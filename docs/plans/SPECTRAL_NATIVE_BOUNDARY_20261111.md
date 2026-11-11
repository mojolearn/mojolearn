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
