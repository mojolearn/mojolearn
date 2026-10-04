# Catalog G1/G5 unscored probe, 2026-10-04

Base fc36f878c8c80381e38d3c2ea05c6739b93c74c1; catalog source pinned
9ab2d3d3fb770498ef025db08f595a0149792bb7. experiments/apple_fast/gemm/mma.mojo
and package init files are copied unchanged from that source. Original catalog
worktree untouched. Source checkpoint c205e2d8e; no production dispatch hook.

Binding gemm_probe, define MOJOLEARN_APPLE_GEMM_PROBE. A build omits the define
and exposes only a disabled callable; B enables three explicit runtime arms:
0 = incumbent core.gemm NT (SDK matmul NN companion), 1 = G1 direct64/BK16,
5 = G5 single shared64/BK16. Thus B contains all three controls; the runtime
arm, ABI version, completed-call counters and SHA are recorded. This is not
an A-vs-B quality comparison of disabled/enabled binaries. No silent dispatch
fallback in G1/G5. IDENTICAL/non-Apple compile the callable refusal; no
IDENTICAL timing authorized. One compile instantiates NN/NT for both new arms.

The build script follows repository platform/link flags and compile lock but
never executes an import or GPU build gate. Its temporary files live under
build/probe-tmp. Manager may use existing compile_arms/intake A empty, B
MOJOLEARN_APPLE_GEMM_PROBE with binding gemm_probe. M2 compilation has not been
performed by this agent; private SDK _mma_apple_8x8 ABI is still unverified.

After staging verified arms, run on serial M3 queue:
MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_VENDOR=apple <python>
tools/catalog_gemm_quality.py SOURCE w2-catalog-g1g5-q-20261004

The helper loads verified B.so directly, verifies exact source, manifest,
binary SHA and ABI, and never assumes or overwrites an installed custom .so.
No timers exist. Twelve quality fixtures cover NN/NT square, ragged/tail,
small, GEMV-shaped incumbent, true device-aliased Gram, cancellation, dynamic
range and K=0. Each arm runs once per fixture; poisoned output catches holes.
G1/G5 must have exactly equal output bytes, finite outputs and scaled FP64
oracle error <=5e-6; both must also have error <=incumbent with zero added
tolerance. Bounds fixed before any run. Every call increments only its chosen
completed-arm counter. Full arrays and raw metrics retained; exact PASS.json
only if every shape meets the criteria. K0 incumbent is explicitly a parallel GPU zero-fill contract, not an SDK
zero-extent matmul; G1/G5 still execute their own zero-K kernels. Zero-length
inputs are never uploaded or dereferenced. Private ABI failures are
infrastructure evidence to repair under a new tag, not a pass.

G1/G5 differ from current decomposition K16 and LU dbuf: the catalog uses
SDK SIMD2 fragments, scalar lane loads, no prefetch, and single shared page
(two barriers/BK); existing kernels use AIR v64 load/MMA, padding/prefetch,
and K16/dbuf use two pages. The G1/G5 control pair tests whether shared staging
helps this independent implementation. It does not prove its advantage over
existing specialized kernels; incumbent control remains essential.

No scored timing is implemented or queued here. If quality passes, choose a
small fixed representative shape set and record one scored call per arm with
first read; separately instrument resident completion to distinguish staging
from allocation/upload. Do not score all ten variants or whole board by default.
Remaining eight variants stay source-only backlog.

Reachability: catalog core NT hook would not reach LU's lfm updates or
x_decomp's generic launcher. PCA covariance_eigh uses its own AFN Gram. Current
FAST Cholesky dominant trailing blocks (>2048) use direct SDK matmul with the
lower-triangle subtraction epilogue, while <=2048 tails call core NT. The old
catalog README's sabotage-only Cholesky description is stale for current main.
A high-leverage Cholesky adaptation must preserve that fused epilogue rather
than turn on sabotage or route the whole factor through this standalone probe.

## Compile-only repair r1

M2 source4d5f42f1c failed parsing before arm A built: `alias` is a reserved
Mojo keyword. New branch lane/apple-fast-catalog-gemm-probe-r1 renames the
local to aliased_inputs. A proactive API-spelling audit also replaces the
catalog's Array fragment containers with the repository-standard InlineArray
and explicit zero initialization; all slots are overwritten before use, so
fragment arithmetic/order is unchanged. This is the only copied-kernel source
change; the original pinned catalog/source4d5 remains preserved. No scored
output existed, and no threshold, fixture or runtime dispatch changed.
Recompile on M2; private intrinsic ABI and actual arithmetic remain unverified.
