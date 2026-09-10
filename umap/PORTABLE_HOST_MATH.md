# IDENTICAL UMAP host math (2026-09-10)

IDENTICAL now routes host binary64 transcendental calls through repository
portable seams. Other numeric modes retain their previous `std.math` calls
through compile-time wrappers; no FAST or DETERMINISTIC tests were run here.

| UMAP module | Replaced calls |
|---|---|
| `curve.mojo` | exp, log, pow |
| `graph.mojo` | exp, log2 |
| `sparse_graph.mojo` | exp, log2 |
| `transform.mojo` | exp, log2, pow |
| `optimizer.mojo` | pow in the serial host optimizer |
| `sparse_optimizer.mojo` | pow in the serial host optimizer |

`identical_log2_64` wraps the existing portable log2 implementation.
`identical_pow64` wraps the new `portable_pow64` only for IDENTICAL.
General finite powers use `portable_exp64(p * portable_log64(abs(x)))`.
This is a specified approximation, not correctly rounded pow and not a
claimed small-ULP replacement for libm. The numerical gate reports both ULP
and relative error instead of hiding the conditioning behind one metric.

Pow's real-valued special cases are explicit: canonical NaNs, x**0 and 1**p,
infinite exponents, signed zeros, infinities, and negative finite bases with
integer exponents. Integer parity is derived from exponent/significand bits,
so huge finite exponents never undergo an overflowing integer conversion.
Identity, reciprocal, square, and integer powers of binary powers have direct
IEEE arithmetic/bit constructions. Subnormal inputs are retained. Subnormal
outputs use a local scaled-exp reconstruction: exp(t+512*ln2)*2**-512,
with split ln2 constants. Existing `portable_exp64` behavior is unchanged,
including its separate output-flush policy for all other callers.

The native primitive gate covers 262,144 deterministic hashed pairs across
UMAP-sized bases/exponents, the complete positive binary64 exponent range,
near-one bases with huge signed exponents, and negative bases with integer
exponents. It adds 234 boundary pairs and exact special-value word checks.
Admission is relative error <=2e-12 with an absolute allowance of two smallest
binary64 subnormals, against host libm `pow` through FFI. This is a tested
approximation bound, not a proof over all binary64 pairs. Input/output FNV
hashes record portable arithmetic independently of the comparator. Agreement
across hosts requires matching captures from separately executed hosts.

`umap/checks/portable_host_math_check.mojo` additionally checks the existing
independent default/custom curve references and scalar optimizer reference,
exact dense/CSR graph and serial layout equivalence, repeated transforms,
frozen training coordinates, refusal behavior, and full stage words/hashes.
`tools/check_umap_portable_host_math.sh OUT` runs those gates and the existing
small/broader UMAP identity and transform fixtures, all in IDENTICAL mode.
Root owns local build execution; cross-host evidence is recorded separately.
A stronger portable seam can change older platform-dependent stage words;
old/new equality must be measured rather than assumed or enforced by changing
numerical tolerances.

The audit found no remaining raw pow/exp/log/log2 calls inside the six UMAP
modules above. The device Jacobi optimizer already calls FP32 `identical_pow`
and remains unchanged; its arithmetic/order is intentionally different from
the serial host optimizer. Spectral initialization reaches the Float32
`symmetric_eig_host` instantiation in Lanczos, whose sqrt uses
`identical_sqrt`; the generic Float64 sqrt branch is not instantiated by this
UMAP route. This work does not add portable sqrt64, trig64, inverse trig64,
or generic FP32 reduction primitives and makes no closure claim for those
independent gaps. No tree code or shared existing primitive arithmetic changed.
