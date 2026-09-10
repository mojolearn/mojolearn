# Portable primitive inventory, 2026-09-10

This is a read-only source inventory at base `1b3f53d6` plus UMAP seam change
`163aaf81` and primitive-gate revision `20061ad5`; it does not certify
execution paths merely by counting names.
No tree implementation or tests were changed. The companion JSON retains
file/line/source for each detected call. Detection strips Mojo docstrings,
comments and string literals, includes named `external_call` targets, and
excludes function declarations. Library helpers under `checks/` are identified
explicitly rather than assuming that directory contains only test code.
Counts are lexical call sites, not dynamic invocation counts or an exhaustive
compiler call graph. Reproduce with `python3 tools/audit_portable_primitive_calls.py
--output /tmp/primitive-inventory.json` from the recorded source.

| Standalone primitive | Executable calls | Other mentions | Interpretation |
|---|---:|---:|---|
| erfc | 0 | 10 | All mentions occur in `checks/portable_gelu_check.mojo` comments, docstrings or diagnostic strings. The complement polynomial inside erf is exercised; there is no standalone erfc call to route. |
| expm1 | 0 | 2 | Both mentions are one proposed expression in the Jones-transform design docstring. |
| log10, asin, acos, atan, atan2 | 0 each | 0 each | No matching Mojo consumer found. Missing standalone functions alone do not establish a product defect. |
| cbrt, sinh, cosh, hypot, tgamma | 0 each | 0 each | Same scope: no matching Mojo consumer found. |
| atanh(Float64) | 2 | 19 total atanh textual occurrences across precisions | Both Float64 calls are ARIMA reference/oracle code. The additional Float32 production call is behind an explicit non-IDENTICAL branch; IDENTICAL Jones uses identical_log. No standalone portable atanh64 added. |
| log2(Float32) | 1 | Counted by explicit argument type, not generic name | A real tree histogram-offset call remains; see the scope note below. No standalone portable log2f added. |
| lgamma | 4 | 16 total textual occurrences | Two executable sites are KDE FAST helpers and two are the independent KDE oracle. IDENTICAL KDE has an existing portable norm construction. |

Concrete reference locations:

- `checks/portable_gelu_check.mojo:282,521,523,768,773,775,788,789,827,828`:
  erfc mentions only; quoted `erfc(2.0)` is an error message, not a call.
- `arima/impl/timeSeries/jones_transform.mojo:73`: proposed expm1 expression.
- `kde/impl/neighbors/kernel_density.mojo:538,553`: lgamma calls in
  `log_kernel_norm_fast` and `_log_vn_fast`.
- `kde/checks/kde_oracle.mojo:359,381`: Float64 reference lgamma calls.

The current shared module has portable FP32 exp/log/sqrt/sin/cos/pow/div/rsqrt,
log1p/sigmoid/silu/softplus/tanh/erf/GELU/fmax/clamp seams. It has portable host
FP64 exp/log/log2, and this change adds pow64 plus IDENTICAL wrappers for
pow64/log2_64. There is no standalone portable sqrt64 or trig64 family here.
That absence must be distinguished from reachability: UMAP Lanczos calls
`symmetric_eig_host[DType.float32]`, which routes sqrt through identical_sqrt;
its generic Float64 branch is not the UMAP instantiation. PCA's shared host
spectrum tail does use Float64 stdlib sqrt, an independent existing seam.
No correctly-rounded general pow64 or generic FP32 reduction is claimed.

The prior IDENTITY_PATHS row18 statement that every IDENTICAL transcendental
was already repository arithmetic was broader than its evidence. The six UMAP
host modules contained real raw exp/log/log2/pow consumers. They now route to
portable wrappers only in IDENTICAL, while preserving the prior non-IDENTICAL
calls. `umap/PORTABLE_HOST_MATH.md` specifies pow approximation bounds and
subnormal/special-value semantics. The primitive's x**0/1**p precedence is
explicit even for signaling NaNs: glibc may return a quiet NaN where ours
returns1. The gate checks our declared policy exactly and reports that libm
policy difference separately from finite numerical accuracy.

Explicit precision-specific findings requested in the continuation:

- `atanh(Float64)`: `arima/checks/fit_check.mojo:919` consumes the Float64
  local `v`, and `arima/checks/fit_oracle.mojo:99` consumes a List[Float64]
  element. `arima/impl/timeSeries/jones_transform.mojo:120` is Float32 and
  occurs only in the non-IDENTICAL else branch; it is not an atanh64 gap.
- `cbrt`: zero textual occurrences and zero executable calls in this scan.
- `log2(Float32)`: `gbdt/methods/kernel/split_properties_helpers.mojo:119`
  explicitly evaluates ceil(log2(Float32(self.fold_count))). It is called
  through data_partition_offset by pointwise_scores and pointwise_kernels.
  Its source docstring explicitly preserves the upstream float expression.
  This is an existing tree seam, left unchanged and untested under the user's
  no-tree scope; it must not be mislabeled as a comment or closed by adding
  the separate binary64 UMAP wrapper. Other generic log2-name hits do not
  imply Float32 arguments. The JSON scanner now includes atanh and cbrt.
