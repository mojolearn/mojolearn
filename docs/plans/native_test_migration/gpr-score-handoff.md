# GPR score native coverage

Authored **17 CPU / 10 GPU cases**, uncompiled and unrun. Import
`gpr_score_cpu_suite.run_checks` and `gpr_score_gpu_suite.run_checks` for the
next root bundle; thin executable wrappers are `gpr_score_cpu.mojo` and
`gpr_score_gpu.mojo`. Both use `MOJOLEARN_NUMERIC_IDENTICAL=1`, with
`MOJOLEARN_COLUMN_CPU=1` additionally for CPU. This is correctness coverage,
not an identity or performance campaign.

Seven pure cases cover native storage movement, zero/adjacent input ranges,
independent copied input ownership, finite checks and truth-first error
precedence. Ten actual-provider cases exercise four independent analytic
score oracles and the original 60×4 GPR workload, each with the range flag
off/on. On CPU the actual metrics capability chooses its existing output
route. On GPU these exercise the actual range/full-output routes, pointer
fit and resident mean-only query when advertised. Kernels are unchanged.
CPU cases use the full shared host pool; GPU graphs are ordered because
shared DeviceContext concurrent submission remains unqualified.

The original workload comes from `_reg_data` in
`python/mojolearn/tests/test_estimator_score_p2.py`, with the same seed 1,
default fixed Constant×RBF kernel, ridge `2**-20`, no normalization or
optimizer, and original absolute score tolerance `1e-3`. Three saved files
total 1,224 bytes. The independent NumPy Gaussian-covariance/solve/R² oracle
imports no Mojolearn code. The analytical fixtures additionally preserve
constant-target policy and the existing Float32-rounded target mean; these
supplementary cases are not counted as migrated Python definitions.

All existing Python tests remain. Only the GPR parameter row's numerical
assertion is a candidate for later removal after equivalent acceptance;
the public call/scalar packing and short-target refusal remain Python seams.
The shared score helper still serves other estimators. Nothing is deleted
from product code or existing tests by these fixture commits.

Dependencies are producer `76bba34249` plus `6f5afa90c8`. Local cherry-picks
`829373938d` and `1171741f6d` are dependency adoptions, not new producers.
The reference producer is `573b722368`. Affected binding roots are listed
in `gpr-score-native.json`; root owns all builds.

Independent source review supports the seven GPR roles without overlap
with the prior 69 GP roles. Its two native roots resolve to 27 local modules
with no Python imports; 127 numerical source files are unchanged. The one
changed existing native file extracts shared R² execution in
`core/linear_score_run.mojo`, preserving the separate linear constant-target
policy. These GPR native sources match root staging `87a3c4ccc3`. The eight
GPC score roles remain outside this audit. Full source/hash/anchor evidence:
`~/mojolearn-evidence/python-mojo-migration-20261110/gpr-score-native-tests/gpr-role-audit.json`.
