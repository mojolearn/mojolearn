# GPC score native coverage

Authored **15 CPU / 14 GPU cases**, uncompiled and unrun. Import
`gpc_score_cpu_suite.run_checks` and `gpc_score_gpu_suite.run_checks` for
the root bundle. Thin executable wrappers use the same names without
`_suite`. Both use `MOJOLEARN_NUMERIC_IDENTICAL=1`, with
`MOJOLEARN_COLUMN_CPU=1` additionally for CPU.

Six analytic graphs each run the actual COUNT descriptor with ranges off
and on. Independently prescribed unions, remapped row codes and matching
fractions cover negative integers, duplicates, unseen truth labels, full
Int64 endpoints, mixed UTF-8/UCS4 spellings, a surrogate, an astral character
and positive/negative `2**90`. These retain the actual host label encoder,
shared key index and native gather; the fixtures do not copy those algorithms.
Native ownership checks destroy key/source/encoder owners before metric
execution, mutate snapshotted codes, and move the complete program while
checking allocation identity. CPU additionally checks the empty native union.

Two further graphs reuse the existing 888 bytes of original 48×3 binary GP
data, seed 0, first ten query rows, Constant×RBF kernel and 100-iteration
limit. They execute actual fit, prediction, label encoding and COUNT, checking
the score against a separate parallel equality predicate and guarded outputs.
The `1e-6` composition tolerance is supplementary; it does not replace or
weaken an existing fit/query assertion. Analytic fractions use `1e-12`.

CPU cases dispatch across the full existing host pool. GPU graphs run in
order on one physical GPU while kernels retain their parallelism. Shared
DeviceContext concurrent submission remains separately unqualified.

No Python definitions or assertions are deleted. GPC is absent from the
original `test_estimator_score_p2.py` classifier parameter table, so its shared
score helper is not claimed as migrated GPC coverage. Existing fit/query and
public score tests remain. Public class identity/rebinding, unused classes,
scalar admission, float-label refusal and error precedence stay Python seams.

Dependencies are `76bba34249`, `6f5afa90c8`, `b8acfbcfdb`, `b9b0edfe7f`,
`4d8e2c64b3`, `6ed4b3b075`, and `4256e469ae`. Local adoption commits are not
producers. The unrelated root type repair `0ef98ae676` is already collected.
There are no product edits in the fixture producer. Binding roots are listed
in `gpc-score-native.json`; the parent owns compilation and execution.

Independent source review supports eight additional GPC roles, making fifteen
for GPR/GPC score composition. None repeats the earlier 69 GP roles, the seven
GPR score roles, or shared snapshot/arena mechanisms. Required Python scalar
identity pins remain explicitly counted as pins, not eliminated objects.
Thirty-six native local modules have no Python imports; 130 numerical/helper
files are unchanged. The five native roots match staging `87a3c4ccc3`.
This is source evidence only, not an integrated or runtime count.

Full evidence is under
`~/mojolearn-evidence/python-mojo-migration-20261110/gpr-score-native-tests/`:
`gpc-role-audit.json`, `gpc-score-independent-oracle-cases.json`, and
`gpc-score-original-python-inventory.json`. All original failures and earlier
accepted results remain untouched.
