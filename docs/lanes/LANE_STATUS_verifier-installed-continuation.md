# Installed CPU verifier continuation — 2026-09-18

Continues the verification recovery handoff in its existing worktree. No new
lane, rental, full native build or publication was started.

## Development wheel and completed checks

The development CPU wheel packages source `6b3399b88` with 32 retained host
bindings. Thirty come from the earlier broad-verifier wheel staging tree;
the kernel-methods and estimators bindings come from the completed kernel
identity work. Source paths, original hashes, packaged hashes and exact wheel
SHA256 are retained in `bench/results/verifier-installed/2026-09-18/`.
Runtime dylibs were staged with the normal closure checker. This mixed-source
native artifact is explicitly not the fresh expanded release candidate.

The wheel passed current Python/public-API/reference-payload completeness.
An isolated install outside the checkout passed a byte-for-byte comparison
with every wheel package member. No external native override was used.

- All 58 bundled models passed both model and batch checks, with two repeats:
  116 IDENTICAL, zero DIVERGENT, REFUSED or OWED.
- The verifier self-test reproduced the clean OLS reference and detected the
  actual one-ULP input perturbation as DIVERGENT.
- This exposed a reporting bug: `models_checked` counted distinct model lanes
  (10) instead of lane/fixture models (58). The fix reports both counts and
  preserves the human table's lane total. The original wheel report remains
  unchanged as evidence. Source runtime validation confirms 58 models across
  10 lanes and the same 116 passing comparisons after the reporting fix.

## Completed ordinary installed replay

All 23 held routes passed the installed public CLI on all nine fixtures with
two repeats: **810 numerical comparisons IDENTICAL**, 225 explicit N/A, zero
DIVERGENT, REFUSED or OWED. The complete ordinary job took 183.70 seconds,
including the model and self-test checks. JSON, stderr, commands, package-byte
verification and receipt are retained per route. Execution used the shared
slot, nice 19 and one numerical worker.

The public reports deliberately still return INCOMPLETE/exit 5 because the
default qualification holds remain. The five neural reasons now say
`qualification pending`, rather than incorrectly continuing to say `unwatched`.
No route was promoted by this development replay.

The all-nine extended Mamba attempt hit its 600-second route limit without a
complete report. Its process group was terminated; the following Transformer
attempt was interrupted before switching to the standard replay. Neither is
a numerical failure or a pass. Extended installed neural properties remain
owed. The earlier shorter interrupted attempt and all logs remain externally.

The 18 classical/kernel routes also passed a separate all-nine, twice-repeated
installed run with `--batch-checks`: 612 IDENTICAL, 846 explicit N/A, zero
DIVERGENT, REFUSED or OWED, in a 55.14-second job. These families declare the
additional neural properties inapplicable; the N/A values are not numerical
comparisons. Their full installed property replay is complete for this artifact.

## Whole loaded model and reference-admission repair

Installed `verify-causal-lm` passed all 24 v2 cases across all seven checkpoint
families and FP32/BF16/int8. All 144 parts match each of the retained CPU, Apple
and AMD captures through the strict comparator, including its source-digest,
checkpoint, complete-case and property checks. Composition controls passed;
this is not new native arithmetic-fault or physical two-device evidence.

The two historical `gp-sample-y/odd` Apple discrepancies were traced to
`2026-09-15_gp-sample-y/metal-transient/`. That directory's contemporaneous
README explicitly quarantined its faulty-device runs, but admission still
accepted them. Admission now rejects only that particular incident directory;
the regression retains clean GP records and unrelated paths.

The already committed installed Apple sampling captures pass strict repeated
input/protocol/revision admission and match the CPU/AMD values. A scoped repair
replaces provenance for the two sampling lanes: 18 selected cells, **every
existing reference hash unchanged**, and all 2,106 other cells byte-for-byte
unchanged. Every selected part now has agreeing CPU/Apple/AMD witnesses, and
no cell anywhere in the resulting table cites the quarantined records. The
original incident files and historical record metadata remain retained. This
repairs evidence selection; it does not diagnose the original device incident
or certify arbitrary Apple inputs.

After the scoped repair, **281 focused tests passed** in 5.44 seconds, with no
skips. The harness/verifier drift test was explicitly scoped to OLS; the
separate installed replay covers all 23 held routes. An initial test invocation
that implicitly launched all public CPU routes was interrupted and retained,
then replaced by this appropriately scoped validation.

External artifacts:
`~/mojolearn-evidence/verifier-installed-continuation-2026-09-18/`.

## Still owed

Current NVIDIA captures, extended installed neural property replay, a fresh
source-pinned expanded wheel/native build and exact-artifact qualification,
physical two-GPU evidence and broader Apple numerical identity remain separate
debts. No reference hash, default admission or `release_qualified` flag changed.
Historical 178-lane records still have mixed
AMD build digests and five missing AMD parallel lanes; they do not qualify this
development artifact.
