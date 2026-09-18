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
  unchanged as evidence; validation of the reporting fix is pending.

## In progress

All 23 held routes are being replayed through the installed public CLI, nine
fixtures, two repeats, with extended properties. Each completed route retains
its JSON, stderr and receipt. Execution uses the shared slot, nice 19 and one
numerical worker. The first neural attempt was interrupted before its short
deadline; completed model/self-test results were retained. The resumed run has
a 600-second per-route limit and a 2400-second overall limit, with process-group
cleanup. Its results must be read before claiming any route passed.

External artifacts:
`~/mojolearn-evidence/verifier-installed-continuation-2026-09-18/`.

## Still owed

Current NVIDIA captures, any uncompleted installed CPU property replay, a fresh
source-pinned expanded wheel/native build and exact-artifact qualification,
physical two-GPU evidence, and the two historical Apple GP sampling/odd
disagreements remain separate debts. No reference hash, default hold or
`release_qualified` flag changed. Historical 178-lane records still have mixed
AMD build digests and five missing AMD parallel lanes; they do not qualify this
development artifact.
