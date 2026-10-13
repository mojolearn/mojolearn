# Published-wheel verification repairs

**Correction, 2026-10-13:** the Apple-only reference policy used in this report was
incorrect. IDENTICAL requires Apple, NVIDIA and AMD to match the same hashes;
Apple FAST is separate. The 72 feature-frequency differences accepted below are
unresolved numerical failures, not repaired cross-vendor identity. The corrected
verifier reports all 72 as DIVERGENT. Original reports and counts are retained as
historical evidence of the affected verifier. The neural cross-vendor comparison
below remains valid because its actual hashes agreed across all three vendors.

Source repair: `87e6005eaad08b3f67a25d4e344a555bb2590e7e`. The starting installation
was PyPI `mojolearn==0.8.42`, source witness
`0db362a62516faef0a52b112e93f7cba5e212a2b`. Native binaries were reused unchanged;
their actual hashes are in the reports. This does not publish a new PyPI release.

The full neural profile ran on Apple M4, RunPod RTX 4090 and DigitalOcean MI325X:
64 lanes, all nine fixtures, all nine properties, one fit per cell. All 5,184
results agree across all three vendors: 2,349 numerical hashes and 2,835 explicit
N/A declarations. The original 1,336 mismatches per vendor were stale references.
The original failing reports are retained byte-for-byte in the compressed
`*.source-report.json.gz` files; their uncompressed SHA-256 values are in the
corresponding `*.from-verify.json` observation records. Those observations were
converted through `tools/identity_columns.py from-verify`, then admitted through
the reference builder. A changed reference alone never makes a failed local
property admissible.

The eight affected routine lanes were executed again on all three vendors.
Every NVIDIA/AMD cell part agrees. Apple differs only in the 72 expected
feature-frequency parts; its own recorded reference is now selected, consistent
with the NVIDIA/AMD identity contract. The eigenvalue test's input now uses a
fixed float32 reduction instead of platform-dependent BLAS. The DART binary
probability path now keeps its borrowed zero column alive until the native
stack call returns, fixing first-row corruption in Kernel SHAP. Its regression
test also fails against the original published implementation as expected.

All four radius metrics execute all nine GPU/CPU fixtures on each machine:
72/72 compared parts agree per vendor, no skipped cases. Fresh 64-lane neural
base-fixture verification exits zero on all three machines; Apple quick exits
zero too. The retained 3,456-cell full routine run, with fresh results for the
eight affected lanes substituted and the new reference applied, has 11,240
IDENTICAL and 19,980 N/A parts, with no mismatches, missing references or refusals.
That is reuse of the completed full run plus affected-lane reruns, not a second
full routine execution.

`summary.json` records scope, source witnesses, tests, retained archives and
rental teardown status. Full logs and original diagnostics remain under
`bench/results/verify_repairs/20261012/` and the original installation evidence
under `bench/results/pip_verify/20261012-205427/`. Both rented machines were
deleted only after their output, original wheels and patched source bytes were
captured locally and the archive SHA-256 verified against the remote archive.

Reference regeneration retained every previously covered part and introduced
no conflicts or PTX differences. No PTX execution or additional hardware claim
is made by this repair.
