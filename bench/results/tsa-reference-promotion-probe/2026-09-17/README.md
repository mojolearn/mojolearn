# Scoped CPU TSA admission and installed-wheel replay

`admit.py` reproduces the two-lane reference extension from baseline `1752ab7c4`
using the repository's strict generator. `admission.json` records its policy,
harness digest and source column. All 1,638 previous cells remain unchanged.
`generation.json` describes the full staging-only regeneration, not the shipped
table; it was not substituted for the legacy references.

`check_installed.py EVIDENCE_DIRECTORY` runs the installed development wheel
outside the checkout and asserts coverage, default selection, all-fixture public
results and historical native controls. The directory contains `installed-env/`
and `dist/`. Successful reports are `coverage.json.gz` and `public-clean.json.gz`;
`source-clean.json.gz` is the earlier source replay. `wheel-receipt.json` records
the successful artifact, native binding and runtime hashes. The wheel uses
reused Mac bindings; no fresh native-source or final-release qualification is
claimed. Artifacts remain under
`~/mojolearn-evidence/tsa-reference-promotion/`.

Retained failures: `failed-runtime-clean.json.gz` is the initial isolated-venv
run whose bindings could not resolve the missing runtime dylibs. All 90 parts
refused, so this is not positive evidence. The successful rerun used the release
runtime staging helper, whose closure checks are in `runtime-staging.log`.
The prior import failure from the omitted `models` subpackage was corrected in
both wheel builders. `tests-unbounded-drift-stopped.log` records stopping the
unintended broad default CPU drift check; `tests.log` is the passing scoped run
(266 tests and two subtests). `matrix-check.log` confirms generated consistency.

This `probe` directory is deliberately excluded from numerical reference
admission. Admitted references come from the committed clean Linux column under
`identity_break/2026-09-17_tsa-negative-controls/`, with its native controls,
source/binding digests and teardown receipts. No cloud rental occurred here.
