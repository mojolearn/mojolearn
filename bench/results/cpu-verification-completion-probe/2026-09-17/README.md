# Installed CPU development-wheel replay

The installed artifact at source `0dcf5b5a0` passed all nine fixtures twice
for 16 changed/reference-filled lanes: 423 IDENTICAL, 297 explicit N/A,
zero DIVERGENT, REFUSED or OWED. Its comparator self-test passed. Coverage
reports 142 available, 37 withheld and 50 parallel exclusions. All available
lanes have all-nine core references. This probe directory is not admitted as
numerical reference evidence.

`check_installed.py LANES LABEL` (copied into EVIDENCE_DIRECTORY) reproduces the check in
an isolated installed environment outside the checkout, without host/harness
or runtime path overrides. The receipt hashes the wheel, bindings and runtime
closure. The wheel reuses Mac bindings except the freshly compiled metrics
binding. This is scoped development-artifact verification, not final native
source qualification or a PyPI publication.

The retained neural preflight timed out during extended batch-size properties.
Its checkpoint is incomplete and was not admitted. The passing scoped tests
number 278. Cloud source records and native control pairs live separately in
`bench/results/identity_break/2026-09-17_cpu-classical-completion/` and
`2026-09-17_cpu-metrics-constant/`; the classical pod was verified deleted.
External artifacts: `~/mojolearn-evidence/cpu-verification-completion/`.

The follow-up wheel at `b74de8f54` also passes UMAP and ordered sum: 45 IDENTICAL,
45 N/A, no failures or missing references. Coverage reports 144 available and
35 withheld. Its training binding is freshly compiled as well as metrics;
other bindings remain reused. `completion-*` files record this separate artifact.
307 focused tests passed. No final release or PyPI upload is claimed.
