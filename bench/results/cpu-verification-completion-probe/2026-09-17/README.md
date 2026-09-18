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

The five-lane wheel replay at `91fdf2387` passes IVF, both low-bit GEMMs,
byte-lm and byte-lm-resident: 126 IDENTICAL, 99 N/A, zero failures or owed
references; coverage is 150 available, 29 withheld, 50 parallel exclusions.
The first artifact refused both low-bit GEMMs because its reused Mac linalg
binding lacked the low-bit API. That failure and artifact identity are retained
in `neural-five-old-linalg-*`. A fresh linalg binding fixes the installed replay.
Metrics, training and linalg are fresh; other native bindings remain reused.
`neural-five-wheel-receipt.json` identifies the successful artifact and runtime
closure; wheel binaries are archived externally by SHA-256. The post-merge
focused suite passes 336 tests. This remains scoped development-wheel evidence.

The 32-family export audit initially found stale forest and GBDT bindings.
Fresh builds remove all missing exports. The wheel at `7f5b786ae` then passes
seven affected tree lanes: 198 IDENTICAL, 117 N/A, zero failures or owed parts.
`host-export-*` receipts/logs and `installed-exports-*` audits retain both
outcomes. Mac and Linux packaging gates now require callable manifest exports;
167 packaging/manifest tests pass. Five families are freshly built at this
point; the subsequent frozen all-family rebuild is separate work.
