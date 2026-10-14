# GPU evidence repair continuation — 2026-10-14

Updated 2026-10-14T05:20:41.247407+00:00. Branch `main`; latest implementation commit `4dce9b50aa68d8561d71cadea49d2bf2c539aad6`. This is an incremental repair of the original 303 NVIDIA/AMD GPU-opponent recipes. Full-board timing remains incomplete. No Apple runs, fast-mode reruns, numerical default changes, paper publication or website publication occurred. The faster-than-opponent publication hold remains.

| Vendor | Original recipes | Latest preflight matched | Latest preflight refused | Newly admitted fit recipes |
| --- | ---: | ---: | ---: | ---: |
| nvidia | 213 | 64 | 149 | 33 |
| amd | 90 | 39 | 51 | 25 |

Preflight counts are parameter checks, not timing completion. Newly admitted fits are complete paired measurements from this continuation, with one excluded warmup and one scored sample. They are not statistical performance equivalence claims. Actual dimensions, parameters, seeds, calls, versions, clocks, quality and retention references remain in the board cells. DBSCAN's saved workload is 1,000,000 rows; HDBSCAN's is the saved 100,000-row block. Dataset caps are explicit recipes, not proof that an entire original source dataset was timed.

An additional 14 existing NVIDIA transform results received recovered quality evidence from verified retained inputs and scored output bytes, with original timings unchanged. Eleven fit comparisons qualified. Inference quality receipts were recovered separately; comparisons lacking a common derivable clock remain excluded. Both power-transformer datasets and the Istella standard-scaler pairing still fail quality requirements. Margins were not relaxed.

The 21 completed recipes shared by NVIDIA and AMD have matching scored output hashes. This reuses release 0.8.44 binaries and does not claim new Apple validation. Twenty-one supported historical audit rows were restored after fixing rebuild bookkeeping invalidation; actual settings, source, worker, quality and clock changes still invalidate approval.

## Repairs delivered

- Effective-parameter read-back for hipVS, optimizer options, cuML kNN/LinearSVC/kernel-ridge and clustering controls; actual layer/optimizer inputs and library call provenance.
- Reference quality for retained transforms, optimizer update deltas, finite output checks, and stochastic dropout channel/scaling/probability checks. Dropout's lost opponent outputs were repaired and the affected pairings reraced on both vendors.
- Verified quality corrections supersede only the original matching measurement. Board rebuilds preserve supported unchanged per-arm audits and audit history.
- Large R2 transfers stream bytes, retry without remeasuring, and verify full GET hashes. Failed uploads and original errors remain retained.
- Full workload retries use established board time ceilings. Longer sessions require an armed provider deadline and a retention reserve. Successful matching opponent records can be reused; failed records remain separate.

## Validation and evidence

235 focused tests passed (`final-focused-tests-03.log`). Full logs are retained under `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014`. The board fairness check is recorded alongside them. The latest main NVIDIA/AMD boards were updated through `tools/main_board_ingest.py`; this file does not substitute for their per-phase audit marks.

- [Summary, freezes, quality recovery and retention references](summary.json)
- [All 303 recipes and latest unresolved preflight reasons](recipes.json)
- [Scored cross-vendor hashes](cross-vendor-scored-hashes.json)
- [Historical audit recovery decisions](supported-audit-recovery.json)

- amd: deleted. Retention verified before deletion; DELETE, GET 404 and absence from the provider listing are recorded.
- nvidia: live. Clustering work and final retention remain active; do not terminate before every required archive is verified.

## Resume without rerunning valid work

Use the live evidence tree's `summarize.py`, `compare-hashes.py`, `retention-audit.py` and `ingest-landed.py` from the repository root. Inspect session exit status and individual race/quality status separately: a controller exiting zero can contain refused or failed recipes. Preserve the original AMD setup failures and all transfer retry evidence. Queued NVIDIA wave 7 selects only unstarted work and incomplete own measurements affected by the earlier time ceiling; successful scored arms require recovery/reuse before any retry.

Remaining work includes unresolved implementation/parameter read-backs, genuine transform-quality failures requiring a separate numerical freeze, missing inference upload clocks, and incomplete full clustering measurements. Reuse accepted compilation/identity evidence and retained library measurements. No dimension-specific route changes, Python product fallbacks, unsupported Mojo workarounds, quality-margin relaxation, or publication-hold removal are authorized by this handoff.
