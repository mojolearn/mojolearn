# GPU evidence repair continuation — 2026-10-14

Updated 2026-10-14T07:12:34.329352+00:00. Branch `main`; latest implementation commit `32fe27974481a93ac10fd1c4b446a968d59c23d0`. This is an incremental repair of the original 303 NVIDIA/AMD GPU-opponent recipes. Full-board timing remains incomplete. No Apple runs, fast-mode reruns, numerical default changes, paper publication or website publication occurred. The faster-than-opponent publication hold remains.

| Vendor | Original recipes | Latest preflight matched | Latest preflight refused | Newly admitted fit recipes |
| --- | ---: | ---: | ---: | ---: |
| nvidia | 213 | 68 | 145 | 35 |
| amd | 90 | 39 | 51 | 25 |

Preflight counts are parameter checks, not timing completion. Newly admitted fits are complete paired measurements from this continuation, with one excluded warmup and one scored sample. They are not statistical performance equivalence claims. Actual dimensions, parameters, seeds, calls, versions, clocks, quality and retention references remain in the board cells. DBSCAN's saved workload is 1,000,000 rows; HDBSCAN's is the saved 100,000-row block. Dataset caps are explicit recipes, not proof that an entire original source dataset was timed.

An additional 14 existing NVIDIA transform results received recovered quality evidence from verified retained inputs and scored output bytes, with original timings unchanged. Eleven fit comparisons qualified. Inference quality receipts were recovered separately; comparisons lacking a common derivable clock remain excluded. Both power-transformer datasets and the Istella standard-scaler pairing still fail quality requirements. Margins were not relaxed.

The 21 completed recipes shared by NVIDIA and AMD have matching full SHA-256 digests of scored outputs. This reuses release 0.8.44 binaries and does not claim new Apple validation. Twenty-one supported historical audit rows were restored after fixing rebuild bookkeeping invalidation; actual settings, source, worker, quality and clock changes still invalidate approval.

## Repairs delivered

- Effective-parameter read-back for hipVS, optimizer options, cuML kNN/LinearSVC/kernel-ridge and clustering controls; actual layer/optimizer inputs and library call provenance.
- Reference quality for retained transforms, optimizer update deltas, finite output checks, and stochastic dropout channel/scaling/probability checks. Dropout's lost opponent outputs were repaired and the affected pairings reraced on both vendors.
- Verified quality corrections supersede only the original matching measurement. Board rebuilds preserve supported unchanged per-arm audits and audit history.
- Large R2 transfers stream bytes, retry without remeasuring, and verify full GET hashes. Failed uploads and original errors remain retained.
- Full workload retries use established board time ceilings. Longer sessions require an armed provider deadline and a retention reserve. Successful matching opponent records can be reused; failed records remain separate.

The classical runner previously deleted scored output arrays before sealing archives. This was discovered after the first AMD rental was deleted. Four AMD classical records have now been remeasured and admitted with independently verified retained outputs. Both million-row NVIDIA DBSCAN recipes are complete and admitted, with exact partition/noise-mask agreement and verified scored output bytes; both are losses against cuML. Both NVIDIA HDBSCAN records have complete retained outputs but fail the unchanged silhouette margin (Taxi difference 0.010455; Istella 0.014372), so remain excluded. Eight older NVIDIA kmeans/kNN/PCA/KDE records also lacked scored arrays and are queued for repair. Archive transfer success alone did not prove output retention. The repaired runner retains NPZ bytes and binds archive membership and content hashes to the scored digest. HDBSCAN now records its consumed slice and full-slice silhouette quality; existing quality margins are unchanged.

## Validation and evidence

252 focused tests passed (`final-focused-tests-07.log`), including classical retention, actual-shape reporting, quality policy, upload retries, controller-failure stop behavior and inference fixture checks. The full host-route checker regression suite also passed locally; its required GitHub workflow passed after repairing the five-minute timeout. Full logs are retained under `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014`. The board fairness check is recorded alongside them. The latest main NVIDIA/AMD boards were updated through `tools/main_board_ingest.py`; this file does not substitute for their per-phase audit marks.

- [Summary, freezes, quality recovery and retention references](summary.json)
- [All 303 recipes and latest unresolved preflight reasons](recipes.json)
- [Retained numerical quality failures](quality-failures.json)
- [Scored cross-vendor hashes](cross-vendor-scored-hashes.json)
- [Historical audit recovery decisions](supported-audit-recovery.json)

- amd: deleted. Current required scored outputs, inputs, accepted binaries and logs were verified before deletion; provider absence is recorded. The first rental had four classical output-array omissions; fresh wave8 measurements repaired those records. The original loss is preserved in the incident record.
- nvidia: live. NVIDIA classical-output repairs remain active; do not terminate before current required outputs and archives are verified.

## Resume without rerunning valid work

Use the live evidence tree's `summarize.py`, `compare-hashes.py`, `retention-audit.py` and `ingest-landed.py` from the repository root. Inspect session exit status and individual race/quality status separately: a controller exiting zero can contain refused or failed recipes. Preserve the original AMD setup failures and all transfer retry evidence. NVIDIA wave 7 was intentionally stopped when missing classical output retention was found. Wave8 repaired those six output omissions; both HDBSCAN quality failures remain excluded. Wave9 is queued to repair the same omission in eight older NVIDIA classical records. Original archives, failure records and deleted-rental receipts remain preserved.

Remaining work includes unresolved implementation/parameter read-backs, genuine transform-quality failures requiring a separate numerical freeze, missing inference upload clocks, and the remaining unmeasured full-board recipes. Reuse accepted compilation/identity evidence and retained library measurements. No dimension-specific route changes, Python product fallbacks, unsupported Mojo workarounds, quality-margin relaxation, or publication-hold removal are authorized by this handoff.
