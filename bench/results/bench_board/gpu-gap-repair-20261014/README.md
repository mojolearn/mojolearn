# GPU evidence repair continuation — 2026-10-14

Updated 2026-10-14T08:16:57.057585+00:00. Branch `main`; latest implementation commit `4fa5f67bc9c85973ae04b8fbcf076aaed903ccdc`. This is an incremental repair of the original 303 NVIDIA/AMD GPU-opponent recipes. Full-board timing remains incomplete. No Apple runs, fast-mode reruns, numerical default changes, paper publication or website publication occurred. The faster-than-opponent publication hold remains.

| Vendor | Original recipes | Latest preflight matched | Latest preflight refused | Newly admitted fit recipes |
| --- | ---: | ---: | ---: | ---: |
| nvidia | 213 | 74 | 139 | 49 |
| amd | 90 | 39 | 51 | 25 |

Preflight counts are parameter checks, not timing completion. Newly admitted fits are complete paired measurements from this continuation, with one excluded warmup and one scored sample. They are not statistical performance equivalence claims. Actual dimensions, parameters, seeds, calls, versions, clocks, quality and retention references remain in the board cells. DBSCAN's saved workload is 1,000,000 rows; HDBSCAN's is the saved 100,000-row block. Dataset caps are explicit recipes, not proof that an entire original source dataset was timed.

An additional 14 existing NVIDIA transform results received recovered quality evidence from verified retained inputs and scored output bytes, with original timings unchanged. Eleven fit comparisons qualified. Inference quality receipts were recovered separately; comparisons lacking a common derivable clock remain excluded. Both power-transformer datasets and the Istella standard-scaler pairing still fail quality requirements. Margins were not relaxed.

The 25 completed recipes shared by NVIDIA and AMD have matching full SHA-256 digests of scored outputs. This reuses release 0.8.44 binaries and does not claim new Apple validation. Twenty-one supported historical audit rows were restored after fixing rebuild bookkeeping invalidation; actual settings, source, worker, quality and clock changes still invalidate approval.

## Repairs delivered

- Effective-parameter read-back for hipVS, optimizer options, cuML kNN/LinearSVC/kernel-ridge and clustering controls; actual layer/optimizer inputs and library call provenance.
- Reference quality for retained transforms, optimizer update deltas, finite output checks, and stochastic dropout channel/scaling/probability checks. Dropout's lost opponent outputs were repaired and the affected pairings reraced on both vendors.
- Verified quality corrections supersede only the original matching measurement. Board rebuilds preserve supported unchanged per-arm audits and audit history.
- Large R2 transfers stream bytes, retry without remeasuring, and verify full GET hashes. Failed uploads and original errors remain retained.
- Linear-algebra receipts preserve complete QR/SVD factors and untruncated solve outputs. QR checks full reconstruction and orthogonality; policy v5 adds an orthogonality bound without changing existing margins. Policy v6 recognizes the recorded PCA inference reference metric under the existing 1e-4 bound.
- The clock admission gate now requires the owner-declared clock. Alternate common clocks remain diagnostic; missing CuPy/GPyTorch whole clocks cannot qualify a pairing. Neural torch whole clocks remain valid. Three completed wave10 CuPy results remain retained but excluded for their missing copy durations.
- Full workload retries use established board time ceilings. Longer sessions require an armed provider deadline and a retention reserve. Successful matching opponent records can be reused; failed records remain separate.

Four generated matrix input arrays are additionally retained as NPY bytes in a verified R2 archive. Exact reconstruction on the worker matched every original scored worker input hash, and an independent download rechecked all four arrays and their linked measurement records. This did not re-time any arm.

The classical runner previously deleted scored output arrays before sealing archives. This was discovered after the first AMD rental was deleted. Four AMD classical records have now been remeasured and admitted with independently verified retained outputs. Both million-row NVIDIA DBSCAN recipes are complete and admitted, with exact partition/noise-mask agreement and verified scored output bytes; both are losses against cuML. Both NVIDIA HDBSCAN records have complete retained outputs but fail the unchanged silhouette margin (Taxi difference 0.010455; Istella 0.014372), so remain excluded. Eight older NVIDIA kmeans/kNN/PCA/KDE records also lacked scored arrays; all eight now have new complete, admitted fit measurements. Both NVIDIA PCA inference comparisons now qualify using their retained FP64-reference errors and recorded copy clocks, without new timing. NVIDIA kmeans inference still needs an applicable quality decision; AMD hipVS kmeans has no wired inference call. Other mixed inference clocks remain explicit gaps. Archive transfer success alone did not prove output retention. The repaired runner retains NPZ bytes and binds archive membership and content hashes to the scored digest. HDBSCAN now records its consumed slice and full-slice silhouette quality; existing quality margins are unchanged.

## Validation and evidence

205 focused tests passed on the final source (`final-focused-tests-08.log`), covering parameter parity, quality, required clocks, complete output retention, inference, board ingestion, session/controller guards and lossless archive transfers. The generated-input retention tool also passed 27 targeted tests, including five new exact-byte reconstruction/refusal cases. Earlier validation logs remain retained, including the prior 252-test suite; these counts overlap and are not added together. The full host-route checker regression suite also passed locally; its required GitHub workflow passed after repairing the five-minute timeout. Full logs are retained under `/Users/andrewhendel/mojolearn-evidence/gpu-gap-resume-20261014`. The board fairness check is recorded alongside them. The latest main NVIDIA/AMD boards were updated through `tools/main_board_ingest.py`; this file does not substitute for their per-phase audit marks.

- [Summary, freezes, quality recovery and retention references](summary.json)
- [All 303 recipes and latest unresolved preflight reasons](recipes.json)
- [Retained numerical quality failures](quality-failures.json)
- [Scored cross-vendor hashes](cross-vendor-scored-hashes.json)
- [Historical audit recovery decisions](supported-audit-recovery.json)

- amd: deleted. Current required scored outputs, inputs, accepted binaries and logs were verified before deletion; provider absence is recorded. The first rental had four classical output-array omissions; fresh wave8 measurements repaired those records. The original loss is preserved in the incident record.
- nvidia: deleted. Current required scored outputs, inputs, accepted binaries and logs were verified before deletion; provider absence is recorded.

## Resume without rerunning valid work

Use the live evidence tree's `summarize.py`, `compare-hashes.py`, `retention-audit.py` and `ingest-landed.py` from the repository root. Inspect session exit status and individual race/quality status separately: a controller exiting zero can contain refused or failed recipes. Preserve the original AMD setup failures and all transfer retry evidence. NVIDIA wave 7 was intentionally stopped when missing classical output retention was found. Wave8 repaired those six output omissions; both HDBSCAN quality failures remain excluded. Wave9 repaired the same omission in eight older NVIDIA classical records. Wave10 completed Cholesky and both LU recipes. Its dispatch controller was intentionally stopped only after the active cell finished with verified archives; session exit 241 is retained as an operator stop, not a failed scored run. Wave11 completed the three unstarted recipes under faster lossless compression, recorded CuPy copy clocks and the required-clock admission fix. All six selected linear-algebra recipes are measured and admitted against supported opponents. No completed cell is repeated. Original archives, failure records and deleted-rental receipts remain preserved.

Remaining work includes unresolved implementation/parameter read-backs, transform/HDBSCAN quality-parity failures to investigate before any numerical repair freeze, missing inference upload clocks, and the remaining unmeasured full-board recipes. Reuse accepted compilation/identity evidence and retained library measurements. No dimension-specific route changes, Python product fallbacks, unsupported Mojo workarounds, quality-margin relaxation, or publication-hold removal are authorized by this handoff.
