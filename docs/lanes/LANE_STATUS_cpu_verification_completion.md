# CPU verification completion — active checkpoint

Updated 2026-09-17. User asks to continue every exposed algorithm and expose
honest verification in the installed wheel. Standing workflow: use separate
worktrees, commit coherent verified batches, and merge them into main.
Worktree: `~/mojolearn-wt/cpu-verification-completion`, branch
`fix/cpu-verification-completion`. External evidence:
`~/mojolearn-evidence/cpu-verification-completion/`.
Main includes completed work through `06622158a`. No PyPI upload or final
release qualification is claimed. Earlier checkpoints remain in git history.

## Current coverage and remaining work

Baseline was 128 available CPU lanes, 51 withheld, 50 parallel exclusions.
Current candidate: **150 available, 29 withheld, 50 excluded** (229 harness
lanes; 246 appendix entries and the source API inventory are separate counts).
All available lanes have all-nine core references, including explicit N/A.
The remaining 29 are:

- Twelve low-bit weight lanes: Transformer, Mamba1/2/3, MLP and Samba, each
  BF16 and INT8. The live cloud run below records these with complete native
  dependencies, clean properties and all-nine native training controls.
- Five ordinary neural lanes: mamba3, transformer, transformer-window, samba,
  samba-untied-dropout-accum. Concurrent `reference-regen` work is recording
  full properties. Inspect its records and admission policy before integrating;
  its starting branch predates our incomplete-record and scoped-admission fixes.
- Twelve vendor-held lanes: gp-optimize, gp-optimize-restarts, svc-poly,
  gbdt-query-rmse, gmm-random-init-sample, gmm-sample, gp-normalize-y,
  gp-sample-y, gp-sample-y-normalize, gpc, gpc-multiclass, ivf-extend.
  The live job adds CPU evidence; NVIDIA and AMD evidence remains required.

Still owed beyond CPU admission: missing optional properties, all-fixture
native controls, physical device-use attestation, multi-GPU pairs and final
frozen artifact qualification. Public CPU training outside verification still
refuses. Do not equate availability or historical evidence with qualification.

## Implemented and verified

`tools/verify_cpu_batch.py` records bounded clean/native sabotage pairs, two
repetitions and all nine fixtures, step/full and applicable RLPAIR by default;
`--properties` adds gradient, batch-size and ragged checks. It requires matching
source/input/device context, actual repeated changed native bytes, and clean
properties STABLE or N/A. It rejects training-only success with property failures.
Scoped strict admission preserves unrelated table cells/policies and rejects
incomplete checkpoints. Installed coverage exposes per-lane admission policy.

Promotions: embedding/embedding-sort/ivf-euclidean; metrics H/C/V; ten classical
GBDT/RF/HDBSCAN/kmeans variants; UMAP; ordered gradient sum; mamba2-dtlimit;
IVF, BF16/INT8 GEMM and byte-lm/byte-lm-resident. Existing BPE and fold lanes
also gained complete references (Python-only, no native-control claim).

Fixed native controls: metric constant-label zero-entropy branch returns wrong
zero in sabotage builds; ordered accumulation zeros the first result (reversing
two addends was inert). Production arithmetic unchanged. Ordered oracle failure
preserves actual repeated hashes as DIVERGENT and exits one; ordinary errors
remain REFUSED. Comparing two equally wrong oracle records remains DIVERGENT.
Mamba2 references use the measured current 0.5/0.9 clamp; old pins were stale.

336 focused tests passed after the compare-commitment merge. Installed development
wheel replays, all nine fixtures twice, no DIVERGENT/REFUSED/OWED:

- Sixteen changed/reference-filled lanes: 423 IDENTICAL, 297 N/A.
- UMAP and ordered sum: 45 IDENTICAL, 45 N/A.
- Mamba2 clamp: 36 IDENTICAL, 9 N/A.
- IVF, both low-bit GEMMs and both byte-LM lanes: 126 IDENTICAL, 99 N/A.

Receipts and logs: `bench/results/cpu-verification-completion-probe/2026-09-17/`.
Wheel binaries are archived externally under `wheel-artifacts/<sha256>/`.
The five-lane wheel at `91fdf2387` has fresh metrics/training/linalg bindings;
others were reused. Its first artifact failed low-bit GEMM because old linalg
lacked the API; that failed report and artifact identity are retained.

A subsequent 32-family installed export audit found stale forest and GBDT
bindings. Both were rebuilt at `06622158a`. Commit `7f5b786ae` makes Mac and
Linux packaging gates reject missing callable manifest exports. 167 focused
packaging/manifest tests pass. Installed rebuilt export/tree replay PASSED: all 32 families have their
exports; seven tree lanes yield 198 IDENTICAL and 117 N/A, no failures/owed.
Receipts and both export audit outcomes are retained in the probe directory.
A fresh build of all 32 host families from frozen `7f5b786ae` is now running
locally in session 21523, `fresh-host-build.log`. Its source archive and
`build_fresh_hosts.py` are external. Output `fresh-host-build-7f5b786ae/`;
per-family build receipt/checkpoints include commands, toolchain and hashes.

## Cloud ownership and restart

Only OUR two-vCPU cloud job plus one local single-thread numerical/compiler
worker may run. Use mac_slot.py, nice 19, numerical thread limits one; compile
at most two cloud workers or one local worker. Other sessions have independent
pods; do not interfere with their jobs. No broad Metal matrix. R2 caches enabled.

First pod `0krigxfoki5p54`: VERIFIED DELETED, $0.0238. Ten classical lanes
passed; original metric control failure retained. Second `4018mzhxmp3prb`:
VERIFIED DELETED, $0.0433. All 25 records retained at
`identity_break/2026-09-17_cpu-neural-completion/`. Only six full lane pairs pass;
18 neural pairs lacked the neural dependency and ordered-sum fault was inert.
Its original training-only success summaries are NOT full qualification;
`reevaluated-controls.json` is authoritative. These issues were subsequently fixed.

**LIVE third pod `17o53wuev9wtbr`**, source `1d7fc53fe`, created 16:19 EDT,
120-minute watchdog and external dead-man, two vCPUs, $0.06/hr. Root session
95156. SSH `-p 39174 root@38.80.152.147`. Remote output `/root/leg_out/`.
External `cloud-complete-dependencies/`, log `cloud-complete-dependencies-run.log`.
Seventeen production and fifteen sabotage families, including neural, run 24
lanes with two single-thread workers and 2400-second per-arm limits. First four
Transformer/Mamba1 low-bit pairs pass; Mamba2 low-bit sabotage arms in progress.
Do not kill an owed run to start another rental. Collect results and verify
DELETE/404/not-listed before renting again. Records are nested
`records/<lane>/<lane>/cpu-*.json`; exits are `records/batch-exits.json`.

Concurrent sessions: `reference-regen` records eight ordinary neural lanes with
all optional properties; `sabotage-sweep` records broader controls on base/ties
only. Their two-fixture controls do not replace our all-nine pairs. Inspect
tracked main and these handoffs before starting overlapping work.

## Next actions

1. Finish all-family Mac host build, stage the runtime closure, install its
   development wheel and replay CPU references. Preserve every failure.
   Commit and merge verified batches, including the export gate fix.
2. Monitor and finish live CPU job; preserve failures and teardown evidence.
3. Admit only fully passing low-bit lanes through scoped strict generation,
   regenerate packaged evidence/matrix, and replay installed wheel. Vendor-held
   lanes stay held even if all CPU checks pass.
4. Audit/integrate concurrent neural records without losing unrelated references.
5. Continue native artifact, optional-property and hardware qualification. No
   full release claim for development wheels with reused native bindings.
