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
Last installed/merged batch: **150 available, 29 withheld, 50 excluded** (229 harness
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
All 32 host families were freshly built from frozen `7f5b786ae` in 244.7s,
and all installed export audits pass. Source/archive, compiler commands and
hashes are retained in the probe `fresh-host-build/` directory. An initial
dependency-free invocation exposed an uncaught missing-NumPy import.
`b6132eec0` fixes it with an actionable CANNOT RUN response, demonstrated in
the rebuilt installed wheel before installing NumPy; 56 verifier tests pass.
The all-150 CPU replay is LIVE in local session 19228, one numerical worker,
`fresh-cpu-wheel-replay.log`, per-lane compressed reports/checkpoints under
external `fresh-cpu-wheel-replay/`. Do not change its installed environment
`fresh-installed-env/` or its frozen wheel in `fresh-dist/` while it runs.
Python source b6132eec0, all native sources 7f5b786ae. Up to 1200s per lane,
7200s outer bound; all nine fixtures, two repetitions, default core properties.

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

Third pod `17o53wuev9wtbr` is VERIFIED DELETED (DELETE 204, GET 404,
not listed). Source `1d7fc53fe`, two vCPUs, 120-minute watchdog/dead-man,
$0.0365 total (build 253s, run 1844s, billed 2189s). No rental from this
workstream remains active. Records and teardown are committed under
`identity_break/2026-09-17_cpu-complete-dependencies/`.
All twelve low-bit lanes and ten vendor-held candidates pass complete clean
properties plus all-nine native training controls. `gp-normalize-y` and
`gp-sample-y-normalize` refused because preprocessing was omitted; keep these
failures. Future batches should build all 32 families to cover transitive
native dependencies. The vendor-held candidates remain held for GPU evidence.

The twelve low-bit lanes now have scoped strict candidate references, bringing
candidate availability to 162 and withholding to 17. Their installed replay
is still owed before merging this promotion into main. The ongoing frozen
150-lane fresh-wheel replay has exposed stale N/A batch references in bootstrap
and cross-val. These are OWED, not numerical divergence. Completed records on
`lane/reference-regen` cover both; inspect and integrate them with strict scoped
admission, preserving the original failed artifact reports.

Concurrent sessions: `reference-regen` records eight ordinary neural lanes with
all optional properties; `sabotage-sweep` records broader controls on base/ties
only. Their two-fixture controls do not replace our all-nine pairs. Inspect
tracked main and these handoffs before starting overlapping work.

## Next actions

1. Finish all-family Mac host build, stage the runtime closure, install its
   development wheel and replay CPU references. Preserve every failure.
   Commit and merge verified batches, including the export gate fix.
2. Admit and replay the twelve low-bit lanes, then merge the verified batch.
   All cloud outputs and verified teardown have been retained.
3. Admit only fully passing low-bit lanes through scoped strict generation,
   regenerate packaged evidence/matrix, and replay installed wheel. Vendor-held
   lanes stay held even if all CPU checks pass.
4. Audit/integrate concurrent neural records without losing unrelated references.
5. Continue native artifact, optional-property and hardware qualification. No
   full release claim for development wheels with reused native bindings.

## Concurrent branch integration and CI follow-up

Completed commits from `lane/reference-regen` and `lane/sabotage-sweep` are
merged into this worktree; the candidate is not yet merged into main. Incomplete
reference columns remain excluded by our strict admission code. Preserve the
reference branch's explicit `one column` hold for Samba until qualified.

The sabotage branch parked its alternative ordered-sum fault behind an unused
define because the old CI gate required STABLE sabotage cells. We retain our
measured zero corruption and explicit NumericalMismatch results. The CI gate
now has an explicit sabotage mode that accepts exit one only with complete
records, native sabotage readback, repeated actual hashes and oracle errors.
Production, ordinary exceptions, incomplete shards and unstable results still
fail. Sabotage CI uses two repetitions. The retained real nine-fixture ordered
control passes this validator; 51 gate/batch tests pass. Direct forest and byte
LM loaders now point at the selected native arm, with clean permissions reset.

The remaining all-nine native-control audit identifies 99 non-parallel lanes
with fewer than nine qualifying training controls. A follow-up CPU batch is
being prepared with all 32 native families and the required saved CTR models;
no rental has been started for it yet. The fresh-wheel 150-lane replay remains
active; keep its frozen installed environment unchanged.
