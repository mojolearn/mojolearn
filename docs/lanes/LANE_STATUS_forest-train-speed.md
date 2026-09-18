# lane/forest-train-speed: ExtraTrees and RandomForest TRAINING speed on NVIDIA

Written 2026-09-17 evening by the resuming agent, for a reader with no memory.
The first agent was hard stopped in its final proof phase and left no note;
everything in the "reconstructed" sections comes from the branch's commits and
DEVIATION comments and from the evidence it pulled to
`~/mojolearn-evidence/forest-train-speed/leg_out/` (467 files, pod
`6x6vfh2zqas3n4`, RTX 4090, driver 580.159.04, Mojo 1.0.0 ed45d567, reaped by
the first agent at 20:53Z). Sections marked "resume pod" were measured by the
resuming agent on the second pod named there.

## The one change (DEVIATION 3022) and the two arms (3020, 3021)

Files on the branch beyond main `86d33fcdf`:
`extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo`
(the change), `tools/forest_train_ab.py` (the training A/B harness),
`tools/forest_train_body.sh` (the pod body), this document.

**DEVIATION 3022, the default flip, NVIDIA only.** The ExtraTrees search's
publish atomics (the range pass's `min`, `max`, two `fetch_add`; the score
pass's `3 + 2 * n_acc` `fetch_add`; the nonconstant flag's `fetch_add`) are
`Ordering.RELAXED` on an NVIDIA build (`has_nvidia_gpu_accelerator()`), the
default ordering (seq_cst) elsewhere. `-D MOJOLEARN_ET_SEQCST_PUBLISH=1`
restores seq_cst for the A/B; `-D MOJOLEARN_ET_RELAXED_PUBLISH=1` opts another
column in for a measurement. Sabotage `-D MOJOLEARN_ET_SAB_RELAXED_PUBLISH=1`
adds one to every count the relaxed path publishes.

Why no bit moves: the ordering changes fences, never the read-modify-write; an
`atomicrmw add`, `min` or `max` loses no update under any ordering and an
integer sum, min or max is the same value under any interleaving; no thread of
a launch reads a published cell, the readers are later launches on the same
in-order queue. The random forest builder pinned its histogram adds to RELAXED
long ago (`ensemble/decisiontree/batched_levelalgo/bins.mojo`); this file never
did, which is why ExtraTrees trained 30x slower than RandomForest on the same
rows.

**DEVIATION 3020, stays an arm.** `MOJOLEARN_ET_SEARCH_RPT_{1,32,64,128,256}`
join DEVIATION 2020's rows-per-thread arms. Under seq_cst the tile divided a
per-block cost (taxi 16 trees: 16.5 s at 1 row per thread, 1.45 s at 64); under
the relaxed publish it is worth 3 to 5 percent (0.89 s to 0.84 s taxi, 2.53 to
2.45 s Istella-S) and costs the level loop a restage and a drain, so the default
stays 1. Bit-inert (integer sums, key-space fold, keyed draws); its own
sabotage `MOJOLEARN_ET_SAB_RPT_TAIL_DROP` moved 15 of 15 ExtraTrees cells.

**DEVIATION 3021, stays an arm.** `-D MOJOLEARN_ET_PLAIN_SINGLE_PUBLISH=1`: a
node that fits in one search block publishes with plain stores (one writer).
Inside the noise once 3022 landed (taxi 0.84 s without, 0.88 s with; Istella-S
2.45 and 2.41 s). Sabotage `MOJOLEARN_ET_SAB_PLAIN_PUBLISH`.

Measurement-only arms, wrong on purpose or vendor-unsafe, never defaults:
`MOJOLEARN_ET_EXP_RACY_RANGE_PUBLISH` (plain read-modify-writes, loses updates;
it priced the range pass: 16.5 s to 10.0 s at 16 trees),
`MOJOLEARN_ET_EXP_NO_SEARCH_BARRIER` (16.5 s to 15.1 s: the barriers were not
the cost).

## What was measured first (reconstructed; pod 6x6vfh2zqas3n4)

Attribution on main `86d33fcdf`'s kernels (`leg_out/profile/main.*`), taxi
4.0M x 16, ET classifier 100 trees depth 16 sqrt features, IDENTICAL, one
process. Untimed fits 94.5, 96.3, 97.3 s (hash df80db81082e30). Stage ledger
(`MOJOLEARN_STAGE_TIMES=1`, drains per stage, a split not a timing):

| stage | seconds |
|---|---:|
| setup (buffers, row fill, workspace) | 0.30 |
| stage + feature sampler | 0.91 |
| range pass (init+range+decode+nonconst) | 39.53 |
| score pass (init+score+finalize) | 56.77 |
| candidate+reduce+splits readback | 0.06 |
| partition (4 kernels) | 0.82 |
| leaf pass | 0.23 |
| host (split records, pop, queue push) | 0.08 |
| total | 98.69 |

nsys on the same fit: the score and range kernels were 99.5 percent of device
time, at a constant 1.4 to 1.9 ns per thread from the root to the deepest
level while the partition pass tiled the same rows and finished in under 1 ms;
the cost was per block, and the block's tail is thread 0's publish atomics.
Istella-S 2.0M x 220 ET classifier 100 trees: 156.8 s. RandomForest on the
same rows for scale: taxi 2.87 to 2.93 s, Istella-S 3.72 to 5.73 s.

The Python side is not the cost: `BOUNDARY_PYTHON binding_ms` equals the fit
within 0.1 percent; the dataset upload is 0.04 s (taxi) after the first fit.

## Identity (done at the tip, pod 6x6vfh2zqas3n4, `leg_out/identity/`)

`tools/identity_break.py --lanes rf-clf,rf-reg,et-clf,et-reg,rf-clf-entropy-log2-noboot,rf-reg-poisson,rf-reg-gamma-ig,et-clf-entropy-bestfirst,rf-score-weighted --fixtures base,ties,odd,dupes,wide --repeats 2`,
BEFORE = main's kernels (MOJOLEARN_COMMIT 5da93bb53, whose tree only adds the
tools), AFTER = the tip 764682f15, both built on the pod (`final_mtimes_sha.txt`:
source 20:35:51Z, GPU binding 20:36:35Z, host binding 20:36:52Z, FAST binding
20:37:59Z). The five N/A cells are `rf-score-weighted`'s scalar `score()`.

| diff | verdict |
|---|---|
| before-cuda vs after-cuda | IDENTICAL=40, N/A=5 |
| before-cpu vs after-cpu | IDENTICAL=40, N/A=5 |
| after-cuda vs after-cpu | IDENTICAL=40, N/A=5 |
| after-cuda vs sabotage-cuda (`MOJOLEARN_ET_SAB_RELAXED_PUBLISH=1`) | DIVERGENT=15 (every et-clf, et-clf-entropy-bestfirst, et-reg cell; infer, model and batch parts), IDENTICAL=25 (the rf lanes, unreached), N/A=5 |
| before-cuda vs sabctl-cuda (sabotage define plus `MOJOLEARN_ET_SEQCST_PUBLISH=1`) | IDENTICAL=40: the sabotage sits inside the relaxed path only |
| before-cuda vs rpt64-cuda (`MOJOLEARN_ET_SEARCH_RPT_64=1`) | IDENTICAL=40 |
| before-cuda vs plain-cuda (`MOJOLEARN_ET_PLAIN_SINGLE_PUBLISH=1`) | IDENTICAL=40 |

The intermediate run at 7c78cd7ba (`leg_out/identity_intermediate/`) read the
same, with the earlier sabotage (`MOJOLEARN_ET_SAB_RPT_TAIL_DROP`) MOVED=15.

## Speed, IDENTICAL, before vs after (interleaved, one process per fit)

Done on pod 6x6vfh2zqas3n4 (`leg_out/fab_*`), model hashes equal across arms
in every cell, 5 rounds, spreads inside the 1.10 gate:

| cell | before median ms (min..max) | after median ms (min..max) | after/before |
|---|---:|---:|---:|
| et-clf taxi 200,000 x 16, 100 trees | 4890.7 (4816.1..5026.0) | 328.0 (323.0..332.9) | 0.0671 |
| et-reg taxireg 4,000,000 x 16, 16 trees, 1.0 features | 44851.8 (44546.0..47136.9) | 4725.3 (4720.4..4734.7) | 0.1054 |

The full-size 100-tree ET A/B on taxi 4.0M and Istella-S 2.0M (`fab_et`) was
stopped before its first sample (`fab_et/STOPPED.txt`); the resume pod below
carries it. Single-process samples at 16 trees from the arm screen
(`quick_*.json`, hashes equal): taxi 16.5 s before, 0.89 s after; Istella-S
28.4 s before, 2.53 s after.

RandomForest, IDENTICAL against the FAST build of the same commit (`fab_rf`,
5 rounds of 2 fits, hashes equal, quality equal):

| cell | IDENTICAL median ms | FAST median ms | FAST/IDENTICAL |
|---|---:|---:|---:|
| rf-clf taxi 4.0M x 16, 100 trees | 2837.1 (spread 1.022) | 2837.9 (1.027) | 1.0003 |
| rf-clf Istella-S 2.0M x 220 | 4439.0 (spread 1.393, u) | 4399.3 (1.409, u) | not quoted |

The Istella-S rf cell is `u` because every process's FIRST full-size fit costs
about 1.4 s more than its second (5.07 s then 3.70 s, both arms alike): a
per-process first-touch cost at 1.76 GB of X that the 20,000-row warmup does
not cover. The resume pod's harness warms up at full size for that reason
(`--warm-rows`).

## The ExtraTrees regressor at 377.8 s (attributed)

The orchestrator's `prepare3.log` (`~/mojolearn-evidence/forest-groves-row/final_pull_pause/out/`)
fitted `rf-taxi-100x16` in 4.8 s, `et-taxireg-100x16` (5,250,086 x 16) in
377.8 s and `rf-istella-100x16` in 5.1 s on main's default build. No ET
CLASSIFIER was fitted in that log; the "RF and ET classifiers take about 5 s"
premise is RandomForest only. Main's ET classifier on taxi 4.11M rows takes
94.5 s (above). The regressor's factor over that is the search width:
`tools/speed_gbdt_arm.py::max_features_for` gives a regressor 1.0 (all 16
columns per node) and a classifier sqrt (4 of 16), and the seq_cst publish cost
scales with columns searched, so 94.5 s x 4 columns x 1.28 rows = 484 s is the
expected order and the measured 377.8 s is inside it (the same log's RF
classifier grew 3.44M nodes at 100 trees, the ET regressor 3.62M; the node
count is not the factor, the per-node search width is). Same code path (the
regressor is the score kernel with `n_acc = 1` and fixed-point labels), no
defect; the resume pod's stage ledgers for et-clf and et-reg at the same rows
and tree count confirm or deny this below. Under
DEVIATION 3022 the same 16-tree regressor fit went 44.85 s to 4.73 s (table
above), so a 100-tree fit is about 30 s on this box.

## Owed (the resume pod carries these)

1. Interleaved A/B, 100 trees, ET classifier taxi 4.0M and Istella-S 2.0M,
   arms before / after / after-FAST, 5 rounds, with the geometric mean.
2. The cost-of-pinning table: IDENTICAL vs FAST of the same commit for ET and
   RF, classifier and regressor, taxi and Istella-S, with the stage ledger in
   both tiers and the ceiling at zero pinning cost.
3. The attribution table of AFTER (stage ledger and nsys).
4. Apple and AMD columns: owed at the next release, not taken now (the flip is
   NVIDIA only; `checks/kernel_matrix.mojo` rows unchanged).

## The resume pod (2026-09-18 01:37Z to 03:15Z, RTX 4090; landed by the orchestrator)

The agent that ran this pod stalled before writing these in; the orchestrator
read them from `~/mojolearn-evidence/forest-train-speed/leg_out2/`
(`resume.out`, `rab_*/summary.txt`, `rab_et.before-arm-missing/`). The pod is
reaped. The lane is landed at `f1fe7057c`; commit `70797b6cf` (DEVIATION 3023,
the regression score pass at width 4) was NEVER BUILT OR RUN and stays on
`lane/forest-train-speed` as a candidate.

**Owed item 1, the 100-tree ET A/B at full size: INCOMPLETE.** The AFTER and
AFTER-FAST arms ran five rounds each on Istella-S 2,000,000 x 220
(`rab_et.before-arm-missing/`): after 4530.3 ms (4514.1..4783.0, spread
1.060), after-FAST 4443.0 (4435.0..4530.6, 1.022), model hashes equal across
the ten fits. The BEFORE arm did not run (its build was missing in that
stage), the three-arm rerun (`resume2.out`) was in flight when the agent
stalled, and the pod ended. For scale: the 2026-09-12 H100 board fitted the
same ET classifier on Istella-S in 9291 ms and this box's main took 28.4 s
at 16 trees (single process, above). The BEFORE-vs-AFTER ratios of record
stay the ones above (200,000 x 16 rows at 100 trees, 0.067; 4,000,000 x 16
at 16 trees, 0.105; hashes equal), plus the single-process 16-tree samples on
both datasets.

**Owed item 2, the cost of pinning (`rab_rf_pin`, `rab_etreg_pin`, 5 rounds
of one fit per process, full-size warmup):**

| cell | IDENTICAL ms (spread) | FAST ms (spread) | FAST / IDENTICAL | hashes |
|---|---:|---:|---:|---|
| rf-clf taxi 4.0M x 16, 100 trees | 2837.3 (1.002) | 2823.3 (1.020) | 0.995 | equal |
| rf-clf Istella-S 2.0M x 220 | 3762.3 (1.052) | 3705.6 (1.052) | 0.985 | equal |
| rf-reg taxireg 4.0M x 16 | 6291.0 (1.005) | 6276.0 (1.063) | 0.998 | differ (FAST RF keeps its own hash) |
| rf-reg Istella-S 2.0M x 220 | 41727.5 (1.099) | 41059.0 (1.102 u) | not quoted | differ |
| et-reg taxireg 4.0M x 16, 16 trees | 4661.3 (1.001) | 4664.0 (1.001) | 1.001 | equal |
| et-reg Istella-S 2.0M x 220, 16 trees | 27065.4 (1.001) | 27047.4 (1.004) | 0.999 | equal |

FAST is IDENTICAL within noise on every forest cell. The ceiling if every
pinned seam cost zero is therefore about 1.0x: the pinning is not where the
time is, and a FAST tree tier can only gain by a different algorithm, not by
relaxing the pins. Same verdict as lane/gbdt-train-speed for boosting.

**Owed item 3, the AFTER attribution:** the `nsys` and stage-ledger profiles
ran (`profile*/`, `prof_*.out`); they are pulled and not tabulated here.
