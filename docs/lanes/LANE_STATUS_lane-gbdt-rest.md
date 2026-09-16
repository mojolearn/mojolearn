# Lane status: lane/gbdt-rest (2026-09-15)

## HELD 2026-09-16: THE FIXTURE SHRINK REACHES THIS LANE. RECORD NOTHING YET.

**Do not take this lane's Metal column, and do not reuse ANY GBDT column or
sabotage verdict taken before the shrink lands.** A sabotage arm proven live at
one fixture size can be INERT at another, so a pre-shrink verdict is not
evidence for the post-shrink lane.

WHY, and why the earlier "safe to record" answer was withdrawn. On the morning
of 2026-09-16 `docs/lanes/FIXTURE_SHRINK_SCOPE.md` (branch
`lane/identity-fixtures-light`) put the GBDT family in bucket B, "leave big",
on the reasoning that the family's cost was the Metal command-queue leak rather
than fixture size. **That reasoning was measured and found wrong, and the
bucket B decision is CANCELLED.** The queues were an Apple SYSTEM SERVICE, not
our processes: measured directly, one of our lanes held a single queue while
DockHelper held thousands. (Corroborated here independently: with our own gp
lane holding the GPU, `ioreg -l | grep -c AGXCommandQueue` read 1.)

With the leak explanation gone, the finished Apple column was read directly and
four GBDT lanes are among the TEN MOST EXPENSIVE IN THE WHOLE RECORD:

| lane | Apple column |
|---|---|
| `gbdt-parametric-losses` | 1701 s |
| `gbdt-nan-modes` | 969 s |
| `gbdt-lossguide-newtoncosine` | 838 s |
| `gbdt-pair-logit` | 794 s |

The fixture-shrink lane is now cutting all four, with authority to choose the
sizes, targeting under 30 s each.

**WHAT THIS HOLD COSTS IF IGNORED:** those four lanes alone are over an hour on
Apple, and a column recorded now is a column for fixtures that no longer exist.

### This lane's specific exposure

- Item 1's own lane is `gbdt-query-softmax`, which is NOT in the cut list, but
  the lane's owed evidence includes a BASE-FIXTURE SPOT CHECK of ten existing
  GBDT lanes, and **two of those ten, `gbdt-pair-logit` and
  `gbdt-parametric-losses`, are being cut.** That spot check must be retaken
  after the shrink, against a post-shrink committed column.
- `bench/results/identity_break/2026-09-15_gbdt-query-softmax/cpu-apple-m4.json`
  (the committed CPU column, cells=3 stable=3) was taken at PRE-SHRINK
  fixtures. Re-read the fixtures before trusting it as the CPU half of a diff.

### State on 2026-09-16: NOTHING WAS RECORDED

The Metal column was **not** taken. The evidence chain was launched at 05:47,
was starved of the shared Metal lock for about ten minutes by another agent's
jobs, and was retired having written **zero JSON** (only lock-wait messages in
`metal-new.log`). No sabotage build was produced. So there is no partial or
tainted GBDT evidence anywhere from that attempt, and nothing is owed to a box.

### What survives the hold and does NOT need redoing (fixture independent)

- The Metal identical GBDT binding built from this tree, sha256
  `88055dbf6d34c0bbdd112a5654ed0a3a628fc6b6accc2298e476cefbb2bc5362`, and the
  GBDT host binding `26b6a71ee3390be2f5982b0d80365be06f9670b7d38bd6c01c487afb2274b2c5`,
  both cached under the session scratchpad `bins/softmax2/`.
- **The refusal trap is cleared.** The worktree now carries a COMPLETE
  `python/mojolearn/identical/` set (17 bindings), verified by IMPORT (not
  `nm`): `transpose_f32` is present. That is what made the first attempt read
  REFUSED on every cell.
- The host sabotage build is still OWED; it must be built and SEEN to diverge
  **at the post-shrink sizes**.

### Resume, after the shrink lands

1. Re-read this lane's fixtures in `tools/identity_break.py` (do not assume the
   sizes in the commands below still hold).
2. Re-run the whole evidence chain with a FRESH tag, so no pre-shrink artifact
   is mixed in: `bash <scratchpad>/rest_evidence.sh <fresh-tag>`.
3. All seven owed items below are still owed, in full.


Andrew, Sep 15 2026: "gbdt as a lane". Work through `gbdt/NOT_IMPLEMENTED.tsv`
(28 rows on origin/main at a76c02d27) for the CatBoost-reference GBDT:
implement the user-facing items, each bitwise identical on the GPU and the
CPU verifier (with public CPU inference where the model changes), and give
every remaining row a precise reason. Reference tree:
`/Users/andrewhendel/CascadeProjects/upstream/catboost` (7055d33d).

Worktree `scratchpad/wt-gbdt-rest`, branch `lane/gbdt-rest`, main only (0.8.7).
Evidence per item is small fixtures only (memory scope-lane-evidence-small).

## Triage

Classes: (a) a user-facing feature a CatBoost user would set (a loss, a
metric, an option or an input type); (b) internal reference plumbing this
implementation does not need; (c) intentionally excluded (dead or unreached in
the reference, nondeterministic by nature, or a reference bug).

Counts: **(a) 10, (b) 13, (c) 5.**

| # | Row (their file, symbol) | Class | Decision and reason |
|---|---|---|---|
| 1 | multilogit.cu RMSEWithUncertainty | a | Loss. Trained in the reference by `TMultiClassTrainer` (train_lib/multiclass.cpp:8), a two-plane approx on a one-column target. PLANNED, item 4. |
| 2 | multilogit.cu MultiCrossEntropy | a | Loss (multilabel). `TMultiClassTrainer` (multiclass.cpp:11); needs a 2D label input. PLANNED, item 5. |
| 3 | multilogit.cu MultiRMSE | a | Loss (multi-target regression). `TMultiClassTrainer` (multiclass.cpp:13); needs a 2D label input. PLANNED, item 3. |
| 4 | multilogit.cu BuildConfusionMatrixBins | a | A GPU METRIC kernel (Accuracy, Precision, Recall, F1 at gpu_metrics.cpp:304-326). GradientBoosting has no `eval_metric` or `custom_metric` surface (`git grep eval_metric` in python/mojolearn and gbdt is empty), so there is nothing for it to serve; it waits on that surface. |
| 5 | pointwise_targets.cu TNumErrorsMetric | c | In the kernel switch, refused by their own Init: unreachable in training. |
| 6 | pointwise_targets.cu MseImpl | c | `ApproximateMse` has no caller in catboost/. |
| 7 | bootstrap.cu GammaBootstrapImpl | c | Its only call is commented out (bootstrap.cu:82). |
| 8 | segmented_sort.cu SortPairsDescending arm | c | No caller passes compareGreater=true. |
| 9 | leaves_estimation_helper.h MakeSupportPairsMatrix, ReorderPairs, FilterZeroLeafBins | b | Plumbing of the pairwise oracle (pairwise_oracle.h), which only the pairwise trainer family uses (row 11). Follows row 11. |
| 10 | targets/kernel/ query_softmax.cu, pfound_f.cu, query_cross_entropy.cu, dcg.cu | a | Split by what trains each. **QuerySoftMax**: the querywise trainer QueryRMSE already runs on (train_lib/querywise.cpp:7, querywise_targets_impl.h:242-257, pointwise oracle). PLANNED, item 1. **QueryCrossEntropy**: `TPairwiseGpuTrainer<TQueryCrossEntropy>` (train_lib/query_cross_entropy.cpp:5-6). **pfound_f.cu** is `TPFoundF`, the target of the loss **YetiRankPairwise** (train_lib/pfound_f.cpp:5-6), not a PFound loss; PFound itself is only a metric. Both need the pairwise GPU learner (pairwise structure searcher, pair matrices, pairwise oracle), which is not implemented. **dcg.cu**: NDCG/DCG GPU metric kernels, which wait on the eval metric surface of row 4. |
| 11 | pair_logit.cu PairLogitPairwiseImpl, RemoveOffsetsBiasImpl | a | Loss PairLogitPairwise, `TPairwiseGpuTrainer<TPairLogitPairwise>` (train_lib/pair_logit_pairwise.cpp:5-6): the pairwise learner, not implemented. |
| 12 | pairs/util.cpp max_pairs | a | Option of PairLogit (loss_description.cpp:230-232). PLANNED, item 7. |
| 13 | query_helper.cu FillQueryEndMask, CreateSortKeys, FillTakenDocsMask, ComputeGroupMax, offsets-only ComputeGroupMeans | b | Reached only through `TQuerywiseSampler` (querywise_helper.cpp:28, :90), held only by QueryCrossEntropy (query_cross_entropy.h:169-191) and TPFoundF (pfound_f.h:133-141, pfound_f.cpp:96 RemoveQueryMax). It is those pairwise targets' query sampler, not a user bootstrap. Follows rows 10 and 11. |
| 14 | overfitting_detector.cpp TOverfittingDetectorWilcoxon | a | Reachable only by spelling the enum (enums.h:10); their Python docs list IncToDec and Iter only (python-package core.py:4974-4977). The statistic is `NStatistics::Wilcoxon` in library/cpp/statistics/statistics.h, which IS in the reference checkout (the old reason implied it was unavailable). Lowest value; not planned. |
| 15 | pointwise_hist1.cu ComputeHist1 | c | Registered, never called. |
| 16 | compute_by_blocks_helper.h TComputeSplitPropertiesByBlocksHelper | b | Histogram block regrouping; a performance layer. Note kept: can move score ties on fold-heterogeneous one-byte policies. |
| 17 | compute_by_blocks_helper.cpp Rebuild, StreamCount, ForceOneBlockPerPolicy | b | Stream and block configuration; one Metal queue. |
| 18 | gather_bins.cu GatherCompressedIndex | b | Multiclass load policy; same histogram values. |
| 19 | hist_single_leaf.cuh TComputeSingleHistKernel | b | Root-level kernel family; same arithmetic. |
| 20 | feature_layout_doc_parallel.h CreateFeaturesMapping shuffle | b | Feature grouping for kernel specialization. |
| 21 | feature_layout_feature_parallel.h TFeatureParallelLayout | b | Collapses to doc-parallel at one device. |
| 22 | feature_layout_single.h TSingleDevLayout | b | One device. |
| 23 | feature_layout_common.h SkipInSplitSearch, SkipFirstBinInScoreCount | b | Driven by exclusive feature bundles (binarizations_manager.h:90, helpers.cpp:158), set by the data layer; the only surface is `dev_efb_max_buckets` (oblivious_tree_options.cpp:26), a dev option. |
| 24 | compressed_index.h TCompressedDataSet container | b | One flat buffer at one device. |
| 25 | doc_parallel_dataset.h TDocParallelDataSet | b | Bundling class; arguments carried separately. |
| 26 | doc_parallel_dataset_builder.cpp | b | Its one decision (blockSize 1) is pinned. |
| 27 | catboost_options.cpp:357-359 l2 0 to 1e-20 | a | Option semantics of an explicit `l2_leaf_reg=0`. PLANNED, item 6; it may move recorded l2=0 cells, which are named in the row. |
| 28 | loss_description.cpp YetiRank permutations, decay | a | Loss parameters. PLANNED, item 2, with CatBoost's `Loss:key=value` spelling. |

## Plan (order of user value)

1. QuerySoftMax (params lambda 0.01, beta 1.0; Gradient leaves at 100
   iterations by default, catboost_options.cpp:100-105).
2. The ranking loss parameters, as wrapper keywords beside the existing
   `loss_q` and `loss_delta` (YetiRank permutations and decay,
   QuerySoftMax lambda and beta). DONE with item 1.
3. RMSEWithUncertainty. 4. MultiRMSE. 5. MultiCrossEntropy.
6. l2_leaf_reg=0 substitution. 7. max_pairs.

**Items 3 and 4 were REORDERED after reading both sides (2026-09-15).**
MultiRMSE and MultiCrossEntropy take a genuinely two-dimensional target:
the reference sets `NumClasses = TargetData->GetTargetDimension()`
(`multiclass_targets.h:155`), while `y` is one-dimensional in every layer
here (`as_f32_c(y, ndim=1)` in the wrapper, one float32 buffer across the
ABI, `train(y: List[Float32])`), so each needs a new n_rows x targetCount
target surface through the wrapper, both bindings, `train`, the target
upload and both oracles. RMSEWithUncertainty does not: it keeps the
one-dimensional target and only widens the approx to two planes
(`NumClasses = 2`, `multiclass_targets.h:158-160`), and the saved-model
path already carries that -- `predict` branches on `approx_dim_ > 1`
rather than on the loss, `model_text` stores `dim` per weak model, and the
host loader reads `approx_dim` from the text and uses the loss name only
for `is_classifier`. Its extension points are therefore the objective
code, the `approx_dim` rule (`gbdt/train.mojo:1971-1975`), the oracle's
`cursor_dim` / `single_bin_dim` (`pointwise_oracle.mojo:1081-1101`), the
two kernels (`RMSEWithUncertaintyValAndFirstDerImpl` and the
`SecondDerRow` rows: row 0 the weight, row 1 `[0, 2 w miss^2 exp(min(-2 a1,
70))]`) with a host twin, and the predict shape rule.

Not planned in this lane, with reasons above: the pairwise learner family
(QueryCrossEntropy, PairLogitPairwise, YetiRankPairwise, rows 9, 10, 11, 13),
the GPU metric kernels (rows 4, 10) behind a missing eval metric surface, and
Wilcoxon (row 14).

## STATE (2026-09-15, read this first)

Item 1 (QuerySoftMax and the ranking loss parameters) is IMPLEMENTED and
PROVEN ON CPU ONLY. Its Metal column is OWED and so is everything that
compares against it. Nothing from this lane has been merged to main.

The Apple M4 Metal GPU was degraded that evening by a command-queue leak
(AGXCommandQueue at 6754 against a limit of 512, about 6400 with no live
creator process); every GBDT Metal fit ran roughly 20x slow and a machine
restart was the clearing action. The coordinator stopped Metal work before the
restart, so the Metal half of the proof was deliberately not taken. Resume it
on a healthy machine with the commands below.

- Branch: `lane/gbdt-rest`; its tip carries item 1. Base: origin/main
  `a76c02d27`. Triage commit: `2355b6a27`.
- Record so far: `bench/results/identity_break/2026-09-15_gbdt-query-softmax/`
  (`cpu-apple-m4.json` plus a README that states plainly what is owed).
- Drivers and raw logs, outside the repo because /private/tmp does not
  survive a restart: `~/mojolearn-evidence/gbdt-rest-2026-09-15/`
  (`rest_evidence.sh`, `run_lanes_rest.sh`, `build_gbdt_rest.sh`,
  `query_softmax_reference.py`, the two build logs, the attempt's logs).
- Bindings built from this tree (sha256): Metal GBDT
  `88055dbf6d34c0bb...`, GBDT host `26b6a71ee3390be2...`. They live in the
  scratchpad and are probably gone after the restart; rebuild them.

## THE TRAP THAT WASTED THE FIRST ATTEMPT

The first Metal column read REFUSED on every cell with
`ImportError: the base binding has no transpose_f32`. That is NOT a defect in
this lane: a fresh worktree has no `python/mojolearn/identical/` at all, and
copying only `_mojolearn_gbdt.so` into it leaves the package without the BASE
`_mojolearn` extension, which is where `transpose_f32` lives
(`bindings/_mojolearn.mojo:1647`, added by `485caa24b`). `nm` cannot answer
this question: Mojo `def_function` exports are registered at module init, not
as exported C symbols, so check with an import, not with `nm`. Place a
COMPLETE identical set in the worktree before any Metal column.

## Owed, with the exact commands

Let `WT` be the lane worktree, `HD` a host dir, `HDS` a sabotage host dir.

    # 0. worktree and a COMPLETE binding set (the trap above)
    cd /Users/andrewhendel/CascadeProjects/mojolearn
    git worktree add $WT lane/gbdt-rest
    mkdir -p $WT/python/mojolearn/identical $HD $HDS
    cp /Users/andrewhendel/CascadeProjects/mojolearn/python/mojolearn/identical/*.so $WT/python/mojolearn/identical/
    cp /Users/andrewhendel/CascadeProjects/mojolearn/python/mojolearn/host/_mojolearn_core_host.so $HD/
    cp /Users/andrewhendel/CascadeProjects/mojolearn/python/mojolearn/host/_mojolearn_core_host.so $HDS/

    # 1. this lane's two bindings, one core each (about 350 s and 125 s)
    cd $WT
    MOJOLEARN_NUMERIC_MODE=identical nice -n 19 sh bindings/build_gbdt.sh
    # build_gbdt.sh writes python/mojolearn/identical/_mojolearn_gbdt.so in place
    MOJOLEARN_GBDT_HOST_OUTDIR=$HD nice -n 19 sh bindings/build_gbdt_host.sh
    MOJOLEARN_GBDT_HOST_OUTDIR=$HDS MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_HOST_SABOTAGE=1" \
        nice -n 19 sh bindings/build_gbdt_host.sh

    export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=$WT/python
    export OMP_NUM_THREADS=1 MOJOLEARN_CPU_THREADS=1 MODULAR_THREAD_BUSY_WAIT_US=0
    PY=/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/test/bin/python
    BENCH_PY=/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/bench/bin/python

    # 2. OWED: the Metal column for the new lane (exclusive Metal slot)
    $PY tools/identity_break.py --lanes gbdt-query-softmax --fixtures base,ties,odd \
        --repeats 2 --vendor apple-m4 --json metal-new.json

    # 3. the CPU column (identical/ moved aside so the CPU route is taken)
    mv python/mojolearn/identical python/mojolearn/identical.aside
    MOJOLEARN_HOST_DIR=$HD $PY tools/identity_break.py --lanes gbdt-query-softmax \
        --fixtures base,ties,odd --repeats 2 --vendor cpu-apple-m4 --json cpu-new.json
    mv python/mojolearn/identical.aside python/mojolearn/identical
    # already taken on 2026-09-15 and recorded: cells=3 stable=3 moved=0 refused=0,
    # infer/model/batch stable=3 (bench/results/.../cpu-apple-m4.json)

    # 4. OWED: the diff that is the identity claim
    $PY tools/identity_break.py --diff metal-new.json cpu-new.json \
        --require-columns 2 --lanes gbdt-query-softmax

    # 5. OWED: batch sabotage on Metal, must read batch_moved
    MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1 $PY tools/identity_break.py \
        --lanes gbdt-query-softmax --fixtures base --repeats 1 --vendor apple-m4 \
        --json metal-batchsab.json

    # 6. OWED: the host sabotage column, must read DIVERGENT on every new cell
    mv python/mojolearn/identical python/mojolearn/identical.aside
    MOJOLEARN_HOST_ALLOW_SABOTAGE=1 MOJOLEARN_HOST_DIR=$HDS $PY tools/identity_break.py \
        --lanes gbdt-query-softmax --fixtures base,ties,odd --repeats 1 \
        --vendor cpu-apple-m4 --json cpu-sab.json
    mv python/mojolearn/identical.aside python/mojolearn/identical
    $PY tools/identity_break.py --diff metal-new.json cpu-sab.json --lanes gbdt-query-softmax

    # 7. OWED: the BASE-FIXTURE spot check of the existing GBDT lanes (small on
    #    purpose, memory scope-lane-evidence-small), Metal then CPU, then against
    #    the committed stage 4 Metal columns
    OLD=gbdt-symmetric,gbdt-rmse,gbdt-depthwise,gbdt-lossguide,gbdt-multiclass,gbdt-onevsall,gbdt-parametric-losses,gbdt-query-rmse,gbdt-pair-logit,gbdt-yeti-rank
    $PY tools/identity_break.py --lanes $OLD --fixtures base --repeats 1 \
        --vendor apple-m4 --json metal-old.json
    # CPU: the same with identical/ aside, MOJOLEARN_HOST_DIR=$HD, --vendor cpu-apple-m4
    $PY tools/identity_break.py --diff metal-old.json cpu-old.json
    $PY tools/identity_break.py --diff \
        bench/results/identity_break/2026-09-15_gbdt-yeti-rank/apple-m4.earlier-18-lanes.json \
        bench/results/identity_break/2026-09-15_gbdt-yeti-rank/apple-m4.json \
        metal-old.json

    # 8. OWED: the test route on Metal (the CPU route already reads 21 passed)
    $PY -m pytest python/mojolearn/tests/test_gbdt_query_softmax.py \
        python/mojolearn/tests/test_gbdt_yeti_rank.py \
        python/mojolearn/tests/test_gbdt_query_rmse.py \
        python/mojolearn/tests/test_host_surface.py -q -p no:cacheprovider

    # 9. OWED: the CatBoost 1.2.10 CPU QuerySoftMax comparison (quality only)
    QS=~/mojolearn-evidence/gbdt-rest-2026-09-15/query_softmax_reference.py
    PYTHONPATH=$WT/python:$WT/tools $PY $QS ours ours-reference.json
    PYTHONPATH=$WT/tools $BENCH_PY $QS catboost catboost-reference.json
    $PY $QS compare ours-reference.json catboost-reference.json

Put every json, log and diff under
`bench/results/identity_break/2026-09-15_gbdt-query-softmax/` and rewrite that
README with the measured verdicts, replacing its OWED section.

## Merging item 1, once the Metal half is in hand

    $PY tools/docs_facts.py --check
    $PY packaging/wheel_ci.py pins .
    $PY packaging/wheel_ci.py inventory python/mojolearn
    git merge origin/main        # 6 commits ahead as of a76c02d27
    git push origin HEAD:main    # main only, 0.8.7; then confirm it landed

## Progress

- Triage committed (this file).
- **Item 1 and item 2 done together (QuerySoftMax and the ranking loss
  parameters).** `gbdt/targets/kernel/query_softmax.mojo` (the four reference
  kernels and their launcher), `gbdt/host/gbdt_oracle_query_softmax.mojo` (the
  same order on the host), the querywise wiring in `doc_parallel_boosting`,
  `pointwise_oracle`, `train`, both bindings and `gbdt_oracle_losses`, and
  `loss_lambda` / `loss_beta` / `loss_permutations` / `loss_decay` on
  `GradientBoosting`, carried as a fifth `strs` entry so no numeric ABI tail
  moves. Identity lane `gbdt-query-softmax` (defaults, both QuerySoftMax
  parameters, and a YetiRank fit with its two), tests in
  `test_gbdt_query_softmax`. Rows removed from the TSV: the YetiRank loss
  parameter row, and query_softmax.cu from the ranking family row.
