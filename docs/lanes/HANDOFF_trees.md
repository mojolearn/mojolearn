# Trees lane handoff (branch `lane/trees-identical`, 2026-09-09)

Current scope and persistent project decisions: [TREE_GROWTH_SCOPE.md](TREE_GROWTH_SCOPE.md).

Symmetric GBDT (CatBoost mirror), RF (cuML mirror), ET. Our IDENTICAL arm
against each opponent's FAST arm on NVIDIA. Wind-down ordered by the
orchestrator before tasks 4 and 5 were measured on the H100; everything
below is either measured with a log path or marked not run.

## What landed (commits, `%h parent %p`)

- `609747b5 parent f3d76e8d` tools/trees_leg.sh (guarded RunPod session:
  rent, arm the on-pod watchdog first, ship source archive, ssh, extend,
  reap+verify) and tools/trees_identical_remote.sh (pod setup: pixi,
  IDENTICAL bindings base/gbdt/rf/trees, CatBoost, cuML, LightGBM built
  with USE_CUDA=ON, HIGGS).
- `4ccccde6 parent 609747b5`
  - core/pinned_reduce.mojo: `two_phase_halving_sum`, the same halving
    tree (same additions, same order) in 3 barriers instead of
    log2(block)+2; `halving_block_sum` calls it.
  - gbdt/targets/kernel/pointwise_targets.mojo: its IDENTICAL
    `pinned_block_sum` arm imports that fold instead of carrying a copy.
  - ensemble/randomforest.mojo, ensemble/decisiontree/batched_levelalgo/
    builder.mojo, .../kernels/builder_kernels_impl.mojo: ROWS_SORTED_SAMPLE,
    HIST_ITEMS_PER_THREAD, HIST_SMEM_COPIES_DEFAULT selected by
    `-D MOJOLEARN_2010_ROWS_SORTED=1`, `-D MOJOLEARN_2011_HIST_ITEMS4=1`,
    `-D MOJOLEARN_2012_SMEM_COPIES4=1`; shipped values unchanged.
  - bindings/build_rf.sh: MOJOLEARN_EXTRA_DEFINES pass-through.
  - tools/speed_gbdt_arm.py, bench/speed/forest_speed_arm.py: dataset
    `higgsreg` (HIGGS label as a float target, the RMSE cell).
  - tools/trees_identical_ab.sh: on-pod A/B helper (named binary sets,
    identity_break fingerprints and diffs, speed cells, RF reach gate).
- `a9ba6818 parent 4ccccde6` L40S artifacts, HIGGS via UCI static zip in
  the setup body, and bench/results/trees_identical/patches/
  kernel_matrix_single_pass_identical_nvidia.patch (checks/kernel_matrix.mojo
  is NOT edited; the kNN lane owns it).
- this commit: H100 artifacts and this file.

## Boxes

- L40S: NVIDIA L40S, driver 580.126.09, CUDA 12.4 image, catboost 1.2.10,
  lightgbm 4.7.0 (USE_CUDA=ON, probe ok), cuml 26.08.00, numpy 2.4.6;
  bench/results/trees_identical/l40s_2026-09-09/logs/{versions,gpu,setup}.txt.
- H100: NVIDIA H100 80GB HBM3, driver 580.126.09, same image and versions;
  bench/results/trees_identical/h100_2026-09-09/logs/{versions,gpu,setup}.txt.
  Source commit a9ba6818 on the box (setup.txt).

## Fingerprints (identity_break, IDENTICAL tier, 9 hostile fixtures x2 repeats)

Files: bench/results/trees_identical/{l40s,h100}_2026-09-09/ib/.
- baseline NVIDIA L40S vs Apple M4 JSON (bench/results/identity_break/
  apple-m4.identical.json, commit e616906e): 81/81 cells IDENTICAL for
  rf-clf, rf-reg, et-clf, et-reg, gbdt-symmetric, gbdt-depthwise,
  gbdt-lossguide, gbdt-rmse, kmeans (run locally, JSON diff only).
- fold (two_phase_halving_sum, base+gbdt rebuilt): 81/81 equal to baseline
  on both boxes (ib/diff.baseline.fold.txt).
- fused (`-D MOJOLEARN_2030_FUSED_EST_MOVE=1`): 36/36 equal (4 gbdt lanes)
  on both boxes (ib/diff.baseline.fused.txt).
- sp (fold + fused + the kernel_matrix patch): 36/36 equal on H100
  (h100 ib/diff.baseline.sp.txt). Note the patch's arm only runs above
  500,000 rows per leaf, which the 20,000-row fixtures never reach; the
  1M/2M prediction hashes below are the witness for that arm and they were
  NOT run (wind-down).
- rf2010 / rf2011 / rf2012: 18/18 equal each (rf-clf, rf-reg) on both boxes.
- rf_perf_candidates_check (reach gates, identical define, source and each
  candidate define): ALL ARMS GREEN on both boxes, logs/rfgate.*.log.

## Task 1: profile and reference table (IDENTICAL baseline, depth 6, 100 iters, lr 0.1, l2 1, 254 borders, no bagging, Plain)

Medians of the timed rounds, ms, Python surface wall (fit only, scoring
outside the timer). Logs under bench/results/trees_identical/<box>/speed/.

| box | cell | ours IDENTICAL | opponent (FAST) | log |
|---|---|---|---|---|
| L40S | symmetric Logloss higgs 1M, 7 rounds | 426 | catboost-gpu 781 | l40s .../speed/profile_cells_console.txt (console capture; the log files expired with the pod) |
| L40S | symmetric RMSE higgsreg 1M, 7 rounds | 318 | catboost-gpu 915 | same |
| H100 | symmetric Logloss higgs 1M, 7 rounds | 775 (697..1161) | catboost-gpu 900 (864..939) | h100 .../speed/baseline.gbdt-symmetric.higgs.r1000000.full.log |
| H100 | symmetric RMSE higgsreg 1M, 7 rounds | 806 (765..1313) | catboost-gpu 699 (680..759) | h100 .../speed/baseline.gbdt-symmetric.higgsreg.r1000000.full.log |
| H100 | rf higgs 1M, 5 rounds (100 trees, depth 16, sqrt, 128 bins, bootstrap) | 5762 (5116..6451) | cuml-rf-gpu 3314 (3257..3950); lightgbm-cuda 469654 (468691..471303, 3 rounds then budget) | h100 .../speed/baseline.rf.higgs.r1000000.full.log |

Quality on the same 500k test tail: Logloss ours 0.542067 / AUC 0.800716,
CatBoost GPU 0.542524 / 0.800431; RMSE ours 0.429173, CatBoost 0.429220.
Our hash is constant across rounds on both boxes (dac2cf366e219cec Logloss,
037aa3188d25c889 RMSE, the same on L40S and H100); CatBoost GPU's Logloss
hash changes every round.

H100 caveat: the Mojo fit wall inside those rounds is 251 ms (Logloss) and
169 ms (RMSE) per the stage clock, so 450-600 ms of every H100 round is
outside the Mojo fit (host-side Python/NumPy prep on a 208-core NUMA box
that prints tcmalloc mbind warnings); on the L40S the same gap is ~200 ms.
The opponent rows are measured on the same box in the same process, so
the ratio is still like for like, but the host tax is where the H100 rows
go before the device does.

Per-phase profile (MOJOLEARN_STAGE_TIMES=1, one untimed replicate, drains
per stage so it is not a benchmark), symmetric depth 6 higgs 1M:

| stage | L40S Logloss | H100 Logloss | H100 RMSE |
|---|---|---|---|
| sym.hist | 48.7 | 67.7 | 68.0 |
| sym.pstats | 9.7 | 16.0 | 16.0 |
| sym.score | 10.6 | 14.8 | 14.6 |
| sym.winner | 4.3 | 6.0 | 5.7 |
| sym.split | 39.2 | 45.5 | 45.3 |
| sym.drain | 1.0 | 1.5 | 1.5 |
| sym.leaves | - | - | 6.1 |
| est.move | 11.2 | 13.1 | - |
| est.approx | 15.8 | 16.6 | - |
| est.pstats | 26.5 | 22.8 | - |
| est.readback | 9.3 | 13.4 | - |
| accounted / fit wall | 176 / 212 | 217 / 251 | 157 / 169 |

Logs: <box>/speed/baseline.gbdt-symmetric.{higgs,higgsreg}.r1000000.stage.log
(L40S: profile_cells_console.txt). The Newton walker (est.*) is 63 ms of
251 on the H100; sym.hist and sym.split are the two largest phases.

RF 1M H100 quality (same 500k tail): ours logloss 0.538850 / AUC 0.809906,
cuml-rf-gpu 0.538814 / 0.809834, lightgbm-cuda 0.638510 / 0.754274 (rf
boosting with the 0.632 bagging LightGBM forces; 470 s per fit, so it
exhausts the 1800 s per-arm budget after round 3). Our hash is constant
(3ffa2951595422d4); cuML's is constant on this box too (a372de7ab27df595).
ET at every rung, RF 2M/5M: not run (wind-down; the ET 1M cell was killed
at its warm-up and its partial log was not kept).

H100 batch status at wind-down (logs/batchH.sh, logs/ab.txt): setup 16:37
to 16:44; PHASE_BUILDS (9 builds, 7 fingerprint sets, 4 reach gates) done
16:51; PHASE_TABLE_1M2M 5 of 8 cells done at 17:28 when the batch was
stopped (the lightgbm-cuda rf arm took 36 min of it); PHASE_SYM_AB (13
cells), PHASE_RF_AB (8 cells), the 5M rungs and the final stack (6 cells)
not started. Pod r8jua8lx0j0urj terminated 17:29 UTC, verified 404.

## Task 2: DEV 2030 fused walker move (`-D MOJOLEARN_2030_FUSED_EST_MOVE=1`)

Fingerprints equal (36/36, both boxes). Timing, L40S, ours only, medians
of 7 rounds at 1M and 5 at 2M (bench/results/trees_identical/
l40s_2026-09-09/speed/ was lost with the pod; the numbers are the FSPEED
lines read from the pod before it expired, transcribed here):
baseline 411 / 585, fold 336 / 570, fused (= fold + 2030) 322 / 557.
Fused is faster than fold at both rungs (4% / 2%), hashes equal
(dac2cf366e219cec at 1M on every set). H100 A/B: NOT RUN (wind-down hit
before PHASE_SYM_AB). NOT FLIPPED: the default stays off because the
only timing is from the L40S and its logs did not come home; flip it with
the H100 rerun below (one-line change in
gbdt/methods/leaves_estimation/pointwise_oracle.mojo, FUSED_EST_MOVE_2030).

## Task 3: halving_block_sum in three barriers

Landed (4ccccde6) and verified: 81/81 fingerprints equal on both boxes,
including kmeans through the base binding. L40S ours-only Logloss:
baseline 411 -> fold 336 ms at 1M, 585 -> 570 at 2M (same transcription
caveat as task 2). H100 timing not run.

## Task 4: RF candidates

Builds, fingerprints (18/18 each) and reach gates green on both boxes.
Timing at 1M/2M: NOT RUN on either box (the L40S batch expired in the RF
cells; the H100 wind-down came before PHASE_RF_AB). Nothing flipped.

## Task 5: structural phase

From the profile the largest phase that identity pays for is sym.split
(the 3-launch stable partition; 45 ms of 251 on the H100). The single-pass
decoupled-lookback partition (DEVIATION 1907) produces the same stable
permutation by construction; the patch under
bench/results/trees_identical/patches/ routes it under IDENTICAL on the
NVIDIA column only (Apple keeps the 3-launch path, AMD stays off). Built on
the H100 as set `sp` (fold + fused + patch): 36/36 fingerprints equal at
the 20k fixtures, but the arm only triggers above 500k rows per leaf, so
the 1M/2M hash witness and the timing were NOT RUN. Not landed.

## Rows for bench/OPPONENT_REFERENCE.md (orchestrator merges; do not create the file on this branch)

All on NVIDIA H100 80GB HBM3, driver 580.126.09, runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04,
HIGGS train = first N rows, test = last 500,000 rows, same process as our arm, ms median (min..max):

| opponent | version | lane, params | rows | median ms | log |
|---|---|---|---|---|---|
| catboost-gpu | 1.2.10 | SymmetricTree, Logloss, iters 100, depth 6, lr 0.1, l2 1, border_count 254, bootstrap No, Plain, seed 7 | 1M | 900 (864..939) | h100_2026-09-09/speed/baseline.gbdt-symmetric.higgs.r1000000.full.log |
| catboost-gpu | 1.2.10 | same, RMSE on the 0/1 label (higgsreg) | 1M | 699 (680..759) | h100_2026-09-09/speed/baseline.gbdt-symmetric.higgsreg.r1000000.full.log |
| cuml-rf-gpu | 26.08.00 | RandomForestClassifier, 100 trees, depth 16, sqrt features, 128 bins, bootstrap, seed 7, n_streams default | 1M | 3314 (3257..3950), logloss 0.538814 | h100_2026-09-09/speed/baseline.rf.higgs.r1000000.full.log |
| lightgbm-cuda | 4.7.0 (USE_CUDA=ON) | rf boosting, 100 trees, depth 16, 32768 leaves, bagging 0.632/1, feature_fraction sqrt, max_bin 255 | 1M | 469654 (468691..471303, 3 rounds), logloss 0.638510 | same |

L40S (same versions, driver 580.126.09): catboost-gpu 781 (771..788) Logloss 1M, 915 (849..941) RMSE 1M;
l40s_2026-09-09/speed/profile_cells_console.txt.

## RUN OWED on the Apple M4 (orchestrator)

The fold changed bytes in core/pinned_reduce.mojo and pointwise_targets.mojo,
and the RF constants moved behind defines. Apple identical fingerprints
must not move:

    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build.sh
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_gbdt.sh
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_rf.sh
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_trees.sh
    MOJOLEARN_NUMERIC_MODE=identical python3 tools/identity_break.py \
        --lanes rf-clf,rf-reg,et-clf,et-reg,gbdt-symmetric,gbdt-depthwise,gbdt-lossguide,gbdt-rmse,kmeans \
        --vendor apple-m4 --json /tmp/apple_trees_lane.json
    python3 tools/identity_break.py --diff bench/results/identity_break/apple-m4.identical.json /tmp/apple_trees_lane.json
    python3 tools/identity_break.py --diff bench/results/trees_identical/h100_2026-09-09/ib/fold.json /tmp/apple_trees_lane.json
    tools/with_build_lock.sh pixi run mojo run -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 ensemble/checks/rf_perf_candidates_check.mojo
    pixi run check-fit-pointwise; pixi run check-logloss-train; pixi run check-ordered-boosting
    MOJOLEARN_NUMERIC_MODE=identical python3 -m mojolearn --conformance   # or the usual identical card task

Expected: every cell IDENTICAL (the NVIDIA baseline already matched the
existing Apple JSON 81/81; the fold matched that baseline 81/81 on NVIDIA).

## Next commands for a fresh agent (in order)

    MOJOLEARN_RUNPOD_KEY_FILE=~/.mojolearn_runpod_key tools/trees_leg.sh rent --gpu "NVIDIA H100 80GB HBM3" --minutes 90
    # then on the pod (see tools/trees_leg.sh ssh; the guard blocks compound ssh text, use `ssh <target> sh -s < script`):
    #   nohup sh tools/trees_identical_remote.sh > /root/trees_out/setup_console.log 2>&1 &   (about 7 min)
    #   scp the batch script; bench/results/trees_identical/h100_2026-09-09/logs/batchH.sh is the one that ran here:
    #   cut the lightgbm-cuda rf/et arms down (470 s per fit) by running rf/et cells with MOJOLEARN_SPEED_BUDGET_S=600, or --devices gpu with rounds 3
    #   remaining phases: PHASE_SYM_AB (baseline/fold/fused/spnf/sp, ours only, 1M x7 and 2M x5, higgsreg 1M), PHASE_RF_AB (baseline/rf2010/rf2011/rf2012, 1M/2M), et 1M/2M/5M full, rf 2M/5M full, sp full vs catboost at 1M/2M/5M
    #   the sp hash at 1M must equal dac2cf366e219cec (Logloss) and 037aa3188d25c889 (RMSE); 2M hashes must equal baseline's
    # flips, if the H100 confirms: FUSED_EST_MOVE_2030 default on (pointwise_oracle.mojo), the winning RF define into its comptime default, and the kernel_matrix patch handed to the kNN lane / orchestrator
    tools/trees_leg.sh reap

## Unfinished and why

- Every 2M/5M rung, ET at every rung, RF 2M/5M, the H100 A/B for fold,
  fused, sp and the RF candidates: the L40S pod expired (lease, the session
  was rate-limited mid-run) with the speed logs unfetched, and the H100 leg
  was wound down by the orchestrator while its first RF cell was in flight
  (lightgbm-cuda at 470 s per fit ate the phase).
- No flag flipped: nothing has an H100 timing with a log on disk.
