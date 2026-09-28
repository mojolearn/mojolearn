# trees-apple3: progress (Apple FAST speed round 3, trees family)

Branch `lane/trees-apple3` (worktree `~/mojolearn-wt/trees-apple3`), off
lane/apple3-merged 6856b5f8f. Brief: `~/mojolearn-evidence/apple3_speed_brief.md`.
Evidence: `~/mojolearn-evidence/trees-apple3/runs/<steward id>.stdout`.
Harness: `tools/trees_apple_ab.sh` (arms) around `tools/trees_apple_speed.sh`
(cells; this round adds `xtstage:` and `rfstage:`, a stage split without the
launch clock). Laptop helpers: `tools/trees_apple3/`.

## Steward jobs

| id | Mac | commit | mode | what |
|---|---|---|---|---|
| 1790626600545 | m4pro-b | 6856b5f8f | FAST | base timing (2 rounds) and phase split (GBDT stages, RF launch clock, cProfile) |
| 1790627269652 | m4pro-b | 7871c85bf | FAST | stage split without the launch clock (DART, AdaBoost, Bagging, DT, RF, ET) |
| 1790627750328 | m4pro-b | 2366629d3 | FAST | Lossguide exact batch A/B, taxi (widths 32, 16, 64); its quality cell failed on a script error (fixed at b7c00d5dd) and is NOT a result |
| 1790629196891 | m3ultra-b | b7c00d5dd | FAST + IDENTICAL | batch 1 (`tools/trees_apple3/job_batch1.sh`): Lossguide exact batch, RF node batch, forest data session; IDENTICAL digests; Lossguide and member quality |

## Base, FAST, M4 Pro m4pro-b (steward 1790626600545, commit 6856b5f8f)

1,000,000 training rows, taxi-shaped data (16 columns; Istella is not staged
on m4pro-b), fit ms, two rounds.

| cell | round 0 | round 1 | digest or hash | quality |
|---|---|---|---|---|
| dart:taxi (100 trees) | 4676 | 4638 | 3424b6dea3350a41 | accuracy 0.760005 |
| dart:taxireg (100 trees) | 4414 | 4360 | 1749a8811fcd9b6c | RMSE 3.9303 |
| rf:taxireg | 4088 | 4081 | 014f11cfa4c5dfff | RMSE 2.7490 |
| gbdt-lossguide:taxi (100 trees) | 3550 | 3683 | 715f109c52d6d7c8 / 14d2bb7846cb434e (FAST Lossguide varies run to run) | |
| adaboost:taxireg (50 members) | 3514 | 3517 | 142950540f150682 | |
| rf:taxi | 2755 | 2665 | 452a173087f86a9d | accuracy 0.75968 |
| adaboost:taxi (50 members) | 2724 | 2718 | 892ffac9df871610 | |
| et:taxireg | 2092 | 2081 | ec62616c8e02c60b | RMSE 2.0166 |
| et:taxi | 1894 | 1891 | ac18d5d8a54b1555 | accuracy 0.759845 |
| gbdt-depthwise:taxi (100 trees) | 1496 | 1506 | d94b1bf480b49280 | |
| gbdt-symmetric:taxi (100 trees) | 829 | 833 | 388959920a036d2d | |
| bagging:taxi | 623 | 631 | 16aaba82631a5774 | accuracy 0.75923 |
| embedding:taxi | 317 | 329 | 90d1b0749de17d44 | |
| dt:taxi / dt:taxireg | 76 / 70 | 74 / 70 | 86b6487420c736c8 / e735a53b7d74025a | accuracy 0.74586 / RMSE 3.6220 |
| iforest:taxi | 73 | 73 | 32881432d39389ec | |

## Where the time goes (same job; splits, never timings)

- GBDT Lossguide: 63 iterations per tree (one leaf per iteration), each with
  about 16 launches, two host waits and the id uploads. Per tree 35.0 ms
  against Depthwise's 14.3 ms for the same 64 leaves in 6 iterations: the
  difference is per-iteration overhead, not kernel work.
- DART: 40.8 ms per member, 33.4 ms of it inside one `rf_regressor_fit` call
  (a fresh forest fit per member: NaN scan, 64 MB host copy, upload,
  quantiles, binning, builder setup, then one 31-leaf tree) and 4.0 ms in a
  Python-side `all_finite` of the same X every member.
- RF taxi, launch counts for 100 trees: 7151 histogram phases (each
  phase_setup + histogram + find_best_splits + merge_split_candidates + an
  upload) and 3276 node-split batches (publish, count_left, copy_back, reset,
  uploads and a download): about 55,000 enqueues for a 2.7 s fit.

## Targets, in order of FAST seconds

1. GBDT Lossguide: exact best-first in batches (the same tree, fewer
   iterations).
2. DART (and the AdaBoost members): keep the device dataset across member fits.
3. RF builder launch count per phase.

## Changes (all OPT-IN until their A/B and quality check pass)

| commit | mode | change | switch | shared code? |
|---|---|---|---|---|
| 2366629d3 | FAST, Apple | GBDT Lossguide: exact best-first in batches. Every round replays best-first on the host over the gains known so far, splits the leaf the replay needs and, in the same round, the leaves best-first would split next; leaves split ahead of time that best-first never reached are folded back on the host at the end. Same tree as one leaf per iteration, about log2(max_leaves) + max_leaves / width rounds | `-D MOJOLEARN_GBDT_LG_EXACT_BATCH` (width 32; `..._BATCH16`, `..._BATCH64`) | GBDT non-symmetric driver (`greedy_search_helper_depthwise.mojo`); Depthwise and IDENTICAL compile the old path |
| f971afd27 | both | Forest data session: the members of DART and AdaBoostClassifier fit the same X, staged on the device once (NaN scan, pinned copy, upload) instead of once per member. `share` (FAST only) also keeps the first member's quantile table and bins | env `MOJOLEARN_FOREST_SESSION=1` or `share` | `ensemble/randomforest.mojo` (`fit_forest` is now a wrapper of `fit_forest_prepared`: every RF-builder estimator), `bindings/_mojolearn_rf.mojo`, `python/mojolearn/_forest_protocol.py`, `randomforest.py`, `_expansion_trees.py` |
| 4744936ed | FAST, Apple | RF builder: wider node batch (16384 or 32768 instead of 4096) for trees without a leaf budget that may grow past 12 levels, capped at 512 MB of histogram workspace per stream | `-D MOJOLEARN_RF_FAST_BATCH16K` or `32K` | `bindings/_mojolearn_rf.mojo` only |

## A/B results

| change | Mac | steward id | mode | cell | before ms | after ms | after/before | digest or quality |
|---|---|---|---|---|---|---|---|---|
| Lossguide exact batch, width 32 | m4pro-b | 1790627750328 | FAST | gbdt-lossguide:taxi (100 trees) | 3705 (3569, 3841) | 1500 (1506, 1493) | 0.405 | 10 trees: cfeb63222f74d866 in every arm. 100 trees: FAST Lossguide varies run to run in both arms (before bcc798f1.., cab8c89e..; after 14d2bb78.., which the base job's before arm also produced) |
| same, width 16 | m4pro-b | 1790627750328 | FAST | gbdt-lossguide:taxi | 3705 | 1531 | 0.413 | 10 trees equal |
| same, width 64 | m4pro-b | 1790627750328 | FAST | gbdt-lossguide:taxi | 3705 | 1509 | 0.407 | 10 trees equal |
| same (control) | m4pro-b | 1790627750328 | FAST | gbdt-depthwise:taxi | 1519 | 1509 | 0.993 | FAST Depthwise varies run to run in both arms |

Per tree, Lossguide went from 35.3 ms to 14.5 ms, Depthwise's 14.5 ms. At this
config (max_depth 6, max_leaves 64) the leaf budget never binds, so nothing
is folded back; the quality cell of batch 1 (max_leaves 31, max_depth 10)
is the one that exercises it.
