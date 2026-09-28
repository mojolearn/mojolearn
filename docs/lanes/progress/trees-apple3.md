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
