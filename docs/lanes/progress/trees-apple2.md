# trees-apple2: progress (Apple speed round 2, trees family)

Branch `lane/trees-apple2` (worktree `~/mojolearn-wt/trees-apple2`), off
lane/apple-merged 037daa353. Brief: `~/mojolearn-evidence/apple2_speed_brief.md`.
Evidence: `~/mojolearn-evidence/trees-apple2/runs/<steward id>.stdout`.

Harness: `tools/trees_apple_ab.sh` puts the before and after arms of one
change in ONE steward job (same commit, same Mac): each arm rebuilds the
named bindings with its own `MOJOLEARN_EXTRA_DEFINES` (the before arm is the
change's opt-out define), then runs `tools/trees_apple_speed.sh` on
`TAP_CELLS`; arms alternate per round.

## Changes

| commit | mode | change | opt-out define | shared code? |
|---|---|---|---|---|
| c7228df55 | IDENTICAL | RF row-major bins on Apple for `n_cols <= 64` (wide data stays column-major) | `MOJOLEARN_RF_BINS_COLUMN_MAJOR` | RF builder: DT, Bagging, DART, AdaBoost, RF |
| bf5ac8dec, a11e74ed9 | IDENTICAL | RF histogram zero-after-read on Apple (no per-round `hist_zero` launch). bf5ac8dec's `air.wg.barrier(3, 1)` did not build (conflicts with the stdlib declaration); a11e74ed9 keeps `barrier()` and zeroes each cell by the thread that wrote its cdf (also FAST's block kernel, multi-class) | `MOJOLEARN_RF_FAST_HIST_ZERO_OFF` | RF builder, as above |
| 44b69e68f | IDENTICAL | GBDT depthwise/lossguide ridx-only splits on Apple (DEVIATION 1902, FAST's already); partstats sweep and the `stats` trace record gather through the index | `MOJOLEARN_GBDT_IDENTICAL_RIDX_OFF` | GBDT non-symmetric driver, `checks/kernel_matrix.mojo` row |
| a5f2c1d34 | IDENTICAL | ET tiled range + regression score kernels on Apple (key-space range fold under IDENTICAL) | `MOJOLEARN_ET_TILED_SEARCH_IDENTICAL_OFF` | ET builder (ExtraTrees, RandomTreesEmbedding if 2k >= n) |

## Steward jobs

| id | Mac | commit | what |
|---|---|---|---|
| 1790603103015 | m3ultra-b | c7228df55 | RF row-major A/B |
| 1790603147569 | m4pro-a | c7228df55 | GBDT stage profile (lossguide, symmetric) |
| 1790603238984 | m4pro-b | bf5ac8dec | RF hist zero-after-read A/B |
| 1790603367483 | m4-a | a5f2c1d34 | ET tiled IDENTICAL A/B: FAIL, build_rf (the barrier above) |
| 1790603238984 | m4pro-b | bf5ac8dec | FAIL, same build |
| 1790604108084 | m4-a | a11e74ed9 | ET tiled IDENTICAL A/B (taxi only: Istella is staged only on m3ultra-b and m4pro-a) |
| 1790604112001 | m4pro-b | a11e74ed9 | RF hist zero A/B, taxi: PASS |
| 1790604496688 | m4pro-a | a11e74ed9 | RF hist zero A/B, Istella |
| 1790604882793 | m3ultra-b | 44b69e68f | GBDT ridx A/B |

## A/B results (IDENTICAL unless noted; ms, median of the rounds; digests equal before and after in every row)

| change | Mac | steward id | cell | before | after | after/before | digest |
|---|---|---|---|---|---|---|---|
| RF hist zero-after-read | m4pro-b | 1790604112001 | rf:taxi | 2986 | 2939 | 0.984 | 452a173087f86a9d |
| RF hist zero-after-read | m4pro-b | 1790604112001 | rf:taxireg | 5274 | 5162 | 0.979 | 58ec783b7afbd7a7 |
| RF hist zero-after-read | m4pro-b | 1790604112001 | dart:taxi | 6141 | 6177 | 1.006 | 8375ab8d60172694 |
| RF hist zero-after-read | m4pro-b | 1790604112001 | bagging:taxi | 642 | 645 | 1.004 | 16aaba82631a5774 |
| RF hist zero-after-read | m4pro-a | 1790604496688 | rf:istellareg | 50669 | 49529 | 0.977 | 3a5e8c09dd0d5fc7 |
| RF hist zero-after-read | m4pro-a | 1790604496688 | dt:istellareg | 837 | 834 | 0.997 | 6f406a1b9436c69f |
| ET tiled IDENTICAL | m4-a | 1790604108084 | et:taxireg | 34585 | 8964 | 0.259 | ec62616c8e02c60b |
| ET tiled IDENTICAL | m4-a | 1790604108084 | et:taxi (control, classifier k=4) | 8576 | 8611 | 1.004 | ac18d5d8a54b1555 |
| ET tiled IDENTICAL | m4pro-a | 1790606161176 | et:istellareg | 63088 | 23539 | 0.373 | 981c3b89e374c91d |
| ET tiled IDENTICAL | m4pro-a | 1790606161176 | et:taxireg | 7107 | 2337 | 0.329 | ec62616c8e02c60b |
| GBDT 2580 level quant (define, not flipped) | m4-a | 1790606781155 | gbdt-symmetric taxi 100 trees | 4960 | 5050 | 1.018 | 8e760782efae56c8 |

Metal enqueue prices (M4 m4-a, steward 1790607460116, `bench/speed/metal_enqueue_cost_main.mojo`):
kernel launch ~20 us host, small host-to-device copy ~20 us, launch + copy
back + synchronize ~180 us, empty synchronize 12 us. A Lossguide leaf split
paid two host waits (~360 us) and about ten small uploads plus ~16 launches;
at ~0.9 ms per split that is nearly all overhead, which is what the id-arena
change (7298ebd92) trims.
| GBDT ridx-only splits IDENTICAL (44b69e68f; both arms carry 7298ebd92) | m4pro-a | 1790608373786 | lossguide taxi | 3980 | 3838 | 0.965 | b1761eecc6dfbc73 |
| same | m4pro-a | 1790608373786 | lossguide istella | 6015 | 5996 | 0.997 | eb0d9510ee08a16f |
| same | m4pro-a | 1790608373786 | depthwise taxi | 1768 | 1713 | 0.969 | 5694af7699036c65 |
| same | m4pro-a | 1790608373786 | depthwise istella | 3587 | 3673 | **1.024 (slower)** | e9c0f7e913af5a8a |
| GBDT id arena IDENTICAL (7298ebd92) | m4-a | 1790608353263 | lossguide taxi | 7354 | 6865 | 0.933 | b1761eecc6dfbc73 |
| same | m4-a | 1790608353263 | depthwise taxi | 3317 | 3266 | 0.985 | 5694af7699036c65 |
| FAST GBDT 2031 sym ridx (define arm, e314b925d) | m4pro-b | 1790608403999 | symmetric taxi | 919 | 835 | 0.909 | 388959920a036d2d |
| FAST GBDT id arena (e314b925d; before = ids separate) | m4pro-b | 1790608403999 | lossguide taxi | 4102 | 3820 | 0.931 | FAST lossguide/depthwise digests vary run to run in BOTH arms (not deterministic in FAST), so no digest comparison |
| same | m4pro-b | 1790608403999 | depthwise taxi | 1538 | 1524 | 0.991 | as above |

Decision on IDENTICAL ridx for the non-symmetric driver: kept on (3 of 4
cells faster, geomean 0.989), with the Istella depthwise 1.024 noted.
| RF row-major bins IDENTICAL n_cols <= 64 (c7228df55; tip bfc552c67) | m4-a | 1790609927653 | rf:taxireg | 16485 | 10177 | 0.617 | 58ec783b7afbd7a7 |
| same | m4-a | 1790609927653 | rf:taxi | 6217 | 5594 | 0.900 | 452a173087f86a9d |
| same | m4-a | 1790609927653 | dt:taxireg | 188 | 143 | 0.760 | e735a53b7d74025a |
| same | m4-a | 1790609927653 | dt:taxi | 146 | 129 | 0.880 | 86b6487420c736c8 |
| same | m4-a | 1790609927653 | bagging:taxi | 1129 | 969 | 0.858 | 16aaba82631a5774 |
| same | m4-a | 1790609927653 | dart:taxireg | 6600 | 6356 | 0.963 | 86f40254833745ec |
| same | m4-a | 1790609927653 | adaboost:taxireg | 2339 | 2348 | 1.004 | 160452cbaf288200 |
| GBDT 1903 deferred copy IDENTICAL (939ea50e9) | m4-a | 1790610574529 | lossguide taxi | 6932 | 6620 | 0.955 | b1761eecc6dfbc73 |
| same | m4-a | 1790610574529 | depthwise taxi | 3285 | 3266 | 0.994 | 5694af7699036c65 |
| RF row-major IDENTICAL (c7228df55) | m3ultra-b | 1790603103015 | rf:taxireg | 3437 | 3295 | 0.959 | 58ec783b7afbd7a7 |
| same | m3ultra-b | 1790603103015 | rf:taxi | 2565 | 2498 | 0.974 | 452a173087f86a9d |
| same | m3ultra-b | 1790603103015 | dt:taxi / dt:taxireg / bagging / dart:taxireg / adaboost:taxireg | 67 / 69 / 597 / 5700 / 1922 | 68 / 70 / 599 / 5679 / 1922 | 1.01 / 1.00 / 1.00 / 1.00 / 1.00 | all equal |
| ET narrow row-major, always-on trial arm (88535136f) | m4-a | 1790610860810 | et:taxi | 3904 | 3745 | 0.959 | ac18d5d8a54b1555 |
| same | m4-a | 1790610860810 | embedding:taxi (k = 1) | 469 | 1086 | **2.317** -> gated on 4k >= n_cols (14706bb29) | 90d1b0749de17d44 |
| ET narrow row-major gated 4k >= n (14706bb29) | m4-a | 1790611318000 | et:taxi | 3880 | 3756 | 0.968 | ac18d5d8a54b1555 |
| same | m4-a | 1790611318000 | embedding:taxi | 467 | 466 | 0.997 | 90d1b0749de17d44 |
| GBDT ridx-only IDENTICAL (206ca2e53) | m3ultra-b | 1790606770142 | depthwise taxi / istella | 1704 / 2894 | 1667 / 2899 | 0.978 / 1.002 | 5694af7699036c65 / e9c0f7e913af5a8a |
| same | m3ultra-b | 1790606770142 | lossguide istella | 6753 | 6714 | 0.994 | eb0d9510ee08a16f |
| GBDT 2031 sym ridx IDENTICAL, ungated (bfc552c67) | m4pro-a | 1790609918603 | symmetric taxi | 1024 | 941 | 0.919 | 8e760782efae56c8 |
| same | m4pro-a | 1790609918603 | symmetric istella | 2813 | 2992 | **1.064 (slower)** -> gated to <= 64 features (63a38c138) | d5571c2a35ea06c8 |
| FAST combined: 2031 + id arena + ET narrow off vs on (63a38c138) | m4-a | 1790611979532 | FAST symmetric taxi | 1706 | 1575 | 0.923 | 8f160400c1defdf5 |
| same | m4-a | 1790611979532 | FAST lossguide taxi | 7054 | 6385 | 0.905 | FAST lossguide is not deterministic run to run (two digests inside EACH arm); the changes are schedule-only |
| same | m4-a | 1790611979532 | FAST depthwise taxi | 3118 | 3080 | 0.988 | 4e615891dc42e914 |
| same | m4-a | 1790611979532 | FAST et:taxi | 3860 | 3728 | 0.966 | ac18d5d8a54b1555 (accuracy 0.759845 both) |
| same | m4-a | 1790611979532 | FAST embedding:taxi | 467 | 466 | 0.999 | 90d1b0749de17d44 |
| GBDT 2031 gated to <= 64 features, IDENTICAL (63a38c138) | m4pro-a | 1790611947758 | symmetric taxi | 1026 | 935 | 0.911 | 8e760782efae56c8 |
| same | m4pro-a | 1790611947758 | symmetric istella | 2827 | 2819 | 0.997 | d5571c2a35ea06c8 |
| RF columns per pass IDENTICAL, trial arms (819d4bd7b) | m4pro-a | 1790612032193 | rf:istellareg | 49533 (10) | 47836 (20) / 47139 (40) | 0.966 / 0.952 | 3a5e8c09dd0d5fc7 |

## M3 Ultra m3ultra-b, IDENTICAL, all changes off vs on (commit 14706bb29, steward 1790611718868)

Before = every opt-out define of this lane (`-D MOJOLEARN_RF_BINS_COLUMN_MAJOR -D MOJOLEARN_RF_FAST_HIST_ZERO_OFF -D MOJOLEARN_ET_TILED_SEARCH_IDENTICAL_OFF -D MOJOLEARN_ET_RM_NARROW_OFF -D MOJOLEARN_GBDT_IDENTICAL_RIDX_OFF -D MOJOLEARN_2031_SYM_RIDX_SPLITS_OFF -D MOJOLEARN_GBDT_ID_UPLOADS_SEPARATE -D MOJOLEARN_GBDT_IDENTICAL_SPLIT_COPY`), after = defaults; same commit, same Mac, arms alternated, 2 rounds, median ms. Every digest equal before and after. This commit still had 2031 ungated (the Istella symmetric 1.021 row) and 10 RF columns per pass.

| cell | before | after | after/before | digest |
|---|---|---|---|---|
| adaboost:taxi | 2947 | 2909 | 0.987 | e8529a04f218dbab |
| adaboost:taxireg | 1905 | 1930 | 1.013 | 160452cbaf288200 |
| bagging:taxi | 600 | 595 | 0.992 | 16aaba82631a5774 |
| dart:taxireg | 5648 | 5772 | 1.022 | 86f40254833745ec |
| dt:istellareg | 446 | 449 | 1.007 | 6f406a1b9436c69f |
| dt:taxireg | 70 | 69 | 0.987 | e735a53b7d74025a |
| embedding:taxi | 187 | 189 | 1.008 | 90d1b0749de17d44 |
| et:istellareg | 16004 | 9373 | 0.586 | 981c3b89e374c91d |
| et:taxi | 1053 | 1046 | 0.994 | ac18d5d8a54b1555 |
| et:taxireg | 1665 | 1161 | 0.698 | ec62616c8e02c60b |
| gbdt-depthwise:istella | 2912 | 2890 | 0.992 | e9c0f7e913af5a8a |
| gbdt-depthwise:taxi | 1734 | 1653 | 0.953 | 5694af7699036c65 |
| gbdt-lossguide:istella | 6948 | 6523 | 0.939 | eb0d9510ee08a16f |
| gbdt-lossguide:taxi | 5396 | 4852 | 0.899 | b1761eecc6dfbc73 |
| gbdt-symmetric:istella | 2001 | 2042 | 1.021 | d5571c2a35ea06c8 |
| gbdt-symmetric:taxi | 730 | 700 | 0.959 | 8e760782efae56c8 |
| iforest:taxi | 68 | 68 | 1.000 | 96ff7aa1dfcef11e |
| rf:istellareg | 19647 | 19681 | 1.002 | 3a5e8c09dd0d5fc7 |
| rf:taxi | 2563 | 2470 | 0.964 | 452a173087f86a9d |
| rf:taxireg | 3439 | 3234 | 0.941 | 58ec783b7afbd7a7 |

## M4 m4-a, IDENTICAL, all changes off vs on (commit 819d4bd7b, steward 1790613322668)

Same arms as the M3 Ultra table (10 RF columns per pass in both arms; 2031 already gated to <= 64 features; non-symmetric ridx not yet gated). Taxi-shaped data only: Istella is not staged on m4-a. Every digest equal.

| cell | before | after | after/before | digest |
|---|---|---|---|---|
| adaboost:taxi | 3423 | 3454 | 1.009 | e8529a04f218dbab |
| adaboost:taxireg | 2318 | 2324 | 1.002 | 160452cbaf288200 |
| bagging:taxi | 1158 | 981 | 0.847 | 16aaba82631a5774 |
| dart:taxireg | 6400 | 6282 | 0.982 | 86f40254833745ec |
| dt:taxireg | 197 | 142 | 0.717 | e735a53b7d74025a |
| embedding:taxi | 466 | 466 | 1.000 | 90d1b0749de17d44 |
| et:taxi | 3875 | 3757 | 0.969 | ac18d5d8a54b1555 |
| et:taxireg | 16346 | 4250 | 0.260 | ec62616c8e02c60b |
| gbdt-depthwise:taxi | 3513 | 3261 | 0.928 | 5694af7699036c65 |
| gbdt-lossguide:taxi | 7496 | 6591 | 0.879 | b1761eecc6dfbc73 |
| gbdt-symmetric:taxi | 1726 | 1553 | 0.900 | 8e760782efae56c8 |
| iforest:taxi | 69 | 70 | 1.013 | 96ff7aa1dfcef11e |
| rf:taxi | 6283 | 5632 | 0.896 | 452a173087f86a9d |
| rf:taxireg | 16722 | 10239 | 0.612 | 58ec783b7afbd7a7 |

## Later single-change A/Bs (IDENTICAL)

| change | Mac | steward id | cell | before | after | after/before | digest |
|---|---|---|---|---|---|---|---|
| non-symmetric ridx gated to <= 64 features (ef756d960) | m4pro-a | 1790613700948 | depthwise taxi | 1768 | 1683 | 0.952 | 5694af7699036c65 |
| same | m4pro-a | 1790613700948 | depthwise istella | 3547 | 3542 | 0.999 | e9c0f7e913af5a8a |
| same | m4pro-a | 1790613700948 | lossguide istella | 5938 | 5912 | 0.996 | eb0d9510ee08a16f |
| 2031 sym ridx IDENTICAL (bfc552c67) | m4pro-b | 1790609413669 | symmetric taxi | 966 | 872 | 0.903 | 8e760782efae56c8 |
| RF 40 columns per pass IDENTICAL (72437f655, workspace uncapped) | m3ultra-b | 1790614067535 | rf:istellareg | 19706 | 18547 | 0.941 | 3a5e8c09dd0d5fc7 |
| same | m3ultra-b | 1790614067535 | rf:taxireg / dt:istellareg / dt:taxireg | 3232 / 446 / 68 | 3142 / 433 / 70 | 0.972 / 0.972 / 1.022 | equal |
| same | m4pro-b | 1790614070834 | rf:taxireg / rf:taxi | 5162 / 2942 | 5019 / 2954 | 0.972 / 1.004 | equal |
| same | m4pro-b | 1790614070834 | dart:taxireg / adaboost:taxireg / bagging:taxi / dt:taxireg | 5839 / 2062 / 634 / 81 | 6422 / 2169 / 665 / 84 | **1.100 / 1.052 / 1.048 / 1.039** -> workspace capped at the sampled columns (a91fe60be) | equal |
| id arena incl. split bins, IDENTICAL (a29c5bffa; before = one copy per slot) | m4pro-b | 1790615807516 | lossguide taxi | 3982 | 3737 | 0.938 | b1761eecc6dfbc73 |
| same | m4pro-b | 1790615807516 | depthwise taxi | 1643 | 1617 | 0.984 | 5694af7699036c65 |
