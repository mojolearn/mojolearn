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
