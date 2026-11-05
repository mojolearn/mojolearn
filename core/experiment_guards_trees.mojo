# SPDX-License-Identifier: Apache-2.0
"""Opt-in strategy admission for the tree bindings (GBDT, random forest, trees, x_trees, the forest host).

Each binding of this family imports TREES_GUARDS_OK and asserts it in its build function, so every compile of one of
those bindings (per-binding or merged) evaluates these guards. A guard whose define is read in code by bindings
of more than one family belongs in core/six_lane_experiment_guards.mojo, which every binding reaches through
checks/numerics.mojo.
"""
from std.sys.compile import is_defined, get_defined_int
# The entries below moved verbatim, in their original order, from core/six_lane_experiment_guards.mojo on
# 2026-11-04 (lane split-hot-groups; the placement of each entry is docs/build/guard_split.json). A header comment
# that headed lines of several families is carried with each of them.


def _check_configuration() -> Bool:
    comptime assert not (is_defined["MOJOLEARN_FOREST_ORDERED_RESIDENT_OFF"]() and is_defined["MOJOLEARN_AFT_P02"]()), "incompatible integrated strategies: MOJOLEARN_FOREST_ORDERED_RESIDENT_OFF / MOJOLEARN_AFT_P02"
    comptime assert not (is_defined["MOJOLEARN_AFT_P07"]() and is_defined["MOJOLEARN_SHAP_FAST_ROW_PAIR"]()), "incompatible integrated strategies: MOJOLEARN_AFT_P07 / MOJOLEARN_SHAP_FAST_ROW_PAIR"
    # Lane grid-prune (2026-10-07): deleted losers and dead arms (rows in docs/apple-fast/EXPERIMENTS.md
    # "IDENTICAL grid prune"); recoverable at main ab554bb4a. A stale build line must not run the incumbent silently.
    comptime assert not is_defined["MOJOLEARN_TREES_T09"](), "removed: MOJOLEARN_TREES_T09 (ExtraTrees bootstrap sort: unreachable at bootstrap=False, host sync in fit)"
    comptime assert not is_defined["MOJOLEARN_TREES_T27"](), "removed: MOJOLEARN_TREES_T27 (alias of MOJOLEARN_2030_FUSED_EST_MOVE; inert on board walks)"
    comptime assert not is_defined["MOJOLEARN_TREES_C47_GBDT"](), "removed: MOJOLEARN_TREES_C47_GBDT (width cap subsumed by MOJOLEARN_TREES_T17_BATCH)"
    comptime assert not is_defined["MOJOLEARN_TREES_HIST_REP_BPSM"](), "merged: use -D MOJOLEARN_TREES_HIST_REP_SM=1 (device SMs x 4 blocks); BPSM=4 alone equalled SM=64"
    # Deleted 2026-10-08 (lane grid-losers-1): IDENTICAL grid ge123e6f9 losers (docs/apple-fast/EXPERIMENTS.md). Recoverable at main ad7ed2370.
    # C25_PROJECTION_REUSE: deleted 2026-10-08 above, restored 2026-10-10 (lane c25-restore-promote) from the tombstone patch after grid pass 2.
    comptime assert not is_defined["MOJOLEARN_TREES_T19"](), "removed: MOJOLEARN_TREES_T19 retired 2026-10-08: quality loss, istella AUC -0.43% and logloss +20.7% despite NV 0.47x / AMD 0.69x on depthwise taxi (grid ge123e6f9); see EXPERIMENTS.md"
    comptime assert not is_defined["MOJOLEARN_TREES_T22"](), "promoted 2026-10-08 (grid ge123e6f9, lane/grid-flips-1): ordered document-keyed storage is the IDENTICAL default; drop -D MOJOLEARN_TREES_T22, use -D MOJOLEARN_TREES_T22_OFF for the old path"
    comptime assert not is_defined["MOJOLEARN_TREES_ORD_STD_GRIDFOLD"](), "promoted 2026-10-08 (grid ge123e6f9, lane/grid-flips-1): ordered grid fold is the IDENTICAL default; drop -D MOJOLEARN_TREES_ORD_STD_GRIDFOLD, use -D MOJOLEARN_TREES_ORD_STD_GRIDFOLD_OFF for the old path"
    # Deleted 2026-10-08 (lane grid-act-2): IDENTICAL grid ge123e6f9 losers (docs/apple-fast/EXPERIMENTS.md). Recoverable at main 42d1e42c6.
    comptime assert get_defined_int["MOJOLEARN_TREES_HIST_MULTISTAT",0]() == 0, "removed: MOJOLEARN_TREES_HIST_MULTISTAT (=4|8) retired 2026-10-08: slower, gbdt-multiclass =4 NV 1.04x / AMD 1.41x istella, 1.07x / 1.11x taxi; =8 0.99x / 1.31x istella, 1.11x / 1.14x taxi; quality SAME (grid ge123e6f9); see EXPERIMENTS.md"
    comptime assert not is_defined["MOJOLEARN_TREES_T29_VERSIONED"](), "removed: MOJOLEARN_TREES_T29_VERSIONED (T29 arm 'versioned') retired 2026-10-08: slower, gbdt-rank-pairlogit istella NV 3.43x / AMD 3.76x (grid ge123e6f9); MOJOLEARN_TREES_T29 alone stays; see EXPERIMENTS.md"
    # Lane owed-deletions-D1 (2026-10-09): DROPPED rows in docs/apple-fast/EXPERIMENTS.md whose code was still on main (docs/TOMBSTONES.md). Deleted code recoverable at main b639a2bd2.
    comptime assert not is_defined["MOJOLEARN_EST_REUSE_PART"](), "removed 2026-10-09 (lane/owed-deletions-D1): FAST Apple leaf estimation in row order over the searcher partition was DROPPED-BUG: symmetric istella auc .980 -> .930, logloss .186 -> 2.15 (wrong leaves, sym-est-rp-ist); code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_EST_SHRINK_FUSED"](), "removed 2026-10-09 (lane/owed-deletions-D1): FAST Apple cursor add fused with the next tree derivative pass was DROPPED-inconclusive: sym-est-sh-1k istella -2.8% on an old base, never verified on main; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_KSHAP_FAST_OVERLAP"](), "removed 2026-10-09 (lane/owed-deletions-D1): KernelExplainer next-chunk synthetic rows and download overlapped with the model was DROPPED-noise: kernel-shap istella neutral (M3 verdicts batch 6, 1 run per arm); code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_PSHAP_FAST_OVERLAP"](), "removed 2026-10-09 (lane/owed-deletions-D1): PermutationExplainer next-chunk delta rows and download overlapped with the model was DROPPED-slower: permutation-shap taxi +26.6% (M3 verdicts batch 6, 1 run per arm); code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_MC_CLASS_BATCH_DERIV"](), "removed 2026-10-09 (lane/owed-deletions-D1): FAST Apple MultiClass softmax in registers was DROPPED-noise: symmulti-mc-deriv istella/taxi within +-2.4%, identical quality (old base); code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_MC_CLASS_BATCH_EST"](), "removed 2026-10-09 (lane/owed-deletions-D1): FAST Apple MultiClass leaf-oracle value, der and der2 in one launch was DROPPED-noise: symmulti-mc-est istella/taxi within +-2.4%, with MC_CLASS_BATCH_DERIV -1.2% (old base), quality identical; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_PL_PAIRS_ONCE"](), "removed 2026-10-09 (lane/owed-deletions-D1): FAST Apple PairLogit each-pair-once group kernel was DROPPED-noise: symmulti-pl-once / pl-both istella within +-2.4%, identical quality (old base); code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_SYM_DERIV_FUSED"](), "removed 2026-10-09 (lane/owed-deletions-D1): FAST Apple symmetric next-tree gradient pass behind the tail drain was DROPPED-noise: sym-iter-fused istella/taxi -0.1% (old base); F12 re-runs 2026-10-06 kept it OFF; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_SYM_LEAF_FROM_STATS"](), "removed 2026-10-09 (lane/owed-deletions-D1): FAST Apple symmetric leaves as one Newton step from the searcher partition stats was DROPPED-noise: sym-iter-leaf istella/taxi -0.2% (old base); F12 re-runs 2026-10-06 kept it OFF; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_SYM_REUSE_PARTITION"](), "removed 2026-10-09 (lane/owed-deletions-D1): FAST Apple symmetric reuse of the searcher final partition for leaf estimation was DROPPED-noise: sym-iter-reuse taxi +0.3% (old base); F12 re-runs 2026-10-06 B/A 0.94-1.03 mixed, kept OFF; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_SYM_GATHER_FUSED"](), "removed 2026-10-09 (lane/owed-deletions-D1): FAST Apple symmetric partition sizes and both gathers in one launch was DROPPED-noise: symhist-gather istella/taxi noise (old base); code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_SYM_PART_STATS_PAR"](), "removed 2026-10-09 (lane/owed-deletions-D1): FAST Apple symmetric per-partition sums on a partitions x chunks grid with float atomics was DROPPED-noise: symhist-part-stats istella/taxi noise (old base); code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_SYM_HIST_MULT"](), "removed 2026-10-09 (lane/owed-deletions-D1): FAST Apple pointwise histogram blocks at 2x the SM count was DROPPED-noise: symhist-hist-mult istella/taxi noise (old base); code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_SYM_SCAN_SUB_FUSED"](), "removed 2026-10-09 (lane/owed-deletions-D1): FAST Apple pointwise fold scan and sibling subtraction in one launch was DROPPED-noise: symhist-scan-sub istella/taxi noise (old base); code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_YETI_TREE_SEARCH_SCORE_GRID"](), "removed 2026-10-03 by 99da8a08f (refusal added 2026-10-09, lane/owed-deletions-D1): FAST Apple one-thread-per-bin-feature symmetric score grid was DROPPED-noise: yetirank 5,304 -> 5,288 ms (-0.3%, overlap); code at 99da8a08f^; see docs/TOMBSTONES.md"
    # wave-2 w2-decide-algos (2026-10-11): promoted (absent = new path; _OFF the old path) and removed switches.
    comptime assert not is_defined["MOJOLEARN_IDN_PSHAP_RESIDENT"](), "promoted 2026-10-11 (lane/w2-decide-algos, same-box A/B on main c8ef6c66d): resident permutation-SHAP linear path is the IDENTICAL default (0.097/0.022); drop -D MOJOLEARN_IDN_PSHAP_RESIDENT, use -D MOJOLEARN_IDN_PSHAP_RESIDENT_OFF for the old path"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_LG_FUSED_CHAIN.patch
    comptime assert not is_defined["MOJOLEARN_IDN_LG_FUSED_CHAIN"](), "removed 2026-10-11 (lane/w2-decide-linalg, 9ea0ceea2): IDN_LG_FUSED_CHAIN Lossguide on the fused split chain (F2) was NOISE: gbdt-lossguide istella 1.001, taxi 0.988, gbdt-categorical 1.024 (nv2 v0113, amd a1695); see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_CTR_PERM_GRID.patch
    comptime assert not is_defined["MOJOLEARN_IDN_CTR_PERM_GRID"](), "removed 2026-10-11 (lane/w2-decide-linalg, d092b6e2d): IDN_CTR_PERM_GRID CTR permutation grid bins were NOISE: gbdt-categorical taxi 1.001 (nv2 v0114, amd a1696); see docs/TOMBSTONES.md"
    # Lane tombstone-qsample-ivfstride (2026-10-15), removed: Apple FAST losers of the M4 A/B on main 7a8f183f6, synthetic non-board data,
    # one warmup + one scored fit per arm (docs/TOMBSTONES.md, docs/apple-fast/EXPERIMENTS.md). Code recoverable at 1a76955b5^.
    # Restore: git apply experiments/removed/MOJOLEARN_RF_FAST_QSAMPLE.patch (or MOJOLEARN_IVF_FAST_STRIDED_COARSE.patch)
    comptime assert not is_defined["MOJOLEARN_RF_FAST_QSAMPLE"](), "removed 2026-10-15 (lane/tombstone-qsample-ivfstride, 1a76955b5): RF_FAST_QSAMPLE (split candidates from >= 65,536 sampled rows) was DROPPED-slower: M4 DecisionTreeRegressor 1,000,000 x 32 7995 -> 8673 ms (8% slower), R2 0.99118 -> 0.99159 (noise). Restore with experiments/removed/MOJOLEARN_RF_FAST_QSAMPLE.patch."
    return True


comptime TREES_GUARDS_OK = _check_configuration()
