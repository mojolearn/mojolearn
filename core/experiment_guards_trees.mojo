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
    # Lane k1-delete-unverdicted-fast (2026-11-09): Apple FAST experiments with no verdict, never to be measured (Apple FAST
    # investment stopped); deleted (docs/TOMBSTONES.md "Unverdicted FAST experiments deleted (lane K1)").
    # Restore: git apply experiments/removed/MOJOLEARN_KSHAP_FAST_DEVICE_MODEL.patch
    comptime assert not is_defined["MOJOLEARN_KSHAP_FAST_DEVICE_MODEL"](), "removed 2026-11-09 (lane k1-delete-unverdicted-fast): MOJOLEARN_KSHAP_FAST_DEVICE_MODEL, KernelExplainer's FAST tier on the IDENTICAL device-model route for a built-in forest, was OPEN (afr2-kshap-dm-rf never run) and will not be measured; the IDENTICAL MOJOLEARN_IDN_SHAP_DEVICE_MODEL stays; code at f59d913b0b; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_PSHAP_FAST_DEVICE_MODEL"](), "removed 2026-11-09 (lane k1-delete-unverdicted-fast): MOJOLEARN_PSHAP_FAST_DEVICE_MODEL, the PermutationExplainer twin of KSHAP_FAST_DEVICE_MODEL, was OPEN (afr2-pshap-dm-rf never run) and will not be measured; code at f59d913b0b; see docs/TOMBSTONES.md"

    # Lane m1-followups-misc (2026-11-09): the escapes from the AdaBoost data session (a GPU tree member's rounds on the device).
    # TOMBSTONE: MOJOLEARN_TE_ADA_SESSION_OFF / MOJOLEARN_IDN_ADA_SESSION_OFF (escapes) deleted 2026-11-09 by lane/m1-followups-misc; code recoverable at 5b5490960d.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ADA_SESSION_OFF.patch
    comptime assert not is_defined["MOJOLEARN_TE_ADA_SESSION_OFF"](), "removed 2026-11-09 (lane/m1-followups-misc): MOJOLEARN_TE_ADA_SESSION_OFF, the FAST escape from the AdaBoost members' data session (it sent a GPU tree member back to the per-round host path; the session was the measured winner, adaboost-clf -9.0%, adaboost-reg -42% on the M3); every build opens the session; code at 5b5490960d; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_IDN_ADA_SESSION_OFF"](), "removed 2026-11-09 (lane/m1-followups-misc): MOJOLEARN_IDN_ADA_SESSION_OFF, the IDENTICAL escape from the AdaBoost members' data session (the session moves no bit); every build opens the session and a GPU binary runs every tree member's round on the device; code at 5b5490960d; see docs/TOMBSTONES.md"
    # integ-w6 (2026-11-09): a retired model-selection define; CalibratedClassifierCV (bindings/trees_public_meta_door.mojo) is in this family.
    comptime assert not is_defined["MOJOLEARN_MSEL_UNITS_SERIAL"](), "removed 2026-11-09 (integ-w6, integ/pre-0867-w3; code at lane/p3-msel-parallel-and-process-pool@9d7f5a2af9): MSEL_UNITS_SERIAL (every CV unit on the calling thread) is redundant: units whose member fits run on a GPU run one at a time by default (bindings/native_unit_threads.mojo native_unit_device_workers); -D MOJOLEARN_MSEL_UNITS_GPU_PARALLEL is the parallel-GPU arm (docs/TOMBSTONES.md)"
    # Lane expdel-05-trees (2026-11-15): Apple FAST GBDT/SYM losers and a superseded old path (docs/plans/
    # EXPERIMENT_DELETIONS_2026-11-15.md, docs/TOMBSTONES.md "Lane expdel-05-trees"); code recoverable at each line's sha.
    # Restore: git apply experiments/removed/MOJOLEARN_SYM_FEAT_ALL.patch
    comptime assert not is_defined["MOJOLEARN_SYM_FEAT_ALL"](), "removed 2026-11-15 (lane/expdel-05-trees): MOJOLEARN_SYM_FEAT_ALL, the FAST Apple umbrella of the six symmetric feature-set switches, was NEUTRAL: rab4-symfeat istella -1.3%, taxi -0.1%, quality noise; the switches it kept keep their own defines; code at 54b6a264fc; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_GBDT_BOOT_DEVICE.patch
    comptime assert not is_defined["MOJOLEARN_GBDT_BOOT_DEVICE"](), "removed 2026-11-15 (lane/expdel-05-trees): MOJOLEARN_GBDT_BOOT_DEVICE, the FAST Apple device bootstrap seed fill, was NEUTRAL: rab7-gbdtbootdevice istella -0.2%; the device seed fill is every build's default, the switch read nothing; code at 2f51f8b97d; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_GBDT_EVAL_SKIP_EMPTY.patch
    comptime assert not is_defined["MOJOLEARN_GBDT_EVAL_SKIP_EMPTY"](), "removed 2026-11-15 (lane/expdel-05-trees): MOJOLEARN_GBDT_EVAL_SKIP_EMPTY, the FAST Apple one-word held-out index for an empty eval set, was NEUTRAL: rab7-gbdtevalskipem istella +0.0%, taxi +0.4%; code at 8815b00d75; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_GBDT_EVAL_FUSED.patch
    comptime assert not is_defined["MOJOLEARN_GBDT_EVAL_FUSED"](), "removed 2026-11-15 (lane/expdel-05-trees): MOJOLEARN_GBDT_EVAL_FUSED, the FAST Apple packed held-out apply and deferred held-out loss readback, was NEUTRAL: rab7-gbdtevalfused istella 14752.15 -> 14703.82 ms (-0.3%); code at 70801eb24c; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_SYM_HIST_ALL.patch
    comptime assert not is_defined["MOJOLEARN_SYM_HIST_ALL"](), "removed 2026-11-15 (lane/expdel-05-trees): MOJOLEARN_SYM_HIST_ALL, the FAST Apple umbrella of SYM_SORT_SWAP + SYM_RESOLVE_BLOCK, was NEUTRAL: rab4-symhist istella -0.1%, taxi -0.2%, quality noise; code at 24c9108159; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_SYM_SORT_SWAP.patch
    comptime assert not is_defined["MOJOLEARN_SYM_SORT_SWAP"](), "removed 2026-11-15 (lane/expdel-05-trees): MOJOLEARN_SYM_SORT_SWAP, the FAST Apple one-pass level sort with a handle swap, was NEUTRAL: rab7-symsortswap istella +0.3%, taxi +0.1%; code at 1805cb6b73; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_SYM_ITER_ALL.patch
    comptime assert not is_defined["MOJOLEARN_SYM_ITER_ALL"](), "removed 2026-11-15 (lane/expdel-05-trees): MOJOLEARN_SYM_ITER_ALL, the FAST Apple umbrella of SYM_BUF_ARENA, was NEUTRAL: rab4-symiter 1000-taxi +0.2%, quality identical; code at a768195223; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_SYM_BUF_ARENA.patch
    comptime assert not is_defined["MOJOLEARN_SYM_BUF_ARENA"](), "removed 2026-11-15 (lane/expdel-05-trees): MOJOLEARN_SYM_BUF_ARENA, the FAST Apple per-fit buffer pool of the pointwise symmetric arm, was NEUTRAL: rab7-symbufarena 1000-taxi 18897.66 -> 18865.75 ms (-0.2%); F12 M3 B/A 0.93..1.01 mixed; code at 0f457a0c03; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_SYM_MULTI_ALL.patch
    comptime assert not is_defined["MOJOLEARN_SYM_MULTI_ALL"](), "removed 2026-11-15 (lane/expdel-05-trees): MOJOLEARN_SYM_MULTI_ALL, the FAST Apple umbrella of YR_TASK_FUSED (its PL_GROUP_NARROW part is the default on its own), was NEUTRAL: rab4-symmulti yetirank 3390.8 -> 3395.46 ms (+0.1%); code at 16f9e02852; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_YR_TASK_FUSED.patch
    comptime assert not is_defined["MOJOLEARN_YR_TASK_FUSED"](), "removed 2026-11-15 (lane/expdel-05-trees): MOJOLEARN_YR_TASK_FUSED, the FAST Apple one-launch YetiRank call, was NEUTRAL: rab7-yrtaskfused yetirank 3386.79 -> 3394.43 ms (+0.2%), ndcg10 .680466 -> .680553; code at 41f0634a67; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_KSHAP_FAST_BATCH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KSHAP_FAST_BATCH_OFF"](), "removed 2026-11-15 (lane/expdel-05-trees): MOJOLEARN_KSHAP_FAST_BATCH_OFF, the FAST Apple rollback of the batched kernel / permutation SHAP buffers, is removed: KSHAP_FAST_BATCH was KEPT (kap2-kshap-batch-istella 27,011 -> 15,325 ms, kap2-pshap-batch-istella 34,198 -> 28,236 ms, M3); every FAST Apple build batches; code at 073a9cc375; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_SCALAR_STAGE_COMPARE.patch
    comptime assert not is_defined["MOJOLEARN_ET_SCALAR_STAGE_COMPARE"](), "removed 2026-11-15 (lane/expdel2-01-base): MOJOLEARN_ET_SCALAR_STAGE_COMPARE (the ExtraTrees scalar staging-copy reference arm) is gone, no A/B was ever recorded; the vector byte copies are the only path; restore: experiments/removed/MOJOLEARN_ET_SCALAR_STAGE_COMPARE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_DT_BINS_QOLD.patch
    comptime assert not is_defined["MOJOLEARN_DT_BINS_QOLD"](), "removed 2026-11-15 (lane/expdel2-01-base): MOJOLEARN_DT_BINS_QOLD (rollback of the FAST DecisionTreeRegressor default to 128 quantile bins) is gone; RF_DT_DEFAULT_BINS is 256 under FAST, 128 otherwise (rab5-dtbins); restore: experiments/removed/MOJOLEARN_DT_BINS_QOLD.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_IF_RESIDENT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_IF_RESIDENT_OFF"](), "removed 2026-11-15 (lane/expdel2-01-base): MOJOLEARN_IDN_IF_RESIDENT_OFF (rollback of IsolationForest to a refit of the whole forest inside every scoring call) is gone, no A/B was ever recorded for it; the resident forest that fit built is the path on every tier; restore: experiments/removed/MOJOLEARN_IDN_IF_RESIDENT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_FOREST_DEVICE_FINITE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_FOREST_DEVICE_FINITE_OFF"](), "removed 2026-11-15 (lane/expdel2-05-base): FOREST_DEVICE_FINITE_OFF (FAST+Apple rollback of the forest fits to the host finite scans) is gone; the device scan is the FAST+Apple path"
    # MOJOLEARN_AFCL_T08: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_FOREST_SHARED_ROWS.patch
    comptime assert not is_defined["MOJOLEARN_FOREST_SHARED_ROWS"](), "removed 2026-11-15 (lane/expdel2-03-base): FOREST_SHARED_ROWS (Forest inference) is gone: no record: no A/B recorded for the opt-in arm (V2 rule a); code at a5502e594d; restore: experiments/removed/MOJOLEARN_FOREST_SHARED_ROWS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_FOREST_FAST_SHARED_ROWS.patch
    comptime assert not is_defined["MOJOLEARN_FOREST_FAST_SHARED_ROWS"](), "removed 2026-11-15 (lane/expdel2-03-base): FOREST_FAST_SHARED_ROWS (Forest inference (FAST Apple)) is gone: no record: no A/B recorded for the opt-in arm (V2 rule a); code at a5502e594d; restore: experiments/removed/MOJOLEARN_FOREST_FAST_SHARED_ROWS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T32_SHARED_ROWS.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T32_SHARED_ROWS"](), "removed 2026-11-15 (lane/expdel2-03-base): TREES_T32_SHARED_ROWS (Forest inference (IDENTICAL)) is gone: no record: no A/B recorded for the opt-in arm (V2 rule a); code at a5502e594d; restore: experiments/removed/MOJOLEARN_TREES_T32_SHARED_ROWS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_FOREST_SEPARATE_NODES.patch
    comptime assert not is_defined["MOJOLEARN_FOREST_SEPARATE_NODES"](), "removed 2026-11-15 (lane/expdel2-03-base): FOREST_SEPARATE_NODES (Forest inference) is gone: EXP:1504 DROP (defines deleted; the packed layout is the default); code at 77aaad1bcc; restore: experiments/removed/MOJOLEARN_FOREST_SEPARATE_NODES.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_FOREST_SCALAR_GROVES.patch
    comptime assert not is_defined["MOJOLEARN_FOREST_SCALAR_GROVES"](), "removed 2026-11-15 (lane/expdel2-03-base): FOREST_SCALAR_GROVES (Forest inference) is gone: no record: no A/B recorded for the opt-in arm (V2 rule a); code at b177479fc7; restore: experiments/removed/MOJOLEARN_FOREST_SCALAR_GROVES.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_PREDICT_ROW_OUTPUTS.patch
    comptime assert not is_defined["MOJOLEARN_TREES_PREDICT_ROW_OUTPUTS"](), "removed 2026-11-15 (lane/expdel2-03-base): TREES_PREDICT_ROW_OUTPUTS (Forest inference (IDENTICAL)) is gone: no record: no A/B recorded for the opt-in arm (V2 rule a); code at 432acbb3df; restore: experiments/removed/MOJOLEARN_TREES_PREDICT_ROW_OUTPUTS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T34_CHUNK_FOLD.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T34_CHUNK_FOLD"](), "removed 2026-11-15 (lane/expdel2-03-base): TREES_T34_CHUNK_FOLD (Forest inference (IDENTICAL, fold version)) is gone: no record: no A/B recorded for the opt-in arm (V2 rule a); code at cfad8788a0; restore: experiments/removed/MOJOLEARN_TREES_T34_CHUNK_FOLD.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_FOREST_ROUTE.patch
    comptime assert not is_defined["MOJOLEARN_TREES_FOREST_ROUTE"](), "removed 2026-11-15 (lane/expdel2-03-base): TREES_FOREST_ROUTE (Forest inference (IDENTICAL, routes)) is gone: no record: no A/B recorded for the opt-in arm (V2 rule a); code at c44dc3c017; restore: experiments/removed/MOJOLEARN_TREES_FOREST_ROUTE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T33_COST_SCHEDULE.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T33_COST_SCHEDULE"](), "removed 2026-11-15 (lane/expdel2-03-base): TREES_T33_COST_SCHEDULE (Forest inference (IDENTICAL, schedule)) is gone: no record: no A/B recorded for the opt-in arm (V2 rule a); code at 1cad80e23b; restore: experiments/removed/MOJOLEARN_TREES_T33_COST_SCHEDULE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T39_AUXILIARY_WALK.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T39_AUXILIARY_WALK"](), "removed 2026-11-15 (lane/expdel2-03-base): TREES_T39_AUXILIARY_WALK (Forest auxiliary outputs (IDENTICAL)) is gone: no record: no A/B recorded for the opt-in arm (V2 rule a); code at 96ee21ad13; restore: experiments/removed/MOJOLEARN_TREES_T39_AUXILIARY_WALK.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T44_METADATA_CACHE.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T44_METADATA_CACHE"](), "removed 2026-11-15 (lane/expdel2-03-base): TREES_T44_METADATA_CACHE (Tree SHAP metadata (IDENTICAL)) is gone: no record: no A/B recorded for the opt-in arm (V2 rule a); code at 3e9cbbd919; restore: experiments/removed/MOJOLEARN_TREES_T44_METADATA_CACHE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T37_WORKSPACE.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T37_WORKSPACE"](), "removed 2026-11-15 (lane/expdel2-03-base): TREES_T37_WORKSPACE (Forest inference (IDENTICAL, workspace)) is gone: no record: no A/B recorded for the opt-in arm (V2 rule a); code at 07f7a0733b; restore: experiments/removed/MOJOLEARN_TREES_T37_WORKSPACE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T38_FUSED_LABELS.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T38_FUSED_LABELS"](), "removed 2026-11-15 (lane/expdel2-03-base): TREES_T38_FUSED_LABELS (Forest inference (IDENTICAL, labels)) is gone: no record: no A/B recorded for the opt-in arm (V2 rule a); code at 81c34fb4fb; restore: experiments/removed/MOJOLEARN_TREES_T38_FUSED_LABELS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_C50_GB_PACKED.patch
    comptime assert not is_defined["MOJOLEARN_TREES_C50_GB_PACKED"](), "removed 2026-11-15 (lane/expdel2-03-base): TREES_C50_GB_PACKED (GBDT predict (IDENTICAL, packed nodes)) is gone: no record: no A/B recorded for the opt-in arm (V2 rule a); code at 03a43eb1ec; restore: experiments/removed/MOJOLEARN_TREES_C50_GB_PACKED.patch"
    # MOJOLEARN_AFT_P01: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_P02: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_AFT_P03.patch
    comptime assert not is_defined["MOJOLEARN_AFT_P03"](), "removed 2026-11-15 (lane/expdel2-03-base): AFT_P03 (Apple FAST trees programme (P)) is gone: never-run Apple FAST programme card, no A/B (V2 rule g); code at 5bc2865240; restore: experiments/removed/MOJOLEARN_AFT_P03.patch"
    # MOJOLEARN_AFT_P04: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_P05: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_P06: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_P07: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_P08: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_P09: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_P10: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_P11: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_P12: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # Lane expdel2-04-base (2026-11-15): unmeasured arms and old paths deleted under "An unmeasured arm is deleted" (plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history), docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_FOREST_ORDERED_RESIDENT.patch
    comptime assert not is_defined["MOJOLEARN_FOREST_ORDERED_RESIDENT"](), "removed 2026-11-15 (lane/expdel2-04-base): FOREST_ORDERED_RESIDENT (opt-in arm, Forest inference (ordered resident policy); no A/B recorded for the opt-in arm (rule a)) is gone; the default stays; code recoverable at 39d9cbb242; restore: experiments/removed/MOJOLEARN_FOREST_ORDERED_RESIDENT.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_FOREST_ORDERED_RESIDENT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_FOREST_ORDERED_RESIDENT_OFF"](), "removed 2026-11-15 (lane/expdel2-04-base): FOREST_ORDERED_RESIDENT_OFF (old path, Forest inference (ordered resident policy); no A/B recorded beside the default (rule b)) is gone; the default stays; code recoverable at 057d3cc56b; restore: experiments/removed/MOJOLEARN_FOREST_ORDERED_RESIDENT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_FOREST_PER_MODEL_IO.patch
    comptime assert not is_defined["MOJOLEARN_FOREST_PER_MODEL_IO"](), "removed 2026-11-15 (lane/expdel2-04-base): FOREST_PER_MODEL_IO (opt-in arm, Forest inference (shared registry IO); no A/B recorded for the opt-in arm (rule a)) is gone; the default stays; code recoverable at c9c333f2d7; restore: experiments/removed/MOJOLEARN_FOREST_PER_MODEL_IO.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_FOREST_PINNED_STAGE.patch
    comptime assert not is_defined["MOJOLEARN_FOREST_PINNED_STAGE"](), "removed 2026-11-15 (lane/expdel2-04-base): FOREST_PINNED_STAGE (opt-in arm, Forest inference (pinned host stage); no A/B recorded for the opt-in arm (rule a)) is gone; the default stays; code recoverable at 28a3c1be2b; restore: experiments/removed/MOJOLEARN_FOREST_PINNED_STAGE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_RF_FAST_BATCH32K.patch
    comptime assert not is_defined["MOJOLEARN_RF_FAST_BATCH32K"](), "removed 2026-11-15 (lane/expdel2-04-base): RF_FAST_BATCH32K (opt-in arm, RandomForest fit node batch (FAST Apple); no A/B recorded (Apple fast opt-in, rule a)) is gone; the default stays; code recoverable at 1f8a4570c6; restore: experiments/removed/MOJOLEARN_RF_FAST_BATCH32K.patch"
    # ---- end lane expdel2-04-base
    # Lane expdel2-08-base (2026-11-15): unmeasured arms and superseded old paths deleted, plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history) (docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_TREE_SHAPE_NATIVE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_TREE_SHAPE_NATIVE_OFF"](), "removed 2026-11-15 (lane/expdel2-08-base): IDN_TREE_SHAPE_NATIVE_OFF (unmeasured old path) is gone; the default is kept; code at 3f0e029eb4; restore: experiments/removed/MOJOLEARN_IDN_TREE_SHAPE_NATIVE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TE_NATIVE_SPLITS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_TE_NATIVE_SPLITS_OFF"](), "removed 2026-11-15 (lane/expdel2-08-base): TE_NATIVE_SPLITS_OFF (old path of the FAST Apple default) is gone; the default is kept; code at 6037d8cab6; restore: experiments/removed/MOJOLEARN_TE_NATIVE_SPLITS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TE_ADA_SESSION_SHARE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_TE_ADA_SESSION_SHARE_OFF"](), "removed 2026-11-15 (lane/expdel2-08-base): TE_ADA_SESSION_SHARE_OFF (old path of the FAST Apple default) is gone; the default is kept; code at 1eebdbdc8f; restore: experiments/removed/MOJOLEARN_TE_ADA_SESSION_SHARE_OFF.patch"
    # Lane expdel2-39-trees (2026-11-15): unmeasured arms and superseded old paths deleted, plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history) (docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_2010_ROWS_SORTED.patch
    comptime assert not is_defined["MOJOLEARN_2010_ROWS_SORTED"](), "removed 2026-11-15 (lane/expdel2-39-trees): 2010_ROWS_SORTED (opt-in arm) is gone; the default is kept; code at c2f33a3d2c^; restore: experiments/removed/MOJOLEARN_2010_ROWS_SORTED.patch"
    # MOJOLEARN_ET_DEVICE_BATCH_16384: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_ET_DEVICE_BATCH_32768.patch
    comptime assert not is_defined["MOJOLEARN_ET_DEVICE_BATCH_32768"](), "removed 2026-11-15 (lane/expdel2-39-trees): ET_DEVICE_BATCH_32768 (opt-in arm) is gone; the default is kept; code at 7264145906^; restore: experiments/removed/MOJOLEARN_ET_DEVICE_BATCH_32768.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_DEVICE_BATCH_4096.patch
    comptime assert not is_defined["MOJOLEARN_ET_DEVICE_BATCH_4096"](), "removed 2026-11-15 (lane/expdel2-39-trees): ET_DEVICE_BATCH_4096 (opt-in arm) is gone; the default is kept; code at d643565405^; restore: experiments/removed/MOJOLEARN_ET_DEVICE_BATCH_4096.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_RF_EXP_NO_SPLIT_MUTEX.patch
    comptime assert not is_defined["MOJOLEARN_RF_EXP_NO_SPLIT_MUTEX"](), "removed 2026-11-15 (lane/expdel2-39-trees): RF_EXP_NO_SPLIT_MUTEX (opt-in arm) is gone; the default is kept; code at ef82188860^; restore: experiments/removed/MOJOLEARN_RF_EXP_NO_SPLIT_MUTEX.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_RF_FUSED_BOOTSTRAP_GATHER.patch
    comptime assert not is_defined["MOJOLEARN_RF_FUSED_BOOTSTRAP_GATHER"](), "removed 2026-11-15 (lane/expdel2-39-trees): RF_FUSED_BOOTSTRAP_GATHER (opt-in arm) is gone; the default is kept; code at f1b48eea9f^; restore: experiments/removed/MOJOLEARN_RF_FUSED_BOOTSTRAP_GATHER.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_RF_FUSED_BOOTSTRAP_GATHER_OFF.patch
    comptime assert not is_defined["MOJOLEARN_RF_FUSED_BOOTSTRAP_GATHER_OFF"](), "removed 2026-11-15 (lane/expdel2-39-trees): RF_FUSED_BOOTSTRAP_GATHER_OFF (old path) is gone; the default is kept; code at 926c443542^; restore: experiments/removed/MOJOLEARN_RF_FUSED_BOOTSTRAP_GATHER_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_RF_ROWS_SORTED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_RF_ROWS_SORTED_OFF"](), "removed 2026-11-15 (lane/expdel2-39-trees): RF_ROWS_SORTED_OFF (old path) is gone; the default is kept; code at f11f6b1142^; restore: experiments/removed/MOJOLEARN_RF_ROWS_SORTED_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_C48.patch
    comptime assert not is_defined["MOJOLEARN_TREES_C48"](), "removed 2026-11-15 (lane/expdel2-39-trees): TREES_C48 (opt-in arm) is gone; the default is kept; code at faf4eceab2^; restore: experiments/removed/MOJOLEARN_TREES_C48.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_RF_SAMPLE.patch
    comptime assert not is_defined["MOJOLEARN_TREES_RF_SAMPLE"](), "removed 2026-11-15 (lane/expdel2-39-trees): TREES_RF_SAMPLE (opt-in arm) is gone; the default is kept; code at e76f952880^; restore: experiments/removed/MOJOLEARN_TREES_RF_SAMPLE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T02.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T02"](), "removed 2026-11-15 (lane/expdel2-39-trees): TREES_T02 (opt-in arm) is gone; the default is kept; code at 2a4a84c366^; restore: experiments/removed/MOJOLEARN_TREES_T02.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T03.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T03"](), "removed 2026-11-15 (lane/expdel2-39-trees): TREES_T03 (opt-in arm) is gone; the default is kept; code at d8533691e0^; restore: experiments/removed/MOJOLEARN_TREES_T03.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T04.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T04"](), "removed 2026-11-15 (lane/expdel2-39-trees): TREES_T04 (opt-in arm) is gone; the default is kept; code at cd242e8278^; restore: experiments/removed/MOJOLEARN_TREES_T04.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T05.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T05"](), "removed 2026-11-15 (lane/expdel2-39-trees): TREES_T05 (opt-in arm) is gone; the default is kept; code at 53f8571104^; restore: experiments/removed/MOJOLEARN_TREES_T05.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T05_ROW_VISITS.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T05_ROW_VISITS"](), "removed 2026-11-15 (lane/expdel2-39-trees): TREES_T05_ROW_VISITS (opt-in knob) is gone; the default is kept; code at 559154b9d2^; restore: experiments/removed/MOJOLEARN_TREES_T05_ROW_VISITS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T06.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T06"](), "removed 2026-11-15 (lane/expdel2-39-trees): TREES_T06 (opt-in arm) is gone; the default is kept; code at 1573ad87b6^; restore: experiments/removed/MOJOLEARN_TREES_T06.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T07.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T07"](), "removed 2026-11-15 (lane/expdel2-39-trees): TREES_T07 (opt-in arm) is gone; the default is kept; code at 793698a946^; restore: experiments/removed/MOJOLEARN_TREES_T07.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T13.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T13"](), "removed 2026-11-15 (lane/expdel2-39-trees): TREES_T13 (opt-in arm) is gone; the default is kept; code at 14fd08ad88^; restore: experiments/removed/MOJOLEARN_TREES_T13.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T13_BYTES.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T13_BYTES"](), "removed 2026-11-15 (lane/expdel2-39-trees): TREES_T13_BYTES (opt-in knob) is gone; the default is kept; code at e4afdba38b^; restore: experiments/removed/MOJOLEARN_TREES_T13_BYTES.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T11_LEVELS.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T11_LEVELS"](), "removed 2026-11-15 (lane/expdel2-39-trees): TREES_T11_LEVELS (opt-in arm) is gone; the default is kept; code at fbc7483ab3^; restore: experiments/removed/MOJOLEARN_TREES_T11_LEVELS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T08.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T08"](), "removed 2026-11-15 (lane/expdel2-39-trees): TREES_T08 (opt-in arm) is gone; the default is kept; code at 5a385aa685^; restore: experiments/removed/MOJOLEARN_TREES_T08.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T08_LAYOUT.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T08_LAYOUT"](), "removed 2026-11-15 (lane/expdel2-39-trees): TREES_T08_LAYOUT (opt-in arm) is gone; the default is kept; code at 5a385aa685^; restore: experiments/removed/MOJOLEARN_TREES_T08_LAYOUT.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T08_BITS.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T08_BITS"](), "removed 2026-11-15 (lane/expdel2-39-trees): TREES_T08_BITS (opt-in arm) is gone; the default is kept; code at 5a385aa685^; restore: experiments/removed/MOJOLEARN_TREES_T08_BITS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T12.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T12"](), "removed 2026-11-15 (lane/expdel2-39-trees): TREES_T12 (opt-in arm) is gone; the default is kept; code at b9e0a62714^; restore: experiments/removed/MOJOLEARN_TREES_T12.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T12_BYTES.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T12_BYTES"](), "removed 2026-11-15 (lane/expdel2-39-trees): TREES_T12_BYTES (opt-in arm) is gone; the default is kept; code at b9e0a62714^; restore: experiments/removed/MOJOLEARN_TREES_T12_BYTES.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T14.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T14"](), "removed 2026-11-15 (lane/expdel2-39-trees): TREES_T14 (opt-in arm) is gone; the default is kept; code at e4c227042e^; restore: experiments/removed/MOJOLEARN_TREES_T14.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T14_EXACT.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T14_EXACT"](), "removed 2026-11-15 (lane/expdel2-39-trees): TREES_T14_EXACT (opt-in arm) is gone; the default is kept; code at e4c227042e^; restore: experiments/removed/MOJOLEARN_TREES_T14_EXACT.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T15.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T15"](), "removed 2026-11-15 (lane/expdel2-39-trees): TREES_T15 (opt-in arm) is gone; the default is kept; code at bcdf8b62a8^; restore: experiments/removed/MOJOLEARN_TREES_T15.patch"
    # Lane expdel2-38-trees (2026-11-15): unmeasured arms and old paths deleted, plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history) (docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_2011_HIST_ITEMS1.patch
    comptime assert not is_defined["MOJOLEARN_2011_HIST_ITEMS1"](), "removed 2026-11-15 (lane/expdel2-38-trees): 2011_HIST_ITEMS1 (old path) is gone; the default is kept; code at 52922acecb; restore: experiments/removed/MOJOLEARN_2011_HIST_ITEMS1.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_2012_SMEM_COPIES4.patch
    comptime assert not is_defined["MOJOLEARN_2012_SMEM_COPIES4"](), "removed 2026-11-15 (lane/expdel2-38-trees): 2012_SMEM_COPIES4 (arm) is gone; the default path is the only one; code at 52922acecb; restore: experiments/removed/MOJOLEARN_2012_SMEM_COPIES4.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_2502_RETRY_PURE.patch
    comptime assert not is_defined["MOJOLEARN_2502_RETRY_PURE"](), "removed 2026-11-15 (lane/expdel2-38-trees): 2502_RETRY_PURE (old path) is gone; the default is kept; code at 52922acecb; restore: experiments/removed/MOJOLEARN_2502_RETRY_PURE.patch"
    # MOJOLEARN_AFT_F01: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_F02: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_F03: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_F04: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_F05: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_RF_COLS40_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_RF_COLS40_OFF"](), "removed 2026-11-15 (lane/expdel2-38-trees): IDN_RF_COLS40_OFF (old path) is gone; the default is kept; code at 52922acecb; restore: experiments/removed/MOJOLEARN_IDN_RF_COLS40_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_RF_FUSED_PARTITION_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_RF_FUSED_PARTITION_OFF"](), "removed 2026-11-15 (lane/expdel2-38-trees): IDN_RF_FUSED_PARTITION_OFF (old path) is gone; the default is kept; code at 52922acecb; restore: experiments/removed/MOJOLEARN_IDN_RF_FUSED_PARTITION_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_RF_HIST_SIMD_AGG_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_RF_HIST_SIMD_AGG_OFF"](), "removed 2026-11-15 (lane/expdel2-38-trees): IDN_RF_HIST_SIMD_AGG_OFF (old path) is gone; the default is kept; code at 52922acecb; restore: experiments/removed/MOJOLEARN_IDN_RF_HIST_SIMD_AGG_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_RF_HIST_ZERO_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_RF_HIST_ZERO_OFF"](), "removed 2026-11-15 (lane/expdel2-38-trees): IDN_RF_HIST_ZERO_OFF (old path) is gone; the default is kept; code at 52922acecb; restore: experiments/removed/MOJOLEARN_IDN_RF_HIST_ZERO_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_RF_QBIN_DEVICE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_RF_QBIN_DEVICE_OFF"](), "removed 2026-11-15 (lane/expdel2-38-trees): IDN_RF_QBIN_DEVICE_OFF (old path) is gone; the default is kept; code at 52922acecb; restore: experiments/removed/MOJOLEARN_IDN_RF_QBIN_DEVICE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_RF_SAMPLE_PER_NODE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_RF_SAMPLE_PER_NODE_OFF"](), "removed 2026-11-15 (lane/expdel2-38-trees): IDN_RF_SAMPLE_PER_NODE_OFF (old path) is gone; the default is kept; code at 52922acecb; restore: experiments/removed/MOJOLEARN_IDN_RF_SAMPLE_PER_NODE_OFF.patch"
    # MOJOLEARN_RF_BINS_COLUMN_MAJOR: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_RF_COLS10: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_RF_FAST_HIST_ZERO_OFF.patch
    comptime assert not is_defined["MOJOLEARN_RF_FAST_HIST_ZERO_OFF"](), "removed 2026-11-15 (lane/expdel2-38-trees): RF_FAST_HIST_ZERO_OFF (old path) is gone; the default is kept; code at 52922acecb; restore: experiments/removed/MOJOLEARN_RF_FAST_HIST_ZERO_OFF.patch"
    # MOJOLEARN_RF_FAST_SAMPLE_PER_COLUMN: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_RF_FBS_WARP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_RF_FBS_WARP_OFF"](), "removed 2026-11-15 (lane/expdel2-38-trees): RF_FBS_WARP_OFF (old path) is gone; the default is kept; code at 52922acecb; restore: experiments/removed/MOJOLEARN_RF_FBS_WARP_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_RF_HIST_COLUMNS2.patch
    comptime assert not is_defined["MOJOLEARN_RF_HIST_COLUMNS2"](), "removed 2026-11-15 (lane/expdel2-38-trees): RF_HIST_COLUMNS2 (arm) is gone; the default path is the only one; code at 52922acecb; restore: experiments/removed/MOJOLEARN_RF_HIST_COLUMNS2.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_RF_HIST_COLUMNS4.patch
    comptime assert not is_defined["MOJOLEARN_RF_HIST_COLUMNS4"](), "removed 2026-11-15 (lane/expdel2-38-trees): RF_HIST_COLUMNS4 (arm) is gone; the default path is the only one; code at 52922acecb; restore: experiments/removed/MOJOLEARN_RF_HIST_COLUMNS4.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_RF_HIST_COLUMNS4_OFF.patch
    comptime assert not is_defined["MOJOLEARN_RF_HIST_COLUMNS4_OFF"](), "removed 2026-11-15 (lane/expdel2-38-trees): RF_HIST_COLUMNS4_OFF (old path) is gone; the default is kept; code at 52922acecb; restore: experiments/removed/MOJOLEARN_RF_HIST_COLUMNS4_OFF.patch"
    # MOJOLEARN_RF_HIST_COLUMNS8: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_RF_HIST_COLUMNS8_OFF.patch
    comptime assert not is_defined["MOJOLEARN_RF_HIST_COLUMNS8_OFF"](), "removed 2026-11-15 (lane/expdel2-38-trees): RF_HIST_COLUMNS8_OFF (old path) is gone; the default is kept; code at 52922acecb; restore: experiments/removed/MOJOLEARN_RF_HIST_COLUMNS8_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_RF_HIST_SIMD_AGG_OFF.patch
    comptime assert not is_defined["MOJOLEARN_RF_HIST_SIMD_AGG_OFF"](), "removed 2026-11-15 (lane/expdel2-38-trees): RF_HIST_SIMD_AGG_OFF (old path) is gone; the default is kept; code at 52922acecb; restore: experiments/removed/MOJOLEARN_RF_HIST_SIMD_AGG_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_RF_SMALL_NODE_FUSED.patch
    comptime assert not is_defined["MOJOLEARN_RF_SMALL_NODE_FUSED"](), "removed 2026-11-15 (lane/expdel2-38-trees): RF_SMALL_NODE_FUSED (arm) is gone; the default path is the only one; code at 52922acecb; restore: experiments/removed/MOJOLEARN_RF_SMALL_NODE_FUSED.patch"
    # MOJOLEARN_RF_SPLIT_CANDIDATES: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_RF_SPLIT_CANDIDATES_OFF.patch
    comptime assert not is_defined["MOJOLEARN_RF_SPLIT_CANDIDATES_OFF"](), "removed 2026-11-15 (lane/expdel2-38-trees): RF_SPLIT_CANDIDATES_OFF (old path) is gone; the default is kept; code at 52922acecb; restore: experiments/removed/MOJOLEARN_RF_SPLIT_CANDIDATES_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_RF_TRIAL_COLS20.patch
    comptime assert not is_defined["MOJOLEARN_RF_TRIAL_COLS20"](), "removed 2026-11-15 (lane/expdel2-38-trees): RF_TRIAL_COLS20 (arm) is gone; the default path is the only one; code at 52922acecb; restore: experiments/removed/MOJOLEARN_RF_TRIAL_COLS20.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_RF_TRIAL_COLS40.patch
    comptime assert not is_defined["MOJOLEARN_RF_TRIAL_COLS40"](), "removed 2026-11-15 (lane/expdel2-38-trees): RF_TRIAL_COLS40 (arm) is gone; the default path is the only one; code at 52922acecb; restore: experiments/removed/MOJOLEARN_RF_TRIAL_COLS40.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREE_EXACT_SIBLING_HIST.patch
    comptime assert not is_defined["MOJOLEARN_TREE_EXACT_SIBLING_HIST"](), "removed 2026-11-15 (lane/expdel2-38-trees): TREE_EXACT_SIBLING_HIST (arm) is gone; the default path is the only one; code at 52922acecb; restore: experiments/removed/MOJOLEARN_TREE_EXACT_SIBLING_HIST.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREE_EXACT_SIBLING_HIST_AUDIT.patch
    comptime assert not is_defined["MOJOLEARN_TREE_EXACT_SIBLING_HIST_AUDIT"](), "removed 2026-11-15 (lane/expdel2-38-trees): TREE_EXACT_SIBLING_HIST_AUDIT (arm) is gone; the default path is the only one; code at 52922acecb; restore: experiments/removed/MOJOLEARN_TREE_EXACT_SIBLING_HIST_AUDIT.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREE_EXACT_SIBLING_HIST_BYTES.patch
    comptime assert not is_defined["MOJOLEARN_TREE_EXACT_SIBLING_HIST_BYTES"](), "removed 2026-11-15 (lane/expdel2-38-trees): TREE_EXACT_SIBLING_HIST_BYTES (arm) is gone; the default path is the only one; code at 52922acecb; restore: experiments/removed/MOJOLEARN_TREE_EXACT_SIBLING_HIST_BYTES.patch"
    # Lane expdel2-40-trees (2026-11-15): unmeasured arms and old paths deleted under "An unmeasured arm is deleted" (plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history), docs/TOMBSTONES.md).
    # MOJOLEARN_AFCL_T09: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_F06: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_F07: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_F08: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_ET_BINNED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_ET_BINNED_OFF"](), "removed 2026-11-15 (lane/expdel2-40-trees): ET_BINNED_OFF (old path, Trees (ExtraTrees FAST Apple float X instead of binned codes); no record (rule b)) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_ET_BINNED_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_DEVICE_LOOP_K1.patch
    comptime assert not is_defined["MOJOLEARN_ET_DEVICE_LOOP_K1"](), "removed 2026-11-15 (lane/expdel2-40-trees): ET_DEVICE_LOOP_K1 (opt-in arm, Trees (ExtraTrees batches per header drain = 1); no record (rule a)) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_ET_DEVICE_LOOP_K1.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_DEVICE_LOOP_K2.patch
    comptime assert not is_defined["MOJOLEARN_ET_DEVICE_LOOP_K2"](), "removed 2026-11-15 (lane/expdel2-40-trees): ET_DEVICE_LOOP_K2 (opt-in arm, Trees (ExtraTrees batches per header drain = 2); no record (rule a)) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_ET_DEVICE_LOOP_K2.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_DEVICE_LOOP_K8.patch
    comptime assert not is_defined["MOJOLEARN_ET_DEVICE_LOOP_K8"](), "removed 2026-11-15 (lane/expdel2-40-trees): ET_DEVICE_LOOP_K8 (opt-in arm, Trees (ExtraTrees batches per header drain = 8); no record (rule a)) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_ET_DEVICE_LOOP_K8.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_MAX_ACC_16.patch
    comptime assert not is_defined["MOJOLEARN_ET_MAX_ACC_16"](), "removed 2026-11-15 (lane/expdel2-40-trees): ET_MAX_ACC_16 (opt-in arm, Trees (ExtraTrees fixed 16-wide score accumulator); no record (rule a)) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_ET_MAX_ACC_16.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_MAX_ACC_4.patch
    comptime assert not is_defined["MOJOLEARN_ET_MAX_ACC_4"](), "removed 2026-11-15 (lane/expdel2-40-trees): ET_MAX_ACC_4 (opt-in arm, Trees (ExtraTrees fixed 4-wide score accumulator); no record (rule a)) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_ET_MAX_ACC_4.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_MAX_ACC_8.patch
    comptime assert not is_defined["MOJOLEARN_ET_MAX_ACC_8"](), "removed 2026-11-15 (lane/expdel2-40-trees): ET_MAX_ACC_8 (opt-in arm, Trees (ExtraTrees fixed 8-wide score accumulator); no record (rule a)) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_ET_MAX_ACC_8.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_PART_FLAGS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_ET_PART_FLAGS_OFF"](), "removed 2026-11-15 (lane/expdel2-40-trees): ET_PART_FLAGS_OFF (old path, Trees (ExtraTrees partition second gather); no record (rule b)) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_ET_PART_FLAGS_OFF.patch"
    # MOJOLEARN_ET_PART_ROWS: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_ET_PART_ROWS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_ET_PART_ROWS_OFF"](), "removed 2026-11-15 (lane/expdel2-40-trees): ET_PART_ROWS_OFF (old path, Trees (ExtraTrees FAST Apple one row per thread partition); EXP:105 KEPT 269ffa57a (rule d): promoted on Apple M3 Ultra only; NVIDIA/AMD A/B owed but FAST is Apple only) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_ET_PART_ROWS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_RANGE_TILED.patch
    comptime assert not is_defined["MOJOLEARN_ET_RANGE_TILED"](), "removed 2026-11-15 (lane/expdel2-40-trees): ET_RANGE_TILED (opt-in arm, Trees (ExtraTrees FAST tiled range pass off Apple); no record (rule a)) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_ET_RANGE_TILED.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_RM_LINE_HW_OFF.patch
    comptime assert not is_defined["MOJOLEARN_ET_RM_LINE_HW_OFF"](), "removed 2026-11-15 (lane/expdel2-40-trees): ET_RM_LINE_HW_OFF (old path, Trees (ExtraTrees row-major 64-byte line); no record (rule b)) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_ET_RM_LINE_HW_OFF.patch"
    # MOJOLEARN_ET_RM_NARROW: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_ET_RM_NARROW_OFF.patch
    comptime assert not is_defined["MOJOLEARN_ET_RM_NARROW_OFF"](), "removed 2026-11-15 (lane/expdel2-40-trees): ET_RM_NARROW_OFF (old path, Trees (ExtraTrees Apple narrow-row row-major copy off); no record (rule b)) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_ET_RM_NARROW_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_ROW_MAJOR.patch
    comptime assert not is_defined["MOJOLEARN_ET_ROW_MAJOR"](), "removed 2026-11-15 (lane/expdel2-40-trees): ET_ROW_MAJOR (opt-in arm, Trees (ExtraTrees FAST feature-major search grid); no record (rule a)) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_ET_ROW_MAJOR.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_SCORE_TILED.patch
    comptime assert not is_defined["MOJOLEARN_ET_SCORE_TILED"](), "removed 2026-11-15 (lane/expdel2-40-trees): ET_SCORE_TILED (opt-in arm, Trees (ExtraTrees FAST tiled regression score off Apple); no record (rule a)) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_ET_SCORE_TILED.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_STAGE_FULL_CAPACITY.patch
    comptime assert not is_defined["MOJOLEARN_ET_STAGE_FULL_CAPACITY"](), "removed 2026-11-15 (lane/expdel2-40-trees): ET_STAGE_FULL_CAPACITY (opt-in arm, Trees (ExtraTrees full-capacity batch staging); no record (rule a)) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_ET_STAGE_FULL_CAPACITY.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_TILED_SEARCH_IDENTICAL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_ET_TILED_SEARCH_IDENTICAL_OFF"](), "removed 2026-11-15 (lane/expdel2-40-trees): ET_TILED_SEARCH_IDENTICAL_OFF (old path, Trees (ExtraTrees Apple IDENTICAL one-feature search kernels); no record (rule b)) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_ET_TILED_SEARCH_IDENTICAL_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_TILED_SEARCH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_ET_TILED_SEARCH_OFF"](), "removed 2026-11-15 (lane/expdel2-40-trees): ET_TILED_SEARCH_OFF (old path, Trees (ExtraTrees Apple one-feature search kernels); no record (rule b)) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_ET_TILED_SEARCH_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_TPB_1024.patch
    comptime assert not is_defined["MOJOLEARN_ET_TPB_1024"](), "removed 2026-11-15 (lane/expdel2-40-trees): ET_TPB_1024 (opt-in arm, Trees (ExtraTrees 1024-thread frontier block); no record (rule a)) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_ET_TPB_1024.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_TPB_128.patch
    comptime assert not is_defined["MOJOLEARN_ET_TPB_128"](), "removed 2026-11-15 (lane/expdel2-40-trees): ET_TPB_128 (opt-in arm, Trees (ExtraTrees 128-thread frontier block on 64-lane devices); no record (rule a)) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_ET_TPB_128.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_TPB_512.patch
    comptime assert not is_defined["MOJOLEARN_ET_TPB_512"](), "removed 2026-11-15 (lane/expdel2-40-trees): ET_TPB_512 (opt-in arm, Trees (ExtraTrees 512-thread frontier block on 32-lane devices); no record (rule a)) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_ET_TPB_512.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ET_PART_ROWS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ET_PART_ROWS_OFF"](), "removed 2026-11-15 (lane/expdel2-40-trees): IDN_ET_PART_ROWS_OFF (old path, Trees (ExtraTrees IDENTICAL NVIDIA/AMD one row per thread partition); no record (rule b)) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_IDN_ET_PART_ROWS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ET_STAGE_LIVE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ET_STAGE_LIVE_OFF"](), "removed 2026-11-15 (lane/expdel2-40-trees): IDN_ET_STAGE_LIVE_OFF (old path, Trees (ExtraTrees IDENTICAL full-capacity staging); no record (rule b)) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_IDN_ET_STAGE_LIVE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ET_TILED_SEARCH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ET_TILED_SEARCH_OFF"](), "removed 2026-11-15 (lane/expdel2-40-trees): IDN_ET_TILED_SEARCH_OFF (old path, Trees (ExtraTrees IDENTICAL NVIDIA/AMD one-feature search kernels); no record (rule b)) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_IDN_ET_TILED_SEARCH_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_LEGACY_NARROW_ET_BINNED.patch
    comptime assert not is_defined["MOJOLEARN_LEGACY_NARROW_ET_BINNED"](), "removed 2026-11-15 (lane/expdel2-40-trees): LEGACY_NARROW_ET_BINNED (old path, Trees (ExtraTrees FAST Apple binning only at n_cols >= 64); no record (rule b)) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_LEGACY_NARROW_ET_BINNED.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_LEGACY_NARROW_ET_RM.patch
    comptime assert not is_defined["MOJOLEARN_LEGACY_NARROW_ET_RM"](), "removed 2026-11-15 (lane/expdel2-40-trees): LEGACY_NARROW_ET_RM (old path, Trees (ExtraTrees FAST quarter gate for the row-major copy); EXP:1330 OWED, never run (rule b)) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_LEGACY_NARROW_ET_RM.patch"
    # Lane expdel2-41-trees (2026-11-15): unmeasured arms, never-run cards and old paths deleted under "An unmeasured arm is deleted" (plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history), docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_ET_SEARCH_RPT_256.patch
    comptime assert not is_defined["MOJOLEARN_ET_SEARCH_RPT_256"](), "removed 2026-11-15 (lane/expdel2-41-trees): ET_SEARCH_RPT_256 (opt-in arm, Trees (ExtraTrees search rows per thread force knob 256); no A/B recorded for the opt-in knob (rule a)) is gone; the default path is the only one; code recoverable at 6641db4b18; restore: experiments/removed/MOJOLEARN_ET_SEARCH_RPT_256.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_SEARCH_RPT_128.patch
    comptime assert not is_defined["MOJOLEARN_ET_SEARCH_RPT_128"](), "removed 2026-11-15 (lane/expdel2-41-trees): ET_SEARCH_RPT_128 (opt-in arm, Trees (ExtraTrees search rows per thread force knob 128); no A/B recorded for the opt-in knob (rule a)) is gone; the default path is the only one; code recoverable at f0fbe787cf; restore: experiments/removed/MOJOLEARN_ET_SEARCH_RPT_128.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_SEARCH_RPT_64.patch
    comptime assert not is_defined["MOJOLEARN_ET_SEARCH_RPT_64"](), "removed 2026-11-15 (lane/expdel2-41-trees): ET_SEARCH_RPT_64 (opt-in arm, Trees (ExtraTrees search rows per thread force knob 64); no A/B recorded for the opt-in knob (rule a)) is gone; the default path is the only one; code recoverable at 7a3591d159; restore: experiments/removed/MOJOLEARN_ET_SEARCH_RPT_64.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_SEARCH_RPT_32.patch
    comptime assert not is_defined["MOJOLEARN_ET_SEARCH_RPT_32"](), "removed 2026-11-15 (lane/expdel2-41-trees): ET_SEARCH_RPT_32 (opt-in arm, Trees (ExtraTrees search rows per thread force knob 32); no A/B recorded for the opt-in knob (rule a)) is gone; the default path is the only one; code recoverable at b1119892ab; restore: experiments/removed/MOJOLEARN_ET_SEARCH_RPT_32.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_SEARCH_RPT_16.patch
    comptime assert not is_defined["MOJOLEARN_ET_SEARCH_RPT_16"](), "removed 2026-11-15 (lane/expdel2-41-trees): ET_SEARCH_RPT_16 (opt-in arm, Trees (ExtraTrees search rows per thread force knob 16); no A/B recorded for the opt-in knob (rule a)) is gone; the default path is the only one; code recoverable at c9254754a9; restore: experiments/removed/MOJOLEARN_ET_SEARCH_RPT_16.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_SEARCH_RPT_8.patch
    comptime assert not is_defined["MOJOLEARN_ET_SEARCH_RPT_8"](), "removed 2026-11-15 (lane/expdel2-41-trees): ET_SEARCH_RPT_8 (opt-in arm, Trees (ExtraTrees search rows per thread force knob 8); no A/B recorded for the opt-in knob (rule a)) is gone; the default path is the only one; code recoverable at 58dd5e245b; restore: experiments/removed/MOJOLEARN_ET_SEARCH_RPT_8.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_SEARCH_RPT_4.patch
    comptime assert not is_defined["MOJOLEARN_ET_SEARCH_RPT_4"](), "removed 2026-11-15 (lane/expdel2-41-trees): ET_SEARCH_RPT_4 (opt-in arm, Trees (ExtraTrees search rows per thread force knob 4); no A/B recorded for the opt-in knob (rule a)) is gone; the default path is the only one; code recoverable at 6bf147a71e; restore: experiments/removed/MOJOLEARN_ET_SEARCH_RPT_4.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_SEARCH_RPT_2.patch
    comptime assert not is_defined["MOJOLEARN_ET_SEARCH_RPT_2"](), "removed 2026-11-15 (lane/expdel2-41-trees): ET_SEARCH_RPT_2 (opt-in arm, Trees (ExtraTrees search rows per thread force knob 2); no A/B recorded for the opt-in knob (rule a)) is gone; the default path is the only one; code recoverable at 1e4bce29cc; restore: experiments/removed/MOJOLEARN_ET_SEARCH_RPT_2.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_SEARCH_RPT_1.patch
    comptime assert not is_defined["MOJOLEARN_ET_SEARCH_RPT_1"](), "removed 2026-11-15 (lane/expdel2-41-trees): ET_SEARCH_RPT_1 (opt-in arm, Trees (ExtraTrees search rows per thread force knob 1); no A/B recorded for the opt-in knob (rule a)) is gone; the default path is the only one; code recoverable at d1cfc6ff2b; restore: experiments/removed/MOJOLEARN_ET_SEARCH_RPT_1.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_EXP_NO_SEARCH_BARRIER.patch
    comptime assert not is_defined["MOJOLEARN_ET_EXP_NO_SEARCH_BARRIER"](), "removed 2026-11-15 (lane/expdel2-41-trees): ET_EXP_NO_SEARCH_BARRIER (opt-in arm, Trees (ExtraTrees search barrier measurement arm); no A/B recorded (rule a)) is gone; the default path is the only one; code recoverable at c989276b4b; restore: experiments/removed/MOJOLEARN_ET_EXP_NO_SEARCH_BARRIER.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_EXP_RACY_RANGE_PUBLISH.patch
    comptime assert not is_defined["MOJOLEARN_ET_EXP_RACY_RANGE_PUBLISH"](), "removed 2026-11-15 (lane/expdel2-41-trees): ET_EXP_RACY_RANGE_PUBLISH (opt-in arm, Trees (ExtraTrees plain-store range publish, wrong on purpose); no A/B recorded (rule a)) is gone; the default path is the only one; code recoverable at 03694cfdb9; restore: experiments/removed/MOJOLEARN_ET_EXP_RACY_RANGE_PUBLISH.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_RELAXED_PUBLISH.patch
    comptime assert not is_defined["MOJOLEARN_ET_RELAXED_PUBLISH"](), "removed 2026-11-15 (lane/expdel2-41-trees): ET_RELAXED_PUBLISH (opt-in arm, Trees (ExtraTrees relaxed publish opt-in for other columns); no A/B recorded (rule a)) is gone; the default path is the only one; code recoverable at 05759052d0; restore: experiments/removed/MOJOLEARN_ET_RELAXED_PUBLISH.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ET_SEQCST_PUBLISH.patch
    comptime assert not is_defined["MOJOLEARN_ET_SEQCST_PUBLISH"](), "removed 2026-11-15 (lane/expdel2-41-trees): ET_SEQCST_PUBLISH (opt-in arm, Trees (ExtraTrees seq_cst publish opt-out); no A/B recorded for the opt-out (rule a)) is gone; the default path is the only one; code recoverable at 7da85bac98; restore: experiments/removed/MOJOLEARN_ET_SEQCST_PUBLISH.patch"
    # MOJOLEARN_AFT_F09: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_ET_TIE_MAX_COLID.patch
    comptime assert not is_defined["MOJOLEARN_ET_TIE_MAX_COLID"](), "removed 2026-11-15 (lane/expdel2-41-trees): ET_TIE_MAX_COLID (opt-in arm, Trees (ExtraTrees max-colid tie rule); no A/B recorded (rule a)) is gone; the default path is the only one; code recoverable at 47831cc734; restore: experiments/removed/MOJOLEARN_ET_TIE_MAX_COLID.patch"
    # MOJOLEARN_AFCL_T01: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFCL_T02: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFCL_T03: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFCL_T04: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFCL_T05: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFCL_T06: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFCL_T07: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFCL_T12: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_N01: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_N02: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_N03: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_N04: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_N05: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_N06: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_N07: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_N08: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_N09: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_N10: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_N11: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_N12: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_SYM_CTR_ALL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SYM_CTR_ALL_OFF"](), "removed 2026-11-15 (lane/expdel2-41-trees): SYM_CTR_ALL_OFF (old path, Trees (CatBoost-style CTR prep umbrella rollback); EXP:1274 ACCEPT/DEFAULT FAST+Apple (rule d)) is gone; the default path is the only one; code recoverable at 2497c8a9cf; restore: experiments/removed/MOJOLEARN_SYM_CTR_ALL_OFF.patch"
    # MOJOLEARN_CTR_PREP_SHARED: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_CTR_SORT_ONCE: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_CTR_INDEX_FUSED: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_CTR_ONEHOT_DEVICE: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # ---- end lane expdel2-41-trees
    # Lane expdel2-45-trees (2026-11-15): unmeasured arms, never-run cards and old paths deleted under "An unmeasured arm is deleted" (plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history), docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_C50_IF_PACKED.patch
    comptime assert not is_defined["MOJOLEARN_TREES_C50_IF_PACKED"](), "removed 2026-11-15 (lane/expdel2-45-trees): TREES_C50_IF_PACKED (opt-in arm, IsolationForest (packed vector nodes for scoring); no record, rule a) is gone; the default path is the only one; code recoverable at 931e7ca76a; restore: experiments/removed/MOJOLEARN_TREES_C50_IF_PACKED.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T43_RADIX_SELECT.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T43_RADIX_SELECT"](), "removed 2026-11-15 (lane/expdel2-45-trees): TREES_T43_RADIX_SELECT (opt-in arm, IsolationForest (radix select for the contamination quantile); no record, rule a) is gone; the default path is the only one; code recoverable at b623cedc91; restore: experiments/removed/MOJOLEARN_TREES_T43_RADIX_SELECT.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_3040_YETI_SEQUENTIAL.patch
    comptime assert not is_defined["MOJOLEARN_3040_YETI_SEQUENTIAL"](), "removed 2026-11-15 (lane/expdel2-45-trees): 3040_YETI_SEQUENTIAL (opt-in arm, GBDT (YetiRank one-thread task kill switch); no record, rule a) is gone; the default path is the only one; code recoverable at 185ae2aa1b; restore: experiments/removed/MOJOLEARN_3040_YETI_SEQUENTIAL.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_3040_YETI_BLOCK.patch
    comptime assert not is_defined["MOJOLEARN_3040_YETI_BLOCK"](), "removed 2026-11-15 (lane/expdel2-45-trees): 3040_YETI_BLOCK (opt-in arm, GBDT (YetiRank block-kernel force-on); no record, rule a) is gone; the default path is the only one; code recoverable at ebec4a4e17; restore: experiments/removed/MOJOLEARN_3040_YETI_BLOCK.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GBDT_YETI_BLOCK_AMD_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GBDT_YETI_BLOCK_AMD_OFF"](), "removed 2026-11-15 (lane/expdel2-45-trees): IDN_GBDT_YETI_BLOCK_AMD_OFF (old path, GBDT (YetiRank AMD identical block kernel rollback); EXP:1584 DROP, define deleted (no-op), rule e) is gone; the default path is the only one; code recoverable at c25340638d; restore: experiments/removed/MOJOLEARN_IDN_GBDT_YETI_BLOCK_AMD_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_YETI_EST_REUSE_SEARCH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_YETI_EST_REUSE_SEARCH_OFF"](), "removed 2026-11-15 (lane/expdel2-45-trees): YETI_EST_REUSE_SEARCH_OFF (old path, GBDT (YetiRank FAST Apple estimation reuse rollback); EXP:125 KEPT 269ffa57a, rule d) is gone; the default path is the only one; code recoverable at 935028c9f7; restore: experiments/removed/MOJOLEARN_YETI_EST_REUSE_SEARCH_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_YETI_FAST_SORT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_YETI_FAST_SORT_OFF"](), "removed 2026-11-15 (lane/expdel2-45-trees): YETI_FAST_SORT_OFF (old path, GBDT (YetiRank FAST Apple simdgroup sort rollback); EXP:228 NEUTRAL, rule e) is gone; the default path is the only one; code recoverable at e04b81c34a; restore: experiments/removed/MOJOLEARN_YETI_FAST_SORT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_YETI_SEARCH_TASK16K_OFF.patch
    comptime assert not is_defined["MOJOLEARN_YETI_SEARCH_TASK16K_OFF"](), "removed 2026-11-15 (lane/expdel2-45-trees): YETI_SEARCH_TASK16K_OFF (old path, GBDT (YetiRank FAST Apple 16 KiB task kernel rollback); EXP:155 DROPPED-noise, rule e) is gone; the default path is the only one; code recoverable at eba0f3a418; restore: experiments/removed/MOJOLEARN_YETI_SEARCH_TASK16K_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_IF_QUERY_DEVICE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_IF_QUERY_DEVICE_OFF"](), "removed 2026-11-15 (lane/expdel2-45-trees): IDN_IF_QUERY_DEVICE_OFF (old path, IsolationForest (host List query route rollback); no record, rule b) is gone; the default path is the only one; code recoverable at 4f2861aaea; restore: experiments/removed/MOJOLEARN_IDN_IF_QUERY_DEVICE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_IF_EPILOGUE_DEVICE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_IF_EPILOGUE_DEVICE_OFF"](), "removed 2026-11-15 (lane/expdel2-45-trees): IDN_IF_EPILOGUE_DEVICE_OFF (old path, IsolationForest (host scoring epilogue rollback); no record, rule b) is gone; the default path is the only one; code recoverable at 0454762fc4; restore: experiments/removed/MOJOLEARN_IDN_IF_EPILOGUE_DEVICE_OFF.patch"
    # MOJOLEARN_TREES_HIST_PACKED_GH: refusal removed 2026-11-15 (restored by lane/restore-run-arms-trees; RUN arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_TREES_HIST_WARP_AGG: refusal removed 2026-11-15 (restored by lane/restore-run-arms-trees; RUN arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_HIST_REP_SM_OFF.patch
    comptime assert not is_defined["MOJOLEARN_TREES_HIST_REP_SM_OFF"](), "removed 2026-11-15 (lane/expdel2-45-trees): TREES_HIST_REP_SM_OFF (old path, GBDT (pinned-32 histogram replicas rollback); EXP:1640 PROMOTED, rule e) is gone; the default path is the only one; code recoverable at 70fbbc4c47; restore: experiments/removed/MOJOLEARN_TREES_HIST_REP_SM_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_HIST_REP_SM.patch
    comptime assert not is_defined["MOJOLEARN_TREES_HIST_REP_SM"](), "removed 2026-11-15 (lane/expdel2-45-trees): TREES_HIST_REP_SM (opt-in arm, GBDT (histogram replica count force values); EXP:1640 PROMOTED, rule e) is gone; the default path is the only one; code recoverable at cf752f933b; restore: experiments/removed/MOJOLEARN_TREES_HIST_REP_SM.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T17_BATCH.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T17_BATCH"](), "removed 2026-11-15 (lane/expdel2-45-trees): TREES_T17_BATCH (opt-in arm, GBDT (lossguide exact batch width force values); EXP:1641 PROMOTED, rule e) is gone; the default path is the only one; code recoverable at fe9794876a; restore: experiments/removed/MOJOLEARN_TREES_T17_BATCH.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T18.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T18"](), "removed 2026-11-15 (lane/expdel2-45-trees): TREES_T18 (opt-in arm, GBDT (T18 opt-in); no record, rule a) is gone; the default path is the only one; code recoverable at 967d4d7eef; restore: experiments/removed/MOJOLEARN_TREES_T18.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T21.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T21"](), "removed 2026-11-15 (lane/expdel2-45-trees): TREES_T21 (opt-in arm, GBDT (T21 replica streams opt-in); no record, rule a) is gone; the default path is the only one; code recoverable at 80865e2e25; restore: experiments/removed/MOJOLEARN_TREES_T21.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T21_STREAMS.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T21_STREAMS"](), "removed 2026-11-15 (lane/expdel2-45-trees): TREES_T21_STREAMS (opt-in arm, GBDT (T21 stream count sweep); no record, rule a) is gone; the default path is the only one; code recoverable at 293d3da2dd; restore: experiments/removed/MOJOLEARN_TREES_T21_STREAMS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T22_OFF.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T22_OFF"](), "removed 2026-11-15 (lane/expdel2-45-trees): TREES_T22_OFF (old path, GBDT (ordered fold-position storage rollback); EXP:1636 PROMOTED, rule e) is gone; the default path is the only one; code recoverable at 0fcc31241b; restore: experiments/removed/MOJOLEARN_TREES_T22_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T25.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T25"](), "removed 2026-11-15 (lane/expdel2-45-trees): TREES_T25 (opt-in arm, GBDT (T25 opt-in); no record, rule a) is gone; the default path is the only one; code recoverable at 5de73a4228; restore: experiments/removed/MOJOLEARN_TREES_T25.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T28.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T28"](), "removed 2026-11-15 (lane/expdel2-45-trees): TREES_T28 (opt-in arm, GBDT (T28 CTR prep sharing opt-in); no record, rule a) is gone; the default path is the only one; code recoverable at 5c0baaca38; restore: experiments/removed/MOJOLEARN_TREES_T28.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T29.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T29"](), "removed 2026-11-15 (lane/expdel2-45-trees): TREES_T29 (opt-in arm, GBDT (T29 generated PairLogit objective opt-in); EXP:1869 KEPT (code stays), rule e) is gone; the default path is the only one; code recoverable at a6ac1678d7; restore: experiments/removed/MOJOLEARN_TREES_T29.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T30.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T30"](), "removed 2026-11-15 (lane/expdel2-45-trees): TREES_T30 (opt-in arm, GBDT (T30 opt-in); no record, rule a) is gone; the default path is the only one; code recoverable at b0f082dd3e; restore: experiments/removed/MOJOLEARN_TREES_T30.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T19_DEFER.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T19_DEFER"](), "removed 2026-11-15 (lane/expdel2-45-trees): TREES_T19_DEFER (opt-in arm, GBDT (T19 deferred histogram copy); EXP:1627 DROP (quality loss), code deleted, rule e) is gone; the default path is the only one; code recoverable at 2f68cbff39; restore: experiments/removed/MOJOLEARN_TREES_T19_DEFER.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T28_SORT.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T28_SORT"](), "removed 2026-11-15 (lane/expdel2-45-trees): TREES_T28_SORT (opt-in arm, GBDT (T28 CTR sort-once sub-arm); no record, rule a) is gone; the default path is the only one; code recoverable at 7745e944ef; restore: experiments/removed/MOJOLEARN_TREES_T28_SORT.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T28_PREFIX.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T28_PREFIX"](), "removed 2026-11-15 (lane/expdel2-45-trees): TREES_T28_PREFIX (opt-in arm, GBDT (T28 CTR prefix reuse sub-arm); no record, rule a) is gone; the default path is the only one; code recoverable at 7ffe9b4185; restore: experiments/removed/MOJOLEARN_TREES_T28_PREFIX.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T23.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T23"](), "removed 2026-11-15 (lane/expdel2-45-trees): TREES_T23 (opt-in arm, GBDT (T23 opt-in); no record, rule a) is gone; the default path is the only one; code recoverable at 35309dcb55; restore: experiments/removed/MOJOLEARN_TREES_T23.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T24.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T24"](), "removed 2026-11-15 (lane/expdel2-45-trees): TREES_T24 (opt-in arm, GBDT (T24 opt-in); no record, rule a) is gone; the default path is the only one; code recoverable at fca63c0b28; restore: experiments/removed/MOJOLEARN_TREES_T24.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T26.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T26"](), "removed 2026-11-15 (lane/expdel2-45-trees): TREES_T26 (opt-in arm, GBDT (T26 opt-in); no record, rule a) is gone; the default path is the only one; code recoverable at 0242b4a432; restore: experiments/removed/MOJOLEARN_TREES_T26.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T30_ADABOOST.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T30_ADABOOST"](), "removed 2026-11-15 (lane/expdel2-45-trees): TREES_T30_ADABOOST (opt-in arm, Trees (T30 AdaBoost sub-arm); no record, rule a) is gone; the default path is the only one; code recoverable at 0469c65bd2; restore: experiments/removed/MOJOLEARN_TREES_T30_ADABOOST.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_C45_GBDT.patch
    comptime assert not is_defined["MOJOLEARN_TREES_C45_GBDT"](), "removed 2026-11-15 (lane/expdel2-45-trees): TREES_C45_GBDT (opt-in arm, GBDT (C45 opt-in); no record, rule a) is gone; the default path is the only one; code recoverable at 92620cf9db; restore: experiments/removed/MOJOLEARN_TREES_C45_GBDT.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_ORD_STD_GRIDFOLD_OFF.patch
    comptime assert not is_defined["MOJOLEARN_TREES_ORD_STD_GRIDFOLD_OFF"](), "removed 2026-11-15 (lane/expdel2-45-trees): TREES_ORD_STD_GRIDFOLD_OFF (old path, GBDT (ordered 256-lane-chain fold rollback); EXP:1637 PROMOTED, rule e) is gone; the default path is the only one; code recoverable at 56ec6a17df; restore: experiments/removed/MOJOLEARN_TREES_ORD_STD_GRIDFOLD_OFF.patch"
    # MOJOLEARN_TREES_LG_LEVEL_ROUNDS: refusal removed 2026-11-15 (restored by lane/restore-run-arms-trees; RUN arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_CTR_SORTFREE_SUMS.patch
    comptime assert not is_defined["MOJOLEARN_TREES_CTR_SORTFREE_SUMS"](), "removed 2026-11-15 (lane/expdel2-45-trees): TREES_CTR_SORTFREE_SUMS (opt-in arm, GBDT (sort-free CTR sums); no record, rule a) is gone; the default path is the only one; code recoverable at a3d500c032; restore: experiments/removed/MOJOLEARN_TREES_CTR_SORTFREE_SUMS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ORD_SKIP_UNSEARCHED_PERM_OFF.patch
    comptime assert not is_defined["MOJOLEARN_ORD_SKIP_UNSEARCHED_PERM_OFF"](), "removed 2026-11-15 (lane/expdel2-45-trees): ORD_SKIP_UNSEARCHED_PERM_OFF (old path, GBDT (ordered unsearched permutation rollback); grid fg-gbdt-ordered.json pending, rule b) is gone; the default path is the only one; code recoverable at b54053d9c0; restore: experiments/removed/MOJOLEARN_ORD_SKIP_UNSEARCHED_PERM_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ORD_CAT_PLANES_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ORD_CAT_PLANES_OFF"](), "removed 2026-11-15 (lane/expdel2-45-trees): IDN_ORD_CAT_PLANES_OFF (old path, GBDT (ordered per-fold cursor planes rollback); grid fg-gbdt-ordered.json pending, rule b) is gone; the default path is the only one; code recoverable at c435213440; restore: experiments/removed/MOJOLEARN_IDN_ORD_CAT_PLANES_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ORD_HIST_FOLD_SKIP.patch
    comptime assert not is_defined["MOJOLEARN_ORD_HIST_FOLD_SKIP"](), "removed 2026-11-15 (lane/expdel2-45-trees): ORD_HIST_FOLD_SKIP (opt-in arm, GBDT (ordered small-fold histogram kernel); grid fg-gbdt-ordered.json pending, rule a) is gone; the default path is the only one; code recoverable at 48ef0658c9; restore: experiments/removed/MOJOLEARN_ORD_HIST_FOLD_SKIP.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ORD_HIST_FOLD_SKIP_DOCS.patch
    comptime assert not is_defined["MOJOLEARN_ORD_HIST_FOLD_SKIP_DOCS"](), "removed 2026-11-15 (lane/expdel2-45-trees): ORD_HIST_FOLD_SKIP_DOCS (opt-in arm, GBDT (ordered small-fold document budget sweep); grid fg-gbdt-ordered.json pending, rule a) is gone; the default path is the only one; code recoverable at 5a5f297498; restore: experiments/removed/MOJOLEARN_ORD_HIST_FOLD_SKIP_DOCS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ORD_INDEX_SHARED.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ORD_INDEX_SHARED"](), "removed 2026-11-15 (lane/expdel2-45-trees): IDN_ORD_INDEX_SHARED (opt-in arm, GBDT (ordered shared permutation index); grid fg-gbdt-ordered.json pending, rule a) is gone; the default path is the only one; code recoverable at d452e2066a; restore: experiments/removed/MOJOLEARN_IDN_ORD_INDEX_SHARED.patch"
    # ---- end lane expdel2-45-trees
    # Lane expdel2-46-trees (2026-11-15): unmeasured arms, never-run cards and old paths deleted under "An unmeasured arm is deleted" (plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history), docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T40_ROWMAJOR.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T40_ROWMAJOR"](), "removed 2026-11-15 (lane/expdel2-46-trees): TREES_T40_ROWMAJOR (opt-in arm, Trees (IsolationForest IDENTICAL row-major upload); no A/B recorded for the opt-in arm (rule a)) is gone; the default path is the only one; code recoverable at 931e7ca76a; restore: experiments/removed/MOJOLEARN_TREES_T40_ROWMAJOR.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T41_TREE_BATCH.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T41_TREE_BATCH"](), "removed 2026-11-15 (lane/expdel2-46-trees): TREES_T41_TREE_BATCH (opt-in arm, Trees (IsolationForest tree batching); no A/B recorded for the opt-in arm (rule a)) is gone; the default path is the only one; code recoverable at aa61101d80; restore: experiments/removed/MOJOLEARN_TREES_T41_TREE_BATCH.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T42_CORRECTION_CACHE.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T42_CORRECTION_CACHE"](), "removed 2026-11-15 (lane/expdel2-46-trees): TREES_T42_CORRECTION_CACHE (opt-in arm, Trees (IsolationForest c(n) correction table); no A/B recorded for the opt-in arm (rule a)) is gone; the default path is the only one; code recoverable at d30e6f5678; restore: experiments/removed/MOJOLEARN_TREES_T42_CORRECTION_CACHE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_C47_IF_LIVE_TASKS.patch
    comptime assert not is_defined["MOJOLEARN_TREES_C47_IF_LIVE_TASKS"](), "removed 2026-11-15 (lane/expdel2-46-trees): TREES_C47_IF_LIVE_TASKS (opt-in arm, Trees (IsolationForest live min/max reduction lanes); no A/B recorded for the opt-in arm (rule a)) is gone; the default path is the only one; code recoverable at 77a0ad6f45; restore: experiments/removed/MOJOLEARN_TREES_C47_IF_LIVE_TASKS.patch"
    # MOJOLEARN_TREES_C48_IF_PARTITION: refusal removed 2026-11-15 (restored by lane/restore-run-arms-trees; RUN arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_IF_ROWMAJOR_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IF_ROWMAJOR_OFF"](), "removed 2026-11-15 (lane/expdel2-46-trees): IF_ROWMAJOR_OFF (old path, Trees (IsolationForest FAST+Apple row-major upload rollback); EXP:115 KEPT 269ffa57a, the FAST+Apple default (rule d: FAST is Apple only, so no NVIDIA/AMD arm is owed)) is gone; the default path is the only one; code recoverable at b35af0836a; restore: experiments/removed/MOJOLEARN_IF_ROWMAJOR_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_IF_BUILD_TPB.patch
    comptime assert not is_defined["MOJOLEARN_TREES_IF_BUILD_TPB"](), "removed 2026-11-15 (lane/expdel2-46-trees): TREES_IF_BUILD_TPB (opt-in arm, Trees (IsolationForest build block width sweep); no A/B recorded for the opt-in int sweep (rule a)) is gone; the default path is the only one; code recoverable at 2e3951cb68; restore: experiments/removed/MOJOLEARN_TREES_IF_BUILD_TPB.patch"
    # MOJOLEARN_AFT_F10: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFCL_T10: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_F12: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_F11: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_IF_DIAG_ENTRY_RETURN.patch
    comptime assert not is_defined["MOJOLEARN_IF_DIAG_ENTRY_RETURN"](), "removed 2026-11-15 (lane/expdel2-46-trees): IF_DIAG_ENTRY_RETURN (diagnostic arm, Trees (IsolationForest hang-bisect truncation, entry return); no A/B recorded (rule a); a 2026-08 RTX 4090 hang bisect guard that no script builds today) is gone; the default path is the only one; code recoverable at 4a6591cfdc; restore: experiments/removed/MOJOLEARN_IF_DIAG_ENTRY_RETURN.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IF_DIAG_GATHER_ONLY.patch
    comptime assert not is_defined["MOJOLEARN_IF_DIAG_GATHER_ONLY"](), "removed 2026-11-15 (lane/expdel2-46-trees): IF_DIAG_GATHER_ONLY (diagnostic arm, Trees (IsolationForest hang-bisect truncation, gather only); no A/B recorded (rule a); a 2026-08 RTX 4090 hang bisect guard that no script builds today) is gone; the default path is the only one; code recoverable at 690463fbcf; restore: experiments/removed/MOJOLEARN_IF_DIAG_GATHER_ONLY.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IF_DIAG_NO_REJECT.patch
    comptime assert not is_defined["MOJOLEARN_IF_DIAG_NO_REJECT"](), "removed 2026-11-15 (lane/expdel2-46-trees): IF_DIAG_NO_REJECT (diagnostic arm, Trees (IsolationForest hang-bisect truncation, no rejection loop); no A/B recorded (rule a); a 2026-08 RTX 4090 hang bisect guard that no script builds today) is gone; the default path is the only one; code recoverable at e1bb1a873e; restore: experiments/removed/MOJOLEARN_IF_DIAG_NO_REJECT.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IF_DIAG_NO_RECORD.patch
    comptime assert not is_defined["MOJOLEARN_IF_DIAG_NO_RECORD"](), "removed 2026-11-15 (lane/expdel2-46-trees): IF_DIAG_NO_RECORD (diagnostic arm, Trees (IsolationForest hang-bisect truncation, no decision record); no A/B recorded (rule a); a 2026-08 RTX 4090 hang bisect guard that no script builds today) is gone; the default path is the only one; code recoverable at 5c5dda5130; restore: experiments/removed/MOJOLEARN_IF_DIAG_NO_RECORD.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREESHAP_FAST_TABLE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_TREESHAP_FAST_TABLE_OFF"](), "removed 2026-11-15 (lane/expdel2-46-trees): TREESHAP_FAST_TABLE_OFF (old path, Trees (TreeSHAP leaf table rollback); EXP:186 KEPT/DEFAULT (rule e)) is gone; the default path is the only one; code recoverable at 7d6b2a07e7; restore: experiments/removed/MOJOLEARN_TREESHAP_FAST_TABLE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SHAP_TREE_TAB_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SHAP_TREE_TAB_OFF"](), "removed 2026-11-15 (lane/expdel2-46-trees): SHAP_TREE_TAB_OFF (old path, Trees (TreeSHAP decision-word row unit rollback); EXP:1271 ACCEPT/DEFAULT on M3 (FAST+Apple, istella 24.75 -> 19.57 ms); promoted on one vendor, the NVIDIA/AMD IDENTICAL A/B is owed (rule d)) is gone; the default path is the only one; code recoverable at 24292c45c2; restore: experiments/removed/MOJOLEARN_SHAP_TREE_TAB_OFF.patch"
    # MOJOLEARN_AFCL_T11: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_SHAP_FAST_ROW_PAIR: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_TREESHAP_ACC_SHARED.patch
    comptime assert not is_defined["MOJOLEARN_TREESHAP_ACC_SHARED"](), "removed 2026-11-15 (lane/expdel2-46-trees): TREESHAP_ACC_SHARED (opt-in arm, Trees (TreeSHAP shared-tile accumulator); no A/B recorded for the opt-in arm (rule a)) is gone; the default path is the only one; code recoverable at 70f8ff00b2; restore: experiments/removed/MOJOLEARN_TREESHAP_ACC_SHARED.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T45_LINEAR_UNWIND.patch
    comptime assert not is_defined["MOJOLEARN_TREES_T45_LINEAR_UNWIND"](), "removed 2026-11-15 (lane/expdel2-46-trees): TREES_T45_LINEAR_UNWIND (opt-in arm, Trees (TreeSHAP linear path unwind); no A/B recorded for the opt-in arm (rule a)) is gone; the default path is the only one; code recoverable at f2de2ddf8b; restore: experiments/removed/MOJOLEARN_TREES_T45_LINEAR_UNWIND.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_PLATT_ONE_READ_OFF.patch
    comptime assert not is_defined["MOJOLEARN_TREES_PLATT_ONE_READ_OFF"](), "removed 2026-11-15 (lane/expdel2-46-trees): TREES_PLATT_ONE_READ_OFF (old path, Trees (calibrated Platt one-read rollback); EXP:229 PENDING: no A/B recorded for the old path (rule b)) is gone; the default path is the only one; code recoverable at 397bc56996; restore: experiments/removed/MOJOLEARN_TREES_PLATT_ONE_READ_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AGN_IDN_SYN_POOL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_AGN_IDN_SYN_POOL_OFF"](), "removed 2026-11-15 (lane/expdel2-46-trees): AGN_IDN_SYN_POOL_OFF (old path, Trees (Kernel/Permutation SHAP pooled synthetic buffer rollback); no A/B recorded for the old path (rule b)) is gone; the default path is the only one; code recoverable at f0bdc4bfd9; restore: experiments/removed/MOJOLEARN_AGN_IDN_SYN_POOL_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SHAP_DEVICE_MODEL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SHAP_DEVICE_MODEL_OFF"](), "removed 2026-11-15 (lane/expdel2-46-trees): IDN_SHAP_DEVICE_MODEL_OFF (old path, Trees (Kernel/Permutation SHAP device model rollback); no A/B recorded for the old path (rule b)) is gone; the default path is the only one; code recoverable at c41fff16e3; restore: experiments/removed/MOJOLEARN_IDN_SHAP_DEVICE_MODEL_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_PSHAP_RESIDENT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_PSHAP_RESIDENT_OFF"](), "removed 2026-11-15 (lane/expdel2-46-trees): IDN_PSHAP_RESIDENT_OFF (old path, Trees (Permutation SHAP resident linear path rollback); EXP:2051 KEEP, promoted 2026-10-11 (NV 0.097, AMD 0.128) (rule e)) is gone; the default path is the only one; code recoverable at 5002a250c9; restore: experiments/removed/MOJOLEARN_IDN_PSHAP_RESIDENT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_PSHAP_DELTA_OFF.patch
    comptime assert not is_defined["MOJOLEARN_PSHAP_DELTA_OFF"](), "removed 2026-11-15 (lane/expdel2-46-trees): PSHAP_DELTA_OFF (old path, Trees (Permutation SHAP delta synthesis rollback); EXP:1595 OPEN: no A/B recorded for the old path (rule b); the default is the M3 -47.6% FAST+Apple one) is gone; the default path is the only one; code recoverable at 7f82ef0243; restore: experiments/removed/MOJOLEARN_PSHAP_DELTA_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SHAP_PERM_CACHE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SHAP_PERM_CACHE_OFF"](), "removed 2026-11-15 (lane/expdel2-46-trees): SHAP_PERM_CACHE_OFF (old path, Trees (Permutation SHAP per-chunk cache rollback); EXP:1403 READY-AB: no A/B recorded for the old path (rule b)) is gone; the default path is the only one; code recoverable at 7669424c25; restore: experiments/removed/MOJOLEARN_SHAP_PERM_CACHE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_DART_DEVICE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_DART_DEVICE_OFF"](), "removed 2026-11-15 (lane/expdel2-46-trees): DART_DEVICE_OFF (old path, Trees (DART FAST device round rollback); EXP:104 KEPT 741a5495d on M3 (dart -46%); promoted on one vendor, the NVIDIA/AMD FAST A/B is owed (rule d)) is gone; the default path is the only one; code recoverable at 9a82517370; restore: experiments/removed/MOJOLEARN_DART_DEVICE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DART_DEVICE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_DART_DEVICE_OFF"](), "removed 2026-11-15 (lane/expdel2-46-trees): IDN_DART_DEVICE_OFF (old path, Trees (DART IDENTICAL device round rollback); no A/B recorded for the old path (rule b)) is gone; the default path is the only one; code recoverable at 4a48287b38; restore: experiments/removed/MOJOLEARN_IDN_DART_DEVICE_OFF.patch"
    # ---- end lane expdel2-46-trees
    # Lane expdel2-43-trees (2026-11-15): unmeasured arms, never-run cards and old paths deleted under "An unmeasured arm is deleted" (plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history), docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_GBDT_DW2_SCAN_SMEM_OFF.patch
    comptime assert not is_defined["MOJOLEARN_GBDT_DW2_SCAN_SMEM_OFF"](), "removed 2026-11-15 (lane/expdel2-43-trees): GBDT_DW2_SCAN_SMEM_OFF (old path, Trees (GBDT depthwise, FAST Apple shared-memory histogram scan); EXP:110 KEPT c55b8c377) is gone; the default path is the only one; code recoverable at 1effc69adf; restore: experiments/removed/MOJOLEARN_GBDT_DW2_SCAN_SMEM_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_GBDT_DW_FUSED_CHAIN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_GBDT_DW_FUSED_CHAIN_OFF"](), "removed 2026-11-15 (lane/expdel2-43-trees): GBDT_DW_FUSED_CHAIN_OFF (old path, Trees (GBDT depthwise, FAST Apple fused split chain); EXP:137 DROPPED-noise for the stacked sync arms) is gone; the default path is the only one; code recoverable at e717a7e088; restore: experiments/removed/MOJOLEARN_GBDT_DW_FUSED_CHAIN_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_GBDT_DW_MODE_SKIP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_GBDT_DW_MODE_SKIP_OFF"](), "removed 2026-11-15 (lane/expdel2-43-trees): GBDT_DW_MODE_SKIP_OFF (old path, Trees (GBDT quantized histogram, FAST Apple mode-bin skip); EXP:112 KEPT 7736fa5b8) is gone; the default path is the only one; code recoverable at 2c8d6bbc8e; restore: experiments/removed/MOJOLEARN_GBDT_DW_MODE_SKIP_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_GBDT_IDENTICAL_SPLIT_COPY.patch
    comptime assert not is_defined["MOJOLEARN_GBDT_IDENTICAL_SPLIT_COPY"](), "removed 2026-11-15 (lane/expdel2-43-trees): GBDT_IDENTICAL_SPLIT_COPY (opt-in arm, Trees (GBDT depthwise/lossguide, Apple IDENTICAL split-time histogram copy); no A/B recorded for the opt-in arm) is gone; the default path is the only one; code recoverable at 7be6f0b80e; restore: experiments/removed/MOJOLEARN_GBDT_IDENTICAL_SPLIT_COPY.patch"
    # MOJOLEARN_GBDT_ID_DEFER_COPY: refusal removed 2026-11-15 (restored by lane/restore-run-arms-trees; RUN arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_GBDT_LG_BATCH: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_GBDT_LG_BATCH2: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_GBDT_LG_BATCH4: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_GBDT_LG_BATCH8: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_GBDT_LG_BATCH32: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_GBDT_LG_BATCH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_GBDT_LG_BATCH_OFF"](), "removed 2026-11-15 (lane/expdel2-43-trees): GBDT_LG_BATCH_OFF (old path, Trees (GBDT lossguide, FAST Apple B-best-leaves batch); no A/B recorded for the old path) is gone; the default path is the only one; code recoverable at ba59d96b6a; restore: experiments/removed/MOJOLEARN_GBDT_LG_BATCH_OFF.patch"
    # MOJOLEARN_GBDT_LG_EXACT_BATCH128: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_GBDT_LG_EXACT_BATCH128_OFF.patch
    comptime assert not is_defined["MOJOLEARN_GBDT_LG_EXACT_BATCH128_OFF"](), "removed 2026-11-15 (lane/expdel2-43-trees): GBDT_LG_EXACT_BATCH128_OFF (old path, Trees (GBDT lossguide, FAST Apple exact-batch width 128); EXP:113 KEPT 9722e5a2b) is gone; the default path is the only one; code recoverable at e50ac0e432; restore: experiments/removed/MOJOLEARN_GBDT_LG_EXACT_BATCH128_OFF.patch"
    # MOJOLEARN_GBDT_LG_EXACT_BATCH32: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_GBDT_LG_EXACT_BATCH64.patch
    comptime assert not is_defined["MOJOLEARN_GBDT_LG_EXACT_BATCH64"](), "removed 2026-11-15 (lane/expdel2-43-trees): GBDT_LG_EXACT_BATCH64 (opt-in arm, Trees (GBDT lossguide, IDENTICAL exact-batch width 64); EXP:114 KEPT is the FAST width) is gone; the default path is the only one; code recoverable at d61b232e98; restore: experiments/removed/MOJOLEARN_GBDT_LG_EXACT_BATCH64.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_GBDT_LG_EXACT_BATCH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_GBDT_LG_EXACT_BATCH_OFF"](), "removed 2026-11-15 (lane/expdel2-43-trees): GBDT_LG_EXACT_BATCH_OFF (old path, Trees (GBDT lossguide, FAST Apple exact best-first batch); no A/B recorded for the old path) is gone; the default path is the only one; code recoverable at c556cd4e1d; restore: experiments/removed/MOJOLEARN_GBDT_LG_EXACT_BATCH_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_GBDT_LG_EXACT_ID_OFF.patch
    comptime assert not is_defined["MOJOLEARN_GBDT_LG_EXACT_ID_OFF"](), "removed 2026-11-15 (lane/expdel2-43-trees): GBDT_LG_EXACT_ID_OFF (old path, Trees (GBDT lossguide, IDENTICAL exact best-first batch); no A/B recorded for the old path) is gone; the default path is the only one; code recoverable at ad8530ee49; restore: experiments/removed/MOJOLEARN_GBDT_LG_EXACT_ID_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_GBDT_NS_INHERIT_ID_OFF.patch
    comptime assert not is_defined["MOJOLEARN_GBDT_NS_INHERIT_ID_OFF"](), "removed 2026-11-15 (lane/expdel2-43-trees): GBDT_NS_INHERIT_ID_OFF (old path, Trees (GBDT non-symmetric, IDENTICAL inherited partition); no A/B recorded for the old path) is gone; the default path is the only one; code recoverable at cb4478ec40; restore: experiments/removed/MOJOLEARN_GBDT_NS_INHERIT_ID_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_GBDT_NS_INHERIT_PARTITION_OFF.patch
    comptime assert not is_defined["MOJOLEARN_GBDT_NS_INHERIT_PARTITION_OFF"](), "removed 2026-11-15 (lane/expdel2-43-trees): GBDT_NS_INHERIT_PARTITION_OFF (old path, Trees (GBDT non-symmetric, FAST Apple inherited partition); no A/B recorded for the old path) is gone; the default path is the only one; code recoverable at 3b8ce77de5; restore: experiments/removed/MOJOLEARN_GBDT_NS_INHERIT_PARTITION_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GBDT_NS_SCALE_DEVICE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GBDT_NS_SCALE_DEVICE_OFF"](), "removed 2026-11-15 (lane/expdel2-43-trees): IDN_GBDT_NS_SCALE_DEVICE_OFF (old path, Trees (GBDT non-symmetric, IDENTICAL device fixed-point scale); no A/B recorded for the old path) is gone; the default path is the only one; code recoverable at 608bbc1936; restore: experiments/removed/MOJOLEARN_IDN_GBDT_NS_SCALE_DEVICE_OFF.patch"
    # MOJOLEARN_SYM_RESOLVE_BLOCK: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # ---- end lane expdel2-43-trees
    # Lane expdel2-42-trees (2026-11-15): unmeasured arms, never-run cards and old paths deleted under "An unmeasured arm is deleted" (plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history), docs/TOMBSTONES.md).
    # MOJOLEARN_2031_SYM_RIDX_SPLITS: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_2031_SYM_RIDX_SPLITS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_2031_SYM_RIDX_SPLITS_OFF"](), "removed 2026-11-15 (lane/expdel2-42-trees): 2031_SYM_RIDX_SPLITS_OFF (old path, Trees (GBDT symmetric permuting reorder on Apple); no A/B recorded for the old path (rule b)) is gone; the default path is the only one; code recoverable at 931e7ca76a; restore: experiments/removed/MOJOLEARN_2031_SYM_RIDX_SPLITS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_2580_LEVEL_QUANT.patch
    comptime assert not is_defined["MOJOLEARN_2580_LEVEL_QUANT"](), "removed 2026-11-15 (lane/expdel2-42-trees): 2580_LEVEL_QUANT (opt-in arm, Trees (GBDT one-byte stats quantized once per level); no A/B recorded for the opt-in arm (rule a)) is gone; the default path is the only one; code recoverable at 7a9e6c5691; restore: experiments/removed/MOJOLEARN_2580_LEVEL_QUANT.patch"
    # MOJOLEARN_2581_GROUP_WIDTH: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_2581_GROUP_WIDTH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_2581_GROUP_WIDTH_OFF"](), "removed 2026-11-15 (lane/expdel2-42-trees): 2581_GROUP_WIDTH_OFF (old path, Trees (GBDT block-widest one-byte ladder under IDENTICAL); no A/B recorded for the old path (rule b)) is gone; the default path is the only one; code recoverable at a0d1816b17; restore: experiments/removed/MOJOLEARN_2581_GROUP_WIDTH_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_GBDT_FUSED_LEVEL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_GBDT_FUSED_LEVEL_OFF"](), "removed 2026-11-15 (lane/expdel2-42-trees): GBDT_FUSED_LEVEL_OFF (old path, Trees (GBDT unfused symmetric level schedule); no A/B recorded for the old path (rule b)) is gone; the default path is the only one; code recoverable at 413f6bdc16; restore: experiments/removed/MOJOLEARN_GBDT_FUSED_LEVEL_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_LEGACY_NARROW_SYM_RIDX.patch
    comptime assert not is_defined["MOJOLEARN_LEGACY_NARROW_SYM_RIDX"](), "removed 2026-11-15 (lane/expdel2-42-trees): LEGACY_NARROW_SYM_RIDX (old path, Trees (GBDT FAST 64-feature cap on the ridx schedule); EXP:1331 OWED) is gone; the default path is the only one; code recoverable at f0b3e68972; restore: experiments/removed/MOJOLEARN_LEGACY_NARROW_SYM_RIDX.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_GBDT_RIDX_GATHER_COST.patch
    comptime assert not is_defined["MOJOLEARN_GBDT_RIDX_GATHER_COST"](), "removed 2026-11-15 (lane/expdel2-42-trees): GBDT_RIDX_GATHER_COST (opt-in arm, Trees (GBDT ridx gather cost sweep knob); no A/B recorded for the sweep (rule a)) is gone; the default path is the only one; code recoverable at 5ee0e727f3; restore: experiments/removed/MOJOLEARN_GBDT_RIDX_GATHER_COST.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GBDT_PAIRLOGIT_GROUP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GBDT_PAIRLOGIT_GROUP_OFF"](), "removed 2026-11-15 (lane/expdel2-42-trees): IDN_GBDT_PAIRLOGIT_GROUP_OFF (old path, Trees (GBDT PairLogit host pair list under IDENTICAL); no A/B recorded for the old path (rule b)) is gone; the default path is the only one; code recoverable at 88680d2aa2; restore: experiments/removed/MOJOLEARN_IDN_GBDT_PAIRLOGIT_GROUP_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GBDT_CTR_PERM_BATCH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GBDT_CTR_PERM_BATCH_OFF"](), "removed 2026-11-15 (lane/expdel2-42-trees): IDN_GBDT_CTR_PERM_BATCH_OFF (old path, Trees (GBDT serial CTR permutation loop under IDENTICAL); no A/B recorded for the old path (rule b)) is gone; the default path is the only one; code recoverable at 4fc6907fab; restore: experiments/removed/MOJOLEARN_IDN_GBDT_CTR_PERM_BATCH_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_GBDT_CTR_PERM_BATCH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_GBDT_CTR_PERM_BATCH_OFF"](), "removed 2026-11-15 (lane/expdel2-42-trees): GBDT_CTR_PERM_BATCH_OFF (old path, Trees (GBDT serial CTR permutation loop, FAST Apple rollback); EXP:108 KEPT 57ddea946 (M3 taxicat 36,772 -> 35,780 ms)) is gone; the default path is the only one; code recoverable at 9adf63d224; restore: experiments/removed/MOJOLEARN_GBDT_CTR_PERM_BATCH_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_GBDT_BT_HW_F64_OFF.patch
    comptime assert not is_defined["MOJOLEARN_GBDT_BT_HW_F64_OFF"](), "removed 2026-11-15 (lane/expdel2-42-trees): GBDT_BT_HW_F64_OFF (old path, Trees (GBDT border-type DP on soft binary64 everywhere); EXP:2272 CANDIDATE) is gone; the default path is the only one; code recoverable at 533f4a23f1; restore: experiments/removed/MOJOLEARN_GBDT_BT_HW_F64_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_GBDT_BT_COLUMN_BLOCKS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_GBDT_BT_COLUMN_BLOCKS_OFF"](), "removed 2026-11-15 (lane/expdel2-42-trees): GBDT_BT_COLUMN_BLOCKS_OFF (old path, Trees (GBDT packed 64-column border-type launch); EXP:2273 CANDIDATE) is gone; the default path is the only one; code recoverable at 4d1a3fc5f3; restore: experiments/removed/MOJOLEARN_GBDT_BT_COLUMN_BLOCKS_OFF.patch"
    # MOJOLEARN_2041_FAST_CHUNKS_PIN: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_2635_LINEAR_BOUNDS.patch
    comptime assert not is_defined["MOJOLEARN_2635_LINEAR_BOUNDS"](), "removed 2026-11-15 (lane/expdel2-42-trees): 2635_LINEAR_BOUNDS (opt-in arm, Trees (GBDT linear LowerBound/UpperBound walks in the border search); no A/B recorded for the opt-in arm (rule a)) is gone; the default path is the only one; code recoverable at 824cf67b84; restore: experiments/removed/MOJOLEARN_2635_LINEAR_BOUNDS.patch"
    # MOJOLEARN_AFT_G01: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_G02: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_G03: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_G04: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_G05: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_G06: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_G07: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_G08: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_G09: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_G10: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_G11: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFT_G12: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_GBDT_PREDICT_PACKED: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_GBDT_QUANT_DEVICE: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_GBDT_INDEX_PACK_DEVICE: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # ---- end lane expdel2-42-trees
    # Lane expdel2-44-trees (2026-11-15): unmeasured arms, force knobs and old paths deleted under "An unmeasured arm is deleted" (plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history), docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_2030_FUSED_EST_MOVE.patch
    comptime assert not is_defined["MOJOLEARN_2030_FUSED_EST_MOVE"](), "removed 2026-11-15 (lane/expdel2-44-trees): 2030_FUSED_EST_MOVE (opt-in arm, Trees (GBDT fused estimation move force knob, DEVIATION 2030)) is gone; the default path is the only one; code recoverable at 5b86e54145; restore: experiments/removed/MOJOLEARN_2030_FUSED_EST_MOVE.patch"
    # MOJOLEARN_2030_NO_FUSED_EST_MOVE: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_2551_DEVICE_PARTITION: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_2551_DEVICE_PARTITION_OFF.patch
    comptime assert not is_defined["MOJOLEARN_2551_DEVICE_PARTITION_OFF"](), "removed 2026-11-15 (lane/expdel2-44-trees): 2551_DEVICE_PARTITION_OFF (old path, Trees (GBDT host leaf partition rollback, DEVIATION 2551)) is gone; the default path is the only one; code recoverable at 269778f8c9; restore: experiments/removed/MOJOLEARN_2551_DEVICE_PARTITION_OFF.patch"
    # MOJOLEARN_EST_ITERS_DEVICE: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_GBDT_NS_PREDICT_BITVEC.patch
    comptime assert not is_defined["MOJOLEARN_GBDT_NS_PREDICT_BITVEC"](), "removed 2026-11-15 (lane/expdel2-44-trees): GBDT_NS_PREDICT_BITVEC (opt-in arm, Trees (GBDT Depthwise/Lossguide bitvector apply)) is gone; the default path is the only one; code recoverable at 4cdd684efa; restore: experiments/removed/MOJOLEARN_GBDT_NS_PREDICT_BITVEC.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_GBDT_ORD_TREE_BUFS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_GBDT_ORD_TREE_BUFS_OFF"](), "removed 2026-11-15 (lane/expdel2-44-trees): GBDT_ORD_TREE_BUFS_OFF (old path, Trees (GBDT Ordered per-tree buffer allocation rollback)) is gone; the default path is the only one; code recoverable at 47147e100f; restore: experiments/removed/MOJOLEARN_GBDT_ORD_TREE_BUFS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_GBDT_WALK_DEV_STEP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_GBDT_WALK_DEV_STEP_OFF"](), "removed 2026-11-15 (lane/expdel2-44-trees): GBDT_WALK_DEV_STEP_OFF (old path, Trees (GBDT device walk step rollback)) is gone; the default path is the only one; code recoverable at a303565f2a; restore: experiments/removed/MOJOLEARN_GBDT_WALK_DEV_STEP_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GBDT_APPLY_WIDE.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GBDT_APPLY_WIDE"](), "removed 2026-11-15 (lane/expdel2-44-trees): IDN_GBDT_APPLY_WIDE (opt-in arm, Trees (GBDT IDENTICAL wide uniform-depth oblivious apply)) is gone; the default path is the only one; code recoverable at 805d22d33d; restore: experiments/removed/MOJOLEARN_IDN_GBDT_APPLY_WIDE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GBDT_EST_ONE_STEP_DEVICE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GBDT_EST_ONE_STEP_DEVICE_OFF"](), "removed 2026-11-15 (lane/expdel2-44-trees): IDN_GBDT_EST_ONE_STEP_DEVICE_OFF (old path, Trees (GBDT IDENTICAL one-step device estimation rollback)) is gone; the default path is the only one; code recoverable at 0ed18e01e0; restore: experiments/removed/MOJOLEARN_IDN_GBDT_EST_ONE_STEP_DEVICE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GBDT_MC_ONE_STEP_DEVICE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GBDT_MC_ONE_STEP_DEVICE_OFF"](), "removed 2026-11-15 (lane/expdel2-44-trees): IDN_GBDT_MC_ONE_STEP_DEVICE_OFF (old path, Trees (GBDT IDENTICAL MultiClass one-step device solve rollback)) is gone; the default path is the only one; code recoverable at 764c283909; restore: experiments/removed/MOJOLEARN_IDN_GBDT_MC_ONE_STEP_DEVICE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GBDT_NS_PREDICT_PACKED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GBDT_NS_PREDICT_PACKED_OFF"](), "removed 2026-11-15 (lane/expdel2-44-trees): IDN_GBDT_NS_PREDICT_PACKED_OFF (old path, Trees (GBDT IDENTICAL packed non-symmetric predict rollback)) is gone; the default path is the only one; code recoverable at c48d22c288; restore: experiments/removed/MOJOLEARN_IDN_GBDT_NS_PREDICT_PACKED_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GBDT_PREDICT_FOUR_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GBDT_PREDICT_FOUR_OFF"](), "removed 2026-11-15 (lane/expdel2-44-trees): IDN_GBDT_PREDICT_FOUR_OFF (old path, Trees (GBDT IDENTICAL four-trees-per-launch predict rollback)) is gone; the default path is the only one; code recoverable at a8ef992718; restore: experiments/removed/MOJOLEARN_IDN_GBDT_PREDICT_FOUR_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GBDT_SYM_EST_DEFER_DRAIN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GBDT_SYM_EST_DEFER_DRAIN_OFF"](), "removed 2026-11-15 (lane/expdel2-44-trees): IDN_GBDT_SYM_EST_DEFER_DRAIN_OFF (old path, Trees (GBDT IDENTICAL symmetric estimation deferred drain rollback)) is gone; the default path is the only one; code recoverable at 71b261799b; restore: experiments/removed/MOJOLEARN_IDN_GBDT_SYM_EST_DEFER_DRAIN_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ORD_ONE_STEP_DEVICE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ORD_ONE_STEP_DEVICE_OFF"](), "removed 2026-11-15 (lane/expdel2-44-trees): IDN_ORD_ONE_STEP_DEVICE_OFF (old path, Trees (GBDT IDENTICAL Ordered one-step device estimation rollback)) is gone; the default path is the only one; code recoverable at 632cd99677; restore: experiments/removed/MOJOLEARN_IDN_ORD_ONE_STEP_DEVICE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ORD_STD_SCALE_DEVICE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ORD_STD_SCALE_DEVICE_OFF"](), "removed 2026-11-15 (lane/expdel2-44-trees): IDN_ORD_STD_SCALE_DEVICE_OFF (old path, Trees (GBDT IDENTICAL Ordered device std/scale rollback)) is gone; the default path is the only one; code recoverable at fce840609d; restore: experiments/removed/MOJOLEARN_IDN_ORD_STD_SCALE_DEVICE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_LEGACY_NARROW_ORD_ALL.patch
    comptime assert not is_defined["MOJOLEARN_LEGACY_NARROW_ORD_ALL"](), "removed 2026-11-15 (lane/expdel2-44-trees): LEGACY_NARROW_ORD_ALL (old path, Trees (GBDT Ordered FAST bundle legacy width gate)) is gone; the default path is the only one; code recoverable at f8ce124897; restore: experiments/removed/MOJOLEARN_LEGACY_NARROW_ORD_ALL.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MULTICLASS_HESSIAN_BATCH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_MULTICLASS_HESSIAN_BATCH_OFF"](), "removed 2026-11-15 (lane/expdel2-44-trees): MULTICLASS_HESSIAN_BATCH_OFF (old path, Trees (GBDT MultiClass Hessian batch rollback, FAST Apple)) is gone; the default path is the only one; code recoverable at 0415c68a0f; restore: experiments/removed/MOJOLEARN_MULTICLASS_HESSIAN_BATCH_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ORDERED_BATCH_EST_OFF.patch
    comptime assert not is_defined["MOJOLEARN_ORDERED_BATCH_EST_OFF"](), "removed 2026-11-15 (lane/expdel2-44-trees): ORDERED_BATCH_EST_OFF (old path, Trees (GBDT Ordered batched estimation rollback, FAST Apple)) is gone; the default path is the only one; code recoverable at accebf0183; restore: experiments/removed/MOJOLEARN_ORDERED_BATCH_EST_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ORD_ALL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_ORD_ALL_OFF"](), "removed 2026-11-15 (lane/expdel2-44-trees): ORD_ALL_OFF (old path, Trees (GBDT Ordered FAST bundle rollback)) is gone; the default path is the only one; code recoverable at 611ff32f4b; restore: experiments/removed/MOJOLEARN_ORD_ALL_OFF.patch"
    # MOJOLEARN_ORD_DOC_ID_STORAGE: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-trees; Apple fast opt-in arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_PAIRLOGIT_EST_REUSE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_PAIRLOGIT_EST_REUSE_OFF"](), "removed 2026-11-15 (lane/expdel2-44-trees): PAIRLOGIT_EST_REUSE_OFF (old path, Trees (GBDT PairLogit estimation reuse rollback, FAST Apple)) is gone; the default path is the only one; code recoverable at 6d03f59f60; restore: experiments/removed/MOJOLEARN_PAIRLOGIT_EST_REUSE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_PAIRLOGIT_GROUP_FUSED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_PAIRLOGIT_GROUP_FUSED_OFF"](), "removed 2026-11-15 (lane/expdel2-44-trees): PAIRLOGIT_GROUP_FUSED_OFF (old path, Trees (GBDT PairLogit group kernel rollback, FAST Apple)) is gone; the default path is the only one; code recoverable at a7ec4e7f91; restore: experiments/removed/MOJOLEARN_PAIRLOGIT_GROUP_FUSED_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_PL_GROUP_NARROW_OFF.patch
    comptime assert not is_defined["MOJOLEARN_PL_GROUP_NARROW_OFF"](), "removed 2026-11-15 (lane/expdel2-44-trees): PL_GROUP_NARROW_OFF (old path, Trees (GBDT PairLogit 128-thread group block rollback, FAST Apple)) is gone; the default path is the only one; code recoverable at 9dc8626747; restore: experiments/removed/MOJOLEARN_PL_GROUP_NARROW_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SYM_EST_ALL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SYM_EST_ALL_OFF"](), "removed 2026-11-15 (lane/expdel2-44-trees): SYM_EST_ALL_OFF (old path, Trees (GBDT symmetric estimation bundle rollback, FAST Apple)) is gone; the default path is the only one; code recoverable at ab7c8ba7f3; restore: experiments/removed/MOJOLEARN_SYM_EST_ALL_OFF.patch"
    # ---- end lane expdel2-44-trees
    # ---- lane fb15-ab-verdicts (2026-11-18): A/B verdicts of ab1 (NVIDIA L40S, wheel d652a4b0a1)
    # TOMBSTONE: MOJOLEARN_TREES_BLOCK_SCRATCH_FILL_OFF (DROPPED-slower: whole-buffer zero 1.38-1.40x on gbdt-lossguide istella fit, NV L40S 2026-11-18) deleted 2026-11-18 by lane/fb15-ab-verdicts; code recoverable at c989eb9222.
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_BLOCK_SCRATCH_FILL_OFF.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_TREES_BLOCK_SCRATCH_FILL_OFF"](), "removed 2026-11-18 (lane/fb15-ab-verdicts, A/B ab1 NVIDIA L40S, wheel d652a4b0a1, one run per arm): the whole-buffer histogram scratch zero was 1.38-1.40x slower on gbdt-lossguide istella fit (13.92 -> 19.24 s, 14.28 -> 20.02 s), digests equal; the block-span zero is the one form; recoverable at c989eb9222"
    # ---- end lane fb15-ab-verdicts
    # ---- lane fx-verdicts2 (2026-11-18): A/B verdicts of ab3 (NVIDIA L40S, wheel d652a4b0a1)
    # TOMBSTONE: MOJOLEARN_IDN_ET_DEVICE_LOOP (LOSER, A/B ab3 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests equal) deleted 2026-11-18 by lane/fx-verdicts2; code recoverable at 85c36e324f.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ET_DEVICE_LOOP.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_IDN_ET_DEVICE_LOOP"](), "removed 2026-11-18 (lane/fx-verdicts2, A/B ab3 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests equal): the depth-wise ET device loop was arm/base 1.030 geomean over 6 cells; the host queue is the depth-wise route (MOJOLEARN_IDN_ET_DEVICE_LOOP_EARLY_EXIT and best-first keep the device loop); recoverable at 85c36e324f"
    # TOMBSTONE: MOJOLEARN_IDN_GBDT_PREBINNED_INFER_OLD (LOSER, A/B ab3 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests equal) deleted 2026-11-18 by lane/fx-verdicts2; code recoverable at 85c36e324f.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GBDT_PREBINNED_INFER_OLD.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_IDN_GBDT_PREBINNED_INFER_OLD"](), "removed 2026-11-18 (lane/fx-verdicts2, A/B ab3 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests equal): the per-call Depthwise/Lossguide resident pack was arm/base 1.034 geomean over 24 cells; the pack held from the first predict is the route; recoverable at 85c36e324f"
    # ---- end lane fx-verdicts2
    # ---- lane fx-old-arms-triage (2026-11-19): measured losers of ab4 (NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T201200Z-d21ba0)
    # TOMBSTONE: MOJOLEARN_IDN_RF_DEVICE_LOOP (LOSER, A/B ab4 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T201200Z-d21ba0, one run per arm, digests equal) deleted 2026-11-19 by lane/fx-old-arms-triage; code recoverable at a7c89e7558.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_RF_DEVICE_LOOP.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_IDN_RF_DEVICE_LOOP"](), "removed 2026-11-19 (lane/fx-old-arms-triage, A/B ab4 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T201200Z-d21ba0, one run per arm): the RandomForest device level loop as the fit route was slower, fit istella 4407 -> 30417 ms (6.90), taxi 5352 -> 8918 (1.67); the host queue stays, the device loop body serves -D MOJOLEARN_IDN_RF_DEVICE_LOOP_EARLY_EXIT; code at a7c89e7558; restore: experiments/removed/MOJOLEARN_IDN_RF_DEVICE_LOOP.patch"
    # ---- end lane fx-old-arms-triage
    # ---- lane fx-verdicts4 (2026-11-19): verdicts of A/B round 2 (NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T154232Z-0642a5)
    # TOMBSTONE: MOJOLEARN_GBDT_FAST_RADIX8 (NOISE, A/B round 2 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T154232Z-0642a5, one run per arm, digests equal) deleted 2026-11-19 by lane/fx-verdicts4; code recoverable at 8f3f639b13.
    # Restore: git apply experiments/removed/MOJOLEARN_GBDT_FAST_RADIX8.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_GBDT_FAST_RADIX8"](), "removed 2026-11-19 (lane/fx-verdicts4, A/B round 2 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T154232Z-0642a5, one run per arm, digests equal): the GBDT low-bit sorts on eight-bit digits were arm/base 0.965-1.03 over ten fits (lossguide faster, symmetric and multiclass slower; noise); code at 8f3f639b13, experiments/removed/MOJOLEARN_GBDT_FAST_RADIX8.patch"
    # ---- end lane fx-verdicts4
    # ---- lane fx-verdicts7 (2026-11-21): verdicts of A/B abn9 (NVIDIA L40S host 79f003b8e8ae, wheel 3880db30a6af, job nvc-ab-20261121T003518Z-d38466)
    # TOMBSTONE: MOJOLEARN_TREES_FAST_HR2_BAGGING_MEMBERS (LOSER, A/B abn9 NVIDIA L40S, wheel 3880db30a6af, job nvc-ab-20261121T003518Z-d38466, one run per arm: bagging-clf and bagging-reg on istella and taxi, 8 cells 3.84-13.71 arm/base, 0 faster) deleted 2026-11-21 by lane/fx-verdicts7; code recoverable at 519388d23a.
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_FAST_HR2_BAGGING_MEMBERS.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_TREES_FAST_HR2_BAGGING_MEMBERS"](), "removed 2026-11-21 (lane/fx-verdicts7, A/B abn9 NVIDIA L40S, wheel 3880db30a6af, job nvc-ab-20261121T003518Z-d38466, one run per arm; LOSER: bagging-clf and bagging-reg on istella and taxi, 8 cells 3.84-13.71 arm/base, 0 faster): the per-member fit loop for a bag the forest fit admits is gone; a bag of best-splitter trees is one forest fit. Remove -D MOJOLEARN_TREES_FAST_HR2_BAGGING_MEMBERS; see docs/TOMBSTONES.md."
    # TOMBSTONE: MOJOLEARN_TREES_FAST_ADABOOST_REENCODE (NOISE, A/B abn9 NVIDIA L40S, wheel 3880db30a6af, job nvc-ab-20261121T003518Z-d38466, one run per arm: adaboost-clf on istella and taxi, 4 cells 0.98-1.04 arm/base) deleted 2026-11-21 by lane/fx-verdicts7; code recoverable at 519388d23a.
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_FAST_ADABOOST_REENCODE.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_TREES_FAST_ADABOOST_REENCODE"](), "removed 2026-11-21 (lane/fx-verdicts7, A/B abn9 NVIDIA L40S, wheel 3880db30a6af, job nvc-ab-20261121T003518Z-d38466, one run per arm; NOISE: adaboost-clf on istella and taxi, 4 cells 0.98-1.04 arm/base): the label re-encode per member is gone; the labels are encoded once per fit. Remove -D MOJOLEARN_TREES_FAST_ADABOOST_REENCODE; see docs/TOMBSTONES.md."
    # ---- end lane fx-verdicts7
    return True


comptime TREES_GUARDS_OK = _check_configuration()
