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
    # Restore: git apply experiments/removed/MOJOLEARN_AFCL_T08.patch
    comptime assert not is_defined["MOJOLEARN_AFCL_T08"](), "removed 2026-11-15 (lane/expdel2-03-base): AFCL_T08 (Forest inference (Apple FAST programme)) is gone: never-run Apple FAST programme card, no A/B (V2 rule g); code at 5bc2865240; restore: experiments/removed/MOJOLEARN_AFCL_T08.patch"
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
    # Restore: git apply experiments/removed/MOJOLEARN_AFT_P01.patch
    comptime assert not is_defined["MOJOLEARN_AFT_P01"](), "removed 2026-11-15 (lane/expdel2-03-base): AFT_P01 (Apple FAST trees programme (P)) is gone: never-run Apple FAST programme card, no A/B (V2 rule g); code at 5bc2865240; restore: experiments/removed/MOJOLEARN_AFT_P01.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AFT_P02.patch
    comptime assert not is_defined["MOJOLEARN_AFT_P02"](), "removed 2026-11-15 (lane/expdel2-03-base): AFT_P02 (Apple FAST trees programme (P)) is gone: never-run Apple FAST programme card, no A/B (V2 rule g); code at 5bc2865240; restore: experiments/removed/MOJOLEARN_AFT_P02.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AFT_P03.patch
    comptime assert not is_defined["MOJOLEARN_AFT_P03"](), "removed 2026-11-15 (lane/expdel2-03-base): AFT_P03 (Apple FAST trees programme (P)) is gone: never-run Apple FAST programme card, no A/B (V2 rule g); code at 5bc2865240; restore: experiments/removed/MOJOLEARN_AFT_P03.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AFT_P04.patch
    comptime assert not is_defined["MOJOLEARN_AFT_P04"](), "removed 2026-11-15 (lane/expdel2-03-base): AFT_P04 (Apple FAST trees programme (P)) is gone: never-run Apple FAST programme card, no A/B (V2 rule g); code at 5bc2865240; restore: experiments/removed/MOJOLEARN_AFT_P04.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AFT_P05.patch
    comptime assert not is_defined["MOJOLEARN_AFT_P05"](), "removed 2026-11-15 (lane/expdel2-03-base): AFT_P05 (Apple FAST trees programme (P)) is gone: never-run Apple FAST programme card, no A/B (V2 rule g); code at 5bc2865240; restore: experiments/removed/MOJOLEARN_AFT_P05.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AFT_P06.patch
    comptime assert not is_defined["MOJOLEARN_AFT_P06"](), "removed 2026-11-15 (lane/expdel2-03-base): AFT_P06 (Apple FAST trees programme (P)) is gone: never-run Apple FAST programme card, no A/B (V2 rule g); code at 5bc2865240; restore: experiments/removed/MOJOLEARN_AFT_P06.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AFT_P07.patch
    comptime assert not is_defined["MOJOLEARN_AFT_P07"](), "removed 2026-11-15 (lane/expdel2-03-base): AFT_P07 (Apple FAST trees programme (P)) is gone: never-run Apple FAST programme card, no A/B (V2 rule g); code at 5bc2865240; restore: experiments/removed/MOJOLEARN_AFT_P07.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AFT_P08.patch
    comptime assert not is_defined["MOJOLEARN_AFT_P08"](), "removed 2026-11-15 (lane/expdel2-03-base): AFT_P08 (Apple FAST trees programme (P)) is gone: never-run Apple FAST programme card, no A/B (V2 rule g); code at 5bc2865240; restore: experiments/removed/MOJOLEARN_AFT_P08.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AFT_P09.patch
    comptime assert not is_defined["MOJOLEARN_AFT_P09"](), "removed 2026-11-15 (lane/expdel2-03-base): AFT_P09 (Apple FAST trees programme (P)) is gone: never-run Apple FAST programme card, no A/B (V2 rule g); code at 5bc2865240; restore: experiments/removed/MOJOLEARN_AFT_P09.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AFT_P10.patch
    comptime assert not is_defined["MOJOLEARN_AFT_P10"](), "removed 2026-11-15 (lane/expdel2-03-base): AFT_P10 (Apple FAST trees programme (P)) is gone: never-run Apple FAST programme card, no A/B (V2 rule g); code at 5bc2865240; restore: experiments/removed/MOJOLEARN_AFT_P10.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AFT_P11.patch
    comptime assert not is_defined["MOJOLEARN_AFT_P11"](), "removed 2026-11-15 (lane/expdel2-03-base): AFT_P11 (Apple FAST trees programme (P)) is gone: never-run Apple FAST programme card, no A/B (V2 rule g); code at 5bc2865240; restore: experiments/removed/MOJOLEARN_AFT_P11.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AFT_P12.patch
    comptime assert not is_defined["MOJOLEARN_AFT_P12"](), "removed 2026-11-15 (lane/expdel2-03-base): AFT_P12 (Apple FAST trees programme (P)) is gone: never-run Apple FAST programme card, no A/B (V2 rule g); code at 5bc2865240; restore: experiments/removed/MOJOLEARN_AFT_P12.patch"
    # Lane expdel2-04-base (2026-11-15): unmeasured arms and old paths deleted under "An unmeasured arm is deleted" (docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md, docs/TOMBSTONES.md).
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
    return True


comptime TREES_GUARDS_OK = _check_configuration()
