# SPDX-License-Identifier: Apache-2.0
"""Opt-in strategy admission, evaluated through GLOBAL_NUMERIC_MODE.

A standalone unused comptime binding is lazy and cannot enforce a guard.
This predicate is required by the numeric-mode constant used in every binding.
"""
from std.sys.compile import is_defined, get_defined_int
#
# Since 2026-11-04 this file holds only the guards whose defines are read in code by bindings of more than one
# family (shared GEMM, row-norm, CV and kNN arms) and the records that could not be tied to one family. Every
# other entry moved, verbatim and in order, to its family's module core/experiment_guards_<family>.mojo, which
# that family's bindings import and assert in their build function; a guard added here rebuilds every binding,
# one added there rebuilds that family only (docs/build/HOT_GROUPS.md, tools/hooks/rebuild_fence.py).


def _check_configuration() -> Bool:
    comptime assert not (is_defined["MOJOLEARN_AFCL_G01"]() and is_defined["MOJOLEARN_KNN_FAST_MMA_OFF"]()), "incompatible integrated strategies: MOJOLEARN_AFCL_G01 / MOJOLEARN_KNN_FAST_MMA_OFF"
    # Retired alternative-override defines: each became ONE define with arms (lane classical-misc).
    comptime assert not (is_defined["MOJOLEARN_C52_PAIR_128"]() or is_defined["MOJOLEARN_C52_PAIR_512"]()), "retired: use -D MOJOLEARN_C52_PAIR_ROWS=128|512"
    comptime assert not is_defined["MOJOLEARN_C52_PAIR_ROWS"]() or get_defined_int["MOJOLEARN_C52_PAIR_ROWS",128]() == 128 or get_defined_int["MOJOLEARN_C52_PAIR_ROWS",128]() == 512, "invalid MOJOLEARN_C52_PAIR_ROWS (128|512)"
    comptime assert not (is_defined["MOJOLEARN_CLASSICAL_C19_ORDERED_128"]() or is_defined["MOJOLEARN_CLASSICAL_C19_ORDERED_32"]()), "retired: use -D MOJOLEARN_CLASSICAL_C19_SGD_CHUNK=32|128|2048"
    comptime assert get_defined_int["MOJOLEARN_CLASSICAL_C19_SGD_CHUNK",2048]() == 32 or get_defined_int["MOJOLEARN_CLASSICAL_C19_SGD_CHUNK",2048]() == 128 or get_defined_int["MOJOLEARN_CLASSICAL_C19_SGD_CHUNK",2048]() == 2048, "invalid MOJOLEARN_CLASSICAL_C19_SGD_CHUNK (32|128|2048)"
    # Lane classical-fixes (2026-10-07): the KMeans convergence chunk is ONE int sweep define.
    comptime assert not (is_defined["MOJOLEARN_IDN_KMEANS_CONV_CHUNK_1"]() or is_defined["MOJOLEARN_IDN_KMEANS_CONV_CHUNK_2"]() or is_defined["MOJOLEARN_IDN_KMEANS_CONV_CHUNK_4"]() or is_defined["MOJOLEARN_IDN_KMEANS_CONV_CHUNK_16"]() or is_defined["MOJOLEARN_IDN_KMEANS_CONV_CHUNK_32"]()), "retired: use -D MOJOLEARN_IDN_KMEANS_CONV_CHUNK=1|2|4|8|16|32"
    comptime KCC = get_defined_int["MOJOLEARN_IDN_KMEANS_CONV_CHUNK",8]()
    comptime assert KCC == 1 or KCC == 2 or KCC == 4 or KCC == 8 or KCC == 16 or KCC == 32, "invalid MOJOLEARN_IDN_KMEANS_CONV_CHUNK (1|2|4|8|16|32)"
    # Lane grid-prune (2026-10-07): c06_norms is ONE control with arms {rows2, rows4, small_d}; small_d returns before
    # norm_rows when d <= 32 (core/row_norms.mojo:220-232), so the pair is never a distinct configuration.
    comptime assert not (is_defined["MOJOLEARN_CLASSICAL_C06_NORM_ROWS"]() and is_defined["MOJOLEARN_CLASSICAL_C06_SMALL_D_THREAD"]()), "c06_norms takes one arm: MOJOLEARN_CLASSICAL_C06_NORM_ROWS=2|4 or MOJOLEARN_CLASSICAL_C06_SMALL_D_THREAD, not both"
    # ENETCV_SCORE_BLOCKS: 1024 promoted to the IDENTICAL default 2026-10-11 (lane/owed-decide, NV v0063/v0064, AMD a1645/a1646,
    # avg ratio 0.935): absent = 1024; =4096 or _OFF the old 4096 default. The unblocked path is FAST only.
    comptime assert get_defined_int["MOJOLEARN_CLASSICAL_ENETCV_SCORE_BLOCKS",1024]() == 1024 or get_defined_int["MOJOLEARN_CLASSICAL_ENETCV_SCORE_BLOCKS",1024]() == 4096, "invalid MOJOLEARN_CLASSICAL_ENETCV_SCORE_BLOCKS (1024|4096; -D MOJOLEARN_CLASSICAL_ENETCV_SCORE_BLOCKS_OFF = 4096)"
    comptime assert not (is_defined["MOJOLEARN_CLASSICAL_ENETCV_SCORE_BLOCKS_OFF"]() and is_defined["MOJOLEARN_CLASSICAL_ENETCV_SCORE_BLOCKS"]()), "MOJOLEARN_CLASSICAL_ENETCV_SCORE_BLOCKS_OFF (4096) and MOJOLEARN_CLASSICAL_ENETCV_SCORE_BLOCKS=N are exclusive arms"
    # ENETCV_FOLD_BLOCKS promoted 2026-10-10 (lane/grid-act-6, grid g50ebe26a5): absent = 64 chunks per fold; =16|32 the other arms; _OFF the old staged means + Gram path.
    comptime assert get_defined_int["MOJOLEARN_CLASSICAL_ENETCV_FOLD_BLOCKS",64]() == 16 or get_defined_int["MOJOLEARN_CLASSICAL_ENETCV_FOLD_BLOCKS",64]() == 32 or get_defined_int["MOJOLEARN_CLASSICAL_ENETCV_FOLD_BLOCKS",64]() == 64, "invalid MOJOLEARN_CLASSICAL_ENETCV_FOLD_BLOCKS (16|32|64; promoted 2026-10-10, absent = 64; the old path is -D MOJOLEARN_CLASSICAL_ENETCV_FOLD_BLOCKS_OFF)"
    comptime assert not (is_defined["MOJOLEARN_CLASSICAL_ENETCV_FOLD_BLOCKS_OFF"]() and is_defined["MOJOLEARN_CLASSICAL_ENETCV_FOLD_BLOCKS"]()), "MOJOLEARN_CLASSICAL_ENETCV_FOLD_BLOCKS_OFF (old staged path) and MOJOLEARN_CLASSICAL_ENETCV_FOLD_BLOCKS=N are exclusive arms"
    # Lane grid-fixups-1 (2026-10-08): C17_LS_TRIALS (2 trials) and the I12 loser IDN_QN_EXACT_TRIALS (4 trials) pick the
    # trial count of one line search (glm/impl/qn/qn_linesearch.mojo:236); together the first silently overrides the second.
    comptime assert not (is_defined["MOJOLEARN_CLASSICAL_C17_LS_TRIALS"]() and is_defined["MOJOLEARN_IDN_QN_EXACT_TRIALS"]()), "qn exact trials take one count: MOJOLEARN_CLASSICAL_C17_LS_TRIALS (2) or MOJOLEARN_IDN_QN_EXACT_TRIALS (4), not both"
    comptime assert not is_defined["MOJOLEARN_C58_TEAM64"](), "retired: use -D MOJOLEARN_C58_TEAM_MIB=64|256"
    comptime assert get_defined_int["MOJOLEARN_C58_TEAM_MIB",256]() == 64 or get_defined_int["MOJOLEARN_C58_TEAM_MIB",256]() == 256, "invalid MOJOLEARN_C58_TEAM_MIB (64|256)"
    # Lane neural-gemm-attn-dedupe (2026-10-07): each merged idea is ONE define
    # with named arms, so its old mutual exclusions hold by construction.
    # Retired defines are refused so a stale catalog arm cannot build as a
    # silent incumbent.
    comptime S = get_defined_int["MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE",0]()
    comptime assert S == 0 or S == 1 or S == 2 or S == 4 or S == 9 or S == 10 or S == 11 or S == 15 or S == 24, "MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE arms: 1 geometry, 2 stream, 4 stream_exact, 9 pages, 10 cost, 11 fold_exact, 15 threadmap, 24 pages_threadmap (8 async deleted by lane/grid-prune; 3 stream_all removed 2026-10-08 by lane/grid-act-3: slower, gemm NV 2.07x / AMD 1.78x, lm-train-step 1.11x / 1.07x (grid ge123e6f9); see EXPERIMENTS.md)"
    comptime assert S != 10 or is_defined["MOJOLEARN_IDN_NEURAL_FILL_BLOCKS"](), "schedule 10 (cost) needs an explicit hardware fill budget"
    comptime assert S != 1 or is_defined["MOJOLEARN_GEMM_ARM_TRIAL"](), "schedule 1 (geometry) needs MOJOLEARN_GEMM_ARM_TRIAL"
    comptime LEAF = get_defined_int["MOJOLEARN_IDN_GEMM_LEAF",0]()
    comptime assert LEAF == 0 or LEAF == 1 or LEAF == 2 or LEAF == 3, "MOJOLEARN_IDN_GEMM_LEAF arms: 1 neural128, 2 neural256, 3 all256"
    comptime assert not (S == 1 and (LEAF == 1 or LEAF == 2 or is_defined["MOJOLEARN_IDN_NEURAL_CHAINS"]())), "schedule 1 (geometry) runs incumbent geometries; it excludes the neural leaf/chains profile"
    comptime assert not (is_defined["MOJOLEARN_IDN_CE_DENOM_ROWFOLD"]() and (LEAF == 1 or LEAF == 2 or is_defined["MOJOLEARN_IDN_NEURAL_CHAINS"]())), "MOJOLEARN_IDN_CE_DENOM_ROWFOLD reproduces the v1 GEMM contract chain and fold; it excludes the neural leaf/chains profile"
    # TOMBSTONE: MOJOLEARN_IDN_ATTN_SOFTMAX=2 (slower) deleted 2026-10-10 by c1bf9d832; code recoverable at ca25d9321.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ATTN_SOFTMAX-arm2.patch
    # Both arms are gone, so the define itself is refused, with its forward rows-per-block control MOJOLEARN_IDN_ATTN_V2_TQ.
    comptime assert not is_defined["MOJOLEARN_IDN_GEMM_FOLD_LEAF_64"](), "MOJOLEARN_IDN_GEMM_FOLD_LEAF_64 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    # Lane grid-prune (2026-10-07): deleted losers and dead arms (rows in docs/apple-fast/EXPERIMENTS.md
    # "IDENTICAL grid prune"); recoverable at main ab554bb4a. A stale build line must not run the incumbent silently.
    comptime assert not is_defined["MOJOLEARN_IDN_GEMM_TILE_SHORT_K"](), "removed: MOJOLEARN_IDN_GEMM_TILE_SHORT_K (arms equal measured pass62 losers / the NVIDIA default; inert on AMD)"
    comptime assert not is_defined["MOJOLEARN_IDN_GEMM_FS2"](), "removed: MOJOLEARN_IDN_GEMM_FS2 (OVN N02 noise vs FS4)"
    comptime assert not is_defined["MOJOLEARN_IDN_GEMM_COMPACT_LIVE_TILE"](), "removed: MOJOLEARN_IDN_GEMM_COMPACT_LIVE_TILE (OVN A05 slower on NVIDIA and AMD)"
    comptime assert not is_defined["MOJOLEARN_IDN_GEMM_GROUP_TILES_BODY"](), "removed: MOJOLEARN_IDN_GEMM_GROUP_TILES_BODY (OVN N01 noise on the L40S)"
    comptime M1S = get_defined_int["MOJOLEARN_IDN_M1_SCAN",3]()
    # TOMBSTONE: MOJOLEARN_IDN_M1_SCAN=1 (NN34 affine_prefix Mamba-1 scan; LOSER: mamba1-forward gaussian NV 5.40 -> 10.41 ms (1.93), AMD 3.14 -> 8.38 ms (2.67), avg 2.30, vs arm 3 persistent) deleted 2026-10-11 by 291629888; code recoverable at 291629888^.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_M1_SCAN_1.patch (neural_scan_profile.mojo, neural_mamba_scan.mojo and the host/gen twins come back with it); record in docs/TOMBSTONES.md.
    comptime assert M1S != 1, "removed: MOJOLEARN_IDN_M1_SCAN=1 (NN34 affine_prefix) deleted 2026-10-11: LOSER, mamba1-forward NV 5.40 -> 10.41 ms (1.93x), AMD 3.14 -> 8.38 ms (2.67x), avg 2.30x; see docs/TOMBSTONES.md"
    comptime assert M1S == 0 or M1S == 2 or M1S == 3, "MOJOLEARN_IDN_M1_SCAN arms: 0 old scan, 2 state_window, 3 persistent (1 removed 2026-10-11)"
    comptime assert M1S == 3 or not (is_defined["MOJOLEARN_IDN_M1_PERSISTENT_SCAN_CH"]() or is_defined["MOJOLEARN_IDN_M1_PERSISTENT_SCAN_TOKENS"]()), "MOJOLEARN_IDN_M1_PERSISTENT_SCAN_CH/_TOKENS are read only by MOJOLEARN_IDN_M1_SCAN=3 (persistent)"
    comptime assert not (is_defined["MOJOLEARN_NI14_BOUNDED_COL2IM"]() or is_defined["MOJOLEARN_NI14_TILED_COL2IM"]()), "merged: use -D MOJOLEARN_NI14_COL2IM=1 (bounded) | 2 (tiled)"
    comptime assert get_defined_int["MOJOLEARN_NI14_COL2IM",0]() >= 0 and get_defined_int["MOJOLEARN_NI14_COL2IM",0]() <= 2, "MOJOLEARN_NI14_COL2IM arms: 1 bounded, 2 tiled"
    comptime assert not (is_defined["MOJOLEARN_TREES_C50_GB_PACKED"]() and is_defined["MOJOLEARN_IDN_GBDT_APPLY_WIDE"]()), "MOJOLEARN_TREES_C50_GB_PACKED returns before MOJOLEARN_IDN_GBDT_APPLY_WIDE (gbdt/resident_model.mojo): the pair == C50"
    # Deleted 2026-10-08 (lane grid-losers-1): IDENTICAL grid ge123e6f9 losers (docs/apple-fast/EXPERIMENTS.md). Recoverable at main ad7ed2370.
    # C25_PROJECTION_REUSE: deleted 2026-10-08 above, restored 2026-10-10 (lane c25-restore-promote) from the tombstone patch after grid pass 2.
    comptime assert not is_defined["MOJOLEARN_IDN_GEMM_GROUP_SLACK_8"](), "promoted 2026-10-08 (grid ge123e6f9, lane/grid-act-2): GEMM group slack 8 is the IDENTICAL default (gemm NV 0.743x / AMD 0.99x); drop -D MOJOLEARN_IDN_GEMM_GROUP_SLACK_8, use -D MOJOLEARN_IDN_GEMM_GROUP_SLACK_OFF for the old slack 4"
    comptime assert not (is_defined["MOJOLEARN_IDN_GEMM_GROUP_SLACK_OFF"]() and is_defined["MOJOLEARN_IDN_GEMM_GROUP_SLACK_2"]()), "MOJOLEARN_IDN_GEMM_GROUP_SLACK_OFF (slack 4) and MOJOLEARN_IDN_GEMM_GROUP_SLACK_2 are exclusive arms"
    # Deleted 2026-10-08 (lane grid-act-2): IDENTICAL grid ge123e6f9 losers (docs/apple-fast/EXPERIMENTS.md). Recoverable at main 42d1e42c6.
    comptime assert not (is_defined["MOJOLEARN_C37_FUSED_ACCUMULATE"]() or is_defined["MOJOLEARN_C37_FUSED_ROWS"]()), "removed: MOJOLEARN_C37_FUSED_ACCUMULATE (and _FUSED_ROWS) retired 2026-10-08: slower, kmeans NV 104.8x / AMD 0.84x istella, NV 68.6x / AMD 0.90x taxi (vendor split; combined 9.4x / 7.8x), inertia SAME (grid ge123e6f9); see EXPERIMENTS.md"
    # Lane grid-act-3 (2026-10-08): IDENTICAL grid ge123e6f9 promotions and losers (docs/apple-fast/EXPERIMENTS.md). Deleted code recoverable at main bc10b8b56.
    comptime TMB = get_defined_int["MOJOLEARN_IDN_GEMM_TILE_MIN_BLOCKS",512]()
    comptime assert TMB == 192 or TMB == 512 or TMB == 1024, "MOJOLEARN_IDN_GEMM_TILE_MIN_BLOCKS legal set {192, 512, 1024}"
    # Lane grid-act-5 (2026-10-08): IDENTICAL grid ge123e6f9 gemm_leaf split and neural_gemm_epilogue loser (docs/apple-fast/EXPERIMENTS.md). Deleted code recoverable at main 8ed94710a.
    comptime assert not (is_defined["MOJOLEARN_IDN_MLP_GEMM_LEAF128_OFF"]() and (get_defined_int["MOJOLEARN_IDN_GEMM_LEAF",0]() == 1 or get_defined_int["MOJOLEARN_IDN_GEMM_LEAF",0]() == 2)), "split 2026-10-08 (grid ge123e6f9, lane/grid-act-5): gemm_leaf=neural128 is the IDENTICAL default for the MLP training step only (mlp-train-step 0.886x, same bits; every other neural model slower, lm-train-step 13.6x); MOJOLEARN_IDN_MLP_GEMM_LEAF128_OFF has no effect with MOJOLEARN_IDN_GEMM_LEAF=1|2, which routes every neural caller to the profile"
    # TOMBSTONE: MOJOLEARN_GEMM_FAST_LEAF512 (NOISE, A/B abf2 and round 3) deleted 2026-11-20 by lane/fx-verdicts6 with this refusal of the arm beside MOJOLEARN_IDN_MLP_GEMM_LEAF128_OFF; the define itself is refused in the lane fx-verdicts6 block below; code recoverable at 06d1e8ff34.
    # Restore: git apply experiments/removed/MOJOLEARN_GEMM_FAST_LEAF512.patch
    # TOMBSTONE: MOJOLEARN_IDN_MLP_GEMM_LEAF128_OFF (its MLP step was deleted, plan row X2) deleted 2026-11-09 by lane/x2-small-mlp-on-executor; code recoverable at f59d913b0b. Refused in core/experiment_guards_neural.mojo.
    # TOMBSTONE restored 2026-11-09 by lane/r1-restore-measured-winners: MOJOLEARN_IDN_MLP_GEMM_LEAF128_OFF is live again (autograd/small_mlp_op.mojo); core/experiment_guards_neural.mojo no longer refuses it.
    # TOMBSTONE: MOJOLEARN_IDN_MLP_GEMM_LEAF128 (the leaf-128 body) and its switch MOJOLEARN_IDN_MLP_GEMM_LEAF128_OFF deleted 2026-11-24 by lane/o3-nmlp (ab-outcome-3; A/B nvc-ab-20261124T115722Z-d8f8b2, LOSER); the switch is refused outright in core/experiment_guards_neural.mojo, so the conditional refusal above no longer has a live arm; code recoverable at 774ef6eb1f.
    # Lane i17-decide-nbr-lin-prep (2026-10-11), removed: A/B on main b2657cc89, one run per arm, NV L40S nv2 / AMD MI325X (docs/TOMBSTONES.md,
    # docs/apple-fast/EXPERIMENTS.md, experiments/removed/*.patch).
    comptime assert not is_defined["MOJOLEARN_IDN_GEMM_NV_MICROTILE"](), "removed 2026-10-11 (lane/i17-decide-nbr-lin-prep, c38d7bc89): IDN_GEMM_NV_MICROTILE (NVIDIA packed GEMM at K window 32) was SLOWER: neural:gemm gaussian NV 48.5 -> 55.9 ms (1.15x, v0050); code at c38d7bc89^, patch experiments/removed/MOJOLEARN_IDN_GEMM_NV_MICROTILE.patch"
    # Lane expdel2-04-base (2026-11-15): unmeasured arms and old paths deleted under "An unmeasured arm is deleted" (plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history), docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_2711_GRAM_STRIDED_DEAD.patch
    comptime assert not is_defined["MOJOLEARN_2711_GRAM_STRIDED_DEAD"](), "removed 2026-11-15 (lane/expdel2-04-base): 2711_GRAM_STRIDED_DEAD (opt-in arm, Decomposition (split-K Gram, IDENTICAL); no A/B recorded for the opt-in arm (rule a)) is gone; the default stays; code recoverable at 7b75b0819e; restore: experiments/removed/MOJOLEARN_2711_GRAM_STRIDED_DEAD.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_537_GEMM_IDENT_SWAP.patch
    comptime assert not is_defined["MOJOLEARN_537_GEMM_IDENT_SWAP"](), "removed 2026-11-15 (lane/expdel2-04-base): 537_GEMM_IDENT_SWAP (opt-in arm, GEMM (IDENTICAL operand swap); no A/B recorded for the opt-in arm (rule a)) is gone; the default stays; code recoverable at 33a25f5e7e; restore: experiments/removed/MOJOLEARN_537_GEMM_IDENT_SWAP.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_APPLE_FAST_STEP_UNROLL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_APPLE_FAST_STEP_UNROLL_OFF"](), "removed 2026-11-15 (lane/expdel2-04-base): APPLE_FAST_STEP_UNROLL_OFF (old path, Strided walks (FAST Apple); no A/B recorded beside the FAST Apple default (rule b)) is gone; the default stays; code recoverable at c34327db85; restore: experiments/removed/MOJOLEARN_APPLE_FAST_STEP_UNROLL_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_APPLE_GEMM_NT_MMA_OFF.patch
    comptime assert not is_defined["MOJOLEARN_APPLE_GEMM_NT_MMA_OFF"](), "removed 2026-11-15 (lane/expdel2-04-base): APPLE_GEMM_NT_MMA_OFF (old path, GEMM (Apple IDENTICAL simdgroup gemm_nt); no A/B recorded beside the default (rule b)) is gone; the default stays; code recoverable at ea1e949b3d; restore: experiments/removed/MOJOLEARN_APPLE_GEMM_NT_MMA_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_APPLE_GEMM_NT_TILED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_APPLE_GEMM_NT_TILED_OFF"](), "removed 2026-11-15 (lane/expdel2-04-base): APPLE_GEMM_NT_TILED_OFF (old path, GEMM (IDENTICAL tiled gemm_nt); no A/B recorded beside the default (rule b)) is gone; the default stays; code recoverable at 337c096a5b; restore: experiments/removed/MOJOLEARN_APPLE_GEMM_NT_TILED_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_APPLE_STEP_UNROLL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_APPLE_STEP_UNROLL_OFF"](), "removed 2026-11-15 (lane/expdel2-04-base): APPLE_STEP_UNROLL_OFF (old path, Strided walks (IDENTICAL); no A/B recorded beside the default (rule b)) is gone; the default stays; code recoverable at 853900db5b; restore: experiments/removed/MOJOLEARN_APPLE_STEP_UNROLL_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_EXPERIMENT_IDENTICAL_CALLPATH.patch
    comptime assert not is_defined["MOJOLEARN_EXPERIMENT_IDENTICAL_CALLPATH"](), "removed 2026-11-15 (lane/expdel2-04-base): EXPERIMENT_IDENTICAL_CALLPATH (opt-in arm, Shared IDENTICAL call session (experiment programme); no A/B recorded for the opt-in arm (rule a)) is gone; the default stays; code recoverable at d1343040f6; restore: experiments/removed/MOJOLEARN_EXPERIMENT_IDENTICAL_CALLPATH.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_FAST_LCG_MODULO.patch
    comptime assert not is_defined["MOJOLEARN_FAST_LCG_MODULO"](), "removed 2026-11-15 (lane/expdel2-04-base): FAST_LCG_MODULO (opt-in arm, Shuffle iterator (FAST); no A/B recorded (Apple fast opt-in, rule a)) is gone; the default stays; code recoverable at b086331d7a; restore: experiments/removed/MOJOLEARN_FAST_LCG_MODULO.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_GEMV_FAST_VENDOR.patch
    comptime assert not is_defined["MOJOLEARN_GEMV_FAST_VENDOR"](), "removed 2026-11-15 (lane/expdel2-04-base): GEMV_FAST_VENDOR (opt-in arm, GEMV (FAST Apple); no A/B recorded (Apple fast opt-in, rule a)) is gone; the default stays; code recoverable at 8fc20d312b; restore: experiments/removed/MOJOLEARN_GEMV_FAST_VENDOR.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_HPDEV_CAST_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_HPDEV_CAST_OFF"](), "removed 2026-11-15 (lane/expdel2-04-base): IDN_HPDEV_CAST_OFF (old path, Hotpath device steps (IDENTICAL cast); no A/B recorded beside the default (rule b)) is gone; the default stays; code recoverable at 8159159cc4; restore: experiments/removed/MOJOLEARN_IDN_HPDEV_CAST_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_HPDEV_ELEM_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_HPDEV_ELEM_OFF"](), "removed 2026-11-15 (lane/expdel2-04-base): IDN_HPDEV_ELEM_OFF (old path, Hotpath device steps (IDENTICAL element ops); no A/B recorded beside the default (rule b)) is gone; the default stays; code recoverable at 18a80bdd0b; restore: experiments/removed/MOJOLEARN_IDN_HPDEV_ELEM_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_HPDEV_ISUM_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_HPDEV_ISUM_OFF"](), "removed 2026-11-15 (lane/expdel2-04-base): IDN_HPDEV_ISUM_OFF (old path, Hotpath device steps (IDENTICAL integer sum); no A/B recorded beside the default (rule b)) is gone; the default stays; code recoverable at 8f026542e3; restore: experiments/removed/MOJOLEARN_IDN_HPDEV_ISUM_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_HPDEV_LABELS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_HPDEV_LABELS_OFF"](), "removed 2026-11-15 (lane/expdel2-04-base): IDN_HPDEV_LABELS_OFF (old path, Hotpath device steps (IDENTICAL label encode); no A/B recorded beside the default (rule b)) is gone; the default stays; code recoverable at 7fe1ce981e; restore: experiments/removed/MOJOLEARN_IDN_HPDEV_LABELS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_HPDEV_REDUCE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_HPDEV_REDUCE_OFF"](), "removed 2026-11-15 (lane/expdel2-04-base): IDN_HPDEV_REDUCE_OFF (old path, Hotpath device steps (IDENTICAL reduce); no A/B recorded beside the default (rule b)) is gone; the default stays; code recoverable at 8ec3732a1b; restore: experiments/removed/MOJOLEARN_IDN_HPDEV_REDUCE_OFF.patch"
    # MOJOLEARN_IDN_OLS_PREDICT_TILED: refusal removed 2026-11-15 (restored; owed B arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_IDN_QUERY_X_HELD_OFF: refusal removed 2026-11-15 (restored; owed B arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SCRATCH_POOL.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SCRATCH_POOL"](), "removed 2026-11-15 (lane/expdel2-04-base): IDN_SCRATCH_POOL (opt-in arm, Scratch buffer pool (IDENTICAL); no A/B recorded for the opt-in arm (rule a)) is gone; the default stays; code recoverable at 25fca5d3e3; restore: experiments/removed/MOJOLEARN_IDN_SCRATCH_POOL.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SCRATCH_POOL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SCRATCH_POOL_OFF"](), "removed 2026-11-15 (lane/expdel2-04-base): IDN_SCRATCH_POOL_OFF (old path, Scratch buffer pool (IDENTICAL); no A/B recorded beside the default (rule b)) is gone; the default stays; code recoverable at 2ec9cce908; restore: experiments/removed/MOJOLEARN_IDN_SCRATCH_POOL_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SEG_RADIX8_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SEG_RADIX8_OFF"](), "removed 2026-11-15 (lane/expdel2-04-base): IDN_SEG_RADIX8_OFF (old path, Segmented sort (IDENTICAL 8-bit radix); no A/B recorded beside the default (rule b)) is gone; the default stays; code recoverable at af0618e292; restore: experiments/removed/MOJOLEARN_IDN_SEG_RADIX8_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_XTY_TILE_1024.patch
    comptime assert not is_defined["MOJOLEARN_IDN_XTY_TILE_1024"](), "removed 2026-11-15 (lane/expdel2-04-base): IDN_XTY_TILE_1024 (opt-in arm, Host tile fold (X^T y tile size); no A/B recorded for the opt-in arm (rule a)) is gone; the default stays; code recoverable at 444a21c6a3; restore: experiments/removed/MOJOLEARN_IDN_XTY_TILE_1024.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_XTY_TILE_64.patch
    comptime assert not is_defined["MOJOLEARN_IDN_XTY_TILE_64"](), "removed 2026-11-15 (lane/expdel2-04-base): IDN_XTY_TILE_64 (opt-in arm, Host tile fold (X^T y tile size); no A/B recorded for the opt-in arm (rule a)) is gone; the default stays; code recoverable at 792ad25b28; restore: experiments/removed/MOJOLEARN_IDN_XTY_TILE_64.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NV_AMD_STEPS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_NV_AMD_STEPS_OFF"](), "removed 2026-11-15 (lane/expdel2-04-base): NV_AMD_STEPS_OFF (old path, Strided walks and gemv_n (NVIDIA / AMD IDENTICAL); no A/B recorded beside the default (rule b)) is gone; the default stays; code recoverable at d1fe4369a6; restore: experiments/removed/MOJOLEARN_NV_AMD_STEPS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SEG_SUMS_SERIAL.patch
    comptime assert not is_defined["MOJOLEARN_SEG_SUMS_SERIAL"](), "removed 2026-11-15 (lane/expdel2-04-base): SEG_SUMS_SERIAL (opt-in arm, Segmented sort block-sum scan (FAST Apple); no A/B recorded for the opt-in arm on current source (rule a) is gone; the default stays; code recoverable at 28ac8cfd29; restore: experiments/removed/MOJOLEARN_SEG_SUMS_SERIAL.patch"
    # ---- end lane expdel2-04-base
    # lane sqrt-comptime-step (2026-11-16): MOJOLEARN_NUMERICS_FAST_SQRT_COMPTIME_STEP is read in checks/numerics.mojo
    # (portable_sqrtf), which every binding of every family reaches, so its guard is here, not in a family module.
    # lane fx-verdicts (2026-11-18): promoted to the default on every column (A/B ab2, NVIDIA L40S, digests equal on
    # all 36 cells); the old opt-in name is refused so a stale build line does not silently mean the default.
    comptime assert not is_defined["MOJOLEARN_NUMERICS_FAST_SQRT_COMPTIME_STEP"](), "promoted 2026-11-18 (lane/fx-verdicts, A/B ab2 NVIDIA L40S, wheel d652a4b0a1, one run per arm, digests equal on 36 cells, hdbscan/knn taxi 0.80-0.88): the comptime candidate step of portable_sqrtf is the default on every column; drop -D MOJOLEARN_NUMERICS_FAST_SQRT_COMPTIME_STEP (the run-time loop and its _OFF switch were deleted 2026-11-23)"
    # TOMBSTONE: MOJOLEARN_NUMERICS_FAST_SQRT_COMPTIME_STEP_OFF (removed: decided A/B ab2) deleted 2026-11-23 by lane/ab-prep-neural; code recoverable at 4d76a9e16e.
    # Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_NUMERICS_FAST_SQRT_COMPTIME_STEP_OFF.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_NUMERICS_FAST_SQRT_COMPTIME_STEP_OFF"](), "removed 2026-11-23 (lane/ab-prep-neural, decided A/B ab2 2026-11-18 NVIDIA L40S c6dd9ebea3e3, digests equal on 36 cells, geomean 0.973, hdbscan/knn taxi 0.80-0.88): the run-time candidate loop of portable_sqrtf is gone, the comptime step is the one form; code at 4d76a9e16e; restore: ~/mojolearn-evidence/experiments-removed/MOJOLEARN_NUMERICS_FAST_SQRT_COMPTIME_STEP_OFF.patch"
    # lane fx-verdicts (2026-11-18): MOJOLEARN_IDN_NV_FMA_ASM_OLD (checks/nv_fma_rn.mojo, reached by the neighbors,
    # GEMM and attention kernels of several families) won its A/B and is the default; the old name is refused.
    comptime assert not is_defined["MOJOLEARN_IDN_NV_FMA_ASM_OLD"](), "promoted 2026-11-18 (lane/fx-verdicts, A/B ab2 NVIDIA L40S, wheel d652a4b0a1, one run per arm, digests equal on all 30 cells, geomean 0.971): the inline-asm fma.rn.f32 helper is the NVIDIA identical default in checks/nv_fma_rn.mojo; drop -D MOJOLEARN_IDN_NV_FMA_ASM_OLD (the llvm.nvvm.fma.rn.f intrinsic and its switch are deleted)"
    # TOMBSTONE: MOJOLEARN_IDN_NV_FMA_ASM_OFF (old path of the promotion of MOJOLEARN_IDN_NV_FMA_ASM_OLD: the llvm.nvvm.fma.rn.f intrinsic LOST A/B ab2 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T140734Z-8f0269, one run per arm, digests equal on all 30 cells, asm/intrinsic geomean 0.971 over 29 cells, ols taxi 0.74-0.80, ridge taxi 0.82) the switch that restored the intrinsic at every NVIDIA identical FMA site (checks/nv_fma_rn.mojo) deleted 2026-11-22 by lane/verdicts-apply; code recoverable at 5dfd4a2e67.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NV_FMA_ASM_OFF.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_IDN_NV_FMA_ASM_OFF"](), "removed 2026-11-22 (lane/verdicts-apply, A/B ab2 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T140734Z-8f0269, one run per arm, digests equal on all 30 cells): the llvm.nvvm.fma.rn.f intrinsic was the slower form (asm/intrinsic geomean 0.971 over 29 cells, ols taxi 0.74-0.80, ridge taxi 0.82); the inline-asm fma.rn.f32 helper is the one form in checks/nv_fma_rn.mojo. docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C06_NORM_ROWS.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C06_NORM_ROWS"](), "removed 2026-11-15 (lane/expdel2-07-base): CLASSICAL_C06_NORM_ROWS (Row norms (rows per norm block 2/4 for KNN / KDE / KMeans, IDENTICAL)) was opt-in arm; EXP:1519 RESTRUCTURED, A/B owed and never run; the opt-in arm is gone and the default path is unchanged; code recoverable at 526172a06e; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C06_SMALL_D_THREAD.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C06_SMALL_D_THREAD"](), "removed 2026-11-15 (lane/expdel2-07-base): CLASSICAL_C06_SMALL_D_THREAD (Row norms (one thread per row at d <= 32, IDENTICAL)) was opt-in arm; EXP:1519 A/B owed and never run; the opt-in arm is gone and the default path is unchanged; code recoverable at 6a40fe1b92; see docs/TOMBSTONES.md"
    # ---- lane fx-verdicts5 (2026-11-20): verdicts of A/B round 3 (NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad)
    # MOJOLEARN_IDENTITY_FAST_NAN_CANON (core/nan_canon.mojo): promoted to the default; the old opt-in name is refused.
    comptime assert not is_defined["MOJOLEARN_IDENTITY_FAST_NAN_CANON"](), "promoted 2026-11-20 (lane/fx-verdicts5, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm, a cross-vendor identity fix, cost: gemm 0.917 inside a 19 % base spread, transformer-decode 1.009): it is the default; drop the define, or build the old form with -D MOJOLEARN_IDENTITY_FAST_NAN_CANON_OFF"
    # ---- end lane fx-verdicts5
    # ---- lane fx-verdicts6 (2026-11-20): verdicts of A/B abf2 (NVIDIA L40S host 79f003b8e8ae, wheel e41b5148b24d, job nvc-ab-20261120T224801Z-458a59)
    # TOMBSTONE: MOJOLEARN_MODE_FAST_BF16_UNITS (ROW WITHOUT A ROUTE, A/B abf2 NVIDIA L40S, wheel e41b5148b24d, job nvc-ab-20261120T224801Z-458a59, one run per arm: its one route MOJOLEARN_GEMM_FAST_BF16_UNITS was dominated by the TF32 route, mlp-clf/taxi infer 1.86) deleted 2026-11-20 by lane/fx-verdicts6; code recoverable at 06d1e8ff34.
    # Restore: git apply experiments/removed/MOJOLEARN_GEMM_FAST_BF16_UNITS.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_MODE_FAST_BF16_UNITS"](), "removed 2026-11-20 (lane/fx-verdicts6, A/B abf2 NVIDIA L40S, wheel e41b5148b24d, job nvc-ab-20261120T224801Z-458a59, one run per arm; ROW WITHOUT A ROUTE: its one route MOJOLEARN_GEMM_FAST_BF16_UNITS was dominated by the TF32 route, mlp-clf/taxi infer 1.86): mode-policy row 7 (bf16 matrix units, mask bit 128) of checks/mode_policy.mojo; the table has seven rows; see docs/TOMBSTONES.md"
    # TOMBSTONE: MOJOLEARN_GEMM_FAST_LEAF512 (NOISE, A/B abf2 NVIDIA L40S, wheel e41b5148b24d, job nvc-ab-20261120T224801Z-458a59, one run per arm: lm-forward 0.96, lm-train-step 0.99, mlp-train-step 1.08, median 0.99; round 3 on wheel 28a780604c 0.990, 1.024, 1.12) deleted 2026-11-20 by lane/fx-verdicts6; code recoverable at 06d1e8ff34.
    # Restore: git apply experiments/removed/MOJOLEARN_GEMM_FAST_LEAF512.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_GEMM_FAST_LEAF512"](), "removed 2026-11-20 (lane/fx-verdicts6, A/B abf2 NVIDIA L40S, wheel e41b5148b24d, job nvc-ab-20261120T224801Z-458a59, one run per arm; NOISE: lm-forward 0.96, lm-train-step 0.99, mlp-train-step 1.08, median 0.99; round 3 on wheel 28a780604c 0.990, 1.024, 1.12): the contract's shortest leaf 512 (profile v2-leaf512, version 4); the leaf is 128 on every column; see docs/TOMBSTONES.md"
    # ---- end lane fx-verdicts6
    # ---- lane ab-outcome-1 (2026-11-24): A/B nvc-ab-20261124T070727Z-69dac2 decisions (NVIDIA L40S only)
    # TOMBSTONE: MOJOLEARN_IDN_GEMM_FILL_TILES (NOISE/LOSER: NV L40S 3 cells, median 0.99, 0.99..1.12, lm-train-step 35.11 -> 39.19 ms) deleted 2026-11-24 by lane/ab-outcome-1; code recoverable at d81b6ba708.
    # Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_IDN_GEMM_FILL_TILES.ab-outcome-1.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GEMM_FILL_TILES"](), "removed 2026-11-24 (lane/ab-outcome-1, A/B nvc-ab-20261124T070727Z-69dac2 NVIDIA L40S, wheel 4519097d380a, one run per arm): IDN_GEMM_FILL_TILES (continuous fill-wave tile step-down) was NOISE/LOSER: 3 cells, median 0.99, lm-train-step 35.11 -> 39.19 ms; the two-rule step-down default is kept; code at d81b6ba708; restore: ~/mojolearn-evidence/experiments-removed/MOJOLEARN_IDN_GEMM_FILL_TILES.ab-outcome-1.patch"
    # TOMBSTONE: MOJOLEARN_IDN_NV_MUL_RN_ASM_OLD (NOISE: NV L40S 30 cells, median 1.01, 0.83..1.11) deleted 2026-11-24 by lane/ab-outcome-1; code recoverable at d81b6ba708.
    # Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_IDN_NV_MUL_RN_ASM_OLD.ab-outcome-1.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NV_MUL_RN_ASM_OLD"](), "removed 2026-11-24 (lane/ab-outcome-1, A/B nvc-ab-20261124T070727Z-69dac2 NVIDIA L40S, wheel 4519097d380a, one run per arm): IDN_NV_MUL_RN_ASM_OLD (the inline-asm mul.rn.f32 / mul.rn.f64 / mul.rn.ftz.f32 x 1 old path) was NOISE: 30 cells, median 1.01, 0.83..1.11; the schedulable fma.rn(a, b, -0.0) default is kept; code at d81b6ba708; restore: ~/mojolearn-evidence/experiments-removed/MOJOLEARN_IDN_NV_MUL_RN_ASM_OLD.ab-outcome-1.patch"
    # ---- end lane ab-outcome-1
    # ---- lane ab-outcome-2 (2026-11-24): A/B nvc-ab-20261124T095727Z-92fd9f decisions (NVIDIA L40S only, job nvc1-0002; sub-lane ab2-classic)
    # TOMBSTONE: MOJOLEARN_IDN_OLS_PREDICT_TILED (NOISE/LOSER: 4 cells median 0.98, 0.93..1.44, ols istella fit 509 -> 733 ms) deleted 2026-11-24 by lane/ab-outcome-2; code recoverable at ba83d4655f.
    # Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_IDN_OLS_PREDICT_TILED.ab-outcome-2.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_IDN_OLS_PREDICT_TILED"](), "removed 2026-11-24 (lane/ab-outcome-2, A/B nvc-ab-20261124T095727Z-92fd9f NVIDIA L40S, wheel 4519097d380a, job nvc1-0002, one run per arm): the tiled gemv for OLS predict was not a winner: 4 cells, median 0.98, 0.93..1.44 (ols infer istella 20.51 -> 19.77 ms, taxi 1.53 -> 1.42 ms, noise; ols istella fit 509 -> 733 ms); the plain pinned_gemv_n_kernel is IDENTICAL OLS predict's only form; code at ba83d4655f; restore: ~/mojolearn-evidence/experiments-removed/MOJOLEARN_IDN_OLS_PREDICT_TILED.ab-outcome-2.patch"
    # TOMBSTONE: MOJOLEARN_MCD_FAST_LOCKSTEP_OFF (old path, LOSER: 6 cells median 1.43, the serial candidate loop 1.42..1.46 slower) deleted 2026-11-24 by lane/ab-outcome-2; code recoverable at ba83d4655f.
    # Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_MCD_FAST_LOCKSTEP_OFF.ab-outcome-2.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_MCD_FAST_LOCKSTEP_OFF"](), "removed 2026-11-24 (lane/ab-outcome-2, A/B nvc-ab-20261124T095727Z-92fd9f NVIDIA L40S, wheel 4519097d380a, job nvc1-0002, one run per arm): the serial MCD candidate loop as the NVIDIA / AMD IDENTICAL path was slower: 6 cells, median 1.43, min-cov-det istella fit 84981 -> 123749 ms (1.46), taxi 4110 -> 5897 (1.43), elliptic-envelope istella 1.45, taxi 1.42; the lockstep (every width) is the only NVIDIA / AMD IDENTICAL form, the serial loop stays for Apple, fast and select_init_best; code at ba83d4655f; restore: ~/mojolearn-evidence/experiments-removed/MOJOLEARN_MCD_FAST_LOCKSTEP_OFF.ab-outcome-2.patch"
    # ---- end lane ab-outcome-2
    return True

comptime SIX_LANE_CONFIGURATION_OK = _check_configuration()
