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
    # TOMBSTONE: MOJOLEARN_IDN_MLP_GEMM_LEAF128_OFF (its MLP step was deleted, plan row X2) deleted 2026-11-09 by lane/x2-small-mlp-on-executor; code recoverable at f59d913b0b. Refused in core/experiment_guards_neural.mojo.
    # TOMBSTONE restored 2026-11-09 by lane/r1-restore-measured-winners: MOJOLEARN_IDN_MLP_GEMM_LEAF128_OFF is live again (autograd/small_mlp_op.mojo); core/experiment_guards_neural.mojo no longer refuses it.
    # Lane i17-decide-nbr-lin-prep (2026-10-11), removed: A/B on main b2657cc89, one run per arm, NV L40S nv2 / AMD MI325X (docs/TOMBSTONES.md,
    # docs/apple-fast/EXPERIMENTS.md, experiments/removed/*.patch).
    comptime assert not is_defined["MOJOLEARN_IDN_GEMM_NV_MICROTILE"](), "removed 2026-10-11 (lane/i17-decide-nbr-lin-prep, c38d7bc89): IDN_GEMM_NV_MICROTILE (NVIDIA packed GEMM at K window 32) was SLOWER: neural:gemm gaussian NV 48.5 -> 55.9 ms (1.15x, v0050); code at c38d7bc89^, patch experiments/removed/MOJOLEARN_IDN_GEMM_NV_MICROTILE.patch"
    return True

comptime SIX_LANE_CONFIGURATION_OK = _check_configuration()
