# SPDX-License-Identifier: Apache-2.0
"""Opt-in strategy admission, evaluated through GLOBAL_NUMERIC_MODE.

A standalone unused comptime binding is lazy and cannot enforce a guard.
This predicate is required by the numeric-mode constant used in every binding.
"""
from std.sys.compile import is_defined, get_defined_int


def _check_configuration() -> Bool:
    comptime assert not (is_defined["MOJOLEARN_NN14_BOUNDED_IM2COL"]() and is_defined["MOJOLEARN_NI12_IMPLICIT_CONV"]()), "incompatible integrated strategies: MOJOLEARN_NN14_BOUNDED_IM2COL / MOJOLEARN_NI12_IMPLICIT_CONV"
    comptime assert not (is_defined["MOJOLEARN_IDN_NEURAL_NN12"]() and is_defined["MOJOLEARN_NI01_TRAINING_WORKSPACE"]()), "incompatible integrated strategies: MOJOLEARN_IDN_NEURAL_NN12 / MOJOLEARN_NI01_TRAINING_WORKSPACE"
    comptime assert not (is_defined["MOJOLEARN_NN48_CSR_TILES"]() and is_defined["MOJOLEARN_NI55_GRAPH_FEATURE4"]()), "incompatible integrated strategies: MOJOLEARN_NN48_CSR_TILES / MOJOLEARN_NI55_GRAPH_FEATURE4"
    comptime assert not (is_defined["MOJOLEARN_NI59_DROPOUT_CHANNEL"]() and is_defined["MOJOLEARN_NI60_DROPOUT_APPLY4"]()), "incompatible integrated strategies: MOJOLEARN_NI59_DROPOUT_CHANNEL / MOJOLEARN_NI60_DROPOUT_APPLY4"
    comptime assert not (is_defined["MOJOLEARN_NN53_HEAD_CHUNK512"]() and is_defined["MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2"]()), "incompatible integrated strategies: MOJOLEARN_NN53_HEAD_CHUNK512 / MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2"
    comptime assert not (is_defined["MOJOLEARN_FOREST_ORDERED_RESIDENT_OFF"]() and is_defined["MOJOLEARN_AFT_P02"]()), "incompatible integrated strategies: MOJOLEARN_FOREST_ORDERED_RESIDENT_OFF / MOJOLEARN_AFT_P02"
    comptime assert not (is_defined["MOJOLEARN_AFT_P07"]() and is_defined["MOJOLEARN_SHAP_FAST_ROW_PAIR"]()), "incompatible integrated strategies: MOJOLEARN_AFT_P07 / MOJOLEARN_SHAP_FAST_ROW_PAIR"
    comptime assert not (is_defined["MOJOLEARN_AFCL_G01"]() and is_defined["MOJOLEARN_KNN_FAST_MMA_OFF"]()), "incompatible integrated strategies: MOJOLEARN_AFCL_G01 / MOJOLEARN_KNN_FAST_MMA_OFF"
    comptime assert not (is_defined["MOJOLEARN_NN53_HEAD_CHUNK512"]() and is_defined["MOJOLEARN_NN53_HEAD_CHUNK2048"]()), "incompatible integrated strategies: MOJOLEARN_NN53_HEAD_CHUNK512 / MOJOLEARN_NN53_HEAD_CHUNK2048"
    comptime assert not (is_defined["MOJOLEARN_AFN26_MAMBA1_CHUNKS16"]() and is_defined["MOJOLEARN_AFN26_MAMBA1_CHUNKS64"]()), "incompatible integrated strategies: MOJOLEARN_AFN26_MAMBA1_CHUNKS16 / MOJOLEARN_AFN26_MAMBA1_CHUNKS64"
    comptime assert not (is_defined["MOJOLEARN_AFN26_MAMBA3_THREADS64"]() and is_defined["MOJOLEARN_AFN26_MAMBA3_THREADS256"]()), "incompatible integrated strategies: MOJOLEARN_AFN26_MAMBA3_THREADS64 / MOJOLEARN_AFN26_MAMBA3_THREADS256"
    comptime assert not (is_defined["MOJOLEARN_AFN26_EMB_THREADS64"]() and is_defined["MOJOLEARN_AFN26_EMB_THREADS128"]()), "incompatible integrated strategies: MOJOLEARN_AFN26_EMB_THREADS64 / MOJOLEARN_AFN26_EMB_THREADS128"
    comptime assert not (is_defined["MOJOLEARN_AFN26_ATTN_NORM_TPB128"]() and is_defined["MOJOLEARN_AFN26_ATTN_NORM_TPB512"]()), "incompatible integrated strategies: MOJOLEARN_AFN26_ATTN_NORM_TPB128 / MOJOLEARN_AFN26_ATTN_NORM_TPB512"
    comptime assert not (is_defined["MOJOLEARN_IDN_SAMBA_RESIDENT_STEP"]() and is_defined["MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2"]()), "incompatible integrated strategies: MOJOLEARN_IDN_SAMBA_RESIDENT_STEP / MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2"
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
    comptime assert get_defined_int["MOJOLEARN_CLASSICAL_ENETCV_SCORE_BLOCKS",0]() == 0 or get_defined_int["MOJOLEARN_CLASSICAL_ENETCV_SCORE_BLOCKS",0]() == 1024 or get_defined_int["MOJOLEARN_CLASSICAL_ENETCV_SCORE_BLOCKS",0]() == 4096, "invalid MOJOLEARN_CLASSICAL_ENETCV_SCORE_BLOCKS (1024|4096)"
    # Lane grid-fixups-1 (2026-10-08): C17_LS_TRIALS (2 trials) and the I12 loser IDN_QN_EXACT_TRIALS (4 trials) pick the
    # trial count of one line search (glm/impl/qn/qn_linesearch.mojo:236); together the first silently overrides the second.
    comptime assert not (is_defined["MOJOLEARN_CLASSICAL_C17_LS_TRIALS"]() and is_defined["MOJOLEARN_IDN_QN_EXACT_TRIALS"]()), "qn exact trials take one count: MOJOLEARN_CLASSICAL_C17_LS_TRIALS (2) or MOJOLEARN_IDN_QN_EXACT_TRIALS (4), not both"
    comptime assert not is_defined["MOJOLEARN_C58_TEAM64"](), "retired: use -D MOJOLEARN_C58_TEAM_MIB=64|256"
    comptime assert get_defined_int["MOJOLEARN_C58_TEAM_MIB",256]() == 64 or get_defined_int["MOJOLEARN_C58_TEAM_MIB",256]() == 256, "invalid MOJOLEARN_C58_TEAM_MIB (64|256)"
    # Deleted 2026-10-07 (lane serial-cleanup): forbidden serial shape; incumbent route is parallel. Recoverable at origin/integration/switches-20261007 608a7cf4a.
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C20_ROW_CACHE"](), "removed: C20_ROW_CACHE was a host loop of 3 launches per working-set row (one 1x1); forbidden serial shape, incumbent square tile is parallel"
    comptime assert not is_defined["MOJOLEARN_C60_DIFF_REUSE"](), "removed: C60_DIFF_REUSE was dead code (select_d never reaches d == 2)"
    # Deleted 2026-10-08 (lane gap-graph, plan 5.3): the batched CC rounds + device relabel are the only path on every vendor; the C33 CC chunking (never compiled) went with it. Recoverable at origin/main 02f7aed95.
    comptime assert not (is_defined["MOJOLEARN_CC_FAST_OFF"]() or is_defined["MOJOLEARN_CC_FAST"]()), "removed: connected_components' batched rounds with the device relabel are the only path (lane gap-graph 2026-10-08); the per-round-wait path is gone"
    # Deleted 2026-10-07 (lane serial-cleanup): forbidden serial shape; incumbent kNN selectors are parallel. Recoverable at origin/integration/switches-20261007 608a7cf4a.
    comptime assert not (is_defined["MOJOLEARN_C29_STREAM_TOPK"]() or is_defined["MOJOLEARN_C29_TILE"]() or is_defined["MOJOLEARN_C29_TILE_128"]()), "removed: C29_STREAM_TOPK (and MOJOLEARN_C29_TILE) was one thread per query walking every index row; forbidden serial shape, incumbent kNN top-k is parallel"
    comptime assert get_defined_int["MOJOLEARN_IDN_NEURAL_STAGE_DEPTH",2]() == 1 or get_defined_int["MOJOLEARN_IDN_NEURAL_STAGE_DEPTH",2]() == 2, "invalid neural STAGE_DEPTH configuration"
    comptime assert get_defined_int["MOJOLEARN_IDN_NEURAL_STAGE_PAD",0]() == 0 or get_defined_int["MOJOLEARN_IDN_NEURAL_STAGE_PAD",0]() == 1, "invalid neural STAGE_PAD configuration"
    comptime assert get_defined_int["MOJOLEARN_IDN_NEURAL_FILL_BLOCKS",1]() > 0, "positive neural resource budget required"
    comptime assert get_defined_int["MOJOLEARN_IDN_NEURAL_STREAM_GROUP",8]() > 0, "positive neural resource budget required"
    comptime assert get_defined_int["MOJOLEARN_IDN_NEURAL_RETAINED_FLOATS",16777216]() > 0, "positive neural resource budget required"
    # Lane neural-gemm-attn-dedupe (2026-10-07): each merged idea is ONE define
    # with named arms, so its old mutual exclusions hold by construction.
    # Retired defines are refused so a stale catalog arm cannot build as a
    # silent incumbent.
    comptime S = get_defined_int["MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE",0]()
    comptime assert S == 0 or S == 1 or S == 2 or S == 3 or S == 4 or S == 9 or S == 10 or S == 11 or S == 15 or S == 24, "MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE arms: 1 geometry, 2 stream, 3 stream_all, 4 stream_exact, 9 pages, 10 cost, 11 fold_exact, 15 threadmap, 24 pages_threadmap (8 async deleted by lane/grid-prune)"
    comptime R = get_defined_int["MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE_ROLES",7]()
    comptime assert R >= 1 and R <= 7, "MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE_ROLES is a mask: 1 projection, 2 head, 4 weight-grad"
    comptime assert S != 3 or R == 7, "schedule 3 (stream_all) is a global GEMM arm; it takes no role mask"
    comptime assert S != 10 or is_defined["MOJOLEARN_IDN_NEURAL_FILL_BLOCKS"](), "schedule 10 (cost) needs an explicit hardware fill budget"
    comptime assert S != 1 or is_defined["MOJOLEARN_GEMM_ARM_TRIAL"](), "schedule 1 (geometry) needs MOJOLEARN_GEMM_ARM_TRIAL"
    comptime LEAF = get_defined_int["MOJOLEARN_IDN_GEMM_LEAF",0]()
    comptime assert LEAF == 0 or LEAF == 1 or LEAF == 2 or LEAF == 3, "MOJOLEARN_IDN_GEMM_LEAF arms: 1 neural128, 2 neural256, 3 all256"
    comptime CH = get_defined_int["MOJOLEARN_IDN_NEURAL_CHAINS",1]()
    comptime assert CH == 1 or CH == 2 or CH == 4, "invalid neural CHAINS configuration"
    comptime assert not (S == 1 and (LEAF == 1 or LEAF == 2 or is_defined["MOJOLEARN_IDN_NEURAL_CHAINS"]())), "schedule 1 (geometry) runs incumbent geometries; it excludes the neural leaf/chains profile"
    comptime assert not (is_defined["MOJOLEARN_IDN_CE_DENOM_ROWFOLD"]() and (LEAF == 1 or LEAF == 2 or is_defined["MOJOLEARN_IDN_NEURAL_CHAINS"]())), "MOJOLEARN_IDN_CE_DENOM_ROWFOLD reproduces the v1 GEMM contract chain and fold; it excludes the neural leaf/chains profile"
    comptime EPI = get_defined_int["MOJOLEARN_IDN_NEURAL_GEMM_EPILOGUE",0]()
    comptime assert EPI >= 0 and EPI <= 3, "MOJOLEARN_IDN_NEURAL_GEMM_EPILOGUE is a mask: 1 mlp, 2 cnn"
    comptime HS = get_defined_int["MOJOLEARN_IDN_ATTN_HEAD_SHARE",1]()
    comptime assert HS == 1 or HS == 4, "MOJOLEARN_IDN_ATTN_HEAD_SHARE legal set {4} (two-head I06/NI19 deleted as a loser)"
    comptime SM = get_defined_int["MOJOLEARN_IDN_ATTN_SOFTMAX",0]()
    comptime assert SM >= 0 and SM <= 2, "MOJOLEARN_IDN_ATTN_SOFTMAX arms: 1 summary_tree, 2 online_tile32"
    # lane/attention-tiled-v2 (2026-10-07): the online_tile32 forward's query rows per block.
    comptime TQ2 = get_defined_int["MOJOLEARN_IDN_ATTN_V2_TQ",32]()
    comptime assert TQ2 == 32 or TQ2 == 64, "MOJOLEARN_IDN_ATTN_V2_TQ legal set 32|64 (256 threads, 8 or 4 lanes per query row)"
    comptime assert not is_defined["MOJOLEARN_IDN_ATTN_V2_TQ"]() or SM == 2, "MOJOLEARN_IDN_ATTN_V2_TQ is only read by the online_tile32 arm (MOJOLEARN_IDN_ATTN_SOFTMAX=2)"
    comptime ST = get_defined_int["MOJOLEARN_IDN_ATTN_STASH",0]()
    comptime assert ST >= 0 and ST <= 3, "MOJOLEARN_IDN_ATTN_STASH arms: 1 recompute, 2 packed, 3 alias_y"
    comptime NO = get_defined_int["MOJOLEARN_IDN_NORM",0]()
    comptime assert NO >= 0 and NO <= 4, "MOJOLEARN_IDN_NORM arms: 1 lanes8, 2 split_scale, 3 row_block, 4 lanes8_split_scale"
    comptime RO = get_defined_int["MOJOLEARN_IDN_ROPE",0]()
    comptime assert RO >= 0 and RO <= 2, "MOJOLEARN_IDN_ROPE arms: 1 qk_pair, 2 k_cache"
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_NN01"](), "MOJOLEARN_IDN_NEURAL_NN01 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_NN02"](), "MOJOLEARN_IDN_NEURAL_NN02 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_NN03"](), "MOJOLEARN_IDN_NEURAL_NN03 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_NN04"](), "MOJOLEARN_IDN_NEURAL_NN04 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_NN08"](), "MOJOLEARN_IDN_NEURAL_NN08 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_NN09"](), "MOJOLEARN_IDN_NEURAL_NN09 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_NN10"](), "MOJOLEARN_IDN_NEURAL_NN10 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_NN11"](), "MOJOLEARN_IDN_NEURAL_NN11 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_NN15"](), "MOJOLEARN_IDN_NEURAL_NN15 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_LEAF"](), "MOJOLEARN_IDN_NEURAL_LEAF is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_NN06"](), "MOJOLEARN_IDN_NEURAL_NN06 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_NI02_GEMM_STREAM_PARTIALS"](), "MOJOLEARN_NI02_GEMM_STREAM_PARTIALS is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_NI08_GEMM_LEAF_256"](), "MOJOLEARN_NI08_GEMM_LEAF_256 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_GEMM_FOLD_LEAF_64"](), "MOJOLEARN_IDN_GEMM_FOLD_LEAF_64 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_NI09_TILED_BIAS"](), "MOJOLEARN_NI09_TILED_BIAS is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_NN17_GQA_FOUR_HEADS"](), "MOJOLEARN_NN17_GQA_FOUR_HEADS is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_ATTN_GQA_HEAD_REUSE"](), "MOJOLEARN_IDN_ATTN_GQA_HEAD_REUSE is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_NN20_BALANCED_SUMMARY_TREE"](), "MOJOLEARN_NN20_BALANCED_SUMMARY_TREE is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_ATTENTION_V2"](), "MOJOLEARN_IDN_ATTENTION_V2 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_NN22_EAGER_DKDV_PAIR"](), "MOJOLEARN_NN22_EAGER_DKDV_PAIR is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_NN23_ROWDOT_DS"](), "MOJOLEARN_NN23_ROWDOT_DS is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_NN24_NORM_LANES8"](), "MOJOLEARN_NN24_NORM_LANES8 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_NN25_RMS_SPLIT_SCALE"](), "MOJOLEARN_NN25_RMS_SPLIT_SCALE is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_RMS_ROW_BLOCK"](), "MOJOLEARN_IDN_RMS_ROW_BLOCK is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_NN26_TRAIN_SWIGLU"](), "MOJOLEARN_NN26_TRAIN_SWIGLU is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_NN27_QK_ROPE_PAIR"](), "MOJOLEARN_NN27_QK_ROPE_PAIR is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_ROPE_CACHE"](), "MOJOLEARN_IDN_ROPE_CACHE is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_NN28_DEAD_TRAINING_CACHE"](), "MOJOLEARN_NN28_DEAD_TRAINING_CACHE is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_ATTN_V1_RECOMPUTE_BACKWARD"](), "MOJOLEARN_ATTN_V1_RECOMPUTE_BACKWARD is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_ATTN_V1_PACKED_ESTASH"](), "MOJOLEARN_ATTN_V1_PACKED_ESTASH is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_ATTN_V1_ALIAS_Y_ESTASH"](), "MOJOLEARN_ATTN_V1_ALIAS_Y_ESTASH is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    # L11 (2026-10-07) neural sequence/training dedupe: retired defines refuse
    # (a stale build line must not silently run the B arm), and every merged
    # switch admits only its documented arms. See grid_controls/neural-seq-train-dedupe.json.
    comptime assert not is_defined["MOJOLEARN_NN38_CACHE_SUFFIX_SEEDS"](), "retired define MOJOLEARN_NN38_CACHE_SUFFIX_SEEDS: use IDN_M3_ANGLE_CARRY_CACHE"
    comptime assert not is_defined["MOJOLEARN_NN36_SHARED_DECAY"](), "retired define MOJOLEARN_NN36_SHARED_DECAY: use IDN_M2_YOFF_EXP=1"
    comptime assert not is_defined["MOJOLEARN_IDN_M2_YOFF_EXP_CACHE"](), "retired define MOJOLEARN_IDN_M2_YOFF_EXP_CACHE: use IDN_M2_YOFF_EXP=2"
    comptime assert not is_defined["MOJOLEARN_NN33_YDIAG_ROWS8"](), "retired define MOJOLEARN_NN33_YDIAG_ROWS8: use IDN_M2_YD_ROWS=8"
    comptime assert not is_defined["MOJOLEARN_NN33_CSTATE_P16"](), "retired define MOJOLEARN_NN33_CSTATE_P16: use IDN_M2_CS_PT=16"
    comptime assert not is_defined["MOJOLEARN_NN39_M2_GRAD_TREE"](), "retired define MOJOLEARN_NN39_M2_GRAD_TREE: use IDN_M2_GRAD_FOLD=1"
    comptime assert not is_defined["MOJOLEARN_IDN_M2_GRAD_LEAF128"](), "retired define MOJOLEARN_IDN_M2_GRAD_LEAF128: use IDN_M2_GRAD_FOLD=2"
    comptime assert not is_defined["MOJOLEARN_NN45_CONV_RELU"](), "retired define MOJOLEARN_NN45_CONV_RELU: use IDN_CNN_CONV_RELU=1"
    comptime assert not is_defined["MOJOLEARN_NI16_CONV_RELU_FUSED"](), "retired define MOJOLEARN_NI16_CONV_RELU_FUSED: use IDN_CNN_CONV_RELU=2"
    comptime assert not is_defined["MOJOLEARN_NN43_WGRAD_FIXED128"](), "retired define MOJOLEARN_NN43_WGRAD_FIXED128: use IDN_SEQ_WGRAD=2"
    comptime assert not is_defined["MOJOLEARN_IDN_SEQ_WGRAD_LEAF256"](), "retired define MOJOLEARN_IDN_SEQ_WGRAD_LEAF256: use IDN_SEQ_WGRAD=1"
    comptime assert not is_defined["MOJOLEARN_IDN_SEQ_LN_LEAF32"](), "retired define MOJOLEARN_IDN_SEQ_LN_LEAF32: use IDN_SEQ_LN_LEAF=32"
    comptime assert not is_defined["MOJOLEARN_IDN_SEQ_ROW_SERIAL_SCAN"](), "retired define MOJOLEARN_IDN_SEQ_ROW_SERIAL_SCAN: use deleted (serial per-row scan)"
    comptime assert not is_defined["MOJOLEARN_NN44_STABLE_GROUP"](), "retired define MOJOLEARN_NN44_STABLE_GROUP: use IDN_MOE_STABLE_PACK"
    comptime assert not is_defined["MOJOLEARN_NN52_CE_WEIGHT_GRAD"](), "retired define MOJOLEARN_NN52_CE_WEIGHT_GRAD: use IDN_CE_GRAD_FUSED"
    comptime assert not is_defined["MOJOLEARN_NN54_LOSS_PROFILE"](), "retired define MOJOLEARN_NN54_LOSS_PROFILE: use IDN_CE_TOKEN_FOLD=1"
    comptime assert not is_defined["MOJOLEARN_IDN_LOSS_TOKEN_TREE_V2"](), "retired define MOJOLEARN_IDN_LOSS_TOKEN_TREE_V2: use IDN_CE_TOKEN_FOLD=2"
    comptime assert not is_defined["MOJOLEARN_NN51_RESIDENT_TOKEN_VALIDATION"](), "retired define MOJOLEARN_NN51_RESIDENT_TOKEN_VALIDATION: use IDN_LM_RESIDENT_TOKENS=2"
    comptime assert not is_defined["MOJOLEARN_IDN_LM_OWNED_TOKENS"](), "retired define MOJOLEARN_IDN_LM_OWNED_TOKENS: use IDN_LM_RESIDENT_TOKENS=1"
    comptime assert not is_defined["MOJOLEARN_NN60_BLOCK_VIEWS"](), "retired define MOJOLEARN_NN60_BLOCK_VIEWS: use IDN_LM_VIEWS=1"
    comptime assert not is_defined["MOJOLEARN_NN60_EMB_HEAD_VIEWS"](), "retired define MOJOLEARN_NN60_EMB_HEAD_VIEWS: use IDN_LM_VIEWS=2"
    comptime assert not is_defined["MOJOLEARN_IDN_LM_PARAM_VIEWS"](), "retired define MOJOLEARN_IDN_LM_PARAM_VIEWS: use IDN_LM_VIEWS=3"
    comptime assert not is_defined["MOJOLEARN_NN56_GROUPED_ADAM"](), "retired define MOJOLEARN_NN56_GROUPED_ADAM: use IDN_LM_GROUPED_ADAM=1"
    comptime assert not is_defined["MOJOLEARN_NN55_BLOCK_STATUS"](), "retired define MOJOLEARN_NN55_BLOCK_STATUS: use IDN_LM_GROUPED_ADAM=2"
    comptime assert not is_defined["MOJOLEARN_NN32_RETAIN_FORWARD"](), "retired define MOJOLEARN_NN32_RETAIN_FORWARD: use IDN_ACT_RETAIN=1"
    comptime assert not is_defined["MOJOLEARN_NN31_BOUNDED_CHECKPOINTS"](), "retired define MOJOLEARN_NN31_BOUNDED_CHECKPOINTS: use IDN_ACT_RETAIN=2"
    comptime assert not is_defined["MOJOLEARN_IDN_SAMBA_FORWARD_TAPE"](), "retired define MOJOLEARN_IDN_SAMBA_FORWARD_TAPE: use IDN_ACT_RETAIN=3"
    comptime assert not is_defined["MOJOLEARN_IDN_TRAIN_BACKWARD_SCRATCH"](), "retired define MOJOLEARN_IDN_TRAIN_BACKWARD_SCRATCH: use IDN_TRAIN_SCRATCH=1"
    comptime assert not is_defined["MOJOLEARN_NN62_LIFETIME_ARENA"](), "retired define MOJOLEARN_NN62_LIFETIME_ARENA: use IDN_TRAIN_SCRATCH=2"
    comptime assert get_defined_int["MOJOLEARN_IDN_M2_YOFF_EXP",0]() == 0 or get_defined_int["MOJOLEARN_IDN_M2_YOFF_EXP",0]() == 1 or get_defined_int["MOJOLEARN_IDN_M2_YOFF_EXP",0]() == 2, "invalid MOJOLEARN_IDN_M2_YOFF_EXP arm (legal: 0|1|2)"
    comptime assert get_defined_int["MOJOLEARN_IDN_M2_YD_ROWS",4]() == 4 or get_defined_int["MOJOLEARN_IDN_M2_YD_ROWS",4]() == 8, "invalid MOJOLEARN_IDN_M2_YD_ROWS arm (legal: 4|8)"
    comptime assert get_defined_int["MOJOLEARN_IDN_M2_CS_PT",8]() == 8 or get_defined_int["MOJOLEARN_IDN_M2_CS_PT",8]() == 16, "invalid MOJOLEARN_IDN_M2_CS_PT arm (legal: 8|16)"
    comptime assert get_defined_int["MOJOLEARN_IDN_M2_GRAD_FOLD",0]() == 0 or get_defined_int["MOJOLEARN_IDN_M2_GRAD_FOLD",0]() == 1 or get_defined_int["MOJOLEARN_IDN_M2_GRAD_FOLD",0]() == 2, "invalid MOJOLEARN_IDN_M2_GRAD_FOLD arm (legal: 0|1|2)"
    comptime assert get_defined_int["MOJOLEARN_IDN_CNN_CONV_RELU",0]() == 0 or get_defined_int["MOJOLEARN_IDN_CNN_CONV_RELU",0]() == 1 or get_defined_int["MOJOLEARN_IDN_CNN_CONV_RELU",0]() == 2, "invalid MOJOLEARN_IDN_CNN_CONV_RELU arm (legal: 0|1|2)"
    comptime assert get_defined_int["MOJOLEARN_IDN_SEQ_WGRAD",0]() == 0 or get_defined_int["MOJOLEARN_IDN_SEQ_WGRAD",0]() == 1 or get_defined_int["MOJOLEARN_IDN_SEQ_WGRAD",0]() == 2, "invalid MOJOLEARN_IDN_SEQ_WGRAD arm (legal: 0|1|2)"
    comptime assert get_defined_int["MOJOLEARN_IDN_SEQ_LN_LEAF",64]() == 32 or get_defined_int["MOJOLEARN_IDN_SEQ_LN_LEAF",64]() == 64, "invalid MOJOLEARN_IDN_SEQ_LN_LEAF arm (legal: 32|64)"
    comptime assert get_defined_int["MOJOLEARN_IDN_CE_TOKEN_FOLD",0]() == 0 or get_defined_int["MOJOLEARN_IDN_CE_TOKEN_FOLD",0]() == 1 or get_defined_int["MOJOLEARN_IDN_CE_TOKEN_FOLD",0]() == 2, "invalid MOJOLEARN_IDN_CE_TOKEN_FOLD arm (legal: 0|1|2)"
    comptime assert get_defined_int["MOJOLEARN_IDN_LM_RESIDENT_TOKENS",0]() == 0 or get_defined_int["MOJOLEARN_IDN_LM_RESIDENT_TOKENS",0]() == 1 or get_defined_int["MOJOLEARN_IDN_LM_RESIDENT_TOKENS",0]() == 2, "invalid MOJOLEARN_IDN_LM_RESIDENT_TOKENS arm (legal: 0|1|2)"
    comptime assert get_defined_int["MOJOLEARN_IDN_LM_VIEWS",0]() == 0 or get_defined_int["MOJOLEARN_IDN_LM_VIEWS",0]() == 1 or get_defined_int["MOJOLEARN_IDN_LM_VIEWS",0]() == 2 or get_defined_int["MOJOLEARN_IDN_LM_VIEWS",0]() == 3, "invalid MOJOLEARN_IDN_LM_VIEWS arm (legal: 0|1|2|3)"
    comptime assert get_defined_int["MOJOLEARN_IDN_LM_GROUPED_ADAM",0]() == 0 or get_defined_int["MOJOLEARN_IDN_LM_GROUPED_ADAM",0]() == 1 or get_defined_int["MOJOLEARN_IDN_LM_GROUPED_ADAM",0]() == 2, "invalid MOJOLEARN_IDN_LM_GROUPED_ADAM arm (legal: 0|1|2)"
    comptime assert get_defined_int["MOJOLEARN_IDN_ACT_RETAIN",0]() == 0 or get_defined_int["MOJOLEARN_IDN_ACT_RETAIN",0]() == 1 or get_defined_int["MOJOLEARN_IDN_ACT_RETAIN",0]() == 2 or get_defined_int["MOJOLEARN_IDN_ACT_RETAIN",0]() == 3, "invalid MOJOLEARN_IDN_ACT_RETAIN arm (legal: 0|1|2|3)"
    comptime assert get_defined_int["MOJOLEARN_IDN_TRAIN_SCRATCH",0]() == 0 or get_defined_int["MOJOLEARN_IDN_TRAIN_SCRATCH",0]() == 1 or get_defined_int["MOJOLEARN_IDN_TRAIN_SCRATCH",0]() == 2 or get_defined_int["MOJOLEARN_IDN_TRAIN_SCRATCH",0]() == 3, "invalid MOJOLEARN_IDN_TRAIN_SCRATCH arm (legal: 0|1|2|3)"
    # Lane grid-prune (2026-10-07): deleted losers and dead arms (rows in docs/apple-fast/EXPERIMENTS.md
    # "IDENTICAL grid prune"); recoverable at main ab554bb4a. A stale build line must not run the incumbent silently.
    comptime assert not is_defined["MOJOLEARN_IDN_GEMM_TILE_SHORT_K"](), "removed: MOJOLEARN_IDN_GEMM_TILE_SHORT_K (arms equal measured pass62 losers / the NVIDIA default; inert on AMD)"
    comptime assert not is_defined["MOJOLEARN_IDN_GEMM_FS2"](), "removed: MOJOLEARN_IDN_GEMM_FS2 (OVN N02 noise vs FS4)"
    comptime assert not is_defined["MOJOLEARN_IDN_GEMM_COMPACT_LIVE_TILE"](), "removed: MOJOLEARN_IDN_GEMM_COMPACT_LIVE_TILE (OVN A05 slower on NVIDIA and AMD)"
    comptime assert not is_defined["MOJOLEARN_IDN_GEMM_GROUP_TILES_BODY"](), "removed: MOJOLEARN_IDN_GEMM_GROUP_TILES_BODY (OVN N01 noise on the L40S)"
    comptime assert not is_defined["MOJOLEARN_TREES_T09"](), "removed: MOJOLEARN_TREES_T09 (ExtraTrees bootstrap sort: unreachable at bootstrap=False, host sync in fit)"
    comptime assert not is_defined["MOJOLEARN_TREES_T27"](), "removed: MOJOLEARN_TREES_T27 (alias of MOJOLEARN_2030_FUSED_EST_MOVE; inert on board walks)"
    comptime assert not is_defined["MOJOLEARN_TREES_C47_GBDT"](), "removed: MOJOLEARN_TREES_C47_GBDT (width cap subsumed by MOJOLEARN_TREES_T17_BATCH)"
    comptime assert not is_defined["MOJOLEARN_TREES_HIST_REP_BPSM"](), "merged: use -D MOJOLEARN_TREES_HIST_REP_SM=1 (device SMs x 4 blocks); BPSM=4 alone equalled SM=64"
    comptime assert not (is_defined["MOJOLEARN_NN34_AFFINE_PREFIX"]() or is_defined["MOJOLEARN_IDN_M1_STATE_WINDOW"]() or is_defined["MOJOLEARN_IDN_M1_PERSISTENT_SCAN"]()), "merged: use -D MOJOLEARN_IDN_M1_SCAN=1 (affine_prefix, NN34) | 2 (state_window, NI38) | 3 (persistent)"
    comptime M1S = get_defined_int["MOJOLEARN_IDN_M1_SCAN",0]()
    comptime assert M1S >= 0 and M1S <= 3, "MOJOLEARN_IDN_M1_SCAN arms: 1 affine_prefix, 2 state_window, 3 persistent"
    comptime assert M1S == 3 or not (is_defined["MOJOLEARN_IDN_M1_PERSISTENT_SCAN_CH"]() or is_defined["MOJOLEARN_IDN_M1_PERSISTENT_SCAN_TOKENS"]()), "MOJOLEARN_IDN_M1_PERSISTENT_SCAN_CH/_TOKENS are read only by MOJOLEARN_IDN_M1_SCAN=3 (persistent)"
    comptime assert not (is_defined["MOJOLEARN_NI14_BOUNDED_COL2IM"]() or is_defined["MOJOLEARN_NI14_TILED_COL2IM"]()), "merged: use -D MOJOLEARN_NI14_COL2IM=1 (bounded) | 2 (tiled)"
    comptime assert get_defined_int["MOJOLEARN_NI14_COL2IM",0]() >= 0 and get_defined_int["MOJOLEARN_NI14_COL2IM",0]() <= 2, "MOJOLEARN_NI14_COL2IM arms: 1 bounded, 2 tiled"
    comptime assert not (is_defined["MOJOLEARN_TREES_C50_GB_PACKED"]() and is_defined["MOJOLEARN_IDN_GBDT_APPLY_WIDE"]()), "MOJOLEARN_TREES_C50_GB_PACKED returns before MOJOLEARN_IDN_GBDT_APPLY_WIDE (gbdt/resident_model.mojo): the pair == C50"
    # Deleted 2026-10-08 (lane grid-losers-1): IDENTICAL grid ge123e6f9 losers (docs/apple-fast/EXPERIMENTS.md). Recoverable at main ad7ed2370.
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C25_PROJECTION_REUSE"](), "removed: MOJOLEARN_CLASSICAL_C25_PROJECTION_REUSE retired 2026-10-08: noise, NV 1.00-1.04x / AMD 0.88-0.98x on nystroem and rbf-sampler (grid ge123e6f9); see EXPERIMENTS.md"
    comptime assert not is_defined["MOJOLEARN_C58_SHARED_PREP"](), "removed: MOJOLEARN_C58_SHARED_PREP retired 2026-10-08: noise, NV 0.996x / AMD 0.980x on ets (grid ge123e6f9); see EXPERIMENTS.md"
    comptime assert not (is_defined["MOJOLEARN_QN_IDN_DCONV"]() or is_defined["MOJOLEARN_QN_IDN_DCONV_POLL_2"]() or is_defined["MOJOLEARN_QN_IDN_DCONV_POLL_8"]()), "removed: MOJOLEARN_QN_IDN_DCONV (and _POLL_2/_POLL_8) retired 2026-10-08: neutral/slower, NV 0.95-1.06x / AMD 0.98-1.08x on logreg, linearsvc, linearsvr (grid ge123e6f9); see EXPERIMENTS.md"
    comptime assert not is_defined["MOJOLEARN_IDN_GMM_COV_SYM"](), "removed: MOJOLEARN_IDN_GMM_COV_SYM retired 2026-10-08: slower, NV 1.84x / AMD 1.45x on gmm taxi, and mean log-likelihood -0.78% (grid ge123e6f9); see EXPERIMENTS.md"
    comptime assert not is_defined["MOJOLEARN_TREES_T19"](), "removed: MOJOLEARN_TREES_T19 retired 2026-10-08: quality loss, istella AUC -0.43% and logloss +20.7% despite NV 0.47x / AMD 0.69x on depthwise taxi (grid ge123e6f9); see EXPERIMENTS.md"
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_LINEAR_GRAM_SOLVE"](), "promoted 2026-10-08 (grid ge123e6f9, lane/grid-flips-1): OLS/Ridge resident Gram is the IDENTICAL default; drop -D MOJOLEARN_CLASSICAL_LINEAR_GRAM_SOLVE, use -D MOJOLEARN_CLASSICAL_LINEAR_GRAM_SOLVE_OFF for the old path"
    comptime assert not is_defined["MOJOLEARN_TREES_T22"](), "promoted 2026-10-08 (grid ge123e6f9, lane/grid-flips-1): ordered document-keyed storage is the IDENTICAL default; drop -D MOJOLEARN_TREES_T22, use -D MOJOLEARN_TREES_T22_OFF for the old path"
    comptime assert not is_defined["MOJOLEARN_TREES_ORD_STD_GRIDFOLD"](), "promoted 2026-10-08 (grid ge123e6f9, lane/grid-flips-1): ordered grid fold is the IDENTICAL default; drop -D MOJOLEARN_TREES_ORD_STD_GRIDFOLD, use -D MOJOLEARN_TREES_ORD_STD_GRIDFOLD_OFF for the old path"
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C24_ROWS2048"](), "promoted 2026-10-08 (grid ge123e6f9, lane/grid-flips-1): TSQR 2048-row leaves is the IDENTICAL default; drop -D MOJOLEARN_CLASSICAL_C24_ROWS2048, use -D MOJOLEARN_CLASSICAL_C24_ROWS2048_OFF for the old path"
    comptime assert not is_defined["MOJOLEARN_IDN_KMEANS_CENTROID_FOLD"](), "promoted 2026-10-08 (grid ge123e6f9, lane/grid-flips-1): two-level centroid fold is the IDENTICAL default; drop -D MOJOLEARN_IDN_KMEANS_CENTROID_FOLD, use -D MOJOLEARN_IDN_KMEANS_CENTROID_FOLD_OFF for the old path"
    comptime assert not is_defined["MOJOLEARN_IVF_DIRECT_DISTANCE"](), "promoted 2026-10-08 (grid ge123e6f9, lane/grid-flips-1): IVF direct distance is the IDENTICAL default; drop -D MOJOLEARN_IVF_DIRECT_DISTANCE, use -D MOJOLEARN_IVF_DIRECT_DISTANCE_OFF for the old path"
    # Lane grid-act-3 (2026-10-08): IDENTICAL grid ge123e6f9 promotions and losers (docs/apple-fast/EXPERIMENTS.md). Deleted code recoverable at main bc10b8b56.
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C08_UNIQUE_SCAN"](), "promoted 2026-10-08 (grid ge123e6f9, lane/grid-act-3): the unique_cols run scan is the IDENTICAL default (onehot/ordinal/target-encoder 0.525x combined, same bits); drop -D MOJOLEARN_CLASSICAL_C08_UNIQUE_SCAN, use -D MOJOLEARN_CLASSICAL_C08_UNIQUE_SCAN_OFF for the old path"
    comptime assert not is_defined["MOJOLEARN_C58_FORECAST4"](), "removed: MOJOLEARN_C58_FORECAST4 retired 2026-10-08: slower, theta/ets family NV 2.53-3.15x / AMD 1.81-3.08x on istella and taxi (2.72x combined), quality SAME (grid ge123e6f9); see EXPERIMENTS.md"
    comptime TMB = get_defined_int["MOJOLEARN_IDN_GEMM_TILE_MIN_BLOCKS",512]()
    comptime assert TMB == 192 or TMB == 512 or TMB == 1024, "MOJOLEARN_IDN_GEMM_TILE_MIN_BLOCKS legal set {192, 512, 1024}"
    return True

comptime SIX_LANE_CONFIGURATION_OK = _check_configuration()
