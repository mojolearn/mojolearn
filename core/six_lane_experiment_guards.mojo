# SPDX-License-Identifier: Apache-2.0
"""Opt-in strategy admission, evaluated through GLOBAL_NUMERIC_MODE.

A standalone unused comptime binding is lazy and cannot enforce a guard.
This predicate is required by the numeric-mode constant used in every binding.
"""
from std.sys.compile import is_defined, get_defined_int


def _check_configuration() -> Bool:
    comptime assert not (is_defined["MOJOLEARN_NN36_SHARED_DECAY"]() and is_defined["MOJOLEARN_IDN_M2_YOFF_EXP_CACHE"]()), "incompatible integrated strategies: MOJOLEARN_NN36_SHARED_DECAY / MOJOLEARN_IDN_M2_YOFF_EXP_CACHE"
    comptime assert not (is_defined["MOJOLEARN_NN43_WGRAD_FIXED128"]() and is_defined["MOJOLEARN_IDN_SEQ_WGRAD_LEAF256"]()), "incompatible integrated strategies: MOJOLEARN_NN43_WGRAD_FIXED128 / MOJOLEARN_IDN_SEQ_WGRAD_LEAF256"
    comptime assert not (is_defined["MOJOLEARN_NN54_LOSS_PROFILE"]() and is_defined["MOJOLEARN_IDN_LOSS_TOKEN_TREE_V2"]()), "incompatible integrated strategies: MOJOLEARN_NN54_LOSS_PROFILE / MOJOLEARN_IDN_LOSS_TOKEN_TREE_V2"
    comptime assert not (is_defined["MOJOLEARN_NN14_BOUNDED_IM2COL"]() and is_defined["MOJOLEARN_NI12_IMPLICIT_CONV"]()), "incompatible integrated strategies: MOJOLEARN_NN14_BOUNDED_IM2COL / MOJOLEARN_NI12_IMPLICIT_CONV"
    comptime assert not (is_defined["MOJOLEARN_IDN_NEURAL_NN12"]() and is_defined["MOJOLEARN_NI01_TRAINING_WORKSPACE"]()), "incompatible integrated strategies: MOJOLEARN_IDN_NEURAL_NN12 / MOJOLEARN_NI01_TRAINING_WORKSPACE"
    comptime assert not (is_defined["MOJOLEARN_NN48_CSR_TILES"]() and is_defined["MOJOLEARN_NI55_GRAPH_FEATURE4"]()), "incompatible integrated strategies: MOJOLEARN_NN48_CSR_TILES / MOJOLEARN_NI55_GRAPH_FEATURE4"
    comptime assert not (is_defined["MOJOLEARN_NI59_DROPOUT_CHANNEL"]() and is_defined["MOJOLEARN_NI60_DROPOUT_APPLY4"]()), "incompatible integrated strategies: MOJOLEARN_NI59_DROPOUT_CHANNEL / MOJOLEARN_NI60_DROPOUT_APPLY4"
    comptime assert not (is_defined["MOJOLEARN_NN34_AFFINE_PREFIX"]() and is_defined["MOJOLEARN_IDN_M1_STATE_WINDOW"]()), "incompatible integrated strategies: MOJOLEARN_NN34_AFFINE_PREFIX / MOJOLEARN_IDN_M1_STATE_WINDOW"
    comptime assert not (is_defined["MOJOLEARN_NN39_M2_GRAD_TREE"]() and is_defined["MOJOLEARN_IDN_M2_GRAD_LEAF128"]()), "incompatible integrated strategies: MOJOLEARN_NN39_M2_GRAD_TREE / MOJOLEARN_IDN_M2_GRAD_LEAF128"
    comptime assert not (is_defined["MOJOLEARN_NN53_HEAD_CHUNK512"]() and is_defined["MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2"]()), "incompatible integrated strategies: MOJOLEARN_NN53_HEAD_CHUNK512 / MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2"
    comptime assert not (is_defined["MOJOLEARN_FOREST_ORDERED_RESIDENT_OFF"]() and is_defined["MOJOLEARN_AFT_P02"]()), "incompatible integrated strategies: MOJOLEARN_FOREST_ORDERED_RESIDENT_OFF / MOJOLEARN_AFT_P02"
    comptime assert not (is_defined["MOJOLEARN_AFT_P07"]() and is_defined["MOJOLEARN_SHAP_FAST_ROW_PAIR"]()), "incompatible integrated strategies: MOJOLEARN_AFT_P07 / MOJOLEARN_SHAP_FAST_ROW_PAIR"
    comptime assert not (is_defined["MOJOLEARN_AFCL_G01"]() and is_defined["MOJOLEARN_KNN_FAST_MMA_OFF"]()), "incompatible integrated strategies: MOJOLEARN_AFCL_G01 / MOJOLEARN_KNN_FAST_MMA_OFF"
    comptime assert not (is_defined["MOJOLEARN_NN53_HEAD_CHUNK512"]() and is_defined["MOJOLEARN_NN53_HEAD_CHUNK2048"]()), "incompatible integrated strategies: MOJOLEARN_NN53_HEAD_CHUNK512 / MOJOLEARN_NN53_HEAD_CHUNK2048"
    comptime assert not (is_defined["MOJOLEARN_AFN26_MAMBA1_CHUNKS16"]() and is_defined["MOJOLEARN_AFN26_MAMBA1_CHUNKS64"]()), "incompatible integrated strategies: MOJOLEARN_AFN26_MAMBA1_CHUNKS16 / MOJOLEARN_AFN26_MAMBA1_CHUNKS64"
    comptime assert not (is_defined["MOJOLEARN_AFN26_MAMBA3_THREADS64"]() and is_defined["MOJOLEARN_AFN26_MAMBA3_THREADS256"]()), "incompatible integrated strategies: MOJOLEARN_AFN26_MAMBA3_THREADS64 / MOJOLEARN_AFN26_MAMBA3_THREADS256"
    comptime assert not (is_defined["MOJOLEARN_AFN26_EMB_THREADS64"]() and is_defined["MOJOLEARN_AFN26_EMB_THREADS128"]()), "incompatible integrated strategies: MOJOLEARN_AFN26_EMB_THREADS64 / MOJOLEARN_AFN26_EMB_THREADS128"
    comptime assert not (is_defined["MOJOLEARN_AFN26_ATTN_NORM_TPB128"]() and is_defined["MOJOLEARN_AFN26_ATTN_NORM_TPB512"]()), "incompatible integrated strategies: MOJOLEARN_AFN26_ATTN_NORM_TPB128 / MOJOLEARN_AFN26_ATTN_NORM_TPB512"
    comptime assert not (is_defined["MOJOLEARN_C52_PAIR_128"]() and is_defined["MOJOLEARN_C52_PAIR_512"]()), "incompatible integrated strategies: MOJOLEARN_C52_PAIR_128 / MOJOLEARN_C52_PAIR_512"
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
    comptime assert S == 0 or S == 1 or S == 2 or S == 3 or S == 4 or S == 8 or S == 9 or S == 10 or S == 11 or S == 15 or S == 24, "MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE arms: 1 geometry, 2 stream, 3 stream_all, 4 stream_exact, 8 async, 9 pages, 10 cost, 11 fold_exact, 15 threadmap, 24 pages_threadmap"
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
    comptime EPI = get_defined_int["MOJOLEARN_IDN_NEURAL_GEMM_EPILOGUE",0]()
    comptime assert EPI >= 0 and EPI <= 3, "MOJOLEARN_IDN_NEURAL_GEMM_EPILOGUE is a mask: 1 mlp, 2 cnn"
    comptime HS = get_defined_int["MOJOLEARN_IDN_ATTN_HEAD_SHARE",1]()
    comptime assert HS == 1 or HS == 4, "MOJOLEARN_IDN_ATTN_HEAD_SHARE legal set {4} (two-head I06/NI19 deleted as a loser)"
    comptime SM = get_defined_int["MOJOLEARN_IDN_ATTN_SOFTMAX",0]()
    comptime assert SM >= 0 and SM <= 2, "MOJOLEARN_IDN_ATTN_SOFTMAX arms: 1 summary_tree, 2 online_tile32"
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
    return True

comptime SIX_LANE_CONFIGURATION_OK = _check_configuration()
