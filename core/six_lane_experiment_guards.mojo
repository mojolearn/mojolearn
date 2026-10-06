# SPDX-License-Identifier: Apache-2.0
"""Opt-in strategy admission, evaluated through GLOBAL_NUMERIC_MODE.

A standalone unused comptime binding is lazy and cannot enforce a guard.
This predicate is required by the numeric-mode constant used in every binding.
"""
from std.sys.compile import is_defined, get_defined_int


def _check_configuration() -> Bool:
    comptime assert not (is_defined["MOJOLEARN_NN20_BALANCED_SUMMARY_TREE"]() and is_defined["MOJOLEARN_IDN_ATTENTION_V2"]()), "incompatible integrated strategies: MOJOLEARN_NN20_BALANCED_SUMMARY_TREE / MOJOLEARN_IDN_ATTENTION_V2"
    comptime assert not (is_defined["MOJOLEARN_NN25_RMS_SPLIT_SCALE"]() and is_defined["MOJOLEARN_IDN_RMS_ROW_BLOCK"]()), "incompatible integrated strategies: MOJOLEARN_NN25_RMS_SPLIT_SCALE / MOJOLEARN_IDN_RMS_ROW_BLOCK"
    comptime assert not (is_defined["MOJOLEARN_NN27_QK_ROPE_PAIR"]() and is_defined["MOJOLEARN_IDN_ROPE_CACHE"]()), "incompatible integrated strategies: MOJOLEARN_NN27_QK_ROPE_PAIR / MOJOLEARN_IDN_ROPE_CACHE"
    comptime assert not (is_defined["MOJOLEARN_NN36_SHARED_DECAY"]() and is_defined["MOJOLEARN_IDN_M2_YOFF_EXP_CACHE"]()), "incompatible integrated strategies: MOJOLEARN_NN36_SHARED_DECAY / MOJOLEARN_IDN_M2_YOFF_EXP_CACHE"
    comptime assert not (is_defined["MOJOLEARN_NN43_WGRAD_FIXED128"]() and is_defined["MOJOLEARN_IDN_SEQ_WGRAD_LEAF256"]()), "incompatible integrated strategies: MOJOLEARN_NN43_WGRAD_FIXED128 / MOJOLEARN_IDN_SEQ_WGRAD_LEAF256"
    comptime assert not (is_defined["MOJOLEARN_NN54_LOSS_PROFILE"]() and is_defined["MOJOLEARN_IDN_LOSS_TOKEN_TREE_V2"]()), "incompatible integrated strategies: MOJOLEARN_NN54_LOSS_PROFILE / MOJOLEARN_IDN_LOSS_TOKEN_TREE_V2"
    comptime assert not (is_defined["MOJOLEARN_NN14_BOUNDED_IM2COL"]() and is_defined["MOJOLEARN_NI12_IMPLICIT_CONV"]()), "incompatible integrated strategies: MOJOLEARN_NN14_BOUNDED_IM2COL / MOJOLEARN_NI12_IMPLICIT_CONV"
    comptime assert not (is_defined["MOJOLEARN_IDN_NEURAL_NN12"]() and is_defined["MOJOLEARN_NI01_TRAINING_WORKSPACE"]()), "incompatible integrated strategies: MOJOLEARN_IDN_NEURAL_NN12 / MOJOLEARN_NI01_TRAINING_WORKSPACE"
    comptime assert not (is_defined["MOJOLEARN_NN48_CSR_TILES"]() and is_defined["MOJOLEARN_NI55_GRAPH_FEATURE4"]()), "incompatible integrated strategies: MOJOLEARN_NN48_CSR_TILES / MOJOLEARN_NI55_GRAPH_FEATURE4"
    comptime assert not (is_defined["MOJOLEARN_NI59_DROPOUT_CHANNEL"]() and is_defined["MOJOLEARN_NI60_DROPOUT_APPLY4"]()), "incompatible integrated strategies: MOJOLEARN_NI59_DROPOUT_CHANNEL / MOJOLEARN_NI60_DROPOUT_APPLY4"
    comptime assert not (is_defined["MOJOLEARN_IDN_NEURAL_NN02"]() and is_defined["MOJOLEARN_NI02_GEMM_STREAM_PARTIALS"]()), "incompatible integrated strategies: MOJOLEARN_IDN_NEURAL_NN02 / MOJOLEARN_NI02_GEMM_STREAM_PARTIALS"
    comptime assert not (is_defined["MOJOLEARN_IDN_NEURAL_NN03"]() and is_defined["MOJOLEARN_NI08_GEMM_LEAF_256"]()), "incompatible integrated strategies: MOJOLEARN_IDN_NEURAL_NN03 / MOJOLEARN_NI08_GEMM_LEAF_256"
    comptime assert not (is_defined["MOJOLEARN_NN34_AFFINE_PREFIX"]() and is_defined["MOJOLEARN_IDN_M1_STATE_WINDOW"]()), "incompatible integrated strategies: MOJOLEARN_NN34_AFFINE_PREFIX / MOJOLEARN_IDN_M1_STATE_WINDOW"
    comptime assert not (is_defined["MOJOLEARN_NN39_M2_GRAD_TREE"]() and is_defined["MOJOLEARN_IDN_M2_GRAD_LEAF128"]()), "incompatible integrated strategies: MOJOLEARN_NN39_M2_GRAD_TREE / MOJOLEARN_IDN_M2_GRAD_LEAF128"
    comptime assert not (is_defined["MOJOLEARN_NN53_HEAD_CHUNK512"]() and is_defined["MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2"]()), "incompatible integrated strategies: MOJOLEARN_NN53_HEAD_CHUNK512 / MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2"
    comptime assert not (is_defined["MOJOLEARN_FOREST_ORDERED_RESIDENT_OFF"]() and is_defined["MOJOLEARN_AFT_P02"]()), "incompatible integrated strategies: MOJOLEARN_FOREST_ORDERED_RESIDENT_OFF / MOJOLEARN_AFT_P02"
    comptime assert not (is_defined["MOJOLEARN_AFT_P07"]() and is_defined["MOJOLEARN_SHAP_FAST_ROW_PAIR"]()), "incompatible integrated strategies: MOJOLEARN_AFT_P07 / MOJOLEARN_SHAP_FAST_ROW_PAIR"
    comptime assert not (is_defined["MOJOLEARN_AFCL_G01"]() and is_defined["MOJOLEARN_KNN_FAST_MMA_OFF"]()), "incompatible integrated strategies: MOJOLEARN_AFCL_G01 / MOJOLEARN_KNN_FAST_MMA_OFF"
    comptime assert not (is_defined["MOJOLEARN_IDN_NEURAL_NN01"]() and is_defined["MOJOLEARN_IDN_NEURAL_NN03"]()), "incompatible integrated strategies: MOJOLEARN_IDN_NEURAL_NN01 / MOJOLEARN_IDN_NEURAL_NN03"
    comptime assert not (is_defined["MOJOLEARN_IDN_NEURAL_NN01"]() and is_defined["MOJOLEARN_IDN_NEURAL_NN04"]()), "incompatible integrated strategies: MOJOLEARN_IDN_NEURAL_NN01 / MOJOLEARN_IDN_NEURAL_NN04"
    comptime assert not (is_defined["MOJOLEARN_IDN_GEMM_FOLD_LEAF_64"]() and is_defined["MOJOLEARN_NI08_GEMM_LEAF_256"]()), "incompatible integrated strategies: MOJOLEARN_IDN_GEMM_FOLD_LEAF_64 / MOJOLEARN_NI08_GEMM_LEAF_256"
    comptime assert not (is_defined["MOJOLEARN_NN53_HEAD_CHUNK512"]() and is_defined["MOJOLEARN_NN53_HEAD_CHUNK2048"]()), "incompatible integrated strategies: MOJOLEARN_NN53_HEAD_CHUNK512 / MOJOLEARN_NN53_HEAD_CHUNK2048"
    comptime assert not (is_defined["MOJOLEARN_AFN26_MAMBA1_CHUNKS16"]() and is_defined["MOJOLEARN_AFN26_MAMBA1_CHUNKS64"]()), "incompatible integrated strategies: MOJOLEARN_AFN26_MAMBA1_CHUNKS16 / MOJOLEARN_AFN26_MAMBA1_CHUNKS64"
    comptime assert not (is_defined["MOJOLEARN_AFN26_MAMBA3_THREADS64"]() and is_defined["MOJOLEARN_AFN26_MAMBA3_THREADS256"]()), "incompatible integrated strategies: MOJOLEARN_AFN26_MAMBA3_THREADS64 / MOJOLEARN_AFN26_MAMBA3_THREADS256"
    comptime assert not (is_defined["MOJOLEARN_AFN26_EMB_THREADS64"]() and is_defined["MOJOLEARN_AFN26_EMB_THREADS128"]()), "incompatible integrated strategies: MOJOLEARN_AFN26_EMB_THREADS64 / MOJOLEARN_AFN26_EMB_THREADS128"
    comptime assert not (is_defined["MOJOLEARN_AFN26_ATTN_NORM_TPB128"]() and is_defined["MOJOLEARN_AFN26_ATTN_NORM_TPB512"]()), "incompatible integrated strategies: MOJOLEARN_AFN26_ATTN_NORM_TPB128 / MOJOLEARN_AFN26_ATTN_NORM_TPB512"
    comptime assert not (is_defined["MOJOLEARN_C52_PAIR_128"]() and is_defined["MOJOLEARN_C52_PAIR_512"]()), "incompatible integrated strategies: MOJOLEARN_C52_PAIR_128 / MOJOLEARN_C52_PAIR_512"
    comptime assert Int(is_defined["MOJOLEARN_IDN_NEURAL_NN01"]())+Int(is_defined["MOJOLEARN_IDN_NEURAL_NN02"]())+Int(is_defined["MOJOLEARN_IDN_NEURAL_NN08"]())+Int(is_defined["MOJOLEARN_IDN_NEURAL_NN10"]())+Int(is_defined["MOJOLEARN_IDN_NEURAL_NN09"]() or is_defined["MOJOLEARN_IDN_NEURAL_NN15"]())+Int(is_defined["MOJOLEARN_IDN_NEURAL_NN11"]() and not is_defined["MOJOLEARN_IDN_NEURAL_NN02"]()) <= 1, "select one neural GEMM schedule; NN11 composes with NN02"
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_NN10"]() or is_defined["MOJOLEARN_IDN_NEURAL_FILL_BLOCKS"](), "NN10 needs an explicit hardware fill budget"
    comptime assert get_defined_int["MOJOLEARN_IDN_NEURAL_LEAF",128]() == 64 or get_defined_int["MOJOLEARN_IDN_NEURAL_LEAF",128]() == 128 or get_defined_int["MOJOLEARN_IDN_NEURAL_LEAF",128]() == 256, "invalid neural LEAF configuration"
    comptime assert get_defined_int["MOJOLEARN_IDN_NEURAL_CHAINS",2]() == 1 or get_defined_int["MOJOLEARN_IDN_NEURAL_CHAINS",2]() == 2 or get_defined_int["MOJOLEARN_IDN_NEURAL_CHAINS",2]() == 4, "invalid neural CHAINS configuration"
    comptime assert get_defined_int["MOJOLEARN_IDN_NEURAL_STAGE_DEPTH",2]() == 1 or get_defined_int["MOJOLEARN_IDN_NEURAL_STAGE_DEPTH",2]() == 2, "invalid neural STAGE_DEPTH configuration"
    comptime assert get_defined_int["MOJOLEARN_IDN_NEURAL_STAGE_PAD",0]() == 0 or get_defined_int["MOJOLEARN_IDN_NEURAL_STAGE_PAD",0]() == 1, "invalid neural STAGE_PAD configuration"
    comptime assert get_defined_int["MOJOLEARN_IDN_NEURAL_FILL_BLOCKS",1]() > 0, "positive neural resource budget required"
    comptime assert get_defined_int["MOJOLEARN_IDN_NEURAL_STREAM_GROUP",8]() > 0, "positive neural resource budget required"
    comptime assert get_defined_int["MOJOLEARN_IDN_NEURAL_RETAINED_FLOATS",16777216]() > 0, "positive neural resource budget required"
    return True

comptime SIX_LANE_CONFIGURATION_OK = _check_configuration()
