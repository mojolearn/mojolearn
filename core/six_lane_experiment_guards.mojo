# SPDX-License-Identifier: Apache-2.0
"""Mutually exclusive integrated strategies. All opt-in; incumbent defaults retained."""
from std.sys.compile import is_defined

def _integration_require_1() -> Bool:
    comptime assert not (is_defined["MOJOLEARN_NN20_BALANCED_SUMMARY_TREE"]() and is_defined["MOJOLEARN_IDN_ATTENTION_V2"]()), "incompatible integrated strategies: MOJOLEARN_NN20_BALANCED_SUMMARY_TREE / MOJOLEARN_IDN_ATTENTION_V2"
    return True

comptime _INTEGRATION_REQUIRE_1 = _integration_require_1()

def _integration_require_2() -> Bool:
    comptime assert not (is_defined["MOJOLEARN_NN25_RMS_SPLIT_SCALE"]() and is_defined["MOJOLEARN_IDN_RMS_ROW_BLOCK"]()), "incompatible integrated strategies: MOJOLEARN_NN25_RMS_SPLIT_SCALE / MOJOLEARN_IDN_RMS_ROW_BLOCK"
    return True

comptime _INTEGRATION_REQUIRE_2 = _integration_require_2()

def _integration_require_3() -> Bool:
    comptime assert not (is_defined["MOJOLEARN_NN27_QK_ROPE_PAIR"]() and is_defined["MOJOLEARN_IDN_ROPE_CACHE"]()), "incompatible integrated strategies: MOJOLEARN_NN27_QK_ROPE_PAIR / MOJOLEARN_IDN_ROPE_CACHE"
    return True

comptime _INTEGRATION_REQUIRE_3 = _integration_require_3()

def _integration_require_4() -> Bool:
    comptime assert not (is_defined["MOJOLEARN_NN36_SHARED_DECAY"]() and is_defined["MOJOLEARN_IDN_M2_YOFF_EXP_CACHE"]()), "incompatible integrated strategies: MOJOLEARN_NN36_SHARED_DECAY / MOJOLEARN_IDN_M2_YOFF_EXP_CACHE"
    return True

comptime _INTEGRATION_REQUIRE_4 = _integration_require_4()

def _integration_require_5() -> Bool:
    comptime assert not (is_defined["MOJOLEARN_NN43_WGRAD_FIXED128"]() and is_defined["MOJOLEARN_IDN_SEQ_WGRAD_LEAF256"]()), "incompatible integrated strategies: MOJOLEARN_NN43_WGRAD_FIXED128 / MOJOLEARN_IDN_SEQ_WGRAD_LEAF256"
    return True

comptime _INTEGRATION_REQUIRE_5 = _integration_require_5()

def _integration_require_6() -> Bool:
    comptime assert not (is_defined["MOJOLEARN_NN54_LOSS_PROFILE"]() and is_defined["MOJOLEARN_IDN_LOSS_TOKEN_TREE_V2"]()), "incompatible integrated strategies: MOJOLEARN_NN54_LOSS_PROFILE / MOJOLEARN_IDN_LOSS_TOKEN_TREE_V2"
    return True

comptime _INTEGRATION_REQUIRE_6 = _integration_require_6()

def _integration_require_7() -> Bool:
    comptime assert not (is_defined["MOJOLEARN_NN14_BOUNDED_IM2COL"]() and is_defined["MOJOLEARN_NI12_IMPLICIT_CONV"]()), "incompatible integrated strategies: MOJOLEARN_NN14_BOUNDED_IM2COL / MOJOLEARN_NI12_IMPLICIT_CONV"
    return True

comptime _INTEGRATION_REQUIRE_7 = _integration_require_7()

def _integration_require_8() -> Bool:
    comptime assert not (is_defined["MOJOLEARN_IDN_NEURAL_NN12"]() and is_defined["MOJOLEARN_NI01_TRAINING_WORKSPACE"]()), "incompatible integrated strategies: MOJOLEARN_IDN_NEURAL_NN12 / MOJOLEARN_NI01_TRAINING_WORKSPACE"
    return True

comptime _INTEGRATION_REQUIRE_8 = _integration_require_8()

def _integration_require_9() -> Bool:
    comptime assert not (is_defined["MOJOLEARN_NN48_CSR_TILES"]() and is_defined["MOJOLEARN_NI55_GRAPH_FEATURE4"]()), "incompatible integrated strategies: MOJOLEARN_NN48_CSR_TILES / MOJOLEARN_NI55_GRAPH_FEATURE4"
    return True

comptime _INTEGRATION_REQUIRE_9 = _integration_require_9()

def _integration_require_10() -> Bool:
    comptime assert not (is_defined["MOJOLEARN_NI59_DROPOUT_CHANNEL"]() and is_defined["MOJOLEARN_NI60_DROPOUT_APPLY4"]()), "incompatible integrated strategies: MOJOLEARN_NI59_DROPOUT_CHANNEL / MOJOLEARN_NI60_DROPOUT_APPLY4"
    return True

comptime _INTEGRATION_REQUIRE_10 = _integration_require_10()

def _integration_require_11() -> Bool:
    comptime assert not (is_defined["MOJOLEARN_IDN_NEURAL_NN02"]() and is_defined["MOJOLEARN_NI02_GEMM_STREAM_PARTIALS"]()), "incompatible integrated strategies: MOJOLEARN_IDN_NEURAL_NN02 / MOJOLEARN_NI02_GEMM_STREAM_PARTIALS"
    return True

comptime _INTEGRATION_REQUIRE_11 = _integration_require_11()

def _integration_require_12() -> Bool:
    comptime assert not (is_defined["MOJOLEARN_IDN_NEURAL_NN03"]() and is_defined["MOJOLEARN_NI08_GEMM_LEAF_256"]()), "incompatible integrated strategies: MOJOLEARN_IDN_NEURAL_NN03 / MOJOLEARN_NI08_GEMM_LEAF_256"
    return True

comptime _INTEGRATION_REQUIRE_12 = _integration_require_12()

def _integration_require_13() -> Bool:
    comptime assert not (is_defined["MOJOLEARN_NN34_AFFINE_PREFIX"]() and is_defined["MOJOLEARN_IDN_M1_STATE_WINDOW"]()), "incompatible integrated strategies: MOJOLEARN_NN34_AFFINE_PREFIX / MOJOLEARN_IDN_M1_STATE_WINDOW"
    return True

comptime _INTEGRATION_REQUIRE_13 = _integration_require_13()

def _integration_require_14() -> Bool:
    comptime assert not (is_defined["MOJOLEARN_NN39_M2_GRAD_TREE"]() and is_defined["MOJOLEARN_IDN_M2_GRAD_LEAF128"]()), "incompatible integrated strategies: MOJOLEARN_NN39_M2_GRAD_TREE / MOJOLEARN_IDN_M2_GRAD_LEAF128"
    return True

comptime _INTEGRATION_REQUIRE_14 = _integration_require_14()

def _integration_require_15() -> Bool:
    comptime assert not (is_defined["MOJOLEARN_NN53_HEAD_CHUNK512"]() and is_defined["MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2"]()), "incompatible integrated strategies: MOJOLEARN_NN53_HEAD_CHUNK512 / MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2"
    return True

comptime _INTEGRATION_REQUIRE_15 = _integration_require_15()

def _integration_require_16() -> Bool:
    comptime assert not (is_defined["MOJOLEARN_FOREST_ORDERED_RESIDENT_OFF"]() and is_defined["MOJOLEARN_AFT_P02"]()), "incompatible integrated strategies: MOJOLEARN_FOREST_ORDERED_RESIDENT_OFF / MOJOLEARN_AFT_P02"
    return True

comptime _INTEGRATION_REQUIRE_16 = _integration_require_16()

def _integration_require_17() -> Bool:
    comptime assert not (is_defined["MOJOLEARN_AFT_P07"]() and is_defined["MOJOLEARN_SHAP_FAST_ROW_PAIR"]()), "incompatible integrated strategies: MOJOLEARN_AFT_P07 / MOJOLEARN_SHAP_FAST_ROW_PAIR"
    return True

comptime _INTEGRATION_REQUIRE_17 = _integration_require_17()

comptime SIX_LANE_CONFIGURATION_OK = True
