# SPDX-License-Identifier: Apache-2.0
"""Mutually exclusive integrated source strategies; all opt-in, no default changes."""
from std.sys.compile import is_defined

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
comptime SIX_LANE_CONFIGURATION_OK = True
