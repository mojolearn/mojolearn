# SPDX-License-Identifier: Apache-2.0
"""RUN half of `providers/python/autograd_ops_frontend_pyapi.mojo` (wave 8 run
split): the session operation codes, their name table (one per process) and
the metadata append helper. Typed Mojo only: no Python import."""
from core.shared_slot import SharedSlot


# Session operation codes (row N089, lesson 12): the operation name is looked up
# once per call in one native table built at first use, and the dispatch below
# compares integers. The codes are private to this file (the public entry still
# takes the operation name); 0 = unknown.  Dispatch only, no arithmetic.
comptime _SOP_PROGRAM_PUBLIC = 1
comptime _SOP_PROGRAM_INIT = 2
comptime _SOP_ENTER = 3
comptime _SOP_FREE_PROGRAM = 4
comptime _SOP_CLOSE = 5
comptime _SOP_EXIT = 6
comptime _SOP_PROFILE_RESET = 7
comptime _SOP_PROFILE_REPORT = 8
comptime _SOP_LOAD_PARTITIONED = 9
comptime _SOP_LOAD_DISTRIBUTED = 10
comptime _SOP_PROGRAM_RUN = 11
comptime _SOP_FIT = 12
comptime _SOP_FIT_IF = 13
comptime _SOP_GRADIENT_REPORT = 14
comptime _SOP_EXECUTION_PROFILE_BEGIN = 15
comptime _SOP_EXECUTION_PROFILE_END = 16
comptime _SOP_SAVE_CONTROL = 17
comptime _SOP_LOAD_CONTROL = 18
comptime _SOP_SAVE_PROGRAM = 19
comptime _SOP_LOAD_WEIGHTS = 20
comptime _SOP_TENSOR = 21
comptime _SOP_IDS = 22
comptime _SOP_SHAPE = 23
comptime _SOP_NUMPY = 24
comptime _SOP_GRAD = 25
comptime _SOP_SCALAR = 26
comptime _SOP_CAPTURE = 27
comptime _SOP_LOAD_PROGRAM = 28
comptime _SOP_SAVE_STATE = 29
comptime _SOP_LOAD_STATE = 30
comptime _SOP_RESET = 31
comptime _SOP_FREE = 32
comptime _SOP_BACKWARD = 33
comptime _SOP_GRAD_TENSOR = 34
comptime _SOP_VJP = 35
comptime _SOP_HVP = 36
comptime _SOP_JVP = 37
comptime _SOP_ZERO_GRAD = 38
comptime _SOP_OPTIMIZER = 39
comptime _SOP_STEP = 40
comptime _SOP_SCHEDULER_STEP = 41
comptime _SOP_LEARNING_RATE = 42
comptime _SOP_OPTIMIZER_STATE = 43
comptime _SOP_LOAD_OPTIMIZER_STATE = 44
comptime _SOP_COPY = 45
comptime _SOP_DETACH = 46
comptime _SOP_MATMUL = 47
comptime _SOP_ADD = 48
comptime _SOP_SUB = 49
comptime _SOP_MUL = 50
comptime _SOP_DIV = 51
comptime _SOP_RELU = 52
comptime _SOP_SIGMOID = 53
comptime _SOP_TANH = 54
comptime _SOP_EXP = 55
comptime _SOP_LOG = 56
comptime _SOP_SQRT = 57
comptime _SOP_ERF = 58
comptime _SOP_SOFTPLUS = 59
comptime _SOP_SILU = 60
comptime _SOP_HARD_SWISH = 61
comptime _SOP_ABS = 62
comptime _SOP_FLOOR = 63
comptime _SOP_CEIL = 64
comptime _SOP_ROUND = 65
comptime _SOP_SIN = 66
comptime _SOP_COS = 67
comptime _SOP_SIGN = 68
comptime _SOP_RECIPROCAL = 69
comptime _SOP_POW = 70
comptime _SOP_MINIMUM = 71
comptime _SOP_MAXIMUM = 72
comptime _SOP_WHERE = 73
comptime _SOP_FULL = 74
comptime _SOP_EXPAND = 75
comptime _SOP_SCALE = 76
comptime _SOP_LEAKY_RELU = 77
comptime _SOP_ELU = 78
comptime _SOP_POW_SCALAR = 79
comptime _SOP_HARD_SIGMOID = 80
comptime _SOP_TRANSPOSE = 81
comptime _SOP_PERMUTE = 82
comptime _SOP_RESHAPE = 83
comptime _SOP_FLATTEN = 84
comptime _SOP_SQUEEZE = 85
comptime _SOP_UNSQUEEZE = 86
comptime _SOP_SUM = 87
comptime _SOP_MEAN = 88
comptime _SOP_MAX = 89
comptime _SOP_ANY = 90
comptime _SOP_ALL = 91
comptime _SOP_SOFTMAX = 92
comptime _SOP_LOG_SOFTMAX = 93
comptime _SOP_CONCAT = 94
comptime _SOP_SPLIT = 95
comptime _SOP_EMBEDDING = 96
comptime _SOP_TAKE = 97
comptime _SOP_CONV2D = 98
comptime _SOP_LINEAR = 99
comptime _SOP_MAX_POOL2D = 100
comptime _SOP_AVG_POOL2D = 101
comptime _SOP_GELU = 102
comptime _SOP_CLIP = 103
comptime _SOP_EQUAL = 104
comptime _SOP_NOT_EQUAL = 105
comptime _SOP_GREATER = 106
comptime _SOP_GREATER_EQUAL = 107
comptime _SOP_LESS = 108
comptime _SOP_LESS_EQUAL = 109
comptime _SOP_LOGICAL_NOT = 110
comptime _SOP_LOGICAL_AND = 111
comptime _SOP_LOGICAL_OR = 112
comptime _SOP_ARGMAX = 113
comptime _SOP_PAD = 114
comptime _SOP_CUMSUM = 115
comptime _SOP_INDEX = 116
comptime _SOP_LAYER_NORM = 117
comptime _SOP_TRANSFORMER_SOFTMAX = 118
comptime _SOP_TRANSFORMER_LINEAR = 119
comptime _SOP_RMS_NORM = 120
comptime _SOP_ROTARY = 121
comptime _SOP_DROPOUT = 122
comptime _SOP_PERMUTATION = 123
comptime _SOP_GATHER_ROWS = 124
comptime _SOP_MSE = 125
comptime _SOP_MAE = 126
comptime _SOP_HUBER = 127
comptime _SOP_BCE_WITH_LOGITS = 128
comptime _SOP_CROSS_ENTROPY = 129
comptime _SOP_ACCURACY = 130
comptime _SOP_LOG_LOSS = 131
comptime _SOP_ROC_AUC = 132
# lane x1: an uploaded tensor declared carried state (OpsSession.declare_state)
comptime _SOP_STATE = 133
# lane fx-ops-session-blocks (2026-11-18): the block ops' public names. The
# two block makers pack a block's weights, zero state and state words
# (OpsSession.upload_block_grad); the two steps run OPS_M{1,2,3}_STEP 81-83 /
# OPS_ATTN_STEP 84; small_mlp is OPS_SMALL_MLP 88, lm_head_loss OPS_LM_HEAD_V2 87.
comptime _SOP_MAMBA_BLOCK = 134
comptime _SOP_ATTENTION_BLOCK = 135
comptime _SOP_MAMBA_STEP = 136
comptime _SOP_ATTENTION_STEP = 137
comptime _SOP_SMALL_MLP = 138
comptime _SOP_LM_HEAD_LOSS = 139
# 2026-11-19: the generation ops (85, 86) with their rows state, and the LM
# trainer's embedding (93)
comptime _SOP_LM_ROWS = 140
comptime _SOP_LM_PICK = 141
comptime _SOP_LM_ALIVE = 142
comptime _SOP_LM_EMBEDDING = 143
# the fused CNN block and head (90, 91)
comptime _SOP_CONV_BLOCK = 144
comptime _SOP_CONV_LINEAR = 145
# one recurrent layer over a sequence (94)
comptime _SOP_LSTM = 146
comptime _SOP_GRU = 147
comptime _SOP_RNN = 148
# nn-blocks-qkv begin: attention over explicit q, k, v (122); its own name code, clear of the shared 1xx run
comptime _SOP_ATTENTION_QKV = 1122
# nn-blocks-qkv end
# nn-blocks-pg begin: optimizer parameter groups (providers/python/nn_pg_pyapi_door.mojo); a range of its own,
# clear of the shared sequence above
comptime _SOP_PG_OPTIMIZER = 1190
comptime _SOP_PG_LEARNING_RATES = 1191
comptime _SOP_PG_STATE_DICT = 1192
comptime _SOP_PG_LOAD_STATE_DICT = 1193
comptime _SOP_PG_FIRST = _SOP_PG_OPTIMIZER
comptime _SOP_PG_LAST = _SOP_PG_LOAD_STATE_DICT
# nn-blocks-pg end
# nn-blocks-scatter begin: scatter_add and the segment reductions (session codes 113..116); private codes
# 1113..1116 so the nn-blocks sub-lanes' additions cannot collide
comptime _SOP_SCATTER_ADD = 1113
comptime _SOP_SEGMENT_SUM = 1114
comptime _SOP_SEGMENT_MEAN = 1115
comptime _SOP_SEGMENT_MAX = 1116
# nn-blocks-scatter end
# nn-blocks-scan begin: the scan over a captured cell (OPS 118; the frontend code is 1000 + 118, outside the
# shared 1..999 sequence so the parallel nn-blocks lanes cannot collide)
comptime _SOP_SCAN = 1118
# nn-blocks-scan end
# nn-blocks-cbw begin: a user-defined backward rule (autograd/nn_cbw.mojo);
# codes from the sub-lane's own range, so no other lane's next code collides
comptime _SOP_CUSTOM_OP = 1260
comptime _SOP_CUSTOM = 1261
# nn-blocks-cbw end
# lane batch-source: Session.load_training (autograd/program_training_checkpoint.mojo), its own range
comptime _SOP_LOAD_TRAINING = 1270
# session-ops-resize begin: interpolate, pixel_shuffle, pixel_unshuffle (1000 + session codes 150..152); pad with
# a mode keeps the name "pad" (_SOP_PAD): the door chooses session code 153 when the call names a mode
comptime _SOP_INTERPOLATE = 1150
comptime _SOP_PIXEL_SHUFFLE = 1151
comptime _SOP_PIXEL_UNSHUFFLE = 1152
# session-ops-resize end
# session-ops-conv begin: Sequential's convolutions as Session ops (session codes 130, 132..134; 131, conv2d with
# dilation or per-axis words, is chosen by Session.conv2d's door and has no name of its own)
comptime _SOP_CONV1D = 1130
comptime _SOP_CONV3D = 1132
comptime _SOP_CONV_TRANSPOSE1D = 1133
comptime _SOP_CONV_TRANSPOSE2D = 1134
# session-ops-conv end
# session-ops-pool begin: Sequential's pools as Session ops (session codes 140..148, autograd/session_ops_pool.mojo);
# frontend codes inside this sub-lane's 1140..1149 (the door chooses the session code: max_pool2d / avg_pool2d keep
# their names' codes and take 142 / 143 only with an option; the adaptive 2-D names use the free 1142 / 1143 and
# record 147 / 148 with rank word 2)
comptime _SOP_MAX_POOL1D = 1140
comptime _SOP_AVG_POOL1D = 1141
comptime _SOP_MAX_POOL3D = 1144
comptime _SOP_AVG_POOL3D = 1145
comptime _SOP_ADAPTIVE_AVG_POOL3D = 1146
comptime _SOP_ADAPTIVE_AVG_POOL1D = 1147
comptime _SOP_ADAPTIVE_MAX_POOL1D = 1148
comptime _SOP_ADAPTIVE_AVG_POOL2D = 1142
comptime _SOP_ADAPTIVE_MAX_POOL2D = 1143
# session-ops-pool end
# session-ops-extra begin: 1000 + the session codes (180..183; the soft-target cross-entropy is cross_entropy's name)
comptime _SOP_SEQUENCE_MASK = 1180
comptime _SOP_TOPK = 1181
comptime _SOP_GATHER = 1182
comptime _SOP_LOGSUMEXP = 1183
# session-ops-extra end

# session-ops-rest begin: Sequential's norms, prelu, mish, multi-head attention, positional encoding, losses and raw
# SSM scans (session codes 160..174, autograd/session_ops_rest.mojo); frontend codes 1000 + the session code
comptime _SOP_SOR_BATCH_NORM = 1160
comptime _SOP_SOR_GROUP_NORM = 1161
comptime _SOP_SOR_INSTANCE_NORM = 1162
comptime _SOP_SOR_PRELU = 1163
comptime _SOP_SOR_MISH = 1164
comptime _SOP_SOR_MULTI_HEAD_ATTENTION = 1165
comptime _SOP_SOR_POSITIONAL_ENCODING = 1166
comptime _SOP_SOR_SMOOTH_L1_LOSS = 1167
comptime _SOP_SOR_NLL_LOSS = 1168
comptime _SOP_SOR_KL_DIV = 1169
comptime _SOP_SOR_BINARY_CROSS_ENTROPY_WITH_LOGITS = 1170
comptime _SOP_SOR_CTC_LOSS = 1171
comptime _SOP_SOR_SELECTIVE_SCAN = 1172
comptime _SOP_SOR_SSD_SCAN = 1173
comptime _SOP_SOR_TRAPEZOID_SCAN = 1174
comptime _SOP_SOR_FIRST = 1160
comptime _SOP_SOR_LAST = 1174
# session-ops-rest end


struct _SessionOpCodes(Defaultable, Movable):
    var codes: Dict[String, Int]

    def __init__(out self):
        self.codes = Dict[String, Int]()
        self.codes["program_public"] = _SOP_PROGRAM_PUBLIC
        self.codes["program_init"] = _SOP_PROGRAM_INIT
        self.codes["enter"] = _SOP_ENTER
        self.codes["free_program"] = _SOP_FREE_PROGRAM
        self.codes["close"] = _SOP_CLOSE
        self.codes["exit"] = _SOP_EXIT
        self.codes["profile_reset"] = _SOP_PROFILE_RESET
        self.codes["profile_report"] = _SOP_PROFILE_REPORT
        self.codes["load_partitioned"] = _SOP_LOAD_PARTITIONED
        self.codes["load_distributed"] = _SOP_LOAD_DISTRIBUTED
        self.codes["program_run"] = _SOP_PROGRAM_RUN
        self.codes["fit"] = _SOP_FIT
        self.codes["fit_if"] = _SOP_FIT_IF
        self.codes["gradient_report"] = _SOP_GRADIENT_REPORT
        self.codes["execution_profile_begin"] = _SOP_EXECUTION_PROFILE_BEGIN
        self.codes["execution_profile_end"] = _SOP_EXECUTION_PROFILE_END
        self.codes["save_control"] = _SOP_SAVE_CONTROL
        self.codes["load_control"] = _SOP_LOAD_CONTROL
        self.codes["save_program"] = _SOP_SAVE_PROGRAM
        self.codes["load_weights"] = _SOP_LOAD_WEIGHTS
        self.codes["tensor"] = _SOP_TENSOR
        self.codes["ids"] = _SOP_IDS
        self.codes["shape"] = _SOP_SHAPE
        self.codes["numpy"] = _SOP_NUMPY
        self.codes["grad"] = _SOP_GRAD
        self.codes["scalar"] = _SOP_SCALAR
        self.codes["capture"] = _SOP_CAPTURE
        self.codes["load_program"] = _SOP_LOAD_PROGRAM
        self.codes["save_state"] = _SOP_SAVE_STATE
        self.codes["load_state"] = _SOP_LOAD_STATE
        self.codes["reset"] = _SOP_RESET
        self.codes["free"] = _SOP_FREE
        self.codes["backward"] = _SOP_BACKWARD
        self.codes["grad_tensor"] = _SOP_GRAD_TENSOR
        self.codes["vjp"] = _SOP_VJP
        self.codes["hvp"] = _SOP_HVP
        self.codes["jvp"] = _SOP_JVP
        self.codes["zero_grad"] = _SOP_ZERO_GRAD
        self.codes["optimizer"] = _SOP_OPTIMIZER
        self.codes["step"] = _SOP_STEP
        self.codes["scheduler_step"] = _SOP_SCHEDULER_STEP
        self.codes["learning_rate"] = _SOP_LEARNING_RATE
        self.codes["optimizer_state"] = _SOP_OPTIMIZER_STATE
        self.codes["load_optimizer_state"] = _SOP_LOAD_OPTIMIZER_STATE
        self.codes["copy"] = _SOP_COPY
        self.codes["detach"] = _SOP_DETACH
        self.codes["matmul"] = _SOP_MATMUL
        self.codes["add"] = _SOP_ADD
        self.codes["sub"] = _SOP_SUB
        self.codes["mul"] = _SOP_MUL
        self.codes["div"] = _SOP_DIV
        self.codes["relu"] = _SOP_RELU
        self.codes["sigmoid"] = _SOP_SIGMOID
        self.codes["tanh"] = _SOP_TANH
        self.codes["exp"] = _SOP_EXP
        self.codes["log"] = _SOP_LOG
        self.codes["sqrt"] = _SOP_SQRT
        self.codes["erf"] = _SOP_ERF
        self.codes["softplus"] = _SOP_SOFTPLUS
        self.codes["silu"] = _SOP_SILU
        self.codes["hard_swish"] = _SOP_HARD_SWISH
        self.codes["abs"] = _SOP_ABS
        self.codes["floor"] = _SOP_FLOOR
        self.codes["ceil"] = _SOP_CEIL
        self.codes["round"] = _SOP_ROUND
        self.codes["sin"] = _SOP_SIN
        self.codes["cos"] = _SOP_COS
        self.codes["sign"] = _SOP_SIGN
        self.codes["reciprocal"] = _SOP_RECIPROCAL
        self.codes["pow"] = _SOP_POW
        self.codes["minimum"] = _SOP_MINIMUM
        self.codes["maximum"] = _SOP_MAXIMUM
        self.codes["where"] = _SOP_WHERE
        self.codes["full"] = _SOP_FULL
        self.codes["expand"] = _SOP_EXPAND
        self.codes["scale"] = _SOP_SCALE
        self.codes["leaky_relu"] = _SOP_LEAKY_RELU
        self.codes["elu"] = _SOP_ELU
        self.codes["pow_scalar"] = _SOP_POW_SCALAR
        self.codes["hard_sigmoid"] = _SOP_HARD_SIGMOID
        self.codes["transpose"] = _SOP_TRANSPOSE
        self.codes["permute"] = _SOP_PERMUTE
        self.codes["reshape"] = _SOP_RESHAPE
        self.codes["flatten"] = _SOP_FLATTEN
        self.codes["squeeze"] = _SOP_SQUEEZE
        self.codes["unsqueeze"] = _SOP_UNSQUEEZE
        self.codes["sum"] = _SOP_SUM
        self.codes["mean"] = _SOP_MEAN
        self.codes["max"] = _SOP_MAX
        self.codes["any"] = _SOP_ANY
        self.codes["all"] = _SOP_ALL
        self.codes["softmax"] = _SOP_SOFTMAX
        self.codes["log_softmax"] = _SOP_LOG_SOFTMAX
        self.codes["concat"] = _SOP_CONCAT
        self.codes["split"] = _SOP_SPLIT
        self.codes["embedding"] = _SOP_EMBEDDING
        self.codes["take"] = _SOP_TAKE
        self.codes["conv2d"] = _SOP_CONV2D
        self.codes["linear"] = _SOP_LINEAR
        self.codes["max_pool2d"] = _SOP_MAX_POOL2D
        self.codes["avg_pool2d"] = _SOP_AVG_POOL2D
        self.codes["gelu"] = _SOP_GELU
        self.codes["clip"] = _SOP_CLIP
        self.codes["equal"] = _SOP_EQUAL
        self.codes["not_equal"] = _SOP_NOT_EQUAL
        self.codes["greater"] = _SOP_GREATER
        self.codes["greater_equal"] = _SOP_GREATER_EQUAL
        self.codes["less"] = _SOP_LESS
        self.codes["less_equal"] = _SOP_LESS_EQUAL
        self.codes["logical_not"] = _SOP_LOGICAL_NOT
        self.codes["logical_and"] = _SOP_LOGICAL_AND
        self.codes["logical_or"] = _SOP_LOGICAL_OR
        self.codes["argmax"] = _SOP_ARGMAX
        self.codes["pad"] = _SOP_PAD
        self.codes["cumsum"] = _SOP_CUMSUM
        self.codes["index"] = _SOP_INDEX
        self.codes["layer_norm"] = _SOP_LAYER_NORM
        self.codes["transformer_softmax"] = _SOP_TRANSFORMER_SOFTMAX
        self.codes["transformer_linear"] = _SOP_TRANSFORMER_LINEAR
        self.codes["rms_norm"] = _SOP_RMS_NORM
        self.codes["rotary"] = _SOP_ROTARY
        self.codes["dropout"] = _SOP_DROPOUT
        self.codes["permutation"] = _SOP_PERMUTATION
        self.codes["gather_rows"] = _SOP_GATHER_ROWS
        self.codes["mse"] = _SOP_MSE
        self.codes["mae"] = _SOP_MAE
        self.codes["huber"] = _SOP_HUBER
        self.codes["bce_with_logits"] = _SOP_BCE_WITH_LOGITS
        self.codes["cross_entropy"] = _SOP_CROSS_ENTROPY
        self.codes["accuracy"] = _SOP_ACCURACY
        self.codes["log_loss"] = _SOP_LOG_LOSS
        self.codes["roc_auc"] = _SOP_ROC_AUC
        self.codes["state"] = _SOP_STATE
        self.codes["mamba_block"] = _SOP_MAMBA_BLOCK
        self.codes["attention_block"] = _SOP_ATTENTION_BLOCK
        self.codes["mamba_step"] = _SOP_MAMBA_STEP
        self.codes["attention_step"] = _SOP_ATTENTION_STEP
        self.codes["small_mlp"] = _SOP_SMALL_MLP
        self.codes["lm_head_loss"] = _SOP_LM_HEAD_LOSS
        self.codes["lm_rows"] = _SOP_LM_ROWS
        self.codes["lm_pick"] = _SOP_LM_PICK
        self.codes["lm_alive"] = _SOP_LM_ALIVE
        self.codes["lm_embedding"] = _SOP_LM_EMBEDDING
        self.codes["conv_block"] = _SOP_CONV_BLOCK
        self.codes["conv_linear"] = _SOP_CONV_LINEAR
        self.codes["lstm"] = _SOP_LSTM
        self.codes["gru"] = _SOP_GRU
        self.codes["rnn"] = _SOP_RNN
        # nn-blocks-qkv begin
        self.codes["attention"] = _SOP_ATTENTION_QKV
        # nn-blocks-qkv end
        # nn-blocks-pg begin
        self.codes["param_group_optimizer"] = _SOP_PG_OPTIMIZER
        self.codes["learning_rates"] = _SOP_PG_LEARNING_RATES
        self.codes["optimizer_state_dict"] = _SOP_PG_STATE_DICT
        self.codes["load_optimizer_state_dict"] = _SOP_PG_LOAD_STATE_DICT
        # nn-blocks-pg end
        # nn-blocks-scatter begin
        self.codes["scatter_add"] = _SOP_SCATTER_ADD
        self.codes["segment_sum"] = _SOP_SEGMENT_SUM
        self.codes["segment_mean"] = _SOP_SEGMENT_MEAN
        self.codes["segment_max"] = _SOP_SEGMENT_MAX
        # nn-blocks-scatter end
        # nn-blocks-scan begin
        self.codes["scan"] = _SOP_SCAN
        # nn-blocks-scan end
        # nn-blocks-cbw begin
        self.codes["custom_op"] = _SOP_CUSTOM_OP
        self.codes["custom"] = _SOP_CUSTOM
        self.codes["load_training"] = _SOP_LOAD_TRAINING
        # nn-blocks-cbw end
        # session-ops-resize begin
        self.codes["interpolate"] = _SOP_INTERPOLATE
        self.codes["pixel_shuffle"] = _SOP_PIXEL_SHUFFLE
        self.codes["pixel_unshuffle"] = _SOP_PIXEL_UNSHUFFLE
        # session-ops-resize end
        # session-ops-conv begin
        self.codes["conv1d"] = _SOP_CONV1D
        self.codes["conv3d"] = _SOP_CONV3D
        self.codes["conv_transpose1d"] = _SOP_CONV_TRANSPOSE1D
        self.codes["conv_transpose2d"] = _SOP_CONV_TRANSPOSE2D
        # session-ops-conv end
        # session-ops-pool begin
        self.codes["max_pool1d"] = _SOP_MAX_POOL1D
        self.codes["avg_pool1d"] = _SOP_AVG_POOL1D
        self.codes["max_pool3d"] = _SOP_MAX_POOL3D
        self.codes["avg_pool3d"] = _SOP_AVG_POOL3D
        self.codes["adaptive_avg_pool3d"] = _SOP_ADAPTIVE_AVG_POOL3D
        self.codes["adaptive_avg_pool1d"] = _SOP_ADAPTIVE_AVG_POOL1D
        self.codes["adaptive_max_pool1d"] = _SOP_ADAPTIVE_MAX_POOL1D
        self.codes["adaptive_avg_pool2d"] = _SOP_ADAPTIVE_AVG_POOL2D
        self.codes["adaptive_max_pool2d"] = _SOP_ADAPTIVE_MAX_POOL2D
        # session-ops-pool end
        # session-ops-extra begin
        self.codes["sequence_mask"] = _SOP_SEQUENCE_MASK
        self.codes["topk"] = _SOP_TOPK
        self.codes["gather"] = _SOP_GATHER
        self.codes["logsumexp"] = _SOP_LOGSUMEXP
        # session-ops-extra end
        # session-ops-rest begin
        self.codes["batch_norm"] = _SOP_SOR_BATCH_NORM
        self.codes["group_norm"] = _SOP_SOR_GROUP_NORM
        self.codes["instance_norm"] = _SOP_SOR_INSTANCE_NORM
        self.codes["prelu"] = _SOP_SOR_PRELU
        self.codes["mish"] = _SOP_SOR_MISH
        self.codes["multi_head_attention"] = _SOP_SOR_MULTI_HEAD_ATTENTION
        self.codes["positional_encoding"] = _SOP_SOR_POSITIONAL_ENCODING
        self.codes["smooth_l1_loss"] = _SOP_SOR_SMOOTH_L1_LOSS
        self.codes["nll_loss"] = _SOP_SOR_NLL_LOSS
        self.codes["kl_div"] = _SOP_SOR_KL_DIV
        self.codes["binary_cross_entropy_with_logits"] = _SOP_SOR_BINARY_CROSS_ENTROPY_WITH_LOGITS
        self.codes["ctc_loss"] = _SOP_SOR_CTC_LOSS
        self.codes["selective_scan"] = _SOP_SOR_SELECTIVE_SCAN
        self.codes["ssd_scan"] = _SOP_SOR_SSD_SCAN
        self.codes["trapezoid_scan"] = _SOP_SOR_TRAPEZOID_SCAN
        # session-ops-rest end


# One table per process (identical contents in every extension that links this file).
comptime _SESSION_OP_CODES = SharedSlot[_SessionOpCodes, "MojolearnSessionOpCodesV4"]


# The name lookup is the door's (providers/python/autograd_ops_frontend_pyapi_door.mojo
# `_session_op_names`): one Python dictionary built once from this table and
# read with the caller's own str. The former `_session_op_code(name: String)`
# (a String copy and a hash of the copy per Session operation) had no caller left.


def _append(mut dest: List[Int], src: List[Int]):
    for k in range(len(src)):  # driver-loop: over operation metadata
        dest.append(src[k])
