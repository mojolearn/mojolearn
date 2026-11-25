# SPDX-License-Identifier: Apache-2.0
"""THE OPERATION SESSION: the tape's forward operations, one call each, for
composing new inference and hybrid algorithms (lane ag-ops-api, 2026-10-19).

An `OpsSession` owns one executor and one `Tape` and a table of live
tensors by integer handle. Every operation is a tape method (or one of the
tape's losses, layers, data steps or metrics) on the session's executor, so
a composition built here runs the kernels the trainers run and gives the
same bits on Apple, NVIDIA, AMD and the host. In inference mode (the
default) the tape keeps no graph; with recording on, the same composition is
differentiable: `backward` then `gradient_into` per leaf.

Handles: `(generation << 24) | (slot + 1)`, so a freed handle never names
the tensor that later takes its slot. Storage: a computed tensor's values
live in tape storage until `reset` or `close`; an uploaded tensor owns its
buffer, which `reset` keeps (re-registered as a fresh leaf of the next
graph) and `free` returns at the next `reset` or `close` (a recorded graph
may still read it before then).

Sessions live in a process-wide table (`std.ffi._Global`, one per numeric
tier, as `sequence/seq_tensor.mojo`); each entry records its executor kind,
so a host-binding call never reaches a device session. No new kernels: the
operations launch the tape's, the losses', the metrics' and the trainers'
existing opcodes (autograd/ops.mojo, autograd/loss_ops.mojo,
autograd/data_ops.mojo, sequence/ops.mojo OP_FILL / OP_COPY /
OP_GATHER_ROWS / OP_MLP_PERM).

Optimizers (lane ag-session-optim, 2026-10-19): `optimizer` makes an
optimizer of the session (SGD with momentum and Nesterov, Adam, AdamW, with
weight decay and Sequential.compile's learning-rate schedules); `step`
updates trainable uploaded tensors IN PLACE on the executor from the last
backward's gradients, one OP_OPT launch per parameter (autograd/optim.mojo
step_lr, the element body and the host scalars every sequence trainer
runs), with the moment buffers and the step count held in the session, per
(optimizer, parameter), allocated at the parameter's first step. The update
is elementwise and its host scalars depend on the step count and the
learning rate alone, so a parameter stepped here gets the words
Sequential.fit's one flat launch gives it (sequence/recurrent.mojo
opt_step: the same OP_OPT arguments, opt_scalars word for word).

Higher order (lane ag-ho-session, plan AUTOGRAD_HIGHER_ORDER (git history)
section 5): `backward_graph` records the backward pass itself
(autograd/higher.mojo backward_scalar_recorded), so `grad_handle` returns a
gradient as a session tensor that later operations and a second backward
can use; `backward_seed`, `vjp`, `hvp` and `jvp` are built from it with tape
operations only (no new kernels).

Captured programs (lane ag-program, autograd/program.mojo): an inference
session logs its operation calls; `capture(inputs, outputs)` keeps the
logged operations the outputs depend on as a program, and `run_program`
replays it in one call (inputs uploaded, every node through `_op_run` with
its recorded arguments on a scratch tape, parameters read in place,
outputs downloaded). No new kernels.

Index tensors (lane ag-program-embedding): `upload_ids` makes a session
tensor of int32 words (the caller's integer array cast on the executor by
upload_cast_i32, the cast the embedding's and take's caller-array form
runs); `embedding` and `take` take such a tensor as their second input, so
a captured program records the ids as a value (an input of kind
PROGRAM_IN_IDS, uploaded through the same cast at each run) instead of an
address. The eager words are the caller-array form's: the same int32 words
reach the same Tape.embedding and rankn_take kernels."""
from std.memory import bitcast
from sequence.exec_trait import Exec
from sequence.ops import (
    FP, Args, OP_FILL, OP_COPY, OPT_SGD, OPT_ADAM, OPT_ADAMW, OPT_LION, OPT_ADAMAX, OPT_NADAM, OPT_RMSPROP,
    OPT_ADAGRAD,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, identical_div
from autograd.tape import Tape, Tensor, IdsHandle
from autograd.ops import RELU, SIGMOID, TANH, SOFTMAX, ADD, SUBTRACT, MULTIPLY, DIVIDE, AXIS_ALL, AXIS_ROWS, AXIS_COLS
from autograd.layers import NeuralRules, layer_norm
from autograd.transformer_primitives import graph_rms_norm, graph_rotary, graph_transformer_linear, graph_transformer_softmax
from autograd.higher import backward_recorded, backward_scalar_recorded
from autograd.losses import mse, mae, huber, binary_cross_entropy_with_logits, cross_entropy
from autograd.data import permutation, gather_rows
from autograd.metrics import accuracy, accuracy_from_scores, log_loss, roc_auc
from core.cast_in import CastIn, cast_in_kind
from autograd.cast_ops import upload_cast, upload_cast_i32
from autograd.program import (
    OpsLog, OpsProgram, PROGRAM_IN_FLOAT, PROGRAM_IN_IDS, PROGRAM_PARAM_WEIGHT, PROGRAM_PARAM_STATE,
    PROGRAM_PARAM_STATE_WORDS, program_refuses_node, same_shape, shape_text,
)
from autograd.program_tools import ProgramTrace, program_op_name
from autograd.program_inference_metadata import PreparedInferenceReplay
from std.time import perf_counter_ns
from autograd.optim import (
    OptimizerConfig, OptimizerState, optimizer_config_packed, optimizer_slots, step_lr, optimizer_clips,
    clip_norm_accumulate, step_lr_clipped,
)
from autograd.optimizer_requests import OpsOptimizerExtras, ops_optimizer_extras_default
from autograd.optimizer_requests import (
    OPS_OPT_INTS, OPS_OPT_FLOATS, OpsOptimizerRequest, OpsOptimizerStateValue, OpsOptimizerRestoreRequest,
    ops_optimizer_request, ops_optimizer_restore_request,
)
from autograd.model import SCHED_CONSTANT, SCHED_STEP, SCHED_EXPONENTIAL, epoch_lr
# nn-blocks-pg begin: optimizer parameter groups (autograd/nn_pg.mojo)
from autograd.nn_pg import (
    ParamGroupTable, pg_check_membership, PG_SABOTAGE_CLIP_PER_GROUP, PG_SABOTAGE_CLIP_PER_GROUP_HOST,
    PG_SABOTAGE_GROUP_LR, PG_SABOTAGE_GROUP_LR_HOST,
)
# nn-blocks-pg end
# Rank-N tensors (lane ag-rankn-python, plan AUTOGRAD_RANKN (git history)): the
# core's shape metadata and views, the fold / bmm / layout / conv lanes'
# entries, and the shared glue (autograd/rankn_api.mojo).
from autograd.rankn import RN_MAX_RANK, ShapeN, rn_normalize_axis
from autograd.rankn_fold import rankn_binary, rankn_sum, rankn_max
from autograd.rankn_bmm import rankn_matmul
from autograd.rankn_layout import rankn_concat, rankn_narrow, rankn_take
from autograd.rankn_conv import rankn_conv2d, rankn_max_pool2d, rankn_avg_pool2d
from autograd.rankn_api import (
    RN_CONCAT_MAX, RankNRules, rn_as_shape, rn_axis_mask, rn_check_permutation, rn_dims,
    rn_mask_axes, rn_reduced, rn_resolve, rn_shape, rn_softmax_along,
)
# The import elementwise family (lane ag-import-kernels, docs/plans/
# MODEL_IMPORT.md 3.3): session codes 48..76 below, kernels in
# autograd/import_ops.mojo, tape calls and ImportRules in autograd/import_api.mojo.
from autograd.import_ops import (
    IMP_ERF, IMP_GELU_ERF, IMP_GELU_TANH, IMP_SOFTPLUS, IMP_SILU, IMP_LEAKY_RELU, IMP_ELU,
    IMP_HARD_SIGMOID, IMP_HARD_SWISH, IMP_CLIP, IMP_ABS, IMP_FLOOR, IMP_CEIL, IMP_ROUND, IMP_SIN,
    IMP_COS, IMP_SIGN, IMP_RECIPROCAL, IMP_NOT, IMP_POW_SCALAR, IMP_POW, IMP_MINIMUM, IMP_MAXIMUM,
    IMP_EQ, IMP_AND, IMP_OR, IMP_BOOL,
)
from autograd.import_api import (
    ImportRules, imp_unary, imp_binary, imp_where, imp_expand, imp_argmax, imp_pad, imp_cumsum, imp_gather_nd,
)
# Block step ops (lane x1-executor-state-and-step-ops): session codes 81..84,
# the resident decode sessions' step bodies over session tensors.
from sequence.exec_device import DeviceExec
from autograd.block_step_ops import (
    OPS_M1_STEP, OPS_M2_STEP, OPS_M3_STEP, OPS_ATTN_STEP, BlockStepScratch, is_block_step, block_step_check,
    ATTN_STEP_TAG, attn_step_train_run, step_train_tag, mamba_step_train_run,
    block_step_run, block_weight_sizes, block_state_sizes, block_offsets, block_guard, block_int_words,
    block_part_floats, BLOCK_PART_WEIGHTS, BLOCK_PART_STATE, BLOCK_PART_WORDS, attn_append_restore,
)
from autograd.rules_step_ops import StepOpRules
from autograd.rules_recurrent import OPS_RNN_SCAN, rnn_scan_op
# Lane rnn-parity-B: the weighted per-step losses (session code 95) and the
# recurrent layer's padded-step zeroing (autograd/per_step_ops.mojo).
from autograd.per_step_ops import OPS_WEIGHTED_LOSS, weighted_loss, rnn_zero_padded
from sequence.ops import CELL_LSTM
from sequence.recurrent import fill as _seq_fill
# Generation ops (lane x45c): the language model's token pick and the decode
# loop's condition, session codes 85, 86 (autograd/generation_ops.mojo).
from autograd.lm_head_v2_op import OPS_LM_HEAD_V2
from autograd.lm_head_v2_graph import graph_lm_head_v2
from autograd.small_mlp_op import (
    OPS_SMALL_MLP, SMALL_MLP_TAG, small_mlp_check, small_mlp_workspace_floats, small_mlp_forward_device,
)
# CNNClassifier's tuned conv block and head (lane r1-restore-measured-winners):
# session codes 90, 91 (autograd/cnn_block_op.mojo).
from autograd.cnn_block_op import (
    OPS_CNN_CONV_BLOCK, OPS_CNN_LINEAR, CNN_CONV_BLOCK_TAG, CNN_LINEAR_TAG, CNN_HOST_BLOCK_FORWARD,
    CNN_HOST_LINEAR_FORWARD, cnn_block_words, cnn_block_out_dims, cnn_block_saved_sizes, cnn_block_keep,
    cnn_block_forward_device, cnn_linear_forward_device,
)
# Lane r1-restore-measured-winners (2026-11-09): the byte LM trainer's
# embedding backward as an op (autograd/lm_embedding_op.mojo), code 93.
from autograd.lm_embedding_op import OPS_LM_EMBEDDING, LM_EMBEDDING_TAG, lm_embedding_scratch_words
from autograd.embedding_ops import (
    AG_EMB_PLAN_SORT, embedding_refuse_shape, embedding_check_words, embedding_refuse_ids, embedding_forward_into,
    embedding_perm_words,
)
from autograd.generation_ops import (
    OPS_LM_PICK, OPS_LM_ALIVE, is_generation_op, generation_op_check, generation_op_name, lm_pick_run,
    lm_alive_run, LM_PICK_SAMPLE,
)
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
# nn-blocks-cbw begin: user-defined backward rules (autograd/nn_cbw.mojo)
from autograd.tape import BackwardRules
from autograd.nn_cbw import (
    OPS_CUSTOM_VJP, NN_CBW_TAG, NN_CBW_SABOTAGE_DROP_PATH, NN_CBW_SABOTAGE_SEED_ONE, CbwOp, CbwRecord, CbwState,
    cbw_bind, cbw_from_words,
)
# nn-blocks-cbw end
# session-ops begin: Sequential's layers as Session ops, codes OPS_LAYER_FIRST..OPS_LAYER_LAST, and the rules that
# differentiate their nodes in a session (autograd/session_layer_rules.mojo)
from autograd.session_layer_rules import (
    LayerRules, OPS_LAYER_FIRST, OPS_LAYER_LAST, OPS_LAYER_POOL_FIRST, OPS_LAYER_RESIZE_FIRST, OPS_LAYER_REST_FIRST,
    OPS_LAYER_EXTRA_FIRST,
)
# session-ops-extra begin: sequence_mask, topk, gather along an axis, logsumexp, soft-target cross-entropy
from autograd.session_ops_extra import (
    OPS_SEQUENCE_MASK, OPS_TOPK_INDICES, OPS_GATHER_AXIS, OPS_LOGSUMEXP, OPS_SOFT_CROSS_ENTROPY,
    OPS_SOFT_CROSS_ENTROPY_WEIGHTED, session_ops_extra_int_result, sox_sequence_mask, sox_topk_indices, sox_gather,
    sox_logsumexp, sox_soft_cross_entropy,
)
# session-ops-extra end
# session-ops end
# session-ops-resize begin: interpolate, pixel_shuffle / unshuffle, pad with a mode (codes 150..153)
from autograd.session_ops_resize import (
    OPS_INTERPOLATE, OPS_PIXEL_SHUFFLE, OPS_PIXEL_UNSHUFFLE, OPS_PAD_MODE,
    session_interpolate, session_pixel_shuffle, session_pad_mode,
)
# session-ops-resize end
# session-ops-conv begin: conv1d, conv2d with dilation / per-axis words, conv3d, conv_transpose1d / 2d (130..134)
from autograd.session_ops_conv import session_conv_forward, session_conv_name
# session-ops-conv end
# session-ops-pool begin: Sequential's pools as Session ops, codes 140..148 (autograd/session_ops_pool.mojo)
from autograd.session_ops_pool import session_pool, pool_int_count, pool_op_name, OPS_POOL_LAST
# session-ops-pool end
# nn-blocks-qkv begin: attention over explicit q, k, v (session code 122, autograd/nn_qkv.mojo)
from autograd.nn_qkv import OPS_ATTENTION_QKV, qkv_attention
# nn-blocks-qkv end
# nn-blocks-scatter begin: scatter_add and the segment reductions, session codes 113..116 (autograd/nn_scatter.mojo)
from autograd.nn_scatter import (
    OPS_SCATTER_ADD, OPS_SEGMENT_SUM, OPS_SEGMENT_MEAN, OPS_SEGMENT_MAX, nn_scatter_add, nn_segment, segment_op_name,
)
# nn-blocks-scatter end
# nn-blocks-scan begin: the public scan over a captured cell, session code 118 (autograd/nn_scan.mojo)
from autograd.nn_scan import (
    OPS_SCAN, ScanSpec, scan_decode, scan_check_binding, SCAN_SABOTAGE_STATE_LAG, SCAN_SABOTAGE_STATE_LAG_HOST,
    SCAN_SABOTAGE_INIT_GRAD, SCAN_SABOTAGE_INIT_GRAD_HOST, SCAN_SABOTAGE_DROP_KEY, SCAN_SABOTAGE_DROP_KEY_HOST,
    scan_dropout_seed,
)
# nn-blocks-scan end
# nn-blocks-composed begin: the composed tagger's sabotage switches only (autograd/nn_composed.mojo)
from autograd.nn_composed import (
    COMPOSED_SABOTAGE_SCAN_KEY, COMPOSED_SABOTAGE_SCAN_KEY_HOST, COMPOSED_SABOTAGE_GROUP_STATE,
    COMPOSED_SABOTAGE_GROUP_STATE_HOST,
)
# nn-blocks-composed end
# session-ops-rest begin: norms, prelu, mish, multi-head attention, positional encoding, losses and raw SSM scans as
# Session ops, codes 160..174 (autograd/session_ops_rest.mojo)
from autograd.session_ops_rest import (
    SOR_FIRST, SOR_LAST, SOR_BATCH_NORM, SOR_GROUP_NORM, SOR_INSTANCE_NORM, SOR_PRELU, SOR_MISH, SOR_MHA,
    SOR_POS_ENC, SOR_SMOOTH_L1, SOR_NLL, SOR_KL_DIV, SOR_BCE_POS_WEIGHT, SOR_CTC, SOR_SELECTIVE_SCAN, SOR_SSD_SCAN,
    SOR_TRAPEZOID_SCAN, sor_name, sor_batch_norm, sor_group_norm, sor_instance_norm, sor_prelu, sor_mish,
    sor_multi_head_attention, sor_positional_encoding, sor_smooth_l1_loss, sor_nll_loss, sor_kl_div,
    sor_bce_with_logits, sor_ctc_loss, sor_selective_scan, sor_ssd_scan, sor_trapezoid_scan,
)
# session-ops-rest end

# Executor kinds of a session (the binding that opened it).
# Lane ag-pass6-host (2026-10-19, PASS6_IDEAS idea G): the packed argument
# block of one operation call (OpsSession.op_packed), int64 words written by
# the Python shell into a session-owned buffer and read here by address, so
# a call crosses no Python list: word 0 the handle count, 1 the integer
# count, 2 the float count, then OPS_PACK_HANDLES handle words, then
# OPS_PACK_INTS integer words, then OPS_PACK_FLOATS float32 values in the
# words from OPS_PACK_F_WORD on (two a word, little-endian order as numpy's
# float32 view of the same buffer).
comptime OPS_PACK_HANDLES = 4
comptime OPS_PACK_INTS = 8
comptime OPS_PACK_FLOATS = 4
comptime OPS_PACK_H_WORD = 3
comptime OPS_PACK_I_WORD = OPS_PACK_H_WORD + OPS_PACK_HANDLES
comptime OPS_PACK_F_WORD = OPS_PACK_I_WORD + OPS_PACK_INTS
comptime OPS_PACK_WORDS = OPS_PACK_F_WORD + (OPS_PACK_FLOATS + 1) // 2

comptime OPS_DEVICE = 1
comptime OPS_HOST = 2

# Operation codes of ops_op (python/mojolearn/ops.py mirrors them).
comptime OPS_MATMUL = 1
comptime OPS_ADD = 2
comptime OPS_SUB = 3
comptime OPS_MUL = 4
comptime OPS_DIV = 5
comptime OPS_RELU = 6
comptime OPS_SIGMOID = 7
comptime OPS_TANH = 8
comptime OPS_EXP = 9
comptime OPS_LOG = 10
comptime OPS_SQRT = 11
comptime OPS_SCALE = 12
comptime OPS_TRANSPOSE = 13
comptime OPS_RESHAPE = 14
comptime OPS_SUM = 15
comptime OPS_MEAN = 16
comptime OPS_SOFTMAX = 17
comptime OPS_LOG_SOFTMAX = 18
comptime OPS_LINEAR = 19
comptime OPS_LAYER_NORM = 20
comptime OPS_RMS_NORM = 77
comptime OPS_ROTARY = 78
comptime OPS_TRANSFORMER_LINEAR = 79
comptime OPS_TRANSFORMER_SOFTMAX = 80
comptime OPS_DROPOUT = 21
comptime OPS_GATHER_ROWS = 22
comptime OPS_PERMUTATION = 23
comptime OPS_MSE = 24
comptime OPS_MAE = 25
comptime OPS_HUBER = 26
comptime OPS_BCE_LOGITS = 27
comptime OPS_CROSS_ENTROPY = 28
comptime OPS_DETACH = 29
comptime OPS_FULL = 30
comptime OPS_COPY = 31
comptime OPS_EMBEDDING = 32
# Rank-N operations (lane ag-rankn-python; plan section 5). Integer
# arguments longer than OPS_PACK_INTS travel by the list entry (ops_op).
comptime OPS_RESHAPE_N = 33      # [mode, ...]: 0 reshape dims (one -1), 1 flatten start end, 2 squeeze axes, 3 unsqueeze axis
comptime OPS_PERMUTE = 34        # [mode, ...]: 0 permute axes, 1 swap two axes
comptime OPS_SUM_N = 35          # [keepdims (-1 default, 0, 1), axes...]; no axes: every element
comptime OPS_MEAN_N = 36
comptime OPS_MAX = 37
comptime OPS_MATMUL_N = 38       # the batched product (OPS_MATMUL routes rank-N operands here too)
comptime OPS_CONCAT = 39         # [axis]; any number of tensors
comptime OPS_NARROW = 40         # [axis, piece, mode, ...]: mode 0 sizes..., mode 1 sections
comptime OPS_TAKE = 41           # [ids descriptor (7 words), axis]
comptime OPS_SOFTMAX_N = 42      # [axis]
comptime OPS_LOG_SOFTMAX_N = 43  # [axis]
comptime OPS_CONV2D = 44         # [stride, padding, groups (optional, 1)]; x, weight, optional bias
comptime OPS_MAXPOOL2D = 45      # [kernel, stride]
comptime OPS_AVGPOOL2D = 46      # [kernel, stride]
# 47 stays reserved (the rank-N plan held it for an argmax result; argmax is
# OPS_ARGMAX 72 of the import family below).
# The import family (lane ag-import-kernels, plan MODEL_IMPORT (git history) 3.3;
# 47 stays reserved). Unary maps keep x's shape; the binary kinds, where and
# the comparisons broadcast as NumPy does.
comptime OPS_ERF = 48
comptime OPS_GELU = 49           # [approximate: 0 erf, 1 tanh]
comptime OPS_SOFTPLUS = 50
comptime OPS_SILU = 51
comptime OPS_LEAKY_RELU = 52     # floats [alpha]
comptime OPS_ELU = 53            # floats [alpha]
comptime OPS_HARD_SIGMOID = 54   # floats [alpha, beta]
comptime OPS_HARD_SWISH = 55
comptime OPS_CLIP = 56           # floats [low, high]
comptime OPS_ABS = 57
comptime OPS_FLOOR = 58
comptime OPS_CEIL = 59
comptime OPS_ROUND = 60          # half to even
comptime OPS_SIN = 61
comptime OPS_COS = 62
comptime OPS_SIGN = 63
comptime OPS_RECIPROCAL = 64
comptime OPS_POW = 65            # two tensors, or one tensor and floats [exponent]
comptime OPS_MINIMUM = 66
comptime OPS_MAXIMUM = 67
comptime OPS_WHERE = 68          # tensors (condition, x, y)
comptime OPS_COMPARE = 69        # [kind: 0 eq, 1 gt, 2 lt, 3 ge, 4 le, 5 ne]
comptime OPS_LOGICAL = 70        # [kind: 0 not (one tensor), 1 and, 2 or]
comptime OPS_EXPAND = 71         # [dims...]
comptime OPS_ARGMAX = 72         # [axis, keepdims (0, 1), form: 0 int32 words, 1 float values]
comptime OPS_PAD = 73            # [before 0 .. before n-1, after 0 .. after n-1], floats [value]
# lane ag-import-lm-ops: the attention-mask ops of transformer exports
comptime OPS_ANY_ALL = 74        # [kind: 0 any, 1 all, keepdims (-1, 0, 1), axes...]; no axes: every element
comptime OPS_CUMSUM = 75         # [axis, exclusive (0, 1), reverse (0, 1)]
comptime OPS_INDEX = 76          # tensors (x, index 0, index 1, ...); [axis of index 0, axis of index 1, ...]
comptime OPS_IMP_FIRST = OPS_ERF
comptime OPS_IMP_LAST = OPS_INDEX
# 81..84: the block step ops, autograd/block_step_ops.mojo (OPS_M1_STEP,
# OPS_M2_STEP, OPS_M3_STEP, OPS_ATTN_STEP), dispatched by their own table.

# What a session-owned buffer holds (OpsSession.buffer_kind, by address): an
# upload, carried float state, or carried int32-word state
# (`declare_state`). Block step ops write state buffers in place.
comptime OPS_BUFFER_UPLOAD = 1
comptime OPS_BUFFER_STATE = 2
comptime OPS_BUFFER_STATE_WORDS = 3

# Metric codes of ops_metric.
comptime OPS_METRIC_ACCURACY = 1
comptime OPS_METRIC_LOG_LOSS = 2
comptime OPS_METRIC_ROC_AUC = 3

# The packed optimizer options of `optimizer`.

comptime _SLOT_BITS = 24
comptime _SLOT_MASK = (1 << _SLOT_BITS) - 1


def _activation(code: Int) raises -> Int:
    """The tape activation of a linear op's code: 0 none, 1 relu, 2 sigmoid,
    3 tanh, 4 softmax."""
    if code == 0:
        return 0
    if code == 1:
        return RELU
    if code == 2:
        return SIGMOID
    if code == 3:
        return TANH
    if code == 4:
        return SOFTMAX
    raise Error("ops: linear activation must be 0 (none), 1 (relu), 2 (sigmoid), 3 (tanh) or 4 (softmax)")


def _need(what: String, have: Int, inputs: Int, ints: Int, have_ints: Int, floats: Int, have_floats: Int) raises:
    if have != inputs or have_ints < ints or have_floats < floats:
        raise Error("ops: " + what + " takes " + String(inputs) + " tensors, " + String(ints)
                    + " integers and " + String(floats) + " floats")


def _seed(lo: Int, hi: Int) raises -> UInt64:
    if lo < 0 or hi < 0 or lo > 0xFFFFFFFF or hi > 0xFFFFFFFF:
        raise Error("ops: the seed words must be 32-bit non-negative values")
    return (UInt64(hi) << 32) | UInt64(lo)


@fieldwise_init
struct _OpsOptimizer(ImplicitlyCopyable, Movable):
    """One optimizer of a session: its OP_OPT configuration (packed as
    Sequential.compile packs it), its learning-rate schedule
    (autograd/model.mojo SCHED_*) and the schedule's epoch. `pg`: -1 for an
    ordinary optimizer, else the index of its parameter-group table
    (`OpsSession.pg_tables`, autograd/nn_pg.mojo; lane nn-blocks-pg)."""
    var cfg: OptimizerConfig
    var sched: Int
    var sched_step: Int
    var gamma: Float32
    var epoch: Int
    var pg: Int


@fieldwise_init
struct _OpsOptState(ImplicitlyCopyable, Movable):
    """The state of one (optimizer, parameter) pair: the executor buffers
    of an autograd/optim.mojo OptimizerState (owned by the session), the
    parameter's step count and the running host scalars."""
    var opt: Int
    var param: Int
    var s1: FP
    var s2: FP
    var s3: FP
    var parts: FP
    var cell: FP
    var n: Int
    var block: Int
    var nparts: Int
    var t: Int
    var pw1: Float32
    var pw2: Float32
    var mu_prod: Float32


struct OpsSession[E: Exec & Movable & Deinitable](Movable):
    """One executor, one tape and the live tensors of a composition."""
    var ex: Self.E
    var tape: Tape
    var tensors: List[Tensor]
    var live: List[Bool]
    var generation: List[Int]
    var rows: List[Int]
    var cols: List[Int]
    #: the full shape of each tensor (rows, cols above are its matrix view)
    var shapes: List[ShapeN]
    #: address of a session-owned buffer (an uploaded tensor), 0 otherwise
    var owned: List[Int]
    var wants_grad: List[Bool]
    #: a permutation output: the row count its indices address (else 0)
    var index_rows: List[Int]
    #: an upload_ids tensor: its storage holds int32 words (embedding and
    #: take ids), so a program naming it as an input casts to int32 words
    var int_words: List[Bool]
    var free_slots: List[Int]
    #: owned buffers of freed handles, returned at the next reset or close
    var retired: List[Int]
    #: the session's optimizers (id = index + 1) and their per-parameter state
    var opts: List[_OpsOptimizer]
    var opt_states: List[_OpsOptState]
    #: native (optimizer, generation-bearing handle) -> retained state index
    var opt_state_indices: List[Dict[Int, Int]]
    #: lane ag-program (autograd/program.mojo): the operation calls of an
    #: inference session since its last reset, the captured programs (id =
    #: index + 1) and the scratch tape their replays run on (its storage
    #: reused run after run, returned at close)
    var oplog: OpsLog
    var programs: List[OpsProgram]
    #: Immutable program metadata is prepared lazily, then owned until free/close.
    #: A one-slot owner transfers the plan into replay and receives it on exit.
    var program_replays: List[List[PreparedInferenceReplay]]
    var prog_tape: Tape
    var program_trace: ProgramTrace
    #: Reusable operand capacity for synchronous native operation dispatch.
    var op_tensors: List[Tensor]
    #: lane x1: OPS_BUFFER_* of every live session-owned buffer, by address
    #: (a replay's parameter leaves name the same storage, so state is a
    #: property of the buffer, not of a handle)
    var buffer_kind: Dict[Int, Int]
    #: lane x1: the resident block structs of this session's step ops
    var steps: BlockStepScratch
    #: lane x45c: append-only state, by address: a full-causal KV cache
    #: (window 0) -> the address of the position word that addresses it.
    #: A write past the committed position lands in slots no committed
    #: position reads, so a bounded loop shadows only the position word
    #: (autograd/program_control.mojo), never the cache itself.
    var append_state: Dict[Int, Int]
    # nn-blocks-cbw begin: the bound custom operations and the custom nodes
    # of the recorded graph (autograd/nn_cbw.mojo)
    var cbw: CbwState
    # nn-blocks-cbw end
    #: lane nn-blocks-pg: the parameter-group tables of the grouped
    #: optimizers (autograd/nn_pg.mojo), indexed by `_OpsOptimizer.pg`
    var pg_tables: List[ParamGroupTable]
    #: lane session-ops: True once a Sequential layer op (OPS_LAYER_FIRST..OPS_LAYER_LAST) recorded a node since
    #: the last reset; each backward entry tests it once per call and then differentiates with
    #: LayerRules[its usual rules] (autograd/session_layer_rules.mojo). A session that never uses one keeps
    #: exactly its old rules.
    var layer_nodes: Bool

    def __init__(out self, var ex: Self.E, record: Bool):
        self.ex = ex^
        self.tape = Tape()
        if not record:
            self.tape.no_grad()
        self.tensors = List[Tensor]()
        self.live = List[Bool]()
        self.generation = List[Int]()
        self.rows = List[Int]()
        self.cols = List[Int]()
        self.shapes = List[ShapeN]()
        self.owned = List[Int]()
        self.wants_grad = List[Bool]()
        self.index_rows = List[Int]()
        self.int_words = List[Bool]()
        self.free_slots = List[Int]()
        self.retired = List[Int]()
        self.opts = List[_OpsOptimizer]()
        self.opt_states = List[_OpsOptState]()
        self.opt_state_indices = List[Dict[Int, Int]]()
        self.oplog = OpsLog()
        self.programs = List[OpsProgram]()
        self.program_replays = List[List[PreparedInferenceReplay]]()
        self.prog_tape = Tape()
        self.prog_tape.no_grad()
        self.program_trace = ProgramTrace()
        self.op_tensors = List[Tensor]()
        self.buffer_kind = Dict[Int, Int]()
        self.steps = BlockStepScratch()
        self.append_state = Dict[Int, Int]()
        self.cbw = CbwState()  # nn-blocks-cbw
        self.pg_tables = List[ParamGroupTable]()
        self.layer_nodes = False  # session-ops

    # ------------------------------------------------------------ handles
    def _slot(self, h: Int) raises -> Int:
        var slot = (h & _SLOT_MASK) - 1
        var gen = h >> _SLOT_BITS
        if h <= 0 or slot < 0 or slot >= len(self.live) or not self.live[slot] or self.generation[slot] != gen:
            raise Error("ValueError: ops: handle " + String(h) + " is not a live tensor of this session")
        return slot

    def _add(mut self, t: Tensor, owned: Int, wants_grad: Bool, index_rows: Int) raises -> Int:
        var r = self.tape.rows(t)
        var c = self.tape.cols(t)
        var sh = self.tape.shape_of(t)
        var slot: Int
        if len(self.free_slots) > 0:
            slot = self.free_slots.pop()
            self.tensors[slot] = t
            self.live[slot] = True
            self.rows[slot] = r
            self.cols[slot] = c
            self.shapes[slot] = sh
            self.owned[slot] = owned
            self.wants_grad[slot] = wants_grad
            self.index_rows[slot] = index_rows
            self.int_words[slot] = False
        else:
            slot = len(self.live)
            if slot >= _SLOT_MASK:
                raise Error("ops: too many live tensors in one session")
            self.tensors.append(t)
            self.live.append(True)
            self.generation.append(0)
            self.rows.append(r)
            self.cols.append(c)
            self.shapes.append(sh)
            self.owned.append(owned)
            self.wants_grad.append(wants_grad)
            self.index_rows.append(index_rows)
            self.int_words.append(False)
        if owned != 0:
            self.buffer_kind[owned] = OPS_BUFFER_UPLOAD
        return (self.generation[slot] << _SLOT_BITS) | (slot + 1)

    def _retire(mut self, slot: Int):
        if self.owned[slot] != 0:
            # a freed parameter's optimizer state goes with it (its buffers
            # are returned with the freed uploads, after the queued updates)
            self._drop_states((self.generation[slot] << _SLOT_BITS) | (slot + 1))
        self.live[slot] = False
        self.generation[slot] += 1
        if self.owned[slot] != 0:
            if self.owned[slot] in self.buffer_kind:
                try:
                    _ = self.buffer_kind.pop(self.owned[slot])
                except:
                    pass
            if self.owned[slot] in self.append_state:
                try:
                    _ = self.append_state.pop(self.owned[slot])
                except:
                    pass
            self.steps_forget(self.owned[slot])
            self.retired.append(self.owned[slot])
            self.owned[slot] = 0
        self.free_slots.append(slot)

    def tensor(self, h: Int) raises -> Tensor:
        return self.tensors[self._slot(h)]

    def shape(self, h: Int) raises -> Tuple[Int, Int]:
        var slot = self._slot(h)
        return (self.rows[slot], self.cols[slot])

    def shape_nd(self, h: Int) raises -> ShapeN:
        """The full shape of tensor h."""
        return self.shapes[self._slot(h)]

    def recording(self) -> Bool:
        return self.tape.is_recording()

    # ------------------------------------------------------------ data in and out
    def upload(mut self, src: CastIn, requires_grad: Bool) raises -> Int:
        """A new (src.rows, src.cols) tensor holding the caller's array as
        float32 (core/cast_in.mojo: any accepted dtype, C or Fortran order,
        cast on the executor by autograd/cast_ops.mojo upload_cast); the
        caller's memory may change once this returns."""
        src.check("ops: tensor")
        var rows = src.rows
        var cols = src.cols
        if rows <= 0 or cols <= 0 or rows > ((1 << 60) // 4) // cols:
            raise Error("ops: tensor dimensions must be positive")
        if requires_grad and not self.tape.is_recording():
            raise Error("ops: requires_grad needs a recording session (Session(record=True))")
        var n = rows * cols
        var buf = self.ex.alloc(n)
        upload_cast(self.ex, buf, src)
        var t = self.tape.variable(self.ex, buf, rows, cols, requires_grad)
        return self._add(t, Int(buf), requires_grad, 0)

    def upload_nd(mut self, src: CastIn, shape: ShapeN, requires_grad: Bool) raises -> Int:
        """`upload` of a C-order array of any rank: `src` is its row-major
        (rows, cols) matrix view and `shape` its full shape (the same
        element count). A rank-2 shape is `upload` itself."""
        if shape.ndim == 2:
            if shape.d[0] != src.rows or shape.d[1] != src.cols:
                raise Error("ops: tensor: the shape does not match the array")
            return self.upload(src, requires_grad)
        src.check("ops: tensor")
        var rows = src.rows
        var cols = src.cols
        if rows <= 0 or cols <= 0 or rows > ((1 << 60) // 4) // cols:
            raise Error("ops: tensor dimensions must be positive")
        if shape.count() != rows * cols:
            raise Error("ops: tensor: the shape does not match the array")
        if requires_grad and not self.tape.is_recording():
            raise Error("ops: requires_grad needs a recording session (Session(record=True))")
        var buf = self.ex.alloc(rows * cols)
        upload_cast(self.ex, buf, src)
        var t = self.tape.variable_nd(self.ex, buf, shape, requires_grad)
        return self._add(t, Int(buf), requires_grad, 0)

    def upload_ids(mut self, src: CastIn, shape: ShapeN) raises -> Int:
        """A new tensor of `shape` holding the caller's integer array as
        int32 words (any integer dtype, cast on the executor by
        upload_cast_i32; values outside int32 saturate and the embedding's
        and take's id checks refuse them): the ids of `embedding` and `take`
        as a tensor, which a captured program records as a value. `src` is
        the array's row-major (rows, cols) matrix view; `shape` has its
        element count. The tensor takes no gradient."""
        src.check("ops: ids")
        var rows = src.rows
        var cols = src.cols
        if rows <= 0 or cols <= 0 or rows > ((1 << 60) // 4) // cols:
            raise Error("ops: ids dimensions must be positive")
        if shape.count() != rows * cols:
            raise Error("ops: ids: the shape does not match the array")
        var buf = self.ex.alloc(rows * cols)
        upload_cast_i32(self.ex, buf, src, "ops: ids")
        var t: Tensor
        if shape.ndim == 2:
            t = self.tape.variable(self.ex, buf, shape.d[0], shape.d[1], False)
        else:
            t = self.tape.variable_nd(self.ex, buf, shape, False)
        var h = self._add(t, Int(buf), False, 0)
        self.int_words[self._slot(h)] = True
        return h

    def _ids_of(mut self, t: Tensor) raises -> IdsHandle:
        """The int32 words of tensor t (an upload_ids tensor, a program's
        ids input or an int32 argmax) as the tape's ids, read in place."""
        var n = self.tape.shape_of(t).count()
        var data = self.tape.value_of(self.ex, t)
        return self.tape.ids(self.ex, data, n)

    def load_into(mut self, h: Int, src: CastIn) raises:
        """Overwrite the values of uploaded tensor h with the caller's
        array (any accepted dtype and strides, bfloat16 included, cast on
        the executor by upload_cast): the weight import's write
        (autograd/weight_import.mojo). The array must hold the tensor's
        element count, row-major; a computed tensor is refused (its storage
        belongs to the tape)."""
        var slot = self._slot(h)
        if self.owned[slot] == 0:
            raise Error("ops: load_weights writes tensors made with tensor(); handle " + String(h)
                        + " is a computed tensor")
        src.check("ops: load_weights")
        if src.count() != self.rows[slot] * self.cols[slot]:
            raise Error("ops: load_weights: the source holds " + String(src.count())
                        + " values, the tensor " + String(self.rows[slot] * self.cols[slot]))
        if self.int_words[slot]:
            # an int32-word tensor (ids, integer state) keeps int32 words
            upload_cast_i32(self.ex, FP(unsafe_from_address=self.owned[slot]), src, "ops: load_weights")
        else:
            upload_cast(self.ex, FP(unsafe_from_address=self.owned[slot]), src)
        # a block step op's resident struct over this buffer is stale now
        self.steps_forget(self.owned[slot])

    def declare_state(mut self, h: Int) raises:
        """Make uploaded tensor h carried state: block step ops (and any
        `*_into` writer) update it in place, and a captured program records
        it as a state parameter. A tensor of int32 words (`upload_ids`)
        becomes integer state (a KV position, a buffer length, token ids).
        Nothing differentiates state, so it takes no gradient."""
        var slot = self._slot(h)
        if self.owned[slot] == 0:
            raise Error("ValueError: ops: state takes a tensor made with tensor() or ids(); handle "
                        + String(h) + " is a computed tensor")
        if self.wants_grad[slot]:
            raise Error("ValueError: ops: state tensors take no gradient (requires_grad=False)")
        self.buffer_kind[self.owned[slot]] = OPS_BUFFER_STATE_WORDS if self.int_words[slot] else OPS_BUFFER_STATE
        self.steps_forget(self.owned[slot])

    def is_state(self, h: Int) raises -> Bool:
        """Whether tensor h is carried state (`declare_state`)."""
        var slot = self._slot(h)
        var addr = self.owned[slot]
        if addr == 0 or addr not in self.buffer_kind:
            return False
        try:
            return self.buffer_kind[addr] != OPS_BUFFER_UPLOAD
        except:
            return False

    def declare_append(mut self, kv: Int, words: Int) raises:
        """Lane x45c: state tensor `kv` is written only at the position the
        int32 state word tensor `words` holds, and read only below it (a
        full-causal attention cache): a bounded loop then restores the
        position word alone when iterations ran past its stop."""
        if not self.is_state(kv) or not self.is_state(words):
            raise Error("ValueError: ops: declare_append takes two state tensors")
        var ws = self._slot(words)
        if not self.int_words[ws]:
            raise Error("ValueError: ops: declare_append: the position must be an int32 state word tensor")
        self.append_state[self.owned[self._slot(kv)]] = self.owned[ws]

    def append_words_of(self, addr: Int) -> Int:
        """The position word address of append-only state at `addr`, 0 for
        any other buffer."""
        if addr not in self.append_state:
            return 0
        try:
            return self.append_state[addr]
        except:
            return 0

    def _buffer_kind_of(self, addr: Int) -> Int:
        """OPS_BUFFER_* of the session-owned buffer at addr, 0 for tape storage."""
        if addr not in self.buffer_kind:
            return 0
        try:
            return self.buffer_kind[addr]
        except:
            return 0

    def steps_address(mut self) -> Int:
        """Address of this session's resident step entries (lane x7): the
        backward rules of a program that trains through a step op read
        them (autograd/rules_step_ops.mojo StepOpRules). Stable for the
        session's life (sessions are heap-allocated, never moved)."""
        return Int(MutPointer(to=self.steps))

    def steps_forget(mut self, addr: Int):
        """The buffer at `addr` was written from outside a step or freed:
        the step ops' entries over it go (the device table's here, the host
        arm's in the executor, lane x7b)."""
        self.steps.forget(addr)
        self.ex.step_forget(addr)

    def steps_append_restore(mut self, kvs: List[Int], words: List[Int]) raises:
        """Lane fx-kv-inplace: a bounded loop ran bodies past its stop over
        the append-only state buffers at `kvs` (`declare_append`; their
        position words at `words`, restored by the loop's gate): take each
        cache back to its restored word, so the state does not depend on the
        loop's chunk (the device table's `attn_append_restore`, the host
        arm's `Exec.step_append_restore`). Called before `steps_forget`
        drops the entries, which hold how far the extra bodies wrote."""
        comptime if Self.E.IS_DEVICE:
            attn_append_restore(rebind[DeviceExec](self.ex), self.steps, kvs, words)
        else:
            self.ex.step_append_restore(kvs, words)

    def steps_weights_changed(mut self, addr: Int):
        """The weights at `addr` were rewritten in place (both arms)."""
        self.steps.weights_changed(addr)
        self.ex.step_weights_changed(addr)

    def steps_clear(mut self):
        self.steps.clear()
        self.ex.step_clear()

    def upload_block(mut self, code: Int, ints: List[Int], part: Int, srcs: List[CastIn],
                     words: List[Int]) raises -> Int:
        """`upload_block_grad` without a gradient: what every model program
        packs its blocks with."""
        return self.upload_block_grad(code, ints, part, srcs, words, False)

    def upload_block_grad(mut self, code: Int, ints: List[Int], part: Int, srcs: List[CastIn],
                          words: List[Int], requires_grad: Bool) raises -> Int:
        """One session tensor holding `part` of a block for step op `code`
        (autograd/block_step_ops.mojo): BLOCK_PART_WEIGHTS packs the
        weights `srcs` (one array per piece, `block_weight_sizes` order, an
        absent optional piece an empty descriptor of zero values) into one
        (1, n) upload; BLOCK_PART_STATE packs the float state pieces the
        same way and declares the tensor state; BLOCK_PART_WORDS makes the
        (1, k) int32-word state from `words`. Each piece is cast into its
        slot on the executor (upload_cast); the gaps between slots are zero.
        `requires_grad` (lane fx-ops-session-blocks, 2026-11-18: weights
        only, a recording session only) makes the packed weights a trainable
        leaf: a recorded step op (`_block_step`) is a custom node over
        [x, weights], so `backward` leaves the block's weight gradients in
        this tensor's gradient at the same `block_offsets` slots (the gaps
        zero) and `step` updates the packed weights in place."""
        if requires_grad:
            if part != BLOCK_PART_WEIGHTS:
                raise Error("ValueError: ops: upload_block: only a block's weights take a gradient")
            if not self.tape.is_recording():
                raise Error("ops: requires_grad needs a recording session (Session(record=True))")
        if part == BLOCK_PART_WORDS:
            var k = block_int_words(code)
            if k < 1 or len(words) != k:
                raise Error("ValueError: ops: upload_block: this block carries " + String(k) + " state words")
            var host = List[Float32](length=k, fill=Float32(0))
            for i in range(k):  # small-loop(k: at most two state words): word encode, no data
                var w = words[i]
                if w < -(1 << 31) or w >= (1 << 31):
                    raise Error("ValueError: ops: upload_block: a state word does not fit in 32 bits")
                host[i] = bitcast[DType.float32](Int32(w))
            var wbuf = self.ex.alloc(k)
            self.ex.upload(wbuf, FP(unsafe_from_address=Int(host.unsafe_ptr())), k)
            _ = len(host)  # keep-alive past the staging copy
            var wt = self.tape.variable(self.ex, wbuf, 1, k, False)
            var wh = self._add(wt, Int(wbuf), False, 0)
            var wslot = self._slot(wh)
            self.int_words[wslot] = True
            self.buffer_kind[Int(wbuf)] = OPS_BUFFER_STATE_WORDS
            return wh
        if part != BLOCK_PART_WEIGHTS and part != BLOCK_PART_STATE:
            raise Error("ValueError: ops: upload_block: part must be weights, state or words")
        var sizes = block_weight_sizes(code, ints) if part == BLOCK_PART_WEIGHTS else block_state_sizes(code, ints)
        if len(srcs) != len(sizes):
            raise Error("ValueError: ops: upload_block: this block has " + String(len(sizes)) + " pieces, got "
                        + String(len(srcs)))
        var offs = block_offsets(sizes, block_guard(code))
        var total = offs[len(offs) - 1]
        var buf = self.ex.alloc(total)
        try:
            for i in range(len(sizes)):  # driver-loop: over the block's weight or state pieces
                if sizes[i] == 0:
                    if srcs[i].count() != 0:
                        raise Error("ValueError: ops: upload_block: piece " + String(i)
                                    + " is absent under these options and must be empty")
                    continue
                srcs[i].check("ops: upload_block")
                if srcs[i].count() != sizes[i]:
                    raise Error("ValueError: ops: upload_block: piece " + String(i) + " holds "
                                + String(srcs[i].count()) + " values, the block " + String(sizes[i]))
                upload_cast(self.ex, buf + offs[i], srcs[i])
        except err:
            self.ex.free(buf)
            raise err
        var t = self.tape.variable(self.ex, buf, 1, total, requires_grad)
        var h = self._add(t, Int(buf), requires_grad, 0)
        if part == BLOCK_PART_STATE:
            self.buffer_kind[Int(buf)] = OPS_BUFFER_STATE
        return h

    def load_block(mut self, h: Int, code: Int, ints: List[Int], part: Int, srcs: List[CastIn],
                   words: List[Int]) raises:
        """Overwrite `part` of block tensor h (made with `upload_block` for the
        same code and ints) IN PLACE, piece by piece (lane x5-mamba-on-executor):
        a source of zero values leaves its piece as it is. The resident step
        struct over the buffer is kept where it stays valid: new float state
        is read in place (nothing to drop), new state words update the
        entry's host integers (`BlockStepScratch.set_words`), new weights
        re-arm the block's refusal of non-finite weights."""
        var slot = self._slot(h)
        var addr = self.owned[slot]
        if addr == 0:
            raise Error("ValueError: ops: load_block writes a tensor made with upload_block")
        var buf = FP(unsafe_from_address=addr)
        var have = self.rows[slot] * self.cols[slot]
        if part == BLOCK_PART_WORDS:
            var k = block_int_words(code)
            if k < 1 or len(words) != k or have != k:
                raise Error("ValueError: ops: load_block: this block carries " + String(k) + " state words")
            var host = List[Float32](length=k, fill=Float32(0))
            for i in range(k):  # small-loop(k: at most two state words): word encode, no data
                var w = words[i]
                if w < -(1 << 31) or w >= (1 << 31):
                    raise Error("ValueError: ops: load_block: a state word does not fit in 32 bits")
                host[i] = bitcast[DType.float32](Int32(w))
            self.steps.set_words(addr, words)
            self.ex.upload(buf, FP(unsafe_from_address=Int(host.unsafe_ptr())), k)
            _ = len(host)  # keep-alive past the staging copy
            return
        if part != BLOCK_PART_WEIGHTS and part != BLOCK_PART_STATE:
            raise Error("ValueError: ops: load_block: part must be weights, state or words")
        var sizes = block_weight_sizes(code, ints) if part == BLOCK_PART_WEIGHTS else block_state_sizes(code, ints)
        if len(srcs) != len(sizes):
            raise Error("ValueError: ops: load_block: this block has " + String(len(sizes)) + " pieces, got "
                        + String(len(srcs)))
        var offs = block_offsets(sizes, block_guard(code))
        if offs[len(offs) - 1] != have:
            raise Error("ValueError: ops: load_block: the tensor does not hold this block's layout")
        for i in range(len(sizes)):  # driver-loop: over the block's weight or state pieces
            if sizes[i] == 0 or srcs[i].count() == 0:
                continue
            srcs[i].check("ops: load_block")
            if srcs[i].count() != sizes[i]:
                raise Error("ValueError: ops: load_block: piece " + String(i) + " holds "
                            + String(srcs[i].count()) + " values, the block " + String(sizes[i]))
            upload_cast(self.ex, buf + offs[i], srcs[i])
        if part == BLOCK_PART_WEIGHTS:
            self.steps_weights_changed(addr)
        else:
            # lane fx-llama-refuse-watermark: loaded state floats were not
            # scanned by any step; an attention entry over them scans its
            # whole cache at the next run (a no-op without the arm's define)
            self.steps.attn_cache_written(addr)

    def download_block(mut self, h: Int, code: Int, ints: List[Int], part: Int, dsts: List[Int]) raises:
        """`part` (weights or float state) of block tensor h into the caller's
        host arrays, one float32 array per piece in `block_*_sizes` order (a
        0 address skips that piece), behind one wait."""
        var slot = self._slot(h)
        var addr = self.owned[slot]
        if addr == 0:
            raise Error("ValueError: ops: download_block reads a tensor made with upload_block")
        if part != BLOCK_PART_WEIGHTS and part != BLOCK_PART_STATE:
            raise Error("ValueError: ops: download_block: part must be weights or state")
        var sizes = block_weight_sizes(code, ints) if part == BLOCK_PART_WEIGHTS else block_state_sizes(code, ints)
        if len(dsts) != len(sizes):
            raise Error("ValueError: ops: download_block: this block has " + String(len(sizes)) + " pieces, got "
                        + String(len(dsts)))
        var offs = block_offsets(sizes, block_guard(code))
        var buf = FP(unsafe_from_address=addr)
        for i in range(len(sizes)):  # driver-loop: over the block's weight or state pieces
            if sizes[i] == 0 or dsts[i] == 0:
                continue
            self.ex.download_async(FP(unsafe_from_address=dsts[i]), buf + offs[i], sizes[i])
        self.ex.sync()
        self.steps.release_parked()

    def download(mut self, h: Int, dst: FP) raises:
        """The tensor's rows x cols floats (row-major) into the caller's memory."""
        var slot = self._slot(h)
        var src_t = self.tensors[slot]
        var src = self.tape.value_of(self.ex, src_t)
        var count = self.rows[slot] * self.cols[slot]
        self.ex.download(dst, src, count)
        # lane x45c: the stream drained; step ops' parked stages can go
        self.steps.release_parked()

    def scalar(mut self, h: Int) raises -> Float32:
        """The one value of a (1, 1) tensor, read back to the caller."""
        var slot = self._slot(h)
        if self.rows[slot] * self.cols[slot] != 1:
            raise Error("ops: scalar() takes a tensor of one value")
        var src_t = self.tensors[slot]
        var src = self.tape.value_of(self.ex, src_t)
        var out = List[Float32](length=1, fill=Float32(0))
        self.ex.download(FP(unsafe_from_address=Int(out.unsafe_ptr())), src, 1)
        self.steps.release_parked()
        return out[0]

    def free(mut self, h: Int) raises:
        self._retire(self._slot(h))

    def reset(mut self) raises:
        """Drop every computed tensor and the recorded graph; uploaded tensors
        stay (each a fresh leaf of the next graph)."""
        self.tape.reset()
        # the computed tensors the log names are gone; captured programs stay
        self.oplog.clear()
        self.cbw.clear_records()  # nn-blocks-cbw: the saved tensors were tape storage
        self.layer_nodes = False  # session-ops: the layer nodes were on the tape
        for slot in range(len(self.live)):  # driver-loop: over session handles
            if not self.live[slot]:
                continue
            if self.owned[slot] == 0:
                self._retire(slot)
            else:
                var buf = FP(unsafe_from_address=self.owned[slot])
                var r = self.rows[slot]
                var c = self.cols[slot]
                var grad = self.wants_grad[slot] and self.tape.is_recording()
                if self.shapes[slot].ndim == 2:
                    var leaf = self.tape.variable(self.ex, buf, r, c, grad)
                    self.tensors[slot] = leaf
                else:
                    var leaf_nd = self.tape.variable_nd(self.ex, buf, self.shapes[slot], grad)
                    self.tensors[slot] = leaf_nd
        for i in range(len(self.retired)):  # driver-loop: over freed uploads
            var old = FP(unsafe_from_address=self.retired[i])
            self.ex.free(old)
        self.retired.clear()

    def close(mut self) raises:
        """Return every buffer to the executor and wait for its queue."""
        # the resident block structs view session buffers: drop them first
        # (lane x45c: after the queue drained, with the parked run stages;
        # integ-w4: then both arms' entries through the x7b hook)
        if self.steps.parked() > 0:
            self.ex.sync()
            self.steps.release_parked()
        self.steps_clear()
        self.buffer_kind.clear()
        self.tape.release(self.ex)
        for slot in range(len(self.live)):  # driver-loop: over session handles
            if self.live[slot] and self.owned[slot] != 0:
                var buf = FP(unsafe_from_address=self.owned[slot])
                self.ex.free(buf)
            self.live[slot] = False
        for i in range(len(self.retired)):  # driver-loop: over freed uploads
            var old = FP(unsafe_from_address=self.retired[i])
            self.ex.free(old)
        self.retired.clear()
        for i in range(len(self.opt_states)):  # driver-loop: over optimizer states
            var r = self.opt_states[i]
            self.ex.free(r.s1)
            self.ex.free(r.s2)
            self.ex.free(r.s3)
            self.ex.free(r.parts)
            self.ex.free(r.cell)
        self.opt_states.clear()
        for i in range(len(self.opt_state_indices)):  # driver-loop: over optimizer index metadata at close
            self.opt_state_indices[i].clear()
        self.prog_tape.release(self.ex)
        self.programs.clear()
        self.program_replays.clear()
        self.op_tensors = List[Tensor]()
        self.oplog.clear()
        self.cbw.clear()  # nn-blocks-cbw
        self.ex.sync()

    # ------------------------------------------------------------ gradients
    def backward(mut self, h: Int, accumulate: Bool) raises:
        """Differentiate the (1, 1) tensor `h` (a recording session)."""
        if not self.tape.is_recording():
            raise Error("ops: backward needs a recording session (Session(record=True))")
        var t = self.tensor(h)
        # StepOpRules: ImportRules (RankNRules, NeuralRules plus the rank-N
        # lanes' custom tags, autograd/rankn_api.mojo, and the import
        # family's tags, autograd/import_api.mojo) plus the block step ops'
        # recorded-run tags (lane x56d: a session that records a step op
        # differentiates it here, not only Program.fit); a graph without
        # those nodes runs as with NeuralRules
        if self.cbw.holds_rules():
            # nn-blocks-cbw: the graph holds a user-defined backward rule
            self.cbw_backward_scalar(t, StepOpRules(self.steps_address()), accumulate)
            return
        if self.layer_nodes:
            # session-ops: the graph holds a Sequential layer node (one Bool test per call)
            self.tape.backward_scalar_with_rules(
                self.ex, t, LayerRules[StepOpRules](StepOpRules(self.steps_address())), accumulate)
            return
        self.tape.backward_scalar_with_rules(self.ex, t, StepOpRules(self.steps_address()), accumulate)

    def gradient_into(mut self, h: Int, dst: FP) raises:
        """The gradient of tensor `h` from the last backward into the caller's memory."""
        var slot = self._slot(h)
        var g = self.tape.gradient(self.tensors[slot])
        var count = self.rows[slot] * self.cols[slot]
        self.ex.download(dst, g, count)

    def zero_grad(mut self) raises:
        """Forget the gradient of every trainable tensor (no launch): the
        next backward with accumulate=True starts each from zero."""
        if not self.tape.is_recording():
            raise Error("ops: zero_grad needs a recording session (Session(record=True))")
        for slot in range(len(self.live)):  # driver-loop: over session handles
            if self.live[slot] and self.wants_grad[slot]:
                var leaf = self.tensors[slot]
                self.tape.clear_gradient(leaf)

    # ------------------------------------------------------------ higher order
    # Lane ag-ho-session (plan AUTOGRAD_HIGHER_ORDER (git history) section 5, H3):
    # the session surface of the recorded backward (autograd/higher.mojo,
    # lane H1). A recorded backward expresses every gradient through tape
    # operations, so a gradient is a session tensor that later operations
    # (and a second backward) can use. No new kernels here: the helpers call
    # backward_recorded / backward_scalar_recorded, Tape.grad_of, the tape's
    # multiply / sum / add, the first-order backward, and OP_FILL / OP_COPY
    # for the returned values. Host loops run over handle lists only.
    def _wrt(self, hs: List[Int], what: String) raises -> List[Tensor]:
        """The tensors of `hs`, each differentiable (an input of a VJP)."""
        if len(hs) < 1:
            raise Error("ops: " + what + " takes at least one input")
        var ts = List[Tensor](capacity=len(hs))
        for i in range(len(hs)):  # driver-loop: over the call's input handles
            var t = self.tensor(hs[i])
            if not self.tape.requires_grad(t):
                raise Error("ops: every input of " + what + " must require a gradient (requires_grad=True or computed from one)")
            ts.append(t)
        return ts^

    def _same_shapes(self, hs: List[Int], vs: List[Int], what: String) raises:
        if len(vs) != len(hs):
            raise Error("ops: " + what + " takes one direction per input")
        for i in range(len(hs)):  # driver-loop: over the call's input handles
            if not same_shape(self.shape_nd(hs[i]), self.shape_nd(vs[i])):
                raise Error("ValueError: ops: each direction of " + what + " must have its input's full shape")

    def _copy_constant(mut self, src: FP, rows: Int, cols: Int) raises -> Int:
        """A new constant session tensor holding a copy of `src`."""
        var dst = self.tape.reserve(self.ex, rows * cols)
        var a = Args()
        a.p0 = src
        a.p1 = dst
        self.ex.launch[OP_COPY](a, rows * cols)
        return self._constant(dst, rows, cols, 0)

    def _copy_constant_nd(mut self, src: FP, shape: ShapeN) raises -> Int:
        # Derivative results retain their full rank, rather than publishing the
        # tape's flattened matrix storage as the public tensor shape.
        var dst = self.tape.reserve(self.ex, shape.count())
        var a = Args()
        a.p0 = src
        a.p1 = dst
        self.ex.launch[OP_COPY](a, shape.count())
        return self._leaf(dst, shape)

    def _zeros_nd(mut self, shape: ShapeN) raises -> Int:
        """A new constant session tensor of zeros."""
        var dst = self.tape.reserve(self.ex, shape.count())
        var a = Args()
        a.p0 = dst
        a.f0 = Float32(0)
        self.ex.launch[OP_FILL](a, shape.count())
        return self._leaf(dst, shape)

    def _directional(mut self, gs: List[Tensor], vs: List[Int]) raises -> Tensor:
        """sum_i sum(gs[i] * vs[i]) as one recorded (1, 1) tensor, the
        terms added in list order."""
        var g0 = gs[0]
        var v0 = self.tensor(vs[0])
        var p0 = self.tape.multiply(self.ex, g0, v0)
        var d = self.tape.sum(self.ex, p0, -1)
        for i in range(1, len(gs)):  # driver-loop: over the call's input handles
            var gi = gs[i]
            var vi = self.tensor(vs[i])
            var pi = self.tape.multiply(self.ex, gi, vi)
            var term = self.tape.sum(self.ex, pi, -1)
            d = self.tape.add(self.ex, d, term)
        return d

    def backward_graph(mut self, h: Int) raises:
        """Differentiate the (1, 1) tensor `h` and record the gradient
        computation itself, so each gradient is a tensor (grad_handle) that
        can be differentiated again. No accumulate form: add the previous
        gradient tensor with an operation instead."""
        if not self.tape.is_recording():
            raise Error("ops: backward needs a recording session (Session(record=True))")
        self.cbw_first_order("backward(create_graph=True)")  # nn-blocks-cbw
        var t = self.tensor(h)
        backward_scalar_recorded(self.tape, self.ex, t)

    def backward_seed(mut self, h: Int, seed_h: Int, create_graph: Bool) raises:
        """Differentiate tensor `h` against the seed `seed_h` (a tensor of
        h's shape): the vector-Jacobian product of h for that seed, recorded
        when `create_graph` is set."""
        if not self.tape.is_recording():
            raise Error("ops: backward needs a recording session (Session(record=True))")
        var t = self.tensor(h)
        var seed = self.tensor(seed_h)
        if not same_shape(self.shape_nd(h), self.shape_nd(seed_h)):
            raise Error("ValueError: ops: the seed must have the output's full shape")
        if create_graph:
            self.cbw_first_order("create_graph=True")  # nn-blocks-cbw
            backward_recorded(self.tape, self.ex, t, seed)
        elif self.cbw.holds_rules():
            # nn-blocks-cbw: the graph holds a user-defined backward rule
            self.cbw_backward_seeded(t, seed)
        elif self.layer_nodes:
            # session-ops: the graph holds a Sequential layer node (one Bool test per call)
            self.tape.backward_with_rules(self.ex, t, seed, LayerRules[ImportRules](ImportRules()), False)
        else:
            # First-order seeded and scalar backward accept the same supported
            # rank-N and import operations. Recorded higher derivatives keep
            # their existing explicit operation contract above.
            self.tape.backward_with_rules(self.ex, t, seed, ImportRules(), False)

    def grad_handle(mut self, h: Int) raises -> Int:
        """The gradient of tensor `h` from the last recorded backward, as a
        new session handle (a computed tensor: it lives until reset)."""
        var t = self.tensor(h)
        var g = self.tape.grad_of(t)
        return self._add(g, 0, False, 0)

    def vjp(mut self, out_h: Int, hs: List[Int], seed_h: Int, create_graph: Bool) raises -> List[Int]:
        """The vector-Jacobian product of `out_h` for the seed `seed_h` with
        respect to each tensor of `hs`; one new handle per input. With
        `create_graph` the results are recorded (differentiable) tensors;
        otherwise they are constant copies of the gradients."""
        var xs = self._wrt(hs, "vjp")
        self.backward_seed(out_h, seed_h, create_graph)
        var res = List[Int](capacity=len(hs))
        for i in range(len(xs)):  # driver-loop: over the call's input handles
            var x = xs[i]
            if create_graph:
                var g = self.tape.grad_of(x)
                res.append(self._add(g, 0, False, 0))
            else:
                var shape = self.shape_nd(hs[i])
                var gbuf = self.tape.gradient(x)
                res.append(self._copy_constant_nd(gbuf, shape))
        return res^

    def hvp(mut self, loss_h: Int, hs: List[Int], vs: List[Int]) raises -> List[Int]:
        """The Hessian-vector product of the (1, 1) tensor `loss_h` with
        respect to the tensors `hs` along the directions `vs` (one per
        input, of its shape); one new constant handle per input. A recorded
        backward gives the gradients g_i as tensors; d = sum_i sum(g_i * v_i)
        is recorded from them; a first-order backward of d gives H v. When
        no gradient depends on an input (zero curvature) the result is
        zeros."""
        if not self.tape.is_recording():
            raise Error("ops: hvp needs a recording session (Session(record=True))")
        self.cbw_first_order("hvp")  # nn-blocks-cbw
        var xs = self._wrt(hs, "hvp")
        self._same_shapes(hs, vs, "hvp")
        var t = self.tensor(loss_h)
        backward_scalar_recorded(self.tape, self.ex, t)
        var gs = List[Tensor](capacity=len(xs))
        for i in range(len(xs)):  # driver-loop: over the call's input handles
            var x = xs[i]
            gs.append(self.tape.grad_of(x))
        var d = self._directional(gs, vs)
        var res = List[Int](capacity=len(hs))
        if not self.tape.requires_grad(d):
            for i in range(len(hs)):  # driver-loop: over the call's input handles
                res.append(self._zeros_nd(self.shape_nd(hs[i])))
            return res^
        self.tape.backward_scalar_with_rules(self.ex, d, NeuralRules(), False)
        for i in range(len(xs)):  # driver-loop: over the call's input handles
            var x = xs[i]
            var gbuf = self.tape.gradient(x)
            res.append(self._copy_constant_nd(gbuf, self.shape_nd(hs[i])))
        return res^

    def jvp(mut self, out_h: Int, hs: List[Int], us: List[Int]) raises -> Int:
        """The Jacobian-vector product of tensor `out_h` along the
        directions `us` (one per tensor of `hs`, of its shape); a new
        constant handle of out_h's shape. By the double-VJP identity: with a
        seed w of out_h's shape (a differentiable leaf; its value does not
        matter, the VJP is linear in it) the recorded VJP gives J_i^T w as
        tensors, and the gradient of sum_i sum(J_i^T w * u_i) with respect
        to w is sum_i J_i u_i."""
        if not self.tape.is_recording():
            raise Error("ops: jvp needs a recording session (Session(record=True))")
        self.cbw_first_order("jvp")  # nn-blocks-cbw
        var xs = self._wrt(hs, "jvp")
        self._same_shapes(hs, us, "jvp")
        var t = self.tensor(out_h)
        var oshape = self.shape_nd(out_h)
        var wbuf = self.tape.reserve(self.ex, oshape.count())
        var fa = Args()
        fa.p0 = wbuf
        fa.f0 = Float32(0)
        self.ex.launch[OP_FILL](fa, oshape.count())
        var w = self.tape.variable_nd(self.ex, wbuf, oshape, True)
        backward_recorded(self.tape, self.ex, t, w)
        var gs = List[Tensor](capacity=len(xs))
        for i in range(len(xs)):  # driver-loop: over the call's input handles
            var x = xs[i]
            gs.append(self.tape.grad_of(x))
        var d = self._directional(gs, us)
        if not self.tape.requires_grad(d):
            return self._zeros_nd(oshape)
        self.tape.backward_scalar_with_rules(self.ex, d, NeuralRules(), False)
        var jbuf = self.tape.gradient(w)
        return self._copy_constant_nd(jbuf, oshape)

    # ------------------------------------------------------------ optimizers
    def optimizer(mut self, ints: List[Int], fs: List[Float32]) raises -> Int:
        """A new optimizer; returns its id (>= 1).

        ints = [kind (OPT_SGD, OPT_ADAM, OPT_ADAMW, OPT_LION, OPT_ADAMAX or
        OPT_NADAM), nesterov, schedule
        (autograd/model.mojo SCHED_*), step size]; fs = [lr, momentum,
        beta1, beta2, eps, weight_decay, gamma] and optionally an eighth,
        SGD's dampening (torch.optim.SGD; 0 when absent). The configuration
        is packed as Sequential.compile packs it (sgd: f1 momentum, f2 0,
        f7 dampening; adam and adamw: f1 beta1, f2 beta2; f7 0), so the
        update's arguments are model_fit's words. Lion, Adamax and NAdam
        pack as adam does (f1 beta1, f2 beta2); NAdam's f7 is torch's
        default momentum_decay 0.004 with coupled weight decay."""
        return self.optimizer(ops_optimizer_request(ints, fs))

    def optimizer(mut self, request: OpsOptimizerRequest) raises -> Int:
        """Create an optimizer from one typed native configuration snapshot."""
        return self.optimizer_with(request, ops_optimizer_extras_default())

    def optimizer_with(mut self, request: OpsOptimizerRequest, extras: OpsOptimizerExtras) raises -> Int:
        """`optimizer` with the settings the request does not carry
        (autograd/optimizer_requests.mojo OpsOptimizerExtras): RMSprop
        (`request.momentum` is its momentum) and Adagrad, and the gradient
        clips applied inside `step`."""
        var entry = self._optimizer_entry(request, extras)
        self.opts.append(entry)
        # Failed control-checkpoint restore pops opts before its caller retires
        # partial parameter states. Keep that index addressable during cleanup,
        # then reuse its table on the next creation instead of accumulating an
        # orphan table for each failed restore. Index storage is bounded by the
        # optimizer-count high-water mark; ordinary creation still adds one.
        if len(self.opt_state_indices) < len(self.opts):
            self.opt_state_indices.append(Dict[Int, Int]())
        return len(self.opts)

    def _optimizer_entry(self, request: OpsOptimizerRequest, extras: OpsOptimizerExtras) raises -> _OpsOptimizer:
        """The validated configuration and schedule of one optimizer (lane
        nn-blocks-pg: split from `optimizer_with` so a parameter group's
        member and a loaded group's settings pass the same checks)."""
        var kind = request.kind
        # Lane fx-ops-session-blocks (2026-11-18): Lion, Adamax and NAdam are
        # OP_OPT kinds (sequence/ops.mojo op_opt) whose whole configuration
        # is this request's fields; `step_lr`, `optimizer_slots` and the
        # state records below already serve every kind.
        if (kind != OPT_SGD and kind != OPT_ADAM and kind != OPT_ADAMW and kind != OPT_LION
                and kind != OPT_ADAMAX and kind != OPT_NADAM and kind != OPT_RMSPROP and kind != OPT_ADAGRAD):
            raise Error("ops: the optimizer must be sgd, adam, adamw, rmsprop, adagrad, lion, adamax or nadam")
        if not (extras.alpha >= Float32(0) and extras.alpha < Float32(1)):
            raise Error("ops: alpha must be in [0, 1)")
        if not (extras.lr_decay >= Float32(0)):
            raise Error("ops: lr_decay must be >= 0")
        if not (extras.initial_accumulator_value >= Float32(0)):
            raise Error("ops: initial_accumulator_value must be >= 0")
        if not (extras.clip_grad_norm >= Float32(0)) or not (extras.clip_grad_value >= Float32(0)):
            raise Error("ops: clip_grad_norm and clip_grad_value must be positive (or absent)")
        if extras.centered and kind != OPT_RMSPROP:
            raise Error("ops: centered applies to rmsprop")
        var lr = request.lr
        var momentum = request.momentum
        var beta1 = request.beta1
        var beta2 = request.beta2
        var eps = request.eps
        var wd = request.weight_decay
        var gamma = request.gamma
        # lane x8-cnn-on-executor: CNNClassifier's SGD dampening on the session optimizer
        var dampening = request.dampening
        if not (lr > Float32(0)):
            raise Error("ops: the learning rate must be positive")
        if not (momentum >= Float32(0) and momentum < Float32(1)):
            raise Error("ops: momentum must be in [0, 1)")
        if not (beta1 >= Float32(0) and beta1 < Float32(1)) or not (beta2 >= Float32(0) and beta2 < Float32(1)):
            raise Error("ops: beta1 and beta2 must be in [0, 1)")
        if not (eps > Float32(0)):
            raise Error("ops: eps must be positive")
        if not (wd >= Float32(0)):
            raise Error("ops: weight_decay must be >= 0")
        var nesterov = request.nesterov
        if nesterov and (kind != OPT_SGD or not (momentum > Float32(0))):
            raise Error("ops: Nesterov momentum needs sgd with momentum > 0")
        if not (dampening >= Float32(0) and dampening <= Float32(1)):
            raise Error("ops: dampening must be in [0, 1]")
        if dampening != Float32(0) and (kind != OPT_SGD or nesterov):
            raise Error("ops: dampening applies to sgd without Nesterov momentum")
        var sched = request.schedule
        if sched < SCHED_CONSTANT or sched > SCHED_EXPONENTIAL:
            raise Error("ops: unknown learning-rate schedule code " + String(sched))
        if sched != SCHED_CONSTANT and not (gamma > Float32(0)):
            raise Error("ops: the schedule's gamma must be positive")
        if sched == SCHED_STEP and request.step_size < 1:
            raise Error("ops: the step schedule's step size must be at least 1")
        var f1 = momentum if kind == OPT_SGD else beta1
        var f2 = Float32(0) if kind == OPT_SGD else beta2
        var flags = 1 if nesterov else 0
        if kind == OPT_RMSPROP:
            # optimizer_config_packed: f2 alpha, f7 momentum, flag bit centered
            f1 = Float32(0)
            f2 = extras.alpha
            flags = 1 if extras.centered else 0
        elif kind == OPT_ADAGRAD:
            # f1 lr_decay; the accumulator's start is the state's fill
            f1 = extras.lr_decay
        # f7: SGD's dampening; NAdam's momentum_decay, torch's default
        # (0.004 as a Float32 word, optimizer_config's own default); nothing
        # else reads it.
        var f7 = Float32(0.004) if kind == OPT_NADAM else dampening
        if kind == OPT_RMSPROP:
            f7 = momentum
        var cfg = optimizer_config_packed(kind, flags, f1, f2, eps, wd, f7, lr)
        cfg.init_acc = extras.initial_accumulator_value if kind == OPT_ADAGRAD else Float32(0)
        # The clips of `step`: the norm over all the step's tensors (the
        # state's partials are sized by cfg.clip) and the elementwise bound.
        if extras.clip_grad_norm > Float32(0):
            cfg.clip = extras.clip_grad_norm
            cfg.clip_global = True
        cfg.clip_value = extras.clip_grad_value
        var sched_step = request.step_size if sched == SCHED_STEP else 1
        return _OpsOptimizer(cfg, sched, sched_step, gamma, 0, -1)

    def _opt_index(self, o: Int) raises -> Int:
        if o < 1 or o > len(self.opts):
            raise Error("ValueError: ops: " + String(o) + " is not an optimizer of this session")
        return o - 1

    def learning_rate(self, o: Int) raises -> Float32:
        """Optimizer `o`'s learning rate at its schedule's current epoch."""
        var op = self.opts[self._opt_index(o)]
        return epoch_lr(op.cfg.lr, op.sched, op.sched_step, op.gamma, op.epoch)

    def scheduler_step(mut self, o: Int) raises:
        """Advance optimizer `o`'s schedule by one epoch."""
        var oi = self._opt_index(o)
        self.opts[oi].epoch += 1
        if self.opts[oi].pg >= 0:
            self._pg_scheduler_step(oi)  # nn-blocks-pg: every group's rate

    def _param_slot(self, h: Int, what: String) raises -> Int:
        var slot = self._slot(h)
        if self.owned[slot] == 0 or not self.wants_grad[slot]:
            raise Error("ops: " + what + " takes tensors made with requires_grad=True")
        return slot

    def _state_of(mut self, oi: Int, h: Int, n: Int) raises -> Int:
        """The index of (optimizer oi, parameter h)'s state, allocated (zero
        moment buffers, step 0) on first use."""
        # Training revisits the same pairs every step. Resolve directly rather
        # than walking all states for every parameter. h includes its handle
        # generation, so a reused tensor slot cannot inherit retired moments.
        if h in self.opt_state_indices[oi]:
            return self.opt_state_indices[oi][h]
        var cfg = self.opts[oi].cfg
        var st = OptimizerState(self.ex, cfg, n, cfg.init_acc)
        self.opt_states.append(_OpsOptState(
            oi, h, st.s1, st.s2, st.s3, st.parts, st.cell, st.n, st.block, st.nparts, 0,
            st.pw1, st.pw2, st.mu_prod,
        ))
        var index = len(self.opt_states) - 1
        self.opt_state_indices[oi][h] = index
        return index

    def _state_view(self, i: Int) -> OptimizerState:
        """An OptimizerState over state i's buffers and running scalars (it
        owns nothing; the session frees the buffers)."""
        var r = self.opt_states[i]
        var st = OptimizerState()
        st.s1 = r.s1
        st.s2 = r.s2
        st.s3 = r.s3
        st.parts = r.parts
        st.cell = r.cell
        st.n = r.n
        st.block = r.block
        st.nparts = r.nparts
        st.pw1 = r.pw1
        st.pw2 = r.pw2
        st.mu_prod = r.mu_prod
        return st^

    def _drop_states(mut self, h: Int):
        """Retire every optimizer state of parameter handle h."""
        var previous_count = len(self.opt_states)
        var i = previous_count - 1
        while i >= 0:  # driver-loop: over optimizer states
            if self.opt_states[i].param == h:
                var r = self.opt_states[i]
                self.retired.append(Int(r.s1))
                self.retired.append(Int(r.s2))
                self.retired.append(Int(r.s3))
                self.retired.append(Int(r.parts))
                self.retired.append(Int(r.cell))
                _ = self.opt_states.pop(i)
            i -= 1
        if len(self.opt_states) != previous_count:
            # Removal compacts the vector. Reindex only at this retirement
            # boundary, never on ordinary optimizer steps or tape resets.
            for j in range(len(self.opt_state_indices)):  # driver-loop: over optimizer index metadata after retirement
                self.opt_state_indices[j].clear()
            for j in range(len(self.opt_states)):  # driver-loop: over surviving optimizer state metadata after retirement
                var oi = self.opt_states[j].opt
                self.opt_state_indices[oi][self.opt_states[j].param] = j

    def step(mut self, o: Int, hs: List[Int]) raises:
        """One update of each parameter in `hs` by optimizer `o`, in place,
        from the last backward's gradients: per parameter ONE OP_OPT launch
        (autograd/optim.mojo step_lr; no clip, no read-back, no sync) at the
        schedule's current learning rate and the parameter's next step
        count. Every parameter is checked before any is updated."""
        var oi = self._opt_index(o)
        if not self.tape.is_recording():
            raise Error("ops: step needs a recording session (Session(record=True))")
        if len(hs) < 1:
            raise Error("ops: step takes at least one parameter")
        if self.opts[oi].pg >= 0:
            self._pg_step(oi, hs)  # nn-blocks-pg: a grouped optimizer
            return
        var slots = List[Int](capacity=len(hs))
        var grads = List[FP](capacity=len(hs))
        var seen = Dict[Int, Int]()
        for i in range(len(hs)):  # driver-loop: over the step's parameters
            var slot = self._param_slot(hs[i], "step")
            if slot in seen:
                raise Error("ops: step lists a parameter twice")
            seen[slot] = i
            slots.append(slot)
            grads.append(self.tape.gradient(self.tensors[slot]))
        var cfg = self.opts[oi].cfg
        var lr = self.learning_rate(o)
        if optimizer_clips(cfg):
            # Lane fx-ops-session-blocks: clip_grad_norm over ALL the listed
            # tensors (autograd/optim.mojo, the fold above op_ago_norm_acc:
            # two launches a tensor, the total and the coefficient in the
            # first tensor's cell, no read-back) and clip_grad_value, then
            # one update launch a tensor.
            var sis = List[Int](capacity=len(slots))
            for i in range(len(slots)):  # driver-loop: over the step's parameters
                sis.append(self._state_of(oi, hs[i], self.rows[slots[i]] * self.cols[slots[i]]))
            var cell = self.opt_states[sis[0]].cell
            if cfg.clip_global:
                for i in range(len(slots)):  # driver-loop: over the step's parameters (the norm's fold order)
                    var sv = self._state_view(sis[i])
                    clip_norm_accumulate(self.ex, cfg, sv, grads[i], self.rows[slots[i]] * self.cols[slots[i]],
                                         cell, i == 0, i == len(slots) - 1)
            for i in range(len(slots)):  # driver-loop: over the step's parameters
                var cslot = slots[i]
                var ct = self.opt_states[sis[i]].t + 1
                var cst = self._state_view(sis[i])
                step_lr_clipped(self.ex, cfg, cst, FP(unsafe_from_address=self.owned[cslot]), grads[i],
                                self.rows[cslot] * self.cols[cslot], ct, lr, cell)
                self.steps_forget(self.owned[cslot])  # the weights moved: a step op re-checks them
                self.opt_states[sis[i]].t = ct
                self.opt_states[sis[i]].pw1 = cst.pw1
                self.opt_states[sis[i]].pw2 = cst.pw2
                self.opt_states[sis[i]].mu_prod = cst.mu_prod
            return
        for i in range(len(slots)):  # driver-loop: over the step's parameters
            var slot = slots[i]
            var n = self.rows[slot] * self.cols[slot]
            var si = self._state_of(oi, hs[i], n)
            var t = self.opt_states[si].t + 1
            var st = self._state_view(si)
            step_lr(self.ex, cfg, st, FP(unsafe_from_address=self.owned[slot]), grads[i], n, t, lr)
            self.steps_forget(self.owned[slot])  # the weights moved: a step op re-checks them
            self.opt_states[si].t = t
            self.opt_states[si].pw1 = st.pw1
            self.opt_states[si].pw2 = st.pw2
            self.opt_states[si].mu_prod = st.mu_prod

    def _snapshot(mut self, src: FP, slot: Int) raises -> Int:
        """A copy of a state buffer shaped as parameter `slot`, as a new
        tensor (valid until reset)."""
        var r = self.rows[slot]
        var c = self.cols[slot]
        var dst = self.tape.reserve(self.ex, r * c)
        var a = Args()
        a.p0 = src
        a.p1 = dst
        self.ex.launch[OP_COPY](a, r * c)
        return self._constant(dst, r, c, 0)

    def optimizer_state(mut self, o: Int, h: Int) raises -> Tuple[Int, Float32, Float32, Float32, Int, Int, Int]:
        """Existing native tuple API, packed from the canonical typed snapshot."""
        var state = self.optimizer_state_value(o, h)
        return (state.step, state.beta1_power, state.beta2_power, state.mu_product,
                state.first, state.second, state.third)

    def optimizer_state_value(mut self, o: Int, h: Int) raises -> OpsOptimizerStateValue:
        """(step count, beta1^t, beta2^t, NAdam's mu product, then one
        handle per state buffer: a copy of each buffer the kind uses, valid
        until reset, and 0 for one it does not use). A parameter never
        stepped reads step 0 and zero buffers."""
        var oi = self._opt_index(o)
        if self.opts[oi].pg >= 0:
            # nn-blocks-pg: the parameter's group member keeps its state
            oi = self._pg_member(oi, h, "optimizer_state")
        var slot = self._param_slot(h, "optimizer_state")
        var si = self._state_of(oi, h, self.rows[slot] * self.cols[slot])
        var r = self.opt_states[si]
        var used = optimizer_slots(self.opts[oi].cfg)
        var h1 = 0
        var h2 = 0
        var h3 = 0
        if used[0]:
            h1 = self._snapshot(r.s1, slot)
        if used[1]:
            h2 = self._snapshot(r.s2, slot)
        if used[2]:
            h3 = self._snapshot(r.s3, slot)
        return OpsOptimizerStateValue(self.opts[oi].cfg.kind, r.t, r.pw1, r.pw2, r.mu_prod, h1, h2, h3)

    def load_optimizer_state(mut self, o: Int, h: Int, ints: List[Int], fs: List[Float32]) raises:
        """Set (optimizer o, parameter h)'s state: ints = [step count, then
        one tensor handle per state buffer (0 for a buffer the kind does
        not use)], fs = [beta1^t, beta2^t, NAdam's mu product], the words
        `optimizer_state` returns. Each buffer is copied on the executor."""
        self.load_optimizer_state(o, h, ops_optimizer_restore_request(ints, fs))

    def load_optimizer_state(mut self, o: Int, h: Int, state: OpsOptimizerRestoreRequest) raises:
        """Restore typed scalar state and borrowed tensor handles on the executor."""
        if state.step < 0:
            raise Error("ops: the optimizer state's step count must be >= 0")
        var oi = self._opt_index(o)
        if self.opts[oi].pg >= 0:
            # nn-blocks-composed begin
            comptime if (COMPOSED_SABOTAGE_GROUP_STATE and Self.E.IS_DEVICE) or (
                    COMPOSED_SABOTAGE_GROUP_STATE_HOST and not Self.E.IS_DEVICE):
                if self._pg_group(oi, h, "load_optimizer_state") >= 1:
                    return  # SABOTAGE: a later group's state is not restored
            # nn-blocks-composed end
            # nn-blocks-pg: the parameter's group member keeps its state
            oi = self._pg_member(oi, h, "load_optimizer_state")
        var slot = self._param_slot(h, "load_optimizer_state")
        var n = self.rows[slot] * self.cols[slot]
        var si = self._state_of(oi, h, n)
        var r = self.opt_states[si]
        var u = optimizer_slots(self.opts[oi].cfg)
        var used = List[Bool]()
        used.append(u[0])
        used.append(u[1])
        used.append(u[2])
        var dsts = List[FP]()
        dsts.append(r.s1)
        dsts.append(r.s2)
        dsts.append(r.s3)
        # every handle checked before any copy
        var srcs = List[FP]()
        for k in range(3):  # driver-loop: over the three state buffers
            var hk = state.first if k == 0 else (state.second if k == 1 else state.third)
            if not used[k]:
                if hk != 0:
                    raise Error("ops: this optimizer keeps no state buffer " + String(k + 1))
                srcs.append(r.s1)
                continue
            if hk == 0:
                raise Error("ops: the optimizer state lacks state buffer " + String(k + 1))
            var ks = self._slot(hk)
            if self.rows[ks] != self.rows[slot] or self.cols[ks] != self.cols[slot]:
                raise Error("ops: an optimizer state buffer must have its parameter's shape")
            var src_t = self.tensors[ks]
            srcs.append(self.tape.value_of(self.ex, src_t))
        for k in range(3):  # driver-loop: over the three state buffers
            if used[k]:
                var a = Args()
                a.p0 = srcs[k]
                a.p1 = dsts[k]
                self.ex.launch[OP_COPY](a, n)
        self.opt_states[si].t = state.step
        self.opt_states[si].pw1 = state.beta1_power
        self.opt_states[si].pw2 = state.beta2_power
        self.opt_states[si].mu_prod = state.mu_product

    # ------------------------------------------------------------ operations
    def _constant(mut self, values: FP, rows: Int, cols: Int, index_rows: Int, integer_words: Bool = False) raises -> Int:
        var t = self.tape.variable(self.ex, values, rows, cols, False)
        var h = self._add(t, 0, False, index_rows)
        self.int_words[self._slot(h)] = integer_words
        return h

    def _result_int_words(self, code: Int, hs: List[Int], ints: List[Int]) raises -> Bool:
        # Dtype metadata follows value selection/layout, independently of
        # storage aliases. These operations do not convert int32 words into
        # floating-point values. Arithmetic outputs keep their existing float
        # semantics; argmax explicitly chooses its result representation.
        if code == OPS_ARGMAX:
            return ints[2] == 0
        if code == OPS_CONCAT:
            for k in range(len(hs)):  # driver-loop: over concatenated value kinds
                if self.int_words[self._slot(hs[k])]:
                    return True
            return False
        if code == OPS_WHERE:
            return self.int_words[self._slot(hs[1])] or self.int_words[self._slot(hs[2])]
        if code == OPS_TRANSPOSE or code == OPS_RESHAPE or code == OPS_DETACH or code == OPS_COPY or code == OPS_EMBEDDING or code == OPS_RESHAPE_N or code == OPS_PERMUTE or code == OPS_NARROW or code == OPS_TAKE or code == OPS_EXPAND or code == OPS_PAD or code == OPS_INDEX:
            return self.int_words[self._slot(hs[0])]
        return False

    def op(mut self, code: Int, hs: List[Int], ints: List[Int], fs: List[Float32]) raises -> Int:
        """Run operation `code` on the tensors `hs` with integer and float
        arguments; returns the new tensor's handle. An inference session
        logs the call (autograd/program.mojo OpsLog) for `capture`."""
        var h = self._op_run(code, hs, ints, fs)
        if not self.tape.is_recording():
            self.oplog.add(code, hs, ints, fs, h)
        return h

    def _op_run[nested: Bool = False](mut self, code: Int, hs: List[Int], ints: List[Int], fs: List[Float32]) raises -> Int:
        """`op` without the log: the eager call and a program's replay (`nested`: a custom operation's own
        nodes, `_op_run_tensors[nested]`)."""
        # Move capacity into the call, so the native operation can mutate self
        # without borrowing session-owned operand storage. Resolve every handle
        # afresh; only allocation capacity survives success or failure.
        var x = List[Tensor]()
        swap(x, self.op_tensors)
        try:
            for i in range(len(hs)):  # small-loop(hs: an op's input handles, three or a concat's list): handle lookup, no data
                x.append(self.tensor(hs[i]))
            return self._op_run_tensors[nested](code, hs, ints, fs, x)
        finally:
            x.clear()
            swap(x, self.op_tensors)

    def _op_run_tensors[nested: Bool = False](
        mut self, code: Int, hs: List[Int], ints: List[Int], fs: List[Float32], x: List[Tensor]
    ) raises -> Int:
        """Execute with this call's freshly checked native tensor references.
        `nested` (nn-blocks-zerocost): True for a scan cell's nodes (`_scan_call`) and a custom operation's
        forward and backward nodes (`_program_op[nested=True]`), whose instance has no OPS_SCAN and no
        OPS_CUSTOM_VJP branch, so `_scan`, `_cbw_forward` and `_cbw_apply` never reach themselves (AGENTS.md
        part 6)."""
        var n = len(x)
        var ni = len(ints)
        var nf = len(fs)
        var t: Tensor
        if is_block_step(code):
            # Lane x1: the step-op table (autograd/block_step_ops.mojo), one
            # branch for all its codes.
            return self._block_step(code, hs, ints, fs, x)
        if is_generation_op(code):
            # Lane x45c: the token pick and the decode condition, one branch.
            return self._generation_op(code, hs, ints, fs, x)
        if code == OPS_LM_HEAD_V2:
            # Lane m2-neural-leftovers: the chunked LM head (loss without
            # the logits), one custom node (autograd/lm_head_v2_op.mojo).
            return self._lm_head_v2(hs, ints, fs, x)
        if code == OPS_SMALL_MLP:
            # Lane r1-restore-measured-winners: the small MLP's tuned forward
            # and hand backward, one custom node (autograd/small_mlp_op.mojo).
            return self._small_mlp(hs, ints, fs, x)
        if code == OPS_CNN_CONV_BLOCK:
            # Lane r1-restore-measured-winners: CNNClassifier's hand-tuned
            # conv block, one custom node (autograd/cnn_block_op.mojo).
            return self._cnn_conv_block(hs, ints, fs, x)
        if code == OPS_CNN_LINEAR:
            return self._cnn_linear(hs, ints, fs, x)
        if code == OPS_RNN_SCAN:
            # Lane fx-ops-session-blocks: one recurrent layer over a sequence,
            # the recurrent trainer's own records (autograd/rules_recurrent.mojo).
            return self._rnn_scan(hs, ints, fs, x)
        if code == OPS_LM_EMBEDDING:
            # Lane r1-restore-measured-winners: the byte LM trainer's tuned
            # embedding backward, one custom node (autograd/lm_embedding_op.mojo).
            return self._lm_embedding(hs, ints, fs, x)
        if code == OPS_RMS_NORM or code == OPS_ROTARY or code == OPS_TRANSFORMER_LINEAR or code == OPS_TRANSFORMER_SOFTMAX:
            # These primitive signatures have at most two handles. Their
            # arity validation below rejects any additional operands.
            if ((n > 0 and self.int_words[self._slot(hs[0])])
                    or (n > 1 and self.int_words[self._slot(hs[1])])):
                raise Error("ValueError: transformer primitives require floating-point tensors")
        if code == OPS_MATMUL or code == OPS_MATMUL_N:
            _need("matmul", n, 2, 0, ni, 0, nf)
            var ma = self.tape.ndim(x[0])
            var mb = self.tape.ndim(x[1])
            if ma < 2 or mb < 2:
                raise Error("ops: matmul takes tensors of at least 2 dimensions")
            if ma == 2 and mb == 2:
                t = self.tape.matmul(self.ex, x[0], x[1])
            else:
                # batched product (rank-N @ rank-2 routes to the 2-D GEMM inside)
                t = rankn_matmul(self.tape, self.ex, x[0], x[1])
        elif code == OPS_ADD:
            _need("add", n, 2, 0, ni, 0, nf)
            t = self._binary(x[0], x[1], ADD)
        elif code == OPS_SUB:
            _need("sub", n, 2, 0, ni, 0, nf)
            t = self._binary(x[0], x[1], SUBTRACT)
        elif code == OPS_MUL:
            _need("mul", n, 2, 0, ni, 0, nf)
            t = self._binary(x[0], x[1], MULTIPLY)
        elif code == OPS_DIV:
            _need("div", n, 2, 0, ni, 0, nf)
            t = self._binary(x[0], x[1], DIVIDE)
        elif code == OPS_RELU:
            _need("relu", n, 1, 0, ni, 0, nf)
            t = self.tape.relu(self.ex, x[0])
        elif code == OPS_SIGMOID:
            _need("sigmoid", n, 1, 0, ni, 0, nf)
            t = self.tape.sigmoid(self.ex, x[0])
        elif code == OPS_TANH:
            _need("tanh", n, 1, 0, ni, 0, nf)
            t = self.tape.tanh(self.ex, x[0])
        elif code == OPS_EXP:
            _need("exp", n, 1, 0, ni, 0, nf)
            t = self.tape.exp(self.ex, x[0])
        elif code == OPS_LOG:
            _need("log", n, 1, 0, ni, 0, nf)
            t = self.tape.log(self.ex, x[0])
        elif code == OPS_SQRT:
            _need("sqrt", n, 1, 0, ni, 0, nf)
            t = self.tape.sqrt(self.ex, x[0])
        elif code == OPS_SCALE:
            _need("scale", n, 1, 0, ni, 1, nf)
            t = self.tape.scale(self.ex, x[0], fs[0])
        elif code == OPS_TRANSPOSE:
            _need("transpose", n, 1, 0, ni, 0, nf)
            t = self._swap(x[0], -2, -1)
        elif code == OPS_RESHAPE:
            _need("reshape", n, 1, 2, ni, 0, nf)
            t = self.tape.reshape(self.ex, x[0], ints[0], ints[1])
        elif code == OPS_SUM:
            _need("sum", n, 1, 1, ni, 0, nf)
            t = self.tape.sum(self.ex, x[0], ints[0])
        elif code == OPS_MEAN:
            _need("mean", n, 1, 1, ni, 0, nf)
            t = self.tape.mean(self.ex, x[0], ints[0])
        elif code == OPS_SOFTMAX:
            _need("softmax", n, 1, 0, ni, 0, nf)
            t = self.tape.softmax(self.ex, x[0])
        elif code == OPS_LOG_SOFTMAX:
            _need("log_softmax", n, 1, 0, ni, 0, nf)
            t = self.tape.log_softmax(self.ex, x[0])
        elif code == OPS_LINEAR:
            if n != 2 and n != 3:
                raise Error("ops: linear takes an input, a weight and an optional bias")
            _need("linear", 2, 2, 1, ni, 0, nf)
            var act = _activation(ints[0])
            if n == 3:
                t = self.tape.linear(self.ex, x[0], x[1], Optional[Tensor](x[2]), act)
            else:
                t = self.tape.linear(self.ex, x[0], x[1], None, act)
            var lx = self.tape.shape_of(x[0])
            if lx.ndim > 2:
                # the last axis is the feature axis: x's leading dims + (outputs,)
                var ld = rn_dims(lx)
                ld[lx.ndim - 1] = self.tape.cols(t)
                t = rn_as_shape(self.tape, self.ex, t, rn_shape(ld, "ops: linear"))
        elif code == OPS_LAYER_NORM:
            _need("layer_norm", n, 3, 0, ni, 1, nf)
            t = layer_norm(self.tape, self.ex, x[0], x[1], x[2], fs[0])
        elif code == OPS_RMS_NORM:
            _need("rms_norm", n, 2, 0, ni, 1, nf)
            t = graph_rms_norm(self.tape, self.ex, x[0], x[1], fs[0])
        elif code == OPS_TRANSFORMER_SOFTMAX:
            _need("transformer_softmax", n, 1, 0, ni, 0, nf)
            t = graph_transformer_softmax(self.tape, self.ex, x[0])
        elif code == OPS_TRANSFORMER_LINEAR:
            _need("transformer_linear", n, 2, 0, ni, 0, nf)
            t = graph_transformer_linear(self.tape, self.ex, x[0], x[1])
        elif code == OPS_ROTARY:
            _need("rotary", n, 1, 2, ni, 1, nf)
            t = graph_rotary(self.tape, self.ex, x[0], ints[0], ints[1], fs[0])
        elif code == OPS_DROPOUT:
            _need("dropout", n, 1, 3, ni, 1, nf)
            t = self.tape.dropout(self.ex, x[0], fs[0], _seed(ints[0], ints[1]), ints[2])
        elif code == OPS_GATHER_ROWS:
            # The gather kernel (OP_GATHER_ROWS) reads X[idx] unchecked, so the
            # indices must be a permutation output: every value lies in
            # [0, its row count), checked here against X's rows, no download.
            _need("gather_rows", n, 2, 2, ni, 0, nf)
            if self.tape.ndim(x[0]) != 2:
                raise Error("ops: gather_rows takes a 2-D tensor")
            var gx = self._slot(hs[0])
            var gj = self._slot(hs[1])
            var g_bound = self.index_rows[gj]
            if g_bound == 0 or self.cols[gj] != 1:
                raise Error("ops: gather_rows takes row indices made by permutation")
            if g_bound > self.rows[gx]:
                raise Error("ops: the indices address more rows than the tensor has")
            if self.tape.is_recording() and self.tape.requires_grad(x[0]):
                # a gradient through a gather needs a scatter-add VJP (not on the tape yet)
                raise Error("ops: gather_rows carries no gradient; detach the input first")
            var g_start = ints[0]
            var g_count = ints[1] if ints[1] > 0 else self.rows[gj] - g_start
            if g_start < 0 or g_count < 1 or g_start + g_count > self.rows[gj]:
                raise Error("ops: gather_rows start and count must lie within the indices")
            var g_cols = self.cols[gx]
            var g_dst = self.tape.reserve(self.ex, g_count * g_cols)
            var g_src = self.tape.value_of(self.ex, x[0])
            var g_idx = self.tape.value_of(self.ex, x[1])
            gather_rows(self.ex, g_src, g_idx, g_start, g_count, g_cols, g_dst)
            return self._constant(g_dst, g_count, g_cols, 0, self.int_words[gx])
        elif code == OPS_PERMUTATION:
            _need("permutation", n, 0, 4, ni, 0, nf)
            var p_rows = ints[0]
            var p_dst = self.tape.reserve(self.ex, p_rows if p_rows > 0 else 1)
            permutation(self.ex, _seed(ints[1], ints[2]), ints[3], p_rows, p_dst)
            return self._constant(p_dst, p_rows, 1, p_rows)
        elif code == OPS_MSE:
            _need("mse", n, 2, 1, ni, 0, nf)
            t = mse(self.tape, self.ex, x[0], x[1], ints[0])
        elif code == OPS_MAE:
            _need("mae", n, 2, 1, ni, 0, nf)
            t = mae(self.tape, self.ex, x[0], x[1], ints[0])
        elif code == OPS_HUBER:
            _need("huber", n, 2, 1, ni, 1, nf)
            t = huber(self.tape, self.ex, x[0], x[1], fs[0], ints[0])
        elif code == OPS_BCE_LOGITS:
            _need("bce_with_logits", n, 2, 1, ni, 0, nf)
            t = binary_cross_entropy_with_logits(self.tape, self.ex, x[0], x[1], ints[0])
        elif code == OPS_CROSS_ENTROPY:
            _need("cross_entropy", n, 2, 1, ni, 0, nf)
            if self.tape.ndim(x[0]) != 2 or self.tape.ndim(x[1]) != 2:
                # logits [..., classes] and codes of the leading shape: the
                # loss's (rows, classes) and (rows, 1) views (free)
                var ce_rows = self.tape.rows(x[0])
                var ce_cols = self.tape.cols(x[0])
                if self.tape.shape_of(x[1]).count() != ce_rows:
                    raise Error("ops: cross_entropy takes one class code per row of logits")
                var ce_z = rn_as_shape(self.tape, self.ex, x[0], ShapeN.matrix(ce_rows, ce_cols))
                var ce_c = rn_as_shape(self.tape, self.ex, x[1], ShapeN.matrix(ce_rows, 1))
                t = cross_entropy(self.tape, self.ex, ce_z, ce_c, ints[0])
            else:
                t = cross_entropy(self.tape, self.ex, x[0], x[1], ints[0])
        elif code == OPS_DETACH:
            _need("detach", n, 1, 0, ni, 0, nf)
            t = self.tape.detach(self.ex, x[0])
        elif code == OPS_FULL:
            if n != 0 or nf < 1 or ni < 1:
                raise Error("ops: full takes a shape and a value")
            if ni == 2:
                var f_rows = ints[0]
                var f_cols = ints[1]
                if f_rows <= 0 or f_cols <= 0 or f_rows > ((1 << 60) // 4) // f_cols:
                    raise Error("ops: full needs positive dimensions")
                var f_dst = self.tape.reserve(self.ex, f_rows * f_cols)
                var fa = Args()
                fa.p0 = f_dst
                fa.f0 = fs[0]
                self.ex.launch[OP_FILL](fa, f_rows * f_cols)
                return self._constant(f_dst, f_rows, f_cols, 0)
            # any other rank: the dims, one fill, a free view of the shape
            var fn_shape = rn_shape(ints, "ops: full")
            var fn_count = fn_shape.count()
            if fn_count > (1 << 60) // 4:
                raise Error("ops: full needs positive dimensions")
            var fn_dst = self.tape.reserve(self.ex, fn_count)
            var fna = Args()
            fna.p0 = fn_dst
            fna.f0 = fs[0]
            self.ex.launch[OP_FILL](fna, fn_count)
            var fn_t = self.tape.variable(self.ex, fn_dst, fn_shape.rows(), fn_shape.cols(), False)
            t = rn_as_shape(self.tape, self.ex, fn_t, fn_shape)
        elif code == OPS_EMBEDDING:
            # lane ag-embedding-op: table[ids] (Tape.embedding). ints: the
            # caller's ids as an input descriptor (lane ag-thin-glue2: address,
            # NumPy dtype number, item size, rows, cols, row and column byte
            # strides; any integer dtype and strides, flattened row-major) and
            # padding_idx (-1 none). The ids are cast on the executor to int32
            # words (autograd/cast_ops.mojo upload_cast_i32; values outside
            # int32 saturate and the embedding's id check refuses them) in a
            # session buffer that lives with the graph: it is returned at the
            # next reset or close, like a freed upload.
            # Lane ag-program-embedding: with two tensors (table, ids) the
            # ids are a tensor of int32 words (upload_ids) and ints is
            # [padding_idx]: the same words reach the same Tape.embedding, and
            # a program records the ids as a value.
            if n == 2:
                _need("embedding", n, 2, 1, ni, 0, nf)
                var e_tids = self._ids_of(x[1])
                t = self.tape.embedding(self.ex, x[0], e_tids, ints[0])
            else:
                _need("embedding", n, 1, 8, ni, 0, nf)
                var e_src = CastIn(
                    ints[0], cast_in_kind(ints[1], ints[2], "ops: embedding ids"), ints[2],
                    ints[3], ints[4], ints[5], ints[6],
                )
                e_src.check("ops: embedding ids")
                var e_n = e_src.count()
                var e_buf = self.ex.alloc(e_n)
                self.retired.append(Int(e_buf))
                upload_cast_i32(self.ex, e_buf, e_src, "ops: embedding ids")
                var e_ids = self.tape.ids(self.ex, e_buf, e_n)
                t = self.tape.embedding(self.ex, x[0], e_ids, ints[7])
        elif code == OPS_COPY:
            _need("copy", n, 1, 0, ni, 0, nf)
            var cx = self._slot(hs[0])
            var c_count = self.rows[cx] * self.cols[cx]
            var c_rows = self.rows[cx]
            var c_cols = self.cols[cx]
            var c_shape = self.shapes[cx]
            var c_dst = self.tape.reserve(self.ex, c_count)
            var ca = Args()
            ca.p0 = self.tape.value_of(self.ex, x[0])
            ca.p1 = c_dst
            self.ex.launch[OP_COPY](ca, c_count)
            if c_shape.ndim == 2:
                return self._constant(c_dst, c_rows, c_cols, 0, self.int_words[cx])
            var c_t = self.tape.variable(self.ex, c_dst, c_rows, c_cols, False)
            t = rn_as_shape(self.tape, self.ex, c_t, c_shape)
        elif code == OPS_RESHAPE_N:
            t = self._reshape_n(x, ints)
        elif code == OPS_PERMUTE:
            if n != 1 or ni < 1:
                raise Error("ops: permute takes one tensor and its axes")
            if ints[0] == 0:
                var p_axes = List[Int](capacity=ni - 1)
                for k in range(1, ni):  # small-loop(ni: at most 8 axes): argument parsing, no data
                    p_axes.append(ints[k])
                var p_nd = self.tape.ndim(x[0])
                t = self.tape.permute(self.ex, x[0], rn_check_permutation(p_axes, p_nd, "ops: permute"))
            else:
                _need("transpose", n, 1, 3, ni, 0, nf)
                t = self._swap(x[0], ints[1], ints[2])
        elif code == OPS_SUM_N or code == OPS_MEAN_N:
            if n != 1 or ni < 1:
                raise Error("ops: sum and mean take one tensor, keepdims and the axes")
            t = self._reduce_n(x[0], ints, code == OPS_MEAN_N, False)
        elif code == OPS_MAX:
            if n != 1 or ni < 1:
                raise Error("ops: max takes one tensor, keepdims and the axes")
            t = self._reduce_n(x[0], ints, False, True)
        elif code == OPS_CONCAT:
            _need("concat", 1, 1, 1, ni, 0, nf)
            if n < 1:
                raise Error("ops: concat takes at least one tensor")
            t = self._concat(x, ints[0])
        elif code == OPS_NARROW:
            if n != 1 or ni < 4:
                raise Error("ops: split takes one tensor, the axis, the piece and the sizes or sections")
            t = self._piece(x[0], ints)
        elif code == OPS_TAKE:
            # lane ag-rankn-python: x's slices at integer ids along an axis
            # (plan 2.6). The ids arrive as embedding's do: an input
            # descriptor cast on the executor to int32 words in a session
            # buffer returned at the next reset or close. With two tensors
            # (x, ids; lane ag-program-embedding) the ids are a tensor of
            # int32 words and ints is [axis], as embedding's.
            if n == 2:
                _need("take", n, 2, 1, ni, 0, nf)
                var k_tids = self._ids_of(x[1])
                var kt_nd = self.tape.ndim(x[0])
                var kt_axis = rn_normalize_axis(ints[0], kt_nd if kt_nd > 0 else 1)
                t = rankn_take(self.tape, self.ex, x[0], k_tids, kt_axis)
            else:
                _need("take", n, 1, 8, ni, 0, nf)
                var k_src = CastIn(
                    ints[0], cast_in_kind(ints[1], ints[2], "ops: take indices"), ints[2],
                    ints[3], ints[4], ints[5], ints[6],
                )
                k_src.check("ops: take indices")
                var k_n = k_src.count()
                var k_buf = self.ex.alloc(k_n)
                self.retired.append(Int(k_buf))
                upload_cast_i32(self.ex, k_buf, k_src, "ops: take indices")
                var k_ids = self.tape.ids(self.ex, k_buf, k_n)
                var k_nd = self.tape.ndim(x[0])
                var k_axis = rn_normalize_axis(ints[7], k_nd if k_nd > 0 else 1)
                t = rankn_take(self.tape, self.ex, x[0], k_ids, k_axis)
        elif code == OPS_SOFTMAX_N or code == OPS_LOG_SOFTMAX_N:
            _need("softmax", n, 1, 1, ni, 0, nf)
            t = rn_softmax_along(self.tape, self.ex, x[0], ints[0], code == OPS_LOG_SOFTMAX_N)
        elif code == OPS_CONV2D:
            if n != 2 and n != 3:
                raise Error("ops: conv2d takes an input, a weight and an optional bias")
            _need("conv2d", 2, 2, 2, ni, 0, nf)
            if self.tape.ndim(x[0]) != 4 or self.tape.ndim(x[1]) != 4:
                raise Error("ops: conv2d takes x (N, C, H, W) and weight (out_channels, C, kh, kw)")
            if ints[0] < 1 or ints[1] < 0:
                raise Error("ops: conv2d needs stride >= 1 and padding >= 0")
            # the optional third integer: groups (1 when absent)
            var cg = ints[2] if ni > 2 else 1
            if cg < 1:
                raise Error("ops: conv2d needs groups >= 1")
            if n == 3:
                var oc = self.tape.dim(x[1], 0)
                if self.tape.shape_of(x[2]).count() != oc:
                    raise Error("ops: conv2d: the bias needs one value per output channel")
                var oc_dims = List[Int]()
                oc_dims.append(oc)
                var cb = rn_as_shape(self.tape, self.ex, x[2], rn_shape(oc_dims, "ops: conv2d"))
                t = rankn_conv2d(self.tape, self.ex, x[0], x[1], Optional[Tensor](cb), ints[0], ints[1], cg)
            else:
                t = rankn_conv2d(self.tape, self.ex, x[0], x[1], None, ints[0], ints[1], cg)
        elif code == OPS_MAXPOOL2D or code == OPS_AVGPOOL2D:
            _need("pool2d", n, 1, 2, ni, 0, nf)
            if self.tape.ndim(x[0]) != 4:
                raise Error("ops: pooling takes x (N, C, H, W)")
            if ints[0] < 1 or ints[1] < 1:
                raise Error("ops: pooling needs kernel_size >= 1 and stride >= 1")
            if code == OPS_MAXPOOL2D:
                t = rankn_max_pool2d(self.tape, self.ex, x[0], ints[0], ints[1])
            else:
                t = rankn_avg_pool2d(self.tape, self.ex, x[0], ints[0], ints[1])
        elif code >= OPS_IMP_FIRST and code <= OPS_IMP_LAST:
            # lane ag-import-kernels: the import elementwise family
            t = self._import_op(code, x, ints, fs)
        # nn-blocks-scatter begin
        elif code == OPS_SCATTER_ADD:
            # [x, index (an ids tensor of src's shape), src], ints [dim] (autograd/nn_scatter.mojo)
            _need("scatter_add", n, 3, 1, ni, 0, nf)
            if self.int_words[self._slot(hs[0])] or self.int_words[self._slot(hs[2])]:
                raise Error("ValueError: ops: scatter_add: x and src must be floating-point tensors")
            var sc_slot = self._slot(hs[1])
            if not self.int_words[sc_slot]:
                raise Error("ValueError: ops: scatter_add: index must be an integer tensor made with Session.ids")
            t = nn_scatter_add(self.tape, self.ex, x[0], self.tape.value_of(self.ex, x[1]), self.shapes[sc_slot],
                               x[2], ints[0])
        elif code == OPS_SEGMENT_SUM or code == OPS_SEGMENT_MEAN or code == OPS_SEGMENT_MAX:
            # [data, segment_ids (an ids tensor, one id per row of data)], ints [num_segments]
            var sg_name = segment_op_name(code - OPS_SEGMENT_SUM)
            _need(sg_name, n, 2, 1, ni, 0, nf)
            if self.int_words[self._slot(hs[0])]:
                raise Error("ValueError: ops: " + sg_name + ": data must be a floating-point tensor")
            var sg_slot = self._slot(hs[1])
            if not self.int_words[sg_slot]:
                raise Error("ValueError: ops: " + sg_name + ": segment_ids must be an integer tensor made with Session.ids")
            t = nn_segment(self.tape, self.ex, x[0], self.tape.value_of(self.ex, x[1]), self.shapes[sg_slot].count(),
                           ints[0], code - OPS_SEGMENT_SUM)
        # nn-blocks-scatter end
        # nn-blocks-qkv begin: attention over q, k, v (reached only by codes no branch above took; nn-blocks-zerocost
        # moved it here from ahead of the primitive checks, where every later operation paid its compare)
        elif code == OPS_ATTENTION_QKV:
            return self._attention_qkv(hs, ints, fs, x)
        # nn-blocks-qkv end
        elif code == OPS_WEIGHTED_LOSS:
            # Lane rnn-parity-B: [prediction, target, weight], ints [loss code 24..28,
            # reduction], floats [delta]; reached only when weight= is given
            # (autograd/per_step_ops.mojo). nn-blocks (2026-11-25): moved from
            # mid-chain (after OPS_CROSS_ENTROPY, where every later operation paid
            # its compare) to the chain's tail, which no older operation reaches.
            _need("weighted loss", n, 3, 2, ni, 1, nf)
            if self.int_words[self._slot(hs[2])]:
                raise Error("ValueError: ops: a loss weight must be a floating-point tensor")
            t = weighted_loss(self.tape, self.ex, x[0], x[1], x[2], ints[0], ints[1], fs[0])
        else:
            # nn-blocks-scan / nn-blocks-cbw begin: the scan over a captured cell and a user-defined backward rule,
            # on the chain's no-match path (no other operation pays a compare for them). Both run captured nodes
            # (a cell, a custom op's programs) through the `nested` instance, which has neither branch, so no
            # function reaches itself (AGENTS.md part 6); cbw_bind and scan_check_cell refuse the nesting itself.
            comptime if not nested:
                if code == OPS_SCAN:
                    return self._scan(hs, ints, fs)
                if code == OPS_CUSTOM_VJP:
                    return self._cbw_forward(hs, ints, fs, x)
            # nn-blocks-scan / nn-blocks-cbw end
            # session-ops begin: Sequential's layers as Session ops and the round's additions (codes 130..189), on the no-match path as well;
            # plain operations, so a scan cell or a custom operation's program (`nested`) may use them
            if code >= OPS_LAYER_FIRST and code <= OPS_LAYER_LAST:
                return self._layer_op(code, hs, ints, fs, x)
            # session-ops end
            raise Error("ops: unknown operation code " + String(code))
        if n >= 1 and _keeps_shape(code):
            # elementwise and row-wise ops keep x's full shape (a free view
            # when the op records only the matrix view)
            var ks = self.tape.shape_of(x[0])
            if ks.ndim != 2:
                t = rn_as_shape(self.tape, self.ex, t, ks)
        var result = self._add(t, 0, False, 0)
        self.int_words[self._slot(result)] = self._result_int_words(code, hs, ints)
        return result

    # ------------------------------------------------------------ the import family
    def _block_step(mut self, code: Int, hs: List[Int], ints: List[Int], fs: List[Float32], x: List[Tensor]) raises -> Int:
        """Block step op `code`: operands [x, weights, state, (words)]
        checked here (x float; weights an upload of floats; state and words
        declared state of the right kind), then the table's run on the
        device (lane x7b: the host arm on a host session), the result a new tensor of x's shape. The state is written
        in place. In a recording session (lanes x7, x56d) every step op is the zero-state prefill with its stages
        kept, recorded as one custom node over [x, weights] (StepOpRules differentiates it)."""
        var info = block_step_check(code, len(x), ints, fs)
        _ = info
        var recording = self.tape.is_recording()
        var bufs = List[FP](capacity=len(x))
        var counts = List[Int](capacity=len(x))
        for i in range(len(x)):  # small-loop(x: the op's three or four operands): operand checks, no data
            var slot = self._slot(hs[i])
            var buf = self.tape.value_of(self.ex, x[i])
            var kind = self._buffer_kind_of(Int(buf))
            if i == 0:
                if self.int_words[slot]:
                    raise Error("ValueError: ops: " + String(info.name) + ": x must hold floating-point values")
            elif i == 1:
                if kind == 0 or self.int_words[slot]:
                    raise Error("ValueError: ops: " + String(info.name)
                                + ": the weights must be one floating-point tensor made with upload_block (or tensor())")
            elif i == 2:
                if kind != OPS_BUFFER_STATE:
                    raise Error("ValueError: ops: " + String(info.name)
                                + ": the state must be a floating-point tensor declared state")
            elif kind != OPS_BUFFER_STATE_WORDS:
                raise Error("ValueError: ops: " + String(info.name)
                            + ": the state words must be an integer tensor declared state")
            bufs.append(buf)
            counts.append(self.shapes[slot].count())
        var sh = self.shapes[self._slot(hs[0])]
        var out = self.tape.reserve(self.ex, sh.count())
        if recording:
            # Lane x7: a recorded attention step (Program.fit's replay) is
            # the zero-state prefill with its stages kept, recorded as one
            # custom node over [x, weights]; its VJP is StepOpRules'
            # (autograd/rules_step_ops.mojo), keyed by the operands' storage.
            # Lane x56d: the Mamba steps the same way (tags 109-111).
            comptime if Self.E.IS_DEVICE:
                if code == OPS_ATTN_STEP:
                    _ = attn_step_train_run(rebind[DeviceExec](self.ex), self.steps, bufs, counts, ints, out)
                else:
                    _ = mamba_step_train_run(rebind[DeviceExec](self.ex), self.steps, code, bufs, counts, ints, fs,
                                             out)
            else:
                # lane x7b: the executor's host arm, the same zero-state
                # prefill over the host block body with its stages kept
                # (autograd/block_step_host.mojo, through Exec.step_run).
                self.ex.step_run(code, bufs, counts, ints, fs, out, True)
            var saved = Args()
            saved.p0 = bufs[1]
            saved.p1 = bufs[2]
            # Mamba-1 carries no words tensor: p2 keeps Args' placeholder (a
            # Pointer is non-nullable, so no address-0 FP)
            if len(bufs) > 3:
                saved.p2 = bufs[3]
            var rx = List[Tensor]()
            rx.append(x[0])
            rx.append(x[1])
            var rt = self.tape.custom_nd(self.ex, rx, out, sh, step_train_tag(code), saved)
            return self._add(rt, 0, self.tape.requires_grad(rt), 0)
        comptime if Self.E.IS_DEVICE:
            block_step_run(rebind[DeviceExec](self.ex), self.steps, code, bufs, counts, ints, fs, out)
        else:
            self.ex.step_run(code, bufs, counts, ints, fs, out, False)
        var t: Tensor
        if sh.ndim == 2:
            t = self.tape.variable(self.ex, out, sh.d[0], sh.d[1], False)
        else:
            t = self.tape.variable_nd(self.ex, out, sh, False)
        return self._add(t, 0, False, 0)

    def _lm_head_v2(mut self, hs: List[Int], ints: List[Int], fs: List[Float32], x: List[Tensor]) raises -> Int:
        """OPS_LM_HEAD_V2: [hidden (rows x width, any rank), weight (vocab,
        width), codes (rows class values)] -> the (1, 1) mean loss. Recorded:
        one custom node over [hidden, weight] (LM_HEAD_V2_TAG, VJP in
        StepOpRules); the row statistics are tape storage until the reset."""
        _need("lm_head_v2", len(x), 3, 0, len(ints), 0, len(fs))
        var hslot = self._slot(hs[0])
        var wslot = self._slot(hs[1])
        var cslot = self._slot(hs[2])
        if self.int_words[hslot] or self.int_words[wslot] or self.int_words[cslot]:
            raise Error("ValueError: ops: lm_head_v2 takes floating-point hidden, weight and class-value codes")
        var wshape = self.shapes[wslot]
        if wshape.ndim != 2:
            raise Error("ValueError: ops: lm_head_v2: the weight must be (vocab, width)")
        var vocab = wshape.d[0]
        var width = wshape.d[1]
        var count = self.shapes[hslot].count()
        if width < 1 or count % width != 0:
            raise Error("ValueError: ops: lm_head_v2: the hidden rows must have the weight's width")
        var rows = count // width
        if self.shapes[cslot].count() != rows:
            raise Error("ValueError: ops: lm_head_v2 takes one class code per hidden row")
        # lane neural-blocks-3-lm: the op's one recording (autograd/lm_head_v2_graph.mojo), shared with the
        # Sequential lm_head layer's token cross-entropy
        var rt = graph_lm_head_v2(self.tape, self.ex, x[0], x[1], x[2], rows, vocab, width)
        if not self.tape.is_recording():
            return self._add(rt, 0, False, 0)
        return self._add(rt, 0, self.tape.requires_grad(rt), 0)

    def _small_mlp(mut self, hs: List[Int], ints: List[Int], fs: List[Float32], x: List[Tensor]) raises -> Int:
        """OPS_SMALL_MLP: [x (rows x in, any rank), w1 (hid, in), b1 (hid
        values), w2 (out, hid), b2 (out values)] -> the (rows, out) logits of
        linear(relu(linear(x, w1^T, b1)), w2^T, b2). Recorded: one custom
        node over the five operands (SMALL_MLP_TAG, VJP in StepOpRules); act,
        the backward scratch and the workspace are tape storage until the
        reset."""
        _need("small_mlp", len(x), 5, 0, len(ints), 0, len(fs))
        var slots = List[Int]()
        for i in range(5):  # small-loop(five operands): handle lookup, no data
            var sl = self._slot(hs[i])
            if self.int_words[sl]:
                raise Error("ValueError: ops: small_mlp takes floating-point operands")
            slots.append(sl)
        var s1 = self.shapes[slots[1]]
        var s3 = self.shapes[slots[3]]
        if s1.ndim != 2 or s3.ndim != 2:
            raise Error("ValueError: ops: small_mlp: the weights must be (hidden, inputs) and (outputs, hidden)")
        var hid = s1.d[0]
        var n_in = s1.d[1]
        var n_out = s3.d[0]
        if s3.d[1] != hid:
            raise Error("ValueError: ops: small_mlp: the second weight must take the hidden width")
        if self.shapes[slots[2]].count() != hid or self.shapes[slots[4]].count() != n_out:
            raise Error("ValueError: ops: small_mlp: each bias must hold one value per output of its layer")
        var count = self.shapes[slots[0]].count()
        if n_in < 1 or count % n_in != 0:
            raise Error("ValueError: ops: small_mlp: the input rows must have the first weight's width")
        var rows = count // n_in
        small_mlp_check(rows, n_in, hid, n_out)
        var recording = self.tape.is_recording()
        var ws_n = small_mlp_workspace_floats(rows, n_in, hid, n_out, recording)
        var vals = List[FP]()
        for i in range(5):  # small-loop(five operands): value pointers, no data
            vals.append(self.tape.value_of(self.ex, x[i]))
        var pre1 = self.tape.reserve(self.ex, rows * hid)
        var act = self.tape.reserve(self.ex, rows * hid)
        var pre2 = self.tape.reserve(self.ex, rows * n_out)
        var logits = self.tape.reserve(self.ex, rows * n_out)
        var ws = self.tape.reserve(self.ex, ws_n)
        var fb: List[FP] = [vals[0], vals[1], vals[2], vals[3], vals[4], pre1, act, pre2, logits, ws]
        var fd: List[Int] = [rows, n_in, hid, n_out, ws_n]
        comptime if Self.E.IS_DEVICE:
            small_mlp_forward_device(rebind[DeviceExec](self.ex), fb, fd)
        else:
            # the host arm (autograd/small_mlp_host.mojo) through the executor
            self.ex.small_mlp_host(False, fb, fd)
        if not recording:
            var t = self.tape.variable(self.ex, logits, rows, n_out, False)
            return self._add(t, 0, False, 0)
        var saved = Args()
        saved.p0 = act
        saved.p1 = self.tape.reserve(self.ex, rows * hid)
        saved.p2 = self.tape.reserve(self.ex, rows * hid)
        saved.p3 = ws
        saved.i0 = rows
        saved.i1 = n_in
        saved.i2 = hid
        saved.i3 = n_out
        saved.i4 = ws_n
        saved.i5 = 1 if self.tape.requires_grad(x[0]) else 0
        var rx = List[Tensor]()
        for i in range(5):  # small-loop(five operands): the node's parents
            rx.append(x[i])
        var rt = self.tape.custom(self.ex, rx, logits, rows, n_out, SMALL_MLP_TAG, saved)
        return self._add(rt, 0, self.tape.requires_grad(rt), 0)

    def _cnn_conv_block(mut self, hs: List[Int], ints: List[Int], fs: List[Float32], x: List[Tensor]) raises -> Int:
        """OPS_CNN_CONV_BLOCK: [x (N, C, H, W), w (OC, C, k, k), b (1, OC)],
        ints [pad, pool (0 / 1), pool size] -> maxpool(relu(conv(x) + b))
        (stride 1), or relu(conv(x) + b) without the pool. Recorded: one
        custom node over [x, w, b] (CNN_CONV_BLOCK_TAG, VJP in StepOpRules);
        the pool winners and the saved im2col / conv output are tape storage
        until the reset (autograd/cnn_block_op.mojo)."""
        _need("cnn_conv_block", len(x), 3, 3, len(ints), 0, len(fs))
        var xslot = self._slot(hs[0])
        var wslot = self._slot(hs[1])
        var bslot = self._slot(hs[2])
        if self.int_words[xslot] or self.int_words[wslot] or self.int_words[bslot]:
            raise Error("ValueError: ops: cnn_conv_block takes floating-point inputs")
        var xs = self.shapes[xslot]
        var ws = self.shapes[wslot]
        if xs.ndim != 4 or ws.ndim != 4:
            raise Error("ValueError: ops: cnn_conv_block takes (N, C, H, W) images and an (OC, C, k, k) weight")
        var n = xs.d[0]
        var c = xs.d[1]
        var oc = ws.d[0]
        var ks = ws.d[2]
        if ws.d[1] != c or ws.d[3] != ks:
            raise Error("ValueError: ops: cnn_conv_block: the weight must be (OC, C, k, k) over the images' channels")
        if self.shapes[bslot].count() != oc:
            raise Error("ValueError: ops: cnn_conv_block takes one bias per output channel")
        var pad = ints[0]
        var pool = ints[1] != 0
        var ps = ints[2]
        if pad < 0 or (ints[1] != 0 and ints[1] != 1) or (pool and ps < 1):
            raise Error("ValueError: ops: cnn_conv_block takes ints [pad >= 0, pool 0 or 1, pool size >= 1]")
        var wd = cnn_block_words(n, c, xs.d[2], xs.d[3], oc, ks, pad, pool, ps)
        var cprm = wd[0].copy()
        var pprm = wd[1].copy()
        var od = cnn_block_out_dims(cprm, pprm, pool)
        var count = n * oc * od[0] * od[1]
        var xp = self.tape.value_of(self.ex, x[0])
        var wp = self.tape.value_of(self.ex, x[1])
        var bp = self.tape.value_of(self.ex, x[2])
        var dst = self.tape.reserve(self.ex, count)
        # the pool winners (int32 words in float storage of the same width)
        var idx = self.tape.reserve(self.ex, count if pool else 1)
        var recording = self.tape.is_recording()
        var keep = recording and cnn_block_keep(cprm, Self.E.IS_DEVICE)
        var sz = cnn_block_saved_sizes(cprm, Self.E.IS_DEVICE)
        # FP is non-nullable: without the saves, cols / yconv hold dst as a
        # placeholder that nothing reads (i10 = 0 says so), and the device
        # entry is handed address 0 ("not saved"), as before.
        var cols = dst
        var yconv = dst
        if keep:
            cols = self.tape.reserve(self.ex, sz[0])
            yconv = self.tape.reserve(self.ex, sz[1])
        comptime if Self.E.IS_DEVICE:
            # a public session: its queue is not the x_cnn entries' (BlockStepScratch). It drains its own queue
            # so the entry reads finished operands, and the entry waits (no may_defer) so that this session's next
            # op, download or reset, on its own queue, reads finished results (autograd/cnn_block_op.mojo Waits).
            var other = self.steps.cnn_other_queue
            if other:
                self.ex.sync()
            cnn_block_forward_device(
                xp, wp, bp, cprm, pprm, pool, dst, idx, Int(cols) if keep else 0, Int(yconv) if keep else 0, not other
            )
        else:
            # the host arm (autograd/cnn_block_host.mojo) through the executor
            var hcols = cols
            var hy = yconv
            var hb: List[FP] = [xp, wp, bp, dst, idx, hcols, hy]
            var hw: List[Int] = [1 if pool else 0, 1 if keep else 0]
            self.ex.cnn_op_host(CNN_HOST_BLOCK_FORWARD, hb, cprm, pprm, hw)
        var od_list: List[Int] = [n, oc, od[0], od[1]]
        var osh = rn_shape(od_list, "ops: cnn_conv_block")
        if not recording:
            var t = self.tape.variable_nd(self.ex, dst, osh, False)
            return self._add(t, 0, False, 0)
        var saved = Args()
        saved.p0 = idx
        saved.p1 = cols
        saved.p2 = yconv
        saved.i0 = n
        saved.i1 = c
        saved.i2 = xs.d[2]
        saved.i3 = xs.d[3]
        saved.i4 = oc
        saved.i5 = ks
        saved.i6 = pad
        saved.i7 = 1 if pool else 0
        saved.i8 = ps
        saved.i9 = 1 if self.tape.requires_grad(x[0]) else 0
        saved.i10 = 1 if keep else 0
        var rx = List[Tensor]()
        rx.append(x[0])
        rx.append(x[1])
        rx.append(x[2])
        var rt = self.tape.custom_nd(self.ex, rx, dst, osh, CNN_CONV_BLOCK_TAG, saved)
        return self._add(rt, 0, self.tape.requires_grad(rt), 0)

    def _cnn_linear(mut self, hs: List[Int], ints: List[Int], fs: List[Float32], x: List[Tensor]) raises -> Int:
        """OPS_CNN_LINEAR: [h (n, d_in), w (d_out, d_in), b (1, d_out)] ->
        h w^T + b: CNNClassifier's head on the CNN's pinned GEMM plan
        (autograd/cnn_block_op.mojo). Recorded: one custom node over
        [h, w, b] (CNN_LINEAR_TAG, VJP in StepOpRules)."""
        _need("cnn_linear", len(x), 3, 0, len(ints), 0, len(fs))
        var hslot = self._slot(hs[0])
        var wslot = self._slot(hs[1])
        var bslot = self._slot(hs[2])
        if self.int_words[hslot] or self.int_words[wslot] or self.int_words[bslot]:
            raise Error("ValueError: ops: cnn_linear takes floating-point inputs")
        var wsh = self.shapes[wslot]
        if wsh.ndim != 2:
            raise Error("ValueError: ops: cnn_linear: the weight must be (outputs, inputs)")
        var d_out = wsh.d[0]
        var d_in = wsh.d[1]
        var count = self.shapes[hslot].count()
        if d_in < 1 or count % d_in != 0:
            raise Error("ValueError: ops: cnn_linear: the input rows must have the weight's width")
        if self.shapes[bslot].count() != d_out:
            raise Error("ValueError: ops: cnn_linear takes one bias per output")
        var n = count // d_in
        var hp = self.tape.value_of(self.ex, x[0])
        var wp = self.tape.value_of(self.ex, x[1])
        var bp = self.tape.value_of(self.ex, x[2])
        var dst = self.tape.reserve(self.ex, n * d_out)
        comptime if Self.E.IS_DEVICE:
            # a public session: drain its own queue first, and the entry waits for its results (as conv_block)
            var other = self.steps.cnn_other_queue
            if other:
                self.ex.sync()
            cnn_linear_forward_device(hp, wp, bp, n, d_in, d_out, dst, not other)
        else:
            var hb: List[FP] = [hp, wp, bp, dst]
            var hw: List[Int] = [n, d_in, d_out]
            self.ex.cnn_op_host(CNN_HOST_LINEAR_FORWARD, hb, List[Int32](), List[Int32](), hw)
        if not self.tape.is_recording():
            var t = self.tape.variable(self.ex, dst, n, d_out, False)
            return self._add(t, 0, False, 0)
        var saved = Args()
        saved.i0 = n
        saved.i1 = d_in
        saved.i2 = d_out
        var rx = List[Tensor]()
        rx.append(x[0])
        rx.append(x[1])
        rx.append(x[2])
        var rt = self.tape.custom(self.ex, rx, dst, n, d_out, CNN_LINEAR_TAG, saved)
        return self._add(rt, 0, self.tape.requires_grad(rt), 0)

    def _rnn_scan(mut self, hs: List[Int], ints: List[Int], fs: List[Float32], x: List[Tensor]) raises -> Int:
        """OPS_RNN_SCAN: [x (steps * batch rows of inputs, time-major),
        weight_ih (G H, inputs), weight_hh (G H, H), bias_ih (G H values),
        bias_hh (G H values)], ints [cell, steps, batch] -> h_1..h_T as a
        (steps * batch, H) tensor, from a zero initial state. The trainer's
        layer (`rnn_scan_op`); recorded as its two custom nodes, which
        StepOpRules hands to RecurrentRules."""
        if len(ints) >= 8:
            # Lane rnn-parity-B: the layer's options (h_0, c_0, lengths, reverse, the final cell state);
            # a call without them sends three integers and runs the words below unchanged.
            return self._rnn_scan_options(hs, ints, fs, x)
        _need("rnn_scan", len(x), 5, 3, len(ints), 0, len(fs))
        for i in range(5):  # small-loop(five operands): operand kinds, no data
            if self.int_words[self._slot(hs[i])]:
                raise Error("ValueError: ops: a recurrent layer takes floating-point tensors")
        var rt = rnn_scan_op(self.tape, self.ex, x[0], x[1], x[2], x[3], x[4], ints[0], ints[1], ints[2])
        return self._add(rt, 0, self.tape.requires_grad(rt), 0)

    def _rnn_scan_options(mut self, hs: List[Int], ints: List[Int], fs: List[Float32], x: List[Tensor]) raises -> Int:
        """OPS_RNN_SCAN with options (lane rnn-parity-B): ints [cell, steps, batch, reverse, has h_0, has c_0,
        has lengths, cell state out] and, after the five operands, h_0 (B x H values), c_0 (B x H, LSTM) and
        lengths (B float32 values, each in [1, steps]) when given. h_0 / c_0 are the initial state (no
        gradient reaches them); `lengths` zeroes each row's padded inputs (rnn_zero_padded) and the cell carries
        its state through them (rnn_scan_op); `reverse` walks time T - 1 down to 0, the output in time order.
        With `cell state out` (an LSTM) the result is ((T + 1) B, H): the T B output rows, then the final cell
        state's B rows (a constant), which the caller splits (autograd/ops_block_frontend.mojo session_rnn)."""
        var cell = ints[0]
        var T = ints[1]
        var B = ints[2]
        var reverse = ints[3] != 0
        var has_h0 = ints[4] != 0
        var has_c0 = ints[5] != 0
        var has_len = ints[6] != 0
        var cell_out = ints[7] != 0
        var want = 5 + (1 if has_h0 else 0) + (1 if has_c0 else 0) + (1 if has_len else 0)
        _need("rnn_scan", len(x), want, 8, len(ints), 0, len(fs))
        for i in range(want):  # small-loop(at most eight operands): operand kinds, no data
            if self.int_words[self._slot(hs[i])]:
                raise Error("ValueError: ops: a recurrent layer takes floating-point tensors")
        if (has_c0 or cell_out) and cell != CELL_LSTM:
            raise Error("ValueError: ops: only an LSTM has a cell state (c_0)")
        var H = self.tape.cols(x[2])
        var at = 5
        var h0 = Optional[FP](None)
        var c0 = Optional[FP](None)
        if has_h0:
            if self.tape.shape_of(x[at]).count() != B * H or self.tape.cols(x[at]) != H:
                raise Error("ValueError: ops: h_0 must be (batch, hidden): " + String(B) + " x " + String(H))
            h0 = Optional[FP](self.tape.value_of(self.ex, x[at]))
            at += 1
        if has_c0:
            if self.tape.shape_of(x[at]).count() != B * H or self.tape.cols(x[at]) != H:
                raise Error("ValueError: ops: c_0 must be (batch, hidden): " + String(B) + " x " + String(H))
            c0 = Optional[FP](self.tape.value_of(self.ex, x[at]))
            at += 1
            if not has_h0:
                # c_0 alone: the hidden state starts at zero (PyTorch's h_0 default)
                var hz = self.tape.reserve(self.ex, B * H)
                _seq_fill(self.ex, hz, B * H, Float32(0.0))
                h0 = Optional[FP](hz)
        var lens = Optional[FP](None)
        var xin = x[0]
        if has_len:
            if self.tape.shape_of(x[at]).count() != B:
                raise Error("ValueError: ops: lengths must hold one length per sequence (" + String(B) + " values)")
            var lp = self.tape.value_of(self.ex, x[at])
            lens = Optional[FP](lp)
            xin = rnn_zero_padded(self.tape, self.ex, x[0], lp, T, B)
        var c_last = Optional[FP](None)
        if cell_out:
            c_last = Optional[FP](self.tape.reserve(self.ex, B * H))
        var rt = rnn_scan_op(self.tape, self.ex, xin, x[1], x[2], x[3], x[4], cell, T, B, lens, reverse, h0, c0,
                             c_last)
        if cell_out:
            var ct = self.tape.variable(self.ex, c_last.value(), B, H, False)
            var parts = List[Tensor]()
            parts.append(rt)
            parts.append(ct)
            rt = rankn_concat(self.tape, self.ex, parts, 0)
        return self._add(rt, 0, self.tape.requires_grad(rt), 0)

    def _lm_embedding(mut self, hs: List[Int], ints: List[Int], fs: List[Float32], x: List[Tensor]) raises -> Int:
        """OPS_LM_EMBEDDING: [table (vocab, width), ids (n int32 words)],
        ints [padding_idx] -> the (n, width) rows. OPS_EMBEDDING's refusals,
        forward and words; recorded as one custom node over [table]
        (LM_EMBEDDING_TAG, VJP in StepOpRules: the trainer's embedding
        backward on a device session, the executor's on the host). The VJP's
        run scratch is tape storage until the reset."""
        _need("lm_embedding", len(x), 2, 1, len(ints), 0, len(fs))
        var tslot = self._slot(hs[0])
        var islot = self._slot(hs[1])
        if self.int_words[tslot] or not self.int_words[islot]:
            raise Error("ValueError: ops: lm_embedding takes a floating-point table and int32 ids")
        var tshape = self.shapes[tslot]
        if tshape.ndim != 2:
            raise Error("ValueError: ops: lm_embedding: the table must be (vocab, width)")
        var vocab = tshape.d[0]
        var width = tshape.d[1]
        var n = self.shapes[islot].count()
        var pad = ints[0]
        embedding_refuse_shape(n, vocab, width, pad)
        var tp = self.tape.value_of(self.ex, x[0])
        var ip = self.tape.value_of(self.ex, x[1])
        var cw = embedding_check_words(n)
        var scratch_a = self.tape.reserve(self.ex, cw)
        var scratch_b = self.tape.reserve(self.ex, cw)
        embedding_refuse_ids(self.ex, ip, n, vocab, scratch_a, scratch_b)
        var y = self.tape.reserve(self.ex, n * width)
        embedding_forward_into(self.ex, tp, ip, y, n, width)
        if not self.tape.is_recording():
            var t = self.tape.variable(self.ex, y, n, width, False)
            return self._add(t, 0, False, 0)
        # Run scratch for either arm: the device's counts | run_begin | perm,
        # the host's run_begin | perm | keys (autograd/embedding_ops.mojo).
        var words = lm_embedding_scratch_words(vocab, n)
        var host_words = vocab + 1 + 2 * embedding_perm_words(AG_EMB_PLAN_SORT, n)
        if host_words > words:
            words = host_words
        var saved = Args()
        saved.p0 = ip
        saved.p1 = self.tape.reserve(self.ex, words)
        saved.i0 = n
        saved.i1 = vocab
        saved.i2 = width
        saved.i3 = pad
        var rx = List[Tensor]()
        rx.append(x[0])
        var rt = self.tape.custom(self.ex, rx, y, n, width, LM_EMBEDDING_TAG, saved)
        return self._add(rt, 0, self.tape.requires_grad(rt), 0)

    # nn-blocks-qkv begin
    # session-ops begin: Sequential's layers as Session ops (autograd/session_layer_rules.mojo has the codes)
    def _layer_op(mut self, code: Int, hs: List[Int], ints: List[Int], fs: List[Float32], x: List[Tensor]) raises -> Int:
        """Session codes OPS_LAYER_FIRST..OPS_LAYER_LAST: the family's method records the SAME tape nodes as the
        Sequential layer (its forward function), and a recording session notes that its graph now holds a layer
        node, so its backward differentiates with LayerRules (tested once per backward call)."""
        var t: Tensor
        if code < OPS_LAYER_POOL_FIRST:
            t = self._layer_conv(code, hs, ints, fs, x)
        elif code < OPS_LAYER_RESIZE_FIRST:
            t = self._layer_pool(code, hs, ints, fs, x)
        elif code < OPS_LAYER_REST_FIRST:
            t = self._layer_resize(code, hs, ints, fs, x)
        elif code < OPS_LAYER_EXTRA_FIRST:
            t = self._layer_rest(code, hs, ints, fs, x)
        else:
            t = self._layer_extra(code, hs, ints, fs, x)
        if self.tape.is_recording():
            self.layer_nodes = True
        var result = self._add(t, 0, False, 0)
        # session-ops-extra: topk's indices are int32 words (an ids tensor); every other layer op returns floats
        self.int_words[self._slot(result)] = session_ops_extra_int_result(code)
        return result

    # session-ops-conv begin: codes 130..139 (sub-lane session-ops-conv replaces this body)
    def _layer_conv(mut self, code: Int, hs: List[Int], ints: List[Int], fs: List[Float32], x: List[Tensor]) raises -> Tensor:
        """The convolutions (autograd/session_ops_conv.mojo): conv1d, conv2d with dilation or per-axis words,
        conv3d, conv_transpose1d / 2d, each the forward function Sequential's layer calls (its node, its bits).
        ints: the words the door resolved (that file's header); no float arguments."""
        if len(fs) != 0:
            raise Error("ValueError: ops: " + session_conv_name(code) + " takes no float arguments")
        for i in range(len(hs)):  # small-loop(hs: x, weight and an optional bias): admission, no data
            if self.int_words[self._slot(hs[i])]:
                raise Error("TypeError: ops: " + session_conv_name(code)
                            + ": x, weight and bias must be floating-point tensors, not Session.ids tensors")
        return session_conv_forward(self.tape, self.ex, code, ints, x)
    # session-ops-conv end


    # session-ops-pool begin: codes 140..149 (sub-lane session-ops-pool replaces this body)
    def _layer_pool(mut self, code: Int, hs: List[Int], ints: List[Int], fs: List[Float32], x: List[Tensor]) raises -> Tensor:
        """The pools (autograd/session_ops_pool.mojo): 1-D, 2-D with PyTorch's options, 3-D max / avg, adaptive
        avg 1-D / 2-D / 3-D and adaptive max 1-D / 2-D, each through the forward function Sequential's layer calls.
        x [input]; ints pool_int_count(code) words (the ops door writes them)."""
        if code > OPS_POOL_LAST:
            raise Error("ops: unknown operation code " + String(code))  # 149: no operation took it
        var what = pool_op_name(code, ints)
        _need(what, len(x), 1, pool_int_count(code), len(ints), 0, len(fs))
        if self.int_words[self._slot(hs[0])]:
            raise Error("ValueError: ops: " + what + ": x must be a floating-point tensor")
        return session_pool(self.tape, self.ex, code, x[0], ints)
    # session-ops-pool end


    # session-ops-resize begin: codes 150..159 (sub-lane session-ops-resize replaces this body)
    def _layer_resize(mut self, code: Int, hs: List[Int], ints: List[Int], fs: List[Float32], x: List[Tensor]) raises -> Tensor:
        """Resampling and rearranging (interpolate, pixel_shuffle / unshuffle, reflection / replication pad):
        the Sequential layers' own forward functions (autograd/session_ops_resize.mojo)."""
        if len(x) != 1:
            raise Error("ops: resampling and padding operations take one tensor")
        if self.int_words[self._slot(hs[0])]:
            raise Error("TypeError: interpolate, pixel_shuffle, pixel_unshuffle and pad with a mode take a "
                        + "floating-point tensor")
        if code == OPS_INTERPOLATE:
            return session_interpolate(self.tape, self.ex, x[0], ints, fs)
        if code == OPS_PIXEL_SHUFFLE or code == OPS_PIXEL_UNSHUFFLE:
            return session_pixel_shuffle(self.tape, self.ex, x[0], ints, code == OPS_PIXEL_UNSHUFFLE)
        if code == OPS_PAD_MODE:
            return session_pad_mode(self.tape, self.ex, x[0], ints)
        raise Error("ops: unknown operation code " + String(code))
    # session-ops-resize end


    # session-ops-rest begin: codes 160..179 (autograd/session_ops_rest.mojo has the forward glue)
    def _layer_rest(mut self, code: Int, hs: List[Int], ints: List[Int], fs: List[Float32], x: List[Tensor]) raises -> Tensor:
        """Norms, activations with parameters, multi-head attention, positional encoding, the losses and the raw
        state-space scans. The operands and words of each code (read once by the ops door at UNPACK):
          160 batch_norm, 162 instance_norm: [x, (running_mean, running_var), (weight, bias)], ints [training
              (instance_norm: use_input_stats), running given, affine], floats [momentum, eps]
          161 group_norm: [x, (weight, bias)], ints [num_groups, affine], floats [eps]
          163 prelu [x, weight]; 164 mish [x]
          165 multi_head_attention: [query, key, in-projection (self: (E, 3E); cross: Wq (E, E), then [Wk | Wv]
              (kdim, 2E)), out_proj_weight, (in_proj_bias (cross: bq, then [bk | bv]), out_proj_bias),
              (key_lengths)], ints [num_heads, self-attention, bias, is_causal, key_lengths given, training,
              seed low, seed high, counter], floats [dropout_p]
          166 positional_encoding: [x], ints [training, seed low, seed high, counter], floats [dropout_p]
          167 smooth_l1_loss [input, target] ints [reduction] floats [beta]; 168 nll_loss [input, target ids]
              ints [reduction]; 169 kl_div [input, target] ints [reduction (2 batchmean)]; 170
              binary_cross_entropy_with_logits [input, target] ints [reduction] floats [pos_weight]
          171 ctc_loss: [log_probs, targets, input_lengths, target_lengths] (each of the last three float values
              or Session.ids), ints [blank, reduction, zero_infinity]
          172 selective_scan [u, delta, A_log, B, C, D]; 173 ssd_scan [x, dt, A_log, D]; 174 trapezoid_scan
              [v, k, q, dt, A, trap, D] ints [heads]"""
        var n = len(x)
        var ni = len(ints)
        var nf = len(fs)
        var name = sor_name(code)
        if code < SOR_FIRST or code > SOR_LAST:
            raise Error("ops: unknown operation code " + String(code))
        # every operand is a floating-point tensor, except the class ids of nll_loss and the targets and lengths
        # of ctc_loss (either kind)
        for k in range(n):  # small-loop(n: the op's operands, at most 9): handle metadata, no data
            var ids_ok = (code == SOR_NLL and k == 1) or (code == SOR_CTC and k >= 1)
            if self.int_words[self._slot(hs[k])] and not ids_ok:
                raise Error("ValueError: ops: " + name + ": operand " + String(k)
                            + " must be a floating-point tensor (made with Session.tensor)")
        if code == SOR_BATCH_NORM or code == SOR_INSTANCE_NORM:
            if ni < 3 or nf < 2:
                _need(name, n, 1, 3, ni, 2, nf)
            var has_run = ints[1] == 1
            var affine = ints[2] == 1
            _need(name, n, 1 + (2 if has_run else 0) + (2 if affine else 0), 3, ni, 2, nf)
            var rm: Optional[Tensor] = None
            var rv: Optional[Tensor] = None
            var w: Optional[Tensor] = None
            var b: Optional[Tensor] = None
            var at = 1
            if has_run:
                if ints[0] == 1:
                    # training (use_input_stats) writes the running statistics in place: carried state only
                    for k in range(1, 3):  # small-loop(2: running_mean, running_var): buffer metadata, no data
                        if self._buffer_kind_of(Int(self.tape.value_of(self.ex, x[k]))) != OPS_BUFFER_STATE:
                            raise Error("ValueError: ops: " + name + ": running_mean and running_var are updated "
                                        + "in place in training; make them with Session.state")
                rm = Optional[Tensor](x[1])
                rv = Optional[Tensor](x[2])
                at = 3
            if affine:
                w = Optional[Tensor](x[at])
                b = Optional[Tensor](x[at + 1])
            if code == SOR_BATCH_NORM:
                return sor_batch_norm(self.tape, self.ex, x[0], rm, rv, w, b, ints[0] == 1, fs[0], fs[1])
            return sor_instance_norm(self.tape, self.ex, x[0], rm, rv, w, b, ints[0] == 1, fs[0], fs[1])
        if code == SOR_GROUP_NORM:
            if ni < 2:
                _need(name, n, 1, 2, ni, 1, nf)
            var g_affine = ints[1] == 1
            _need(name, n, 3 if g_affine else 1, 2, ni, 1, nf)
            var gw: Optional[Tensor] = None
            var gb: Optional[Tensor] = None
            if g_affine:
                gw = Optional[Tensor](x[1])
                gb = Optional[Tensor](x[2])
            return sor_group_norm(self.tape, self.ex, x[0], ints[0], gw, gb, fs[0])
        if code == SOR_PRELU:
            _need(name, n, 2, 0, ni, 0, nf)
            return sor_prelu(self.tape, self.ex, x[0], x[1])
        if code == SOR_MISH:
            _need(name, n, 1, 0, ni, 0, nf)
            return sor_mish(self.tape, self.ex, x[0])
        if code == SOR_MHA:
            if ni < 9:
                _need(name, n, 4, 9, ni, 1, nf)
            var self_attn = ints[1] == 1
            var bias = ints[2] == 1
            var lens = ints[4] == 1
            var cross = 0 if self_attn else 1
            _need(name, n, 4 + cross + ((2 + cross) if bias else 0) + (1 if lens else 0), 9, ni, 1, nf)
            var w_kv: Optional[Tensor] = None
            if not self_attn:
                w_kv = Optional[Tensor](x[3])
            var wo = x[3 + cross]
            var b_in: Optional[Tensor] = None
            var b_kv: Optional[Tensor] = None
            var bo: Optional[Tensor] = None
            var at = 4 + cross
            if bias:
                b_in = Optional[Tensor](x[at])
                if not self_attn:
                    b_kv = Optional[Tensor](x[at + 1])
                bo = Optional[Tensor](x[at + 1 + cross])
                at += 2 + cross
            var kl: Optional[Tensor] = None
            if lens:
                kl = Optional[Tensor](x[at])
            return sor_multi_head_attention(self.tape, self.ex, x[0], x[0] if self_attn else x[1], self_attn, ints[0],
                                            x[2], b_in, w_kv, b_kv, wo, bo, kl, ints[3] == 1, fs[0], ints[5] == 1,
                                            _seed(ints[6], ints[7]), ints[8])
        if code == SOR_POS_ENC:
            _need(name, n, 1, 4, ni, 1, nf)
            return sor_positional_encoding(self.tape, self.ex, x[0], fs[0], ints[0] == 1, _seed(ints[1], ints[2]),
                                           ints[3])
        if code == SOR_SMOOTH_L1:
            _need(name, n, 2, 1, ni, 1, nf)
            return sor_smooth_l1_loss(self.tape, self.ex, x[0], x[1], ints[0], fs[0])
        if code == SOR_NLL:
            _need(name, n, 2, 1, ni, 0, nf)
            if not self.int_words[self._slot(hs[1])]:
                raise Error("ValueError: ops: nll_loss: target must be class ids made with Session.ids")
            return sor_nll_loss(self.tape, self.ex, x[0], x[1], ints[0])
        if code == SOR_KL_DIV:
            _need(name, n, 2, 1, ni, 0, nf)
            return sor_kl_div(self.tape, self.ex, x[0], x[1], ints[0])
        if code == SOR_BCE_POS_WEIGHT:
            _need(name, n, 2, 1, ni, 1, nf)
            return sor_bce_with_logits(self.tape, self.ex, x[0], x[1], ints[0], fs[0])
        if code == SOR_CTC:
            _need(name, n, 4, 3, ni, 0, nf)
            return sor_ctc_loss(self.tape, self.ex, x[0], x[1], self.int_words[self._slot(hs[1])], x[2],
                                self.int_words[self._slot(hs[2])], x[3], self.int_words[self._slot(hs[3])], ints[0],
                                ints[1], ints[2] == 1)
        if code == SOR_SELECTIVE_SCAN:
            _need(name, n, 6, 0, ni, 0, nf)
            return sor_selective_scan(self.tape, self.ex, x[0], x[1], x[2], x[3], x[4], x[5])
        if code == SOR_SSD_SCAN:
            _need(name, n, 4, 0, ni, 0, nf)
            return sor_ssd_scan(self.tape, self.ex, x[0], x[1], x[2], x[3])
        if code == SOR_TRAPEZOID_SCAN:
            _need(name, n, 7, 1, ni, 0, nf)
            return sor_trapezoid_scan(self.tape, self.ex, x[0], x[1], x[2], x[3], x[4], x[5], x[6], ints[0])
        raise Error("ops: unknown operation code " + String(code))
    # session-ops-rest end


    # session-ops-extra begin: codes 180..189 (sub-lane session-ops-extra replaces this body)
    def _layer_extra(mut self, code: Int, hs: List[Int], ints: List[Int], fs: List[Float32], x: List[Tensor]) raises -> Tensor:
        """sequence_mask, topk, gather along an axis, logsumexp and the soft-target cross-entropy
        (autograd/session_ops_extra.mojo has the words, launches and refusals of each)."""
        var n = len(x)
        var ni = len(ints)
        var nf = len(fs)
        if code == OPS_SEQUENCE_MASK:
            # [lengths], ints [max_length]
            _need("sequence_mask", n, 1, 1, ni, 0, nf)
            if self.int_words[self._slot(hs[0])]:
                raise Error("ValueError: ops: sequence_mask: lengths must be a floating-point tensor (one length per"
                            + " sequence, as Session.scan takes)")
            return sox_sequence_mask(self.tape, self.ex, x[0], ints[0])
        if code == OPS_TOPK_INDICES:
            # [x], ints [k, axis, largest (0, 1)]
            _need("topk", n, 1, 3, ni, 0, nf)
            if self.int_words[self._slot(hs[0])]:
                raise Error("ValueError: ops: topk: x must be a floating-point tensor")
            return sox_topk_indices(self.tape, self.ex, x[0], ints[0], ints[1], ints[2] != 0)
        if code == OPS_GATHER_AXIS:
            # [x, index (an ids tensor)], ints [axis]
            _need("gather", n, 2, 1, ni, 0, nf)
            if self.int_words[self._slot(hs[0])]:
                raise Error("ValueError: ops: gather: x must be a floating-point tensor")
            var gi = self._slot(hs[1])
            if not self.int_words[gi]:
                raise Error("ValueError: ops: gather: index must be an integer tensor made with Session.ids or"
                            + " returned by Session.topk")
            return sox_gather(self.tape, self.ex, x[0], ints[0], self.tape.value_of(self.ex, x[1]), self.shapes[gi])
        if code == OPS_LOGSUMEXP:
            # [x], ints [axis, keepdims (0, 1)]
            _need("logsumexp", n, 1, 2, ni, 0, nf)
            if self.int_words[self._slot(hs[0])]:
                raise Error("ValueError: ops: logsumexp: x must be a floating-point tensor")
            return sox_logsumexp(self.tape, self.ex, x[0], ints[0], ints[1] != 0)
        if code == OPS_SOFT_CROSS_ENTROPY or code == OPS_SOFT_CROSS_ENTROPY_WEIGHTED:
            # [logits, target (a distribution of the logits' shape), (weight)], ints [reduction]
            var weighted = code == OPS_SOFT_CROSS_ENTROPY_WEIGHTED
            _need("cross_entropy", n, 3 if weighted else 2, 1, ni, 0, nf)
            for q in range(n):  # small-loop(n: two or three operands): operand kinds, no data
                if self.int_words[self._slot(hs[q])]:
                    raise Error("ValueError: ops: cross_entropy: logits, a target distribution and a weight must be"
                                + " floating-point tensors")
            if weighted:
                return sox_soft_cross_entropy(self.tape, self.ex, x[0], x[1], ints[0], Optional[Tensor](x[2]))
            return sox_soft_cross_entropy(self.tape, self.ex, x[0], x[1], ints[0], None)
        raise Error("ops: unknown operation code " + String(code))
    # session-ops-extra end
    # session-ops end

    def _attention_qkv(mut self, hs: List[Int], ints: List[Int], fs: List[Float32], x: List[Tensor]) raises -> Int:
        """OPS_ATTENTION_QKV: [q, k, v] or [q, k, v, mask], ints [causal (0, 1), scale given (0, 1), mask kind
        (0 none, 1 boolean: ids words, nonzero attends, PyTorch's boolean mask; 2 additive: float values added to
        the scaled scores, PyTorch's float mask)], floats [scale] -> softmax(scale q k^T + mask) v
        (autograd/nn_qkv.mojo qkv_attention: its shapes, refusals and launches). The door reads the kind from
        the mask tensor once and the call records it, so a program's replay reads the mask the same way."""
        if len(x) != 3 and len(x) != 4:
            raise Error("ValueError: attention takes q, k, v and an optional mask")
        if len(ints) < 3 or len(fs) < 1:
            raise Error("ops: attention takes 3 integers and 1 float")
        if (len(x) == 4) != (ints[2] == 1 or ints[2] == 2) or ints[2] < 0 or ints[2] > 2:
            raise Error("ops: attention: the mask kind must be 1 or 2 with a mask and 0 without one")
        for i in range(3):  # small-loop(three operands): operand kinds, no data
            if self.int_words[self._slot(hs[i])]:
                raise Error("ValueError: attention: q, k and v must be floating-point tensors")
        var mask = Optional[Tensor](None)
        if len(x) == 4:
            mask = Optional[Tensor](x[3])
        var mask_bool = ints[2] == 1
        var t = qkv_attention(self.tape, self.ex, x[0], x[1], x[2], mask, mask_bool, ints[0] != 0, ints[1] != 0,
                              fs[0])
        return self._add(t, 0, False, 0)
    # nn-blocks-qkv end

    def _generation_op(mut self, code: Int, hs: List[Int], ints: List[Int], fs: List[Float32], x: List[Tensor]) raises -> Int:
        """Generation op `code` (autograd/generation_ops.mojo): operand 0 the
        logits (pick, floating point) or the ids (alive), operand 1 the rows
        int32 state words, written in place by the pick."""
        generation_op_check(code, len(x), ints, fs)
        var name = String(generation_op_name(code))
        if self.tape.is_recording():
            raise Error("ValueError: ops: " + name + " runs in an inference session (Session(record=False))")
        var rslot = self._slot(hs[1])
        var rp = self.tape.value_of(self.ex, x[1])
        if self._buffer_kind_of(Int(rp)) != OPS_BUFFER_STATE_WORDS:
            raise Error("ValueError: ops: " + name + ": the rows must be an integer tensor declared state")
        var n_rows = self.shapes[rslot].count()
        var b = ints[0]
        var t: Tensor
        var int_result = False
        comptime if Self.E.IS_DEVICE:
            if code == OPS_LM_PICK:
                var lslot = self._slot(hs[0])
                if self.int_words[lslot]:
                    raise Error("ValueError: ops: lm_pick: the logits must hold floating-point values")
                var lp = self.tape.value_of(self.ex, x[0])
                var out = self.tape.reserve(self.ex, b)
                var n_scratch = b * ints[1] if ints[3] == LM_PICK_SAMPLE else 1
                var order = self.tape.reserve(self.ex, n_scratch)
                var merge = self.tape.reserve(self.ex, n_scratch)
                var weights = self.tape.reserve(self.ex, n_scratch)
                lm_pick_run(rebind[DeviceExec](self.ex), lp, self.shapes[lslot].count(), rp, n_rows, ints, fs,
                            out, order, merge, weights)
                t = self.tape.variable(self.ex, out, b, 1, False)
                int_result = True
            else:
                var out = self.tape.reserve(self.ex, 1)
                lm_alive_run(rebind[DeviceExec](self.ex), rp, n_rows, b, out)
                t = self.tape.variable(self.ex, out, 1, 1, False)
        else:
            # lane x56d: the executor's host arm (autograd/generation_host.mojo
            # through Exec.gen_pick / gen_alive), the same rule over host storage
            if code == OPS_LM_PICK:
                var lslot = self._slot(hs[0])
                if self.int_words[lslot]:
                    raise Error("ValueError: ops: lm_pick: the logits must hold floating-point values")
                var lp = self.tape.value_of(self.ex, x[0])
                var out = self.tape.reserve(self.ex, b)
                self.ex.gen_pick(lp, self.shapes[lslot].count(), rp, n_rows, ints, fs, out)
                t = self.tape.variable(self.ex, out, b, 1, False)
                int_result = True
            else:
                var out = self.tape.reserve(self.ex, 1)
                self.ex.gen_alive(rp, n_rows, b, out)
                t = self.tape.variable(self.ex, out, 1, 1, False)
        var h = self._add(t, 0, False, 0)
        self.int_words[self._slot(h)] = int_result
        return h

    def _import_op(mut self, code: Int, x: List[Tensor], ints: List[Int], fs: List[Float32]) raises -> Tensor:
        """Session codes OPS_ERF .. OPS_PAD (lane ag-import-kernels): the
        argument checks, then one call of autograd/import_api.mojo."""
        var n = len(x)
        var ni = len(ints)
        var nf = len(fs)
        if code == OPS_ERF:
            _need("erf", n, 1, 0, ni, 0, nf)
            return imp_unary(self.tape, self.ex, x[0], IMP_ERF)
        if code == OPS_GELU:
            _need("gelu", n, 1, 1, ni, 0, nf)
            if ints[0] != 0 and ints[0] != 1:
                raise Error("ops: gelu approximate must be 0 (erf) or 1 (tanh)")
            return imp_unary(self.tape, self.ex, x[0], IMP_GELU_TANH if ints[0] == 1 else IMP_GELU_ERF)
        if code == OPS_SOFTPLUS:
            _need("softplus", n, 1, 0, ni, 0, nf)
            return imp_unary(self.tape, self.ex, x[0], IMP_SOFTPLUS)
        if code == OPS_SILU:
            _need("silu", n, 1, 0, ni, 0, nf)
            return imp_unary(self.tape, self.ex, x[0], IMP_SILU)
        if code == OPS_LEAKY_RELU:
            _need("leaky_relu", n, 1, 0, ni, 1, nf)
            return imp_unary(self.tape, self.ex, x[0], IMP_LEAKY_RELU, fs[0])
        if code == OPS_ELU:
            _need("elu", n, 1, 0, ni, 1, nf)
            return imp_unary(self.tape, self.ex, x[0], IMP_ELU, fs[0])
        if code == OPS_HARD_SIGMOID:
            _need("hard_sigmoid", n, 1, 0, ni, 2, nf)
            return imp_unary(self.tape, self.ex, x[0], IMP_HARD_SIGMOID, fs[0], fs[1])
        if code == OPS_HARD_SWISH:
            _need("hard_swish", n, 1, 0, ni, 0, nf)
            # x * hard_sigmoid(x) with alpha = 1/6 and beta = 1/2 (ONNX HardSwish)
            return imp_unary(self.tape, self.ex, x[0], IMP_HARD_SWISH, identical_div(Float32(1), Float32(6)), Float32(0.5))
        if code == OPS_CLIP:
            _need("clip", n, 1, 0, ni, 2, nf)
            if fs[0] > fs[1]:
                raise Error("ops: clip needs low <= high")
            return imp_unary(self.tape, self.ex, x[0], IMP_CLIP, fs[0], fs[1])
        if code >= OPS_ABS and code <= OPS_RECIPROCAL:
            _need("elementwise", n, 1, 0, ni, 0, nf)
            var kind = IMP_ABS
            if code == OPS_FLOOR:
                kind = IMP_FLOOR
            elif code == OPS_CEIL:
                kind = IMP_CEIL
            elif code == OPS_ROUND:
                kind = IMP_ROUND
            elif code == OPS_SIN:
                kind = IMP_SIN
            elif code == OPS_COS:
                kind = IMP_COS
            elif code == OPS_SIGN:
                kind = IMP_SIGN
            elif code == OPS_RECIPROCAL:
                kind = IMP_RECIPROCAL
            return imp_unary(self.tape, self.ex, x[0], kind)
        if code == OPS_POW:
            if n == 1:
                _need("pow", n, 1, 0, ni, 1, nf)
                return imp_unary(self.tape, self.ex, x[0], IMP_POW_SCALAR, fs[0])
            _need("pow", n, 2, 0, ni, 0, nf)
            return imp_binary(self.tape, self.ex, x[0], x[1], IMP_POW)
        if code == OPS_MINIMUM or code == OPS_MAXIMUM:
            _need("minimum and maximum", n, 2, 0, ni, 0, nf)
            return imp_binary(self.tape, self.ex, x[0], x[1], IMP_MINIMUM if code == OPS_MINIMUM else IMP_MAXIMUM)
        if code == OPS_WHERE:
            _need("where", n, 3, 0, ni, 0, nf)
            return imp_where(self.tape, self.ex, x[0], x[1], x[2])
        if code == OPS_COMPARE:
            _need("compare", n, 2, 1, ni, 0, nf)
            if ints[0] < 0 or ints[0] > 5:
                raise Error("ops: the comparison must be 0 (eq), 1 (gt), 2 (lt), 3 (ge), 4 (le) or 5 (ne)")
            return imp_binary(self.tape, self.ex, x[0], x[1], IMP_EQ + ints[0])
        if code == OPS_LOGICAL:
            if ni < 1:
                raise Error("ops: logical takes its kind")
            if ints[0] == 0:
                _need("logical_not", n, 1, 1, ni, 0, nf)
                return imp_unary(self.tape, self.ex, x[0], IMP_NOT)
            if ints[0] != 1 and ints[0] != 2:
                raise Error("ops: the logical kind must be 0 (not), 1 (and) or 2 (or)")
            _need("logical_and and logical_or", n, 2, 1, ni, 0, nf)
            return imp_binary(self.tape, self.ex, x[0], x[1], IMP_AND if ints[0] == 1 else IMP_OR)
        if code == OPS_EXPAND:
            if n != 1 or ni < 1:
                raise Error("ops: expand takes one tensor and a shape")
            return imp_expand(self.tape, self.ex, x[0], ints)
        if code == OPS_ARGMAX:
            _need("argmax", n, 1, 3, ni, 0, nf)
            if ints[1] != 0 and ints[1] != 1:
                raise Error("ops: argmax keepdims must be 0 or 1")
            return imp_argmax(self.tape, self.ex, x[0], ints[0], ints[1] == 1, ints[2])
        if code == OPS_PAD:
            if n != 1 or nf < 1:
                raise Error("ops: pad takes one tensor, its pads and the value")
            return imp_pad(self.tape, self.ex, x[0], ints, fs[0])
        if code == OPS_ANY_ALL:
            # any: the largest of the nonzero tests (0 or 1, order-free);
            # all: not any of the zero tests. No gradient.
            if n != 1 or ni < 2:
                raise Error("ops: any and all take one tensor, the kind, keepdims and the axes")
            if ints[0] != 0 and ints[0] != 1:
                raise Error("ops: the kind must be 0 (any) or 1 (all)")
            var tests = imp_unary(self.tape, self.ex, x[0], IMP_BOOL if ints[0] == 0 else IMP_NOT)
            var rest = List[Int]()
            for k in range(1, ni):  # small-loop(ni: keepdims and at most 8 axes): argument parsing, no data
                rest.append(ints[k])
            var m = self._reduce_n(tests, rest, False, True)
            if ints[0] == 1:
                return imp_unary(self.tape, self.ex, m, IMP_NOT)
            return m
        if code == OPS_CUMSUM:
            _need("cumsum", n, 1, 3, ni, 0, nf)
            if (ints[1] != 0 and ints[1] != 1) or (ints[2] != 0 and ints[2] != 1):
                raise Error("ops: cumsum exclusive and reverse must be 0 or 1")
            return imp_cumsum(self.tape, self.ex, x[0], ints[0], ints[1] == 1, ints[2] == 1)
        if code == OPS_INDEX:
            if n < 2 or ni != n - 1:
                raise Error("ops: index takes a tensor, its index tensors and one axis for each")
            var idx = List[Tensor]()
            for k in range(1, n):  # small-loop(n: the index tensors, at most 8): handle lists, no data
                idx.append(x[k])
            return imp_gather_nd(self.tape, self.ex, x[0], idx, ints)
        raise Error("ops: unknown operation code " + String(code))

    # ------------------------------------------------------------ rank-N helpers
    def _binary(mut self, x: Tensor, y: Tensor, op: Int) raises -> Tensor:
        """x op y: two matrices take the tape's 2-D op (its words and
        broadcasts, unchanged); any other ranks the fold lane's rankn_binary
        (NumPy broadcasting; trailing broadcasts route back to the 2-D op)."""
        if self.tape.ndim(x) == 2 and self.tape.ndim(y) == 2:
            if op == ADD:
                return self.tape.add(self.ex, x, y)
            if op == SUBTRACT:
                return self.tape.subtract(self.ex, x, y)
            if op == MULTIPLY:
                return self.tape.multiply(self.ex, x, y)
            return self.tape.divide(self.ex, x, y)
        return rankn_binary(self.tape, self.ex, x, y, op)

    def _swap(mut self, x: Tensor, a: Int, b: Int) raises -> Tensor:
        """x with axes a and b swapped: on a matrix the tape's transpose
        (its view and words), otherwise a permute view."""
        var nd = self.tape.ndim(x)
        if nd < 2:
            raise Error("ops: transpose takes a tensor of at least 2 dimensions")
        var aa = rn_normalize_axis(a, nd)
        var bb = rn_normalize_axis(b, nd)
        if nd == 2 and aa != bb:
            return self.tape.transpose(self.ex, x)
        var axes = List[Int](capacity=nd)
        for k in range(nd):  # small-loop(nd: tensor rank, at most 8): shape metadata, no data
            axes.append(k)
        axes[aa] = bb
        axes[bb] = aa
        return self.tape.permute(self.ex, x, axes)

    def _reshape_n(mut self, x: List[Tensor], ints: List[Int]) raises -> Tensor:
        """The free views: reshape (mode 0, dims with one -1), flatten
        (mode 1, start and end axes), squeeze (mode 2, axes or none for
        every length-1 axis), unsqueeze (mode 3, the new axis)."""
        if len(x) != 1 or len(ints) < 1:
            raise Error("ops: reshape takes one tensor and its new shape")
        var s = self.tape.shape_of(x[0])
        var mode = ints[0]
        var dims = List[Int]()
        if mode == 0:
            for k in range(1, len(ints)):  # small-loop(ints: at most 8 dims): argument parsing, no data
                dims.append(ints[k])
            return self.tape.view_nd(self.ex, x[0], rn_resolve(s, dims, "ops: reshape"))
        if mode == 1:
            if len(ints) != 3:
                raise Error("ops: flatten takes a start and an end axis")
            var nd = s.ndim if s.ndim > 0 else 1
            var start = rn_normalize_axis(ints[1], nd)
            var end = rn_normalize_axis(ints[2], nd)
            if end < start:
                raise Error("ops: flatten: end must not come before start")
            for k in range(s.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
                if k == start:
                    dims.append(-1)
                elif k < start or k > end:
                    dims.append(s.d[k])
            if s.ndim == 0:
                dims.append(1)
            return self.tape.view_nd(self.ex, x[0], rn_resolve(s, dims, "ops: flatten"))
        if mode == 2:
            var listed = List[Int]()
            for k in range(1, len(ints)):  # small-loop(ints: at most 8 axes): argument parsing, no data
                listed.append(ints[k])
            var mask = rn_axis_mask(s.ndim, listed, "ops: squeeze")
            for k in range(s.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
                if len(listed) == 0:
                    mask[k] = s.d[k] == 1
                elif mask[k] and s.d[k] != 1:
                    raise Error("ops: squeeze: axis " + String(k) + " has length " + String(s.d[k]) + ", not 1")
            return self.tape.view_nd(self.ex, x[0], rn_reduced(s, mask, False, "ops: squeeze"))
        if mode == 3:
            if len(ints) != 2:
                raise Error("ops: unsqueeze takes one axis")
            var at = rn_normalize_axis(ints[1], s.ndim + 1)
            for k in range(s.ndim + 1):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
                if k == at:
                    dims.append(1)
                if k < s.ndim:
                    dims.append(s.d[k])
            return self.tape.view_nd(self.ex, x[0], rn_shape(dims, "ops: unsqueeze"))
        raise Error("ops: unknown reshape mode " + String(mode))

    def _reduce_n(mut self, x: Tensor, ints: List[Int], mean: Bool, is_max: Bool) raises -> Tensor:
        """sum, mean or max over the listed axes (none: every element).
        ints[0] is keepdims: 1 keeps each reduced axis with length 1, 0
        removes it, -1 (the default) keeps them for a tensor of at most 2
        dimensions and removes them otherwise. Every element (no axes,
        default keepdims) gives a (1, 1) tensor. Matrices take the tape's
        sums (AXIS_ALL / AXIS_ROWS / AXIS_COLS: today's words); other
        ranks the fold lane's rankn_sum / rankn_max."""
        var s = self.tape.shape_of(x)
        var kd = ints[0]
        if kd < -1 or kd > 1:
            raise Error("ops: keepdims must be -1, 0 or 1")
        var listed = List[Int]()
        for k in range(1, len(ints)):  # small-loop(ints: at most 8 axes): argument parsing, no data
            listed.append(ints[k])
        var mask = rn_axis_mask(s.ndim, listed, "ops: reduce")
        if len(listed) == 0:
            for k in range(s.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
                mask[k] = True
        var axes = rn_mask_axes(mask)
        var every = len(axes) == s.ndim
        var t: Tensor
        if is_max:
            t = rankn_max(self.tape, self.ex, x, axes)
        elif every:
            t = self.tape.mean(self.ex, x, AXIS_ALL) if mean else self.tape.sum(self.ex, x, AXIS_ALL)
        elif s.ndim == 2:
            var axis = AXIS_ROWS if mask[0] else AXIS_COLS
            t = self.tape.mean(self.ex, x, axis) if mean else self.tape.sum(self.ex, x, axis)
        else:
            t = rankn_sum(self.tape, self.ex, x, axes, mean)
        if len(listed) == 0 and kd == -1:
            return rn_as_shape(self.tape, self.ex, t, ShapeN.matrix(1, 1))
        var keep = kd == 1 or (kd == -1 and s.ndim <= 2)
        return rn_as_shape(self.tape, self.ex, t, rn_reduced(s, mask, keep, "ops: reduce"))

    def _concat(mut self, x: List[Tensor], axis: Int) raises -> Tensor:
        """x[0], x[1], ... joined along axis: one layout-lane concat node
        per RN_CONCAT_MAX operands, chained (copies: the same words)."""
        var nd = self.tape.ndim(x[0])
        var a = rn_normalize_axis(axis, nd if nd > 0 else 1)
        if len(x) == 1:
            return self.tape.view_nd(self.ex, x[0], self.tape.shape_of(x[0]))
        var first = List[Tensor]()
        var i = 0
        while i < len(x) and i < RN_CONCAT_MAX:  # small-loop(x: the first concat node's operands, at most 10): handle lists, no data
            first.append(x[i])
            i += 1
        var acc = rankn_concat(self.tape, self.ex, first, a)
        while i < len(x):  # driver-loop: over concat operands (one node per RN_CONCAT_MAX - 1 more)
            var part = List[Tensor]()
            part.append(acc)
            while i < len(x) and len(part) < RN_CONCAT_MAX:  # driver-loop: over concat operands
                part.append(x[i])
                i += 1
            acc = rankn_concat(self.tape, self.ex, part, a)
        return acc

    def _piece(mut self, x: Tensor, ints: List[Int]) raises -> Tensor:
        """Piece ints[1] of split(x, axis=ints[0]): mode 0 lists the
        sizes (summing to the axis length), mode 1 the number of equal
        sections. A free strided view (the layout lane's narrow)."""
        var s = self.tape.shape_of(x)
        var nd = s.ndim if s.ndim > 0 else 1
        var a = rn_normalize_axis(ints[0], nd)
        var length_a = s.d[a] if s.ndim > 0 else 1
        var piece = ints[1]
        var mode = ints[2]
        var start = 0
        var length = 0
        if mode == 0:
            var pieces = len(ints) - 3
            if piece < 0 or piece >= pieces:
                raise Error("ops: split: piece out of range")
            var total = 0
            for k in range(pieces):  # small-loop(pieces: the split sizes, a handful): argument parsing, no data
                var sz = ints[3 + k]
                if sz < 1:
                    raise Error("ops: split: every size must be positive")
                if k < piece:
                    start += sz
                if k == piece:
                    length = sz
                total += sz
            if total != length_a:
                raise Error("ops: split: the sizes sum to " + String(total) + ", the axis has "
                            + String(length_a))
        elif mode == 1:
            var sections = ints[3]
            if sections < 1 or length_a % sections != 0:
                raise Error("ops: split: " + String(length_a) + " is not divisible into "
                            + String(sections) + " equal sections")
            if piece < 0 or piece >= sections:
                raise Error("ops: split: piece out of range")
            length = length_a // sections
            start = piece * length
        else:
            raise Error("ops: unknown split mode " + String(mode))
        return rankn_narrow(self.tape, self.ex, x, a, start, length)

    # nn-blocks-scan begin: the scan over a captured cell (autograd/nn_scan.mojo, session code 118)
    def _scan_call(
        mut self, code: Int, hs: List[Int], ints: List[Int], fs: List[Float32], mut x: List[Tensor],
    ) raises -> Int:
        """One existing operation inside a scan: `_op_run_tensors` without the scan branch (nested, so a scan
        never re-enters it), the operand list `x` the scan's own reused capacity."""
        x.clear()
        for i in range(len(hs)):  # small-loop(hs: one node's input handles): handle lookup, no data
            x.append(self.tensor(hs[i]))
        return self._op_run_tensors[nested=True](code, hs, ints, fs, x)

    def _scan(mut self, hs: List[Int], ints: List[Int], fs: List[Float32]) raises -> Int:
        """OPS_SCAN: [xs (steps, ...), state_1 .. state_k, (lengths), param_1 .. param_m], ints and floats the
        cell's description (autograd/nn_scan.mojo). A driver loop over steps: step t's input is a free view of
        xs (operation 40, then 33 to the cell's step shape), the cell's nodes run on the step's handles through
        `_op_run_tensors` (the same kernels as the operations one by one, on this session's executor), and the
        next state replaces the state handles; nothing is read back and no buffer is copied between steps. With
        `lengths` (one float length per sequence, batch on axis 1 of xs) a step t >= length keeps the row's
        state (where) and gives a zero output row; the mask is built once on the executor (full, cumsum,
        compare). `reverse` walks t = steps - 1 down to 0; the outputs stay in time order. Returns one (n, 1)
        column: the outputs of every step in time order, then the final states, each flattened (one concat
        node: each step's output copied once). A dropout in the cell draws a new mask each step. Launches per step: two views, the
        cell's nodes, and with lengths one view, one view per state rank and one where per state and output.
        Every temporary handle is retired before the return; the tape keeps its nodes, so a recording session
        differentiates the whole loop (backward through time over the stored per-step values)."""
        var spec = scan_decode(ints, fs, len(hs))
        ref cell = spec.cell
        var n_state = spec.n_state
        var reverse = spec.reverse
        var has_lengths = spec.has_lengths
        var xs = hs[0]
        var xs_slot = self._slot(xs)
        var xs_shape = self.shapes[xs_slot]
        var state = List[Int](capacity=n_state)
        var state_shapes = List[ShapeN](capacity=n_state)
        var state_ids = List[Bool](capacity=n_state)
        for k in range(n_state):  # driver-loop: over state tensors (handles and metadata)
            var slot = self._slot(hs[1 + k])
            state.append(hs[1 + k])
            state_shapes.append(self.shapes[slot])
            state_ids.append(self.int_words[slot])
        var at = 1 + n_state
        var lengths = 0
        var lengths_count = -1
        if has_lengths:
            lengths = hs[at]
            var lslot = self._slot(lengths)
            if self.int_words[lslot]:
                raise Error("ValueError: scan: lengths must hold floating-point values")
            lengths_count = self.shapes[lslot].count()
            at += 1
        var params = List[Int](capacity=spec.n_params)
        var param_shapes = List[ShapeN](capacity=spec.n_params)
        var param_ids = List[Bool](capacity=spec.n_params)
        for j in range(spec.n_params):  # driver-loop: over the cell's bound parameters (handles and metadata)
            var slot = self._slot(hs[at + j])
            params.append(hs[at + j])
            param_shapes.append(self.shapes[slot])
            param_ids.append(self.int_words[slot])
        scan_check_binding(cell, xs_shape, self.int_words[xs_slot], state_shapes, state_ids, lengths_count,
                           param_shapes, param_ids)
        # Resolved once per call: the nodes' arguments, the step views' words, the mask.
        var n_nodes = cell.n_nodes()
        var node_ints = List[List[Int]](capacity=n_nodes)
        var node_fs = List[List[Float32]](capacity=n_nodes)
        for k in range(n_nodes):  # driver-loop: over cell nodes (their arguments, once per call)
            var ni = List[Int]()
            for j in range(cell.i_at[k], cell.i_at[k + 1]):  # small-loop(ints: a node's integer arguments)
                ni.append(cell.ints[j])
            node_ints.append(ni^)
            var nf = List[Float32]()
            for j in range(cell.f_at[k], cell.f_at[k + 1]):  # small-loop(fs: a node's float arguments)
                nf.append(cell.fs[j])
            node_fs.append(nf^)
        # Dropout nodes (OPS_DROPOUT 21: ints [seed lo, seed hi, counter]) draw a new mask every step: a seed
        # derived from (captured seed, the call's key, the node, the time index) (autograd/nn_scan.mojo note).
        var drop_nodes = List[Int]()
        var drop_seed = List[UInt64]()
        for k in range(n_nodes):  # driver-loop: over cell nodes (dropout ones), once per call
            if cell.code[k] == OPS_DROPOUT and len(node_ints[k]) >= 3:
                drop_nodes.append(k)
                drop_seed.append((UInt64(node_ints[k][1]) << 32) | UInt64(node_ints[k][0] & 0xFFFFFFFF))
        var steps = xs_shape.d[0]
        var batch = xs_shape.d[1]
        var x_dims = cell.in_shapes[n_state].dims()
        var view_x: List[Int] = [0]
        for d in range(len(x_dims)):  # small-loop(ndim: the step input's rank, at most 8)
            view_x.append(x_dims[d])
        var piece: List[Int] = [0, 0, 1, steps]
        var no_ints = List[Int]()
        var no_fs = List[Float32]()
        var x = List[Tensor]()
        var one_h: List[Int] = [xs]
        var xs_h: List[Int] = [xs]
        var where_h: List[Int] = [0, 0, 0]
        var temps = List[Int]()
        var mask = 0
        var zero = 0
        var rank_view = List[List[Int]]()
        var mask_by_rank = List[Int](length=9, fill=0)
        var packed = 0
        var step_at = -1
        try:
            if has_lengths:
                var ones_i: List[Int] = [steps, 1]
                var one_f: List[Float32] = [Float32(1.0)]
                var ones = self._scan_call(OPS_FULL, no_ints, ones_i, one_f, x)
                temps.append(ones)
                one_h[0] = ones
                var cs_i: List[Int] = [0, 1, 0]
                var times = self._scan_call(OPS_CUMSUM, one_h, cs_i, no_fs, x)
                temps.append(times)
                one_h[0] = lengths
                var lr_i: List[Int] = [0, 1, batch]
                var lrow = self._scan_call(OPS_RESHAPE_N, one_h, lr_i, no_fs, x)
                temps.append(lrow)
                var cmp_h: List[Int] = [lrow, times]
                var cmp_i: List[Int] = [1]
                mask = self._scan_call(OPS_COMPARE, cmp_h, cmp_i, no_fs, x)
                temps.append(mask)
                var z_i: List[Int] = [1, 1]
                var z_f: List[Float32] = [Float32(0.0)]
                zero = self._scan_call(OPS_FULL, no_ints, z_i, z_f, x)
                temps.append(zero)
            for r in range(9):  # small-loop(ranks 0..8): the mask's view words per rank, once per call
                var rv = List[Int]()
                if r >= 2:
                    rv.append(0)
                    rv.append(batch)
                    for _q in range(r - 1):  # small-loop(rank - 1 unit axes)
                        rv.append(1)
                rank_view.append(rv^)
            comptime if (SCAN_SABOTAGE_INIT_GRAD and Self.E.IS_DEVICE) or (
                    SCAN_SABOTAGE_INIT_GRAD_HOST and not Self.E.IS_DEVICE):
                for k in range(n_state):  # driver-loop: over state tensors  # SABOTAGE: the initial state detached
                    one_h[0] = state[k]
                    var dk = self._scan_call(OPS_DETACH, one_h, no_ints, no_fs, x)
                    temps.append(dk)
                    state[k] = dk
            var lag1 = state.copy()
            var lag2 = state.copy()
            var outs_by_time = List[Int](length=steps, fill=0)
            var vals = List[Int]()
            var args = List[Int]()
            var nxt = List[Int](capacity=n_state + 1)
            for i in range(steps):  # driver-loop: over steps
                var t = steps - 1 - i if reverse else i
                step_at = t
                piece[1] = t
                var x3 = self._scan_call(OPS_NARROW, xs_h, piece, no_fs, x)
                temps.append(x3)
                one_h[0] = x3
                var x_t = self._scan_call(OPS_RESHAPE_N, one_h, view_x, no_fs, x)
                temps.append(x_t)
                vals.clear()
                comptime if (SCAN_SABOTAGE_STATE_LAG and Self.E.IS_DEVICE) or (
                        SCAN_SABOTAGE_STATE_LAG_HOST and not Self.E.IS_DEVICE):
                    for k in range(n_state):  # driver-loop: over state tensors  # SABOTAGE: step t fed state t - 2
                        vals.append(lag2[k] if i >= 2 else state[k])
                else:
                    for k in range(n_state):  # driver-loop: over state tensors (this step's handles)
                        vals.append(state[k])
                vals.append(x_t)
                for q in range(len(drop_nodes)):  # driver-loop: over the cell's dropout nodes (this step's seeds)
                    var dk = drop_nodes[q]
                    var seed = scan_dropout_seed(drop_seed[q], spec.key, dk, t)
                    # nn-blocks-composed begin
                    comptime if (COMPOSED_SABOTAGE_SCAN_KEY and Self.E.IS_DEVICE) or (
                            COMPOSED_SABOTAGE_SCAN_KEY_HOST and not Self.E.IS_DEVICE):
                        seed = scan_dropout_seed(drop_seed[q], 0, dk, t)  # SABOTAGE: the key stays at the capture's 0
                    # nn-blocks-composed end
                    comptime if (SCAN_SABOTAGE_DROP_KEY and Self.E.IS_DEVICE) or (
                            SCAN_SABOTAGE_DROP_KEY_HOST and not Self.E.IS_DEVICE):
                        seed = drop_seed[q]  # SABOTAGE: the captured seed at every step and key
                    node_ints[dk][0] = Int(seed & UInt64(0xFFFFFFFF))
                    node_ints[dk][1] = Int(seed >> 32)
                for k in range(n_nodes):  # driver-loop: over the cell's nodes (one step's launch sequence)
                    args.clear()
                    for j in range(cell.a_at[k], cell.a_at[k + 1]):  # small-loop(args: a node's input refs)
                        var rf = cell.args[j]
                        args.append(params[-rf - 1] if rf < 0 else vals[rf])
                    var h = self._scan_call(cell.code[k], args, node_ints[k], node_fs[k], x)
                    temps.append(h)
                    vals.append(h)
                if has_lengths:
                    one_h[0] = mask
                    var mrow = self._scan_call(OPS_NARROW, one_h, piece, no_fs, x)
                    temps.append(mrow)
                    for r in range(9):  # small-loop(ranks 0..8): this step's mask views, made on first use
                        mask_by_rank[r] = 0
                    mask_by_rank[0] = mrow
                nxt.clear()
                for k in range(n_state + 1):  # driver-loop: over the cell's outputs (next state, then the output)
                    var rf = cell.outs[k]
                    var h = params[-rf - 1] if rf < 0 else vals[rf]
                    var slot = self._slot(h)
                    var want = cell.in_shapes[k] if k < n_state else cell.out_shapes[n_state]
                    if self.int_words[slot]:
                        raise Error("ValueError: scan: the cell's output " + String(k)
                                    + " holds integer ids; a scan carries floating-point values")
                    if not self.shapes[slot].same(want):
                        raise Error("ValueError: scan: the cell's output " + String(k) + " is "
                                    + shape_text(self.shapes[slot]) + " here but was captured as "
                                    + shape_text(want) + ": the output shape must not vary per step")
                    if has_lengths:
                        var r = want.ndim
                        if mask_by_rank[r] == 0:
                            one_h[0] = mask_by_rank[0]
                            mask_by_rank[r] = self._scan_call(OPS_RESHAPE_N, one_h, rank_view[r], no_fs, x)
                            temps.append(mask_by_rank[r])
                        where_h[0] = mask_by_rank[r]
                        where_h[1] = h
                        where_h[2] = vals[k] if k < n_state else zero
                        h = self._scan_call(OPS_WHERE, where_h, no_ints, no_fs, x)
                        temps.append(h)
                    nxt.append(h)
                outs_by_time[t] = nxt[n_state]
                comptime if (SCAN_SABOTAGE_STATE_LAG and Self.E.IS_DEVICE) or (
                        SCAN_SABOTAGE_STATE_LAG_HOST and not Self.E.IS_DEVICE):
                    lag2 = lag1.copy()
                    lag1 = state.copy()
                for k in range(n_state):  # driver-loop: over state tensors (the next step's handles)
                    state[k] = nxt[k]
            step_at = -1
            # The stack: each step's output and each final state as an (n, 1) column (free views), joined once.
            var cols = List[Int](capacity=steps)
            var col_i: List[Int] = [0, cell.out_shapes[n_state].count(), 1]
            for t in range(steps):  # driver-loop: over steps (the stack's columns, views)
                one_h[0] = outs_by_time[t]
                var c = self._scan_call(OPS_RESHAPE_N, one_h, col_i, no_fs, x)
                temps.append(c)
                cols.append(c)
            var extra = List[Int](capacity=n_state)
            for k in range(n_state):  # driver-loop: over state tensors (the final states' columns, views)
                var si: List[Int] = [0, cell.in_shapes[k].count(), 1]
                one_h[0] = state[k]
                var c = self._scan_call(OPS_RESHAPE_N, one_h, si, no_fs, x)
                temps.append(c)
                extra.append(c)
            # One concat node over every column (autograd/rankn_layout.mojo rankn_concat: one slab copy per
            # operand into the reserved (n, 1) output, one slab copy per operand back in the backward; it takes
            # any number of operands, so each step's output is copied once).
            for k in range(len(extra)):  # driver-loop: over the final states' columns
                cols.append(extra[k])
            var parts = List[Tensor](capacity=len(cols))
            for q in range(len(cols)):  # driver-loop: over the stack's columns (handle lookup, no data)
                parts.append(self.tensor(cols[q]))
            var stacked = rankn_concat(self.tape, self.ex, parts, 0)
            packed = self._add(stacked, 0, self.tape.requires_grad(stacked), 0)
        except err:
            self._drop_temps(temps)
            if step_at < 0:
                raise err
            var message = String(err)
            if message.startswith("ValueError: "):
                raise Error("ValueError: scan step " + String(step_at) + ": " + String(message[byte=12:]))
            raise Error("scan step " + String(step_at) + ": " + message)
        self._drop_temps(temps)
        return packed
    # nn-blocks-scan end

    def op_packed(mut self, code: Int, block: Int) raises -> Int:
        """`op` with its arguments in a packed block at address `block`
        (OPS_PACK_* layout): the same checks and the same call."""
        if block == 0:
            raise Error("ops: null argument block")
        var w = UnsafePointer[Int64, MutUntrackedOrigin](unsafe_from_address=block)
        var nh = Int(w[0])
        var ni = Int(w[1])
        var nf = Int(w[2])
        if nh < 0 or nh > OPS_PACK_HANDLES or ni < 0 or ni > OPS_PACK_INTS or nf < 0 or nf > OPS_PACK_FLOATS:
            raise Error("ops: the argument block's counts are out of range")
        var hs = List[Int](capacity=nh)
        for i in range(nh):  # small-loop(nh: an op's input handles, at most four): argument parsing, no data
            hs.append(Int(w[OPS_PACK_H_WORD + i]))
        var ints = List[Int](capacity=ni)
        for i in range(ni):  # small-loop(ni: an op's integer arguments, at most eight): argument parsing, no data
            ints.append(Int(w[OPS_PACK_I_WORD + i]))
        var f = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=block + 8 * OPS_PACK_F_WORD)
        var fs = List[Float32](capacity=nf)
        for i in range(nf):  # small-loop(nf: an op's float arguments, at most four): argument parsing, no data
            fs.append(f[i])
        return self.op(code, hs, ints, fs)

    # ------------------------------------------------------------ captured programs
    def _capture_param(self, h: Int, mut p: OpsProgram) raises -> Int:
        """The ref of session tensor h as a parameter of p: an uploaded
        tensor, read in place at each run."""
        var slot = (h & _SLOT_MASK) - 1
        var gen = h >> _SLOT_BITS
        if h <= 0 or slot < 0 or slot >= len(self.live) or not self.live[slot] or self.generation[slot] != gen:
            raise Error("ops: capture: a tensor the outputs depend on was freed")
        if self.owned[slot] == 0:
            raise Error("ops: capture: the outputs depend on a tensor computed before the last reset; "
                        + "name it as an input or compute it again")
        for j in range(len(p.params)):  # driver-loop: over program parameters
            if p.params[j] == h:
                return -(j + 1)
        p.params.append(h)
        p.param_shapes.append(self.shapes[slot])
        # lane x1: carried state is a state parameter (record version 3)
        var kind = self._buffer_kind_of(self.owned[slot])
        while len(p.param_kinds) < len(p.params) - 1:  # driver-loop: over parameter kinds owed by earlier weights
            p.param_kinds.append(PROGRAM_PARAM_WEIGHT)
        p.param_kinds.append(PROGRAM_PARAM_STATE if kind == OPS_BUFFER_STATE else (
            PROGRAM_PARAM_STATE_WORDS if kind == OPS_BUFFER_STATE_WORDS else PROGRAM_PARAM_WEIGHT))
        return -len(p.params)

    def capture(mut self, inputs: List[Int], outputs: List[Int]) raises -> Int:
        """A program of the logged operations that compute `outputs` from
        `inputs` (autograd/program.mojo); returns its id (>= 1). Uploaded
        tensors the outputs read that are not inputs become parameters,
        read in place at each run."""
        if self.tape.is_recording():
            raise Error("ops: capture takes an inference session (Session(record=False))")
        if len(inputs) < 1 or len(outputs) < 1:
            raise Error("ops: capture takes at least one input and one output")
        var p = OpsProgram()
        p.n_in = len(inputs)
        var value_of = Dict[Int, Int]()
        for k in range(len(inputs)):  # driver-loop: over program inputs
            var s = self._slot(inputs[k])
            if inputs[k] in value_of:
                raise Error("ops: capture: an input is named twice")
            value_of[inputs[k]] = k
            p.in_shapes.append(self.shapes[s])
            p.in_kinds.append(PROGRAM_IN_IDS if self.int_words[s] else PROGRAM_IN_FLOAT)
        var need = Dict[Int, Int]()
        for k in range(len(outputs)):  # driver-loop: over program outputs
            var s = self._slot(outputs[k])
            p.out_shapes.append(self.shapes[s])
            if outputs[k] not in value_of:
                need[outputs[k]] = 1
        # back from the outputs: keep the logged operations they depend on
        var node_count = self.oplog.count()
        var keep = List[Bool](length=node_count, fill=False)
        var back = node_count - 1
        while back >= 0:  # driver-loop: over logged operations, last to first
            var r = self.oplog.res[back]
            if r in need:
                _ = need.pop(r)
                keep[back] = True
                for j in range(self.oplog.h_at[back], self.oplog.h_at[back + 1]):  # driver-loop: over a logged operation input references
                    var h = self.oplog.hs[j]
                    if h not in value_of:
                        need[h] = 1
            back -= 1
        # forward: number the kept operations, in their order
        for i in range(node_count):  # driver-loop: over logged operations
            if not keep[i]:
                continue
            var code = self.oplog.code[i]
            if program_refuses_node(code, self.oplog.h_at[i + 1] - self.oplog.h_at[i]):
                raise Error("ops: capture: embedding and take read the caller's index array, "
                            + "which a program cannot hold; pass the ids as a tensor (ids) instead")
            p.code.append(code)
            for j in range(self.oplog.h_at[i], self.oplog.h_at[i + 1]):  # driver-loop: over logged operation input references
                var h = self.oplog.hs[j]
                if h in value_of:
                    p.args.append(value_of[h])
                else:
                    var pr = self._capture_param(h, p)
                    value_of[h] = pr
                    p.args.append(pr)
            for j in range(self.oplog.i_at[i], self.oplog.i_at[i + 1]):  # driver-loop: over logged operation integer arguments
                p.ints.append(self.oplog.ints[j])
            for j in range(self.oplog.f_at[i], self.oplog.f_at[i + 1]):  # driver-loop: over logged operation float arguments
                p.fs.append(self.oplog.fs[j])
            p.end_node()
            value_of[self.oplog.res[i]] = p.n_in + p.n_nodes() - 1
        for o in range(len(outputs)):  # driver-loop: over program outputs
            var h = outputs[o]
            if h in value_of:
                p.outs.append(value_of[h])
            else:
                var pr = self._capture_param(h, p)
                value_of[h] = pr
                p.outs.append(pr)
        return self.add_program(p^)

    def add_program(mut self, var p: OpsProgram) raises -> Int:
        """Register a program (captured, or read from a file with its
        parameters uploaded); returns its id."""
        if self.tape.is_recording():
            raise Error("ops: programs run in an inference session (Session(record=False))")
        self.programs.append(p^)
        self.program_replays.append(List[PreparedInferenceReplay]())
        return len(self.programs)

    def _program_index(self, pid: Int) raises -> Int:
        if pid < 1 or pid > len(self.programs) or not self.programs[pid - 1].live:
            raise Error("ops: program " + String(pid) + " is not a program of this session")
        return pid - 1

    def program(self, pid: Int) raises -> OpsProgram:
        """A copy of program pid's description."""
        return self.programs[self._program_index(pid)].copy()

    def program_free(mut self, pid: Int) raises:
        var pi = self._program_index(pid)
        self.programs[pi] = OpsProgram()
        self.programs[pi].live = False
        self.program_replays[pi].clear()

    def _leaf(mut self, buf: FP, sh: ShapeN, integer_words: Bool = False) raises -> Int:
        """A temporary handle of a leaf over `buf` on the current tape."""
        var t: Tensor
        if sh.ndim == 2:
            t = self.tape.variable(self.ex, buf, sh.d[0], sh.d[1], False)
        else:
            t = self.tape.variable_nd(self.ex, buf, sh, False)
        var h = self._add(t, 0, False, 0)
        self.int_words[self._slot(h)] = integer_words
        return h

    def _drop_temps(mut self, temps: List[Int]):
        for i in range(len(temps)):  # driver-loop: over a replay's temporary handles
            var slot = (temps[i] & _SLOT_MASK) - 1
            if slot >= 0 and slot < len(self.live) and self.live[slot]:
                self._retire(slot)

    # ------------------------------------------------------------ executor surface
    # Thin public doors for autograd/program_executor.mojo (the per-device
    # ProgramExecutor used by distributed and partitioned programs). Each one
    # forwards to the private helper above unchanged: same launches, same order.
    def drop_temps(mut self, temps: List[Int]):
        """Retire the live handles in `temps` (a replay's temporaries)."""
        self._drop_temps(temps)

    def buffer(self, h: Int) raises -> FP:
        """The session-owned executor buffer of uploaded tensor h (null for
        a computed tensor)."""
        return FP(unsafe_from_address=self.owned[self._slot(h)])

    def is_integer(self, h: Int) raises -> Bool:
        """Whether tensor h's storage holds int32 words (ids)."""
        return self.int_words[self._slot(h)]

    def leaf_nd(mut self, buf: FP, shape: ShapeN, requires_grad: Bool, integer_words: Bool) raises -> Int:
        """A temporary handle of a leaf over the caller's `buf` on the
        current tape (always recorded with `variable_nd`)."""
        var t = self.tape.variable_nd(self.ex, buf, shape, requires_grad)
        var h = self._add(t, 0, requires_grad, 0)
        self.int_words[self._slot(h)] = integer_words
        return h

    def run_op(mut self, code: Int, hs: List[Int], ints: List[Int], fs: List[Float32]) raises -> Int:
        """`op` without the inference log."""
        return self._op_run(code, hs, ints, fs)

    def has_gradient(self, h: Int) raises -> Bool:
        """Whether the last backward wrote a gradient for tensor h."""
        return self.tape._nodes[self.tensor(h)._index].grad_ready

    def optimizer_settings(self, o: Int) raises -> _OpsOptimizer:
        """A copy of optimizer o's configuration and schedule (one group:
        a partitioned or distributed program trains one configuration)."""
        return self.opts[self._opt_index_single(o, "a partitioned or distributed program")]

    def optimizer_state_index(self, o: Int, h: Int) raises -> Int:
        """The index of (optimizer o, parameter handle h)'s retained state, or
        -1 when it was never stepped. Allocates nothing."""
        var oi = self._opt_index(o)
        var si = -1
        for k in range(len(self.opt_states)):  # driver-loop: over retained optimizer states
            if self.opt_states[k].opt == oi and self.opt_states[k].param == h:
                si = k
        return si

    def optimizer_state_record(self, i: Int) -> _OpsOptState:
        """A copy of retained optimizer state i (its buffers stay owned by
        the session)."""
        return self.opt_states[i]

    def _program_op[nested: Bool = False](
        mut self, node: Int, code: Int, hs: List[Int], ints: List[Int], fs: List[Float32]
    ) raises -> Int:
        var start = 0
        if self.program_trace.enabled:
            start = Int(perf_counter_ns())
            self.ex.sync()
            self.program_trace.sync_ns += Int(perf_counter_ns()) - start
            self.program_trace.sync_count += 1
            start = Int(perf_counter_ns())
        var h: Int
        try:
            h = self._op_run[nested](code, hs, ints, fs)
        except err:
            var context = String("program node ") + String(node) + " (" + program_op_name(code) + ") inputs=["
            for j in range(len(hs)):  # driver-loop: over failed node's input shapes
                if j > 0:
                    context += ", "
                context += shape_text(self.shapes[self._slot(hs[j])])
            var message = String(err)
            var kind = String("")
            var message_start = 0
            if message.startswith("ValueError: "):
                kind = "ValueError: "
                message_start = 12
            elif message.startswith("TypeError: "):
                kind = "TypeError: "
                message_start = 11
            elif message.startswith("OverflowError: "):
                kind = "OverflowError: "
                message_start = 15
            raise Error(kind + context + "]: " + String(message[byte=message_start:]))
        if self.program_trace.enabled:
            var issued = Int(perf_counter_ns())
            self.ex.sync()
            var waited = Int(perf_counter_ns()) - issued
            self.program_trace.sync_count += 1
            self.program_trace.sync_ns += waited
            self.program_trace.nodes.append(node)
            self.program_trace.codes.append(code)
            self.program_trace.enqueue_ns.append(issued - start)
            self.program_trace.wait_ns.append(waited)
            var shape = self.shapes[self._slot(h)]
            self.program_trace.shapes.append(shape_text(shape))
            self.program_trace.output_bytes.append(shape.count() * 4)
            var reserved = 0
            for j in range(len(self.tape._capacities)):  # driver-loop: over scratch allocation slots, profiling only
                reserved += self.tape._capacities[j] * 4
            self.program_trace.peak_tape_bytes = max(self.program_trace.peak_tape_bytes, reserved)
        return h

    def run_program(mut self, pid: Int, srcs: List[CastIn], dsts: List[FP]) raises:
        """Replay program pid on the caller's input arrays `srcs` and write
        its outputs into `dsts` (each float32, row-major, of the output's
        shape): the inputs are uploaded into the session's scratch tape,
        every node runs through `_op_run` with its recorded arguments (the
        eager call's kernels and words), the parameters are read in place,
        and the scratch tape's storage is kept for the next run."""
        if self.tape.is_recording():
            raise Error("ops: programs run in an inference session (Session(record=False))")
        var pi = self._program_index(pid)
        var n_in = self.programs[pi].n_in
        if len(srcs) != n_in:
            raise Error("ops: this program takes " + String(n_in) + " inputs")
        if len(dsts) != len(self.programs[pi].outs):
            raise Error("ops: this program has " + String(len(self.programs[pi].outs)) + " outputs")
        for k in range(n_in):  # driver-loop: over program inputs
            var sh = self.programs[pi].in_shapes[k]
            srcs[k].check("ops: program input")
            if srcs[k].rows != sh.rows() or srcs[k].cols != sh.cols():
                raise Error("ops: program input " + String(k) + " must have the shape " + shape_text(sh))
        if len(self.program_replays[pi]) == 0:
            self.program_replays[pi].append(PreparedInferenceReplay(
                self.programs[pi], _SLOT_BITS, _SLOT_MASK))
        # Loan private metadata/scratch to this invocation, never aliasing the
        # mutable session during dispatch. Return its capacity on every exit.
        var prepared = self.program_replays[pi].pop()
        try:
            prepared.clear_live()
            for j in range(len(self.programs[pi].params)):  # driver-loop: over program parameters; refresh and validate live ownership once per invocation
                var h = self.programs[pi].params[j]
                var slot = prepared.slots[j]
                var gen = prepared.generations[j]
                if (h <= 0 or slot < 0 or slot >= len(self.live) or not self.live[slot]
                        or self.generation[slot] != gen or self.owned[slot] == 0):
                    raise Error("ops: a parameter of program " + String(pid) + " was freed")
                if not same_shape(self.shapes[slot], self.programs[pi].param_shapes[j]):
                    raise Error("ops: a parameter of program " + String(pid) + " changed shape")
                prepared.buffers.append(self.owned[slot])
                prepared.integer_words.append(self.int_words[slot])
            # Scratch tape in; session tape held aside until all handles retire.
            var held = Tape()
            swap(held, self.prog_tape)
            swap(held, self.tape)
            try:
                self.tape.reset()
                self._replay(pi, srcs, dsts, prepared)
            finally:
                self._drop_temps(prepared.temporaries)
                swap(held, self.tape)
                swap(held, self.prog_tape)
        finally:
            prepared.clear_live()
            self.program_replays[pi].append(prepared^)

    def _replay(
        mut self, pi: Int, srcs: List[CastIn], dsts: List[FP], mut prepared: PreparedInferenceReplay,
    ) raises:
        prepared.replay.values.clear()
        for k in range(len(srcs)):  # driver-loop: over program inputs
            var sh = self.programs[pi].in_shapes[k]
            var buf = self.tape.reserve(self.ex, sh.count())
            if self.programs[pi].in_kinds[k] == PROGRAM_IN_IDS:
                # ids: the int32 words upload_ids gives (embedding and take read them)
                upload_cast_i32(self.ex, buf, srcs[k], "ops: program input")
            else:
                upload_cast(self.ex, buf, srcs[k])
            var h = self._leaf(buf, sh, self.programs[pi].in_kinds[k] == PROGRAM_IN_IDS)
            prepared.temporaries.append(h)
            prepared.replay.values.append(h)
        for j in range(len(prepared.buffers)):  # driver-loop: over program parameters
            var sh = self.programs[pi].param_shapes[j]
            var integer_words = prepared.integer_words[j]
            var h = self._leaf(FP(unsafe_from_address=prepared.buffers[j]), sh, integer_words)
            prepared.temporaries.append(h)
            prepared.leaves.append(h)
        var nn = self.programs[pi].n_nodes()
        for k in range(nn):  # driver-loop: over program nodes
            var code = self.programs[pi].code[k]
            var a0 = self.programs[pi].a_at[k]
            var a1 = self.programs[pi].a_at[k + 1]
            prepared.replay.handles.clear()
            for j in range(a0, a1):  # small-loop(args: a node's input refs): handle lookup, no data
                var rf = self.programs[pi].args[j]
                if rf < 0:
                    prepared.replay.handles.append(prepared.leaves[-rf - 1])
                else:
                    prepared.replay.handles.append(prepared.replay.values[rf])
            var h = self._program_op(k, code, prepared.replay.handles,
                prepared.replay.nodes[k].ints, prepared.replay.nodes[k].fs)
            prepared.temporaries.append(h)
            prepared.replay.values.append(h)
        for k in range(len(dsts)):  # driver-loop: over program outputs
            var rf = self.programs[pi].outs[k]
            var h: Int
            if rf < 0:
                h = prepared.leaves[-rf - 1]
            else:
                h = prepared.replay.values[rf]
            if not same_shape(self.shapes[self._slot(h)], self.programs[pi].out_shapes[k]):
                raise Error("ops: program output " + String(k) + " no longer has its recorded shape")
            self.download(h, dsts[k])

    def metric(mut self, kind: Int, hs: List[Int]) raises -> Float32:
        """A forward-only metric of (y_true, prediction); one value read back."""
        if len(hs) != 2:
            raise Error("ops: a metric takes y_true and a prediction")
        var ys = self._slot(hs[0])
        var ps = self._slot(hs[1])
        var rows = self.rows[ys]
        if self.cols[ys] != 1 or self.rows[ps] != rows:
            raise Error("ops: y_true must be (n, 1) and the prediction must have n rows")
        var yt_t = self.tensors[ys]
        var yt = self.tape.value_of(self.ex, yt_t)
        var pv_t = self.tensors[ps]
        var pv = self.tape.value_of(self.ex, pv_t)
        var cols = self.cols[ps]
        if kind == OPS_METRIC_ACCURACY:
            if cols == 1:
                return accuracy(self.ex, yt, pv, rows)
            return accuracy_from_scores(self.ex, yt, pv, rows, cols)
        if kind == OPS_METRIC_LOG_LOSS:
            return log_loss(self.ex, yt, pv, rows, cols)
        if kind == OPS_METRIC_ROC_AUC:
            if cols != 1:
                raise Error("ops: roc_auc takes one score column")
            return roc_auc(self.ex, yt, pv, rows)
        raise Error("ops: unknown metric code " + String(kind))

    # nn-blocks-pg begin: optimizer parameter groups (autograd/nn_pg.mojo)
    def _opt_index_single(self, o: Int, what: String) raises -> Int:
        """`_opt_index` for a surface that trains ONE configuration: an
        optimizer with parameter groups is refused by name."""
        var oi = self._opt_index(o)
        if self.opts[oi].pg >= 0:
            raise Error("ValueError: ops: " + what + " takes an optimizer without parameter groups; optimizer "
                        + String(o) + " has parameter groups (update its parameters with Session.step)")
        return oi

    def param_group_optimizer(
        mut self, request: OpsOptimizerRequest, extras: OpsOptimizerExtras,
        group_requests: List[OpsOptimizerRequest], group_extras: List[OpsOptimizerExtras], groups: List[List[Int]],
    ) raises -> Int:
        """A new optimizer with parameter groups; returns its id (>= 1).

        `request` / `extras` are the optimizer-wide settings (its kind, its
        schedule, the gradient clips); group g's settings are
        `group_requests[g]` / `group_extras[g]` (the door fills them from the
        optimizer-wide ones and the group's own keys) and its parameters are
        `groups[g]`. Every setting and every membership is checked before any
        optimizer is added: a refusal leaves the session unchanged. Each group
        becomes a member optimizer of its own (an id after the returned one);
        the table is resolved here, once (autograd/nn_pg.mojo)."""
        var n = len(groups)
        if len(group_requests) != n or len(group_extras) != n:
            raise Error("ops: param_group_optimizer takes one setting per parameter group")
        var table = pg_check_membership(groups)
        for g in range(n):  # driver-loop: over parameter groups (construction, once)
            for i in range(len(groups[g])):  # driver-loop: over one group's parameters (construction, once)
                # a session-owned tensor (made with `tensor`): Program.fit
                # trains captured parameters of an inference session, so
                # requires_grad is checked where the step runs, not here
                if self.owned[self._slot(groups[g][i])] == 0:
                    raise Error("ValueError: ops: param_groups[" + String(g) + "] parameter " + String(i)
                                + " is not a tensor made with Session.tensor")
            var r = group_requests[g]
            var x = group_extras[g]
            if (r.kind != request.kind or r.schedule != request.schedule or r.step_size != request.step_size
                    or r.gamma != request.gamma or x.clip_grad_norm != extras.clip_grad_norm
                    or x.clip_grad_value != extras.clip_grad_value):
                raise Error("ops: param_groups[" + String(g) + "] differs from the optimizer in an optimizer-wide "
                            + "setting (kind, schedule or clip)")
        var head = self._optimizer_entry(request, extras)
        var entries = List[_OpsOptimizer](capacity=n)
        for g in range(n):  # driver-loop: over parameter groups (validation before any change)
            var entry = self._optimizer_entry(group_requests[g], group_extras[g])
            entries.append(entry)
        var first = len(self.opts)
        head.pg = len(self.pg_tables)
        self.opts.append(head)
        for g in range(n):  # driver-loop: over parameter groups (one member optimizer each)
            table.members.append(len(self.opts))
            self.opts.append(entries[g])
        while len(self.opt_state_indices) < len(self.opts):  # driver-loop: over new optimizers' index tables
            self.opt_state_indices.append(Dict[Int, Int]())
        self.pg_tables.append(table^)
        return first + 1

    def _pg_index(self, o: Int) raises -> Int:
        """The table index of grouped optimizer o."""
        var oi = self._opt_index(o)
        var g = self.opts[oi].pg
        if g < 0:
            raise Error("ValueError: ops: optimizer " + String(o) + " has no parameter groups")
        return g

    def _pg_group(self, oi: Int, h: Int, what: String) raises -> Int:
        """The group index of parameter h in grouped optimizer oi."""
        ref table = self.pg_tables[self.opts[oi].pg]
        if h not in table.group_of:
            raise Error("ValueError: ops: " + what + ": the parameter is in no parameter group of optimizer "
                        + String(oi + 1))
        return table.group_of[h]

    def _pg_member(self, oi: Int, h: Int, what: String) raises -> Int:
        """The member optimizer index (opts) of parameter h's group."""
        var gi = self._pg_group(oi, h, what)
        return self.pg_tables[self.opts[oi].pg].members[gi]

    def _pg_scheduler_step(mut self, oi: Int):
        """Every group member at the grouped optimizer's epoch: one schedule
        scales every group's rate (torch's schedulers scale every group)."""
        var g = self.opts[oi].pg
        var epoch = self.opts[oi].epoch
        for k in range(len(self.pg_tables[g].members)):  # driver-loop: over parameter groups
            var m = self.pg_tables[g].members[k]
            self.opts[m].epoch = epoch

    def _pg_rates(self, oi: Int) -> List[Float32]:
        """Each group's learning rate at the schedule's current epoch, once
        per step (a host scalar of each group, never data)."""
        var g = self.opts[oi].pg
        var ngroups = len(self.pg_tables[g].members)
        var rates = List[Float32](capacity=ngroups)
        for k in range(ngroups):  # driver-loop: over parameter groups (one rate each)
            var op = self.opts[self.pg_tables[g].members[k]]
            rates.append(epoch_lr(op.cfg.lr, op.sched, op.sched_step, op.gamma, op.epoch))
        comptime if (PG_SABOTAGE_GROUP_LR and Self.E.IS_DEVICE) or (PG_SABOTAGE_GROUP_LR_HOST and not Self.E.IS_DEVICE):
            for k in range(ngroups):  # driver-loop: over parameter groups
                rates[k] = rates[0]  # SABOTAGE: every group steps at group 0's rate
        return rates^

    def _pg_step(mut self, oi: Int, hs: List[Int]) raises:
        """`step` of a grouped optimizer: each listed parameter updated by its
        group's member (its settings, its rate, its own state), the
        member's ordinary launches (autograd/nn_pg.mojo). With
        clip_grad_norm, ONE norm over every listed gradient of every group,
        in the order `hs` lists them, and ONE coefficient for all.
        Every parameter is checked before any is updated."""
        var rates = self._pg_rates(oi)
        var slots = List[Int](capacity=len(hs))
        var grads = List[FP](capacity=len(hs))
        var groups_of = List[Int](capacity=len(hs))
        var members_of = List[Int](capacity=len(hs))
        var seen = Dict[Int, Int]()
        var g = self.opts[oi].pg
        for i in range(len(hs)):  # driver-loop: over the step's parameters
            var slot = self._param_slot(hs[i], "step")
            if slot in seen:
                raise Error("ops: step lists a parameter twice")
            seen[slot] = i
            var gi = self._pg_group(oi, hs[i], "step")
            slots.append(slot)
            groups_of.append(gi)
            members_of.append(self.pg_tables[g].members[gi])
            grads.append(self.tape.gradient(self.tensors[slot]))
        var head_cfg = self.opts[oi].cfg
        if optimizer_clips(head_cfg):
            # The clip across the step's tensors of EVERY group: the fold of
            # autograd/optim.mojo `clip_norm_accumulate` (the norm in the first
            # tensor's cell), then each tensor's update reads that one
            # coefficient (`step_lr_clipped`, one launch a tensor).
            var sis = List[Int](capacity=len(slots))
            for i in range(len(slots)):  # driver-loop: over the step's parameters
                sis.append(self._state_of(members_of[i], hs[i], self.rows[slots[i]] * self.cols[slots[i]]))
            var cells = List[FP](capacity=len(slots))
            var cell = self.opt_states[sis[0]].cell
            for i in range(len(slots)):  # driver-loop: over the step's parameters
                comptime if (PG_SABOTAGE_CLIP_PER_GROUP and Self.E.IS_DEVICE) or (
                        PG_SABOTAGE_CLIP_PER_GROUP_HOST and not Self.E.IS_DEVICE):
                    if i > 0 and groups_of[i] != groups_of[i - 1]:
                        cell = self.opt_states[sis[i]].cell  # SABOTAGE: a norm of each group's own
                cells.append(cell)
            if head_cfg.clip_global:
                for i in range(len(slots)):  # driver-loop: over the step's parameters (the norm's fold order)
                    var first = i == 0
                    var last = i == len(slots) - 1
                    comptime if (PG_SABOTAGE_CLIP_PER_GROUP and Self.E.IS_DEVICE) or (
                            PG_SABOTAGE_CLIP_PER_GROUP_HOST and not Self.E.IS_DEVICE):
                        first = i == 0 or groups_of[i] != groups_of[i - 1]
                        last = i == len(slots) - 1 or groups_of[i + 1] != groups_of[i]
                    var sv = self._state_view(sis[i])
                    clip_norm_accumulate(self.ex, head_cfg, sv, grads[i], self.rows[slots[i]] * self.cols[slots[i]],
                                         cells[i], first, last)
            for i in range(len(slots)):  # driver-loop: over the step's parameters
                var cslot = slots[i]
                var cfg = self.opts[members_of[i]].cfg
                var ct = self.opt_states[sis[i]].t + 1
                var cst = self._state_view(sis[i])
                step_lr_clipped(self.ex, cfg, cst, FP(unsafe_from_address=self.owned[cslot]), grads[i],
                                self.rows[cslot] * self.cols[cslot], ct, rates[groups_of[i]], cells[i])
                self.steps_forget(self.owned[cslot])  # the weights moved: a step op re-checks them
                self.opt_states[sis[i]].t = ct
                self.opt_states[sis[i]].pw1 = cst.pw1
                self.opt_states[sis[i]].pw2 = cst.pw2
                self.opt_states[sis[i]].mu_prod = cst.mu_prod
            return
        for i in range(len(slots)):  # driver-loop: over the step's parameters
            var slot = slots[i]
            var n = self.rows[slot] * self.cols[slot]
            var si = self._state_of(members_of[i], hs[i], n)
            var t = self.opt_states[si].t + 1
            var st = self._state_view(si)
            var cfg = self.opts[members_of[i]].cfg
            step_lr(self.ex, cfg, st, FP(unsafe_from_address=self.owned[slot]), grads[i], n, t, rates[groups_of[i]])
            self.steps_forget(self.owned[slot])  # the weights moved: a step op re-checks them
            self.opt_states[si].t = t
            self.opt_states[si].pw1 = st.pw1
            self.opt_states[si].pw2 = st.pw2
            self.opt_states[si].mu_prod = st.mu_prod

    def pg_groups(self, o: Int) raises -> Int:
        """The number of parameter groups of grouped optimizer o."""
        return len(self.pg_tables[self._pg_index(o)].members)

    def pg_group_params(self, o: Int, gi: Int) raises -> List[Int]:
        """Group gi's parameter handles, in the order the group listed them."""
        var g = self._pg_index(o)
        if gi < 0 or gi >= len(self.pg_tables[g].members):
            raise Error("ValueError: ops: optimizer " + String(o) + " has no parameter group " + String(gi))
        var out = List[Int]()
        for k in range(self.pg_tables[g].starts[gi], self.pg_tables[g].starts[gi + 1]):  # driver-loop: over one group's parameters
            out.append(self.pg_tables[g].params[k])
        return out^

    def pg_group_config(self, o: Int, gi: Int) raises -> OptimizerConfig:
        """Group gi's configuration (its base learning rate and settings)."""
        var g = self._pg_index(o)
        if gi < 0 or gi >= len(self.pg_tables[g].members):
            raise Error("ValueError: ops: optimizer " + String(o) + " has no parameter group " + String(gi))
        return self.opts[self.pg_tables[g].members[gi]].cfg

    def pg_learning_rates(self, o: Int) raises -> List[Float32]:
        """Every group's learning rate at the schedule's current epoch."""
        _ = self._pg_index(o)
        return self._pg_rates(o - 1)

    def pg_epoch(self, o: Int) raises -> Int:
        """The schedule's epoch of grouped optimizer o (every group's)."""
        _ = self._pg_index(o)
        return self.opts[o - 1].epoch

    def pg_set_epoch(mut self, o: Int, epoch: Int) raises:
        """Set the schedule's epoch of grouped optimizer o and every group."""
        _ = self._pg_index(o)
        if epoch < 0:
            raise Error("ValueError: ops: the schedule's epoch must be >= 0")
        self.opts[o - 1].epoch = epoch
        self._pg_scheduler_step(o - 1)

    def pg_group_request(self, o: Int, gi: Int) raises -> Tuple[OpsOptimizerRequest, OpsOptimizerExtras]:
        """Group gi's settings as the request that builds them (the base a
        loaded group's keys are applied over)."""
        var g = self._pg_index(o)
        if gi < 0 or gi >= len(self.pg_tables[g].members):
            raise Error("ValueError: ops: optimizer " + String(o) + " has no parameter group " + String(gi))
        var m = self.opts[self.pg_tables[g].members[gi]]
        var c = m.cfg
        var request = OpsOptimizerRequest(c.kind, c.nesterov, m.sched, m.sched_step, c.lr, c.momentum, c.beta1,
                                          c.beta2, c.eps, c.weight_decay, m.gamma, c.dampening)
        var extras = OpsOptimizerExtras(c.alpha, c.centered, c.lr_decay, c.init_acc,
                                        c.clip if c.clip_global else Float32(0), c.clip_value)
        return (request, extras)

    def pg_load_group(
        mut self, o: Int, gi: Int, request: OpsOptimizerRequest, extras: OpsOptimizerExtras, commit: Bool,
    ) raises:
        """Replace group gi's settings (a loaded state dict's group): the
        same checks as construction; the kind, the schedule, the clips and the
        state buffers the group keeps (`optimizer_slots`) must not change, so
        its retained state stays valid. `commit` False checks only (the
        caller checks every group before it changes any)."""
        var g = self._pg_index(o)
        if gi < 0 or gi >= len(self.pg_tables[g].members):
            raise Error("ValueError: ops: optimizer " + String(o) + " has no parameter group " + String(gi))
        var m = self.pg_tables[g].members[gi]
        var entry = self._optimizer_entry(request, extras)
        var now = self.opts[m]
        var head = self.opts[o - 1]
        if (entry.cfg.kind != now.cfg.kind or entry.sched != now.sched or entry.sched_step != now.sched_step
                or entry.gamma != now.gamma or entry.cfg.clip != head.cfg.clip
                or entry.cfg.clip_global != head.cfg.clip_global or entry.cfg.clip_value != head.cfg.clip_value):
            raise Error("ValueError: ops: param_groups[" + String(gi) + "] of the loaded state changes an "
                        + "optimizer-wide setting (kind, schedule or clip)")
        var a = optimizer_slots(entry.cfg)
        var b = optimizer_slots(now.cfg)
        if a[0] != b[0] or a[1] != b[1] or a[2] != b[2]:
            raise Error("ValueError: ops: param_groups[" + String(gi) + "] of the loaded state changes which "
                        + "state buffers the group keeps (momentum or centered); build the optimizer with "
                        + "those settings")
        if entry.cfg.init_acc != now.cfg.init_acc and len(self.opt_state_indices[m]) > 0:
            raise Error("ValueError: ops: param_groups[" + String(gi) + "] of the loaded state changes "
                        + "initial_accumulator_value after its state was made")
        if commit:
            self.opts[m].cfg = entry.cfg
    # nn-blocks-pg end

    # nn-blocks-cbw begin: user-defined backward rules (autograd/nn_cbw.mojo).
    # A custom operation is two captured programs (forward, backward) bound
    # once (cbw_define, or the first replay of a loaded program holding one).
    # A call runs the forward without recording and, in a recording session,
    # records ONE tape node (tag NN_CBW_TAG) whose rule (CbwRules below) runs
    # the backward program on the node's incoming gradient and hands the tape
    # one gradient per input, which the tape accumulates like any other
    # contribution. No Python is called in either direction.
    def cbw_define(mut self, var forward: OpsProgram, var backward: OpsProgram, saved: List[Int], n_out: Int) raises -> Int:
        """Bind a custom operation (the checks of autograd/nn_cbw.mojo
        cbw_bind); returns its id (>= 1). Binding the same pair again returns
        the same id."""
        var op = cbw_bind(forward^, backward^, saved, n_out)
        return self.cbw.add(op^) + 1

    def cbw_call(mut self, op_id: Int, hs: List[Int]) raises -> List[Int]:
        """Call custom operation `op_id` on the tensors `hs`; one handle per
        returned output. It is one OPS_CUSTOM_VJP operation holding both
        programs (an inference session logs it, so a capture holds it and a
        saved program carries both graphs), then, for several outputs, one
        narrow and one reshape per output over its (1, total) result."""
        if op_id < 1 or op_id > len(self.cbw.ops):
            raise Error("ValueError: custom: " + String(op_id) + " is not a custom operation of this session")
        var words = self.cbw.ops[op_id - 1].words.copy()
        var floats = self.cbw.ops[op_id - 1].floats.copy()
        var n_out = self.cbw.ops[op_id - 1].n_out
        var res = self.op(OPS_CUSTOM_VJP, hs, words, floats)
        var outs = List[Int](capacity=n_out)
        if n_out == 1:
            outs.append(res)
            return outs^
        var shapes = self.cbw.last_out_shapes.copy()
        var spec = List[Int]()
        var dims = List[Int]()
        var one = List[Int]()
        for k in range(n_out):  # driver-loop: over the operation's outputs
            spec.clear()
            spec.append(1)
            spec.append(k)
            spec.append(0)
            for j in range(n_out):  # small-loop(n_out: the operation's outputs): narrow sizes
                spec.append(shapes[j].count())
            one.clear()
            one.append(res)
            var piece = self.op(OPS_NARROW, one, spec, List[Float32]())
            dims.clear()
            dims.append(0)
            for j in range(shapes[k].ndim):  # small-loop(ndim: tensor rank, at most 8): shape words
                dims.append(shapes[k].d[j])
            one.clear()
            one.append(piece)
            outs.append(self.op(OPS_RESHAPE_N, one, dims, List[Float32]()))
        return outs^

    def cbw_first_order(self, what: String) raises:
        """Refuse a recorded (create_graph) backward through a custom rule."""
        if self.cbw.holds_rules():
            raise Error("ValueError: custom: " + what + " through a user-defined backward rule is not supported: "
                        + "custom rules are first order only (the engine does not differentiate the backward "
                        + "program); call reset() before recording a graph without one")

    def cbw_backward_scalar(mut self, t: Tensor, rules: StepOpRules, accumulate: Bool) raises:
        """The scalar backward of a graph that may hold custom nodes: `rules`
        plus the custom tag (identical to `rules` on every other node)."""
        var wrapped = CbwRules[Self.E](rules, Int(UnsafePointer(to=self)))
        if self.layer_nodes:
            # session-ops: custom nodes and Sequential layer nodes in one graph
            self.tape.backward_scalar_with_rules(self.ex, t, LayerRules[CbwRules[Self.E]](wrapped), accumulate)
            return
        self.tape.backward_scalar_with_rules(self.ex, t, wrapped, accumulate)

    def cbw_backward_seeded(mut self, t: Tensor, seed: Tensor) raises:
        """The seeded first-order backward of a graph that may hold custom
        nodes (StepOpRules: ImportRules plus the step ops' tags)."""
        var wrapped = CbwRules[Self.E](StepOpRules(self.steps_address()), Int(UnsafePointer(to=self)))
        if self.layer_nodes:
            # session-ops: custom nodes and Sequential layer nodes in one graph
            self.tape.backward_with_rules(self.ex, t, seed, LayerRules[CbwRules[Self.E]](wrapped), False)
            return
        self.tape.backward_with_rules(self.ex, t, seed, wrapped, False)

    def _cbw_op(mut self, ints: List[Int], fs: List[Float32]) raises -> Int:
        """The bound op a node's words name: one dictionary read, or (a loaded
        program's first replay) decoded, bound and kept."""
        var i = self.cbw.find(ints)
        if i >= 0:
            return i
        var op = cbw_from_words(ints, fs)
        return self.cbw.add(op^)

    def _cbw_forward(mut self, hs: List[Int], ints: List[Int], fs: List[Float32], x: List[Tensor]) raises -> Int:
        """OPS_CUSTOM_VJP: the forward program on `hs` without recording; in a
        recording session one custom tape node over the inputs (its value the
        output, or the outputs packed in one (1, total) row)."""
        var oi = self._cbw_op(ints, fs)
        var recording = self.tape.is_recording()
        var op = CbwOp()
        swap(op, self.cbw.ops[oi])
        var temps = List[Int]()
        var result = 0
        try:
            result = self._cbw_run_forward(oi, op, hs, x, recording, temps)
        finally:
            if recording:
                self.tape.record()
            swap(op, self.cbw.ops[oi])
            self._drop_temps(temps)
        return result

    def _cbw_run_forward(
        mut self, oi: Int, op: CbwOp, hs: List[Int], x: List[Tensor], recording: Bool, mut temps: List[Int],
    ) raises -> Int:
        var n_in = op.forward.n_in
        if len(hs) != n_in:
            raise Error("ValueError: custom: this custom operation takes " + String(n_in) + " inputs, got "
                        + String(len(hs)))
        for k in range(n_in):  # driver-loop: over the operation's inputs
            var sh = self.shape_nd(hs[k])
            if sh.ndim != op.forward.in_shapes[k].ndim:
                raise Error("ValueError: custom: input " + String(k) + " has shape " + shape_text(sh)
                            + "; the forward program was captured with " + shape_text(op.forward.in_shapes[k])
                            + " (the rank must match)")
            var ids = op.forward.in_kinds[k] == PROGRAM_IN_IDS
            if self.int_words[self._slot(hs[k])] != ids:
                if ids:
                    raise Error("ValueError: custom: input " + String(k) + " must be an ids tensor (Session.ids)")
                raise Error("ValueError: custom: input " + String(k) + " must be a float tensor, not ids")
        if recording:
            self.tape.no_grad()
        var values = List[Int](capacity=n_in + op.forward.n_nodes())
        for k in range(n_in):  # driver-loop: over the operation's inputs
            values.append(hs[k])
        var args = List[Int]()
        for k in range(op.forward.n_nodes()):  # driver-loop: over the forward program's nodes
            args.clear()
            for j in range(op.forward.a_at[k], op.forward.a_at[k + 1]):  # small-loop(args: a node's input refs): handle lookup
                args.append(values[op.forward.args[j]])
            var h = self._program_op[nested=True](k, op.forward.code[k], args, op.f_ints[k], op.f_fs[k])
            temps.append(h)
            values.append(h)
        # every forward output; one the forward returns unchanged from its
        # inputs is copied (a node never shares an input's storage)
        var outs = List[Int](capacity=len(op.forward.outs))
        var no_ints = List[Int]()
        var no_fs = List[Float32]()
        for k in range(len(op.forward.outs)):  # driver-loop: over the forward program's outputs
            var rf = op.forward.outs[k]
            var h = values[rf]
            if rf < n_in:
                args.clear()
                args.append(h)
                h = self._op_run[nested=True](OPS_COPY, args, no_ints, no_fs)
                temps.append(h)
            outs.append(h)
        if recording:
            self.tape.record()
        var n_out = op.n_out
        self.cbw.last_out_shapes.clear()
        var total = 0
        for k in range(n_out):  # driver-loop: over the returned outputs
            var sh = self.shape_nd(outs[k])
            self.cbw.last_out_shapes.append(sh)
            total += sh.count()
        var out_fp: FP
        var out_shape: ShapeN
        var offsets = List[Int](capacity=n_out)
        if n_out == 1:
            var t0 = self.tensor(outs[0])
            out_fp = self.tape.value_of(self.ex, t0)
            out_shape = self.shape_nd(outs[0])
            offsets.append(0)
        else:
            out_fp = self.tape.reserve(self.ex, total)
            var at = 0
            for k in range(n_out):  # driver-loop: over the returned outputs (one copy each into the packed row)
                var tk = self.tensor(outs[k])
                var a = Args()
                a.p0 = self.tape.value_of(self.ex, tk)
                a.p1 = out_fp + at
                var n = self.cbw.last_out_shapes[k].count()
                self.ex.launch[OP_COPY](a, n)
                offsets.append(at)
                at += n
            out_shape = ShapeN.matrix(1, total)
        if not recording:
            if n_out == 1:
                # the forward's own result: kept, not dropped with the temporaries
                for j in range(len(temps)):  # driver-loop: over this call's temporary handles
                    if temps[j] == outs[0]:
                        temps[j] = temps[len(temps) - 1]
                        _ = temps.pop()
                        break
                return outs[0]
            return self._constant(out_fp, 1, total, 0)
        var rec = CbwRecord()
        rec.op = oi
        rec.out_shapes = self.cbw.last_out_shapes.copy()
        rec.out_offsets = offsets^
        for k in range(n_in):  # driver-loop: over the operation's inputs
            rec.in_shapes.append(self.shape_nd(hs[k]))
        for j in range(len(op.saved)):  # driver-loop: over the saved tensors
            var h = outs[op.saved[j]]
            var ts = self.tensor(h)
            var fp = self.tape.value_of(self.ex, ts)
            rec.saved.append(Int(fp))
            rec.saved_shapes.append(self.shape_nd(h))
            rec.saved_words.append(self.int_words[self._slot(h)])
        # Records live with the graph: a reset tape (Program.fit resets it per
        # batch; Session.reset clears them) restarts its node indices, so a
        # node index at or below the last record's starts a new graph.
        var node = len(self.tape._nodes)
        var last = len(self.cbw.records) - 1
        if last >= 0 and node <= self.cbw.records[last].node:
            self.cbw.records.clear()
        rec.node = node
        var saved_args = Args()
        saved_args.i0 = len(self.cbw.records)
        saved_args.i1 = node
        self.cbw.records.append(rec^)
        var t = self.tape.custom_nd(self.ex, x, out_fp, out_shape, NN_CBW_TAG, saved_args)
        if t._index != node:
            raise Error("custom: the tape recorded the custom node at an unexpected index")
        return self._add(t, 0, False, 0)

    def _cbw_apply(mut self, ri: Int, node: Int, upstream: FP, gradients: List[FP]) raises:
        """The rule of custom node `node` (record `ri`): the backward program on
        the saved tensors and the incoming gradient, on the session's scratch
        tape, each gradient copied into its destination (the tape adds it to
        the input's other contributions). Called by CbwRules.apply while the
        session's tape runs its backward: the tape is moved aside for the call
        and put back word for word, as Session.run_program does."""
        if ri < 0 or ri >= len(self.cbw.records) or self.cbw.records[ri].node != node:
            raise Error("ValueError: custom: the recorded graph no longer holds this custom operation's saved "
                        + "tensors; run its forward again")
        var rec = CbwRecord()
        swap(rec, self.cbw.records[ri])
        var oi = rec.op
        var op = CbwOp()
        swap(op, self.cbw.ops[oi])
        var held = Tape()
        swap(held, self.prog_tape)
        swap(held, self.tape)
        var temps = List[Int]()
        try:
            self.tape.reset()
            self.tape.no_grad()
            self._cbw_run_backward(op, rec, upstream, gradients, temps)
        finally:
            self._drop_temps(temps)
            swap(held, self.tape)
            swap(held, self.prog_tape)
            swap(op, self.cbw.ops[oi])
            swap(rec, self.cbw.records[ri])

    def _cbw_run_backward(
        mut self, op: CbwOp, rec: CbwRecord, upstream: FP, gradients: List[FP], mut temps: List[Int],
    ) raises:
        var values = List[Int](capacity=op.backward.n_in + op.backward.n_nodes())
        for j in range(len(rec.saved)):  # driver-loop: over the saved tensors
            var h = self._leaf(FP(unsafe_from_address=rec.saved[j]), rec.saved_shapes[j], rec.saved_words[j])
            temps.append(h)
            values.append(h)
        for k in range(op.n_out):  # driver-loop: over the incoming gradients, one per output
            var g = upstream + rec.out_offsets[k]
            comptime if NN_CBW_SABOTAGE_SEED_ONE:
                # fault injection: a rule that assumes the incoming gradient is 1
                var n = rec.out_shapes[k].count()
                g = self.tape.reserve(self.ex, n)
                var fill = Args()
                fill.p0 = g
                fill.f0 = Float32(1)
                self.ex.launch[OP_FILL](fill, n)
            var h = self._leaf(g, rec.out_shapes[k])
            temps.append(h)
            values.append(h)
        var args = List[Int]()
        for k in range(op.backward.n_nodes()):  # driver-loop: over the backward program's nodes
            args.clear()
            for j in range(op.backward.a_at[k], op.backward.a_at[k + 1]):  # small-loop(args: a node's input refs): handle lookup
                args.append(values[op.backward.args[j]])
            var h = self._program_op[nested=True](k, op.backward.code[k], args, op.b_ints[k], op.b_fs[k])
            temps.append(h)
            values.append(h)
        for j in range(len(op.grad_inputs)):  # driver-loop: over the differentiable inputs
            var i = op.grad_inputs[j]
            var h = values[op.backward.outs[j]]
            var sh = self.shape_nd(h)
            if not same_shape(sh, rec.in_shapes[i]):
                raise Error("ValueError: custom: the backward program's gradient " + String(j) + " has shape "
                            + shape_text(sh) + " at run time; input " + String(i) + " has shape "
                            + shape_text(rec.in_shapes[i]))
            if self.int_words[self._slot(h)]:
                raise Error("ValueError: custom: the backward program's gradient " + String(j)
                            + " holds integer words, not float values")
            var tg = self.tensor(h)
            var a = Args()
            a.p0 = self.tape.value_of(self.ex, tg)
            a.p1 = gradients[i]
            self.ex.launch[OP_COPY](a, sh.count())
    # nn-blocks-cbw end


# nn-blocks-cbw begin: the rule set of a graph that may hold custom nodes
@fieldwise_init
struct CbwRules[S: Exec & Movable & Deinitable](BackwardRules, ImplicitlyCopyable, Movable):
    """StepOpRules plus NN_CBW_TAG, whose rule runs the custom operation's
    captured backward program in the session at address `session` (an
    OpsSession[S]; an open session never moves). Every other tag is
    StepOpRules' own, so a graph without custom nodes differentiates exactly
    as with StepOpRules."""
    var inner: StepOpRules
    var session: Int

    def supports(self, tag: Int) -> Bool:
        if tag == NN_CBW_TAG:
            return True
        return self.inner.supports(tag)

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        if tag == NN_CBW_TAG:
            # the rule overwrites every element of each differentiable
            # input's destination; an ids input's is never read
            return False
        return self.inner.zeroed_destination(tag, saved, edge)

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        if tag == NN_CBW_TAG:
            comptime if NN_CBW_SABOTAGE_DROP_PATH:
                # fault injection: the copy lands on the input's whole
                # gradient, dropping another consumer's earlier term
                return True
            return False
        return self.inner.writes_total(tag, saved, edge)

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        if tag != NN_CBW_TAG:
            self.inner.apply(ex, tag, saved, output, upstream, inputs, gradients, rows, cols)
            return
        # The session's own executor (the same object as `ex`) runs the
        # backward program's operations.
        var s = UnsafePointer[OpsSession[Self.S], MutUntrackedOrigin](unsafe_from_address=self.session)
        s[]._cbw_apply(saved.i0, saved.i1, upstream, gradients)
# nn-blocks-cbw end


def _keeps_shape(code: Int) -> Bool:
    """Ops whose output has x's shape: elementwise maps, dropout, the row
    softmaxes and layer norm, detach."""
    return (code == OPS_RELU or code == OPS_SIGMOID or code == OPS_TANH or code == OPS_EXP
            or code == OPS_LOG or code == OPS_SQRT or code == OPS_SCALE or code == OPS_SOFTMAX
            or code == OPS_LOG_SOFTMAX or code == OPS_LAYER_NORM or code == OPS_DROPOUT
            or code == OPS_DETACH or code == OPS_RMS_NORM or code == OPS_ROTARY or code == OPS_TRANSFORMER_SOFTMAX)


# ---------------------------------------------------------------- the session table
# One lock word guards the table (ids, addr, kind, next_id): open, close and the
# lookup take it, nothing else. A session's storage is allocated once and never
# moves, so the pointer a lookup returns stays valid after the lock is released
# until that session is closed; its owner (a door's prepared record, a member
# state) serializes its own calls. Every caller looks a session up once per
# public operation and runs its forwards, layers, batches and tokens through
# the returned pointer: no loop takes the lock. The table used to rely on the
# GIL, which member workers (GIL released, in parallel) do not hold.
# (partition_entries and distributed_entries reuse this struct for their own
# tables; they carry the word and do not take it yet.)
struct _OpsStore(Defaultable, Movable):
    var ids: List[Int]
    var addr: List[Int]
    var kind: List[Int]
    var next_id: Int
    var synchronization: List[Int64]

    def __init__(out self):
        self.ids = List[Int]()
        self.addr = List[Int]()
        self.kind = List[Int]()
        self.next_id = 1
        self.synchronization = List[Int64](length=2, fill=0)


comptime _OPS_NAME = "MojolearnAutogradOpsSessionsIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnAutogradOpsSessionsFastV1"
comptime _OPS_SESSIONS = SharedSlot[_OpsStore, _OPS_NAME, 2]


@always_inline
def _ops_lock_word(s: UnsafePointer[_OpsStore, MutUntrackedOrigin]) -> MutPointer[Int64, MutUntrackedOrigin]:
    # The list's buffer is allocated once at creation and never grows.
    return rebind[MutPointer[Int64, MutUntrackedOrigin]](s[].synchronization.unsafe_ptr())


def ops_session_registry_token() raises -> Int:
    # Identity of this native registry image, never an external object cache.
    # A frontend may retain/compare it, but must not dereference it.
    return Int(_OPS_SESSIONS.get_or_create_ptr())


def ops_session_open[E: Exec & Movable & Deinitable, KIND: Int](var ex: E, record: Bool) raises -> Int:
    """A new session on `ex`; returns its id (>= 1)."""
    var p = alloc[OpsSession[E]](1)
    p.unsafe_write(OpsSession[E](ex^, record))
    var s = _OPS_SESSIONS.get_or_create_ptr()
    var lock = _ops_lock_word(s)
    shared_slot_lock(lock)
    var sid = s[].next_id
    s[].next_id += 1
    s[].ids.append(sid)
    s[].addr.append(Int(p))
    s[].kind.append(KIND)
    shared_slot_unlock(lock)
    return sid


def _find_held(s: UnsafePointer[_OpsStore, MutUntrackedOrigin], sid: Int, kind: Int) -> Int:
    # Under the table's lock. -1: not open; -2: another binding's session.
    for i in range(len(s[].ids)):  # small-loop(ids: open sessions): a session lookup, no data
        if s[].ids[i] == sid:
            if s[].kind[i] != kind:
                return -2
            return i
    return -1


def _find_refused(sid: Int, found: Int) -> Error:
    if found == -2:
        return Error("ops: session " + String(sid) + " belongs to another binding")
    return Error("ops: session " + String(sid) + " is not open")


def ops_session[E: Exec & Movable & Deinitable, KIND: Int](sid: Int) raises -> UnsafePointer[OpsSession[E], MutUntrackedOrigin]:
    """The open session `sid` of this executor kind. One lookup per public
    operation: everything that operation runs goes through the pointer."""
    var s = _OPS_SESSIONS.get_or_create_ptr()
    var lock = _ops_lock_word(s)
    shared_slot_lock(lock)
    var i = _find_held(s, sid, KIND)
    var address = 0
    if i >= 0:
        address = s[].addr[i]
    shared_slot_unlock(lock)
    if i < 0:
        raise _find_refused(sid, i)
    return UnsafePointer[OpsSession[E], MutUntrackedOrigin](unsafe_from_address=address)


def ops_session_close[E: Exec & Movable & Deinitable, KIND: Int](sid: Int) raises:
    """Release session `sid`'s storage and drop it from the table."""
    var s = _OPS_SESSIONS.get_or_create_ptr()
    var lock = _ops_lock_word(s)
    shared_slot_lock(lock)
    var i = _find_held(s, sid, KIND)
    var address = 0
    if i >= 0:
        address = s[].addr[i]
        _ = s[].ids.pop(i)
        _ = s[].addr.pop(i)
        _ = s[].kind.pop(i)
    shared_slot_unlock(lock)
    if i < 0:
        raise _find_refused(sid, i)
    # Out of the table: no lookup reaches it while it closes.
    var p = UnsafePointer[OpsSession[E], MutUntrackedOrigin](unsafe_from_address=address)
    try:
        p[].close()
    except e:
        p.unsafe_deinit_pointee()
        p.free()
        raise e
    p.unsafe_deinit_pointee()
    p.free()
