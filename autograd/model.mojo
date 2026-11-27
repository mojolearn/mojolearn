# SPDX-License-Identifier: Apache-2.0
"""A user-defined model, trained through the tape from a flat description.

Python assembles the description (an integer program and a float table,
plan AUTOGRAD_API (git history) "Model description"); `parse_model` validates it
once per call and `model_fit` / `model_predict` interpret it. Each batch
records the forward graph on the tape, differentiates the loss and runs one
optimizer step; every number is computed by executor launches (parallel, with
the tape's fixed blocked folds), so the bits are the same on Apple, NVIDIA
and AMD. Host loops run over layers, batches and epochs only.
"""
from std.memory import bitcast, unsafe_memcpy
from sequence.exec_trait import Exec
from core.fit_loop_clock import FitLoop, fit_loop_start, fit_loop_stop
from sequence.fit_loop_words import fit_loop_publish
from sequence.ops import (
    FP, Args, OP_COPY, OP_FILL, OP_GATHER_ROWS, OP_MLP_PERM, OP_ONE_HOT, OP_PROBA2, OP_SEQ_IOTA,
    OPT_ADAM, OPT_ADAMW, OPT_SGD, dummy_ptr,
)
from sequence.mlp import mlp_epoch_key, mlp_perm_args
from sequence.adafactor_update import adafactor_partial_words, adafactor_update
from sequence.recurrent import OptConfig, OptState, opt_step
from sequence.gemm_split import gemm_split_shape
from checks.numerics import ftz, identical_div, identical_pow64, identical_sqrt
from autograd.tape import Tape, Tensor
from autograd.ops import RELU, SIGMOID, TANH, SOFTMAX
from autograd.layers import NeuralRules, layer_norm
# TOMBSTONE: MOJOLEARN_AUTOGRAD_GATHER_EPOCH_OFF (per-batch gather, old path) folded here 2026-11-15 by
# lane/expdel2-24-neural (owner expdel2-23-neural, autograd/data.mojo); code recoverable at 988a3742d1.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_GATHER_EPOCH_OFF.expdel2-24-neural.patch
from autograd.data import gather_epoch, gather_epoch_fits
from autograd.epoch_monitor import EpochMonitor
from autograd.model_ops import OP_AGM_INIT_UNIFORM, OP_AGM_ROW_ARGMAX, OP_AGM_THRESHOLD
from core.cast_in import CAST_KIND_SIGNED, CastIn
from autograd.cast_ops import gather_labels_device, upload_cast, validation_codes
# Description version 2 (lane ag-rankn-python, plan AUTOGRAD_RANKN (git history)
# section 5): per-sample input shapes of any rank and the shape layers.
from autograd.rankn import RN_MAX_RANK, ShapeN
from autograd.rankn_conv import (
    rankn_conv1d, rankn_conv2d, rankn_max_pool1d, rankn_max_pool2d, rankn_avg_pool1d, rankn_avg_pool2d,
)
from autograd.rankn_layout import rankn_concat
from autograd.rankn_api import RankNRules, rn_as_shape, rn_dims, rn_resolve, rn_shape, rn_with_batch
# Lane neural-blocks (2026-11-22): the neural building blocks (codes 32..41) and the version-2 backward rules
from autograd.model_blocks import (
    LAYER_MULTIPLY, LAYER_RECURRENT, ModelRules, block_forward, block_init_bounds, block_param_sizes, block_reads,
    block_shape, bn_state, is_block, BLOCK_INIT_S4D,
)
# lm_head (lane neural-blocks-3-lm): the token cross-entropy's head layer and its fused loss
from autograd.model_blocks import LAYER_LM_HEAD, block_lm_head_loss
# nn-conv begin
from autograd.nn_conv_blocks import nnc_is_projection
# nn-conv end
# nn-losses begin
from autograd.losses import REDUCTION_MEAN, REDUCTION_SUM, REDUCTION_BATCHMEAN, ls_per_row, cross_entropy
from autograd.loss_ops import LS_MSE, LS_MAE, LS_HUBER, LS_BCE_LOGITS
from autograd.nn_losses import LS_SMOOTH_L1, LS_NLL, LS_KL_DIV
# nn-losses end
# nn-losses-2 begin
from autograd.nn_log_softmax import is_nn_log_softmax, log_softmax_axis
from autograd.nn_ctc import ctc_loss, ctc_blank_of, ctc_zero_infinity_of, ctc_admit, ctc_pack_targets
from core.cast_in import CAST_KIND_FLOAT
# nn-losses-2 end
from autograd.ssm_ops import OP_SSM_S4D_INIT
# nn-attention begin
from autograd.nn_attention import (
    nn_attention_is, nn_attention_source, nn_attention_check_source, nn_attention_forward,
    nn_attention_key_padding_length, NN_MHA_SELF,
)
# nn-attention end
# lane nn-optim (2026-11-23): rmsprop, adagrad, lamb and the schedules multi_step .. reduce_on_plateau
from autograd.nn_optim import (
    NN_OPT_ADAFACTOR, NN_OPT_ADAGRAD, NN_SCHED_MULTI_STEP, NN_SCHED_PLATEAU, NnLamb, NnPlateau, nn_epoch_lr, nn_multi_step_count,
    nn_opt_step, nn_optim_admit, nn_sched_per_step, nn_step_lrs, nn_opt_keeps_third,
)
# lane stream-fits (2026-11-25): Sequential.fit(source=), the streamed instance of the fit loop
from std.math import nan
from autograd.batch_source import (
    BatchSource, BS_FIELD_STEPS, BS_FIELD_LENGTHS, BS_FIELD_TARGETS, BATCH_SOURCE_STAGE_WORDS, ProgramTrainingSource,
    SOURCE_PART_WORDS, batch_source_upload, batch_source_fit_saved, batch_source_check_classes, source_part_store,
    batch_source_class_scan,
)
# nn-blocks-pgseq begin: Sequential.compile's parameter groups (one step a run of one group's tensors)
from autograd.nn_pgseq import (
    PgSeqLamb, PgSeqState, PgSeqTable, pgseq_admit_fit, pgseq_fill_accumulators, pgseq_step,
)
# nn-blocks-pgseq end

# Tape.dropout is part of the native tape. Its mask is keyed by seed,
# optimizer step and layer, including across checkpoint/resume.
comptime HAS_DROPOUT = True
# TOMBSTONE: MOJOLEARN_AUTOGRAD_HAS_INFERENCE (tape.no_grad around predict, opt-in arm, no A/B recorded) deleted
# 2026-11-15 by lane/expdel2-24-neural; code recoverable at fd8abcf837.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_HAS_INFERENCE.patch

# LAUNCHES (lane ag-pass1-launches, read from the code; tape.mojo's
# _linear_to and softmax_cross_entropy have the per-node detail).
# Default: a Linear layer and the activation layer right after it are one
# tape.linear node, and cross_entropy against class codes is one
# tape.softmax_cross_entropy node.
#   Linear(+bias)+activation, per layer: forward 1 (gemm_epi; + the row
#     softmax for Softmax), backward 3 (dW, db, dX with the activation
#     derivative of the layer below in its epilogue; 2 when the input needs
#     no gradient); was forward 3, backward 7 (activation VJP 1, add VJP
#     1, bias VJP 3, 2 GEMMs; 6 for the first layer), and the unfused
#     composition is now 6 (the bias fold reads the upstream directly).
#   cross_entropy on codes: forward 3, backward 1; was the dense one-hot
#     composition, forward 7, backward 5 (and a one-hot target of N x out
#     floats where the codes take N).
#   Seed: one fill straight into the loss's gradient (was fill + copy).
# Bits: the fused path is a bit version of its own (its GEMM epilogue and
# column-sum bias gradient sum in a different order than the composed ops);
# TOMBSTONE: MOJOLEARN_AUTOGRAD_MODEL_UNFUSED (the composed nodes, opt-in check arm, no A/B recorded) deleted
# 2026-11-15 by lane/expdel2-24-neural; code recoverable at 5cc4c35385.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_MODEL_UNFUSED.patch
# (Sweep 2026-11-15: the MODEL_UNFUSED = False constant removed; nothing read it.)

# ---------------------------------------------------------------- whole-set predict (lane seq-predict-tiled, 2026-10-28)
#: Whole-set predict (REPORT idea 7; the default since 2026-11-24, see PROMOTED below): model_predict
#: runs the rows in one chunk (one launch chain: one fused GEMM per Linear, the activations, the output
#: transform) instead of the caller's 4096-row chunks (sequential_execution.mojo passes 4096), as long as
#: the inference scratch fits PREDICT_WHOLE_SET_BYTES. At 4096 rows a 64 x 64 micro-tiled GEMM is 64
#: blocks a 64-wide output, under one block per multiprocessor on a 142-SM / 304-CU part, and every
#: chunk repeats the launches; one chunk fills the device and launches each op once.
#: SAME BITS (identical, every column): every row's values come from per-row chains. The GEMM routes
#: (cell, 16 x 16 tile, 64 x 64 micro-tile, thin tiles, coop) all fold acc = fma3(A(m, k), B(k, n), acc),
#: k ascending from +0.0 (sequence/gemm_tiled.mojo), so the row count only picks the schedule; the one
#: shape-keyed fold is the split-K order (sequence/gemm_split.mojo gemm_split_shape, a function of M as
#: well), so predict_chunk_rows keeps the caller's chunk unless every Linear takes the same split
#: decision at the old full chunk, the old last chunk, the new chunk and the new last chunk. The
#: activations, LayerNorm forward (one row a thread or a coop cell, the same chain), residual adds and
#: the transforms are per row or per element. Version-2 descriptions (conv2d, pools: row counts become
#: B H W inside) keep the caller's chunk.
#: The budget is cost/hardware reasoning, never a shape: 1 GiB of float32 scratch is a third of the
#: sequence pool's cap (SEQ_POOL_BYTES 3 GiB, so a predict never evicts the pooled fit buffers), a small
#: share of every supported GPU's memory, and keeps every buffer under 2^28 floats so the kernels' Int32
#: element offsets cannot overflow.
# TOMBSTONE: MOJOLEARN_IDN_SEQ_TILED_PREDICT (whole-set predict chunk, opt-in arm, A/B never run) deleted 2026-11-15
# by lane/expdel2-24-neural with PREDICT_WHOLE_SET_BYTES and _predict_split_same; code recoverable at d19ca4f0c2.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_SEQ_TILED_PREDICT.patch
# RESTORED 2026-11-15 by lane/restore-run-arms-neural: RUN #4 of plan DELETED_ARMS_WORTH_RUNNING_2026-11-15 (git history)
# (sequential-predict, NVIDIA L40S A/B owed). The MOJOLEARN_IDN_ALL_OFF term is not restored (that define stays deleted).
# PROMOTED 2026-11-24 (lane ab2-nseq of ab-outcome-2): A/B nvc-ab-20261124T095727Z-92fd9f (NVIDIA L40S, wheel
# 4519097d380a, job nvc1-0002, base and arm back to back, one scored run per arm, the small-row median of 20 calls):
# neural/sequential-predict full 2.23 -> 1.87 ms (0.84, 1 cell); quality equal (accuracy 0.89892, auc 0.96895,
# log_loss 0.22510 both arms), output hash 8624513c57beb4fe and receipt 20ccc12ce430f7d3 on both arms. AMD not
# measured (NVIDIA-only campaign, AGENTS.md part 8). No bits change on any column: the chunk grows only when
# every Linear's split-K decision (gemm_split_shape, the one rule HostExec and DeviceExec share) is the same at
# the old and new chunk and tails, so every row's words are the caller-chunk words on the host and every vendor.
# The on-define is refused (core/experiment_guards_neural.mojo).
# OLD PATH SWITCH DELETED 2026-11-24 (lane ab-outcome-4): A/B nvc-ab-20261124T215509Z-b82da4 (NVIDIA L40S, wheel
# 71db45de5d, job nvc1-0007, one scored run per arm, the small-row median of 20 calls): `-D
# MOJOLEARN_IDN_SEQ_TILED_PREDICT_OFF` (the caller's chunk always) was slower: neural/sequential-predict full
# 2.29 -> 2.65 ms (1.16, 1 cell). The whole-set chunk is the only rule; the caller's chunk stays only where
# `predict_chunk_rows` finds a Linear's split-K decision could move (or the scratch does not fit). AMD not measured.
# TOMBSTONE: MOJOLEARN_IDN_SEQ_TILED_PREDICT_OFF (old path slower: NV L40S sequential-predict 2.29 -> 2.65 ms, 1.16) deleted 2026-11-24 by lane/ab-outcome-4; code recoverable at 652b59771f.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_IDN_SEQ_TILED_PREDICT_OFF.ab-outcome-4.patch
comptime PREDICT_WHOLE_SET_BYTES = 1 << 30

# ---------------------------------------------------------------- description
comptime SPEC_VERSION = 1
comptime SPEC_HEADER = 3     # version, input features, layer count
comptime SPEC_RECORD = 4     # ints per layer: code, arg0, arg1, reserved

comptime LAYER_LINEAR = 1        # arg0 output units, arg1 bias (0/1)
comptime LAYER_RELU = 2
comptime LAYER_SIGMOID = 3
comptime LAYER_TANH = 4
comptime LAYER_SOFTMAX = 5
comptime LAYER_LAYER_NORM = 6    # float: epsilon
comptime LAYER_DROPOUT = 7       # float: drop probability
comptime LAYER_RESIDUAL = 8      # arg0: -1 the model input, else an earlier layer's index
# Version 2 only (shape layers; per-sample shapes exclude the batch axis):
comptime LAYER_RESHAPE = 9       # args: the new per-sample dims (one may be -1), 0 pads
comptime LAYER_PERMUTE = 10      # args: a permutation of the per-sample axes, -1 pads
comptime LAYER_FLATTEN = 11      # the per-sample shape becomes one axis
comptime LAYER_CONV2D = 12       # (out_channels, kernel, stride, padding, bias 0/1, groups; 0 = 1); input (C, H, W)
comptime LAYER_MAXPOOL2D = 13    # (kernel, stride); input (C, H, W)
comptime LAYER_AVGPOOL2D = 14    # (kernel, stride); input (C, H, W)
# The 1-D sequence blocks (lane seq-blocks, 2026-11-22): input (C, L).
comptime LAYER_CONV1D = 15       # (out_channels, kernel, stride, padding, bias 0/1, groups; 0 = 1)
comptime LAYER_MAXPOOL1D = 16    # (kernel, stride, padding)
comptime LAYER_AVGPOOL1D = 17    # (kernel, stride, padding)
comptime LAYER_ADAPTIVE_AVGPOOL1D = 18  # (output_size): (C, L) -> (C, output_size), output_size divides L
# Branches (the containers' flat form): a layer reads an earlier output, not only the previous one.
comptime LAYER_BRANCH = 19       # (source): the output is source's output (-1 the model input)
comptime LAYER_CONCAT = 20       # (sources..., -2 pads): earlier outputs joined on the sample's first axis
comptime CONCAT_UNUSED = -2

# Version 2: [2, ndim, d0 .. d7 (unused 1), layer count, then per layer
# (code, arg0 .. arg5, 0)]; one float per layer as in version 1.
comptime SPEC2_VERSION = 2
comptime SPEC2_HEADER = 2 + RN_MAX_RANK + 1
comptime SPEC2_RECORD = 8
comptime SPEC2_ARGS = 6

comptime LOSS_MSE = 0
comptime LOSS_CROSS_ENTROPY = 1
comptime LOSS_BCE_WITH_LOGITS = 2
comptime LOSS_HUBER = 3
# lm_head (lane neural-blocks-3-lm): the mean next-token cross-entropy of a model ending in lm_head(vocab);
# y is the (N, tokens) int32 token ids (target code 0: the ids are the classes, no labels sort); the fit
# stops before the head and runs the fused head and loss (autograd/lm_head_v2_graph.mojo), no logits
comptime LOSS_TOKEN_CROSS_ENTROPY = 4
# nn-losses begin: elementwise losses of autograd/nn_losses.mojo and autograd/loss_ops.mojo, one
# Tape.loss_reduced node each. The loss word of the fit's integer words is code + 256 * reduction
# (REDUCTION_MEAN 0, REDUCTION_SUM 1, REDUCTION_BATCHMEAN 2); FitConfig decodes it once per fit.
comptime LOSS_L1 = 5             # torch.nn.L1Loss
comptime LOSS_SMOOTH_L1 = 6      # torch.nn.SmoothL1Loss(beta): the loss scalar is beta
comptime LOSS_NLL = 7            # torch.nn.NLLLoss on log-probabilities, class-code targets (one-hot on the executor)
comptime LOSS_KL_DIV = 8         # torch.nn.KLDivLoss(log_target=False) on log-probabilities, dense targets
comptime LOSS_WORD_REDUCTION = 256
# nn-losses end
# nn-losses-2 begin: torch.nn.CTCLoss on (T, C) log-probabilities per sample (autograd/nn_ctc.mojo); the loss scalar
# word is the blank index, or -1 - blank with zero_infinity (nn_ctc.ctc_blank_word); targets (N, S) labels plus the
# fit's target lengths, packed once per fit into (N, 2 + S) rows [input length, target length, labels]
comptime LOSS_CTC = 9
comptime LOSS_LAST = LOSS_CTC
# nn-losses-2 end

comptime TARGET_DENSE = 0        # Y is (N, out) float32
comptime TARGET_CODES = 1        # Y is N int32 class codes; one-hot built on the executor
comptime TARGET_CODE_VALUES = 2  # Y is N int32 codes; the (N, 1) target is the codes as floats

comptime SCHED_CONSTANT = 0
comptime SCHED_STEP = 1          # lr * gamma^(epoch // step)
comptime SCHED_EXPONENTIAL = 2   # lr * gamma^epoch

comptime OUT_RAW = 0
comptime OUT_SOFTMAX = 1
comptime OUT_SIGMOID = 2
comptime OUT_PROBA2 = 3          # one output: sigmoid p, written as the (N, 2) [1 - p, p]
comptime OUT_ARGMAX = 4          # the row argmax of the raw outputs, N float32 class codes
comptime OUT_THRESHOLD = 5       # one output: sigmoid p > 0.5, N float32 class codes (0/1)

# Row indices travel as float32 words (the shuffle and the gather), exact
# below 2^24; the parameter counter of the initialisation is an Int32 slot.
comptime MODEL_MAX_ROWS = 16777216
comptime MODEL_MAX_PARAMS = 2147483647


struct ModelSpec(Copyable, Movable):
    """A parsed model description: per layer its code, integer arguments,
    float argument, input and output widths and parameter offset (-1 when it
    has none). Parameters are one flat float32 vector: a Linear layer's
    weight (in, out) row-major, then its bias (out); a LayerNorm's weight
    (width), then its bias (width); layers in order."""
    var n_inputs: Int
    var kinds: List[Int]
    var arg0: List[Int]
    var arg1: List[Int]
    var farg: List[Float32]
    var in_width: List[Int]
    var out_width: List[Int]
    var param_off: List[Int]
    var n_params: Int
    #: 1 or 2 (SPEC2_VERSION: the per-sample shapes below are set)
    var version: Int
    #: version 2: the per-sample input shape, each layer's per-sample input
    #: and output shapes, and SPEC2_ARGS integer arguments per layer
    var sample: ShapeN
    var in_shape: List[ShapeN]
    var out_shape: List[ShapeN]
    var xargs: List[Int]

    def __init__(out self):
        self.n_inputs = 0
        self.kinds = List[Int]()
        self.arg0 = List[Int]()
        self.arg1 = List[Int]()
        self.farg = List[Float32]()
        self.in_width = List[Int]()
        self.out_width = List[Int]()
        self.param_off = List[Int]()
        self.n_params = 0
        self.version = SPEC_VERSION
        self.sample = ShapeN.matrix(1, 1)
        self.in_shape = List[ShapeN]()
        self.out_shape = List[ShapeN]()
        self.xargs = List[Int]()

    def n_layers(self) -> Int:
        return len(self.kinds)

    def out_features(self) -> Int:
        return self.out_width[len(self.out_width) - 1]

    def has_dropout(self) -> Bool:
        for i in range(len(self.kinds)):  # driver-loop: over layers
            if self.kinds[i] == LAYER_DROPOUT and self.farg[i] > Float32(0):
                return True
        return False


def _layer_name(i: Int) -> String:
    return "model: layer " + String(i)


def parse_model(prog: List[Int], ptab: List[Float32]) raises -> ModelSpec:
    """Validate a description and compute its widths and parameter offsets.

    prog = [version, input features, layer count, then per layer
    (code, arg0, arg1, 0)]; ptab = one float per layer (LayerNorm epsilon,
    Dropout probability, 0 otherwise). Raises with a message naming the
    first offending layer.
    """
    var s = ModelSpec()
    if len(prog) < SPEC_HEADER:
        raise Error("model: the description is empty")
    if prog[0] == SPEC2_VERSION:
        return parse_model_v2(prog, ptab)
    if prog[0] != SPEC_VERSION:
        raise Error("model: unknown description version " + String(prog[0]))
    var d = prog[1]
    var n = prog[2]
    if d < 1:
        raise Error("model: the input needs at least one feature")
    if n < 1:
        raise Error("model: add at least one layer")
    if len(prog) != SPEC_HEADER + SPEC_RECORD * n:
        raise Error("model: the description holds " + String(len(prog)) + " integers, "
                    + String(SPEC_HEADER + SPEC_RECORD * n) + " expected for " + String(n) + " layers")
    if len(ptab) != n:
        raise Error("model: the float table holds " + String(len(ptab)) + " values, one per layer ("
                    + String(n) + ") expected")
    s.n_inputs = d
    var width = d
    var off = 0
    var has_linear = False
    for i in range(n):  # driver-loop: over layers
        var base = SPEC_HEADER + SPEC_RECORD * i
        var k = prog[base]
        var a0 = prog[base + 1]
        var a1 = prog[base + 2]
        var f = ptab[i]
        var p_off = -1
        s.in_width.append(width)
        if k == LAYER_LINEAR:
            if a0 < 1 or a0 >= MODEL_MAX_ROWS:
                raise Error(_layer_name(i) + " (Linear): output units must be between 1 and 2^24 - 1")
            if a1 != 0 and a1 != 1:
                raise Error(_layer_name(i) + " (Linear): bias must be 0 or 1")
            p_off = off
            off += width * a0 + (a0 if a1 == 1 else 0)
            width = a0
            has_linear = True
        elif k == LAYER_RELU or k == LAYER_SIGMOID or k == LAYER_TANH or k == LAYER_SOFTMAX:
            pass
        elif k == LAYER_LAYER_NORM:
            if not (f > Float32(0)):
                raise Error(_layer_name(i) + " (LayerNorm): epsilon must be positive")
            p_off = off
            off += 2 * width
        elif k == LAYER_DROPOUT:
            if not (f >= Float32(0) and f < Float32(1)):
                raise Error(_layer_name(i) + " (Dropout): the drop probability must be in [0, 1)")
        elif k == LAYER_RESIDUAL:
            if a0 < -1 or a0 >= i:
                raise Error(_layer_name(i) + " (Residual): the source must be -1 (the model input) "
                            + "or the index of an earlier layer")
            var w_from = d if a0 == -1 else s.out_width[a0]
            if w_from != width:
                raise Error(_layer_name(i) + " (Residual): adds a width-" + String(w_from)
                            + " tensor to a width-" + String(width) + " tensor")
        else:
            raise Error(_layer_name(i) + ": unknown layer code " + String(k))
        if off > MODEL_MAX_PARAMS:
            raise Error("model: more than 2^31 - 1 parameters")
        s.kinds.append(k)
        s.arg0.append(a0)
        s.arg1.append(a1)
        s.farg.append(f)
        s.out_width.append(width)
        s.param_off.append(p_off)
    if not has_linear:
        raise Error("model: add at least one Linear layer")
    s.n_params = off
    return s^


def _arg(prog: List[Int], base: Int, k: Int) -> Int:
    return prog[base + 1 + k]


def conv_groups(spec: ModelSpec, i: Int) -> Int:
    """Layer i's (a Conv2d) channel groups: its sixth argument, 0 (the
    encoding before groups) read as 1."""
    return max(1, spec.xargs[SPEC2_ARGS * i + 5])


def conv_in_per_group(spec: ModelSpec, i: Int) -> Int:
    """Layer i's (a Conv2d) input channels per group: the weight's
    second dimension."""
    return spec.in_shape[i].d[0] // conv_groups(spec, i)


def parse_model_v2(prog: List[Int], ptab: List[Float32]) raises -> ModelSpec:
    """Validate a version-2 description (SPEC2_*): the per-sample input
    shape, then the layers with their per-sample shapes, widths (the last
    axis) and parameter offsets. A Linear layer acts on the last axis; the
    model's output must be one vector per sample."""
    var s = ModelSpec()
    s.version = SPEC2_VERSION
    if len(prog) < SPEC2_HEADER:
        raise Error("model: the description is empty")
    var nd = prog[1]
    if nd < 1 or nd > RN_MAX_RANK - 1:
        raise Error("model: the input shape takes 1 to " + String(RN_MAX_RANK - 1) + " dimensions")
    var idims = List[Int](capacity=nd)
    for k in range(nd):  # small-loop(nd: the input rank, at most 7): description parsing, no data
        idims.append(prog[2 + k])
    var cur = rn_shape(idims, "model: input shape")
    var n = prog[SPEC2_HEADER - 1]
    if n < 1:
        raise Error("model: add at least one layer")
    if len(prog) != SPEC2_HEADER + SPEC2_RECORD * n:
        raise Error("model: the description holds " + String(len(prog)) + " integers, "
                    + String(SPEC2_HEADER + SPEC2_RECORD * n) + " expected for " + String(n) + " layers")
    if len(ptab) != n:
        raise Error("model: the float table holds " + String(len(ptab)) + " values, one per layer ("
                    + String(n) + ") expected")
    s.sample = cur
    s.n_inputs = cur.count()
    if s.n_inputs >= MODEL_MAX_ROWS:
        raise Error("model: the input holds more than 2^24 - 1 values per sample")
    var off = 0
    var has_linear = False
    for i in range(n):  # driver-loop: over layers
        var base = SPEC2_HEADER + SPEC2_RECORD * i
        var k = prog[base]
        var a0 = _arg(prog, base, 0)
        var a1 = _arg(prog, base, 1)
        var f = ptab[i]
        var p_off = -1
        var width = cur.cols()
        s.in_width.append(width)
        s.in_shape.append(cur)
        var nxt = cur
        if k == LAYER_LINEAR:
            if a0 < 1 or a0 >= MODEL_MAX_ROWS:
                raise Error(_layer_name(i) + " (Linear): output units must be between 1 and 2^24 - 1")
            if a1 != 0 and a1 != 1:
                raise Error(_layer_name(i) + " (Linear): bias must be 0 or 1")
            p_off = off
            off += width * a0 + (a0 if a1 == 1 else 0)
            var ld = rn_dims(cur)
            ld[cur.ndim - 1] = a0
            nxt = rn_shape(ld, _layer_name(i) + " (Linear)")
            has_linear = True
        elif k == LAYER_RELU or k == LAYER_SIGMOID or k == LAYER_TANH or k == LAYER_SOFTMAX:
            pass
        elif k == LAYER_LAYER_NORM:
            if not (f > Float32(0)):
                raise Error(_layer_name(i) + " (LayerNorm): epsilon must be positive")
            p_off = off
            off += 2 * width
        elif k == LAYER_DROPOUT:
            if not (f >= Float32(0) and f < Float32(1)):
                raise Error(_layer_name(i) + " (Dropout): the drop probability must be in [0, 1)")
        elif k == LAYER_RESIDUAL:
            if a0 < -1 or a0 >= i:
                raise Error(_layer_name(i) + " (Residual): the source must be -1 (the model input) "
                            + "or the index of an earlier layer")
            var from_shape = s.sample if a0 == -1 else s.out_shape[a0]
            if not from_shape.same(cur):
                raise Error(_layer_name(i) + " (Residual): adds tensors of different shapes")
        elif k == LAYER_RESHAPE:
            var rd = List[Int]()
            for j in range(SPEC2_ARGS):  # small-loop(SPEC2_ARGS: six layer arguments): description parsing, no data
                var v = _arg(prog, base, j)
                if v == 0:
                    break
                rd.append(v)
            if len(rd) < 1:
                raise Error(_layer_name(i) + " (Reshape): give the new shape")
            nxt = rn_resolve(cur, rd, _layer_name(i) + " (Reshape)")
        elif k == LAYER_PERMUTE:
            var seen = List[Bool](length=cur.ndim, fill=False)
            var pd = List[Int](capacity=cur.ndim)
            for j in range(SPEC2_ARGS):  # small-loop(SPEC2_ARGS: six layer arguments): description parsing, no data
                var v = _arg(prog, base, j)
                if j < cur.ndim:
                    if v < 0 or v >= cur.ndim or seen[v]:
                        raise Error(_layer_name(i) + " (Permute): the axes must list each of the "
                                    + String(cur.ndim) + " sample axes once")
                    seen[v] = True
                    pd.append(cur.d[v])
                elif v != -1:
                    raise Error(_layer_name(i) + " (Permute): the axes must list each of the "
                                + String(cur.ndim) + " sample axes once")
            nxt = rn_shape(pd, _layer_name(i) + " (Permute)")
        elif k == LAYER_FLATTEN:
            var fd = List[Int]()
            fd.append(cur.count())
            nxt = rn_shape(fd, _layer_name(i) + " (Flatten)")
        elif k == LAYER_CONV2D or k == LAYER_MAXPOOL2D or k == LAYER_AVGPOOL2D:
            if cur.ndim != 3:
                raise Error(_layer_name(i) + ": convolution and pooling take a (channels, height, width) input")
            var c = cur.d[0]
            var h = cur.d[1]
            var w = cur.d[2]
            var ch: Int
            var ks: Int
            var st: Int
            var pad: Int
            if k == LAYER_CONV2D:
                ch = a0
                ks = a1
                st = _arg(prog, base, 2)
                pad = _arg(prog, base, 3)
                var bias = _arg(prog, base, 4)
                var groups = max(1, _arg(prog, base, 5))
                if ch < 1 or ks < 1 or st < 1 or pad < 0 or (bias != 0 and bias != 1):
                    raise Error(_layer_name(i) + " (Conv2d): out_channels, kernel_size and stride must be "
                                + "positive, padding >= 0 and bias 0 or 1")
                if c % groups != 0 or ch % groups != 0:
                    raise Error(_layer_name(i) + " (Conv2d): groups must divide the input channels and "
                                + "out_channels")
                p_off = off
                off += ch * (c // groups) * ks * ks + (ch if bias == 1 else 0)
            else:
                ch = c
                ks = a0
                st = a1
                pad = 0
                if ks < 1 or st < 1:
                    raise Error(_layer_name(i) + " (pooling): kernel_size and stride must be positive")
            var oh = (h + 2 * pad - ks) // st + 1
            var ow = (w + 2 * pad - ks) // st + 1
            if h + 2 * pad < ks or w + 2 * pad < ks or oh < 1 or ow < 1:
                raise Error(_layer_name(i) + ": the kernel is larger than the (padded) input")
            var cd = List[Int]()
            cd.append(ch)
            cd.append(oh)
            cd.append(ow)
            nxt = rn_shape(cd, _layer_name(i))
        elif k == LAYER_CONV1D or k == LAYER_MAXPOOL1D or k == LAYER_AVGPOOL1D or k == LAYER_ADAPTIVE_AVGPOOL1D:
            if cur.ndim != 2:
                raise Error(_layer_name(i) + ": 1-D convolution and pooling take a (channels, length) input")
            var c = cur.d[0]
            var length = cur.d[1]
            var ch = c
            var ol: Int
            if k == LAYER_ADAPTIVE_AVGPOOL1D:
                if a0 < 1 or length % a0 != 0:
                    raise Error(_layer_name(i) + " (AdaptiveAvgPool1d): output_size must be positive and divide "
                                + "the length " + String(length))
                ol = a0
            else:
                var ks: Int
                var st: Int
                var pad = _arg(prog, base, 2)
                if k == LAYER_CONV1D:
                    ch = a0
                    ks = a1
                    st = pad
                    pad = _arg(prog, base, 3)
                    var bias = _arg(prog, base, 4)
                    var groups = max(1, _arg(prog, base, 5))
                    if ch < 1 or ks < 1 or st < 1 or pad < 0 or (bias != 0 and bias != 1):
                        raise Error(_layer_name(i) + " (Conv1d): out_channels, kernel_size and stride must be "
                                    + "positive, padding >= 0 and bias 0 or 1")
                    if c % groups != 0 or ch % groups != 0:
                        raise Error(_layer_name(i) + " (Conv1d): groups must divide the input channels and "
                                    + "out_channels")
                    p_off = off
                    off += ch * (c // groups) * ks + (ch if bias == 1 else 0)
                else:
                    ks = a0
                    st = a1
                    if ks < 1 or st < 1 or pad < 0 or 2 * pad > ks:
                        raise Error(_layer_name(i) + " (pooling): kernel_size and stride must be positive and "
                                    + "padding between 0 and kernel_size / 2")
                if length + 2 * pad < ks:
                    raise Error(_layer_name(i) + ": the kernel is longer than the (padded) input")
                ol = (length + 2 * pad - ks) // st + 1
            var cd = List[Int]()
            cd.append(ch)
            cd.append(ol)
            nxt = rn_shape(cd, _layer_name(i))
        elif k == LAYER_BRANCH:
            if a0 < -1 or a0 >= i:
                raise Error(_layer_name(i) + " (Branch): the source must be -1 (the model input) "
                            + "or the index of an earlier layer")
            nxt = s.sample if a0 == -1 else s.out_shape[a0]
        elif k == LAYER_CONCAT:
            # lane neural-blocks-2: the float argument is the sample axis to join on (0, the channels, by
            # default; the last axis joins the features of (length, features) samples, e.g. Mamba's
            # [u | delta | B | C])
            var cax = Int(f)
            var count = 0
            var total = 0
            var first = cur
            for j in range(SPEC2_ARGS):  # small-loop(SPEC2_ARGS: six layer arguments): description parsing, no data
                var v = _arg(prog, base, j)
                if v == CONCAT_UNUSED:
                    continue
                if v < -1 or v >= i:
                    raise Error(_layer_name(i) + " (Concat): each source must be -1 (the model input) "
                                + "or the index of an earlier layer")
                var sj = s.sample if v == -1 else s.out_shape[v]
                if count == 0:
                    first = sj
                elif sj.ndim != first.ndim:
                    raise Error(_layer_name(i) + " (Concat): the sources have different ranks")
                else:
                    for q in range(sj.ndim):  # small-loop(ndim: the sample rank, at most 7): shape metadata, no data
                        if q != cax and sj.d[q] != first.d[q]:
                            raise Error(_layer_name(i) + " (Concat): the sources differ outside the joined axis")
                if cax < 0 or cax >= sj.ndim:
                    raise Error(_layer_name(i) + " (Concat): axis must be a sample axis")
                total += sj.d[cax]
                count += 1
            if count < 1:
                raise Error(_layer_name(i) + " (Concat): give at least one source")
            var jd = rn_dims(first)
            jd[cax] = total
            nxt = rn_shape(jd, _layer_name(i) + " (Concat)")
        elif is_block(k):
            # lane neural-blocks: autograd/model_blocks.mojo validates the block and sizes its parameters
            var la = List[Int](capacity=SPEC2_ARGS)
            for j in range(SPEC2_ARGS):  # small-loop(SPEC2_ARGS: six layer arguments): description parsing, no data
                la.append(_arg(prog, base, j))
            if k == LAYER_MULTIPLY:
                if a0 < -1 or a0 >= i:
                    raise Error(_layer_name(i) + " (Multiply): the source must be -1 (the model input) "
                                + "or the index of an earlier layer")
                var src_shape = s.sample if a0 == -1 else s.out_shape[a0]
                if not src_shape.same(cur):
                    raise Error(_layer_name(i) + " (Multiply): multiplies tensors of different shapes")
            # nn-attention begin
            if nn_attention_is(k):
                var nsrc = nn_attention_source(k, la, 0)
                var nshape = cur
                if nsrc == -1:
                    nshape = s.sample
                elif nsrc >= 0 and nsrc < i:
                    nshape = s.out_shape[nsrc]
                nn_attention_check_source(k, la, 0, i, cur, s.sample, nshape, _layer_name(i))
            # nn-attention end
            nxt = block_shape(k, la, 0, f, cur, _layer_name(i))
            # lm_head (lane neural-blocks-3-lm): the LM head is the model's projection layer
            if k == LAYER_LM_HEAD:
                has_linear = True
            # nn-conv begin
            # a transposed convolution is a learned projection (a convolutional autoencoder's output layer)
            if nnc_is_projection(k):
                has_linear = True
            # nn-conv end
            var sizes = block_param_sizes(k, la, 0, cur)
            if len(sizes) > 0:
                p_off = off
                for j in range(len(sizes)):  # small-loop(at most four parameter tensors): description parsing
                    off += sizes[j]
        else:
            raise Error(_layer_name(i) + ": unknown layer code " + String(k))
        if nxt.ndim > RN_MAX_RANK - 1:
            raise Error(_layer_name(i) + ": a sample has at most " + String(RN_MAX_RANK - 1) + " dimensions")
        if off > MODEL_MAX_PARAMS:
            raise Error("model: more than 2^31 - 1 parameters")
        s.kinds.append(k)
        s.arg0.append(a0)
        s.arg1.append(a1)
        for j in range(SPEC2_ARGS):  # small-loop(SPEC2_ARGS: six layer arguments): description parsing, no data
            s.xargs.append(_arg(prog, base, j))
        s.farg.append(f)
        s.out_shape.append(nxt)
        s.out_width.append(nxt.cols())
        s.param_off.append(p_off)
        cur = nxt
    if not has_linear:
        raise Error("model: add at least one Linear layer")
    # lm_head (lane neural-blocks-3-lm): a final lm_head gives every position's logits, (length, vocab) per
    # sample; the outputs are that sample flat (n_samples, length * vocab), in the layer's row-major order
    var head_last = s.kinds[len(s.kinds) - 1] == LAYER_LM_HEAD
    # nn-losses-2 begin: a final log_softmax may give a sequence of log-probabilities per sample, (length, classes)
    # for the ctc loss, flat like lm_head's; check_fit refuses every other loss on such an output
    if is_nn_log_softmax(s.kinds[len(s.kinds) - 1]):
        head_last = True
    # nn-losses-2 end
    if head_last:
        s.out_width[len(s.out_width) - 1] = cur.count()
    elif cur.ndim != 1:
        raise Error("model: the output must be one vector per sample; add flatten() before the last "
                    + "linear layer")
    s.n_params = off
    return s^


struct FitConfig(Copyable, Movable):
    """Training options of one fit (see `parse_fit` for the encoding)."""
    var loss: Int
    var target: Int
    var opt: OptConfig
    var lr: Float32
    var sched: Int
    var sched_step: Int
    var sched_gamma: Float32
    var batch: Int
    var epochs: Int
    var shuffle: Bool
    var seed: UInt64
    var patience: Int
    var tol: Float64
    # the loss's scalar: the Huber delta, the smooth L1 beta or the bce_with_logits pos_weight (lane nn-losses)
    var huber_delta: Float32
    var init: Bool
    # nn-losses: REDUCTION_MEAN, REDUCTION_SUM or REDUCTION_BATCHMEAN (the loss word's high part)
    var reduction: Int
    # lane nn-optim: the schedule's NN_SCHED_ARGS words (autograd/nn_optim.mojo; empty for codes 0..2) and the
    # parameter tensors' offsets (LAMB's per-tensor trust ratios; empty: one tensor)
    var sched_args: List[Float64]
    var tensor_offs: List[Int]
    # nn-blocks-pgseq: the parameter groups (autograd/nn_pgseq.mojo; empty: none, the single-launch step)
    var af_tensors: List[Tuple[Int, Int, Int, Int]]  # parameter offset, rows, columns (0 for vector), group
    var groups: PgSeqTable

    def __init__(
        out self, loss: Int, target: Int, opt: OptConfig, lr: Float32, sched: Int, sched_step: Int,
        sched_gamma: Float32, batch: Int, epochs: Int, shuffle: Bool, seed: UInt64, patience: Int, tol: Float64,
        huber_delta: Float32, init: Bool, var sched_args: List[Float64], var tensor_offs: List[Int],
    ):
        """`loss` is the loss word: the loss code + LOSS_WORD_REDUCTION * the reduction (a plain code is the
        mean). `sched_args` and `tensor_offs` may be empty (lane nn-optim)."""
        self.loss = loss % LOSS_WORD_REDUCTION if loss >= 0 else loss
        self.reduction = loss // LOSS_WORD_REDUCTION if loss >= 0 else REDUCTION_MEAN
        self.target = target
        self.opt = opt
        self.lr = lr
        self.sched = sched
        self.sched_step = sched_step
        self.sched_gamma = sched_gamma
        self.batch = batch
        self.epochs = epochs
        self.shuffle = shuffle
        self.seed = seed
        self.patience = patience
        self.tol = tol
        self.huber_delta = huber_delta
        self.init = init
        self.sched_args = sched_args^
        self.tensor_offs = tensor_offs^
        self.af_tensors = List[Tuple[Int, Int, Int, Int]]()
        self.groups = PgSeqTable()


struct FitValidation(Copyable, Movable):
    var x: CastIn
    var y: CastIn
    var classes: CastIn
    var curve: FP
    var restore: Bool
    # Native result: executed epochs in this call, retained absolute epoch.
    var result: MutPointer[Int64, MutUntrackedOrigin]
    # Lane rnn-parity-c: the validation rows' float32 sequence lengths on the host (validation_data=(X, y, lengths)),
    # read by the recurrent layers as the training lengths are; required exactly when the fit has lengths.
    var lengths: Optional[FP]

    def __init__(out self, x: CastIn, y: CastIn, classes: CastIn, curve: FP, restore: Bool,
                 result: MutPointer[Int64, MutUntrackedOrigin], lengths: Optional[FP]):
        self.x = x
        self.y = y
        self.classes = classes
        self.curve = curve
        self.restore = restore
        self.result = result
        self.lengths = lengths


def check_fit(spec: ModelSpec, cfg: FitConfig, N: Int) raises:
    """User-facing refusals of a fit's options against its model."""
    var O = spec.out_features()
    var last = spec.kinds[spec.n_layers() - 1]
    if N < 1 or N >= MODEL_MAX_ROWS:
        raise Error("model fit: the number of samples must be between 1 and 2^24 - 1")
    if cfg.loss < LOSS_MSE or cfg.loss > LOSS_LAST:
        raise Error("model fit: unknown loss code " + String(cfg.loss))
    # nn-losses begin
    if cfg.reduction != REDUCTION_MEAN and cfg.reduction != REDUCTION_SUM and cfg.reduction != REDUCTION_BATCHMEAN:
        raise Error("model fit: reduction must be 'mean', 'sum' or 'batchmean'")
    if cfg.reduction == REDUCTION_BATCHMEAN and cfg.loss != LOSS_KL_DIV:
        raise Error("model fit: reduction='batchmean' is the kl_div loss's; use 'mean' or 'sum'")
    if cfg.reduction == REDUCTION_SUM and cfg.loss == LOSS_TOKEN_CROSS_ENTROPY:
        raise Error("model fit: token_cross_entropy takes reduction='mean'")
    if cfg.loss == LOSS_BCE_WITH_LOGITS and not (cfg.huber_delta > Float32(0)):
        raise Error("model fit: pos_weight must be positive")
    if cfg.loss == LOSS_SMOOTH_L1 and not (cfg.huber_delta >= Float32(0)):
        raise Error("model fit: the smooth_l1 beta must be zero or positive")
    if cfg.loss == LOSS_NLL and O < 2:
        raise Error("model fit: nll needs at least two outputs (one log-probability a class)")
    if (cfg.loss == LOSS_NLL or cfg.loss == LOSS_KL_DIV) and (last == LAYER_SOFTMAX or last == LAYER_SIGMOID):
        raise Error("model fit: nll and kl_div take log-probabilities; a final Softmax or Sigmoid gives "
                    + "probabilities")
    # nn-losses end
    # nn-losses-2 begin
    if cfg.loss == LOSS_CTC:
        _ = model_ctc_dims(spec)
        if cfg.target != TARGET_DENSE:
            raise Error("model fit: ctc takes the label sequences as they are (no class labels)")
        if cfg.reduction == REDUCTION_BATCHMEAN:
            raise Error("model fit: ctc takes reduction='mean' or 'sum'")
    elif model_sequence_output(spec):
        raise Error("model fit: the model's output is a sequence of log-probabilities per sample (a final "
                    + "log_softmax over (length, classes)); that output fits with loss='ctc'. For another loss, "
                    + "flatten() before the last linear layer")
    # nn-losses-2 end
    # lm_head (lane neural-blocks-3-lm)
    if cfg.loss == LOSS_TOKEN_CROSS_ENTROPY:
        if spec.version != SPEC2_VERSION or last != LAYER_LM_HEAD:
            raise Error("model fit: token_cross_entropy needs a model ending in an lm_head layer")
        if cfg.target != TARGET_DENSE:
            raise Error("model fit: token_cross_entropy takes the token ids as they are (no class labels)")
    # end lm_head
    if cfg.target < TARGET_DENSE or cfg.target > TARGET_CODE_VALUES:
        raise Error("model fit: unknown target code " + String(cfg.target))
    if cfg.target == TARGET_CODES and cfg.loss != LOSS_CROSS_ENTROPY and cfg.loss != LOSS_NLL:
        raise Error("model fit: class-code targets need the cross_entropy or nll loss")
    if cfg.target == TARGET_CODE_VALUES and O != 1:
        raise Error("model fit: code-value targets need a model with one output")
    if cfg.loss == LOSS_CROSS_ENTROPY and O < 2:
        raise Error("model fit: cross_entropy needs at least two outputs")
    if (cfg.loss == LOSS_CROSS_ENTROPY or cfg.loss == LOSS_BCE_WITH_LOGITS) and (
        last == LAYER_SOFTMAX or last == LAYER_SIGMOID
    ):
        raise Error("model fit: cross_entropy and bce_with_logits take logits; remove the final "
                    + "Softmax or Sigmoid layer")
    if cfg.loss == LOSS_HUBER and not (cfg.huber_delta > Float32(0)):
        raise Error("model fit: the Huber delta must be positive")
    # lane nn-optim: every optimizer code and this lane's schedule codes, one admission body with compile's
    nn_optim_admit(cfg.opt.kind, cfg.opt.flags, Float64(cfg.opt.f1), Float64(cfg.opt.f2), Float64(cfg.opt.eps),
                   Float64(cfg.opt.wd), Float64(cfg.opt.f7), cfg.sched, cfg.sched_args)
    if not (cfg.lr > Float32(0)):
        raise Error("model fit: the learning rate must be positive")
    if cfg.batch < 1 or cfg.epochs < 1:
        raise Error("model fit: batch size and epochs must be at least 1")
    if cfg.sched < SCHED_CONSTANT or cfg.sched > NN_SCHED_PLATEAU:
        raise Error("model fit: unknown learning-rate schedule code " + String(cfg.sched))
    if (cfg.sched == SCHED_STEP or cfg.sched == SCHED_EXPONENTIAL or cfg.sched == NN_SCHED_MULTI_STEP) and not (
        cfg.sched_gamma > Float32(0)
    ):
        raise Error("model fit: the schedule's gamma must be positive")
    if cfg.sched == SCHED_STEP and cfg.sched_step < 1:
        raise Error("model fit: the step schedule's step size must be at least 1 epoch")
    if cfg.patience < 0:
        raise Error("model fit: n_iter_no_change must be >= 0 (0 trains every epoch)")
    comptime if not HAS_DROPOUT:
        if spec.has_dropout():
            raise Error("model fit: Dropout is not available in this build")


# ---------------------------------------------------------------- helpers
def _copy[E: Exec](mut ex: E, dst: FP, src: FP, n: Int) raises:
    var a = Args()
    a.p0 = src
    a.p1 = dst
    ex.launch[OP_COPY](a, n)


def _fill[E: Exec](mut ex: E, dst: FP, n: Int, value: Float32) raises:
    var a = Args()
    a.p0 = dst
    a.f0 = value
    ex.launch[OP_FILL](a, n)


def _init_uniform[E: Exec](mut ex: E, dst: FP, n: Int, first: Int, seed: UInt64, bound: Float32) raises:
    var a = Args()
    a.p0 = dst
    a.i2 = first
    a.i3 = Int(seed & UInt64(0xFFFF))
    a.i4 = Int((seed >> 16) & UInt64(0xFFFF))
    a.i5 = Int((seed >> 32) & UInt64(0xFFFF))
    a.i6 = Int((seed >> 48) & UInt64(0xFFFF))
    a.f0 = bound
    ex.launch[OP_AGM_INIT_UNIFORM](a, n)


def model_init_params[E: Exec](mut ex: E, spec: ModelSpec, P: FP, seed: UInt64) raises:
    """Initialise every parameter on the executor from the seed: a Linear
    layer's weight and bias uniform in [-b, b], b = sqrt(6 / (in + out))
    (Glorot uniform, scikit-learn's MLP bound); LayerNorm weight 1, bias 0."""
    for i in range(spec.n_layers()):  # driver-loop: over layers
        var k = spec.kinds[i]
        var off = spec.param_off[i]
        if k == LAYER_LINEAR:
            var fi = spec.in_width[i]
            var fo = spec.arg0[i]
            var bound = ftz(identical_sqrt(ftz(identical_div(Float32(6), Float32(fi + fo)))))
            var n = fi * fo + (fo if spec.arg1[i] == 1 else 0)
            _init_uniform(ex, P + off, n, off, seed, bound)
        elif k == LAYER_LAYER_NORM:
            var w = spec.in_width[i]
            _fill(ex, P + off, w, Float32(1))
            _fill(ex, P + off + w, w, Float32(0))
        elif k == LAYER_CONV2D:
            # Glorot uniform over the receptive field: fan_in = C kh kw,
            # fan_out = out_channels kh kw; the bias from the same draw, as Linear
            var oc = spec.arg0[i]
            var ks = spec.arg1[i]
            # (with groups: fan_in = C / groups kh kw, the weight's second dimension)
            var c = conv_in_per_group(spec, i)
            var fan = c * ks * ks + oc * ks * ks
            var bound = ftz(identical_sqrt(ftz(identical_div(Float32(6), Float32(fan)))))
            var n = oc * c * ks * ks + (oc if spec.xargs[SPEC2_ARGS * i + 4] == 1 else 0)
            _init_uniform(ex, P + off, n, off, seed, bound)
        elif k == LAYER_CONV1D:
            # conv2d's rule over a length-ks receptive field
            var oc = spec.arg0[i]
            var ks = spec.arg1[i]
            var c = conv_in_per_group(spec, i)
            var fan = c * ks + oc * ks
            var bound = ftz(identical_sqrt(ftz(identical_div(Float32(6), Float32(fan)))))
            var n = oc * c * ks + (oc if spec.xargs[SPEC2_ARGS * i + 4] == 1 else 0)
            _init_uniform(ex, P + off, n, off, seed, bound)
        elif is_block(k) and off >= 0:
            # lane neural-blocks: per parameter tensor a seeded uniform bound, or a constant fill (-v: v - 1)
            var sizes = block_param_sizes(k, spec.xargs, i, spec.in_shape[i])
            var bounds = block_init_bounds(k, spec.xargs, i, spec.in_shape[i])
            var o = off
            for j in range(len(sizes)):  # small-loop(at most four parameter tensors of the layer)
                if bounds[j] == BLOCK_INIT_S4D:
                    # Mamba's S4D-real A_log: log(1) .. log(state) along each row
                    var sa = Args()
                    sa.p0 = P + o
                    sa.i0 = spec.xargs[SPEC2_ARGS * i]
                    ex.launch[OP_SSM_S4D_INIT](sa, sizes[j])
                elif bounds[j] < Float32(0):
                    _fill(ex, P + o, sizes[j], -bounds[j] - Float32(1))
                else:
                    _init_uniform(ex, P + o, sizes[j], o, seed, bounds[j])
                o += sizes[j]


def _param[E: Exec](
    mut tape: Tape, mut ex: E, P: FP, Gr: FP, off: Int, rows: Int, cols: Int, train: Bool,
) raises -> Tensor:
    # Training: the gradient lands straight in its slot of Gr (variable_into).
    if train:
        return tape.variable_into(ex, P + off, Gr + off, rows, cols)
    return tape.variable(ex, P + off, rows, cols, False)


def model_forward[E: Exec](
    mut tape: Tape, mut ex: E, spec: ModelSpec, P: FP, Gr: FP, x: Tensor, train: Bool,
    seed: UInt64, step: Int,
) raises -> Tensor:
    """Record the model's forward on the tape from input x (B, n_inputs).
    Dropout acts only when `train`; `step` (the optimizer step) and the
    layer index pick its masks."""
    # Standalone callers may change their ModelSpec between calls. Only the
    # fit/predict loops below retain this topology across forwards.
    var topology = _ModelForwardTopology(spec)
    return _model_forward_prepared(tape, ex, spec, P, Gr, x, train, seed, step, topology)


struct _ModelForwardTopology(Movable):
    """Fusion eligibility for one operation's immutable parsed description.

    No parameters, shapes, execution options or borrowed addresses are cached.
    Keep this owner private to the call that owns the matching ModelSpec.
    """
    var activations: List[Int]
    # lm_head (lane neural-blocks-3-lm): the layers the forward runs (all; the token cross-entropy's fit
    # stops before its final lm_head layer)
    var stop: Int

    def __init__(out self, spec: ModelSpec):
        self.stop = spec.n_layers()
        self.activations = List[Int]()
        self.activations = List[Int](length=spec.n_layers(), fill=0)
        for i in range(spec.n_layers()):  # driver-loop: operation setup over model topology
            if spec.kinds[i] == LAYER_LINEAR:
                self.activations[i] = _absorbed_activation(spec, i)


def _model_forward_prepared[E: Exec](
    mut tape: Tape, mut ex: E, spec: ModelSpec, P: FP, Gr: FP, x: Tensor, train: Bool,
    seed: UInt64, step: Int, topology: _ModelForwardTopology, lengths: Optional[FP] = None,
) raises -> Tensor:
    if spec.version == SPEC2_VERSION:
        return _model_forward_v2(tape, ex, spec, P, Gr, x, train, seed, step, topology, lengths)
    var outs = List[Tensor]()
    var h = x
    var i = 0
    while i < spec.n_layers():  # driver-loop: over layers
        var k = spec.kinds[i]
        if k == LAYER_LINEAR:
            var fi = spec.in_width[i]
            var fo = spec.arg0[i]
            var off = spec.param_off[i]
            var w = _param(tape, ex, P, Gr, off, fi, fo, train)
            var bias: Optional[Tensor] = None
            if spec.arg1[i] == 1:
                bias = Optional[Tensor](_param(tape, ex, P, Gr, off + fi * fo, 1, fo, train))
            var act = topology.activations[i]
            h = tape.linear(ex, h, w, bias, act)
            if act != 0:
                # the activation layer i + 1 rode in this node; layer i's
                # own (pre-activation) output is referenced by no layer
                outs.append(h)
                i += 1
        elif k == LAYER_RELU:
            h = tape.relu(ex, h)
        elif k == LAYER_SIGMOID:
            h = tape.sigmoid(ex, h)
        elif k == LAYER_TANH:
            h = tape.tanh(ex, h)
        elif k == LAYER_SOFTMAX:
            h = tape.softmax(ex, h)
        elif k == LAYER_LAYER_NORM:
            var d = spec.in_width[i]
            var off = spec.param_off[i]
            var weight = _param(tape, ex, P, Gr, off, 1, d, train)
            var bias = _param(tape, ex, P, Gr, off + d, 1, d, train)
            h = layer_norm(tape, ex, h, weight, bias, spec.farg[i])
        elif k == LAYER_DROPOUT:
            if train and spec.farg[i] > Float32(0):
                comptime if HAS_DROPOUT:
                    # depends on lane ag-data (Tape.dropout): one mask per
                    # (seed, step, layer), the same on every vendor
                    h = tape.dropout(ex, h, spec.farg[i], seed,
                                     step * spec.n_layers() + i)
                else:
                    raise Error("model fit: Dropout is not available in this build")
        elif k == LAYER_RESIDUAL:
            var src = x if spec.arg0[i] == -1 else outs[spec.arg0[i]]
            h = tape.add(ex, h, src)
        outs.append(h)
        i += 1
    return h


def _model_forward_v2[E: Exec](
    mut tape: Tape, mut ex: E, spec: ModelSpec, P: FP, Gr: FP, x: Tensor, train: Bool,
    seed: UInt64, step: Int, topology: _ModelForwardTopology, lengths: Optional[FP] = None,
) raises -> Tensor:
    """model_forward of a version-2 description: x is the (B, n_inputs)
    batch, viewed as (B,) + the per-sample shape (free). Each layer's output
    is held in its recorded shape (a free view when the op records only the
    matrix view). Linear, the activations, LayerNorm and Dropout act on the
    last axis through the matrix view (their 2-D words); reshape and
    flatten are views, permute a strided view, conv2d and the pools the
    conv lane's rank-4 ops (autograd/rankn_conv.mojo)."""
    var B = tape.rows(x)
    var x0 = rn_as_shape(tape, ex, x, rn_with_batch(B, spec.sample, "model"))
    var outs = List[Tensor]()
    var h = x0
    var i = 0
    while i < topology.stop:  # driver-loop: over layers
        var k = spec.kinds[i]
        var xa = SPEC2_ARGS * i
        if k == LAYER_LINEAR:
            var fi = spec.in_width[i]
            var fo = spec.arg0[i]
            var off = spec.param_off[i]
            var w = _param(tape, ex, P, Gr, off, fi, fo, train)
            var bias: Optional[Tensor] = None
            if spec.arg1[i] == 1:
                bias = Optional[Tensor](_param(tape, ex, P, Gr, off + fi * fo, 1, fo, train))
            var act = topology.activations[i]
            h = tape.linear(ex, h, w, bias, act)
            h = rn_as_shape(tape, ex, h, rn_with_batch(B, spec.out_shape[i], "model"))
            if act != 0:
                outs.append(h)
                i += 1
        elif k == LAYER_RELU:
            h = tape.relu(ex, h)
        elif k == LAYER_SIGMOID:
            h = tape.sigmoid(ex, h)
        elif k == LAYER_TANH:
            h = tape.tanh(ex, h)
        elif k == LAYER_SOFTMAX:
            h = tape.softmax(ex, h)
        elif k == LAYER_LAYER_NORM:
            var d = spec.in_width[i]
            var off = spec.param_off[i]
            var weight = _param(tape, ex, P, Gr, off, 1, d, train)
            var lbias = _param(tape, ex, P, Gr, off + d, 1, d, train)
            h = layer_norm(tape, ex, h, weight, lbias, spec.farg[i])
        elif k == LAYER_DROPOUT:
            if train and spec.farg[i] > Float32(0):
                comptime if HAS_DROPOUT:
                    h = tape.dropout(ex, h, spec.farg[i], seed, step * spec.n_layers() + i)
                else:
                    raise Error("model fit: Dropout is not available in this build")
        elif k == LAYER_RESIDUAL:
            var src = x0 if spec.arg0[i] == -1 else outs[spec.arg0[i]]
            h = tape.add(ex, h, src)
        elif k == LAYER_RESHAPE or k == LAYER_FLATTEN:
            h = tape.view_nd(ex, h, rn_with_batch(B, spec.out_shape[i], "model"))
        elif k == LAYER_PERMUTE:
            var axes = List[Int]()
            axes.append(0)
            for j in range(spec.in_shape[i].ndim):  # small-loop(ndim: the sample rank, at most 7): shape metadata, no data
                axes.append(spec.xargs[xa + j] + 1)
            h = tape.permute(ex, h, axes)
        elif k == LAYER_CONV2D:
            var oc = spec.arg0[i]
            var ks = spec.arg1[i]
            var c = conv_in_per_group(spec, i)
            var groups = conv_groups(spec, i)
            var off = spec.param_off[i]
            var wd = List[Int]()
            wd.append(oc)
            wd.append(c)
            wd.append(ks)
            wd.append(ks)
            var w2 = _param(tape, ex, P, Gr, off, oc, c * ks * ks, train)
            var w4 = tape.view_nd(ex, w2, rn_shape(wd, "model"))
            if spec.xargs[xa + 4] == 1:
                var bd = List[Int]()
                bd.append(oc)
                var b2 = _param(tape, ex, P, Gr, off + oc * c * ks * ks, 1, oc, train)
                var b1 = tape.view_nd(ex, b2, rn_shape(bd, "model"))
                h = rankn_conv2d(tape, ex, h, w4, Optional[Tensor](b1), spec.xargs[xa + 2], spec.xargs[xa + 3],
                                 groups)
            else:
                h = rankn_conv2d(tape, ex, h, w4, None, spec.xargs[xa + 2], spec.xargs[xa + 3], groups)
        elif k == LAYER_MAXPOOL2D:
            h = rankn_max_pool2d(tape, ex, h, spec.arg0[i], spec.arg1[i])
        elif k == LAYER_AVGPOOL2D:
            h = rankn_avg_pool2d(tape, ex, h, spec.arg0[i], spec.arg1[i])
        elif k == LAYER_CONV1D:
            var oc = spec.arg0[i]
            var ks = spec.arg1[i]
            var c = conv_in_per_group(spec, i)
            var groups = conv_groups(spec, i)
            var off = spec.param_off[i]
            var wd = List[Int]()
            wd.append(oc)
            wd.append(c)
            wd.append(ks)
            var w2 = _param(tape, ex, P, Gr, off, oc, c * ks, train)
            var w3 = tape.view_nd(ex, w2, rn_shape(wd, "model"))
            if spec.xargs[xa + 4] == 1:
                var bd = List[Int]()
                bd.append(oc)
                var b2 = _param(tape, ex, P, Gr, off + oc * c * ks, 1, oc, train)
                var b1 = tape.view_nd(ex, b2, rn_shape(bd, "model"))
                h = rankn_conv1d(tape, ex, h, w3, Optional[Tensor](b1), spec.xargs[xa + 2], spec.xargs[xa + 3],
                                 groups)
            else:
                h = rankn_conv1d(tape, ex, h, w3, None, spec.xargs[xa + 2], spec.xargs[xa + 3], groups)
        elif k == LAYER_MAXPOOL1D:
            h = rankn_max_pool1d(tape, ex, h, spec.arg0[i], spec.arg1[i], spec.xargs[xa + 2])
        elif k == LAYER_AVGPOOL1D:
            h = rankn_avg_pool1d(tape, ex, h, spec.arg0[i], spec.arg1[i], spec.xargs[xa + 2])
        elif k == LAYER_ADAPTIVE_AVGPOOL1D:
            var length = spec.in_shape[i].d[1]
            if spec.arg0[i] == 1:
                # the global mean of each (sample, channel) row: one blocked fold over the length
                # (the (B C, L) matrix view), parallel over rows and blocks
                h = tape.mean(ex, h, 1)
            else:
                # output_size divides the length: equal windows, an average pool
                var win = length // spec.arg0[i]
                h = rankn_avg_pool1d(tape, ex, h, win, win, 0)
        elif k == LAYER_BRANCH:
            h = x0 if spec.arg0[i] == -1 else outs[spec.arg0[i]]
        elif k == LAYER_CONCAT:
            var parts = List[Tensor]()
            for j in range(SPEC2_ARGS):  # small-loop(SPEC2_ARGS: six layer arguments): tensor handles, no data
                var v = spec.xargs[xa + j]
                if v == CONCAT_UNUSED:
                    continue
                parts.append(x0 if v == -1 else outs[v])
            # the sample's axis f (0: the first axis, the channels), after the batch axis
            h = rankn_concat(tape, ex, parts, Int(spec.farg[i]) + 1)
        # nn-attention begin
        elif nn_attention_is(k):
            var msrc = nn_attention_source(k, spec.xargs, i)
            var mem = h if msrc == NN_MHA_SELF else (x0 if msrc == -1 else outs[msrc])
            h = nn_attention_forward(tape, ex, k, spec.xargs, i, spec.farg[i], spec.in_shape[i], P, Gr,
                                     spec.param_off[i], h, mem, train, seed, step * spec.n_layers() + i, lengths)
        # nn-attention end
        elif is_block(k):
            var other: Optional[Tensor] = None
            if k == LAYER_MULTIPLY:
                other = Optional[Tensor](x0 if spec.arg0[i] == -1 else outs[spec.arg0[i]])
            h = block_forward(tape, ex, k, spec.xargs, i, spec.farg[i], spec.in_shape[i], P, Gr,
                              spec.param_off[i], h, other, train, lengths)
        h = rn_as_shape(tape, ex, h, rn_with_batch(B, spec.out_shape[i], "model"))
        outs.append(h)
        i += 1
    # the (B, outputs) result is read densely by the loss and the prediction copy
    tape.materialize(ex, h)
    return h


def _absorbed_activation(spec: ModelSpec, i: Int) -> Int:
    """The tape activation code (RELU, SIGMOID, TANH or SOFTMAX) of the
    activation layer right after Linear layer i, when tape.linear can take
    it into the same node; 0 when there is none or a Residual layer reads
    layer i's pre-activation output."""
    if i + 1 >= spec.n_layers():
        return 0
    for j in range(i + 1, spec.n_layers()):  # driver-loop: over layers
        var kj = spec.kinds[j]
        if (kj == LAYER_RESIDUAL or kj == LAYER_BRANCH) and spec.arg0[j] == i:
            return 0
        if is_block(kj) and block_reads(kj, spec.xargs, j, i):
            return 0
        if kj == LAYER_CONCAT:
            for q in range(SPEC2_ARGS):  # small-loop(SPEC2_ARGS: six layer arguments): metadata, no data
                if spec.xargs[SPEC2_ARGS * j + q] == i:
                    return 0
    var nk = spec.kinds[i + 1]
    if nk == LAYER_RELU:
        return RELU
    if nk == LAYER_SIGMOID:
        return SIGMOID
    if nk == LAYER_TANH:
        return TANH
    if nk == LAYER_SOFTMAX:
        return SOFTMAX
    return 0


def _fused_codes(cfg: FitConfig) -> Bool:
    """True when the batch target is the (B, 1) class codes and the loss is
    one softmax_cross_entropy node (the default build, cross_entropy on
    class codes)."""
    return cfg.loss == LOSS_CROSS_ENTROPY and cfg.target == TARGET_CODES


# lm_head (lane neural-blocks-3-lm)
def _token_ids[E: Exec](mut ex: E, Y: CastIn, N: Int, T: Int, dst: FP, what: String) raises:
    """dst (N * T float values) = the int32 token ids of Y (N, T), exactly (OP_ONE_HOT's code-value form)."""
    if Y.count() != N * T or Y.kind != CAST_KIND_SIGNED or Y.itemsize != 4 or not Y.is_c_order():
        raise Error(what + " must be the (n_samples, " + String(T) + ") C-contiguous int32 token ids")
    var ids = ex.alloc(N * T)
    ex.upload(ids, FP(unsafe_from_address=Y.addr), N * T)
    var oh = Args()
    oh.p0 = ids
    oh.p1 = dst
    oh.i0 = 1
    oh.i1 = 1
    ex.launch[OP_ONE_HOT](oh, N * T)
    ex.free(ids)


def _fit_loss[E: Exec](
    mut tape: Tape, mut ex: E, cfg: FitConfig, spec: ModelSpec, P: FP, Gr: FP, y_out: Tensor, target: Tensor,
    train: Bool, ctc_dims: Tuple[Int, Int],
) raises -> Tensor:
    """The batch loss of the fit and validation loops: model_loss, or the token cross-entropy, where `y_out` is
    the final lm_head layer's input (the forward stopped before it) and the fused op is the head. `ctc_dims` is
    the ctc output's (T, C), resolved once per fit by model_ctc_dims ((0, 0) for every other loss)."""
    if cfg.loss == LOSS_TOKEN_CROSS_ENTROPY:
        var i = spec.n_layers() - 1
        return block_lm_head_loss(tape, ex, spec.xargs, i, spec.in_shape[i], P, Gr, spec.param_off[i], y_out,
                                  target, train)
    # nn-losses-2 begin
    if cfg.loss == LOSS_CTC:
        return ctc_loss(tape, ex, y_out, target, ctc_dims[0], ctc_dims[1], ctc_blank_of(cfg.huber_delta), cfg.reduction,
                        ctc_zero_infinity_of(cfg.huber_delta))
    # nn-losses-2 end
    return model_loss(tape, ex, cfg, y_out, target)


def model_token_loss[E: Exec](
    mut tape: Tape, mut ex: E, spec: ModelSpec, P: FP, Gr: FP, x: Tensor, codes: Tensor, train: Bool,
    seed: UInt64, step: Int,
) raises -> Tensor:
    """The token cross-entropy of a model ending in lm_head as the fit records it: the forward stops before
    the head (`_ModelForwardTopology.stop`) and the fused head and loss run on its input. codes holds one
    class value per token, B * (the head input's rows per sample)."""
    var i = spec.n_layers() - 1
    if spec.version != SPEC2_VERSION or spec.kinds[i] != LAYER_LM_HEAD:
        raise Error("model: token_cross_entropy needs a model ending in an lm_head layer")
    var topology = _ModelForwardTopology(spec)
    topology.stop = i
    var h = _model_forward_prepared(tape, ex, spec, P, Gr, x, train, seed, step, topology)
    return block_lm_head_loss(tape, ex, spec.xargs, i, spec.in_shape[i], P, Gr, spec.param_off[i], h, codes, train)
# end lm_head


def model_loss[E: Exec](
    mut tape: Tape, mut ex: E, cfg: FitConfig, pred: Tensor, target: Tensor,
) raises -> Tensor:
    """The batch loss under cfg.reduction: the mean over every element (mse,
    l1, smooth_l1, huber, bce_with_logits, kl_div 'mean'), over the rows
    (cross_entropy, nll, kl_div 'batchmean'), or the sum. Every loss but
    cross_entropy is one Tape.loss_reduced node: the loss body of
    autograd/loss_ops.mojo / autograd/nn_losses.mojo folded in the leaf
    chain, one launch forward and one backward for the prediction (lane
    nn-losses: bce_with_logits and huber were compositions of up to twelve
    tape operations, a second copy of those bodies; mse was three)."""
    if cfg.loss == LOSS_CROSS_ENTROPY:
        if _fused_codes(cfg):
            # target holds the (B, 1) codes as float values
            if cfg.reduction == REDUCTION_SUM:
                return cross_entropy(tape, ex, pred, target, REDUCTION_SUM)
            return tape.softmax_cross_entropy(ex, pred, target)
        var ce = tape.cross_entropy(ex, pred, target)
        if cfg.reduction == REDUCTION_SUM:
            return tape.scale(ex, ce, Float32(tape.rows(pred)))
        return ce
    var kind = model_loss_kind(cfg.loss)
    var averaged = cfg.reduction != REDUCTION_SUM
    return tape.loss_reduced(ex, pred, target, kind, cfg.huber_delta, averaged, ls_per_row(kind, cfg.reduction))


# nn-losses begin
def model_loss_kind(loss: Int) raises -> Int:
    """The elementwise loss kind of a Sequential loss code."""
    if loss == LOSS_MSE:
        return LS_MSE
    if loss == LOSS_BCE_WITH_LOGITS:
        return LS_BCE_LOGITS
    if loss == LOSS_HUBER:
        return LS_HUBER
    if loss == LOSS_L1:
        return LS_MAE
    if loss == LOSS_SMOOTH_L1:
        return LS_SMOOTH_L1
    if loss == LOSS_NLL:
        return LS_NLL
    if loss == LOSS_KL_DIV:
        return LS_KL_DIV
    raise Error("model: loss code " + String(loss) + " is not an elementwise loss")


def model_curve_factor(cfg: FitConfig, rows: Int) -> Float32:
    """The factor of a batch's loss in the epoch's curve word (the curve is the words' sum over the fit's rows):
    the batch's rows for a mean, 1 for a sum (the curve is then the summed loss per sample)."""
    return Float32(1) if cfg.reduction == REDUCTION_SUM else Float32(rows)
# nn-losses end


def epoch_lr(lr: Float32, sched: Int, sched_step: Int, gamma: Float32, epoch: Int) -> Float32:
    """The learning rate of zero-based epoch `epoch` under schedule `sched`
    (SCHED_CONSTANT, SCHED_STEP: lr gamma^(epoch // sched_step),
    SCHED_EXPONENTIAL: lr gamma^epoch). A per-epoch host scalar computed in
    Mojo; model_fit and the operation session's optimizers
    (autograd/ops_api.mojo) both take it from here, so the same schedule
    gives the same word."""
    if sched == SCHED_CONSTANT:
        return lr
    var k = epoch // sched_step if sched == SCHED_STEP else epoch
    if k == 0:
        return lr
    return Float32(Float64(lr) * identical_pow64(Float64(gamma), Float64(k)))


def model_lr(cfg: FitConfig, epoch: Int) -> Float32:
    """The epoch's learning rate (a per-epoch scalar, computed in Mojo)."""
    return epoch_lr(cfg.lr, cfg.sched, cfg.sched_step, cfg.sched_gamma, epoch)


def model_epoch_lr(cfg: FitConfig, base: Float32, epoch: Int) raises -> Float32:
    """The rate of zero-based epoch `epoch` of cfg's per-epoch schedule from the base rate `base` (cfg.lr, or a
    parameter group's own rate: one schedule scales every group, as a torch scheduler scales each group's
    initial rate). reduce_on_plateau's rate is the fit's state, not this value."""
    if cfg.sched == NN_SCHED_MULTI_STEP:
        # torch MultiStepLR: lr gamma^k, k the milestones passed (epoch_lr's exponential form at epoch k)
        return epoch_lr(base, SCHED_EXPONENTIAL, 1, cfg.sched_gamma, nn_multi_step_count(cfg.sched_args, epoch))
    if cfg.sched > NN_SCHED_MULTI_STEP and cfg.sched != NN_SCHED_PLATEAU:
        return nn_epoch_lr(cfg.sched, base, cfg.sched_args, epoch)
    return epoch_lr(base, cfg.sched, cfg.sched_step, cfg.sched_gamma, epoch)


# ---------------------------------------------------------------- resumable state
# The state a fit carries from one epoch to the next beyond the parameters
# and the two optimizer buffers: one int64 word each, floats by their bits.
# `model_fit_state` writes it at the end of every fit; with `resume` it
# starts from it, so 2 + 2 epochs give the words of 4 epochs (the epoch
# index keys the shuffle (mlp_epoch_key) and the schedule, the step counter
# keys the optimizer's bias correction and the dropout masks).
comptime FIT_STATE_WORDS = 16 + SOURCE_PART_WORDS
comptime FS_VALID = 0      # 1 once a fit has written the state
comptime FS_EPOCH = 1      # epochs completed: the next epoch's index
comptime FS_STEP = 2       # optimizer steps completed
comptime FS_STALL = 3      # early stopping: epochs without improvement
comptime FS_BEST = 4       # early stopping: the best epoch loss (float64 bits)
comptime FS_PW1 = 5        # OptState.pw1, beta1^t (float32 bits)
comptime FS_PW2 = 6        # OptState.pw2, beta2^t (float32 bits)
comptime FS_MU = 7         # OptState.mu_prod (float32 bits)
comptime FS_OPT = 8        # the optimizer kind the buffers belong to
comptime FS_NPARAMS = 9    # the parameter count they cover
comptime FS_SEED_LO = 10   # the fit's seed, low 32 bits
comptime FS_SEED_HI = 11   # the fit's seed, high 32 bits
# lane nn-optim: reduce_on_plateau's state (autograd/nn_optim.mojo NnPlateau), kept across resume
comptime FS_PL_LR = 12     # the current rate (float64 bits; 0: not started, the compiled lr)
comptime FS_PL_BEST = 13   # the best monitored loss (float64 bits)
comptime FS_PL_BAD = 14    # epochs without improvement
comptime FS_PL_COOL = 15   # cooldown epochs left
# lane stream-fits (2026-11-25): a streamed fit's batch source part (autograd/batch_source.mojo source_part_store:
# the consumed position and the source's binding words, SOURCE_PART_WORDS words), so save_checkpoint and a resume
# in a fresh process continue on the same next batch; all 0 after a fit on arrays
comptime FS_SOURCE = 16

comptime IP64 = UnsafePointer[Int64, MutUntrackedOrigin]


def _f32_word(v: Float32) -> Int64:
    return Int64(bitcast[DType.uint32](v))


def _word_f32(w: Int64) -> Float32:
    return bitcast[DType.float32](UInt32(w & Int64(0xFFFFFFFF)))


# ---------------------------------------------------------------- fit / predict
def model_fit[E: Exec](
    mut ex: E, spec: ModelSpec, cfg: FitConfig, X: CastIn, Y: CastIn, N: Int, Pio: FP, curve: FP,
    loop_words: Int = 0,
) raises -> Int:
    """`model_fit_state` from a fresh start, the optimizer state discarded.
    `loop_words`: as `model_fit_state`'s."""
    var m = List[Float32](length=spec.n_params, fill=Float32(0))
    var v = List[Float32](length=spec.n_params, fill=Float32(0))
    var w = List[Float32](length=spec.n_params if nn_opt_keeps_third(cfg.opt.kind, cfg.opt.flags) else 1,
                          fill=Float32(0))
    var st = List[Int64](length=FIT_STATE_WORDS, fill=Int64(0))
    var n_iter = model_fit_state(
        ex, spec, cfg, X, Y, N, Pio, curve, FP(unsafe_from_address=Int(m.unsafe_ptr())),
        FP(unsafe_from_address=Int(v.unsafe_ptr())), IP64(unsafe_from_address=Int(st.unsafe_ptr())),
        curve, False, loop_words=loop_words, Wio=Int(w.unsafe_ptr()),
    )
    _ = m^
    _ = v^
    _ = w^
    _ = st^
    return n_iter


def model_fit_state[E: Exec](
    mut ex: E, spec: ModelSpec, cfg_in: FitConfig, X: CastIn, Y: CastIn, N: Int, Pio: FP, curve: FP,
    Mio: FP, Vio: FP, St: IP64, curve_prev: FP, resume: Bool,
    validation: Optional[FitValidation] = None, codes_dev: Int = 0, lengths: Optional[FP] = None,
    target_lengths: Optional[FP] = None, loop_words: Int = 0, Wio: Int = 0,
) raises -> Int:
    """Train the model (the loop is `_model_fit_state`, documented there).

    nn-blocks-zerocost (2026-11-25): an optimizer with parameter groups (Sequential.compile(param_groups=),
    autograd/nn_pgseq.mojo) is decided here, once per fit, as the loop's instance: the ungrouped instance's epoch
    and batch loop is the loop as it was (no per-epoch or per-batch test).

    `Wio` (lane stream-fits): the address of the third optimizer state buffer (n_params float32, `optimizer_w_`)
    when the optimizer keeps one (autograd/nn_optim.mojo nn_opt_keeps_third: centered RMSprop), else 0."""
    return _dispatch_fit_state[E, False](
        ex, spec, cfg_in, X, Y, N, Pio, curve, Mio, Vio, St, curve_prev, resume, validation, codes_dev,
        lengths, target_lengths, loop_words, Wio, 0, 0, 0)


def model_fit_state_source[E: Exec](
    mut ex: E, spec: ModelSpec, cfg_in: FitConfig, mut stream: BatchSource, max_steps: Int, Pio: FP, curve: FP,
    Mio: FP, Vio: FP, St: IP64, curve_prev: FP, resume: Bool, loop_words: Int, Wio: Int, source_out: Int,
    validation: Optional[FitValidation] = None,
) raises -> Int:
    """Sequential.fit(source=) (lane stream-fits, 2026-11-25): the fit loop's STREAMED instance
    (`_model_fit_state[..., streamed=True]`), chosen once per fit like the grouped one, so a fit on arrays runs the
    instance it ran before. The batches come from `stream` (autograd/batch_source.mojo, kept by the caller and set
    at the position to train from): one loop iteration (one `loss_curve_` word, the mean loss of its rows) is the
    rest of
    the source's current epoch, at most `max_steps` batches in the whole call when > 0, and the next batch is read
    ahead after each step is queued. Every batch is padded to the model's sequence length; the lengths field masks
    the recurrent layers. The schedule's epoch is the SOURCE's epoch, so a run stopped mid-epoch and resumed rates
    every batch as the uninterrupted run does. St receives the source part (FS_SOURCE: the consumed position and the
    source's binding words) and source_out (2 int64 words) the consumed position for the handle.

    Lane stream-fits-2: `validation` (validation_data arrays, as the array fit's) is evaluated, monitored (early
    stopping with n_iter_no_change, reduce_on_plateau) and restored (restore_best_weights) at the end of every PASS
    over the source, the iteration that trains an epoch's last batch; an iteration cut by `steps` mid-pass has no
    validation word (NaN) and no decision. Monitoring starts afresh each call, as for arrays. Early stopping and
    reduce_on_plateau without validation data are refused: a pass split across calls has two partial training
    means, so the training loss is not a monitor a resume repeats. Losses: the dense ones, cross_entropy and
    bce_with_logits on a class index a record, nll (its one-hot target made on the executor from the class index,
    one launch a batch) and token_cross_entropy (each record's target row holds its token ids); ctc is refused (a
    target row holds no target length)."""
    var address = Int(UnsafePointer(to=stream))
    return _dispatch_fit_state[E, True](
        ex, spec, cfg_in, CastIn.f32_c(0, 1, 1), CastIn.f32_c(0, 1, 1), stream.n, Pio, curve,
        Mio, Vio, St, curve_prev, resume, validation, 0, None, None, loop_words, Wio, address, max_steps, source_out)


def _dispatch_fit_state[E: Exec, streamed: Bool](
    mut ex: E, spec: ModelSpec, cfg_in: FitConfig, X: CastIn, Y: CastIn, N: Int, Pio: FP, curve: FP,
    Mio: FP, Vio: FP, St: IP64, curve_prev: FP, resume: Bool,
    validation: Optional[FitValidation], codes_dev: Int, lengths: Optional[FP],
    target_lengths: Optional[FP], loop_words: Int, Wio: Int, stream_address: Int, max_steps: Int,
    source_out: Int,
) raises -> Int:
    """Select the minibatch update once for this array or streamed fit."""
    if cfg_in.opt.kind == NN_OPT_ADAFACTOR:
        if cfg_in.groups.on():
            return _model_fit_state[E, True, streamed, True](
                ex, spec, cfg_in, X, Y, N, Pio, curve, Mio, Vio, St, curve_prev, resume, validation,
                codes_dev, lengths, target_lengths, loop_words, Wio, stream_address, max_steps, source_out)
        return _model_fit_state[E, False, streamed, True](
            ex, spec, cfg_in, X, Y, N, Pio, curve, Mio, Vio, St, curve_prev, resume, validation,
            codes_dev, lengths, target_lengths, loop_words, Wio, stream_address, max_steps, source_out)
    if cfg_in.groups.on():
        return _model_fit_state[E, True, streamed](
            ex, spec, cfg_in, X, Y, N, Pio, curve, Mio, Vio, St, curve_prev, resume, validation,
            codes_dev, lengths, target_lengths, loop_words, Wio, stream_address, max_steps, source_out)
    return _model_fit_state[E, False, streamed](
        ex, spec, cfg_in, X, Y, N, Pio, curve, Mio, Vio, St, curve_prev, resume, validation,
        codes_dev, lengths, target_lengths, loop_words, Wio, stream_address, max_steps, source_out)


def _model_fit_state[E: Exec, grouped: Bool, streamed: Bool, adafactor: Bool = False](
    mut ex: E, spec: ModelSpec, cfg_in: FitConfig, X: CastIn, Y: CastIn, N: Int, Pio: FP, curve: FP,
    Mio: FP, Vio: FP, St: IP64, curve_prev: FP, resume: Bool,
    validation: Optional[FitValidation], codes_dev: Int, lengths: Optional[FP],
    target_lengths: Optional[FP], loop_words: Int, Wio: Int, stream_address: Int, max_steps: Int,
    source_out: Int,
) raises -> Int:
    """Train the model on X (N, n_inputs) and Y; returns the epochs run.

    `loop_words` (lane alm-neural, 2026-11-24): the address of the fit
    loop's record words (sequence/fit_loop_words.mojo), or 0. The loop
    region is the epoch loop: the clock is read once before epoch 0's first
    launch (after the uploads, allocations and tapes) and once after the
    first existing `ex.sync()` that follows the loop (the validation
    readback's, else the parameter readback's): no synchronization added,
    nothing inside the loop; its iterations are the epochs executed in
    this call (before a restore of the best epoch).

    X and Y are the caller's arrays as they are (core/cast_in.mojo): any
    accepted dtype in C or Fortran order, cast to row-major float32 on the
    executor (autograd/cast_ops.mojo upload_cast; float32 C order is the
    plain upload). Y is (N, out) for TARGET_DENSE, or N C-contiguous int32
    class codes (their bits in float32 words) for TARGET_CODES (one-hot
    target) and TARGET_CODE_VALUES (the codes as the (N, 1) float target). Pio holds the starting
    parameters when cfg.init is False and receives the trained ones;
    curve[0:epochs run] receives each epoch's mean loss (the batch losses
    weighted by their row counts, a blocked fold, divided by N).

    Each epoch draws its row order on the executor (the MLP's keyed
    permutation, OP_MLP_PERM with mlp_epoch_key) or keeps the identity
    order; each batch gathers its rows, records the forward and the loss,
    runs backward into the gradient vector and one optimizer step
    (sequence/recurrent.mojo opt_step). With patience > 0 training stops
    after `patience` epochs whose loss did not improve on the best by more
    than tol (one word read back per epoch).

    Mio and Vio (n_params each) receive the optimizer's two state buffers
    (the momentum buffer, or Adam's first and second moments) and St the
    FIT_STATE_WORDS counters. With `resume` they are also the starting
    state: training continues at epoch St[FS_EPOCH] and step St[FS_STEP]
    with the first fit's seed, Pio holds the parameters (cfg.init must be
    False), curve_prev holds the St[FS_EPOCH] earlier epochs' losses and
    curve receives all of them, the earlier ones first. Without `resume`,
    curve[0:epochs run] receives this fit's.

    `codes_dev` (lane migrate-neural-2): the N class codes already in an
    executor buffer (the label sort wrote them there); Y is then not read.
    The codes feed the same one-hot launch, so the same words.

    `lengths` (lane seq-blocks-3): N float32 sequence lengths on the host, one
    a row, for the recurrent layers (model_lengths_check admits them). They
    are uploaded once, gathered in each epoch's row order with one launch and
    read by every batch's recurrent layers (autograd/model_blocks.mojo
    block_recurrent: packed-sequence steps).

    `streamed` (lane stream-fits): the instance of `model_fit_state_source`;
    `stream_address` is its BatchSource, `max_steps` its batch cap (0: none)
    and `source_out` its two position words. Every `comptime if streamed`
    below is that instance's only difference; the array instances
    (streamed False) compile the loop as it was.
    """
    var stream = UnsafePointer[BatchSource, MutUntrackedOrigin](unsafe_from_address=stream_address)
    var cfg = cfg_in.copy()
    var start = 0
    var step = 0
    # the generic EpochMonitor block (autograd/epoch_monitor.mojo, lane rnn-parity-c): scikit-learn's
    # `loss > best - tol` stall rule, the one the recurrent estimators' rnn_fit runs; FS_STALL / FS_BEST resume it
    var monitor = EpochMonitor(Float64(cfg.tol), cfg.patience)
    var st = OptState()
    var plateau = NnPlateau(Float64(cfg.lr))
    if resume:
        if cfg.init:
            raise Error("model fit: resuming continues from the saved parameters (init must be 0)")
        if St[FS_VALID] != 1 or Int(St[FS_OPT]) != cfg.opt.kind or Int(St[FS_NPARAMS]) != spec.n_params:
            raise Error("model fit: resume needs the training state of an earlier fit of this model "
                        + "with the same optimizer")
        start = Int(St[FS_EPOCH])
        step = Int(St[FS_STEP])
        monitor = EpochMonitor.resumed(Float64(cfg.tol), cfg.patience, bitcast[DType.float64](St[FS_BEST]),
                                       Int(St[FS_STALL]))
        st.pw1 = _word_f32(St[FS_PW1])
        st.pw2 = _word_f32(St[FS_PW2])
        st.mu_prod = _word_f32(St[FS_MU])
        cfg.seed = (UInt64(St[FS_SEED_HI]) << 32) | UInt64(St[FS_SEED_LO])
        if St[FS_PL_LR] != 0:
            plateau.lr = bitcast[DType.float64](St[FS_PL_LR])
            plateau.best = bitcast[DType.float64](St[FS_PL_BEST])
            plateau.bad = Int(St[FS_PL_BAD])
            plateau.cool = Int(St[FS_PL_COOL])
        if start < 0 or step < 0 or start + cfg.epochs >= MODEL_MAX_ROWS:
            raise Error("model fit: the saved epoch and step counters are out of range")
    check_fit(spec, cfg, N)
    var D = spec.n_inputs
    var O = spec.out_features()
    var np_ = spec.n_params
    var bs = cfg.batch if cfg.batch < N else N
    # lane stream-fits: the third optimizer state buffer (centered RMSprop's momentum buffer, optimizer_w_)
    var three = nn_opt_keeps_third(cfg.opt.kind, cfg.opt.flags)
    if three and Wio == 0:
        raise Error("model fit: centered RMSprop keeps a third optimizer state buffer; its address is missing")
    var has_lens = Bool(lengths)
    var pad = 0  # streamed: every batch padded to the model's sequence length
    var dX = dummy_ptr()
    comptime if streamed:
        if (cfg.patience > 0 or cfg.sched == NN_SCHED_PLATEAU) and not validation:
            raise Error("ValueError: Sequential.fit(source=) monitors validation_data for early stopping "
                        + "(n_iter_no_change) and reduce_on_plateau: a pass over the source split across calls has "
                        + "two partial training means, so the training loss is not a monitor a resume repeats")
        if cfg.loss == LOSS_CTC:
            raise Error("ValueError: Sequential.fit(source=) does not train the ctc loss: it reads each sample's "
                        + "label sequence AND its target length, and a source's target row holds no target length")
        var steps = model_sequence_steps(spec, "Sequential.fit(source=)")
        var F = stream[].width
        if steps > 0:
            # a recurrent layer reads the padded sequences, masked by each record's length
            if steps < stream[].max_len or D != steps * F:
                raise Error("ValueError: Sequential.fit(source=): the model reads sequences of " + String(steps)
                            + " steps of " + String(D // steps if steps > 0 else D) + " features; the source's "
                            + "records have up to " + String(stream[].max_len) + " steps of " + String(F))
            pad = steps
            has_lens = True
        elif stream[].min_len != stream[].max_len or D != stream[].max_len * F:
            raise Error("ValueError: Sequential.fit(source=): a model without recurrent layers reads records of one "
                        + "length, " + String(D) + " values each; the source's records have "
                        + String(stream[].min_len) + " to " + String(stream[].max_len) + " steps of " + String(F)
                        + " features")
        else:
            pad = stream[].max_len
    else:
        if X.rows != N or X.cols != D:
            raise Error("model fit: X must be (n_samples, " + String(D) + ")")
        dX = ex.alloc(N * D)
        upload_cast(ex, dX, X)
    var dLen = dummy_ptr()
    var Lp = dummy_ptr()
    comptime if streamed:
        if has_lens:
            Lp = ex.alloc(bs)  # the batch's lengths, uploaded with it
            if validation and not validation.value().lengths:
                raise Error("ValueError: Sequential.fit(source=): the records are masked by their lengths, so "
                            + "validation_data takes (X, y, lengths)")
    else:
        if has_lens:
            # lane rnn-parity-c: validation rows carry their own lengths (validation_data=(X, y, lengths)), checked
            # below
            if validation and not validation.value().lengths:
                raise Error("ValueError: model fit: with lengths, validation_data takes (X, y, lengths)")
            model_lengths_check(spec, lengths.value(), N, "fit")
            dLen = ex.alloc(N)
            ex.upload(dLen, lengths.value(), N)
            Lp = ex.alloc(N)
    # the batch target's width: the outputs, or 1 for the class codes of the
    # fused cross-entropy (no one-hot target)
    var TW = 1 if _fused_codes(cfg) else O
    # lm_head (lane neural-blocks-3-lm): one token id per row of the head's input sample
    var token = cfg.loss == LOSS_TOKEN_CROSS_ENTROPY
    if token:
        TW = spec.in_shape[spec.n_layers() - 1].rows()
    # nn-losses-2 begin: ctc's batch target is the packed (N, 2 + S) rows; Y is the (N, S) float32 labels, checked
    # on the host with the target lengths (and the input lengths) before anything is uploaded
    var ctc = cfg.loss == LOSS_CTC
    var ctc_S = 0
    # (T, C) of the ctc output, resolved once per fit (lane nn-integrate: _fit_loss resolved it every batch)
    var ctc_dims = (0, 0)
    if ctc:
        var dims = model_ctc_dims(spec)
        ctc_dims = dims
        if validation:
            raise Error("ValueError: model fit: validation_data with the ctc loss is not supported")
        if not target_lengths:
            raise Error("ValueError: model fit: the ctc loss needs target_lengths, one a sample")
        if has_lens and spec.sample.d[0] != dims[0]:
            raise Error("ValueError: model fit: with lengths, ctc needs one output step per input step ("
                        + String(spec.sample.d[0]) + " input steps, " + String(dims[0]) + " output steps)")
        if Y.rows != N or Y.kind != CAST_KIND_FLOAT or Y.itemsize != 4 or not Y.is_c_order():
            raise Error("model fit: ctc targets must be the (n_samples, max target length) C-contiguous float32 labels")
        ctc_S = Y.cols
        ctc_admit(FP(unsafe_from_address=Y.addr), target_lengths.value(), lengths, N, ctc_S, dims[0], dims[1],
                  ctc_blank_of(cfg.huber_delta))
        TW = ctc_S + 2
    elif target_lengths:
        raise Error("ValueError: model fit: target_lengths apply to the ctc loss")
    # nn-losses-2 end
    var dY = dummy_ptr()
    comptime if streamed:
        # lane stream-fits: the targets come with each batch (one row of floats a record): a dense loss reads O
        # words a record; cross_entropy and bce_with_logits read one class index, admitted once a fit
        if stream[].spec.target != BS_FIELD_TARGETS:
            raise Error("ValueError: Sequential.fit(source=) needs the source's target='targets'")
        if token:
            # lane stream-fits-2: a record's target row is its token ids (float32 words, whole numbers below the
            # head's vocabulary): the words _token_ids makes from an array's int32 ids
            if stream[].twidth != TW:
                raise Error("ValueError: Sequential.fit(source=): token_cross_entropy reads " + String(TW)
                            + " token ids a record; the source's targets have width " + String(stream[].twidth))
            # whole numbers >= 0, as the array fit admits int32 ids (the head reads them against its vocabulary)
            _ = batch_source_class_scan(stream[], 0, "Sequential.fit(source=)")
        elif cfg.target == TARGET_DENSE:
            if stream[].twidth != TW:
                raise Error("ValueError: Sequential.fit(source=): the model has " + String(TW) + " outputs; the "
                            + "source's targets have width " + String(stream[].twidth))
        else:
            if stream[].twidth != 1:
                raise Error("ValueError: Sequential.fit(source=): a classification loss reads one class index a "
                            + "record (targets of width 1); the source's targets have width "
                            + String(stream[].twidth))
            batch_source_check_classes(stream[], 2 if cfg.target == TARGET_CODE_VALUES else O,
                                       "Sequential.fit(source=)")
    else:
        dY = ex.alloc(N * TW)
        if token:
            _token_ids(ex, Y, N, TW, dY, "model fit: y")
        # nn-losses-2 begin: the target rows, one pack launch once per fit
        elif ctc:
            var dLab = ex.alloc(N * ctc_S)
            ex.upload(dLab, FP(unsafe_from_address=Y.addr), N * ctc_S)
            var dTL = ex.alloc(N)
            ex.upload(dTL, target_lengths.value(), N)
            var in_lens = Optional[FP](dLen) if has_lens else Optional[FP]()
            ctc_pack_targets(ex, dY, dLab, dTL, in_lens, N, ctc_S, ctc_dims[0])
            ex.free(dLab)
            ex.free(dTL)
        # nn-losses-2 end
        elif cfg.target == TARGET_CODES or cfg.target == TARGET_CODE_VALUES:
            # the codes' int32 bits travel in float32 words (op_one_hot reads them)
            var dC: FP
            if codes_dev != 0:
                dC = FP(unsafe_from_address=codes_dev)
            else:
                if Y.count() != N or Y.kind != CAST_KIND_SIGNED or Y.itemsize != 4 or not Y.is_c_order():
                    raise Error("model fit: the class codes must be N C-contiguous int32")
                dC = ex.alloc(N)
                ex.upload(dC, FP(unsafe_from_address=Y.addr), N)
            var oh = Args()
            oh.p0 = dC
            oh.p1 = dY
            oh.i0 = TW
            oh.i1 = 1 if (cfg.target == TARGET_CODE_VALUES or TW == 1) else 0
            ex.launch[OP_ONE_HOT](oh, N * TW)
        else:
            if Y.rows != N or Y.cols != O:
                raise Error("model fit: y must have " + String(O) + " outputs per row")
            upload_cast(ex, dY, Y)
    var P = ex.alloc(np_)
    if cfg.init:
        model_init_params(ex, spec, P, cfg.seed)
    else:
        ex.upload(P, Pio, np_)
    var Gr = ex.alloc(np_)
    # Lane seq-blocks-3: state words (a batch norm's running statistics) live in the parameter vector, so
    # checkpoints, state_dict and the digest carry them, but the optimizer must not move them: their gradient
    # words are zero (filled once; no rule writes them) and each step restores them from Sv after opt_step,
    # undoing a weight decay. Two copy launches a state range a step; none for a model without state.
    var ranges = model_state_ranges(spec)
    var n_state = 0
    for r in range(len(ranges) // 2):  # driver-loop: over state ranges (description metadata)
        n_state += ranges[2 * r + 1]
    var Sv = dummy_ptr()
    if n_state > 0:
        Sv = ex.alloc(n_state)
        _fill(ex, Gr, np_, Float32(0))
    var s1 = ex.alloc(np_)
    var s2 = ex.alloc(np_)
    var s3 = ex.alloc(np_)
    if resume:
        # the optimizer buffers where the earlier fit left them (s3 is read only by centered RMSprop, which saves
        # it: nn_opt_keeps_third, lane stream-fits)
        ex.upload(s1, Mio, np_)
        ex.upload(s2, Vio, np_)
        if three:
            ex.upload(s3, FP(unsafe_from_address=Wio), np_)
    elif cfg.opt.kind == NN_OPT_ADAGRAD and cfg.opt.f2 != Float32(0) and not grouped:
        # torch Adagrad's initial_accumulator_value: the sum starts there (one fill a fit)
        _fill(ex, s2, np_, cfg.opt.f2)
    # lane nn-optim: LAMB's buffers and tensor table, once a fit (nothing for the other optimizers)
    var af_update = dummy_ptr()
    var af_scales = dummy_ptr()
    var af_parts = dummy_ptr()
    comptime if adafactor:
        if len(cfg.af_tensors) == 0:
            raise Error("model fit: Adafactor needs the model's vector and matrix parameter shapes")
        var max_size = 1
        var max_parts = 1
        for j in range(len(cfg.af_tensors)):  # driver-loop: over parameter shapes, once per fit
            var tensor = cfg.af_tensors[j]
            var size = tensor[1] * max(1, tensor[2])
            max_size = max(max_size, size)
            max_parts = max(max_parts, adafactor_partial_words(size))
        af_update = ex.alloc(max_size)
        af_scales = ex.alloc(4)
        af_parts = ex.alloc(max_parts)
    var lamb = NnLamb(ex, cfg.opt, cfg.tensor_offs, np_)
    # nn-blocks-pgseq begin: the parameter groups, resolved once a fit (nothing without groups): each group's
    # running products from the step count, LAMB's per-run tables, adagrad's starting sums per group
    pgseq_admit_fit(cfg.groups, cfg.sched)
    var pg_state = PgSeqState(cfg.groups, step)
    var pg_lamb = PgSeqLamb(ex, cfg.groups)
    var pg_rates = List[Float32](length=cfg.groups.groups(), fill=Float32(0))
    comptime if grouped:  # the loop's instance (model_fit_state)
        if not resume:
            pgseq_fill_accumulators(ex, cfg.groups, s2)
    # nn-blocks-pgseq end
    var didx = dummy_ptr()
    var xb = ex.alloc(bs * D)
    var yb = ex.alloc(bs * TW)
    # lane stream-fits-2: a streamed nll fit uploads each batch's class indices here and makes yb's one-hot rows
    var one_hot = False
    var yc = dummy_ptr()
    comptime if streamed:
        if cfg.target == TARGET_CODES and not _fused_codes(cfg):
            one_hot = True
            yc = ex.alloc(bs)
    # AUTOGRAD_GATHER_EPOCH (autograd/data.mojo, lane ag-pass5-composed): one
    # gather an epoch into a permuted copy when it fits; each batch is then
    # a slice of it (xb and yb point into the copy). The same words.
    var Xp = dummy_ptr()
    var Yp = dummy_ptr()
    var epoch_gather = False
    var n_batches = (N + bs - 1) // bs
    comptime if streamed:
        n_batches = stream[].n_batches  # xb, yb (and Lp) receive each batch's upload
    else:
        didx = ex.alloc(N)
        epoch_gather = gather_epoch_fits(N, D + TW)
        if epoch_gather:
            Xp = ex.alloc(N * D)
            Yp = ex.alloc(N * TW)
    # lane nn-optim: a per-step schedule's rates for every step this fit can run, filled once (host scalars)
    var per_step = nn_sched_per_step(cfg.sched)
    var step0 = step
    var step_lrs = List[Float32]()
    if per_step:
        step_lrs = nn_step_lrs(cfg.sched, cfg.lr, cfg.sched_args, step + 1, cfg.epochs * n_batches)
    # nn-blocks-pgseq: a per-step schedule's rates for every group, filled once (empty without groups)
    var pg_step_lrs = List[List[Float32]]()
    comptime if grouped:
        if per_step:
            for g in range(cfg.groups.groups()):  # driver-loop: over parameter groups (once per fit)
                pg_step_lrs.append(nn_step_lrs(cfg.sched, cfg.groups.lrs[g], cfg.sched_args, step + 1,
                                               cfg.epochs * n_batches))
    var plateau_on = cfg.sched == NN_SCHED_PLATEAU
    var dloss = ex.alloc(n_batches)
    var dcurve = ex.alloc(cfg.epochs)
    var nv = 0
    var vx = dummy_ptr()
    var vy = dummy_ptr()
    var vloss = dummy_ptr()
    var vcurve = dummy_ptr()
    var best_p = dummy_ptr()
    var best_m = dummy_ptr()
    var best_v = dummy_ptr()
    var best_w = dummy_ptr()  # lane stream-fits: the third buffer's best copy (centered RMSprop)
    var vLen = dummy_ptr()
    var best_step = step
    var best_src_epoch = -1  # lane stream-fits-2: the source epoch whose end the kept (best) pass is
    var best_pw1 = st.pw1
    var best_pw2 = st.pw2
    var best_mu = st.mu_prod
    var restore = False
    if validation:
        var v = validation.value().copy()
        nv = v.x.rows
        if nv < 1 or v.x.cols != D or v.y.rows != nv:
            raise Error("validation_data must have matching nonempty rows and the training input width")
        if v.lengths:
            if not has_lens:
                raise Error("ValueError: model fit: validation lengths need the training lengths (fit(..., lengths=))")
            model_lengths_check(spec, v.lengths.value(), nv, "fit (validation_data)")
            # uploaded once in the validation rows' order (the validation batches never shuffle)
            vLen = ex.alloc(nv)
            ex.upload(vLen, v.lengths.value(), nv)
        vx = ex.alloc(nv * D)
        vy = ex.alloc(nv * TW)
        upload_cast(ex, vx, v.x)
        if token:
            _token_ids(ex, v.y, nv, TW, vy, "validation y")
        elif cfg.target == TARGET_DENSE:
            if v.y.cols != O:
                raise Error("validation targets must have the model's output width")
            upload_cast(ex, vy, v.y)
        else:
            var codes = ex.alloc(nv)
            validation_codes(ex, v.y, v.classes, codes)
            var oh = Args()
            oh.p0 = codes
            oh.p1 = vy
            oh.i0 = TW
            oh.i1 = 1 if (cfg.target == TARGET_CODE_VALUES or TW == 1) else 0
            ex.launch[OP_ONE_HOT](oh, nv * TW)
            ex.free(codes)
        vloss = ex.alloc((nv + bs - 1) // bs)
        vcurve = ex.alloc(cfg.epochs)
        restore = v.restore
        if restore:
            best_p = ex.alloc(np_)
            best_m = ex.alloc(np_)
            best_v = ex.alloc(np_)
            if three:
                best_w = ex.alloc(np_)
        # Validation monitoring starts afresh for each fit call. A resumed
        # call can use a different validation set without stale comparisons.
        monitor.restart()
    var tape = Tape()
    var topology = _ModelForwardTopology(spec)
    if token:
        topology.stop = spec.n_layers() - 1
    var epoch_tape = Tape()
    var word = List[Float32](length=1, fill=Float32(0))
    var inv_n = ftz(identical_div(Float32(1), Float32(N)))
    var n_iter = 0
    var loop_record = FitLoop.empty()
    # lane alm-left: the queued setup (uploads, fills) drained ONCE before the
    # clock, outside the loop, so the loop region holds loop work only
    ex.sync()
    # lane stream-fits: the streamed instance's batches trained in this call, the iteration's first and end batch,
    # its rows, and the upload words staged since the loop's last wait (autograd/batch_source.mojo THE WAIT)
    var taken = 0
    var staged = 0
    var first_b = 0
    var last_b = n_batches
    var rows_run = N
    var loop_start = fit_loop_start()
    for e in range(cfg.epochs):  # driver-loop: over epochs
        # the absolute epoch index: a resumed fit continues the row orders
        # and the schedule of the fit it resumes
        var epoch = start + e
        comptime if streamed:
            # one iteration is the rest of the source's current epoch, within the call's batch cap; its schedule
            # epoch is the source's, so a run stopped mid-epoch and resumed rates each batch as the uninterrupted
            # run does
            if max_steps > 0 and taken >= max_steps:
                break
            var at = stream[].pending()
            stream[].epoch = at[0]
            stream[].consumed = at[1]
            epoch = at[0]
            first_b = at[1]
            last_b = n_batches
            if max_steps > 0:
                last_b = min(n_batches, first_b + (max_steps - taken))
            rows_run = 0
        else:
            if cfg.shuffle:
                ex.launch[OP_MLP_PERM](mlp_perm_args(N, mlp_epoch_key(cfg.seed, epoch), didx), N)
            elif e == 0:
                var ia = Args()
                ia.p0 = didx
                ex.launch[OP_SEQ_IOTA](ia, N)
        # the epoch's rate (lane nn-blocks-pgseq: one body, model_epoch_lr, for the fit's rate and each group's)
        var lr = Float32(plateau.lr) if plateau_on else model_epoch_lr(cfg, cfg.lr, epoch)
        comptime if grouped:
            for g in range(cfg.groups.groups()):  # driver-loop: over parameter groups (a rate a group an epoch)
                pg_rates[g] = model_epoch_lr(cfg, cfg.groups.lrs[g], epoch)
        comptime if not streamed:
            # the order changes only when shuffling: an unshuffled fit keeps epoch 0's copy
            if epoch_gather and (cfg.shuffle or e == 0):
                gather_epoch(ex, dX, dY, didx, N, D, TW, Xp, Yp)
            if has_lens and (cfg.shuffle or e == 0):
                # the lengths in this epoch's row order, one launch (op_gather_rows, one word a row)
                var gl = Args()
                gl.p0 = dLen
                gl.p1 = didx
                gl.p2 = Lp
                gl.p3 = dLen
                gl.p4 = Lp
                gl.i0 = N
                gl.i1 = 1
                gl.i2 = 0
                gl.i3 = 0
                ex.launch[OP_GATHER_ROWS](gl, N)
        for bi in range(first_b, last_b):  # driver-loop: over batches
            var jb = bi - first_b  # the batch's word of dloss (bi for the array instances)
            var off = bi * bs
            var B = 0
            var loff = off  # the batch's first word of Lp
            comptime if streamed:
                # the batch read ahead after the previous step (or, for the call's first batch, read here), padded
                # to the model's sequence length; the waits bound the host staging (autograd/batch_source.mojo)
                if not stream[].holds(stream[].epoch, bi, pad):
                    stream[].read(stream[].epoch, bi, pad)
                if staged >= BATCH_SOURCE_STAGE_WORDS:
                    ex.sync()
                    staged = 0
                B = stream[].held_rows
                rows_run += B
                loff = 0
                staged += batch_source_upload(ex, stream[], BS_FIELD_STEPS, xb).count()
                if one_hot:
                    # nll: the batch's class indices, then their one-hot rows (op_one_hot's float form)
                    staged += batch_source_upload(ex, stream[], BS_FIELD_TARGETS, yc).count()
                    var oh = Args()
                    oh.p0 = yc
                    oh.p1 = yb
                    oh.i0 = TW
                    oh.i1 = 2
                    ex.launch[OP_ONE_HOT](oh, B * TW)
                else:
                    staged += batch_source_upload(ex, stream[], BS_FIELD_TARGETS, yb).count()
                if has_lens:
                    staged += batch_source_upload(ex, stream[], BS_FIELD_LENGTHS, Lp).count()
            else:
                B = bs if off + bs <= N else N - off
                var gathered = False
                if epoch_gather:
                    xb = Xp + off * D
                    yb = Yp + off * TW
                    gathered = True
                if not gathered:
                    # X and Y rows of the batch in one launch
                    var gx = Args()
                    gx.p0 = dX
                    gx.p1 = didx
                    gx.p2 = xb
                    gx.p3 = dY
                    gx.p4 = yb
                    gx.i0 = B
                    gx.i1 = D
                    gx.i2 = off
                    gx.i3 = TW
                    ex.launch[OP_GATHER_ROWS](gx, B * (D + TW))
            tape.reset()
            var x = tape.variable(ex, xb, B, D, False)
            var target = tape.variable(ex, yb, B, TW, False)
            var blens = Optional[FP]()
            if has_lens:
                blens = Optional[FP](Lp + loff)
            var out = _model_forward_prepared(tape, ex, spec, P, Gr, x, True, cfg.seed, step, topology, blens)
            # lane ag-pass6-fold (PASS6 idea H(1)): the loss's own fold
            # launch stores the batch's curve word mul(loss, B), the word
            # tape.scale stores; cross-entropy's loss is not a fold
            # output (a scale of one), so it keeps the two launches below
            if cfg.loss != LOSS_CROSS_ENTROPY and not token and not ctc:  # nn-losses-2: ctc's curve word is the scale below
                tape.arm_fold_word(dloss + jb, model_curve_factor(cfg, B))
            var loss = _fit_loss(tape, ex, cfg, spec, P, Gr, out, target, True, ctc_dims)
            var word_done = tape.disarm_fold_word(loss)
            if spec.version == SPEC2_VERSION:
                # the rank-N lanes' custom tags (conv2d, pools) beside NeuralRules; lane neural-blocks:
                # ModelRules is RankNRules plus the import family (the block activations, the attention
                # mask, the embedding's ids) and the recurrent layer's rules; RankNRules' tags unchanged
                tape.backward_scalar_with_rules(ex, loss, ModelRules())
            else:
                tape.backward_scalar_with_rules(ex, loss, NeuralRules())
            if not word_done:
                # the batch's loss times its rows, for the epoch's weighted mean
                var weighted = tape.scale(ex, loss, model_curve_factor(cfg, B))
                _copy(ex, dloss + jb, tape.value(weighted), 1)
            step += 1
            var so = 0
            for r in range(len(ranges) // 2):  # driver-loop: over state ranges (at most one a batch-norm layer)
                _copy(ex, Sv + so, P + ranges[2 * r], ranges[2 * r + 1])
                so += ranges[2 * r + 1]
            var step_lr = step_lrs[step - step0 - 1] if per_step else lr
            comptime if grouped:
                # nn-blocks-pgseq: one step a run of one group's tensors, each at its group's rate
                if per_step:
                    for g in range(cfg.groups.groups()):  # driver-loop: over parameter groups (a rate a group)
                        pg_rates[g] = pg_step_lrs[g][step - step0 - 1]
            comptime if adafactor:
                for j in range(len(cfg.af_tensors)):  # driver-loop: over parameter tensors on the executor
                    var tensor = cfg.af_tensors[j]
                    var off = tensor[0]
                    var rate = step_lr
                    var eps = cfg.opt.eps
                    var decay = cfg.opt.wd
                    comptime if grouped:
                        var group = tensor[3]
                        rate = pg_rates[group]
                        eps = cfg.groups.cfgs[group].eps
                        decay = cfg.groups.cfgs[group].wd
                    adafactor_update(ex, P + off, Gr + off, s1 + off, s2 + off, tensor[1], tensor[2], step,
                                     Float64(rate), -0.8, Float64(eps), 0.001, 1.0, Float64(decay),
                                     af_update, af_scales, af_parts)
            elif grouped:
                pgseq_step(ex, cfg.groups, pg_state, step, pg_rates, P, Gr, s1, s2, s3, lamb, pg_lamb)
            else:
                nn_opt_step(ex, cfg.opt, st, step, step_lr, P, Gr, s1, s2, s3, np_, lamb)
            so = 0
            for r in range(len(ranges) // 2):  # driver-loop: over state ranges (at most one a batch-norm layer)
                _copy(ex, P + ranges[2 * r], Sv + so, ranges[2 * r + 1])
                so += ranges[2 * r + 1]
            comptime if streamed:
                # this batch is consumed (its step is queued); then the next batch is read ahead while the device
                # trains this one. The training state records the consumed position, never the read-ahead.
                taken += 1
                stream[].consumed = bi + 1
                var ahead = stream[].next_position(stream[].epoch, bi)
                stream[].read(ahead[0], ahead[1], pad)
        epoch_tape.reset()
        var parts = epoch_tape.variable(ex, dloss, 1, last_b - first_b, False)
        var inv_rows = inv_n
        comptime if streamed:
            inv_rows = ftz(identical_div(Float32(1), Float32(rows_run)))
        var mean = epoch_tape.scale(ex, epoch_tape.sum(ex, parts), inv_rows)
        _copy(ex, dcurve + e, epoch_tape.value(mean), 1)
        n_iter = e + 1
        var monitored = dcurve + e
        # lane stream-fits-2: a streamed iteration is watched only when it ends a pass over the source (an iteration
        # cut by the call's step cap mid-pass has no validation word and no decision)
        var watched = True
        comptime if streamed:
            watched = last_b == n_batches
            if nv > 0 and not watched:
                _fill(ex, vcurve + e, 1, nan[DType.float32]())
        if nv > 0 and watched:
            var nb = (nv + bs - 1) // bs
            for vi in range(nb):  # driver-loop: over validation batches
                var off = vi * bs
                var B = min(bs, nv - off)
                tape.reset()
                var x = tape.variable(ex, vx + off * D, B, D, False)
                var target = tape.variable(ex, vy + off * TW, B, TW, False)
                # lane rnn-parity-c: the validation lengths mask the recurrent layers as the training lengths do
                var vlens = Optional[FP]()
                if has_lens:
                    vlens = Optional[FP](vLen + off)
                var pred = _model_forward_prepared(tape, ex, spec, P, Gr, x, False, cfg.seed, step, topology, vlens)
                var loss = _fit_loss(tape, ex, cfg, spec, P, Gr, pred, target, False, ctc_dims)
                var weighted = tape.scale(ex, loss, model_curve_factor(cfg, B))
                _copy(ex, vloss + vi, tape.value(weighted), 1)
            epoch_tape.reset()
            var parts = epoch_tape.variable(ex, vloss, 1, nb, False)
            var average = epoch_tape.scale(ex, epoch_tape.sum(ex, parts), identical_div(Float32(1), Float32(nv)))
            _copy(ex, vcurve + e, epoch_tape.value(average), 1)
            monitored = vcurve + e
        if (cfg.patience > 0 or nv > 0 or plateau_on) and watched:
            ex.sync()
            ex.download(FP(unsafe_from_address=Int(word.unsafe_ptr())), monitored, 1)
            var value = Float64(word[0])
            # the generic block's decision (refuses a loss that is not finite before the plateau reads it)
            var improved = monitor.observe(value, e + 1)
            if plateau_on:
                # torch ReduceLROnPlateau.step(loss) on the loss read above (autograd/nn_optim.mojo)
                plateau.epoch_end(value, cfg.sched_args)
            if improved:
                if restore:
                    _copy(ex, best_p, P, np_)
                    _copy(ex, best_m, s1, np_)
                    _copy(ex, best_v, s2, np_)
                    if three:
                        _copy(ex, best_w, s3, np_)
                    best_step = step
                    best_pw1 = st.pw1
                    best_pw2 = st.pw2
                    best_mu = st.mu_prod
                    comptime if streamed:
                        best_src_epoch = stream[].epoch  # this pass's source epoch (its end is the kept position)
            if monitor.stop():
                break
    if validation:
        var v = validation.value().copy()
        ex.sync()
        loop_record = fit_loop_stop(loop_start, n_iter)
        ex.download(v.curve, vcurve, n_iter)
        v.result[0] = Int64(n_iter)
        v.result[1] = Int64(start + monitor.best_epoch)
        if restore:
            _copy(ex, P, best_p, np_)
            _copy(ex, s1, best_m, np_)
            _copy(ex, s2, best_v, np_)
            if three:
                _copy(ex, s3, best_w, np_)
            step = best_step
            st.pw1 = best_pw1
            st.pw2 = best_pw2
            st.mu_prod = best_mu
            n_iter = monitor.best_epoch
            comptime if streamed:
                # the source at the best pass's end, so the state's position matches its parameters and counters
                if best_src_epoch >= 0:
                    stream[].epoch = best_src_epoch
                    stream[].consumed = n_batches
    ex.sync()
    if not loop_record.recorded:
        loop_record = fit_loop_stop(loop_start, n_iter)
    fit_loop_publish(loop_words, loop_record)
    ex.download(Pio, P, np_)
    ex.download(Mio, s1, np_)
    ex.download(Vio, s2, np_)
    if three:
        ex.download(FP(unsafe_from_address=Wio), s3, np_)
    if resume and start > 0:
        unsafe_memcpy(dest=curve, src=curve_prev, count=start)
    ex.download(curve + start, dcurve, n_iter)
    St[FS_VALID] = 1
    St[FS_EPOCH] = Int64(start + n_iter)
    St[FS_STEP] = Int64(step)
    # Validation monitoring is per call. Do not compare its loss with a
    # different validation set, or with training loss after a resumed fit.
    St[FS_STALL] = Int64(0 if validation else monitor.stall)
    St[FS_BEST] = bitcast[DType.int64](Float64(1.0e308) if validation else monitor.best)
    St[FS_PW1] = _f32_word(st.pw1)
    St[FS_PW2] = _f32_word(st.pw2)
    St[FS_MU] = _f32_word(st.mu_prod)
    St[FS_OPT] = Int64(cfg.opt.kind)
    St[FS_NPARAMS] = Int64(np_)
    St[FS_SEED_LO] = Int64(cfg.seed & UInt64(0xFFFFFFFF))
    St[FS_SEED_HI] = Int64(cfg.seed >> 32)
    St[FS_PL_LR] = bitcast[DType.int64](plateau.lr) if plateau_on else Int64(0)
    St[FS_PL_BEST] = bitcast[DType.int64](plateau.best) if plateau_on else Int64(0)
    St[FS_PL_BAD] = Int64(plateau.bad) if plateau_on else Int64(0)
    St[FS_PL_COOL] = Int64(plateau.cool) if plateau_on else Int64(0)
    # lane stream-fits: the source part (the consumed position and the source's binding words) of a streamed fit;
    # all 0 after a fit on arrays
    comptime if streamed:
        source_part_store(batch_source_fit_saved[E.IS_DEVICE](stream[]), St + FS_SOURCE)
        var out = IP64(unsafe_from_address=source_out)
        out[0] = Int64(stream[].epoch)
        out[1] = Int64(stream[].consumed)
    else:
        source_part_store(ProgramTrainingSource(), St + FS_SOURCE)
    lamb.release(ex)
    pg_lamb.release(ex)  # nn-blocks-pgseq
    tape.release(ex)
    epoch_tape.release(ex)
    _ = word^
    _ = pg_rates^
    _ = pg_step_lrs^
    _ = pg_state^
    return n_iter


# nn-losses-2 begin
def model_sequence_output(spec: ModelSpec) -> Bool:
    """True when the model ends in a log_softmax whose per-sample output has more than one axis (a sequence of
    log-probabilities, ctc's input)."""
    var last = spec.n_layers() - 1
    return spec.version == SPEC2_VERSION and is_nn_log_softmax(spec.kinds[last]) and spec.out_shape[last].ndim > 1


def model_ctc_dims(spec: ModelSpec) raises -> Tuple[Int, Int]:
    """(T, C) of a ctc model's output: a final log_softmax over the classes of (length, classes) samples."""
    var last = spec.n_layers() - 1
    if spec.version != SPEC2_VERSION or not is_nn_log_softmax(spec.kinds[last]) or spec.out_shape[last].ndim != 2:
        raise Error("model fit: ctc takes (length, classes) log-probabilities per sample: end the model with "
                    + "log_softmax() over (length, classes) samples")
    var o = spec.out_shape[last]
    if log_softmax_axis(spec.xargs[SPEC2_ARGS * last], o) != 1:
        raise Error("model fit: ctc takes log-probabilities over the classes: the final log_softmax must be over the "
                    + "last axis (dim=-1)")
    if o.d[1] < 2:
        raise Error("model fit: ctc needs at least two classes (the blank and one label)")
    return (o.d[0], o.d[1])
# nn-losses-2 end


def model_sequence_steps(spec: ModelSpec, operation: String) raises -> Int:
    """The padded sequence length T the model's recurrent layers (and multihead_attention's key_padding_mask) read,
    or 0 when it has none; refuses a model whose layers read sequences of different lengths. (Split out of
    model_lengths_check by lane stream-fits: Sequential.fit(source=) pads every batch to it.)"""
    var T = 0
    for i in range(spec.n_layers()):  # driver-loop: over layers (description metadata)
        if spec.version == SPEC2_VERSION and spec.kinds[i] == LAYER_RECURRENT:
            var t = spec.in_shape[i].d[0]
            if T != 0 and t != T:
                raise Error("ValueError: " + operation + ": lengths need every recurrent layer to read sequences "
                            + "of one length (" + String(T) + " and " + String(t) + " here)")
            T = t
    # nn-attention begin: multihead_attention's key_padding_mask reads the lengths (of the model input)
    for i in range(spec.n_layers()):  # driver-loop: over layers (description metadata)
        if spec.version == SPEC2_VERSION:
            var t = nn_attention_key_padding_length(spec.kinds[i], spec.xargs, i, spec.sample)
            if t != 0:
                if T != 0 and t != T:
                    raise Error("ValueError: " + operation + ": lengths need the recurrent layers and the "
                                + "key_padding_mask to read sequences of one length (" + String(T) + " and "
                                + String(t) + " here)")
                T = t
    # nn-attention end
    return T


def model_lengths_check(spec: ModelSpec, lengths: FP, N: Int, operation: String) raises:
    """Admission of a Sequential `lengths` argument (lane seq-blocks-3): the model has a recurrent layer, all its
    recurrent layers read sequences of one padded length T, and each of the N float32 lengths is a whole number
    from 1 to T. One comparison a row on the host, before any launch (as the recurrent estimators' admission,
    sequence/ops_impl.mojo rnn_lengths_of)."""
    var T = model_sequence_steps(spec, operation)
    if T == 0:
        raise Error("ValueError: " + operation + ": lengths apply to the recurrent layers (rnn, lstm, gru); this "
                    + "model has none")
    for r in range(N):  # admission: one comparison per row on the host, before any launch
        var v = lengths.unsafe_load(r)
        if v != v or v < Float32(1.0) or v > Float32(T) or Float32(Int(v)) != v:
            raise Error("ValueError: " + operation + ": every length must be a whole number from 1 to the "
                        + "padded sequence length " + String(T))


def model_state_ranges(spec: ModelSpec) -> List[Int]:
    """The parameter vector's state ranges, (offset, words) pairs: words a forward writes and the
    optimizer never updates (a batch norm's running statistics). Empty for most models."""
    var out = List[Int]()
    if spec.version != SPEC2_VERSION:
        return out^
    for i in range(spec.n_layers()):  # driver-loop: over layers (description metadata)
        var k = spec.kinds[i]
        if is_block(k) and spec.param_off[i] >= 0:
            var r = bn_state(k, spec.xargs, i)
            if r[1] > 0:
                out.append(spec.param_off[i] + r[0])
                out.append(r[1])
    return out^


def _predict_split_same(spec: ModelSpec, rows_a: Int, rows_b: Int) -> Bool:
    """True when every Linear of `spec` takes the same split-K decision at
    rows_a and rows_b output rows (rows <= 0: no such chunk, always same)."""
    if rows_a <= 0 or rows_b <= 0:
        return True
    for i in range(spec.n_layers()):  # driver-loop: over layers (shape rule only, no data)
        if spec.kinds[i] != LAYER_LINEAR:
            continue
        var fi = spec.in_width[i]
        var fo = spec.arg0[i]
        if gemm_split_shape(rows_a, fo, fi) != gemm_split_shape(rows_b, fo, fi):
            return False
    return True


def predict_chunk_rows(spec: ModelSpec, N: Int, chunk: Int) -> Int:
    """The rows model_predict runs a chunk: the most rows whose inference
    scratch fits PREDICT_WHOLE_SET_BYTES (all N for most models) when no
    row's GEMM fold can move, else the caller's `chunk` (capped at N; see
    PREDICT_WHOLE_SET_BYTES above)."""
    var bs = chunk if chunk < N else N
    if bs >= N or spec.version == SPEC2_VERSION:
        return bs
    # float32 words a row: every layer's output (absorbed activations
    # counted anyway), LayerNorm's mean and rstd, the output, the transform
    # output and the (N, 2) probabilities (an over-estimate, never under)
    var per_row = 4 * spec.out_features()
    for i in range(spec.n_layers()):  # driver-loop: over layers (scratch estimate, no data)
        per_row += spec.out_width[i] + 2
    var cap = PREDICT_WHOLE_SET_BYTES // (4 * per_row)
    if cap <= bs:
        return bs
    var c = N if cap >= N else cap
    var old_tail = N - (N // bs) * bs
    var new_tail = N - (N // c) * c
    if not (
        _predict_split_same(spec, bs, c)
        and _predict_split_same(spec, bs, new_tail)
        and _predict_split_same(spec, bs, old_tail)
    ):
        return bs
    return c


def model_predict[E: Exec](
    mut ex: E, spec: ModelSpec, X: CastIn, N: Int, Pin: FP, dst: FP, chunk: Int, transform: Int,
    classes_addr: Int = 0, n_classes: Int = 0, class_itemsize: Int = 0, lengths: Optional[FP] = None,
) raises:
    """dst (N, out) = the model's output on X (N, n_inputs), optionally
    through a row softmax (OUT_SOFTMAX) or an elementwise sigmoid
    (OUT_SIGMOID); OUT_PROBA2 (one output) writes the (N, 2) [1 - p, p] of
    the sigmoid p into dst instead; OUT_ARGMAX (the first-max-wins row
    argmax) and OUT_THRESHOLD (one output, sigmoid p > 0.5) write N float32
    class codes. Dropout is the identity. Rows run in chunks of `chunk`
    so the tape's scratch stays bounded.

    With `classes_addr` (OUT_ARGMAX / OUT_THRESHOLD; lane migrate-neural-2)
    the codes stay on the executor and `gather_labels_device` writes the
    N labels classes[code] (`n_classes` items of `class_itemsize` bytes) to
    `dst`: the same launches, and only the labels come down.

    `lengths`: N float32 sequence lengths on the host for the recurrent
    layers (as model_fit_state's), uploaded once; each chunk reads its rows'."""
    if N < 1 or N >= MODEL_MAX_ROWS or chunk < 1:
        raise Error("model predict: the number of samples must be between 1 and 2^24 - 1 and chunk >= 1")
    if transform < OUT_RAW or transform > OUT_THRESHOLD:
        raise Error("model predict: unknown output transform " + String(transform))
    if (transform == OUT_PROBA2 or transform == OUT_THRESHOLD) and spec.out_features() != 1:
        raise Error("model predict: binary probabilities and labels need a model with one output")
    var D = spec.n_inputs
    var O = spec.out_features()
    var bs = predict_chunk_rows(spec, N, chunk)
    if X.rows != N or X.cols != D:
        raise Error("model predict: X must be (n_samples, " + String(D) + ")")
    var dX = ex.alloc(N * D)
    upload_cast(ex, dX, X)  # the caller's array as it is (autograd/cast_ops.mojo)
    var dLen = dummy_ptr()
    if lengths:
        model_lengths_check(spec, lengths.value(), N, "predict")
        dLen = ex.alloc(N)
        ex.upload(dLen, lengths.value(), N)
    var P = ex.alloc(spec.n_params)
    ex.upload(P, Pin, spec.n_params)
    var dout = ex.alloc(N * O)
    var tape = Tape()
    var topology = _ModelForwardTopology(spec)
    var b0 = 0
    while b0 < N:  # driver-loop: over row chunks (bounded tape scratch)
        var B = bs if b0 + bs <= N else N - b0
        tape.reset()
        var x = tape.variable(ex, dX + b0 * D, B, D, False)
        var blens = Optional[FP]()
        if lengths:
            blens = Optional[FP](dLen + b0)
        var out = _model_forward_prepared(tape, ex, spec, P, P, x, False, UInt64(0), 0, topology, blens)
        if transform == OUT_SOFTMAX:
            out = tape.softmax(ex, out)
        elif transform == OUT_SIGMOID or transform == OUT_PROBA2 or transform == OUT_THRESHOLD:
            out = tape.sigmoid(ex, out)
        _copy(ex, dout + b0 * O, tape.value(out), B * O)
        b0 += B
    if transform == OUT_PROBA2:
        var two = ex.alloc(N * 2)
        var q = Args()
        q.p0 = dout
        q.p1 = two
        ex.launch[OP_PROBA2](q, N * 2)
        ex.sync()
        ex.download(dst, two, N * 2)
    elif transform == OUT_ARGMAX or transform == OUT_THRESHOLD:
        var codes = ex.alloc(N)
        var q = Args()
        q.p0 = dout
        q.p1 = codes
        q.i0 = O
        q.f0 = Float32(0.5)
        if transform == OUT_ARGMAX:
            ex.launch[OP_AGM_ROW_ARGMAX](q, N)
        else:
            ex.launch[OP_AGM_THRESHOLD](q, N)
        if classes_addr != 0:
            gather_labels_device(ex, codes, N, classes_addr, n_classes, class_itemsize, Int(dst))
        else:
            ex.sync()
            ex.download(dst, codes, N)
    else:
        ex.sync()
        ex.download(dst, dout, N * O)
    tape.release(ex)
