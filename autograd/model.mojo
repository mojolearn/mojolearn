# SPDX-License-Identifier: Apache-2.0
"""A user-defined model, trained through the tape from a flat description.

Python assembles the description (an integer program and a float table,
docs/plans/AUTOGRAD_API.md "Model description"); `parse_model` validates it
once per call and `model_fit` / `model_predict` interpret it. Each batch
records the forward graph on the tape, differentiates the loss and runs one
optimizer step; every number is computed by executor launches (parallel, with
the tape's fixed blocked folds), so the bits are the same on Apple, NVIDIA
and AMD. Host loops run over layers, batches and epochs only.
"""
from std.sys.compile import is_defined
from sequence.exec_trait import Exec
from sequence.ops import (
    FP, Args, OP_COPY, OP_FILL, OP_GATHER_ROWS, OP_MLP_PERM, OP_ONE_HOT, OP_PROBA2, OP_SEQ_IOTA,
    OPT_ADAM, OPT_ADAMW, OPT_SGD,
)
from sequence.mlp import mlp_epoch_key, mlp_perm_args
from sequence.recurrent import OptConfig, OptState, opt_step
from checks.numerics import ftz, identical_div, identical_pow64, identical_sqrt
from autograd.tape import Tape, Tensor
from autograd.ops import RELU, SIGMOID, TANH, SOFTMAX
from autograd.layers import NeuralRules, layer_norm
from autograd.model_ops import OP_AGM_INIT_UNIFORM, OP_AGM_ROW_ARGMAX, OP_AGM_THRESHOLD

# Lane ag-data adds `Tape.dropout`; lane ag-forward adds the tape's
# inference mode. Until they land these stay off and the model trainer does
# not reach either (a Dropout layer is refused at fit time with a clear error;
# prediction records a plain forward with gradient-free leaves).
comptime HAS_DROPOUT = is_defined["MOJOLEARN_AUTOGRAD_HAS_DROPOUT"]()
comptime HAS_INFERENCE = is_defined["MOJOLEARN_AUTOGRAD_HAS_INFERENCE"]()

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
# -D MOJOLEARN_AUTOGRAD_MODEL_UNFUSED keeps the composed nodes, for checks.
comptime MODEL_UNFUSED = is_defined["MOJOLEARN_AUTOGRAD_MODEL_UNFUSED"]()

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

comptime LOSS_MSE = 0
comptime LOSS_CROSS_ENTROPY = 1
comptime LOSS_BCE_WITH_LOGITS = 2
comptime LOSS_HUBER = 3

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


@fieldwise_init
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
    var huber_delta: Float32
    var init: Bool


def check_fit(spec: ModelSpec, cfg: FitConfig, N: Int) raises:
    """User-facing refusals of a fit's options against its model."""
    var O = spec.out_features()
    var last = spec.kinds[spec.n_layers() - 1]
    if N < 1 or N >= MODEL_MAX_ROWS:
        raise Error("model fit: the number of samples must be between 1 and 2^24 - 1")
    if cfg.loss < LOSS_MSE or cfg.loss > LOSS_HUBER:
        raise Error("model fit: unknown loss code " + String(cfg.loss))
    if cfg.target < TARGET_DENSE or cfg.target > TARGET_CODE_VALUES:
        raise Error("model fit: unknown target code " + String(cfg.target))
    if cfg.target == TARGET_CODES and cfg.loss != LOSS_CROSS_ENTROPY:
        raise Error("model fit: class-code targets need the cross_entropy loss")
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
    var kind = cfg.opt.kind
    if kind != OPT_SGD and kind != OPT_ADAM and kind != OPT_ADAMW:
        raise Error("model fit: the optimizer must be sgd, momentum, adam or adamw")
    if not (cfg.lr > Float32(0)):
        raise Error("model fit: the learning rate must be positive")
    if cfg.batch < 1 or cfg.epochs < 1:
        raise Error("model fit: batch size and epochs must be at least 1")
    if cfg.sched < SCHED_CONSTANT or cfg.sched > SCHED_EXPONENTIAL:
        raise Error("model fit: unknown learning-rate schedule code " + String(cfg.sched))
    if cfg.sched != SCHED_CONSTANT and not (cfg.sched_gamma > Float32(0)):
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
            comptime if MODEL_UNFUSED:
                h = tape.matmul(ex, h, w)
                if spec.arg1[i] == 1:
                    var b = _param(tape, ex, P, Gr, off + fi * fo, 1, fo, train)
                    h = tape.add(ex, h, b)
            else:
                var bias: Optional[Tensor] = None
                if spec.arg1[i] == 1:
                    bias = Optional[Tensor](_param(tape, ex, P, Gr, off + fi * fo, 1, fo, train))
                var act = _absorbed_activation(spec, i)
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


def _absorbed_activation(spec: ModelSpec, i: Int) -> Int:
    """The tape activation code (RELU, SIGMOID, TANH or SOFTMAX) of the
    activation layer right after Linear layer i, when tape.linear can take
    it into the same node; 0 when there is none or a Residual layer reads
    layer i's pre-activation output."""
    if i + 1 >= spec.n_layers():
        return 0
    for j in range(i + 1, spec.n_layers()):  # driver-loop: over layers
        if spec.kinds[j] == LAYER_RESIDUAL and spec.arg0[j] == i:
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
    comptime if MODEL_UNFUSED:
        return False
    return cfg.loss == LOSS_CROSS_ENTROPY and cfg.target == TARGET_CODES


def model_loss[E: Exec](
    mut tape: Tape, mut ex: E, cfg: FitConfig, pred: Tensor, target: Tensor,
) raises -> Tensor:
    """The batch loss, a mean over the batch (mse, bce_with_logits and huber:
    over every element; cross_entropy: over rows). bce_with_logits and huber
    are compositions of tape operations, so their folds are the tape's
    blocked folds. (They move to autograd/losses.mojo when lane ag-loss
    lands.)"""
    if cfg.loss == LOSS_MSE:
        return tape.mse_loss(ex, pred, target)
    if cfg.loss == LOSS_CROSS_ENTROPY:
        if _fused_codes(cfg):
            # target holds the (B, 1) codes as float values
            return tape.softmax_cross_entropy(ex, pred, target)
        return tape.cross_entropy(ex, pred, target)
    if cfg.loss == LOSS_BCE_WITH_LOGITS:
        # max(x, 0) - x y + log(1 + exp(-|x|)); its gradient is sigmoid(x) - y
        # on either side of 0, and no exp overflows.
        var pos = tape.relu(ex, pred)
        var neg = tape.relu(ex, tape.scale(ex, pred, Float32(-1)))
        var absx = tape.add(ex, pos, neg)
        var one = tape.scalar(ex, Float32(1))
        var soft = tape.log(ex, tape.add(ex, tape.exp(ex, tape.scale(ex, absx, Float32(-1))), one))
        var per = tape.add(ex, tape.subtract(ex, pos, tape.multiply(ex, pred, target)), soft)
        return tape.mean(ex, per)
    # huber(delta): q = min(|r|, delta), over = |r| - q;
    # 0.5 q^2 + delta over (quadratic inside delta, linear outside)
    var r = tape.subtract(ex, pred, target)
    var a = tape.add(ex, tape.relu(ex, r), tape.relu(ex, tape.scale(ex, r, Float32(-1))))
    var delta = tape.scalar(ex, cfg.huber_delta)
    var over = tape.relu(ex, tape.subtract(ex, a, delta))
    var q = tape.subtract(ex, a, over)
    var per = tape.add(ex, tape.scale(ex, tape.multiply(ex, q, q), Float32(0.5)),
                       tape.scale(ex, over, cfg.huber_delta))
    return tape.mean(ex, per)


def model_lr(cfg: FitConfig, epoch: Int) -> Float32:
    """The epoch's learning rate (a per-epoch scalar, computed in Mojo)."""
    if cfg.sched == SCHED_CONSTANT:
        return cfg.lr
    var k = epoch // cfg.sched_step if cfg.sched == SCHED_STEP else epoch
    if k == 0:
        return cfg.lr
    return Float32(Float64(cfg.lr) * identical_pow64(Float64(cfg.sched_gamma), Float64(k)))


# ---------------------------------------------------------------- fit / predict
def model_fit[E: Exec](
    mut ex: E, spec: ModelSpec, cfg: FitConfig, X: FP, Y: FP, N: Int, Pio: FP, curve: FP,
) raises -> Int:
    """Train the model on X (N, n_inputs) and Y; returns the epochs run.

    Y is (N, out) float32 for TARGET_DENSE, or N int32 class codes (their
    bits in float32 words) for TARGET_CODES (one-hot target) and
    TARGET_CODE_VALUES (the codes as the (N, 1) float target). Pio holds the starting
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
    """
    check_fit(spec, cfg, N)
    var D = spec.n_inputs
    var O = spec.out_features()
    var np_ = spec.n_params
    var bs = cfg.batch if cfg.batch < N else N
    var dX = ex.alloc(N * D)
    ex.upload(dX, X, N * D)
    # the batch target's width: the outputs, or 1 for the class codes of the
    # fused cross-entropy (no one-hot target)
    var TW = 1 if _fused_codes(cfg) else O
    var dY = ex.alloc(N * TW)
    if cfg.target == TARGET_CODES or cfg.target == TARGET_CODE_VALUES:
        # the codes' int32 bits travel in float32 words (op_one_hot reads them)
        var dC = ex.alloc(N)
        ex.upload(dC, Y, N)
        var oh = Args()
        oh.p0 = dC
        oh.p1 = dY
        oh.i0 = TW
        oh.i1 = 1 if (cfg.target == TARGET_CODE_VALUES or TW == 1) else 0
        ex.launch[OP_ONE_HOT](oh, N * TW)
    else:
        ex.upload(dY, Y, N * O)
    var P = ex.alloc(np_)
    if cfg.init:
        model_init_params(ex, spec, P, cfg.seed)
    else:
        ex.upload(P, Pio, np_)
    var Gr = ex.alloc(np_)
    var s1 = ex.alloc(np_)
    var s2 = ex.alloc(np_)
    var s3 = ex.alloc(np_)
    var didx = ex.alloc(N)
    var xb = ex.alloc(bs * D)
    var yb = ex.alloc(bs * TW)
    var n_batches = (N + bs - 1) // bs
    var dloss = ex.alloc(n_batches)
    var dcurve = ex.alloc(cfg.epochs)
    var tape = Tape()
    var epoch_tape = Tape()
    var st = OptState()
    var word = List[Float32](length=1, fill=Float32(0))
    var inv_n = ftz(identical_div(Float32(1), Float32(N)))
    var best = Float64(1.0e308)
    var stall = 0
    var n_iter = 0
    var step = 0
    for epoch in range(cfg.epochs):  # driver-loop: over epochs
        if cfg.shuffle:
            ex.launch[OP_MLP_PERM](mlp_perm_args(N, mlp_epoch_key(cfg.seed, epoch), didx), N)
        elif epoch == 0:
            var ia = Args()
            ia.p0 = didx
            ex.launch[OP_SEQ_IOTA](ia, N)
        var lr = model_lr(cfg, epoch)
        for bi in range(n_batches):  # driver-loop: over batches
            var off = bi * bs
            var B = bs if off + bs <= N else N - off
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
            var out = model_forward(tape, ex, spec, P, Gr, x, True, cfg.seed, step)
            var loss = model_loss(tape, ex, cfg, out, target)
            tape.backward_scalar_with_rules(ex, loss, NeuralRules())
            # the batch's loss times its rows, for the epoch's weighted mean
            var weighted = tape.scale(ex, loss, Float32(B))
            _copy(ex, dloss + bi, tape.value(weighted), 1)
            step += 1
            opt_step(ex, cfg.opt, st, step, lr, P, Gr, s1, s2, s3, np_)
        epoch_tape.reset()
        var parts = epoch_tape.variable(ex, dloss, 1, n_batches, False)
        var mean = epoch_tape.scale(ex, epoch_tape.sum(ex, parts), inv_n)
        _copy(ex, dcurve + epoch, epoch_tape.value(mean), 1)
        n_iter = epoch + 1
        if cfg.patience > 0:
            ex.sync()
            ex.download(FP(unsafe_from_address=Int(word.unsafe_ptr())), dcurve + epoch, 1)
            var value = Float64(word[0])
            if value > best - cfg.tol:
                stall += 1
            else:
                stall = 0
            if value < best:
                best = value
            if stall >= cfg.patience:
                break
    ex.sync()
    ex.download(Pio, P, np_)
    ex.download(curve, dcurve, n_iter)
    tape.release(ex)
    epoch_tape.release(ex)
    _ = word^
    return n_iter


def model_predict[E: Exec](
    mut ex: E, spec: ModelSpec, X: FP, N: Int, Pin: FP, dst: FP, chunk: Int, transform: Int,
) raises:
    """dst (N, out) = the model's output on X (N, n_inputs), optionally
    through a row softmax (OUT_SOFTMAX) or an elementwise sigmoid
    (OUT_SIGMOID); OUT_PROBA2 (one output) writes the (N, 2) [1 - p, p] of
    the sigmoid p into dst instead; OUT_ARGMAX (the first-max-wins row
    argmax) and OUT_THRESHOLD (one output, sigmoid p > 0.5) write N float32
    class codes. Dropout is the identity. Rows run in chunks of `chunk`
    so the tape's scratch stays bounded."""
    if N < 1 or N >= MODEL_MAX_ROWS or chunk < 1:
        raise Error("model predict: the number of samples must be between 1 and 2^24 - 1 and chunk >= 1")
    if transform < OUT_RAW or transform > OUT_THRESHOLD:
        raise Error("model predict: unknown output transform " + String(transform))
    if (transform == OUT_PROBA2 or transform == OUT_THRESHOLD) and spec.out_features() != 1:
        raise Error("model predict: binary probabilities and labels need a model with one output")
    var D = spec.n_inputs
    var O = spec.out_features()
    var bs = chunk if chunk < N else N
    var dX = ex.alloc(N * D)
    ex.upload(dX, X, N * D)
    var P = ex.alloc(spec.n_params)
    ex.upload(P, Pin, spec.n_params)
    var dout = ex.alloc(N * O)
    var tape = Tape()
    var b0 = 0
    while b0 < N:  # driver-loop: over row chunks (bounded tape scratch)
        var B = bs if b0 + bs <= N else N - b0
        tape.reset()
        comptime if HAS_INFERENCE:
            tape.no_grad()  # depends on lane ag-forward (inference mode)
        var x = tape.variable(ex, dX + b0 * D, B, D, False)
        var out = model_forward(tape, ex, spec, P, P, x, False, UInt64(0), 0)
        if transform == OUT_SOFTMAX:
            out = tape.softmax(ex, out)
        elif transform == OUT_SIGMOID or transform == OUT_PROBA2 or transform == OUT_THRESHOLD:
            out = tape.sigmoid(ex, out)
        _copy(ex, dout + b0 * O, tape.value(out), B * O)
        comptime if HAS_INFERENCE:
            tape.record()
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
        ex.sync()
        ex.download(dst, codes, N)
    else:
        ex.sync()
        ex.download(dst, dout, N * O)
    tape.release(ex)
