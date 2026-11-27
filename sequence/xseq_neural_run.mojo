# SPDX-License-Identifier: Apache-2.0
"""RUN half of the sequence-expansion neural estimators: the recurrent
models (LSTM/GRU/RNN regressors and classifiers) and the multi-layer
perceptrons (`bindings/xseq_neural_native.mojo` is the UNPACK/PACK
boundary; lane doorsplit-cnn-xseq, 2026-11-11).

Typed values only: the descriptor entry of the model's connected image and
the address of the admitted provider's call workspace (`XseqNeuralEntry`; the
boundary unwraps both from the provider it admitted), address words, shape
words, integer and Float64 scalars. Here: the cell, layout and schedule
words, the optimizer slots and their refusals, the seeded uniform draws, the
fit and predict descriptor calls with their result-count refusals, the
argmax, threshold and curve-minimum outputs, and the fitted-layer copies.

GIL: RUN never touches the GIL. The boundary releases it around each RUN
call that reaches a native entry, where `neural_descriptor_execute` released
it before the split; the plain word copies run with it held, as before.
"""
from std.atomic import Atomic, Ordering
from std.math import sqrt
from std.memory import unsafe_memcpy
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_leased_call
from std.sys.compile import is_defined
from sequence.lengths import recurrent_length_valid
from sequence.ops_impl import net_of, rnn_lengths_of
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from sequence.neural_model_dispatch import (
    RNN_FIT, RNN_PREDICT, MLP_FIT, MLP_PREDICT, MODEL_ARGMAX, MODEL_MIN, MODEL_THRESHOLD, MODEL_UNIFORM,
)

comptime XseqRunF32P = MutPointer[Float32, MutUntrackedOrigin]


# ---------------------------------------------------------------- shape words

def xseq_run_shape_text(shape: List[Int]) -> String:
    """`str(shape)` of a tuple of ints: `()`, `(5,)`, `(2, 3)`."""
    if len(shape) == 1:
        return "(" + String(shape[0]) + ",)"
    var text = String("(")
    for i in range(len(shape)):  # driver-loop: over shape words
        if i:
            text += ", "
        text += String(shape[i])
    return text + ")"


def xseq_run_size(shape: List[Int]) -> Int:
    var n = 1
    for i in range(len(shape)):  # driver-loop: over shape words
        n *= shape[i]
    return n


def xseq_run_as_f32_check(shape: List[Int], ndim: Int, name: String) raises:
    """`_buffer.as_f32_c`'s rank and emptiness refusals; `ndim` 0 accepts any
    rank."""
    var rank = len(shape)
    if ndim != 0 and rank != ndim:
        var hint = String("")
        if ndim == 2 and rank < 2:
            hint = (". Reshape your data either using array.reshape(-1, 1) if your data "
                    + "has a single feature or array.reshape(1, -1) if it contains a single sample.")
        raise Error("mojolearn: " + name + " must be " + String(ndim) + "-D, got " + String(rank) + "-D shape "
                    + xseq_run_shape_text(shape) + hint)
    if xseq_run_size(shape) == 0:
        var what = String("feature(s)") if rank == 2 and shape[0] != 0 else String("sample(s)")
        raise Error("mojolearn: " + name + " is empty, shape " + xseq_run_shape_text(shape) + ": Found array with 0 "
                    + what + " (shape=" + xseq_run_shape_text(shape) + ") while a minimum of 1 is required.")


# ---------------------------------------------------------------- native entry

@fieldwise_init
struct XseqNeuralEntry(ImplicitlyCopyable, Movable):
    """The model descriptor entry of the connected image and the address of
    the admitted provider's `NativeCallWorkspace` (0: a private temporary
    workspace per call). The boundary keeps the provider referenced across
    every call that uses the address."""
    var call: NativeDescriptorCall
    var workspace: Int


def xseq_neural_run_call(entry: XseqNeuralEntry, operation: Int, addresses: List[Int], ints: List[Int],
                         floats: List[Float64]) raises -> Int:
    """One synchronous call of the image's model descriptor entry under the
    provider's workspace lease (`neural_descriptor_execute` did the same
    before the door split): the ABI output and its error storage are the
    provider's retained scratch, reused across calls; a nested or concurrent
    call that finds it busy gets a private one. The provider's call counters
    are not bumped (nothing product-side reads them). The caller keeps every
    addressed buffer alive and has released the GIL."""
    return native_descriptor_leased_call(entry.call, entry.workspace, operation, addresses, ints, floats)


def xseq_run_uniform(entry: XseqNeuralEntry, address: Int, n: Int, seed: UInt64, offset: Int, low: Float64,
                     high: Float64) raises -> Int:
    """`XseqStream.fill_uniform_native`: n float32 draws U(low, high) at
    `address` from the counter-based stream; the advanced counter."""
    if n:
        var scalars: List[Float64] = [low, high]
        _ = xseq_neural_run_call(entry, MODEL_UNIFORM, [address],
                                 [n, Int(seed & 0xFFFFFFFF), Int(seed >> 32), offset], scalars)
    return offset + n


def xseq_run_argmax(entry: XseqNeuralEntry, scores: Int, codes: Int, rows: Int, cols: Int) raises:
    """First-max-wins row argmax of a C-ordered float32 block into int64
    codes (`argmax_rows_f32`)."""
    if rows:
        _ = xseq_neural_run_call(entry, MODEL_ARGMAX, [scores, codes], [rows, cols], List[Float64]())


def xseq_run_threshold(entry: XseqNeuralEntry, scores: Int, codes: Int, rows: Int) raises:
    """p > 0.5 per row (`threshold_labels_i64`; NaN takes class 0)."""
    if rows:
        _ = xseq_neural_run_call(entry, MODEL_THRESHOLD, [scores, codes], [rows], List[Float64]())


def xseq_run_curve_min(entry: XseqNeuralEntry, curve: Int, count: Int) raises -> Float64:
    """The first-wins minimum of `count` float32 curve words (`reduce_stat` 0)."""
    var best = List[Float64](length=1, fill=0)
    _ = xseq_neural_run_call(entry, MODEL_MIN, [curve, Int(best.unsafe_ptr())], [count], List[Float64]())
    var result = best[0]
    _ = best^
    return result


# ---------------------------------------------------------------- optimizer slots

@fieldwise_init
struct XseqOptimizer(Copyable, Movable):
    """The binding's optimizer kind, flag word and six float slots, as the
    native words the RNN_FIT descriptor reads."""
    var kind: Int
    var flags: Int
    var floats: List[Float64]


def xseq_run_opt_sgd(kind: Int, nesterov: Bool, momentum: Float64, weight_decay: Float64,
                     dampening: Float64) raises -> XseqOptimizer:
    if nesterov and (momentum <= 0.0 or dampening != 0.0):
        raise Error("sgd: nesterov needs momentum > 0 and dampening 0 (torch.optim.SGD)")
    return XseqOptimizer(kind, 1 if nesterov else 0, [momentum, 0.0, 0.0, weight_decay, dampening, 0.0])


def xseq_run_opt_adam(label: String, kind: Int, b1: Float64, b2: Float64, eps: Float64,
                      weight_decay: Float64) raises -> XseqOptimizer:
    """adam, adamw and adamax."""
    if not (0.0 <= b1 and b1 < 1.0 and 0.0 <= b2 and b2 < 1.0):
        raise Error(label + ": betas must lie in [0, 1)")
    return XseqOptimizer(kind, 0, [b1, b2, eps, weight_decay, 0.0, 0.0])


def xseq_run_opt_nadam(kind: Int, b1: Float64, b2: Float64, eps: Float64, weight_decay: Float64,
                       momentum_decay: Float64, decoupled: Bool) raises -> XseqOptimizer:
    if momentum_decay < 0.0:
        raise Error("nadam: momentum_decay must be >= 0")
    return XseqOptimizer(kind, 1 if decoupled else 0, [b1, b2, eps, weight_decay, momentum_decay, 0.0])


def xseq_run_opt_lion(kind: Int, b1: Float64, b2: Float64, weight_decay: Float64) -> XseqOptimizer:
    return XseqOptimizer(kind, 0, [b1, b2, 0.0, weight_decay, 0.0, 0.0])


def xseq_run_opt_rmsprop(kind: Int, centered: Bool, alpha: Float64, eps: Float64, weight_decay: Float64,
                         momentum: Float64) -> XseqOptimizer:
    return XseqOptimizer(kind, 1 if centered else 0, [0.0, alpha, eps, weight_decay, momentum, 0.0])


def xseq_run_opt_adagrad(kind: Int, lr_decay: Float64, eps: Float64, weight_decay: Float64,
                         initial: Float64) -> XseqOptimizer:
    return XseqOptimizer(kind, 0, [lr_decay, 0.0, eps, weight_decay, 0.0, initial])


# ---------------------------------------------------------------- recurrent models

def xseq_run_rnn_cell(cell: String, tanh: Bool, relu: Bool) raises -> Int:
    """`_RecurrentBase._cell()`: the binding's cell code; `tanh` / `relu` are
    the plain RNN's nonlinearity tests (read only for `cell == "rnn"`)."""
    if cell == "rnn":
        if tanh:
            return 0
        if relu:
            return 1
        raise Error("nonlinearity must be 'tanh' or 'relu' (nn.RNN)")
    return 2 if cell == "lstm" else 3


def xseq_run_rnn_gates(cell: Int) -> Int:
    return 4 if cell == 2 else (3 if cell == 3 else 1)


def xseq_run_rnn_rank_check(rank: Int) raises:
    if rank != 3:
        raise Error("X must be 3-D: (n_samples, n_timesteps, n_features)")


def xseq_run_rnn_dims_check(n: Int, steps_t: Int) raises:
    if n < 1 or steps_t < 1:
        raise Error("X needs at least one sample and one time step")


def xseq_run_rnn_param_count(gates: Int, d: Int, hidden: Int, layers: Int, outputs: Int) -> Int:
    """The parameters in layout order: per layer w_ih, w_hh and both biases,
    then the head."""
    var total = 0
    for l in range(layers):  # driver-loop: over recurrent layers of the parameter layout
        var din = d if l == 0 else hidden
        total += gates * hidden * din + gates * hidden * hidden + 2 * gates * hidden
    return total + outputs * hidden + outputs


def xseq_run_rnn_bound(hidden: Float64) -> Float64:
    """U(-1/sqrt(H), 1/sqrt(H))."""
    return 1.0 / sqrt(hidden)


@fieldwise_init
struct XseqRnnSchedule(Copyable, Movable):
    var batch: Int
    var steps: Int


def xseq_run_rnn_schedule(batch_size: Int, n: Int, epochs: Int) raises -> XseqRnnSchedule:
    """The minibatch schedule's scalars; each epoch's row order is built by
    rnn_fit."""
    var batch = max(1, min(batch_size, n))
    if epochs * n >= 2147483647:
        raise Error("max_epochs * n_samples must be below 2^31 - 1, got " + String(epochs) + " * " + String(n))
    return XseqRnnSchedule(batch, epochs * ((n + batch - 1) // batch))


@fieldwise_init
struct XseqRnnFitRequest(Copyable, Movable):
    var cell: Int
    var d: Int
    var hidden: Int
    var layers: Int
    var outputs: Int
    var task: Int
    var n: Int
    var steps_t: Int
    var epochs: Int
    var batch: Int
    var shuffle: Bool
    var seed: UInt64


def xseq_run_rnn_fit(entry: XseqNeuralEntry, req: XseqRnnFitRequest, optimizer: XseqOptimizer, x: Int,
                     target: Int, params: Int, losses: Int, rates: Int, lengths: Int = 0,
                     loop_words: Int = 0) raises:
    """`_ints()` (cell, D, H, L, O), then the schedule words, then the
    RNN_FIT call. `lengths`: the address of n float32 sequence lengths, or
    0 (none: the call is the one it was before lengths existed).
    `loop_words`: the address of the fit loop's record words
    (sequence/fit_loop_words.mojo), or 0 (none)."""
    var ip: List[Int] = [req.cell, req.d, req.hidden, req.layers, req.outputs, req.task, req.n, req.steps_t,
                         req.epochs, req.batch, optimizer.kind, optimizer.flags, 1 if req.shuffle else 0,
                         Int(req.seed & 0xFFFFFFFF), Int(req.seed >> 32)]
    var addresses: List[Int] = [x, target, params, losses, rates]
    if lengths != 0 or loop_words != 0:
        addresses.append(lengths)
    if loop_words != 0:
        addresses.append(loop_words)
    _ = xseq_neural_run_call(entry, RNN_FIT, addresses, ip, optimizer.floats)


def xseq_run_rnn_layout(ints: List[Int], params_size: Int) raises -> Int:
    """The fitted network's layout from `_ints()` (cell, D, H, L, O), its
    refusals, and the size check of `params_`; the parameter count."""
    var network = net_of(ints)
    if params_size != network.n_params():
        raise Error("ValueError: recurrent params_ size differs from the fitted network layout")
    return network.n_params()


@fieldwise_init
struct XseqRnnPredictRequest(Copyable, Movable):
    var ints: List[Int]         # `_ints()`: cell, D, H, L, O
    var n_params: Int
    var task: Int
    var n: Int
    var steps_t: Int
    var chunk: Int
    var want_seq: Bool
    var model: Int              # the recorded block's owner word (with `token`)
    var token: Int              # 0: no resident record, upload per call


def xseq_run_rnn_predict(entry: XseqNeuralEntry, req: XseqRnnPredictRequest, x: Int, params: Int, out_words: Int,
                         seq: Int, lengths: Int = 0) raises:
    """RNN_PREDICT and its result-count refusal. `lengths` as rnn_fit's (0:
    none, the ten words it was)."""
    var ip = req.ints.copy()
    ip.append(req.task)
    ip.append(req.n)
    ip.append(req.steps_t)
    ip.append(req.chunk)
    ip.append(1 if req.want_seq else 0)
    if lengths != 0:
        ip.append(lengths)
    var addresses: List[Int] = [x, params, out_words, seq]
    if req.token:
        # the recorded block's name: its device copy stays resident
        addresses.append(req.model)
        addresses.append(req.token)
        addresses.append(req.n_params)
    var written = xseq_neural_run_call(entry, RNN_PREDICT, addresses, ip, List[Float64]())
    if written != req.n * req.ints[4]:
        raise Error("RuntimeError: recurrent prediction returned an invalid result count")


# ---------------------------------------------------------------- multi-layer perceptrons

def xseq_run_mlp_scalars_check(learning_rate_init: Float64, max_iter: Float64, alpha: Float64) raises:
    if learning_rate_init <= 0.0 or max_iter < 1.0 or alpha < 0.0:
        raise Error("learning_rate_init > 0, max_iter >= 1 and alpha >= 0 are required")


def xseq_run_mlp_hidden_check(hidden: List[Int]) raises:
    for i in range(len(hidden)):  # driver-loop: over hidden layer sizes
        if hidden[i] < 1:
            raise Error("hidden_layer_sizes must be > 0")


def xseq_run_mlp_out_name(code: Int) -> String:
    if code == 0:
        return "identity"
    if code == 1:
        return "logistic"
    if code == 2:
        return "tanh"
    if code == 3:
        return "relu"
    return "softmax"


def xseq_run_mlp_sizes(d: Int, hidden: List[Int], o: Int) -> List[Int]:
    """Layer widths: input, hidden..., output."""
    var sizes: List[Int] = [d]
    for i in range(len(hidden)):  # driver-loop: over hidden layer widths
        sizes.append(hidden[i])
    sizes.append(o)
    return sizes^


def xseq_run_mlp_param_count(sizes: List[Int]) -> Int:
    var total = 0
    for l in range(len(sizes) - 1):  # driver-loop: over network layers
        total += sizes[l] * sizes[l + 1] + sizes[l + 1]
    return total


def xseq_run_mlp_glorot(entry: XseqNeuralEntry, base: Int, sizes: List[Int], logistic: Bool, seed: UInt64,
                        offset: Int) raises -> Int:
    """Glorot-uniform per layer, drawn straight into the flat block at `base`
    the entry trains, w then b of each layer, as the packed layout orders
    them; the advanced stream counter."""
    var counter = offset
    var position = 0
    for l in range(len(sizes) - 1):  # driver-loop: over network layers
        var fi = sizes[l]
        var fo = sizes[l + 1]
        var factor = 2.0 if logistic else 6.0
        var bound = sqrt(factor / Float64(fi + fo))
        counter = xseq_run_uniform(entry, base + 4 * position, fi * fo, seed, counter, -bound, bound)
        position += fi * fo
        counter = xseq_run_uniform(entry, base + 4 * position, fo, seed, counter, -bound, bound)
        position += fo
    return counter


@fieldwise_init
struct XseqMlpFitRequest(Copyable, Movable):
    var rows: Int
    var d: Int
    var o: Int
    var activation: Int
    var out_act: Int
    var loss_code: Int
    var solver: Int
    var rate_code: Int
    var nesterov: Bool
    var batch: Int
    var max_iter: Int
    var shuffle: Bool
    var seed: Int
    var n_iter_no_change: Int
    var hidden: List[Int]
    var n_out: Int              # 0: a float target (no class-count word)


def xseq_run_mlp_fit(entry: XseqNeuralEntry, req: XseqMlpFitRequest, floats: List[Float64], x: Int, target: Int,
                     params: Int, curve: Int, loop_words: Int = 0, progress: Int = 0,
                     previous_moments: Int = 0, next_moments: Int = 0, previous_order: Int = 0,
                     next_order: Int = 0) raises -> Int:
    """The descriptor's int words, then MLP_FIT; the epochs run.
    `loop_words`: the address of the fit loop's record words
    (sequence/fit_loop_words.mojo), or 0 (none)."""
    var ip: List[Int] = [req.rows, req.d, req.o, req.activation, req.out_act, req.loss_code, req.solver,
                         req.rate_code, 1 if req.nesterov else 0, req.batch, req.max_iter, 1 if req.shuffle else 0,
                         req.seed, req.n_iter_no_change, len(req.hidden)]
    for i in range(len(req.hidden)):  # driver-loop: over hidden layer widths
        ip.append(req.hidden[i])
    if req.n_out != 0:
        ip.append(1 if req.o > 1 else 2)
    var addresses: List[Int] = [x, target, params, curve]
    if loop_words != 0 or progress != 0:
        addresses.append(loop_words)
    if progress:
        addresses.append(progress)
        addresses.append(previous_moments)
        addresses.append(next_moments)
        addresses.append(previous_order)
        addresses.append(next_order)
    return xseq_neural_run_call(entry, MLP_FIT, addresses, ip, floats)


def xseq_run_mlp_check_order(address: Int, rows: Int) raises:
    """Require each training row exactly once in the saved fast-profile shuffle order."""
    var order = XseqRunF32P(unsafe_from_address=address)
    var workers = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, workers)
    var bad = List[Int](length=workers, fill=0)
    var bp = bad.unsafe_ptr()
    var seen = List[Int32](length=rows, fill=0)
    var marks = seen.unsafe_ptr()
    def task(worker: Int) {imm order, imm rows, imm chunk, imm bp, imm marks}:
        for i in range(worker * chunk, min(rows, (worker + 1) * chunk)):
            var word = order[i]
            if not (word >= 0 and word < Float32(rows)) or Float32(Int(word)) != word:
                bp[worker] = 1
                break
            # The prior count identifies a duplicate even across worker chunks.
            if Atomic.fetch_add[ordering = Ordering.RELAXED](marks + Int(word), Int32(1)) != 0:
                bp[worker] = 2
                break
    host_parallelize(task, workers)
    _ = seen^
    for worker in range(workers):
        if bad[worker] == 2:
            raise Error("ValueError: duplicate row index in saved MLP shuffle order")
        if bad[worker]:
            raise Error("ValueError: invalid row index in saved MLP shuffle order")


def xseq_run_copy_f32(dst: Int, src: Int, count: Int):
    """`count` float32 words from `src` to `dst` (byte addresses)."""
    if count:
        unsafe_memcpy(dest=XseqRunF32P(unsafe_from_address=dst), src=XseqRunF32P(unsafe_from_address=src),
                      count=count)


def xseq_run_mlp_layers_check(layers: Int, intercepts: Int) raises:
    if layers < 1 or intercepts != layers:
        raise Error("ValueError: MLP fitted weights require matching nonempty layer lists")


def xseq_run_mlp_layer_check(weight_shape: List[Int], bias_shape: List[Int], input_width: Int) raises:
    """One admitted fitted layer: a matrix and a vector that continue the
    layout from `input_width`."""
    if len(weight_shape) != 2 or len(bias_shape) != 1:
        raise Error("ValueError: MLP weights must be matrices and biases must be vectors")
    var fo = weight_shape[1]
    if weight_shape[0] != input_width or fo < 1 or bias_shape[0] != fo:
        raise Error("ValueError: MLP fitted parameter shapes do not match the layer layout")


def xseq_run_mlp_pack(dst: Int, weights: List[Int], weight_counts: List[Int], biases: List[Int],
                      bias_counts: List[Int]):
    """Pack the fitted layers, w then b of each layer, into the float32
    block at `dst`."""
    var offset = 0
    for l in range(len(weights)):  # driver-loop: over network layers
        xseq_run_copy_f32(dst + 4 * offset, weights[l], weight_counts[l])
        offset += weight_counts[l]
        xseq_run_copy_f32(dst + 4 * offset, biases[l], bias_counts[l])
        offset += bias_counts[l]


def xseq_run_mlp_output_check(output_width: Int, n_outputs: Int) raises:
    if output_width != n_outputs:
        raise Error("ValueError: MLP output weights differ from n_outputs_")


@fieldwise_init
struct XseqMlpPredictRequest(Copyable, Movable):
    var rows: Int
    var columns: Int
    var n_outputs: Int
    var activation: Int
    var out_act: Int
    var hidden: List[Int]
    var proba2: Bool
    var width: Int              # output columns: 2 for the two-column probability, else n_outputs
    var params_size: Int
    var model: Int              # the recorded block's owner word (with `token`)
    var token: Int              # 0: no resident record, upload per call


def xseq_run_mlp_predict(entry: XseqNeuralEntry, req: XseqMlpPredictRequest, x: Int, params: Int,
                         out_words: Int) raises:
    """MLP_PREDICT (4096-row chunks) and its result-count refusal."""
    var ip: List[Int] = [req.rows, req.columns, req.n_outputs, req.activation, req.out_act, 4096, len(req.hidden)]
    for i in range(len(req.hidden)):  # driver-loop: over hidden layer widths
        ip.append(req.hidden[i])
    if req.proba2:
        ip.append(1)
    var addresses: List[Int] = [x, params, out_words]
    if req.token:
        # the recorded block's name: its device copy stays resident
        addresses.append(req.model)
        addresses.append(req.token)
        addresses.append(req.params_size)
    var written = xseq_neural_run_call(entry, MLP_PREDICT, addresses, ip, List[Float64]())
    if written != req.rows * req.width:
        raise Error("RuntimeError: mlp_predict: this binding predates the two-column probability; rebuild it")


# ---------------------------------------------------------------- recurrent options (lane rnn-parity)

@fieldwise_init
struct XseqRnnFitOptions(Copyable, Movable):
    """The options form of RNN_FIT (sequence/ops_impl.mojo RNN_FIT_OPTION_INTS / _FLOATS and its nine addresses):
    every word resolved once by the boundary. Addresses are 0 when absent."""
    var dirs: Int
    var seq_out: Bool
    var no_bias: Bool
    var sw: Int
    var sw_steps: Bool
    var cw: Int
    var h0: Int
    var c0: Int
    var nv: Int
    var patience: Int
    var restore: Bool
    var vx: Int
    var vy: Int
    var vl: Int
    var vcurve: Int
    var state: Int
    var dropout: Float64
    var clip: Float64
    var tol: Float64


def _b(v: Bool) -> Int:
    return 1 if v else 0


def xseq_run_rnn_fit_options(entry: XseqNeuralEntry, req: XseqRnnFitRequest, optimizer: XseqOptimizer, x: Int,
                             target: Int, params: Int, losses: Int, rates: Int, lengths: Int, loop_words: Int,
                             o: XseqRnnFitOptions) raises:
    """RNN_FIT in its options form: the fifteen words of xseq_run_rnn_fit, then the option words."""
    var ip: List[Int] = [req.cell, req.d, req.hidden, req.layers, req.outputs, req.task, req.n, req.steps_t,
                         req.epochs, req.batch, optimizer.kind, optimizer.flags, 1 if req.shuffle else 0,
                         Int(req.seed & 0xFFFFFFFF), Int(req.seed >> 32),
                         o.dirs, _b(o.seq_out), _b(o.no_bias), _b(o.sw != 0), _b(o.sw_steps), _b(o.cw != 0),
                         _b(o.h0 != 0), _b(o.c0 != 0), o.nv, o.patience, _b(o.restore), _b(o.vl != 0),
                         _b(o.state != 0)]
    var addresses: List[Int] = [x, target, params, losses, rates, lengths, loop_words, o.sw, o.cw, o.h0, o.c0,
                                o.vx, o.vy, o.vl, o.vcurve, o.state]
    var floats = optimizer.floats.copy()
    floats.append(o.dropout)
    floats.append(o.clip)
    floats.append(o.tol)
    _ = xseq_neural_run_call(entry, RNN_FIT, addresses, ip, floats)


def xseq_run_rnn_fit_stream(entry: XseqNeuralEntry, req: XseqRnnFitRequest, optimizer: XseqOptimizer, params: Int,
                            losses: Int, rates: Int, loop_words: Int, o: XseqRnnFitOptions, words: List[Int],
                            saved: Int, position: Int, flags: Int) raises:
    """RNN_FIT in its STREAMED form (lane stream-fits; sequence/ops_impl.mojo rnn_fit_impl): the options form's
    28 words, then the batch source's words (autograd/batch_source.mojo batch_source_fit_words: the kept source's
    address and max_steps; lane w3i11-source-keep) and the stream flags (lane stream-fits-2: RNN_STREAM_BALANCED)
    last; the options form's 16 addresses (X and y 0), the estimator's saved source part (out) and the position
    (out: 3 int64 words, the consumed (epoch, batch) and the steps run). The caller holds the source's owner
    through the call."""
    var ip: List[Int] = [req.cell, req.d, req.hidden, req.layers, req.outputs, req.task, req.n, req.steps_t,
                         req.epochs, req.batch, optimizer.kind, optimizer.flags, 1 if req.shuffle else 0,
                         Int(req.seed & 0xFFFFFFFF), Int(req.seed >> 32),
                         o.dirs, _b(o.seq_out), _b(o.no_bias), _b(o.sw != 0), _b(o.sw_steps), _b(o.cw != 0),
                         _b(o.h0 != 0), _b(o.c0 != 0), o.nv, o.patience, _b(o.restore), _b(o.vl != 0),
                         _b(o.state != 0)]
    for k in range(len(words)):  # small-loop(the source words)
        ip.append(words[k])
    ip.append(flags)
    var addresses: List[Int] = [0, 0, params, losses, rates, 0, loop_words, o.sw, o.cw, o.h0, o.c0,
                                o.vx, o.vy, o.vl, o.vcurve, o.state, saved, position]
    var floats = optimizer.floats.copy()
    floats.append(o.dropout)
    floats.append(o.clip)
    floats.append(o.tol)
    _ = xseq_neural_run_call(entry, RNN_FIT, addresses, ip, floats)


@fieldwise_init
struct XseqRnnPredictOptions(Copyable, Movable):
    """The options form of RNN_PREDICT: directions, every step's head, initial and final states (addresses, 0
    when absent)."""
    var dirs: Int
    var seq_out: Bool
    var h0: Int
    var c0: Int
    var state_h: Int
    var state_c: Int


def xseq_run_rnn_layout_dirs(ints: List[Int], dirs: Int, params_size: Int) raises -> Int:
    """xseq_run_rnn_layout with `dirs` directions (bidirectional)."""
    var network = net_of(ints)
    network.dirs = dirs
    if params_size != network.n_params():
        raise Error("ValueError: recurrent params_ size differs from the fitted network layout")
    return network.n_params()


def xseq_run_rnn_predict_options(entry: XseqNeuralEntry, req: XseqRnnPredictRequest, x: Int, params: Int,
                                 out_words: Int, seq: Int, lengths: Int, o: XseqRnnPredictOptions) raises:
    """RNN_PREDICT in its options form (sequence/ops_impl.mojo RNN_PREDICT_OPTION_INTS). Final states asked
    (o.state_h != 0) is the estimators' `step`, the streaming form: refused for a bidirectional model, whose
    reverse direction at a step reads every later step, which a live call has not seen yet, so no piece of a
    sequence can be scored before the sequence ends (lane rnn-step-bits; predict on the whole sequence is that
    model's answer, and its h_0 / c_0 stay accepted there)."""
    if o.state_h != 0 and o.dirs != 1:
        raise Error(
            "ValueError: step: a bidirectional model cannot stream: its reverse direction reads the steps that come"
            " after each step; call predict on the whole sequence"
        )
    var ip = req.ints.copy()
    ip.append(req.task)
    ip.append(req.n)
    ip.append(req.steps_t)
    ip.append(req.chunk)
    ip.append(1 if req.want_seq else 0)
    ip.append(lengths)
    ip.append(o.dirs)
    ip.append(_b(o.seq_out))
    ip.append(_b(o.h0 != 0))
    ip.append(_b(o.c0 != 0))
    ip.append(_b(o.state_h != 0))
    ip.append(o.h0)
    ip.append(o.c0)
    ip.append(o.state_h)
    ip.append(o.state_c)
    var addresses: List[Int] = [x, params, out_words, seq]
    if req.token:
        addresses.append(req.model)
        addresses.append(req.token)
        addresses.append(req.n_params)
    var written = xseq_neural_run_call(entry, RNN_PREDICT, addresses, ip, List[Float64]())
    if written != req.n * req.ints[4]:
        raise Error("RuntimeError: recurrent prediction returned an invalid result count")


# SABOTAGE ARMS (lane rnn-review-fixes, 2026-11-25; tools/identity_lanes/coverage_sabotage.tsv). Each, forced on,
# restores one defect of before on ONE side, the device route's fit (`_DEVICE`) or the host route's (`_HOST`, an
# explicit device="cpu"); the side is the boundary's route (bindings/xseq_neural_native_door.mojo). The verify lane
# sequence-rnn-padding-labels (tools/identity_lanes/rnn_parity.py) must MOVE or REFUSE on that side under each.
# LENGTHS: fit admits `lengths` only after the class histogram read them (that side's admission is skipped and the
# histogram's own check is compiled out). PADDING: the sequence target's classes are discovered over every step,
# padded ones included, and a padding label of a validation or partial_fit target is refused as unknown.
comptime RNN_REVIEW_SABOTAGE_LENGTHS_DEVICE = is_defined["MOJOLEARN_RNN_REVIEW_SABOTAGE_LENGTHS_DEVICE"]()
comptime RNN_REVIEW_SABOTAGE_LENGTHS_HOST = is_defined["MOJOLEARN_RNN_REVIEW_SABOTAGE_LENGTHS_HOST"]()
comptime RNN_REVIEW_SABOTAGE_PADDING_DEVICE = is_defined["MOJOLEARN_RNN_REVIEW_SABOTAGE_PADDING_DEVICE"]()
comptime RNN_REVIEW_SABOTAGE_PADDING_HOST = is_defined["MOJOLEARN_RNN_REVIEW_SABOTAGE_PADDING_HOST"]()
comptime _RNN_REVIEW_COUNTS_CHECK_LENGTHS = not (RNN_REVIEW_SABOTAGE_LENGTHS_DEVICE or RNN_REVIEW_SABOTAGE_LENGTHS_HOST)


def xseq_run_lengths_check(lengths: Int, n: Int, steps: Int, operation: String) raises:
    """The admission of n float32 sequence lengths at `lengths` (0: none) against `steps` padded steps: each a
    whole number from 1 to `steps`, refused with the recurrent fit's own words (sequence/ops_impl.mojo
    rnn_lengths_of: the one body). The boundary calls it once, before any consumer reads a length (the class
    histogram, the valid-step label encoding, the step weights)."""
    _ = rnn_lengths_of(lengths, n, steps, operation)


def xseq_run_class_counts[dt: DType](codes: Int, n: Int, steps: Int, lengths: Int, k: Int) raises -> List[Int]:
    """How many valid steps hold each of the k class codes (dt: float32 fit codes or int32 label-encoder codes):
    n sequences of `steps` codes each; when `lengths` (n float32 words; 0: none) is given, only each sequence's
    first length steps count. Per-worker histograms over contiguous chunks of sequences, added in worker order
    (integers: exact in any order). A length is checked before its row's codes are read (finite, whole, 1 ..
    steps; lane rnn-review-fixes: it bounds the read): a bad one refuses with the admission's words and no read
    past its sequence happens. Codes are the caller's own, in [0, k)."""
    var p = MutPointer[Scalar[dt], MutUntrackedOrigin](unsafe_from_address=codes)
    var lp = XseqRunF32P(unsafe_from_address=lengths if lengths != 0 else codes)
    var has_lens = lengths != 0
    var tasks = host_predict_task_count_work(max(1, n), max(1, n * steps))
    var chunk = host_predict_chunk(max(1, n), tasks)
    var hist = List[Int](length=tasks * k, fill=0)
    var bad = List[Int](length=tasks, fill=0)
    var hp = hist.unsafe_ptr()
    var bp = bad.unsafe_ptr()
    def task(worker: Int) {imm p, imm lp, imm hp, imm bp, imm n, imm steps, imm k, imm chunk, imm has_lens}:
        for i in range(worker * chunk, min((worker + 1) * chunk, n)):
            var valid = steps
            if has_lens:
                var v = lp[i]
                comptime if _RNN_REVIEW_COUNTS_CHECK_LENGTHS:
                    if not recurrent_length_valid(v, steps):
                        bp[worker] = 1
                        continue
                valid = Int(v)
            for s in range(valid):
                hp[worker * k + Int(p[i * steps + s])] += 1
    host_parallelize(task, tasks)
    for worker in range(tasks):  # worker summaries after the join
        if bad[worker] != 0:
            raise Error("ValueError: fit: every length must be a whole number from 1 to the sequence length "
                        + String(steps))
    var counts = List[Int](length=k, fill=0)
    for worker in range(tasks):  # worker summaries after the join
        for c in range(k):  # small-loop(classes)
            counts[c] += hist[worker * k + c]
    _ = hist^
    _ = bad^
    return counts^


def xseq_run_balanced_class_weights(codes: Int, n: Int, steps: Int, lengths: Int, k: Int) raises -> List[Float64]:
    """class_weight='balanced' (scikit-learn's compute_class_weight): n_rows / (k * count of class c) for each of
    the k classes, over the float32 class codes at `codes` (n sequences of `steps` rows each; with output=
    "sequence" `steps` is T and, when `lengths` (n float32 words) is given, only each sequence's valid steps
    count: xseq_run_class_counts, which checks each length before it bounds a read)."""
    var counts = xseq_run_class_counts[DType.float32](codes, n, steps, lengths, k)
    var total = 0
    for c in range(k):  # small-loop(classes)
        total += counts[c]
    var out = List[Float64]()
    for c in range(k):  # small-loop(classes)
        if counts[c] == 0:
            raise Error("ValueError: class_weight='balanced' needs every class in y")
        out.append(Float64(total) / (Float64(k) * Float64(counts[c])))
    return out^


def xseq_run_remap_codes[T: DType = DType.float32](codes: Int, n: Int, steps: Int, lengths: Int, mapping: List[Int], dst: Int) raises:
    """dst[i T + s] = Float32(mapping[codes[i T + s]]) at the valid steps of n sequences of `steps` int32 codes
    (a label encoding of its own classes, as float32 codes of the kept or fitted classes: the fit's valid-step
    classes, partial_fit, validation targets), and 0 at each padded step (s >= length_i; `lengths`: n float32
    words, admitted by the boundary, or 0 for none): a padded step's label is never read through `mapping`,
    and its code 0 is a class index the loss weighs 0 there (lane rnn-review-fixes). Contiguous chunks of
    sequences across the host's workers."""
    var src = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=codes)
    var out = MutPointer[Scalar[T], MutUntrackedOrigin](unsafe_from_address=dst)
    var lp = XseqRunF32P(unsafe_from_address=lengths if lengths != 0 else dst)
    var has_lens = lengths != 0
    var mp = mapping.unsafe_ptr()
    var tasks = host_predict_task_count_work(max(1, n), max(1, n * steps))
    var chunk = host_predict_chunk(max(1, n), tasks)
    def task(worker: Int) {imm src, imm out, imm lp, imm mp, imm has_lens, imm n, imm steps, imm chunk}:
        for i in range(worker * chunk, min((worker + 1) * chunk, n)):
            var valid = Int(lp[i]) if has_lens else steps
            for s in range(steps):
                if s < valid:
                    out[i * steps + s] = Scalar[T](mp[Int(src[i * steps + s])])
                else:
                    out[i * steps + s] = Scalar[T](0)
    host_parallelize(task, tasks)
    _ = mapping


def xseq_run_step_weights(n: Int, steps: Int, lengths: Int, sw: Int, sw_steps: Bool, dst: Int) raises:
    """The score weights of every step of n sequences (output="sequence"): dst[i T + s] = (s < length_i ? 1 : 0)
    times the sample weight (per sequence, or per step when `sw_steps`); 1.0 factors when absent (address 0)."""
    var lp = XseqRunF32P(unsafe_from_address=lengths if lengths != 0 else dst)
    var wp = XseqRunF32P(unsafe_from_address=sw if sw != 0 else dst)
    var out = XseqRunF32P(unsafe_from_address=dst)
    var has_l = lengths != 0
    var has_w = sw != 0
    var tasks = host_predict_task_count_work(max(1, n), max(1, n * steps))
    var chunk = host_predict_chunk(max(1, n), tasks)
    def task(worker: Int) {imm lp, imm wp, imm out, imm has_l, imm has_w, imm sw_steps, imm n, imm steps, imm chunk}:
        for i in range(worker * chunk, min((worker + 1) * chunk, n)):
            var valid = Int(lp[i]) if has_l else steps
            for s in range(steps):
                var w = Float32(1.0) if s < valid else Float32(0.0)
                if has_w and s < valid:
                    w = wp[i * steps + s] if sw_steps else wp[i]
                out[i * steps + s] = w
    host_parallelize(task, tasks)
