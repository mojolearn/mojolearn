# SPDX-License-Identifier: Apache-2.0
"""CNNClassifier fit and predict on the program executor (lane
x8-cnn-on-executor, docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md row X8).

The classifier's whole fit is ONE captured program trained by `Program.fit`
(autograd/program_runtime.mojo `train_program_control`) in an inference
`OpsSession` on the caller's executor (the device or the host column, the same
code): per conv block `OPS_CONV2D` (stride 1, 'same' padding k // 2, the bias),
`OPS_RELU` and, where the map still holds a window, `OPS_MAXPOOL2D` (window =
stride = pool_size), then `OPS_RESHAPE_N` (flatten from axis 1) and the head
`OPS_LINEAR` over `OPS_TRANSPOSE` of the (classes, flat) weight; the loss is
the program loss `cross_entropy` (mean softmax cross-entropy on the class
codes); the optimizer is the session's (SGD with momentum, dampening,
Nesterov and weight decay, Adam, AdamW: one OP_OPT launch per parameter).
The epoch order is Program.fit's `permutation(seed, epoch, n)` (OP_MLP_PERM,
the Feistel order the old x_cnn epoch entry drew) with the classifier's
splitmix64 seed. Every forward and backward kernel is the tape's (the rank-4
conv, pool and layout ops of autograd/rankn_conv.mojo, the tape's linear and
loss), one fold order on the host and on every vendor.

predict / predict_proba (lane x8b-cnn-resident-predict): the fitted
parameters live ONCE in a resident registry session (CNN_OPEN, made at the
model's first predict after a fit, a weight replacement or an unpickle, and
closed with the binding's lifetime owner through SESSION_CLOSE) as the
parameters of one captured program, the forward and its `OPS_SOFTMAX`. Each
CNN_PREDICT uploads only the images, once, replays that program over row
chunks on the executor (each chunk's probabilities copied into one (n, k)
buffer), and makes ONE download: the probabilities, or the int32
`OPS_ARGMAX` codes of the whole buffer. The same kernels and fold order as
the per-call forward it replaces, so no word changes.

TOMBSTONE: the per-call CNN predict (a fresh session that uploaded every
parameter at each call and downloaded each chunk) deleted 2026-11-09 by
lane/x8b-cnn-resident-predict; code recoverable at 7c9d9d5575 (it never
reached main).

Python reaches these through the x_sequence whole-model descriptor ABI
(sequence/neural_model_dispatch.mojo CNN_FIT, CNN_PREDICT); no Python runs
here.

TOMBSTONE: the hand CNN epoch (x_cnn/device_entries_impl.mojo fit_epoch_r_device /
fit_epoch_d_device, their x_cnn_fit_epoch_r / x_cnn_fit_epoch_d / x_cnn_fit_d /
x_cnn_predict_r / x_cnn_epoch_rows / x_cnn_adam_hyper_d entries and the tape
wrapper autograd/rules_cnn.mojo) deleted 2026-11-09 by lane/x8-cnn-on-executor;
code recoverable at f59d913b0b.
Restore: git apply experiments/removed/X8_CNN_HAND_EPOCH.patch
"""
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_COPY, OPT_SGD, OPT_ADAM, OPT_ADAMW
from core.cast_in import CastIn, CAST_KIND_SIGNED
from autograd.rankn import ShapeN
from autograd.rankn_api import rn_shape
from autograd.model import SCHED_CONSTANT
from autograd.ops_api import (
    OpsSession, OPS_CONV2D, OPS_RELU, OPS_MAXPOOL2D, OPS_RESHAPE_N, OPS_TRANSPOSE, OPS_LINEAR, OPS_SOFTMAX,
    OPS_ARGMAX, ops_session_open, ops_session_close, ops_session, ops_session_registry_token,
)
from autograd.program_runtime import train_program_control, parameter_buffers, replay_values_prepared
from autograd.program_replay_metadata import PreparedProgramReplay
from autograd.tape import Tape
from autograd.cast_ops import upload_cast

# Program.fit's loss kind 5: cross_entropy over (rows, 1) class codes.
comptime _LOSS_CROSS_ENTROPY = 5

# CNN_FIT options: [n, C, H, W, k, nb, ks, ps, batch, epochs, shuffle, seed_lo,
# seed_hi, optimizer (0 sgd, 1 adam, 2 adamw), nesterov, oc_0 .. oc_{nb-1},
# pool_0 .. pool_{nb-1}]; floats [lr, momentum, beta1, beta2, eps,
# weight_decay, dampening]; addresses [X (n, C*H*W) float32 C order, y codes
# (n) int32, curve (epochs) float32, steps (epochs * ceil(n / batch)) float32,
# then every parameter array, float32 C order, updated in place: w_0, b_0, ..,
# head weight (k, flat), head bias (k)].
comptime CNN_FIT_HEAD = 15
comptime CNN_FIT_FLOATS = 7
comptime CNN_FIT_ARRAYS = 4
# CNN_OPEN options: [1, C, H, W, k, nb, ks, ps, oc_0 .., pool_0 ..];
# addresses [scope (two words written: registry token, executor kind), then
# the parameters as for CNN_FIT]. Returns the session id.
comptime CNN_OPEN_HEAD = 8
comptime CNN_OPEN_ARRAYS = 1
# CNN_PREDICT options: [registry token, executor kind, session id, n, codes
# (0 proba, 1 codes), chunk rows]; addresses [X (n, C*H*W) float32 C order,
# dst (n, k) float32 or (n) int32].
comptime CNN_PREDICT_WORDS = 6
comptime CNN_PREDICT_ARRAYS = 2
# the resident session's one program: the forward and its softmax
comptime _RESIDENT_PROGRAM = 1


struct CnnGeometry(Copyable, Movable):
    """The classifier's layer stack, as CNNClassifier._build lays it out."""

    var c: Int
    var h: Int
    var w: Int
    var k: Int
    var ks: Int
    var ps: Int
    var oc: List[Int]
    var pool: List[Bool]
    var flat: Int

    def __init__(out self, ints: List[Int], head: Int) raises:
        if len(ints) < head:
            raise Error("native CNN: invalid descriptor")
        var c = ints[1]
        var h = ints[2]
        var w = ints[3]
        var k = ints[4]
        var nb = ints[5]
        var ks = ints[6]
        var ps = ints[7]
        if c < 1 or h < 1 or w < 1 or k < 1 or nb < 0 or ks < 1 or ps < 1:
            raise Error("native CNN: positive image dimensions, classes, kernel and pool sizes required")
        if len(ints) != head + 2 * nb:
            raise Error("native CNN: one width and one pool flag per conv block")
        var ocs = List[Int](capacity=nb)
        var pools = List[Bool](capacity=nb)
        var ch = c
        var hh = h
        var ww = w
        var pad = ks // 2
        for j in range(nb):  # driver-loop: over conv blocks (model structure)
            var oc = ints[head + j]
            var pooled = ints[head + nb + j] != 0
            if oc < 1:
                raise Error("native CNN: convolution channels must be positive")
            hh = hh + 2 * pad - ks + 1
            ww = ww + 2 * pad - ks + 1
            if hh < 1 or ww < 1:
                raise Error("native CNN: the kernel is larger than the padded image")
            if pooled:
                if hh < ps or ww < ps:
                    raise Error("native CNN: a pooled block's map is smaller than the window")
                hh = hh // ps
                ww = ww // ps
            ocs.append(oc)
            pools.append(pooled)
            ch = oc
        self.c = c
        self.h = h
        self.w = w
        self.k = k
        self.ks = ks
        self.ps = ps
        self.oc = ocs^
        self.pool = pools^
        self.flat = ch * hh * ww

    def blocks(self) -> Int:
        return len(self.oc)

    def row(self) -> Int:
        return self.c * self.h * self.w

    def image_shape(self, rows: Int) raises -> ShapeN:
        var d: List[Int] = [rows, self.c, self.h, self.w]
        return rn_shape(d, "native CNN: images")

    def param_shapes(self) raises -> List[ShapeN]:
        """Every parameter's shape in descriptor order: per block the
        (oc, C, ks, ks) weight and the (1, oc) bias, then the (k, flat) head
        weight and the (1, k) head bias."""
        var out = List[ShapeN]()
        var ch = self.c
        for j in range(self.blocks()):  # driver-loop: over conv blocks (model structure)
            var wd: List[Int] = [self.oc[j], ch, self.ks, self.ks]
            var bd: List[Int] = [1, self.oc[j]]
            out.append(rn_shape(wd, "native CNN: conv weight"))
            out.append(rn_shape(bd, "native CNN: conv bias"))
            ch = self.oc[j]
        var hw: List[Int] = [self.k, self.flat]
        var hb: List[Int] = [1, self.k]
        out.append(rn_shape(hw, "native CNN: head weight"))
        out.append(rn_shape(hb, "native CNN: head bias"))
        return out^


def _upload_params[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], g: CnnGeometry, addresses: List[Int], first: Int,
) raises -> List[Int]:
    """Upload every parameter array (float32 C order at addresses[first:])
    as a session tensor of its layer shape; returns the handles in order."""
    var shapes = g.param_shapes()
    if len(addresses) != first + len(shapes):
        raise Error("native CNN: one address per parameter array")
    var out = List[Int](capacity=len(shapes))
    for i in range(len(shapes)):  # driver-loop: over parameter arrays (model structure)
        if addresses[first + i] == 0:
            raise Error("native CNN: null parameter address")
        var count = shapes[i].count()
        var src = CastIn.f32_c(addresses[first + i], shapes[i].rows(), shapes[i].cols())
        if src.count() != count:
            raise Error("native CNN: parameter shape mismatch")
        out.append(s.upload_nd(src, shapes[i], False))
    return out^


def _forward[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], x: Int, params: List[Int], g: CnnGeometry,
) raises -> Int:
    """The classifier's logits (rows, k) of the images `x` (rows, C, H, W),
    as session operations (logged for `capture` in an inference session)."""
    var none = List[Float32]()
    var h = x
    var nb = g.blocks()
    for j in range(nb):  # driver-loop: over conv blocks (model structure)
        var hs: List[Int] = [h, params[2 * j], params[2 * j + 1]]
        var conv_ints: List[Int] = [1, g.ks // 2]
        h = s.op(OPS_CONV2D, hs, conv_ints, none)
        var rs: List[Int] = [h]
        h = s.op(OPS_RELU, rs, List[Int](), none)
        if g.pool[j]:
            var ps: List[Int] = [h]
            var pool_ints: List[Int] = [g.ps, g.ps]
            h = s.op(OPS_MAXPOOL2D, ps, pool_ints, none)
    var fs: List[Int] = [h]
    var flat_ints: List[Int] = [1, 1, -1]
    h = s.op(OPS_RESHAPE_N, fs, flat_ints, none)
    # OPS_LINEAR takes the (inputs, outputs) weight; the head keeps torch's
    # (outputs, inputs) layout, so its parameter downloads as it uploaded.
    var ts: List[Int] = [params[2 * nb]]
    var wt = s.op(OPS_TRANSPOSE, ts, List[Int](), none)
    var ls: List[Int] = [h, wt, params[2 * nb + 1]]
    var act: List[Int] = [0]
    return s.op(OPS_LINEAR, ls, act, none)


def _download_params[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], params: List[Int], addresses: List[Int], first: Int,
) raises:
    for i in range(len(params)):  # driver-loop: over parameter arrays (model structure)
        s.download(params[i], FP(unsafe_from_address=addresses[first + i]))


def cnn_fit_program[E: Exec & Movable & Deinitable](
    var ex: E, addresses: List[Int], ints: List[Int], floats: List[Float64],
) raises -> Int:
    """CNNClassifier.fit as one captured program and Program.fit (header).
    Writes the trained parameters over the caller's arrays, the epoch curve
    and every minibatch loss; returns the number of optimizer steps."""
    var g = CnnGeometry(ints, CNN_FIT_HEAD)
    var n = ints[0]
    var batch = ints[8]
    var epochs = ints[9]
    var shuffle = ints[10] != 0
    var seed = (UInt64(ints[12]) << 32) | UInt64(ints[11] & 0xFFFFFFFF)
    var kind = ints[13]
    var nesterov = ints[14] != 0
    if n < 1 or batch < 1 or epochs < 1:
        raise Error("native CNN: fit needs rows, a positive batch size and at least one epoch")
    if kind != OPT_SGD and kind != OPT_ADAM and kind != OPT_ADAMW:
        raise Error("native CNN: the optimizer must be sgd, adam or adamw")
    if len(floats) != CNN_FIT_FLOATS:
        raise Error("native CNN: fit takes seven optimizer scalars")
    if len(addresses) < CNN_FIT_ARRAYS:
        raise Error("native CNN: fit takes the data, label, curve and step arrays")
    for i in range(CNN_FIT_ARRAYS):  # small-loop(the four data arrays)
        if addresses[i] == 0:
            raise Error("native CNN: null data address")
    var sgd = kind == OPT_SGD
    var opt_ints: List[Int] = [kind, 1 if nesterov else 0, SCHED_CONSTANT, 1]
    # [lr, momentum, beta1, beta2, eps, weight_decay, gamma, dampening]; the
    # momentum and dampening are SGD's, the betas Adam's
    var opt_fs: List[Float32] = [
        Float32(floats[0]), Float32(floats[1]) if sgd else Float32(0), Float32(floats[2]), Float32(floats[3]),
        Float32(floats[4]), Float32(floats[5]), Float32(0), Float32(floats[6]) if sgd else Float32(0),
    ]
    var s = OpsSession[E](ex^, False)
    try:
        var params = _upload_params(s, g, addresses, CNN_FIT_ARRAYS)
        # The program is captured from one eager forward over the first
        # minibatch's rows; Program.fit replays it at every minibatch size.
        var bs = min(batch, n)
        var x0 = s.upload_nd(CastIn.f32_c(addresses[0], bs, g.row()), g.image_shape(bs), False)
        var logits = _forward(s, x0, params, g)
        var ins: List[Int] = [x0]
        var outs: List[Int] = [logits]
        var pid = s.capture(ins, outs)
        var opt = s.optimizer(opt_ints, opt_fs)
        var srcs: List[CastIn] = [CastIn.f32_c(addresses[0], n, g.row())]
        var shapes: List[ShapeN] = [g.image_shape(n)]
        var target = CastIn(addresses[1], CAST_KIND_SIGNED, 4, n, 1, 4, 4)
        target.check("y")
        _ = train_program_control(
            s, pid, srcs, shapes, target, _LOSS_CROSS_ENTROPY, Float32(1), opt, params, epochs, batch, shuffle,
            seed, FP(unsafe_from_address=addresses[2]), 0, 0, step_curve=addresses[3],
        )
        _download_params(s, params, addresses, CNN_FIT_ARRAYS)
    except err:
        s.close()
        raise err^
    s.close()
    return epochs * ((n + batch - 1) // batch)




def cnn_open_program[E: Exec & Movable & Deinitable, KIND: Int](
    var ex: E, addresses: List[Int], ints: List[Int],
) raises -> Int:
    """CNNClassifier's resident predict session (CNN_OPEN): a registry
    inference session on `ex` that holds every fitted parameter, uploaded
    once, and the captured forward (the conv blocks, the head and the row
    softmax, program 1) that reads them in place. Writes the registry token
    and the executor kind to the two scope words at addresses[0] (the
    SESSION_CLOSE context the binding's lifetime owner keeps) and returns
    the session id. A later CNN_PREDICT uploads only the images."""
    var g = CnnGeometry(ints, CNN_OPEN_HEAD)
    if len(addresses) < CNN_OPEN_ARRAYS or addresses[0] == 0:
        raise Error("native CNN: null session scope address")
    var sid = ops_session_open[E, KIND](ex^, False)
    try:
        var sp = ops_session[E, KIND](sid)
        var params = _upload_params(sp[], g, addresses, CNN_OPEN_ARRAYS)
        # The forward is captured from one eager pass over one zero image;
        # every op is per row, so the program replays at any chunk size.
        var zeros = List[Float32](length=g.row(), fill=Float32(0))
        var x0 = sp[].upload_nd(CastIn.f32_c(Int(zeros.unsafe_ptr()), 1, g.row()), g.image_shape(1), False)
        var logits = _forward(sp[], x0, params, g)
        var ls: List[Int] = [logits]
        var proba = sp[].op(OPS_SOFTMAX, ls, List[Int](), List[Float32]())
        var ins: List[Int] = [x0]
        var outs: List[Int] = [proba]
        var pid = sp[].capture(ins, outs)
        _ = len(zeros)  # keep-alive past the staging copy and the eager pass
        if pid != _RESIDENT_PROGRAM:
            raise Error("native CNN: the resident session must hold only the forward program")
        sp[].free(x0)
        sp[].reset()
    except err:
        ops_session_close[E, KIND](sid)
        raise err^
    var scope = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=addresses[0])
    scope[0] = ops_session_registry_token()
    scope[1] = KIND
    return sid


def _resident_predict[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], x_addr: Int, dst: Int, n: Int, codes: Bool, chunk: Int,
) raises:
    """Replay the resident forward over the n images at x_addr (uploaded
    once) in chunks of `chunk` rows, each chunk's (m, k) probabilities
    copied on the executor into one (n, k) buffer, then ONE download: the
    probabilities, or (codes) the int32 OPS_ARGMAX of the whole buffer.
    The parameters are the session's, read in place; nothing else moves."""
    if len(s.programs) != _RESIDENT_PROGRAM:
        raise Error("native CNN: the resident session holds no forward program")
    var p = s.program(_RESIDENT_PROGRAM)
    if p.n_in != 1 or len(p.outs) != 1 or p.in_shapes[0].ndim != 4 or p.out_shapes[0].ndim != 2:
        raise Error("native CNN: the resident program is not a classifier forward")
    var shape = p.in_shapes[0]
    var row = shape.count() // shape.d[0]
    var k = p.out_shapes[0].d[1]
    var params = parameter_buffers(s, p)
    var trainable = List[Bool](length=len(p.params), fill=False)
    var prepared = PreparedProgramReplay(p)
    var temps = List[Int]()
    var phs = List[Int]()
    var none = List[Float32]()
    var xb = s.ex.alloc(n * row)
    try:
        var ob = s.ex.alloc(n * k)
        # the scratch tape in, the session tape aside (Session.run_program's
        # order): the replay's tensors never touch the parameters' tape
        var held = Tape()
        swap(held, s.prog_tape)
        swap(held, s.tape)
        try:
            s.tape.reset()
            upload_cast(s.ex, xb, CastIn.f32_c(x_addr, n, row))
            var start = 0
            while start < n:  # driver-loop: over prediction chunks (rows in device-sized batches)
                var m = min(chunk, n - start)
                s._drop_temps(temps)
                temps.clear()
                phs.clear()
                s.tape.reset()
                var ins: List[FP] = [xb + start * row]
                shape.d[0] = m
                var shapes: List[ShapeN] = [shape]
                var outs = replay_values_prepared(s, p, prepared, ins, shapes, params, trainable, temps, phs)
                var a = Args()
                a.p0 = s.tape.value_of(s.ex, s.tensor(outs[0]))
                a.p1 = ob + start * k
                s.ex.launch[OP_COPY](a, m * k)
                start += m
            s._drop_temps(temps)
            temps.clear()
            s.tape.reset()
            if codes:
                var ph = s._leaf(ob, ShapeN.matrix(n, k))
                temps.append(ph)
                var hs: List[Int] = [ph]
                var am_ints: List[Int] = [1, 0, 0]
                var am = s.run_op(OPS_ARGMAX, hs, am_ints, none)
                temps.append(am)
                s.download(am, FP(unsafe_from_address=dst))
            else:
                s.ex.download(FP(unsafe_from_address=dst), ob, n * k)
        finally:
            s._drop_temps(temps)
            s.tape.reset()
            swap(held, s.tape)
            swap(held, s.prog_tape)
            s.ex.free(ob)
    finally:
        s.ex.free(xb)


def cnn_predict_program[E: Exec & Movable & Deinitable, KIND: Int](
    addresses: List[Int], ints: List[Int],
) raises -> Int:
    """CNNClassifier.predict_proba (codes 0: (n, k) float32 probabilities)
    or predict (codes 1: (n) int32 class codes) on the resident session
    `ints[2]` that CNN_OPEN made (header); returns n."""
    if len(ints) != CNN_PREDICT_WORDS:
        raise Error("native CNN: invalid predict descriptor")
    if ints[0] != ops_session_registry_token() or ints[1] != KIND:
        raise Error("native CNN: the resident session belongs to another registry or executor")
    var n = ints[3]
    var codes = ints[4] != 0
    var chunk = ints[5]
    if n < 1 or chunk < 1:
        raise Error("native CNN: predict needs rows and a positive chunk size")
    if len(addresses) != CNN_PREDICT_ARRAYS or addresses[0] == 0 or addresses[1] == 0:
        raise Error("native CNN: null data address")
    var sp = ops_session[E, KIND](ints[2])
    _resident_predict(sp[], addresses[0], addresses[1], n, codes, chunk)
    return n
