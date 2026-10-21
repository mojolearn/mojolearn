# SPDX-License-Identifier: Apache-2.0
"""Sequential and operation-session checkpoints, in the checkpoint container
of training/checkpoint.mojo (training/CHECKPOINT_FORMAT.md section 11).

A `Sequential` checkpoint is section kind CKPT_SECTION_SEQUENTIAL: the
layout table names every parameter tensor of the description (layer i's
weight and bias, `l<i>.weight` / `l<i>.bias`), the payload holds the
parameters and the optimizer's two buffers (the momentum buffer, or Adam's
first and second moments; zeros in an inference export), the header's `t`
the optimizer steps completed, the descriptor the seed and the optimizer's
hyperparameters, and the extension record the rest:

  ints:   [record version, export flag, description length P, the P
           description words, the 9 option words (loss, target, optimizer,
           optimizer flags, schedule, schedule step, n_iter_no_change,
           batch size, shuffle), n_features, n_outputs, y_1d, tol (float64
           bits), the FIT_STATE_WORDS state words (autograd/model.mojo FS_*),
           epochs run, class count k, class kind (core/cast_in.mojo
           CAST_KIND_*), class item size, then k class values (their bytes,
           little-endian, zero-extended to 64 bits)]
  floats: [lr, f1, f2, eps, weight_decay, f7, schedule gamma, huber delta,
           one float per layer (the description's table), the epochs-run
           loss curve]

An operation-session checkpoint is CKPT_SECTION_SESSION: tensors `p<i>` in
the layout, the parameters and, per parameter, the optimizer's first and
second state buffers in the payload (zeros where the optimizer keeps none),
and in the extension
  ints:   [record version, K, (rows, cols) per parameter, has optimizer,
           then per parameter (step count, has first buffer, has second
           buffer), seed low, seed high, epoch, step]
  floats: per parameter (beta1^t, beta2^t, NAdam's mu product)
the words OpsSession.optimizer_state returns; a restore goes through
OpsSession.load_optimizer_state, the one restore path.

A captured-program checkpoint is CKPT_SECTION_PROGRAM: tensors `p<i>` (the
program's parameters) in the layout and payload, and in the extension the
description words of autograd/program.mojo program_words (lane ag-program).

Python passes paths, addresses and the option lists; every byte is written
and read here, host side (a checkpoint is a host file). No arithmetic on the
values: they move by memcpy and by their bits. The Python doors (argument
unpacking and results) are in model_checkpoint_pyapi.mojo.
"""
from std.memory import bitcast, memcpy
from sequence.exec_trait import Exec
from sequence.ops import FP
from autograd.model import (
    FIT_STATE_WORDS, FS_SEED_HI, FS_SEED_LO, FS_STEP, IP64, LAYER_LAYER_NORM, LAYER_LINEAR, ModelSpec,
    parse_model,
)
from autograd.program import program_from_words, program_words
from autograd.ops_api import OpsSession, ops_session
from core.cast_in import CAST_KIND_FLOAT, CastIn
from autograd.arg_words import cast_in_words, fptr_word, fval_word
from training.checkpoint import (
    CKPT_REFUSE_SHAPE, CKPT_SECTION_PROGRAM, CKPT_SECTION_SEQUENTIAL, CKPT_SECTION_SESSION, Checkpoint,
    read_checkpoint_unchecked,
    save_checkpoint,
)

comptime BP = UnsafePointer[UInt8, MutUntrackedOrigin]

comptime SEQ_RECORD_VERSION = 1
comptime SESSION_RECORD_VERSION = 1
comptime SEQ_OPT_INTS = 9      # loss, target, optimizer, flags, schedule, step, patience, batch, shuffle
comptime SEQ_MODEL_INTS = 3    # n_features, n_outputs, y_1d
comptime SEQ_OPT_FLOATS = 8    # lr, f1, f2, eps, weight_decay, f7, gamma, huber delta
# model_save_checkpoint's ip: [export] + the option words + the model words
# + [epochs run, class count]
comptime SEQ_SAVE_INTS = 1 + SEQ_OPT_INTS + SEQ_MODEL_INTS + 2
comptime SEQ_FIT_FLOATS = 9    # model_fit's fp head: lr, f1, f2, eps, wd, f7, tol, gamma, huber delta


# ---------------------------------------------------------------- layout
def model_layout(spec: ModelSpec, mut offsets: List[Int], mut names: List[String]):
    """The checkpoint layout of a description: one entry per parameter
    tensor, in the flat vector's order."""
    offsets.append(0)
    for i in range(spec.n_layers()):  # driver-loop: over layers
        var k = spec.kinds[i]
        var off = spec.param_off[i]
        var tag = String("l") + String(i)
        if k == LAYER_LINEAR:
            var nw = spec.in_width[i] * spec.arg0[i]
            names.append(tag + ".weight")
            offsets.append(off + nw)
            if spec.arg1[i] == 1:
                names.append(tag + ".bias")
                offsets.append(off + nw + spec.arg0[i])
        elif k == LAYER_LAYER_NORM:
            var w = spec.in_width[i]
            names.append(tag + ".weight")
            offsets.append(off + w)
            names.append(tag + ".bias")
            offsets.append(off + 2 * w)


def _check_layout(ck: Checkpoint, offsets: List[Int], names: List[String], path: String) raises:
    """training/checkpoint.mojo load_checkpoint's comparison, against the
    layout of the description the file itself carries."""
    if ck.j_count() != len(names) or ck.n_total() != offsets[len(offsets) - 1]:
        raise Error(String(CKPT_REFUSE_SHAPE) + ": '" + path + "' holds " + String(ck.n_total())
                    + " parameters in " + String(ck.j_count()) + " tensors and its model description has "
                    + String(offsets[len(offsets) - 1]) + " in " + String(len(names)))
    for j in range(len(names)):  # driver-loop: over parameter tensors
        if ck.names[j] != names[j] or ck.offsets[j + 1] != offsets[j + 1]:
            raise Error(String(CKPT_REFUSE_SHAPE) + ": '" + path + "' tensor " + String(j) + " is '"
                        + ck.names[j] + "' and the description's is '" + names[j] + "'")


def _floats_of(p: FP, n: Int) -> List[Float32]:
    var out = List[Float32](length=n, fill=Float32(0))
    if n > 0:
        memcpy(dest=out.unsafe_ptr(), src=p, count=n)
    return out^


def _put_floats(dst: FP, src: List[Float32]):
    if len(src) > 0:
        memcpy(dest=dst, src=src.unsafe_ptr(), count=len(src))


# ---------------------------------------------------------------- Sequential save


# ---------------------------------------------------------------- Sequential load
struct _SeqRecord(Movable):
    var is_export: Bool
    var prog: List[Int]
    var opts: List[Int]       # option words then model words
    var tol_bits: Int64
    var state: List[Int64]
    var n_run: Int
    var k: Int
    var class_kind: Int
    var class_size: Int
    var class_at: Int         # index of the first class word in ext_ints
    var floats: List[Float32] # the 8 option floats
    var table: List[Float32]
    var curve_at: Int         # index of the first curve value in ext_floats

    def __init__(out self):
        self.is_export = False
        self.prog = List[Int]()
        self.opts = List[Int]()
        self.tol_bits = Int64(0)
        self.state = List[Int64]()
        self.n_run = 0
        self.k = 0
        self.class_kind = 0
        self.class_size = 0
        self.class_at = 0
        self.floats = List[Float32]()
        self.table = List[Float32]()
        self.curve_at = 0


def _damaged(path: String) -> Error:
    return Error("mojolearn: '" + path + "' holds a damaged Sequential record")


def _seq_record(ck: Checkpoint, path: String) raises -> _SeqRecord:
    """The extension record of a Sequential checkpoint, every count checked
    against the record's length before it is used."""
    if ck.section_kind != CKPT_SECTION_SEQUENTIAL:
        raise Error("mojolearn: '" + path + "' is not a Sequential checkpoint")
    var xi = ck.ext_ints.copy()
    var xf = ck.ext_floats.copy()
    if len(xi) < 3 or Int(xi[0]) != SEQ_RECORD_VERSION:
        raise Error("mojolearn: '" + path + "' holds an unknown Sequential record version")
    var r = _SeqRecord()
    r.is_export = xi[1] != 0
    var P = Int(xi[2])
    var fixed = SEQ_OPT_INTS + SEQ_MODEL_INTS + 1 + FIT_STATE_WORDS + 4
    if P < 3 or len(xi) < 3 + P + fixed:
        raise _damaged(path)
    var at = 3
    for _ in range(P):  # driver-loop: over description words
        r.prog.append(Int(xi[at]))
        at += 1
    for _ in range(SEQ_OPT_INTS + SEQ_MODEL_INTS):  # driver-loop: over option words
        r.opts.append(Int(xi[at]))
        at += 1
    r.tol_bits = xi[at]
    at += 1
    for _ in range(FIT_STATE_WORDS):  # driver-loop: over state words
        r.state.append(xi[at])
        at += 1
    r.n_run = Int(xi[at])
    r.k = Int(xi[at + 1])
    r.class_kind = Int(xi[at + 2])
    r.class_size = Int(xi[at + 3])
    at += 4
    r.class_at = at
    if r.n_run < 0 or r.k < 0 or len(xi) != at + r.k or r.class_size < 0 or r.class_size > 8:
        raise _damaged(path)
    var n_layers = r.prog[2]
    if n_layers < 1 or len(xf) != SEQ_OPT_FLOATS + n_layers + r.n_run:
        raise _damaged(path)
    for i in range(SEQ_OPT_FLOATS):  # driver-loop: over option floats
        r.floats.append(xf[i])
    for i in range(n_layers):  # driver-loop: over layers
        r.table.append(xf[SEQ_OPT_FLOATS + i])
    r.curve_at = SEQ_OPT_FLOATS + n_layers
    return r^


def _read_sequential(path: String) raises -> Checkpoint:
    """A Sequential checkpoint, its layout checked against the description
    it carries."""
    var ck = read_checkpoint_unchecked(path)
    var r = _seq_record(ck, path)
    var spec = parse_model(r.prog, r.table)
    var offsets = List[Int]()
    var names = List[String]()
    model_layout(spec, offsets, names)
    _check_layout(ck, offsets, names, path)
    return ck^


# ---------------------------------------------------------------- operation sessions
comptime SESSION_PER_PARAM_INTS = 3    # step count, has first buffer, has second buffer
comptime SESSION_PER_PARAM_FLOATS = 3  # beta1^t, beta2^t, mu product


def _session_record(ck: Checkpoint, path: String) raises -> List[Int64]:
    if ck.section_kind != CKPT_SECTION_SESSION:
        raise Error("mojolearn: '" + path + "' is not an operation-session checkpoint")
    var xi = ck.ext_ints.copy()
    if len(xi) < 2 or Int(xi[0]) != SESSION_RECORD_VERSION:
        raise Error("mojolearn: '" + path + "' holds an unknown session record version")
    var K = Int(xi[1])
    if (K < 1 or K != ck.j_count() or len(xi) != 2 + 2 * K + 1 + SESSION_PER_PARAM_INTS * K + 4
            or len(ck.ext_floats) != SESSION_PER_PARAM_FLOATS * K):
        raise Error("mojolearn: '" + path + "' holds a damaged session record")
    for i in range(K):  # driver-loop: over parameter tensors
        var rows = Int(xi[2 + 2 * i])
        var cols = Int(xi[3 + 2 * i])
        if rows < 1 or cols < 1 or ck.offsets[i + 1] - ck.offsets[i] != rows * cols:
            raise Error(String(CKPT_REFUSE_SHAPE) + ": '" + path + "' tensor p" + String(i)
                        + " does not hold its recorded shape")
    return xi^


def _upload_slice[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], src: List[Float32], off: Int, rows: Int, cols: Int, grad: Bool,
) raises -> Int:
    """A new (rows, cols) session tensor holding src[off : off + rows * cols]."""
    var c = CastIn(Int(src.unsafe_ptr()) + 4 * off, CAST_KIND_FLOAT, 4, rows, cols, 4 * cols, 4)
    return s.upload(c, grad)


# ---------------------------------------------------------------- captured programs
# A program checkpoint is CKPT_SECTION_PROGRAM (lane ag-program): tensors
# `p<i>` in the layout, the program's parameters in the payload (the two
# state arrays zeros), and in the extension the description words of
# autograd/program.mojo program_words. A load uploads the parameters into
# the session as new tensors and registers the program over them, so the
# file runs on any install's session (the CPU-only install's host executor
# included).


# ---------------------------------------------------------------- entries
# The typed entries behind model_checkpoint_pyapi.mojo: the Python door
# converts its arguments to Ints, Float64s and Strings and calls one of
# these; every check, copy and session call runs here, in its old order.
def model_save_checkpoint(
    file: String, words: List[Int], addrs: List[Int], ip: List[Int], fp: List[Float64], classes: List[Int],
) raises -> Int:
    """Write a Sequential checkpoint to `file` (model_save_checkpoint_py's
    contract; `classes` the converted classes_ descriptor, empty when the
    class count is 0). Returns the file's byte count."""
    if len(addrs) != 5 or len(ip) != SEQ_SAVE_INTS or len(fp) < SEQ_FIT_FLOATS + 1:
        raise Error("model_save_checkpoint: requires 5 addresses, " + String(SEQ_SAVE_INTS)
                    + " integers and 9 floats plus one per layer")
    var table = List[Float32]()
    for i in range(SEQ_FIT_FLOATS, len(fp)):  # driver-loop: over layers' float arguments
        table.append(fval_word(fp, i))
    var spec = parse_model(words, table)
    var is_export = ip[0] != 0
    var n_run = ip[1 + SEQ_OPT_INTS + SEQ_MODEL_INTS]
    var k = ip[2 + SEQ_OPT_INTS + SEQ_MODEL_INTS]
    var np_ = spec.n_params
    var ck = Checkpoint()
    ck.section_kind = CKPT_SECTION_SEQUENTIAL
    model_layout(spec, ck.offsets, ck.names)
    for _ in range(len(ck.names)):  # driver-loop: over parameter tensors
        ck.buf_initialized.append(False)
    ck.param = _floats_of(fptr_word(addrs[0], "params"), np_)
    var state = List[Int64](length=FIT_STATE_WORDS, fill=Int64(0))
    if is_export:
        ck.m_state = List[Float32](length=np_, fill=Float32(0))
        ck.v_state = List[Float32](length=np_, fill=Float32(0))
        n_run = 0
    else:
        ck.m_state = _floats_of(fptr_word(addrs[1], "m"), np_)
        ck.v_state = _floats_of(fptr_word(addrs[2], "v"), np_)
        var sa = addrs[3]
        if sa == 0:
            raise Error("model_save_checkpoint: null training-state buffer")
        memcpy(dest=state.unsafe_ptr(), src=IP64(unsafe_from_address=sa), count=FIT_STATE_WORDS)
    var opt = ip[3]
    var flags = ip[4]
    ck.t = Int(state[FS_STEP])
    ck.seed = (UInt64(state[FS_SEED_HI]) << 32) | UInt64(state[FS_SEED_LO])
    ck.opt_kind = opt
    ck.lr = fval_word(fp, 0)
    if opt == 0:
        ck.momentum = fval_word(fp, 1)
    else:
        ck.beta1 = fval_word(fp, 1)
        ck.beta2 = fval_word(fp, 2)
    ck.eps = fval_word(fp, 3)
    ck.weight_decay = fval_word(fp, 4)
    ck.dampening = fval_word(fp, 5)
    ck.nesterov = (flags & 1) != 0
    # the extension record
    var xi = List[Int64]()
    xi.append(Int64(SEQ_RECORD_VERSION))
    xi.append(Int64(1) if is_export else Int64(0))
    xi.append(Int64(len(words)))
    for i in range(len(words)):  # driver-loop: over description words
        xi.append(Int64(words[i]))
    for i in range(SEQ_OPT_INTS + SEQ_MODEL_INTS):  # driver-loop: over option words
        xi.append(Int64(ip[1 + i]))
    xi.append(bitcast[DType.int64](fp[6]))
    for i in range(FIT_STATE_WORDS):  # driver-loop: over state words
        xi.append(state[i])
    xi.append(Int64(n_run))
    xi.append(Int64(k))
    if k > 0:
        var c = cast_in_words(classes, "classes")
        if c.count() != k or c.itemsize > 8 or not c.is_c_order():
            raise Error("model_save_checkpoint: classes must be " + String(k) + " contiguous values")
        xi.append(Int64(c.kind))
        xi.append(Int64(c.itemsize))
        var bp = BP(unsafe_from_address=c.addr)
        for j in range(k):  # driver-loop: over classes
            var w = UInt64(0)
            for b in range(c.itemsize):  # driver-loop: over the value's bytes
                w = w | (UInt64(bp[j * c.itemsize + b]) << UInt64(8 * b))
            xi.append(bitcast[DType.int64](w))
    else:
        xi.append(Int64(0))
        xi.append(Int64(0))
    var xf = List[Float32]()
    for i in range(6):  # driver-loop: over option floats (lr .. f7)
        xf.append(fval_word(fp, i))
    xf.append(fval_word(fp, 7))
    xf.append(fval_word(fp, 8))
    for i in range(len(table)):  # driver-loop: over layers
        xf.append(table[i])
    if n_run > 0:
        var curve = _floats_of(fptr_word(addrs[4], "loss_curve"), n_run)
        for e in range(n_run):  # driver-loop: over epochs
            xf.append(curve[e])
    ck.ext_ints = xi^
    ck.ext_floats = xf^
    var h = save_checkpoint(file, ck)
    return h.bytes


struct SeqCheckpointInfo(Movable):
    """model_checkpoint_info_py's answer as typed words: per layer (code,
    arg0, arg1) in `layer_ints` and its float in `layer_floats`; `ints` =
    the option words, n_features, n_outputs, y_1d, epochs run, class count,
    class kind, class item size, n_params; `floats` = lr, f1, f2, eps,
    weight_decay, f7, tol, schedule gamma, huber delta."""
    var is_export: Bool
    var layer_ints: List[Int]
    var layer_floats: List[Float64]
    var ints: List[Int]
    var floats: List[Float64]

    def __init__(out self):
        self.is_export = False
        self.layer_ints = List[Int]()
        self.layer_floats = List[Float64]()
        self.ints = List[Int]()
        self.floats = List[Float64]()


def model_checkpoint_info(file: String) raises -> SeqCheckpointInfo:
    var ck = _read_sequential(file)
    var r = _seq_record(ck, file)
    var spec = parse_model(r.prog, r.table)
    var out = SeqCheckpointInfo()
    out.is_export = r.is_export
    for i in range(spec.n_layers()):  # driver-loop: over layers
        out.layer_ints.append(spec.kinds[i])
        out.layer_ints.append(spec.arg0[i])
        out.layer_ints.append(spec.arg1[i])
        out.layer_floats.append(Float64(spec.farg[i]))
    for i in range(len(r.opts)):  # driver-loop: over option words
        out.ints.append(r.opts[i])
    out.ints.append(r.n_run)
    out.ints.append(r.k)
    out.ints.append(r.class_kind)
    out.ints.append(r.class_size)
    out.ints.append(spec.n_params)
    for i in range(6):  # driver-loop: over option floats (lr .. f7)
        out.floats.append(Float64(r.floats[i]))
    out.floats.append(bitcast[DType.float64](r.tol_bits))
    out.floats.append(Float64(r.floats[6]))
    out.floats.append(Float64(r.floats[7]))
    return out^


def model_load_checkpoint(file: String, addrs: List[Int]) raises -> Int:
    """Fill the caller's buffers from a Sequential checkpoint
    (model_load_checkpoint_py's contract). Returns n_params."""
    if len(addrs) != 6:
        raise Error("model_load_checkpoint: requires 6 addresses")
    var ck = _read_sequential(file)
    var r = _seq_record(ck, file)
    _put_floats(fptr_word(addrs[0], "params"), ck.param)
    if addrs[1] != 0:
        _put_floats(fptr_word(addrs[1], "m"), ck.m_state)
    if addrs[2] != 0:
        _put_floats(fptr_word(addrs[2], "v"), ck.v_state)
    if addrs[3] != 0:
        memcpy(dest=IP64(unsafe_from_address=addrs[3]), src=r.state.unsafe_ptr(), count=FIT_STATE_WORDS)
    if addrs[4] != 0 and r.n_run > 0:
        var curve = fptr_word(addrs[4], "loss_curve")
        memcpy(dest=curve, src=ck.ext_floats.unsafe_ptr() + r.curve_at, count=r.n_run)
    if addrs[5] != 0 and r.k > 0:
        var bp = BP(unsafe_from_address=addrs[5])
        for j in range(r.k):  # driver-loop: over classes
            var w = bitcast[DType.uint64](ck.ext_ints[r.class_at + j])
            for b in range(r.class_size):  # driver-loop: over the value's bytes
                bp[j * r.class_size + b] = UInt8((w >> UInt64(8 * b)) & UInt64(0xFF))
    return ck.n_total()


def ops_save_state[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, file: String, hs: List[Int], opt: Int, counters: List[Int],
) raises -> Int:
    """Write the session tensors `hs`, the state of optimizer `opt` (0 for
    none) for each of them and counters = [seed low, seed high, epoch,
    step] to `file`. Returns the byte count."""
    if len(counters) != 4:
        raise Error("ops_save_state: counters are [seed low, seed high, epoch, step]")
    var K = len(hs)
    if K < 1:
        raise Error("ops_save_state: no parameters")
    var p = ops_session[E, KIND](sid)
    var ck = Checkpoint()
    ck.section_kind = CKPT_SECTION_SESSION
    var xi = List[Int64]()
    xi.append(Int64(SESSION_RECORD_VERSION))
    xi.append(Int64(K))
    var n = 0
    ck.offsets.append(0)
    for i in range(K):  # driver-loop: over parameter handles
        var shape = p[].shape(hs[i])
        xi.append(Int64(shape[0]))
        xi.append(Int64(shape[1]))
        n += shape[0] * shape[1]
        ck.offsets.append(n)
        ck.names.append(String("p") + String(i))
        ck.buf_initialized.append(False)
    ck.param = List[Float32](length=n, fill=Float32(0))
    ck.m_state = List[Float32](length=n, fill=Float32(0))
    ck.v_state = List[Float32](length=n, fill=Float32(0))
    xi.append(Int64(1) if opt != 0 else Int64(0))
    var xf = List[Float32]()
    var first_step = 0
    for i in range(K):  # driver-loop: over parameter handles
        var off = ck.offsets[i]
        p[].download(hs[i], FP(unsafe_from_address=Int(ck.param.unsafe_ptr())) + off)
        if opt == 0:
            xi.append(Int64(0))
            xi.append(Int64(0))
            xi.append(Int64(0))
            xf.append(Float32(1))
            xf.append(Float32(1))
            xf.append(Float32(1))
            continue
        # (step, beta1^t, beta2^t, mu product, buffer copies): the words
        # load_optimizer_state takes back
        var w = p[].optimizer_state(opt, hs[i])
        if w[6] != 0:
            raise Error("ops_save_state: this optimizer keeps a third state buffer, which the "
                        + "session checkpoint does not hold")
        if w[4] != 0:
            p[].download(w[4], FP(unsafe_from_address=Int(ck.m_state.unsafe_ptr())) + off)
            p[].free(w[4])
        if w[5] != 0:
            p[].download(w[5], FP(unsafe_from_address=Int(ck.v_state.unsafe_ptr())) + off)
            p[].free(w[5])
        if i == 0:
            first_step = w[0]
        xi.append(Int64(w[0]))
        xi.append(Int64(1) if w[4] != 0 else Int64(0))
        xi.append(Int64(1) if w[5] != 0 else Int64(0))
        xf.append(w[1])
        xf.append(w[2])
        xf.append(w[3])
    var lo = counters[0]
    var hi = counters[1]
    xi.append(Int64(lo))
    xi.append(Int64(hi))
    xi.append(Int64(counters[2]))
    xi.append(Int64(counters[3]))
    ck.seed = (UInt64(hi) << 32) | UInt64(lo)
    ck.t = first_step
    ck.ext_ints = xi^
    ck.ext_floats = xf^
    var h = save_checkpoint(file, ck)
    return h.bytes


struct SessionStateLoad(Movable):
    """ops_load_state's answer: the new tensor handles, whether the file
    holds optimizer state, and [seed low, seed high, epoch, step]."""
    var handles: List[Int]
    var has_opt: Bool
    var counters: List[Int]

    def __init__(out self):
        self.handles = List[Int]()
        self.has_opt = False
        self.counters = List[Int]()


def ops_load_state[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, file: String, grad: Bool, opt: Int,
) raises -> SessionStateLoad:
    """Upload the parameters of a session checkpoint as new tensors and,
    with optimizer `opt` (or 0), restore each one's saved optimizer state
    through OpsSession.load_optimizer_state (the buffers are copied from
    temporary tensors, freed after)."""
    var ck = read_checkpoint_unchecked(file)
    var xi = _session_record(ck, file)
    var K = Int(xi[1])
    var has_opt = xi[2 + 2 * K] != 0
    if opt != 0 and not has_opt:
        raise Error("mojolearn: '" + file + "' holds no optimizer state")
    var p = ops_session[E, KIND](sid)
    var res = SessionStateLoad()
    var at = 2 + 2 * K + 1
    for i in range(K):  # driver-loop: over parameter tensors
        var rows = Int(xi[2 + 2 * i])
        var cols = Int(xi[3 + 2 * i])
        var off = ck.offsets[i]
        var h = _upload_slice(p[], ck.param, off, rows, cols, grad)
        res.handles.append(h)
        if opt == 0:
            continue
        var base = at + SESSION_PER_PARAM_INTS * i
        var hm = 0
        var hv = 0
        if xi[base + 1] != 0:
            hm = _upload_slice(p[], ck.m_state, off, rows, cols, False)
        if xi[base + 2] != 0:
            hv = _upload_slice(p[], ck.v_state, off, rows, cols, False)
        var ints = List[Int]()
        ints.append(Int(xi[base]))
        ints.append(hm)
        ints.append(hv)
        ints.append(0)
        var fs = List[Float32]()
        for k in range(SESSION_PER_PARAM_FLOATS):  # driver-loop: over the three bias-correction words
            fs.append(ck.ext_floats[SESSION_PER_PARAM_FLOATS * i + k])
        p[].load_optimizer_state(opt, h, ints, fs)
        if hm != 0:
            p[].free(hm)
        if hv != 0:
            p[].free(hv)
    var c = at + SESSION_PER_PARAM_INTS * K
    res.has_opt = has_opt
    for i in range(4):  # driver-loop: over the four counters
        res.counters.append(Int(xi[c + i]))
    _ = ck^
    return res^


def ops_save_program[E: Exec & Movable & Deinitable, KIND: Int](sid: Int, pid: Int, file: String) raises -> Int:
    """Write program `pid` of session `sid` and its parameters' current
    values to `file`. Returns the byte count."""
    var p = ops_session[E, KIND](sid)
    var prog = p[].program(pid)
    var K = len(prog.params)
    var ck = Checkpoint()
    ck.section_kind = CKPT_SECTION_PROGRAM
    var n = 0
    ck.offsets.append(0)
    for i in range(K):  # driver-loop: over program parameters
        n += prog.param_shapes[i].count()
        ck.offsets.append(n)
        ck.names.append(String("p") + String(i))
        ck.buf_initialized.append(False)
    # a program without parameters still writes one (unused) word
    var n_store = n if n > 0 else 1
    ck.param = List[Float32](length=n_store, fill=Float32(0))
    ck.m_state = List[Float32](length=n_store, fill=Float32(0))
    ck.v_state = List[Float32](length=n_store, fill=Float32(0))
    if K == 0:
        ck.offsets.append(1)
        ck.names.append(String("unused"))
        ck.buf_initialized.append(False)
    for i in range(K):  # driver-loop: over program parameters
        p[].download(prog.params[i], FP(unsafe_from_address=Int(ck.param.unsafe_ptr())) + ck.offsets[i])
    var words = program_words(prog)
    ck.ext_ints = words[0].copy()
    ck.ext_floats = words[1].copy()
    ck.seed = 0
    ck.t = 0
    var h = save_checkpoint(file, ck)
    return h.bytes


def ops_load_program[E: Exec & Movable & Deinitable, KIND: Int](sid: Int, file: String) raises -> Int:
    """Read a program checkpoint into session `sid`: its parameters become
    new session tensors and the program runs over them. Returns the new
    program id."""
    var ck = read_checkpoint_unchecked(file)
    if ck.section_kind != CKPT_SECTION_PROGRAM:
        raise Error("mojolearn: '" + file + "' is not a captured-program checkpoint")
    var prog = program_from_words(ck.ext_ints, ck.ext_floats, "mojolearn: '" + file + "'")
    var K = len(prog.params)
    if (K > 0 and ck.j_count() != K) or (K == 0 and ck.j_count() != 1):
        raise Error("mojolearn: '" + file + "' holds a damaged program record")
    var p = ops_session[E, KIND](sid)
    if p[].recording():
        raise Error("mojolearn: programs run in an inference session (Session(record=False))")
    for i in range(K):  # driver-loop: over program parameters
        var sh = prog.param_shapes[i]
        var off = ck.offsets[i]
        if ck.offsets[i + 1] - off != sh.count():
            raise Error(String(CKPT_REFUSE_SHAPE) + ": '" + file + "' tensor p" + String(i)
                        + " does not hold its recorded shape")
        var c = CastIn(Int(ck.param.unsafe_ptr()) + 4 * off, CAST_KIND_FLOAT, 4, sh.rows(), sh.cols(), 4 * sh.cols(), 4)
        prog.params[i] = p[].upload_nd(c, sh, False)
    var pid = p[].add_program(prog^)
    _ = ck^
    return pid
