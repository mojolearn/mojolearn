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
           the schedule's NN_SCHED_ARGS argument words as float64 bits
           (autograd/nn_optim.mojo), the 8 option floats and then each
           layer's float as passed (float64 bits),
           epochs run, class count k, class kind (core/cast_in.mojo
           CAST_KIND_*), class item size, then k class values (their bytes,
           little-endian, zero-extended to 64 bits)]
  floats: [lr, f1, f2, eps, weight_decay, f7, schedule gamma, huber delta,
           one float per layer (the description's table), the epochs-run
           loss curve, then (version 4, lane stream-fits) the third
           optimizer state buffer when the optimizer keeps one (centered
           RMSprop, autograd/nn_optim.mojo nn_opt_keeps_third: its momentum
           buffer, n_params words; absent for every other optimizer)]

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
from std.memory import bitcast, unsafe_memcpy
from sequence.exec_trait import Exec
from sequence.ops import FP
from autograd.model_blocks import block_param_names, block_param_sizes, is_block
from autograd.model import (
    FIT_STATE_WORDS, FS_SEED_HI, FS_SEED_LO, FS_STEP, IP64, LAYER_CONV1D, LAYER_CONV2D, LAYER_LAYER_NORM,
    LAYER_LINEAR, ModelSpec, SPEC2_ARGS, SPEC2_HEADER, SPEC2_VERSION, conv_in_per_group, parse_model,
)
from autograd.nn_optim import NN_SCHED_ARGS, nn_opt_keeps_third
from autograd.nn_pgseq import PGSEQ_RECORD_MARK  # nn-blocks-pgseq
from autograd.program import program_from_words, program_words, PROGRAM_PARAM_WEIGHT, PROGRAM_PARAM_STATE_WORDS
from autograd.ops_api import OpsSession, ops_session
from core.cast_in import CAST_KIND_FLOAT, CastIn
from autograd.arg_words import cast_in_words, fptr_word, fval_word
from training.checkpoint import (
    CKPT_REFUSE_SHAPE, CKPT_SECTION_PROGRAM, CKPT_SECTION_SEQUENTIAL, CKPT_SECTION_SESSION, Checkpoint,
    read_checkpoint_unchecked,
    save_checkpoint,
    save_checkpoint_sha256,
)

comptime BP = UnsafePointer[UInt8, MutUntrackedOrigin]

# ONE record format (no users yet, no upgrade path; lane nn-fix-3, 2026-11-23): version 3 holds the schedule's
# argument words, the FIT_STATE_WORDS state words and the float64 bits of the option floats (lr .. huber delta;
# the float32 words beside them stay what the kernels read) and of each layer's float argument, so a loaded
# model reads back the values the user passed. A file of any other version is refused by name; the version-1
# and version-2 readers are gone (recoverable at 48117b8854).
# Version 4 (lane stream-fits, 2026-11-25): the state words grew a batch source part (FIT_STATE_WORDS, autograd/
# model.mojo FS_SOURCE: a streamed fit's consumed position and the source's binding words), and centered RMSprop's
# third optimizer state buffer follows the loss curve in the floats. Version 3 files are refused by name.
comptime SEQ_RECORD_VERSION = 4
# Session record version 2 (lane fable-finalize, 2026-11-25): a fourth per-parameter integer says whether the
# optimizer keeps a third state buffer for the parameter (centered RMSprop's gradient average), and those buffers
# follow the per-parameter scalars in the extension's floats, in parameter order. Version 1 refused such an
# optimizer at save while Session.save_state's docstring promised every buffer.
comptime SESSION_RECORD_VERSION = 2
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
        elif k == LAYER_CONV2D or k == LAYER_CONV1D:
            # (seq-blocks 2026-11-22: the conv layers were missing here, so a conv2d model's
            # checkpoint layout did not cover its parameters)
            var ks = spec.arg1[i]
            var taps = ks * ks if k == LAYER_CONV2D else ks
            var nw = spec.arg0[i] * conv_in_per_group(spec, i) * taps
            names.append(tag + ".weight")
            offsets.append(off + nw)
            if spec.xargs[SPEC2_ARGS * i + 4] == 1:
                names.append(tag + ".bias")
                offsets.append(off + nw + spec.arg0[i])
        elif is_block(k) and off >= 0:
            # lane neural-blocks: one entry per parameter tensor of a block layer
            var sizes = block_param_sizes(k, spec.xargs, i, spec.in_shape[i])
            var suffixes = block_param_names(k, spec.xargs, i)
            var end = off
            for j in range(len(sizes)):  # small-loop(at most four parameter tensors of the layer)
                end += sizes[j]
                names.append(tag + suffixes[j])
                offsets.append(end)


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
    var out = List[Float32](unsafe_uninit_length=max(n, 0))
    if n > 0:
        unsafe_memcpy(dest=out.unsafe_ptr(), src=p, count=n)
    return out^


def _put_floats(dst: FP, src: List[Float32]):
    if len(src) > 0:
        unsafe_memcpy(dest=dst, src=src.unsafe_ptr(), count=len(src))


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
    var sched_args: List[Float64]  # the schedule's argument words
    var table: List[Float32]
    var curve_at: Int         # index of the first curve value in ext_floats
    var third_at: Int         # lane stream-fits: index of the third optimizer state buffer in ext_floats
    var third: Int            # its word count (0: the optimizer keeps two buffers)
    var opt_f64: List[Float64]    # the SEQ_OPT_FLOATS option floats as passed
    var layer_f64: List[Float64]  # each layer's float argument as passed
    var groups: List[Float64]     # nn-blocks-pgseq: the parameter groups' words (empty: none)

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
        self.sched_args = List[Float64](length=NN_SCHED_ARGS, fill=0.0)
        self.table = List[Float32]()
        self.curve_at = 0
        self.third_at = 0
        self.third = 0
        self.opt_f64 = List[Float64]()
        self.layer_f64 = List[Float64]()
        self.groups = List[Float64]()


def _damaged(path: String) -> Error:
    return Error("mojolearn: '" + path + "' holds a damaged Sequential record")


def _seq_record(ck: Checkpoint, path: String) raises -> _SeqRecord:
    """The extension record of a Sequential checkpoint, every count checked
    against the record's length before it is used."""
    if ck.section_kind != CKPT_SECTION_SEQUENTIAL:
        raise Error("mojolearn: '" + path + "' is not a Sequential checkpoint")
    var xi = ck.ext_ints.copy()
    var xf = ck.ext_floats.copy()
    if len(xi) < 3:
        raise _damaged(path)
    if Int(xi[0]) != SEQ_RECORD_VERSION:
        raise Error("mojolearn: '" + path + "' holds Sequential record version " + String(Int(xi[0]))
                    + "; this version reads version " + String(SEQ_RECORD_VERSION) + " only (save the model again)")
    var r = _SeqRecord()
    r.is_export = xi[1] != 0
    var P = Int(xi[2])
    var fixed = SEQ_OPT_INTS + SEQ_MODEL_INTS + 1 + FIT_STATE_WORDS + NN_SCHED_ARGS + 4
    if P < 3 or len(xi) < 3 + P + fixed:
        raise _damaged(path)
    var at = 3
    for _ in range(P):  # driver-loop: over description words
        r.prog.append(Int(xi[at]))
        at += 1
    # lane nn-norms (2026-11-23): a version-2 description (input_shape) keeps its layer count after the shape
    # slots; prog[2] is its first sample axis, so every rank-n model's checkpoint was refused as damaged
    var n_layers = r.prog[SPEC2_HEADER - 1] if r.prog[0] == SPEC2_VERSION and len(r.prog) >= SPEC2_HEADER else r.prog[2]
    if n_layers < 1 or len(xi) < 3 + P + fixed + SEQ_OPT_FLOATS + n_layers:
        raise _damaged(path)
    for _ in range(SEQ_OPT_INTS + SEQ_MODEL_INTS):  # driver-loop: over option words
        r.opts.append(Int(xi[at]))
        at += 1
    r.tol_bits = xi[at]
    at += 1
    for _ in range(FIT_STATE_WORDS):  # driver-loop: over state words
        r.state.append(xi[at])
        at += 1
    for k in range(NN_SCHED_ARGS):  # small-loop(8: the schedule's argument words)
        r.sched_args[k] = bitcast[DType.float64](xi[at])
        at += 1
    for _ in range(SEQ_OPT_FLOATS):  # small-loop(8: the option floats' float64 bits)
        r.opt_f64.append(bitcast[DType.float64](xi[at]))
        at += 1
    for _ in range(n_layers):  # driver-loop: over the layers' float arguments (float64 bits)
        r.layer_f64.append(bitcast[DType.float64](xi[at]))
        at += 1
    r.n_run = Int(xi[at])
    r.k = Int(xi[at + 1])
    r.class_kind = Int(xi[at + 2])
    r.class_size = Int(xi[at + 3])
    at += 4
    r.class_at = at
    # nn-blocks-pgseq: after the class words, optionally [PGSEQ_RECORD_MARK, count, the group words' float64 bits]
    var end = at + r.k
    if r.k >= 0 and len(xi) > end + 1 and xi[end] == Int64(PGSEQ_RECORD_MARK):
        var count = Int(xi[end + 1])
        if count < 0 or len(xi) != end + 2 + count:
            raise _damaged(path)
        for j in range(count):  # driver-loop: over the saved group words
            r.groups.append(bitcast[DType.float64](xi[end + 2 + j]))
    elif len(xi) != end:
        raise _damaged(path)
    if r.n_run < 0 or r.k < 0 or r.class_size < 0 or r.class_size > 8:
        raise _damaged(path)
    # lane stream-fits: centered RMSprop's third buffer after the curve (a checkpoint, not an export)
    r.third_at = SEQ_OPT_FLOATS + n_layers + r.n_run
    r.third = len(xf) - r.third_at
    var keeps = not r.is_export and nn_opt_keeps_third(r.opts[2], r.opts[3])
    if r.third < 0 or (r.third > 0) != keeps:
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
comptime SESSION_PER_PARAM_INTS = 4    # step count, has first buffer, has second buffer, has third buffer
comptime SESSION_PER_PARAM_FLOATS = 3  # beta1^t, beta2^t, mu product


def _session_record(ck: Checkpoint, path: String) raises -> List[Int64]:
    if ck.section_kind != CKPT_SECTION_SESSION:
        raise Error("mojolearn: '" + path + "' is not an operation-session checkpoint")
    var xi = ck.ext_ints.copy()
    if len(xi) < 2 or Int(xi[0]) != SESSION_RECORD_VERSION:
        raise Error("mojolearn: '" + path + "' holds an unknown session record version")
    var K = Int(xi[1])
    if K < 1 or K != ck.j_count() or len(xi) != 2 + 2 * K + 1 + SESSION_PER_PARAM_INTS * K + 4:
        raise Error("mojolearn: '" + path + "' holds a damaged session record")
    var thirds = 0
    for i in range(K):  # driver-loop: over parameter tensors
        var rows = Int(xi[2 + 2 * i])
        var cols = Int(xi[3 + 2 * i])
        if rows < 1 or cols < 1 or ck.offsets[i + 1] - ck.offsets[i] != rows * cols:
            raise Error(String(CKPT_REFUSE_SHAPE) + ": '" + path + "' tensor p" + String(i)
                        + " does not hold its recorded shape")
        var third = Int(xi[2 + 2 * K + 1 + SESSION_PER_PARAM_INTS * i + 3])
        if third != 0 and third != 1:
            raise Error("mojolearn: '" + path + "' holds a damaged session record")
        thirds += third * rows * cols
    # the per-parameter scalars, then the third state buffers of the parameters that have one
    if len(ck.ext_floats) != SESSION_PER_PARAM_FLOATS * K + thirds:
        raise Error("mojolearn: '" + path + "' holds a damaged session record")
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
    if (len(addrs) != 5 and len(addrs) != 6) or len(ip) != SEQ_SAVE_INTS or len(fp) < SEQ_FIT_FLOATS + 1:
        raise Error("model_save_checkpoint: requires 5 addresses (6 with the third optimizer state buffer), "
                    + String(SEQ_SAVE_INTS) + " integers and 9 floats plus one per layer")
    # lane nn-optim: fp may end in the schedule's NN_SCHED_ARGS words after the layers' floats
    # (lane nn-integrate: a version-2 description, an input_shape model, keeps its layer count after the shape slots,
    # as _seq_record reads it; words[2] is its first sample axis)
    var n_layers = words[SPEC2_HEADER - 1] if len(words) >= SPEC2_HEADER and words[0] == SPEC2_VERSION else (
        words[2] if len(words) > 2 else 0)
    var table_end = len(fp)
    var sched_args = List[Float64](length=NN_SCHED_ARGS, fill=0.0)
    # nn-blocks-pgseq: the parameter groups' words may follow the schedule's
    if len(fp) >= SEQ_FIT_FLOATS + n_layers + NN_SCHED_ARGS:
        table_end = SEQ_FIT_FLOATS + n_layers
        for k in range(NN_SCHED_ARGS):  # small-loop(8: the schedule's argument words)
            sched_args[k] = fp[table_end + k]
    var table = List[Float32]()
    for i in range(SEQ_FIT_FLOATS, table_end):  # driver-loop: over layers' float arguments
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
        unsafe_memcpy(dest=state.unsafe_ptr(), src=IP64(unsafe_from_address=sa), count=FIT_STATE_WORDS)
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
    for k in range(NN_SCHED_ARGS):  # small-loop(8: the schedule's argument words, float64 bits)
        xi.append(bitcast[DType.int64](sched_args[k]))
    # version 3: the option floats and the layers' floats as passed (float64 bits); xf keeps their float32
    # words, which the kernels read
    for i in range(6):  # small-loop(6: option floats lr .. f7)
        xi.append(bitcast[DType.int64](fp[i]))
    xi.append(bitcast[DType.int64](fp[7]))
    xi.append(bitcast[DType.int64](fp[8]))
    for i in range(SEQ_FIT_FLOATS, table_end):  # driver-loop: over layers' float arguments
        xi.append(bitcast[DType.int64](fp[i]))
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
    # nn-blocks-pgseq: the parameter groups' words (float64 bits) after a mark, when the model has groups
    var group_at = SEQ_FIT_FLOATS + n_layers + NN_SCHED_ARGS
    if len(fp) > group_at:
        xi.append(Int64(PGSEQ_RECORD_MARK))
        xi.append(Int64(len(fp) - group_at))
        for i in range(group_at, len(fp)):  # driver-loop: over the group words
            xi.append(bitcast[DType.int64](fp[i]))
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
    # lane stream-fits: centered RMSprop's third optimizer state buffer (record version 4)
    if not is_export and nn_opt_keeps_third(opt, flags):
        if len(addrs) != 6 or addrs[5] == 0:
            raise Error("model_save_checkpoint: centered RMSprop's third optimizer state buffer is missing")
        var third = _floats_of(fptr_word(addrs[5], "optimizer_w"), np_)
        for i in range(np_):  # driver-loop: over the buffer's words
            xf.append(third[i])
    ck.ext_ints = xi^
    ck.ext_floats = xf^
    var h = save_checkpoint(file, ck)
    return h.bytes


struct SeqCheckpointInfo(Movable):
    """model_checkpoint_info_py's answer as typed words: per layer (code,
    arg0, arg1) in `layer_ints` and its float in `layer_floats`; `ints` =
    the option words, n_features, n_outputs, y_1d, epochs run, class count,
    class kind, class item size, n_params; `floats` = lr, f1, f2, eps,
    weight_decay, f7, tol, schedule gamma, huber delta, then the schedule's
    NN_SCHED_ARGS argument words (lane nn-optim). `layer_args` holds each
    layer's SPEC2_ARGS argument words and `sample` the per-sample input shape
    of a version-2 description (empty: a 2-D model), so a model with an
    input_shape, flatten, convolutions or pools loads whole (lane
    nn-integrate, 2026-11-23)."""
    var is_export: Bool
    var layer_ints: List[Int]
    var layer_floats: List[Float64]
    var ints: List[Int]
    var floats: List[Float64]
    var layer_args: List[Int]
    var sample: List[Int]

    def __init__(out self):
        self.is_export = False
        self.layer_ints = List[Int]()
        self.layer_floats = List[Float64]()
        self.ints = List[Int]()
        self.floats = List[Float64]()
        self.layer_args = List[Int]()
        self.sample = List[Int]()


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
        out.layer_floats.append(r.layer_f64[i])
        for j in range(SPEC2_ARGS):  # small-loop(SPEC2_ARGS: six layer arguments): description words
            if len(spec.xargs) == SPEC2_ARGS * spec.n_layers():
                out.layer_args.append(spec.xargs[SPEC2_ARGS * i + j])
            else:
                out.layer_args.append(spec.arg0[i] if j == 0 else (spec.arg1[i] if j == 1 else 0))
    if spec.version == SPEC2_VERSION:
        for k in range(spec.sample.ndim):  # small-loop(at most RN_MAX_RANK - 1 sample axes)
            out.sample.append(spec.sample.d[k])
    for i in range(len(r.opts)):  # driver-loop: over option words
        out.ints.append(r.opts[i])
    out.ints.append(r.n_run)
    out.ints.append(r.k)
    out.ints.append(r.class_kind)
    out.ints.append(r.class_size)
    out.ints.append(spec.n_params)
    # the option floats as passed
    for i in range(6):  # driver-loop: over option floats (lr .. f7)
        out.floats.append(r.opt_f64[i])
    out.floats.append(bitcast[DType.float64](r.tol_bits))
    out.floats.append(r.opt_f64[6])
    out.floats.append(r.opt_f64[7])
    # lane nn-optim: then the schedule's NN_SCHED_ARGS argument words
    for k in range(NN_SCHED_ARGS):  # small-loop(8: the schedule's argument words)
        out.floats.append(r.sched_args[k])
    # nn-blocks-pgseq: then the parameter groups' words, when the model has groups
    for k in range(len(r.groups)):  # driver-loop: over the saved group words
        out.floats.append(r.groups[k])
    return out^


def model_load_checkpoint(file: String, addrs: List[Int]) raises -> Int:
    """Fill the caller's buffers from a Sequential checkpoint
    (model_load_checkpoint_py's contract). Returns n_params."""
    if len(addrs) != 6 and len(addrs) != 7:
        raise Error("model_load_checkpoint: requires 6 addresses (7 with the third optimizer state buffer)")
    var ck = _read_sequential(file)
    var r = _seq_record(ck, file)
    # lane stream-fits: centered RMSprop's third optimizer state buffer (record version 4)
    if r.third > 0:
        if r.third != len(ck.param):
            raise _damaged(file)
        if len(addrs) == 7 and addrs[6] != 0:
            unsafe_memcpy(dest=fptr_word(addrs[6], "optimizer_w"), src=ck.ext_floats.unsafe_ptr() + r.third_at,
                          count=r.third)
    _put_floats(fptr_word(addrs[0], "params"), ck.param)
    if addrs[1] != 0:
        _put_floats(fptr_word(addrs[1], "m"), ck.m_state)
    if addrs[2] != 0:
        _put_floats(fptr_word(addrs[2], "v"), ck.v_state)
    if addrs[3] != 0:
        unsafe_memcpy(dest=IP64(unsafe_from_address=addrs[3]), src=r.state.unsafe_ptr(), count=FIT_STATE_WORDS)
    if addrs[4] != 0 and r.n_run > 0:
        var curve = fptr_word(addrs[4], "loss_curve")
        unsafe_memcpy(dest=curve, src=ck.ext_floats.unsafe_ptr() + r.curve_at, count=r.n_run)
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
    var thirds = List[Int](length=K, fill=0)  # each parameter's third state buffer snapshot (0: none)
    for i in range(K):  # driver-loop: over parameter handles
        var off = ck.offsets[i]
        p[].download(hs[i], FP(unsafe_from_address=Int(ck.param.unsafe_ptr())) + off)
        if opt == 0:
            xi.append(Int64(0))
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
        thirds[i] = w[6]
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
        xi.append(Int64(1) if w[6] != 0 else Int64(0))
        xf.append(w[1])
        xf.append(w[2])
        xf.append(w[3])
    # the third state buffers (centered RMSprop's gradient average) after every parameter's scalars, in order
    for i in range(K):  # driver-loop: over parameter handles (their third state buffers)
        if thirds[i] == 0:
            continue
        var words = ck.offsets[i + 1] - ck.offsets[i]
        var third_at = len(xf)
        xf.resize(third_at + words, Float32(0))
        # the download writes the host words before it returns (sequence/exec_device.mojo download)
        p[].download(thirds[i], FP(unsafe_from_address=Int(xf.unsafe_ptr())) + third_at)
        p[].free(thirds[i])
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
    var third_at = SESSION_PER_PARAM_FLOATS * K  # the next third state buffer's words in the extension's floats
    for i in range(K):  # driver-loop: over parameter tensors
        var rows = Int(xi[2 + 2 * i])
        var cols = Int(xi[3 + 2 * i])
        var off = ck.offsets[i]
        var h = _upload_slice(p[], ck.param, off, rows, cols, grad)
        res.handles.append(h)
        var base = at + SESSION_PER_PARAM_INTS * i
        var third_here = third_at
        if xi[base + 3] != 0:
            third_at += rows * cols
        if opt == 0:
            continue
        var hm = 0
        var hv = 0
        var h3 = 0
        if xi[base + 1] != 0:
            hm = _upload_slice(p[], ck.m_state, off, rows, cols, False)
        if xi[base + 2] != 0:
            hv = _upload_slice(p[], ck.v_state, off, rows, cols, False)
        if xi[base + 3] != 0:
            h3 = _upload_slice(p[], ck.ext_floats, third_here, rows, cols, False)
        var ints = List[Int]()
        ints.append(Int(xi[base]))
        ints.append(hm)
        ints.append(hv)
        ints.append(h3)
        var fs = List[Float32]()
        for k in range(SESSION_PER_PARAM_FLOATS):  # driver-loop: over the three bias-correction words
            fs.append(ck.ext_floats[SESSION_PER_PARAM_FLOATS * i + k])
        p[].load_optimizer_state(opt, h, ints, fs)
        if hm != 0:
            p[].free(hm)
        if hv != 0:
            p[].free(hv)
        if h3 != 0:
            p[].free(h3)
    var c = at + SESSION_PER_PARAM_INTS * K
    res.has_opt = has_opt
    for i in range(4):  # driver-loop: over the four counters
        res.counters.append(Int(xi[c + i]))
    _ = ck^
    return res^


def _program_checkpoint[E: Exec & Movable & Deinitable, KIND: Int](sid: Int, pid: Int) raises -> Checkpoint:
    """The checkpoint record of program `pid` of session `sid` with its
    parameters' current values (downloaded once)."""
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
    return ck^


def ops_save_program[E: Exec & Movable & Deinitable, KIND: Int](sid: Int, pid: Int, file: String) raises -> Int:
    """Write program `pid` of session `sid` and its parameters' current
    values to `file`. Returns the byte count."""
    var ck = _program_checkpoint[E, KIND](sid, pid)
    var h = save_checkpoint(file, ck)
    return h.bytes


def ops_save_program_sha256[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, pid: Int, file: String,
) raises -> Tuple[String, Int]:
    """`ops_save_program`, returning (the SHA-256 of the written bytes, their
    count) from the in-memory body, so the caller never re-reads the file
    (row N167)."""
    var ck = _program_checkpoint[E, KIND](sid, pid)
    var saved = save_checkpoint_sha256(file, ck)
    return (saved[1].copy(), saved[0].bytes)


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
        # lane x1 (record version 3): carried state comes back as state; int32
        # words were saved as their bit patterns and keep them (a float32 copy)
        var pk = prog.param_kind(i)
        if pk == PROGRAM_PARAM_STATE_WORDS:
            p[].int_words[p[]._slot(prog.params[i])] = True
        if pk != PROGRAM_PARAM_WEIGHT:
            p[].declare_state(prog.params[i])
    var pid = p[].add_program(prog^)
    _ = ck^
    return pid
