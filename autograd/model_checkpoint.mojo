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
