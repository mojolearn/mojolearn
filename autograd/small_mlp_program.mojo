# SPDX-License-Identifier: Apache-2.0
"""The public small MLP (8 -> 16 -> 3, `SmallMLPTrainer`) as ONE captured
program on the executor (plan EXECUTOR_CONSOLIDATION_AND_CLEANUP row X2).

A trainer opens one inference session (`OpsSession`) and uploads its four
weights into it once: they are the program's parameters, read in place at
every run and updated in place by the AdamW step, so the session IS the
trainer's resident copy (no per-call upload, no second resident session).
The forward is captured once, at open, as ONE op, the small MLP's tuned
forward and hand backward (autograd/small_mlp_op.mojo; the leaf-128 GEMMs,
`_mlp_kernel` launches and backward chain of the 0.8.66 step):

    logits = small_mlp(x, weight1, bias1, weight2, bias2)  OPS_SMALL_MLP

or, under the A/B define -D MOJOLEARN_SMALL_MLP_GENERIC_OPS, the generic
form lane x2 introduced:

    t1     = transpose(weight1)                       OPS_TRANSPOSE
    hidden = linear(x, t1, bias1, relu)               OPS_LINEAR (act 1)
    t2     = transpose(weight2)                       OPS_TRANSPOSE
    logits = linear(hidden, t2, bias2)                OPS_LINEAR (act 0)

`train_step` replays it with recording on (the weights trainable leaves over
their session buffers), takes the mean cross-entropy against the class
codes, runs the tape backward (`backward_scalar_with_rules` with
StepOpRules, which carry the small_mlp VJP) and applies one AdamW update per weight through the
session's optimizer (`step_lr`, one OP_OPT launch each). `predict_logits`
is the same program with recording off. One wait per step: the loss, the
logits and the gradients come down together, the loss is refused when it is
not finite BEFORE the update is queued, so a refused step publishes nothing.

Bits: one algorithm on every executor (the small_mlp op's GEMM cells and
`_mlp_kernel` body, then the tape's cross-entropy and OP_OPT bodies, the
same on NVIDIA, AMD, Apple and the host column). This replaced the hand chain of training/mlp_ops.mojo
(`mlp_train_step_host`, its tape rules autograd/rules_small_mlp.mojo and
the Apple FAST fused/resident sessions of training/mlp_fast.mojo); the
small MLP's bits changed with it on every column together.

Entry words (`small_mlp_program_open` / `small_mlp_program_run`):

    OPEN      addresses [w1 16x8, b1 16, w2 3x16, b2 3, m 195, v 195]
              ints [t] (completed steps), floats [lr, beta1, beta2, eps, wd]
              -> the handle
    STEP      addresses [x, y (int32), loss (k floats), logits, dw1, db1,
              dw2, db2, dx]; ints [rows, mode, want_dx, k] -> rows * 3.
              mode 0 forward (y, loss, gradients unread; any rows >= 1),
              1 loss and gradients (weights read only; the parallel
              trainer's shard gradients), 2 the whole step (rows 1..256
              for modes 1 and 2). k > 1 (mode
              2 only, want_dx 0): k consecutive minibatches of `rows` rows,
              one loss each; logits and gradients are the last step's.
    DOWNLOAD  addresses [w1, b1, w2, b2, m, v] -> 195
    APPLY     addresses [g (195 packed gradients)] -> 195
    CLOSE     -> 0
    SESSIONS  (no handle; `small_mlp_sessions_forward`) addresses [y, x,
              w1 (hidden x width), b1, w2 (out x hidden), b2], ints [rows,
              width, hidden, out] -> rows * out: the same forward (OPS_SMALL_MLP,
              or the generic ops under the A/B define) for any Linear-ReLU-Linear
              shape on a session opened for the call. The public
              `training.mlp_inference_sessions` runs its packed session rows
              through it.
"""
# Inline tombstones carried here verbatim from files and blocks lane/m2-neural-leftovers deleted
# (training/mlp_ops.mojo, training/neural_session_mlp.mojo, bindings/parallel_mlp_native.mojo's old shard route):
# TOMBSTONE: the fused-gradient route through mlp_train_step (the deleted hand small-MLP step, plan row X2) deleted 2026-11-09 by lane/x2-small-mlp-on-executor; code recoverable at f59d913b0b. The worker gradients take the per-operation primitives below.
# Restore: git apply experiments/removed/MOJOLEARN_X2_SMALL_MLP_HAND_ROUTES.patch
# TOMBSTONE: mlp_train_step_host, its step GEMM dispatch (MOJOLEARN_IDN_MLP_GEMM_LEAF128_OFF), the tape rules autograd/rules_small_mlp.mojo and the Apple FAST fused step training/mlp_fast.mojo (superseded by SmallMLPTrainer's captured program, plan row X2) deleted 2026-11-09 by lane/x2-small-mlp-on-executor; code recoverable at f59d913b0b.
# Restore: git apply experiments/removed/MOJOLEARN_X2_SMALL_MLP_HAND_ROUTES.patch
# TOMBSTONE restored 2026-11-09 by lane/r1-restore-measured-winners as op small_mlp (OPS_SMALL_MLP, autograd/small_mlp_op.mojo + small_mlp_host.mojo): the step's leaf-128 GEMM dispatch (MOJOLEARN_IDN_MLP_GEMM_LEAF128_OFF honoured again), its `_mlp_kernel` launches and the rules_small_mlp backward chain, from f59d913b0b; the program captures it by default (A/B -D MOJOLEARN_SMALL_MLP_GENERIC_OPS). The per-call upload, host scans and the Apple FAST fused/resident sessions stay deleted (plumbing).
# TOMBSTONE: MOJOLEARN_AUTOGRAD_SMALL_MLP_OFF (removed: hand glue superseded by the tape) deleted 2026-10-19 by 3cd0063d7; code recoverable at 3cd0063d7^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_SMALL_MLP_OFF.patch
# TOMBSTONE (lane grid-act-5, 2026-10-08): neural_gemm_epilogue=mlp (NN06,
# -D MOJOLEARN_IDN_NEURAL_GEMM_EPILOGUE bit 1: the forward's GEMM + bias (+
# ReLU) as one thread-per-cell kernel, _mlp_gemm_epilogue_kernel /
# _mlp_fused_projection) was DELETED as an IDENTICAL grid ge123e6f9 loser:
# mlp-train-step NV 2.0 -> 3.8 ms / AMD 3.5 -> 3.4 ms (1.367x combined
# SLOWER), same bits. Recoverable at main 8ed94710a; the guard refuses bit 1.
# TOMBSTONE: the `registry` argument (sessions reading an open Apple FAST resident training session's weights; its only caller, mlp_sessions_resident, went with the session, plan row X2) deleted 2026-11-09 by lane/x2-small-mlp-on-executor; code recoverable at f59d913b0b.

from std.math import isfinite
from sequence.exec_trait import Exec
from sequence.ops import FP, OPT_ADAMW, Args, OP_FILL, OP_MLP_NONFINITE
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul
from core.cast_in import CastIn, CAST_KIND_SIGNED
from autograd.ops_api import (
    OpsSession, ops_session_open, ops_session, ops_session_close, OPS_LINEAR, OPS_TRANSPOSE,
)
from autograd.model import SCHED_CONSTANT
from autograd.program import OpsProgram
from autograd.tape import Tape
from autograd.program_replay_metadata import PreparedProgramReplay
from autograd.rankn import ShapeN
from autograd.cast_ops import upload_cast
from autograd.losses import cross_entropy
from autograd.rules_step_ops import StepOpRules
from autograd.small_mlp_op import OPS_SMALL_MLP
from std.sys.compile import is_defined
from autograd.optim import step_lr
from core.shared_slot import SharedSlot

#: Lane r1-restore-measured-winners (2026-11-09): the program captures the
#: small MLP's tuned op (OPS_SMALL_MLP, autograd/small_mlp_op.mojo: the
#: leaf-128 GEMMs, the `_mlp_kernel` launches and the hand backward chain
#: of the 0.8.66 step, which lane x2 had replaced by the generic ops below).
#: A/B: -D MOJOLEARN_SMALL_MLP_GENERIC_OPS captures lane x2's transpose +
#: OPS_LINEAR form instead. Run owed: mlp-train-step and mlp-predict, tuned
#: op vs generic ops, one run per arm on NVIDIA and AMD.
comptime SMALL_MLP_GENERIC_OPS = is_defined["MOJOLEARN_SMALL_MLP_GENERIC_OPS"]()

comptime SMALL_MLP_PROGRAM_OPEN = 0
comptime SMALL_MLP_PROGRAM_STEP = 1
comptime SMALL_MLP_PROGRAM_DOWNLOAD = 2
comptime SMALL_MLP_PROGRAM_APPLY = 3
comptime SMALL_MLP_PROGRAM_CLOSE = 4
comptime SMALL_MLP_PROGRAM_SESSIONS = 5

comptime SMALL_MLP_PROGRAM_FORWARD = 0
comptime SMALL_MLP_PROGRAM_GRADS = 1
comptime SMALL_MLP_PROGRAM_TRAIN = 2

comptime _IN = 8
comptime _HID = 16
comptime _OUT = 3
comptime _TOTAL = 195
comptime _MAX_ROWS = 256


def _offset(i: Int) -> Int:
    """Start of weight `i` in the packed registry [w1, b1, w2, b2]."""
    if i == 0:
        return 0
    if i == 1:
        return 128
    if i == 2:
        return 144
    if i == 3:
        return 192
    return _TOTAL


def _size(i: Int) -> Int:
    return _offset(i + 1) - _offset(i)


@fieldwise_init
struct _MlpProgram(ImplicitlyCopyable, Movable):
    var sid: Int
    var kind: Int
    var pid: Int
    var opt: Int
    #: session handles of weight1, bias1, weight2, bias2
    var w0: Int
    var w1: Int
    var w2: Int
    var w3: Int
    #: executor scratch reused call after call (addresses; capacity in rows)
    var x_buf: Int
    var y_buf: Int
    var g_buf: Int
    var cap_rows: Int

    def weight(self, i: Int) -> Int:
        if i == 0:
            return self.w0
        if i == 1:
            return self.w1
        if i == 2:
            return self.w2
        return self.w3


struct _MlpProgramStore(Defaultable, Movable):
    var entries: List[_MlpProgram]

    def __init__(out self):
        self.entries = List[_MlpProgram]()


comptime _STORE_NAME = "MojolearnSmallMlpProgramsIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnSmallMlpProgramsFastV1"
comptime _STORE = SharedSlot[_MlpProgramStore, _STORE_NAME]


def _find(handle: Int, kind: Int) raises -> Int:
    var store = _STORE.get_or_create_ptr()
    for i in range(len(store[].entries)):  # small-loop(open small MLP programs): a handle lookup, no data
        if store[].entries[i].sid == handle:
            if store[].entries[i].kind != kind:
                raise Error("small MLP program: handle belongs to another binding")
            return i
    raise Error("small MLP program: handle is not open")


def _bias_powers(t: Int, beta1: Float32, beta2: Float32) -> Tuple[Float32, Float32]:
    """beta1^t and beta2^t as the optimizer's running products reach them
    (autograd/optim.mojo optimizer_scalars: one FTZ product per step), so a
    resumed trainer continues with the words an uninterrupted one holds.
    Once both products flush to zero they stay zero, which bounds the walk
    (about 9e4 steps for beta2 = 0.999) whatever t is."""
    var pw1 = Float32(1.0)
    var pw2 = Float32(1.0)
    var k = 0
    while k < t and (pw1 != Float32(0.0) or pw2 != Float32(0.0)):  # driver-loop: over completed optimizer steps until both products flush
        pw1 = ftz(identical_mul(pw1, beta1))
        pw2 = ftz(identical_mul(pw2, beta2))
        k += 1
    return (pw1, pw2)


def small_mlp_program_open[E: Exec & Movable & Deinitable, KIND: Int](
    var ex: E, a: List[Int], p: List[Int], f: List[Float32],
) raises -> Int:
    """Open the trainer's session: upload the weights, capture the forward,
    create the AdamW optimizer and load the moments at step `t`."""
    if len(a) != 6 or len(p) != 1 or len(f) != 5:
        raise Error("small MLP program open takes 6 addresses, 1 integer and 5 floats")
    var t = p[0]
    if t < 0:
        raise Error("small MLP program open: the step count must be >= 0")
    var sid = ops_session_open[E, KIND](ex^, False)
    try:
        var s = ops_session[E, KIND](sid)
        var w0 = s[].upload(CastIn.f32_c(a[0], _HID, _IN), False)
        var w1 = s[].upload(CastIn.f32_c(a[1], 1, _HID), False)
        var w2 = s[].upload(CastIn.f32_c(a[2], _OUT, _HID), False)
        var w3 = s[].upload(CastIn.f32_c(a[3], 1, _OUT), False)
        # The capture's example input: one row (a run takes any row count;
        # no node's words depend on it).
        var probe = List[Float32](length=_IN, fill=Float32(0))
        var x = s[].upload(CastIn.f32_c(Int(probe.unsafe_ptr()), 1, _IN), False)
        var none_i = List[Int]()
        var none_f = List[Float32]()
        var logits: Int
        comptime if SMALL_MLP_GENERIC_OPS:
            var t1 = s[].op(OPS_TRANSPOSE, [w0], none_i, none_f)
            var hidden = s[].op(OPS_LINEAR, [x, t1, w1], [1], none_f)
            var t2 = s[].op(OPS_TRANSPOSE, [w2], none_i, none_f)
            logits = s[].op(OPS_LINEAR, [hidden, t2, w3], [0], none_f)
        else:
            logits = s[].op(OPS_SMALL_MLP, [x, w0, w1, w2, w3], none_i, none_f)
        var pid = s[].capture([x], [logits])
        s[].free(x)
        s[].reset()
        _ = probe^
        var o = s[].optimizer([OPT_ADAMW, 0, SCHED_CONSTANT, 1],
                              [f[0], Float32(0), f[1], f[2], f[3], f[4], Float32(0)])
        var oi = s[]._opt_index(o)
        var powers = _bias_powers(t, s[].opts[oi].cfg.f1(), s[].opts[oi].cfg.f2())
        var ws: List[Int] = [w0, w1, w2, w3]
        for i in range(4):  # driver-loop: over the four weight tensors
            var si = s[]._state_of(oi, ws[i], _size(i))
            s[].ex.upload(s[].opt_states[si].s1, FP(unsafe_from_address=a[4] + 4 * _offset(i)), _size(i))
            s[].ex.upload(s[].opt_states[si].s2, FP(unsafe_from_address=a[5] + 4 * _offset(i)), _size(i))
            s[].opt_states[si].t = t
            s[].opt_states[si].pw1 = powers[0]
            s[].opt_states[si].pw2 = powers[1]
        var x_buf = s[].ex.alloc(_MAX_ROWS * _IN)
        var y_buf = s[].ex.alloc(_MAX_ROWS)
        var g_buf = s[].ex.alloc(_TOTAL)
        s[].ex.sync()
        var store = _STORE.get_or_create_ptr()
        store[].entries.append(_MlpProgram(sid, KIND, pid, o, w0, w1, w2, w3,
                                           Int(x_buf), Int(y_buf), Int(g_buf), _MAX_ROWS))
    except err:
        ops_session_close[E, KIND](sid)
        raise err
    return sid


def _update[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], e: _MlpProgram, grads: List[FP],
) raises:
    """One AdamW update of the four weights, in place, from `grads` (one
    executor buffer per weight, registry order): one OP_OPT launch each."""
    var oi = s._opt_index(e.opt)
    var cfg = s.opts[oi].cfg
    var lr = s.learning_rate(e.opt)
    for i in range(4):  # driver-loop: over the four weight tensors
        var h = e.weight(i)
        var si = s._state_of(oi, h, _size(i))
        var st = s._state_view(si)
        var t = s.opt_states[si].t + 1
        step_lr(s.ex, cfg, st, s.buffer(h), grads[i], _size(i), t, lr)
        s.opt_states[si].t = t
        s.opt_states[si].pw1 = st.pw1
        s.opt_states[si].pw2 = st.pw2
        s.opt_states[si].mu_prod = st.mu_prod


def _grow[E: Exec & Movable & Deinitable](mut s: OpsSession[E], mut e: _MlpProgram, rows: Int) raises:
    """Scratch for `rows` input rows (k minibatches of train_steps)."""
    if rows <= e.cap_rows:
        return
    s.ex.free(FP(unsafe_from_address=e.x_buf))
    s.ex.free(FP(unsafe_from_address=e.y_buf))
    e.x_buf = Int(s.ex.alloc(rows * _IN))
    e.y_buf = Int(s.ex.alloc(rows))
    e.cap_rows = rows


def _run_steps[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], e: _MlpProgram, a: List[Int], rows: Int, mode: Int, want_dx: Bool, k: Int,
) raises:
    var prog = s.program(e.pid)
    var replay = PreparedProgramReplay(prog)
    var train = mode != SMALL_MLP_PROGRAM_FORWARD
    # program parameter j -> registry index (the capture numbers them by
    # first use; the registry order is the trainer's)
    var reg = List[Int]()
    var bufs = List[FP]()
    for j in range(len(prog.params)):  # driver-loop: over the program's four parameters
        var found = -1
        for i in range(4):  # small-loop(four weights): handle lookup, no data
            if e.weight(i) == prog.params[j]:
                found = i
        if found < 0:
            raise Error("small MLP program: a parameter is not one of the four weights")
        reg.append(found)
        bufs.append(s.buffer(prog.params[j]))
    var x_dev = FP(unsafe_from_address=e.x_buf)
    var y_dev = FP(unsafe_from_address=e.y_buf)
    var held = Tape()
    var temps = List[Int]()
    # The non-finite refusal of the returned words, on the device (lane m1-followups-misc; X2 had moved it to host
    # scans of the downloads): one flag word each for the logits, the gradients and the input gradient, filled 0
    # and set by OP_MLP_NONFINITE, read with the step's one wait.
    var dflag = s.ex.alloc(3)
    var flags = List[Float32](length=3, fill=Float32(0))
    var fa = Args()
    fa.p0 = dflag
    fa.f0 = Float32(0)
    s.ex.launch[OP_FILL](fa, 3)
    swap(held, s.prog_tape)
    swap(held, s.tape)
    try:
        for step in range(k):  # driver-loop: over the call's k minibatches
            s._drop_temps(temps)
            temps.clear()
            s.tape.reset()
            if train:
                s.tape.record()
            else:
                s.tape.no_grad()
            var last = step == k - 1
            var xh = s.leaf_nd(x_dev + step * rows * _IN, ShapeN.matrix(rows, _IN), want_dx and last, False)
            temps.append(xh)
            var phs = List[Int]()
            for j in range(len(bufs)):  # driver-loop: over the program's parameters
                var leaf = s.tape.variable_nd(s.ex, bufs[j], prog.param_shapes[j], train)
                var h = s._add(leaf, 0, train, 0)
                temps.append(h)
                phs.append(h)
            ref vals = replay.values
            vals.clear()
            vals.append(xh)
            ref hs = replay.handles
            for n in range(prog.n_nodes()):  # driver-loop: over the captured operations
                hs.clear()
                for j in range(prog.a_at[n], prog.a_at[n + 1]):  # small-loop(a node's input refs): handle lookup, no data
                    var rf = prog.args[j]
                    hs.append(phs[-rf - 1] if rf < 0 else vals[rf])
                var h = s._program_op(n, prog.code[n], hs, replay.nodes[n].ints, replay.nodes[n].fs)
                temps.append(h)
                vals.append(h)
            var out_ref = prog.outs[0]
            var out_h = phs[-out_ref - 1] if out_ref < 0 else vals[out_ref]
            var out = s.tensor(out_h)
            if last:
                var lv = s.tape.value_of(s.ex, out)
                _flag(s.ex, lv, rows * _OUT, dflag, 0)
                s.ex.download_async(FP(unsafe_from_address=a[3]), lv, rows * _OUT)
            if not train:
                s.ex.download_async(FP(unsafe_from_address=Int(flags.unsafe_ptr())), dflag, 3)
                s.ex.sync()
                _refuse(flags)
                continue
            var codes = s.tape.variable(s.ex, y_dev + step * rows, rows, 1, False)
            var loss = cross_entropy(s.tape, s.ex, out, codes)
            # StepOpRules: ImportRules plus the small_mlp op's VJP (no step
            # entries, so no scratch)
            s.tape.backward_scalar_with_rules(s.ex, loss, StepOpRules(0))
            var loss_dst = a[2] + 4 * step
            s.ex.download_async(FP(unsafe_from_address=loss_dst), s.tape.value(loss), 1)
            var grads = List[FP](length=4, fill=bufs[0])
            for j in range(len(phs)):  # driver-loop: over the program's parameters
                grads[reg[j]] = s.tape.gradient(s.tensor(phs[j]))
            if last:
                for i in range(4):  # driver-loop: over the four gradients
                    _flag(s.ex, grads[i], _size(i), dflag, 1)
                    s.ex.download_async(FP(unsafe_from_address=a[4 + i]), grads[i], _size(i))
                if want_dx:
                    var gx = s.tape.gradient(s.tensor(xh))
                    _flag(s.ex, gx, rows * _IN, dflag, 2)
                    s.ex.download_async(FP(unsafe_from_address=a[8]), gx, rows * _IN)
                s.ex.download_async(FP(unsafe_from_address=Int(flags.unsafe_ptr())), dflag, 3)
            # ONE wait per step: the refusal reads the loss before any update
            # is queued, so a refused step leaves the weights as they were.
            s.ex.sync()
            if not isfinite(FP(unsafe_from_address=loss_dst)[]):
                raise Error("small MLP train step: loss is not finite")
            if last:
                _refuse(flags)
            if mode == SMALL_MLP_PROGRAM_TRAIN:
                _update(s, e, grads)
        s._drop_temps(temps)
        s.tape.no_grad()
    except err:
        s._drop_temps(temps)
        s.tape.no_grad()
        swap(held, s.tape)
        swap(held, s.prog_tape)
        s.ex.free(dflag)
        raise err
    swap(held, s.tape)
    swap(held, s.prog_tape)
    s.ex.free(dflag)
    _ = flags^


def _flag[E: Exec](mut ex: E, src: FP, n: Int, dflag: FP, slot: Int) raises:
    """dflag[slot] = 1 when any of src[0:n] is NaN or infinite (OP_MLP_NONFINITE)."""
    var a = Args()
    a.p0 = src
    a.p1 = dflag
    a.i0 = slot
    ex.launch[OP_MLP_NONFINITE](a, n)


def _refuse(flags: List[Float32]) raises:
    """The refusal the binding's host scans raised, from the device flags."""
    if flags[0] != 0:
        raise Error("RuntimeError: SmallMLPTrainer logits returned an invalid result")
    if flags[1] != 0:
        raise Error("RuntimeError: SmallMLPTrainer gradient returned an invalid result")
    if flags[2] != 0:
        raise Error("RuntimeError: SmallMLPTrainer input gradient returned an invalid result")


def small_mlp_program_run[E: Exec & Movable & Deinitable, KIND: Int](
    op: Int, a: List[Int], p: List[Int], f: List[Float32],
) raises -> Int:
    """STEP, DOWNLOAD, APPLY or CLOSE on an open program (header)."""
    if len(p) < 1:
        raise Error("small MLP program: the handle comes first")
    var handle = p[0]
    var index = _find(handle, KIND)
    var store = _STORE.get_or_create_ptr()
    var e = store[].entries[index]
    var s = ops_session[E, KIND](e.sid)
    if op == SMALL_MLP_PROGRAM_CLOSE:
        _ = store[].entries.pop(index)
        s[].ex.free(FP(unsafe_from_address=e.x_buf))
        s[].ex.free(FP(unsafe_from_address=e.y_buf))
        s[].ex.free(FP(unsafe_from_address=e.g_buf))
        ops_session_close[E, KIND](e.sid)
        return 0
    if op == SMALL_MLP_PROGRAM_DOWNLOAD:
        if len(a) != 6:
            raise Error("small MLP program download takes 6 addresses")
        var oi = s[]._opt_index(e.opt)
        for i in range(4):  # driver-loop: over the four weight tensors
            var h = e.weight(i)
            var si = s[]._state_of(oi, h, _size(i))
            s[].ex.download_async(FP(unsafe_from_address=a[i]), s[].buffer(h), _size(i))
            s[].ex.download_async(FP(unsafe_from_address=a[4] + 4 * _offset(i)), s[].opt_states[si].s1, _size(i))
            s[].ex.download_async(FP(unsafe_from_address=a[5] + 4 * _offset(i)), s[].opt_states[si].s2, _size(i))
        s[].ex.sync()
        return _TOTAL
    if op == SMALL_MLP_PROGRAM_APPLY:
        if len(a) != 1:
            raise Error("small MLP program apply takes the packed gradients")
        var g = FP(unsafe_from_address=e.g_buf)
        s[].ex.upload(g, FP(unsafe_from_address=a[0]), _TOTAL)
        var grads = List[FP]()
        for i in range(4):  # driver-loop: over the four weight tensors
            grads.append(g + _offset(i))
        _update(s[], e, grads)
        s[].ex.sync()
        return _TOTAL
    if op != SMALL_MLP_PROGRAM_STEP:
        raise Error("small MLP program: unknown operation " + String(op))
    if len(a) != 9 or len(p) != 5:
        raise Error("small MLP program step takes 9 addresses and 5 integers")
    var rows = p[1]
    var mode = p[2]
    var want_dx = p[3] != 0
    var k = p[4]
    # Training modes keep the trainer's 1..256 batch; the forward takes any
    # positive row count (the inference sessions run their packed rows as one
    # forward; _grow sizes the scratch, no node depends on the row count).
    if rows < 1 or (rows > _MAX_ROWS and mode != SMALL_MLP_PROGRAM_FORWARD):
        raise Error("small MLP operations require rows 1..256")
    if mode < SMALL_MLP_PROGRAM_FORWARD or mode > SMALL_MLP_PROGRAM_TRAIN:
        raise Error("small MLP train step: mode must be 0 (forward), 1 (grads) or 2 (train)")
    if k < 1 or (k > 1 and (mode != SMALL_MLP_PROGRAM_TRAIN or want_dx)):
        raise Error("small MLP train step: k > 1 runs whole steps without the input gradient")
    _grow(s[], e, k * rows)
    store[].entries[index] = e
    upload_cast(s[].ex, FP(unsafe_from_address=e.x_buf), CastIn.f32_c(a[0], k * rows, _IN))
    if mode != SMALL_MLP_PROGRAM_FORWARD:
        upload_cast(s[].ex, FP(unsafe_from_address=e.y_buf),
                    CastIn(a[1], CAST_KIND_SIGNED, 4, k * rows, 1, 4, 4))
    _run_steps(s[], e, a, rows, mode, want_dx, k)
    return rows * _OUT


def small_mlp_sessions_forward[E: Exec & Movable & Deinitable, KIND: Int](
    var ex: E, a: List[Int], p: List[Int],
) raises -> Int:
    """SESSIONS: `y = linear(relu(linear(x, w1^T, b1)), w2^T, b2)` for any
    (width, hidden, out), on an inference session opened for this call:
    the trainer's forward op (OPS_SMALL_MLP; the A/B define: OPS_TRANSPOSE,
    OPS_LINEAR act 1 / act 0), so
    an 8-16-3 model gives `predict_logits`' words. `x` is the sessions' rows
    packed in order; each row's logits depend only on that row. Refuses a
    non-finite input, weight or logit from device flags, one wait."""
    if len(a) != 6 or len(p) != 4:
        raise Error("neural MLP sessions: 6 addresses and [rows, width, hidden, out]")
    var rows = p[0]
    var width = p[1]
    var hidden = p[2]
    var out_w = p[3]
    if rows < 1 or width < 1 or hidden < 1 or out_w < 1:
        raise Error("neural MLP sessions: invalid metadata")
    var limit = 2147483647
    var widest = max(width, max(hidden, out_w))
    if hidden > limit // width or out_w > limit // hidden or rows > limit // widest:
        raise Error("neural MLP sessions: activation exceeds index range")
    var sid = ops_session_open[E, KIND](ex^, False)
    try:
        var s = ops_session[E, KIND](sid)
        var w0 = s[].upload(CastIn.f32_c(a[2], hidden, width), False)
        var w1 = s[].upload(CastIn.f32_c(a[3], 1, hidden), False)
        var w2 = s[].upload(CastIn.f32_c(a[4], out_w, hidden), False)
        var w3 = s[].upload(CastIn.f32_c(a[5], 1, out_w), False)
        var x = s[].upload(CastIn.f32_c(a[1], rows, width), False)
        var dflag = s[].ex.alloc(2)
        var fa = Args()
        fa.p0 = dflag
        fa.f0 = Float32(0)
        s[].ex.launch[OP_FILL](fa, 2)
        var bx = s[].buffer(x)
        var b0 = s[].buffer(w0)
        var b1 = s[].buffer(w1)
        var b2 = s[].buffer(w2)
        var b3 = s[].buffer(w3)
        _flag(s[].ex, bx, rows * width, dflag, 0)
        _flag(s[].ex, b0, hidden * width, dflag, 0)
        _flag(s[].ex, b1, hidden, dflag, 0)
        _flag(s[].ex, b2, out_w * hidden, dflag, 0)
        _flag(s[].ex, b3, out_w, dflag, 0)
        var none_i = List[Int]()
        var none_f = List[Float32]()
        # the trainer's forward op (OPS_SMALL_MLP, or lane x2's generic ops
        # under the A/B define), so an 8-16-3 model gives predict_logits' words
        var logits: Int
        comptime if SMALL_MLP_GENERIC_OPS:
            var t1 = s[].op(OPS_TRANSPOSE, [w0], none_i, none_f)
            var hid = s[].op(OPS_LINEAR, [x, t1, w1], [1], none_f)
            var t2 = s[].op(OPS_TRANSPOSE, [w2], none_i, none_f)
            logits = s[].op(OPS_LINEAR, [hid, t2, w3], [0], none_f)
        else:
            logits = s[].op(OPS_SMALL_MLP, [x, w0, w1, w2, w3], none_i, none_f)
        var lt = s[].tensor(logits)
        var lv = s[].tape.value_of(s[].ex, lt)
        _flag(s[].ex, lv, rows * out_w, dflag, 1)
        var flags = List[Float32](length=2, fill=Float32(0))
        s[].ex.download_async(FP(unsafe_from_address=a[0]), lv, rows * out_w)
        s[].ex.download_async(FP(unsafe_from_address=Int(flags.unsafe_ptr())), dflag, 2)
        s[].ex.sync()
        s[].ex.free(dflag)
        if flags[0] != 0:
            raise Error("neural MLP sessions: nonfinite input or weights")
        if flags[1] != 0:
            raise Error("neural MLP sessions: nonfinite logits")
        _ = flags^
    except err:
        ops_session_close[E, KIND](sid)
        raise err
    ops_session_close[E, KIND](sid)
    return rows * out_w
