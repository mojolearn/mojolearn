# SPDX-License-Identifier: Apache-2.0
"""The byte LM (`SmallByteLanguageModelTrainer`) as ONE captured program on
the executor (plan EXECUTOR_CONSOLIDATION_AND_CLEANUP row X7, lane
x7-byte-lm-on-executor, 2026-11-09).

A trainer opens one inference session (`OpsSession` on the trainer's
DeviceContext) whose tensors are the model: the embedding table (V, d),
each decoder layer's nine weights packed into ONE step-op weights tensor
(`block_weight_sizes(OPS_ATTN_STEP)` order, `block_offsets` slots), the
head (V, d), and per layer the attention step's carried state (keys and
values of `length` positions, the cached-token word). The training forward
is captured once, at open:

    x      = embedding(embed, ids)                          OPS_EMBEDDING
    x      = attention_step(x, block_l, kv_l, words_l)      OPS_ATTN_STEP, l = 0 .. n_layers - 1
    logits = x @ transpose(lm_head)                         OPS_TRANSPOSE, OPS_MATMUL
    loss   = cross_entropy(logits, reshape(targets))        OPS_RESHAPE, OPS_CROSS_ENTROPY (mean)

with inputs (ids, targets) = (row[:length], row[1:]) of every (length + 1)
token row. `train_step` / `train_steps` are Program.fit over it
(autograd/program_runtime.mojo `train_program_control`): one epoch whose
minibatches are the call's step batches in order (no shuffle), the AdamW
step of the session's optimizer per trained tensor after every minibatch,
the step losses as Program.fit's per-minibatch curve, and the last step's
pre-update gradients copied out (`grads_out`). The attention steps run
recorded (a zero-state prefill at position 0, stages kept) and their
backward is the transformer stage rule (StepOpRules,
autograd/rules_step_ops.mojo). Evaluation is the same captured forward
without the backward.

The trainer's flat buffers (training/byte_lm.mojo `ByteBuffers` on the
device, `ByteHostSession`'s lists on the host column: param, m, v, grad in
`ByteConfig.offsets` order) stay the COMMITTED state that export, logits
and rollback read: `byte_program_load` / `byte_host_program_load` copy
them into the session (once after open and after any outside write),
`byte_program_store` / `byte_host_program_store` copy a finished call
back. Both are copies per tensor piece, no arithmetic.

Bits: one algorithm on every column. The program is generic over the
executor (`ByteLmProgram[E, KIND]`): the GPU columns run it on a device
session, the CPU column (lane x7b-step-ops-host-arm) on a host session,
where the attention step is the step op's host arm (the host block body
and its backward, autograd/block_step_ops.mojo) and the embedding, GEMM,
cross-entropy and OP_OPT AdamW are the executor's host bodies. This
replaced the hand per-layer forward (`_byte_forward_loss`), the tape
backward wrapper (`_byte_backward_tape`, autograd/rules_lm.mojo), the
certified optimizer step of the byte LM and the host column's composed
oracle step (`byte_host_train_step`); the bits changed with them on every
column together.
"""
from std.memory import memcpy
from max.gpu.host import DeviceBuffer, DeviceContext
from sequence.ops import FP, Args, OP_COPY, OPT_ADAMW
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from core.cast_in import CastIn, CAST_KIND_SIGNED, CAST_KIND_FLOAT
from autograd.ops_api import (
    OpsSession, ops_session_open, ops_session, ops_session_close, OPS_DEVICE, OPS_HOST,
    OPS_EMBEDDING, OPS_TRANSPOSE, OPS_MATMUL, OPS_RESHAPE, OPS_CROSS_ENTROPY,
)
from autograd.block_step_ops import (
    OPS_ATTN_STEP, BLOCK_PART_WEIGHTS, BLOCK_PART_STATE, BLOCK_PART_WORDS, block_weight_sizes, block_offsets,
    block_guard, block_part_floats, block_buffer_view,
)
from autograd.model import SCHED_CONSTANT
from autograd.losses import REDUCTION_MEAN
from autograd.program_runtime import train_program_control, parameter_buffers, replay_values_prepared
from autograd.program_replay_metadata import PreparedProgramReplay
from autograd.rankn import ShapeN
from autograd.tape import Tape
from autograd.cast_ops import upload_cast, upload_cast_i32
from autograd.optim import step_lr
from autograd.rules_step_ops import StepOpRules
from autograd.small_mlp_program import _bias_powers
from training.byte_lm_config import ByteConfig
from training.checks.optimizer_contract import OptimizerConfig

#: the parts of the state `_move` copies
comptime BYTE_PART_PARAM = 0
comptime BYTE_PART_M = 1
comptime BYTE_PART_V = 2
comptime BYTE_PART_GRAD = 3


def _piece(k: Int) -> Int:
    """Slot in the step op's weights tensor of the byte LM's block tensor k
    (`byte_param_name` order norm1, q, k, v, o, norm2, gate, up, down; the
    step op's order norm1, norm2, q, k, v, o, gate, up, down)."""
    if k == 0:
        return 0
    if k == 5:
        return 1
    if k <= 4:
        return k + 1
    return k


struct ByteLmProgram[E: Exec & Movable & Deinitable, KIND: Int](Movable):
    """One trainer's session (on executor E: a GPU column's DeviceExec or
    the CPU column's HostExec), its captured training forward, its
    optimizer and the tensor handles (header)."""

    var sid: Int
    var pid: Int
    var opt: Int
    var emb: Int
    var head: Int
    var blocks: List[Int]
    var kv: List[Int]
    var words: List[Int]
    #: the attention step's integer arguments (b, d_model, n_heads, n_kv,
    #: head_dim, intermediate, s_max = length, window 0)
    var ints: List[Int]
    #: weights tensor floats and piece starts of one block
    var wn: Int
    var at: List[Int]
    var config: ByteConfig
    #: executor scratch: the last step's gradient per trained tensor
    #: (embedding, the blocks, the head), and one batch of ids and targets
    var grads: List[Int]
    var ids_buf: Int
    var tgt_buf: Int

    def __init__(out self, sid: Int, config: ByteConfig):
        self.sid = sid
        self.pid = -1
        self.opt = -1
        self.emb = -1
        self.head = -1
        self.blocks = List[Int]()
        self.kv = List[Int]()
        self.words = List[Int]()
        self.ints = List[Int]()
        self.wn = 0
        self.at = List[Int]()
        self.config = config.copy()
        self.grads = List[Int]()
        self.ids_buf = 0
        self.tgt_buf = 0

    def __deinit__(deinit self):
        """The session and its storage go with the program (the scratch
        buffers are executor allocations, returned at the session's close)."""
        try:
            ops_session_close[Self.E, Self.KIND](self.sid)
        except:
            pass

    def trained(self) -> List[Int]:
        """The trained tensors, in the gradient scratch order."""
        var out = List[Int]()
        out.append(self.emb)
        for l in range(len(self.blocks)):  # driver-loop: over decoder layers
            out.append(self.blocks[l])
        out.append(self.head)
        return out^


comptime ByteDeviceProgram = ByteLmProgram[DeviceExec, OPS_DEVICE]
comptime ByteHostProgram = ByteLmProgram[HostExec, OPS_HOST]


def _zeros[E: Exec & Movable & Deinitable](mut s: OpsSession[E], n: Int) raises -> FP:
    """n zero floats of the executor's own storage (the host executor's
    allocation is zeroed; the device's is filled)."""
    var buf = s.ex.alloc(n)
    comptime if E.IS_DEVICE:
        var view = block_buffer_view(rebind[DeviceExec](s.ex), buf, n)
        view.enqueue_fill(Float32(0.0))
    return buf


def _blank[E: Exec & Movable & Deinitable](mut s: OpsSession[E], rows: Int, cols: Int) raises -> Int:
    """A zero (rows, cols) session tensor of the executor's own storage
    (`upload` without a host source: the trainer's committed state is
    copied into it by `byte_program_load` / `byte_host_program_load`)."""
    var buf = _zeros(s, rows * cols)
    var t = s.tape.variable(s.ex, buf, rows, cols, False)
    return s._add(t, Int(buf), False, 0)


def byte_program_open(ctx: DeviceContext, config: ByteConfig, cfg: OptimizerConfig) raises -> ByteDeviceProgram:
    """The trainer's program on a GPU column (`ctx`)."""
    return _program_open[DeviceExec, OPS_DEVICE](DeviceExec(ctx), config, cfg)


def byte_host_program_open(config: ByteConfig, cfg: OptimizerConfig) raises -> ByteHostProgram:
    """The trainer's program on the CPU column (lane x7b): the same capture
    on a host session."""
    return _program_open[HostExec, OPS_HOST](HostExec(), config, cfg)


def _program_open[E: Exec & Movable & Deinitable, KIND: Int](
    var ex: E, config: ByteConfig, cfg: OptimizerConfig,
) raises -> ByteLmProgram[E, KIND]:
    """Open the trainer's session on `ex`: the model tensors (zero until
    the load), the AdamW optimizer and the captured training forward."""
    config.validate()
    var sid = ops_session_open[E, KIND](ex^, False)
    var e = ByteLmProgram[E, KIND](sid, config)
    var s = ops_session[E, KIND](sid)
    var b = config.batch
    var length = config.length
    var dm = config.d_model
    var vocab = config.vocab_size
    var ints: List[Int] = [b, dm, config.n_heads, config.n_kv, config.head_dim, config.intermediate, length, 0]
    e.ints = ints.copy()
    var sizes = block_weight_sizes(OPS_ATTN_STEP, ints)
    e.at = block_offsets(sizes, block_guard(OPS_ATTN_STEP))
    e.wn = block_part_floats(OPS_ATTN_STEP, ints, BLOCK_PART_WEIGHTS)
    var sn = block_part_floats(OPS_ATTN_STEP, ints, BLOCK_PART_STATE)
    e.emb = _blank(s[], vocab, dm)
    for _l in range(config.n_layers):  # driver-loop: over decoder layers
        e.blocks.append(_blank(s[], 1, e.wn))
        var st = _blank(s[], 1, sn)
        s[].declare_state(st)
        e.kv.append(st)
        var w0: List[Int] = [0]
        e.words.append(s[].upload_block(OPS_ATTN_STEP, ints, BLOCK_PART_WORDS, List[CastIn](), w0))
    e.head = _blank(s[], vocab, dm)
    e.opt = s[].optimizer([OPT_ADAMW, 0, SCHED_CONSTANT, 1],
                          [cfg.lr, Float32(0), cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay, Float32(0)])
    var trained = e.trained()
    var oi = s[]._opt_index(e.opt)
    for i in range(len(trained)):  # driver-loop: over the trained tensors (moments and gradient scratch)
        var n = s[].shapes[s[]._slot(trained[i])].count()
        _ = s[]._state_of(oi, trained[i], n)
        e.grads.append(Int(_zeros(s[], n)))
    e.ids_buf = Int(s[].ex.alloc(b * length))
    e.tgt_buf = Int(s[].ex.alloc(b * length))
    # The capture's example batch (zeros): the forward runs once here, in
    # the inference session, and is logged; no node's words depend on it.
    var probe = List[Int32](length=b * length, fill=Int32(0))
    var src = CastIn(Int(probe.unsafe_ptr()), CAST_KIND_SIGNED, 4, b, length, length * 4, 4)
    var ids = s[].upload_ids(src, ShapeN.matrix(b, length))
    var tgt = s[].upload(src, False)
    var none_i = List[Int]()
    var none_f = List[Float32]()
    var pad: List[Int] = [-1]
    var x = s[].op(OPS_EMBEDDING, [e.emb, ids], pad, none_f)
    for l in range(config.n_layers):  # driver-loop: over decoder layers
        x = s[].op(OPS_ATTN_STEP, [x, e.blocks[l], e.kv[l], e.words[l]], ints, none_f)
    var t = s[].op(OPS_TRANSPOSE, [e.head], none_i, none_f)
    var logits = s[].op(OPS_MATMUL, [x, t], none_i, none_f)
    var shape: List[Int] = [b * length, 1]
    var codes = s[].op(OPS_RESHAPE, [tgt], shape, none_f)
    var mean: List[Int] = [REDUCTION_MEAN]
    var loss = s[].op(OPS_CROSS_ENTROPY, [logits, codes], mean, none_f)
    e.pid = s[].capture([ids, tgt], [loss])
    s[].free(ids)
    s[].free(tgt)
    s[].reset()
    _ = probe^
    s[].ex.sync()
    return e^


def _part_fp[E: Exec & Movable & Deinitable, KIND: Int](
    mut s: OpsSession[E], e: ByteLmProgram[E, KIND], h: Int, n: Int, part: Int, slot: Int,
) raises -> FP:
    if part == BYTE_PART_PARAM:
        return s.buffer(h)
    if part == BYTE_PART_GRAD:
        return FP(unsafe_from_address=e.grads[slot])
    var si = s._state_of(s._opt_index(e.opt), h, n)
    return s.opt_states[si].s1 if part == BYTE_PART_M else s.opt_states[si].s2


def _piece_at[E: Exec & Movable & Deinitable, KIND: Int](
    mut s: OpsSession[E], e: ByteLmProgram[E, KIND], j: Int, part: Int,
) raises -> FP:
    """Where flat tensor j (`ByteConfig.offsets` order) of `part` lives in
    the session: the embedding, a block's piece slot, or the head."""
    var c = e.config.copy()
    var o = c.offsets()
    var n = o[j + 1] - o[j]
    if j == 0:
        return _part_fp(s, e, e.emb, n, part, 0)
    if j == c.n_tensors() - 1:
        return _part_fp(s, e, e.head, n, part, c.n_layers + 1)
    var layer = (j - 1) // 9
    return _part_fp(s, e, e.blocks[layer], e.wn, part, 1 + layer) + e.at[_piece((j - 1) % 9)]


def _move(mut s: OpsSession[DeviceExec], e: ByteDeviceProgram, flat: DeviceBuffer[DType.float32], part: Int,
          into_program: Bool) raises:
    """Copy `part` between the trainer's flat device buffer
    (`ByteConfig.offsets` order) and the session's tensors, one device copy
    per tensor piece, enqueued on the session's stream (the trainer's
    context)."""
    var c = e.config.copy()
    var o = c.offsets()
    var ctx = s.ex.ctx.copy()
    for j in range(c.n_tensors()):  # driver-loop: over model parameter tensors
        var n = o[j + 1] - o[j]
        var pv = block_buffer_view(s.ex, _piece_at(s, e, j, part), n)
        var fv = flat.create_sub_buffer[DType.float32](o[j], n)
        if into_program:
            ctx.enqueue_copy(dst_buf=pv, src_buf=fv)
        else:
            ctx.enqueue_copy(dst_buf=fv, src_buf=pv)


def _move_host(mut s: OpsSession[HostExec], e: ByteHostProgram, flat: Int, part: Int, into_program: Bool) raises:
    """`_move` on the CPU column: `flat` is the host address of the
    session owner's flat list; one block copy per tensor piece."""
    var c = e.config.copy()
    var o = c.offsets()
    for j in range(c.n_tensors()):  # driver-loop: over model parameter tensors
        var n = o[j + 1] - o[j]
        var p = _piece_at(s, e, j, part)
        var f = FP(unsafe_from_address=flat) + o[j]
        if n > 0:
            if into_program:
                memcpy(dest=p, src=f, count=n)
            else:
                memcpy(dest=f, src=p, count=n)


def _loaded[E: Exec & Movable & Deinitable, KIND: Int](mut e: ByteLmProgram[E, KIND], completed: Int) raises:
    """After a load: every trained tensor's step count `completed` with the
    running bias-correction products of an uninterrupted run, and the
    attention entries re-check their weights."""
    var s = ops_session[E, KIND](e.sid)
    var oi = s[]._opt_index(e.opt)
    var powers = _bias_powers(completed, s[].opts[oi].cfg.f1(), s[].opts[oi].cfg.f2())
    var trained = e.trained()
    for i in range(len(trained)):  # driver-loop: over the trained tensors (host optimizer scalars)
        var n = s[].shapes[s[]._slot(trained[i])].count()
        var si = s[]._state_of(oi, trained[i], n)
        s[].opt_states[si].t = completed
        s[].opt_states[si].pw1 = powers[0]
        s[].opt_states[si].pw2 = powers[1]
    # the weights were written from outside a step: the attention entries
    # re-check them at their next run
    for l in range(len(e.blocks)):  # driver-loop: over decoder layers
        s[].steps.weights_changed(Int(s[].buffer(e.blocks[l])))


def byte_program_load(mut e: ByteDeviceProgram, param: DeviceBuffer[DType.float32], m: DeviceBuffer[DType.float32],
                      v: DeviceBuffer[DType.float32], completed: Int) raises:
    """The trainer's committed state into the session: parameters, AdamW
    moments, and every trained tensor's step count `completed`."""
    var s = ops_session[DeviceExec, OPS_DEVICE](e.sid)
    _move(s[], e, param, BYTE_PART_PARAM, True)
    _move(s[], e, m, BYTE_PART_M, True)
    _move(s[], e, v, BYTE_PART_V, True)
    _loaded(e, completed)


def byte_host_program_load(mut e: ByteHostProgram, param: Int, m: Int, v: Int, completed: Int) raises:
    """`byte_program_load` on the CPU column: the session owner's flat
    lists (host addresses) into the session."""
    var s = ops_session[HostExec, OPS_HOST](e.sid)
    _move_host(s[], e, param, BYTE_PART_PARAM, True)
    _move_host(s[], e, m, BYTE_PART_M, True)
    _move_host(s[], e, v, BYTE_PART_V, True)
    _loaded(e, completed)


def byte_host_program_load_params(mut e: ByteHostProgram, param: Int) raises:
    """The owner's parameters alone into the session (a loss-only call)."""
    var s = ops_session[HostExec, OPS_HOST](e.sid)
    _move_host(s[], e, param, BYTE_PART_PARAM, True)
    for l in range(len(e.blocks)):  # driver-loop: over decoder layers
        s[].steps.weights_changed(Int(s[].buffer(e.blocks[l])))


def byte_host_program_store(mut e: ByteHostProgram, param: Int, m: Int, v: Int, grad: Int) raises:
    """`byte_program_store` on the CPU column: parameters, moments and the
    last step's pre-update gradient into the owner's flat lists."""
    var s = ops_session[HostExec, OPS_HOST](e.sid)
    _move_host(s[], e, param, BYTE_PART_PARAM, False)
    _move_host(s[], e, m, BYTE_PART_M, False)
    _move_host(s[], e, v, BYTE_PART_V, False)
    _move_host(s[], e, grad, BYTE_PART_GRAD, False)


def byte_program_load_params(mut e: ByteDeviceProgram, param: DeviceBuffer[DType.float32]) raises:
    """The trainer's committed parameters alone into the session (a pooled
    optimizer's shard: its moments are a slice and its update is the pool's
    group update, so the program only takes gradients there)."""
    var s = ops_session[DeviceExec, OPS_DEVICE](e.sid)
    _move(s[], e, param, BYTE_PART_PARAM, True)
    for l in range(len(e.blocks)):  # driver-loop: over decoder layers
        s[].steps.weights_changed(Int(s[].buffer(e.blocks[l])))


def byte_program_store(mut e: ByteDeviceProgram, param: DeviceBuffer[DType.float32], m: DeviceBuffer[DType.float32],
                       v: DeviceBuffer[DType.float32], grad: DeviceBuffer[DType.float32]) raises:
    """The session's state after a call into the trainer's flat buffers:
    parameters, moments and the last step's pre-update gradient. No wait
    (the caller's next device scan is on the same stream)."""
    var s = ops_session[DeviceExec, OPS_DEVICE](e.sid)
    _move(s[], e, param, BYTE_PART_PARAM, False)
    _move(s[], e, m, BYTE_PART_M, False)
    _move(s[], e, v, BYTE_PART_V, False)
    _move(s[], e, grad, BYTE_PART_GRAD, False)


def byte_program_fit[E: Exec & Movable & Deinitable, KIND: Int](
    mut e: ByteLmProgram[E, KIND], stack: Int, k: Int, losses: Int,
) raises -> Float32:
    """`k` training steps as ONE Program.fit: `stack` is the host address of
    k * batch rows of length + 1 int32 token ids (the step batches in
    order), `losses` the host address of k float32 step losses. The last
    step's pre-update gradients land in the gradient scratch. Returns the
    call's mean step loss (Program.fit's epoch curve: non-finite when any
    step's loss is)."""
    if k < 1:
        raise Error("byte LM: a training call takes at least one step")
    var s = ops_session[E, KIND](e.sid)
    var b = e.config.batch
    var length = e.config.length
    var rows = k * b
    var stride = (length + 1) * 4
    var srcs: List[CastIn] = [
        CastIn(stack, CAST_KIND_SIGNED, 4, rows, length, stride, 4),
        CastIn(stack + 4, CAST_KIND_SIGNED, 4, rows, length, stride, 4),
    ]
    var shapes: List[ShapeN] = [ShapeN.matrix(rows, length), ShapeN.matrix(rows, length)]
    var curve = List[Float32](length=1, fill=Float32(0))
    var none = CastIn(0, CAST_KIND_FLOAT, 4, 0, 0, 0, 0)
    _ = train_program_control(
        s[], e.pid, srcs, shapes, none, 0, Float32(0), e.opt, e.trained(), 1, b, False, UInt64(0),
        FP(unsafe_from_address=Int(curve.unsafe_ptr())), 0, 0, Float32(0), 0, losses, e.grads,
    )
    return curve[0]


def _replay[E: Exec & Movable & Deinitable, KIND: Int](
    mut e: ByteLmProgram[E, KIND], ids: Int, backward: Bool,
) raises -> Float32:
    """One forward of the captured program on one batch (`ids`: batch rows
    of length + 1 int32 ids), recorded; with `backward` the tape backward
    writes every trained tensor's gradient into the gradient scratch. No
    optimizer step. Returns the loss (one wait)."""
    var s = ops_session[E, KIND](e.sid)
    var b = e.config.batch
    var length = e.config.length
    var stride = (length + 1) * 4
    var ids_dev = FP(unsafe_from_address=e.ids_buf)
    var tgt_dev = FP(unsafe_from_address=e.tgt_buf)
    upload_cast_i32(s[].ex, ids_dev, CastIn(ids, CAST_KIND_SIGNED, 4, b, length, stride, 4), "byte LM ids")
    upload_cast(s[].ex, tgt_dev, CastIn(ids + 4, CAST_KIND_SIGNED, 4, b, length, stride, 4))
    var p = s[].program(e.pid)
    var prepared = PreparedProgramReplay(p)
    var params = parameter_buffers(s[], p)
    var trained = e.trained()
    var trainable = List[Bool](length=len(p.params), fill=False)
    var slot_of = List[Int](length=len(p.params), fill=-1)
    for j in range(len(p.params)):  # driver-loop: over the program's parameters
        for i in range(len(trained)):  # small-loop(trained: the model's tensors): handle lookup, no data
            if p.params[j] == trained[i]:
                trainable[j] = backward
                slot_of[j] = i
    var inputs: List[FP] = [ids_dev, tgt_dev]
    var shapes: List[ShapeN] = [ShapeN.matrix(b, length), ShapeN.matrix(b, length)]
    var loss_host = List[Float32](length=1, fill=Float32(0))
    var held = Tape()
    var temps = List[Int]()
    var phs = List[Int]()
    swap(held, s[].prog_tape)
    swap(held, s[].tape)
    try:
        s[].tape.reset()
        s[].tape.record()
        var outs = replay_values_prepared(s[], p, prepared, inputs, shapes, params, trainable, temps, phs)
        var loss = s[].tensor(outs[0])
        if backward:
            s[].tape.backward_scalar_with_rules(s[].ex, loss, StepOpRules(s[].steps_address()))
            for j in range(len(phs)):  # driver-loop: over the program's parameters (gradient copies)
                if slot_of[j] < 0:
                    continue
                var gc = Args()
                gc.p0 = s[].tape.gradient(s[].tensor(phs[j]))
                gc.p1 = FP(unsafe_from_address=e.grads[slot_of[j]])
                s[].ex.launch[OP_COPY](gc, p.param_shapes[j].count())
        s[].ex.download(FP(unsafe_from_address=Int(loss_host.unsafe_ptr())), s[].tape.value(loss), 1)
        s[]._drop_temps(temps)
        s[].tape.no_grad()
    except err:
        s[]._drop_temps(temps)
        s[].tape.no_grad()
        swap(held, s[].tape)
        swap(held, s[].prog_tape)
        raise err
    swap(held, s[].tape)
    swap(held, s[].prog_tape)
    return loss_host[0]


def byte_program_eval[E: Exec & Movable & Deinitable, KIND: Int](
    mut e: ByteLmProgram[E, KIND], ids: Int,
) raises -> Float32:
    """The training forward's mean loss on one batch, no backward, no
    update (the same launches as a step's forward, so the same loss)."""
    return _replay(e, ids, False)


def byte_program_gradient[E: Exec & Movable & Deinitable, KIND: Int](
    mut e: ByteLmProgram[E, KIND], ids: Int,
) raises -> Float32:
    """One batch's loss and pre-update gradients (into the gradient
    scratch; `byte_program_store` or `byte_program_grad_out` brings them to
    the trainer), no update: the shard half of ParallelByteLM."""
    return _replay(e, ids, True)


def byte_program_grad_out(mut e: ByteDeviceProgram, grad: DeviceBuffer[DType.float32]) raises:
    """The gradient scratch into the trainer's flat `grad`."""
    var s = ops_session[DeviceExec, OPS_DEVICE](e.sid)
    _move(s[], e, grad, BYTE_PART_GRAD, False)


def byte_program_apply(mut e: ByteDeviceProgram, grad: DeviceBuffer[DType.float32]) raises:
    """One AdamW step of every trained tensor from the trainer's flat
    `grad` (ParallelByteLM's reduced gradient): the gradient into the
    scratch, then the session optimizer's update per tensor (the update
    Program.fit takes after each minibatch). No wait."""
    var s = ops_session[DeviceExec, OPS_DEVICE](e.sid)
    _move(s[], e, grad, BYTE_PART_GRAD, True)
    var oi = s[]._opt_index(e.opt)
    var cfg = s[].opts[oi].cfg
    var lr = s[].learning_rate(e.opt)
    var trained = e.trained()
    for i in range(len(trained)):  # driver-loop: over the trained tensors
        var h = trained[i]
        var n = s[].shapes[s[]._slot(h)].count()
        var si = s[]._state_of(oi, h, n)
        var st = s[]._state_view(si)
        var t = s[].opt_states[si].t + 1
        step_lr(s[].ex, cfg, st, s[].buffer(h), FP(unsafe_from_address=e.grads[i]), n, t, lr)
        s[].opt_states[si].t = t
        s[].opt_states[si].pw1 = st.pw1
        s[].opt_states[si].pw2 = st.pw2
        s[].opt_states[si].mu_prod = st.mu_prod
