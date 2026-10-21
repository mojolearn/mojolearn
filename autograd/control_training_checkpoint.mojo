# SPDX-License-Identifier: Apache-2.0
# Optional v2 control checkpoint suffix: native session optimizer state.
# Component/parameter records keep their v1 layout. Only Session's supported
# optimizer kinds are accepted; no Python serialization or state preparation.
from autograd.ops_api import OpsSession
from autograd.optim import optimizer_slots
from autograd.program import _Reader
from autograd.rankn import ShapeN
from sequence.exec_trait import Exec
from sequence.ops import FP
from training.checkpoint import Checkpoint


def save_control_training[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], handles: List[Int], shapes: List[ShapeN],
    mut ck: Checkpoint, optimizer: Int, seed: UInt64,
) raises:
    var oi = s._opt_index(optimizer)
    var option = s.opts[oi]
    var slots = optimizer_slots(option.cfg)
    if slots[2]:
        raise Error("ValueError: control checkpoint cannot encode a third optimizer state buffer")
    ck.seed = seed
    ck.opt_kind = option.cfg.kind
    ck.lr = option.cfg.lr
    ck.beta1 = option.cfg.beta1
    ck.beta2 = option.cfg.beta2
    ck.eps = option.cfg.eps
    ck.weight_decay = option.cfg.weight_decay
    ck.momentum = option.cfg.momentum
    ck.nesterov = option.cfg.nesterov
    ck.ext_ints.append(Int64(option.epoch))
    ck.ext_ints.append(Int64(option.sched))
    ck.ext_ints.append(Int64(option.sched_step))
    ck.ext_floats.append(option.gamma)
    for k in range(len(handles)):  # driver-loop: over unique checkpoint parameter state
        var si = -1
        for j in range(len(s.opt_states)):  # driver-loop: over existing native optimizer states
            if s.opt_states[j].opt == oi and s.opt_states[j].param == handles[k]:
                si = j
                break
        ck.buf_initialized[k] = si >= 0
        if si < 0:
            ck.ext_ints.append(Int64(0))
            ck.ext_floats.append(Float32(1))
            ck.ext_floats.append(Float32(1))
            ck.ext_floats.append(Float32(1))
            continue
        var state = s.opt_states[si]
        if state.n != shapes[k].count():
            raise Error("ValueError: control checkpoint optimizer state shape changed")
        ck.t = max(ck.t, state.t)
        ck.ext_ints.append(Int64(state.t))
        ck.ext_floats.append(state.pw1)
        ck.ext_floats.append(state.pw2)
        ck.ext_floats.append(state.mu_prod)
        if slots[0]:
            s.ex.download(FP(unsafe_from_address=Int(ck.m_state.unsafe_ptr())) + ck.offsets[k], state.s1, state.n)
        if slots[1]:
            s.ex.download(FP(unsafe_from_address=Int(ck.v_state.unsafe_ptr())) + ck.offsets[k], state.s2, state.n)


struct ControlTraining(Movable):
    var present: Bool
    var epoch: Int
    var ints: List[Int]
    var floats: List[Float32]
    var steps: List[Int]
    var scalar_at: Int

    def __init__(out self):
        self.present = False
        self.epoch = 0
        self.ints = List[Int]()
        self.floats = List[Float32]()
        self.steps = List[Int]()
        self.scalar_at = 0


def read_control_training(
    ck: Checkpoint, kinds: List[Int], mut r: _Reader, mut float_at: Int,
) raises -> ControlTraining:
    var what = String("ValueError: malformed control checkpoint optimizer state")
    var out = ControlTraining()
    out.present = True
    out.epoch = r.take(ck.ext_ints, what)
    var sched = r.take(ck.ext_ints, what)
    var interval = r.take(ck.ext_ints, what)
    if out.epoch < 0 or interval < 1 or sched < 0 or sched > 2:
        raise Error(what + ": invalid epoch or schedule")
    if len(ck.ext_floats) - float_at != 1 + 3 * len(kinds):
        raise Error(what + ": invalid scalar count")
    out.ints = [ck.opt_kind, Int(ck.nesterov), sched, interval]
    out.floats = [ck.lr, ck.momentum, ck.beta1, ck.beta2, ck.eps, ck.weight_decay, ck.ext_floats[float_at]]
    float_at += 1
    out.scalar_at = float_at
    for k in range(len(kinds)):  # driver-loop: over serialized optimizer counters
        var step = r.take(ck.ext_ints, what)
        if step < 0 or (not ck.buf_initialized[k] and step != 0) or (ck.buf_initialized[k] and kinds[k] != 0):
            raise Error(what + ": invalid parameter step or state flag")
        out.steps.append(step)
        for j in range(3):  # small-loop(3: bias-correction scalars): optimizer metadata
            var scalar = ck.ext_floats[float_at]
            if not (scalar >= 0 and scalar <= 1) or (not ck.buf_initialized[k] and scalar != 1):
                raise Error(what + ": invalid bias-correction scalar")
            float_at += 1
    return out^


def restore_control_training[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], handles: List[Int], shapes: List[ShapeN],
    ck: Checkpoint, metadata: ControlTraining,
) raises -> Int:
    # The ordinary native constructor validates the exact same options as the
    # public optimizer call. A failed restore removes this new optimizer, and
    # the caller retires every partial graph/parameter/state allocation.
    var opt = s.optimizer(metadata.ints, metadata.floats)
    var oi = s._opt_index(opt)
    try:
        s.opts[oi].epoch = metadata.epoch
        var slots = optimizer_slots(s.opts[oi].cfg)
        if slots[2]:
            raise Error("ValueError: control checkpoint cannot restore a third optimizer state buffer")
        for k in range(len(handles)):  # driver-loop: over restored optimizer parameter identities
            if not ck.buf_initialized[k]:
                continue
            s.wants_grad[s._slot(handles[k])] = True
            var si = s._state_of(oi, handles[k], shapes[k].count())
            var state = s.opt_states[si]
            if slots[0]:
                s.ex.upload(state.s1, FP(unsafe_from_address=Int(ck.m_state.unsafe_ptr())) + ck.offsets[k], state.n)
            if slots[1]:
                s.ex.upload(state.s2, FP(unsafe_from_address=Int(ck.v_state.unsafe_ptr())) + ck.offsets[k], state.n)
            s.opt_states[si].t = metadata.steps[k]
            s.opt_states[si].pw1 = ck.ext_floats[metadata.scalar_at + 3 * k]
            s.opt_states[si].pw2 = ck.ext_floats[metadata.scalar_at + 3 * k + 1]
            s.opt_states[si].mu_prod = ck.ext_floats[metadata.scalar_at + 3 * k + 2]
        s.ex.sync()
    except error:
        _ = s.opts.pop()
        raise error
    return opt
