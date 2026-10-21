# SPDX-License-Identifier: Apache-2.0
# Executed-path control-flow VJPs. Branch decisions are discrete: predicates
# run without recording, while selected operations and loop-carried state use
# one recording tape. No state copy or reset severs an iteration's parents.
from std.memory import bitcast
from sequence.exec_trait import Exec
from sequence.ops import FP
from autograd.ops_api import OpsSession, ops_session
from autograd.program import OpsProgram, PROGRAM_IN_IDS, same_shape
from autograd.program_control import _signature
from autograd.cast_ops import upload_cast, upload_cast_i32
from autograd.tape import Tape
from autograd.import_api import ImportRules
from autograd.program_io import ProgramInput, HostArray
from autograd.rankn import ShapeN


def _parameter_leaf[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], handle: Int, mut leaves: Dict[Int, Int], mut temps: List[Int],
) raises -> Int:
    if handle in leaves:
        return leaves[handle]
    var slot = s._slot(handle)
    if s.owned[slot] == 0:
        raise Error("ValueError: control-flow parameters must be uploaded tensors")
    var grad = not s.int_words[slot]
    var t = s.tape.variable_nd(s.ex, FP(unsafe_from_address=s.owned[slot]), s.shapes[slot], grad)
    var h = s._add(t, 0, grad, 0)
    s.int_words[s._slot(h)] = not grad
    temps.append(h)
    leaves[handle] = h
    return h


def _replay_handles[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], p: OpsProgram, inputs: List[Int], mut leaves: Dict[Int, Int],
    mut temps: List[Int], recording: Bool,
) raises -> List[Int]:
    if len(inputs) != p.n_in:
        raise Error("ValueError: control-flow state has the wrong number of tensors")
    s.tape.record()
    var params = List[Int]()
    for j in range(len(p.params)):  # driver-loop: over parameter references
        var slot = s._slot(p.params[j])
        if not same_shape(s.shapes[slot], p.param_shapes[j]):
            raise Error("ValueError: control-flow parameter changed shape")
        params.append(_parameter_leaf(s, p.params[j], leaves, temps))
    if not recording:
        s.tape.no_grad()
    var values = inputs.copy()
    for k in range(p.n_nodes()):  # driver-loop: over the executed graph nodes
        var args = List[Int]()
        for j in range(p.a_at[k], p.a_at[k + 1]):  # driver-loop: over node references
            var value_ref = p.args[j]
            args.append(params[-value_ref - 1] if value_ref < 0 else values[value_ref])
        var ints = List[Int]()
        for j in range(p.i_at[k], p.i_at[k + 1]):  # driver-loop: over integer node options
            ints.append(p.ints[j])
        var floats = List[Float32]()
        for j in range(p.f_at[k], p.f_at[k + 1]):  # driver-loop: over scalar node options
            floats.append(p.fs[j])
        var h = s._program_op(k, p.code[k], args, ints, floats)
        temps.append(h)
        values.append(h)
    var outputs = List[Int]()
    for j in range(len(p.outs)):  # driver-loop: over output references
        var value_ref = p.outs[j]
        outputs.append(params[-value_ref - 1] if value_ref < 0 else values[value_ref])
    s.tape.record()
    return outputs^


def _decision[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], condition: OpsProgram, state: List[Int],
    mut leaves: Dict[Int, Int], mut temps: List[Int],
) raises -> Bool:
    var outs = _replay_handles(s, condition, state, leaves, temps, False)
    if len(outs) != 1 or s.shape_nd(outs[0]).count() != 1:
        raise Error("ValueError: control-flow condition must return one scalar")
    var value = s.scalar(outs[0])
    if (bitcast[DType.uint32](value) & UInt32(0x7fffffff)) >= UInt32(0x7f800000):
        raise Error("ValueError: control-flow condition must be finite")
    return value != 0


# ---------------------------------------------------------------- entries
# The typed entries behind program_control_grad_pyapi.mojo, in the old
# order: control_vjp_prepare checks the programs, the door converts the
# inputs, control_vjp_forward records the executed path, the door converts
# one seed per output, and control_vjp_finish runs the seeded backward and
# downloads values and gradients to the host. control_vjp_abort releases a
# recorded path when the seeds cannot be converted.
struct ControlVjpPlan(Movable):
    var body: OpsProgram
    var condition: OpsProgram
    var other: OpsProgram
    var is_loop: Bool
    var maximum: Int

    def __init__(out self, var body: OpsProgram, var condition: OpsProgram, var other: OpsProgram,
                 is_loop: Bool, maximum: Int):
        self.body = body^
        self.condition = condition^
        self.other = other^
        self.is_loop = is_loop
        self.maximum = maximum


struct ControlVjpState(Movable):
    var held: Tape
    var buffers: List[FP]
    var temps: List[Int]
    var initial: List[Int]
    var leaves: Dict[Int, Int]
    var outputs: List[Int]
    var parameters: List[Int]
    var explicit_parameters: Bool
    var iterations: Int
    var stopped: Bool

    def __init__(out self, var held: Tape, var parameters: List[Int], explicit_parameters: Bool):
        self.held = held^
        self.buffers = List[FP]()
        self.temps = List[Int]()
        self.initial = List[Int]()
        self.leaves = Dict[Int, Int]()
        self.outputs = List[Int]()
        self.parameters = parameters^
        self.explicit_parameters = explicit_parameters
        self.iterations = 0
        self.stopped = False


struct ControlVjpResult(Movable):
    """Output values, then per initial input and per parameter (in
    `parameters` order) its gradient; `*_present` is False where the tensor
    holds int32 ids (no gradient)."""
    var values: List[HostArray]
    var input_gradients: List[HostArray]
    var input_present: List[Bool]
    var parameters: List[Int]
    var parameter_gradients: List[HostArray]
    var parameter_present: List[Bool]
    var iterations: Int
    var stopped: Bool

    def __init__(out self):
        self.values = List[HostArray]()
        self.input_gradients = List[HostArray]()
        self.input_present = List[Bool]()
        self.parameters = List[Int]()
        self.parameter_gradients = List[HostArray]()
        self.parameter_present = List[Bool]()
        self.iterations = 0
        self.stopped = False


def _gradient_host[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], h: Int, mut out: List[HostArray], mut present: List[Bool],
) raises:
    if s.int_words[s._slot(h)]:
        out.append(HostArray(ShapeN.of([1])))
        present.append(False)
        return
    var shape = s.shape_nd(h)
    var a = HostArray(shape)
    try:
        var gradient = s.tape.gradient(s.tensor(h))
        s.ex.download(FP(unsafe_from_address=Int(a.data.unsafe_ptr())), gradient, shape.count())
    except error:
        # A disconnected leaf has the exact zero VJP. HostArray's zero fill
        # supplies zero bits; do not hide other failures.
        if not String(error).startswith("tensor has no gradient:"):
            raise error
    out.append(a^)
    present.append(True)


def control_vjp_prepare[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, body_id: Int, programs: List[Int], maximum: Int,
) raises -> ControlVjpPlan:
    """programs = [condition sid, condition id, other sid, other id (0 for a
    while loop)]."""
    if programs[0] != sid or programs[2] != sid:
        raise Error("ValueError: control-flow programs must belong to the same session")
    var s = ops_session[E, KIND](sid)
    if s[].tape.is_recording():
        raise Error("ValueError: captured control flow requires an inference session")
    var body = s[].program(body_id)
    var condition = s[].program(programs[1])
    var is_loop = programs[3] == 0
    var other = body.copy() if is_loop else s[].program(programs[3])
    _signature(body, condition)
    _signature(body, other)
    if is_loop and len(body.outs) != body.n_in:
        raise Error("ValueError: while_loop body must return one tensor per input")
    if is_loop:
        for k in range(body.n_in):  # driver-loop: over loop state input kinds
            if body.in_kinds[k] == PROGRAM_IN_IDS:
                raise Error("ValueError: while_loop state inputs must hold floating-point values")
        for k in range(len(body.outs)):  # driver-loop: over loop state output metadata
            var value_ref = body.outs[k]
            if value_ref >= body.n_in:
                var node = value_ref - body.n_in
                if body.code[node] == 72 and body.ints[body.i_at[node] + 2] == 0:
                    raise Error("ValueError: while_loop state outputs must hold floating-point values")
    return ControlVjpPlan(body^, condition^, other^, is_loop, maximum)


def _control_vjp_release[E: Exec & Movable & Deinitable](mut s: OpsSession[E], mut st: ControlVjpState) raises:
    s._drop_temps(st.temps)
    s.tape.no_grad()
    swap(st.held, s.tape)
    swap(st.held, s.prog_tape)
    for k in range(len(st.buffers)):  # driver-loop: over input and seed owners
        s.ex.free(st.buffers[k])


def control_vjp_forward[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, plan: ControlVjpPlan, var prepared: List[ProgramInput], requested: List[Int],
    explicit_parameters: Bool,
) raises -> ControlVjpState:
    var s = ops_session[E, KIND](sid)
    var body = plan.body.copy()
    var parameters = List[Int]()
    if explicit_parameters:
        for j in range(len(requested)):  # driver-loop: over requested VJP parameters
            var h = requested[j]
            for old in parameters:  # driver-loop: over previous requested parameters
                if h == old:
                    raise Error("ValueError: control VJP lists a parameter twice")
            parameters.append(h)
    var held = Tape()
    swap(held, s[].prog_tape)
    swap(held, s[].tape)
    var st = ControlVjpState(held^, parameters^, explicit_parameters)
    try:
        s[].tape.reset()
        s[].tape.record()
        for k in range(len(prepared)):  # driver-loop: over initial state inputs
            var shape = prepared[k].shape
            var buf = s[].ex.alloc(shape.count())
            st.buffers.append(buf)
            var ids = body.in_kinds[k] == PROGRAM_IN_IDS
            if ids:
                upload_cast_i32(s[].ex, buf, prepared[k].src, "control-flow ids")
            else:
                upload_cast(s[].ex, buf, prepared[k].src)
            var t = s[].tape.variable_nd(s[].ex, buf, shape, not ids)
            var h = s[]._add(t, 0, not ids, 0)
            s[].int_words[s[]._slot(h)] = ids
            st.temps.append(h)
            st.initial.append(h)
        _ = prepared^  # retain scalar-list owners through every upload
        for j in range(len(st.parameters)):  # driver-loop: over requested parameter leaves
            _ = _parameter_leaf(s[], st.parameters[j], st.leaves, st.temps)
        var outputs = st.initial.copy()
        if plan.is_loop:
            while True:  # driver-loop: over graph iterations bounded by max_iterations
                if not _decision(s[], plan.condition, outputs, st.leaves, st.temps):
                    st.stopped = True
                    break
                if st.iterations == plan.maximum:
                    break
                outputs = _replay_handles(s[], body, outputs, st.leaves, st.temps, True)
                for k in range(len(outputs)):  # driver-loop: over loop state output kinds
                    if s[].shape_nd(outputs[k]).ndim != body.in_shapes[k].ndim:
                        raise Error("ValueError: while_loop body must preserve each state tensor's rank")
                    if s[].int_words[s[]._slot(outputs[k])]:
                        raise Error("ValueError: while_loop state outputs must hold floating-point values")
                st.iterations += 1
        elif _decision(s[], plan.condition, st.initial, st.leaves, st.temps):
            outputs = _replay_handles(s[], body, st.initial, st.leaves, st.temps, True)
        else:
            outputs = _replay_handles(s[], plan.other, st.initial, st.leaves, st.temps, True)
        st.outputs = outputs^
    except error:
        _control_vjp_release(s[], st)
        raise error
    return st^


def control_vjp_abort[E: Exec & Movable & Deinitable, KIND: Int](sid: Int, var st: ControlVjpState) raises:
    var s = ops_session[E, KIND](sid)
    _control_vjp_release(s[], st)


def control_vjp_finish[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, var st: ControlVjpState, var seeds: List[ProgramInput],
) raises -> ControlVjpResult:
    var s = ops_session[E, KIND](sid)
    var result = ControlVjpResult()
    try:
        var accumulated = False
        for k in range(len(st.outputs)):  # driver-loop: over seeded output VJPs
            var h = st.outputs[k]
            var shape = s[].shape_nd(h)
            if not same_shape(shape, seeds[k].shape):
                raise Error("ValueError: control VJP seed must have its output's shape")
            var buf = s[].ex.alloc(shape.count())
            st.buffers.append(buf)
            upload_cast(s[].ex, buf, seeds[k].src)
            var value = HostArray(shape)
            s[].download(h, FP(unsafe_from_address=Int(value.data.unsafe_ptr())))
            result.values.append(value^)
            var out = s[].tensor(h)
            if s[].tape.requires_grad(out):
                var seed_tensor = s[].tape.variable_nd(s[].ex, buf, shape, False)
                s[].tape.backward_with_rules(s[].ex, out, seed_tensor, ImportRules(), accumulated)
                accumulated = True
        _ = seeds^  # retain scalar-list owners through every seed upload
        for k in range(len(st.initial)):  # driver-loop: over initial input VJPs
            _gradient_host(s[], st.initial[k], result.input_gradients, result.input_present)
        if not st.explicit_parameters:
            for item in st.leaves.items():  # driver-loop: over executed parameter leaves
                st.parameters.append(item.key)
        for k in range(len(st.parameters)):  # driver-loop: over requested parameter VJPs
            result.parameters.append(st.parameters[k])
            _gradient_host(s[], st.leaves[st.parameters[k]], result.parameter_gradients, result.parameter_present)
        result.iterations = st.iterations
        result.stopped = st.stopped
    except error:
        _control_vjp_release(s[], st)
        raise error
    _control_vjp_release(s[], st)
    return result^
