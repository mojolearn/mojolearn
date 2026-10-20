# SPDX-License-Identifier: Apache-2.0
# Executed-path control-flow VJPs. Branch decisions are discrete: predicates
# run without recording, while selected operations and loop-carried state use
# one recording tape. No state copy or reset severs an iteration's parents.
from std.python import Python, PythonObject
from std.memory import bitcast
from sequence.exec_trait import Exec
from sequence.ops import FP
from autograd.ops_api import OpsSession, ops_session
from autograd.program import OpsProgram, PROGRAM_IN_IDS, same_shape
from autograd.program_control import _signature
from autograd.program_pyapi import ProgramInput, program_inputs, output_array, _integer, _result, _dict
from autograd.cast_ops import upload_cast, upload_cast_i32
from autograd.tape import Tape
from autograd.import_api import ImportRules


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
            var ref = p.args[j]
            args.append(params[-ref - 1] if ref < 0 else values[ref])
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
        var ref = p.outs[j]
        outputs.append(params[-ref - 1] if ref < 0 else values[ref])
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


def _gradient_array[E: Exec & Movable & Deinitable](mut s: OpsSession[E], h: Int) raises -> PythonObject:
    if s.int_words[s._slot(h)]:
        return PythonObject(None)
    var shape = s.shape_nd(h)
    var out = output_array(shape)
    try:
        var gradient = s.tape.gradient(s.tensor(h))
        s.ex.download(FP(unsafe_from_address=Int(py=out.__array_interface__["data"][0])), gradient, shape.count())
    except error:
        # A disconnected leaf has the exact zero VJP. output_array's native
        # bytearray allocation supplies zero bits; do not hide other failures.
        if not String(error).startswith("tensor has no gradient:"):
            raise error
    return out


def ops_program_control_vjp_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, body_id: PythonObject, programs: PythonObject,
    arrays: PythonObject, limit: PythonObject,
) raises -> PythonObject:
    if Int(py=programs[0]) != Int(py=sid) or Int(py=programs[2]) != Int(py=sid):
        raise Error("ValueError: control-flow programs must belong to the same session")
    var s = ops_session[E, KIND](Int(py=sid))
    if s[].tape.is_recording():
        raise Error("ValueError: captured control flow requires an inference session")
    var body = s[].program(Int(py=body_id))
    var condition = s[].program(Int(py=programs[1]))
    var is_loop = Int(py=programs[3]) == 0
    var other = body.copy() if is_loop else s[].program(Int(py=programs[3]))
    _signature(body, condition)
    _signature(body, other)
    var maximum = _integer(limit, "max_iterations", 0)
    if is_loop and len(body.outs) != body.n_in:
        raise Error("ValueError: while_loop body must return one tensor per input")
    var prepared = program_inputs(body, arrays, True, True)
    var buffers = List[FP]()
    var temps = List[Int]()
    var initial = List[Int]()
    var leaves = Dict[Int, Int]()
    var requested = programs[5]
    var explicit_parameters = requested is not PythonObject(None)
    var parameters = List[Int]()
    if explicit_parameters:
        for j in range(len(requested)):  # driver-loop: over requested VJP parameters
            var h = _integer(requested[j], "parameter", 1)
            for old in parameters:  # driver-loop: over previous requested parameters
                if h == old:
                    raise Error("ValueError: control VJP lists a parameter twice")
            parameters.append(h)
    var held = Tape()
    swap(held, s[].prog_tape)
    swap(held, s[].tape)
    var result = _dict()
    var iterations = 0
    var stopped = False
    try:
        s[].tape.reset()
        s[].tape.record()
        for k in range(len(prepared)):  # driver-loop: over initial state inputs
            var shape = prepared[k].shape
            var buf = s[].ex.alloc(shape.count())
            buffers.append(buf)
            var ids = body.in_kinds[k] == PROGRAM_IN_IDS
            if ids:
                upload_cast_i32(s[].ex, buf, prepared[k].src, "control-flow ids")
            else:
                upload_cast(s[].ex, buf, prepared[k].src)
            var t = s[].tape.variable_nd(s[].ex, buf, shape, not ids)
            var h = s[]._add(t, 0, not ids, 0)
            s[].int_words[s[]._slot(h)] = ids
            temps.append(h)
            initial.append(h)
        _ = prepared^  # retain scalar-list owners through every upload
        for j in range(len(parameters)):  # driver-loop: over requested parameter leaves
            _ = _parameter_leaf(s[], parameters[j], leaves, temps)
        var outputs = initial.copy()
        if is_loop:
            while True:  # driver-loop: over graph iterations bounded by max_iterations
                if not _decision(s[], condition, outputs, leaves, temps):
                    stopped = True
                    break
                if iterations == maximum:
                    break
                outputs = _replay_handles(s[], body, outputs, leaves, temps, True)
                iterations += 1
        elif _decision(s[], condition, initial, leaves, temps):
            outputs = _replay_handles(s[], body, initial, leaves, temps, True)
        else:
            outputs = _replay_handles(s[], other, initial, leaves, temps, True)
        var seeds = programs[4]
        var seed_arrays = Python.list()
        if len(outputs) == 1:
            seed_arrays.append(seeds)
        else:
            if len(seeds) != len(outputs):
                raise Error("ValueError: control VJP needs one seed per output")
            seed_arrays = seeds
        var values = Python.list()
        var accumulated = False
        for k in range(len(outputs)):  # driver-loop: over seeded output VJPs
            var h = outputs[k]
            var shape = s[].shape_nd(h)
            var seed = ProgramInput(seed_arrays[k], False)
            if not same_shape(shape, seed.shape):
                raise Error("ValueError: control VJP seed must have its output's shape")
            var buf = s[].ex.alloc(shape.count())
            buffers.append(buf)
            upload_cast(s[].ex, buf, seed.src)
            _ = seed^
            var value = output_array(shape)
            s[].download(h, FP(unsafe_from_address=Int(py=value.__array_interface__["data"][0])))
            values.append(value)
            var out = s[].tensor(h)
            if s[].tape.requires_grad(out):
                var seed_tensor = s[].tape.variable_nd(s[].ex, buf, shape, False)
                s[].tape.backward_with_rules(s[].ex, out, seed_tensor, ImportRules(), accumulated)
                accumulated = True
        var input_gradients = Python.list()
        for k in range(len(initial)):  # driver-loop: over initial input VJPs
            input_gradients.append(_gradient_array(s[], initial[k]))
        var parameter_gradients = _dict()
        if not explicit_parameters:
            for item in leaves.items():  # driver-loop: over executed parameter leaves
                parameters.append(item.key)
        for k in range(len(parameters)):  # driver-loop: over requested parameter VJPs
            parameter_gradients[parameters[k]] = _gradient_array(s[], leaves[parameters[k]])
        result["outputs"] = _result(values)
        result["input_gradients"] = Python.import_module("builtins").tuple(input_gradients)
        result["parameter_gradients"] = parameter_gradients
        result["iterations"] = iterations
        result["stopped"] = stopped
    except error:
        s[]._drop_temps(temps)
        s[].tape.no_grad()
        swap(held, s[].tape)
        swap(held, s[].prog_tape)
        for k in range(len(buffers)):  # driver-loop: over input and seed owners
            s[].ex.free(buffers[k])
        raise error
    s[]._drop_temps(temps)
    s[].tape.no_grad()
    swap(held, s[].tape)
    swap(held, s[].prog_tape)
    for k in range(len(buffers)):  # driver-loop: over input and seed owners
        s[].ex.free(buffers[k])
    return result
