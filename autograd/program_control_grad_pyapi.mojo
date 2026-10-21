# SPDX-License-Identifier: Apache-2.0
# Python door for control-flow VJPs (program_control_grad.mojo): argument
# unpacking and ndarray gradient results around the typed entries.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_setitem
from sequence.exec_trait import Exec
from autograd.program_io import ProgramInput, HostArray
from autograd.program_pyapi import program_input, program_inputs, _integer, _result, _dict
from autograd.program_arrays_pyapi import host_array_py, host_arrays_py
from autograd.program_control_grad import (
    control_vjp_abort, control_vjp_finish, control_vjp_forward, control_vjp_prepare,
)


def _gradient_py(arrays: List[HostArray], present: List[Bool], k: Int) raises -> PythonObject:
    if not present[k]:
        return PythonObject(None)
    return host_array_py(arrays[k])


def ops_program_control_vjp_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, body_id: PythonObject, programs: PythonObject,
    arrays: PythonObject, limit: PythonObject,
) raises -> PythonObject:
    var words = List[Int]()
    words.append(Int(py=programs[0]))
    words.append(Int(py=programs[1]))
    words.append(Int(py=programs[2]))
    words.append(Int(py=programs[3]))
    var plan = control_vjp_prepare[E, KIND](Int(py=sid), Int(py=body_id), words, _integer(limit, "max_iterations", 0))
    # A while VJP differentiates the same state domain as inference while:
    # dimensions may grow, but each input and carried state keeps its rank.
    var prepared = program_inputs(plan.body, arrays, True, not plan.is_loop)
    var requested = programs[5]
    var explicit_parameters = requested is not PythonObject(None)
    var parameters = List[Int]()
    if explicit_parameters:
        for j in range(len(requested)):  # small-loop(requested: the requested parameter handles): argument parsing, no data
            parameters.append(_integer(requested[j], "parameter", 1))
    var st = control_vjp_forward[E, KIND](Int(py=sid), plan, prepared^, parameters, explicit_parameters)
    var seed_list = List[ProgramInput]()
    try:
        var seeds = programs[4]
        var seed_arrays = Python.list()
        if len(st.outputs) == 1:
            _ = py_call(seed_arrays.append, seeds)
        else:
            if len(seeds) != len(st.outputs):
                raise Error("ValueError: control VJP needs one seed per output")
            seed_arrays = seeds
        for k in range(len(st.outputs)):  # small-loop(outputs: one seed per program output): argument parsing, no data
            seed_list.append(program_input(seed_arrays[k], False))
    except error:
        control_vjp_abort[E, KIND](Int(py=sid), st^)
        raise error
    var r = control_vjp_finish[E, KIND](Int(py=sid), st^, seed_list^)
    var input_gradients = Python.list()
    for k in range(len(r.input_gradients)):  # small-loop(inputs: one gradient per program input): result packing, no data
        _ = py_call(input_gradients.append, _gradient_py(r.input_gradients, r.input_present, k))
    var parameter_gradients = _dict()
    for k in range(len(r.parameters)):  # small-loop(parameters: one gradient per parameter): result packing, no data
        py_setitem(parameter_gradients, r.parameters[k], _gradient_py(r.parameter_gradients, r.parameter_present, k))
    var result = _dict()
    py_setitem(result, "outputs", _result(host_arrays_py(r.values)))
    py_setitem(result, "input_gradients", py_call(Python.import_module("builtins").tuple, input_gradients))
    py_setitem(result, "parameter_gradients", parameter_gradients)
    py_setitem(result, "iterations", r.iterations)
    py_setitem(result, "stopped", r.stopped)
    return result
