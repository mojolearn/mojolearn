# SPDX-License-Identifier: Apache-2.0
# The `small_mlp_program` entry of the x_sequence bindings (device and host):
# the argument words of bindings/small_mlp_native.mojo's calls, unpacked into
# native lists, then one call into autograd/small_mlp_program.mojo. No
# arithmetic here.
from std.python import Python, PythonObject
from sequence.exec_trait import Exec
from autograd.small_mlp_program import (
    SMALL_MLP_PROGRAM_OPEN, small_mlp_program_open, small_mlp_program_run,
)


def _words(value: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for i in range(len(value)):  # driver-loop: over the call's argument words
        out.append(Int(py=value[i]))
    return out^


def _floats(value: PythonObject) raises -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(value)):  # driver-loop: over the call's optimizer settings
        out.append(Float32(Float64(py=value[i])))
    return out^


def small_mlp_program_open_py[E: Exec & Movable & Deinitable, KIND: Int](
    var ex: E, addresses: PythonObject, ints: PythonObject, floats: PythonObject,
) raises -> PythonObject:
    return PythonObject(small_mlp_program_open[E, KIND](ex^, _words(addresses), _words(ints), _floats(floats)))


def small_mlp_program_run_py[E: Exec & Movable & Deinitable, KIND: Int](
    op: Int, addresses: PythonObject, ints: PythonObject,
) raises -> PythonObject:
    return PythonObject(small_mlp_program_run[E, KIND](op, _words(addresses), _words(ints), List[Float32]()))
