# SPDX-License-Identifier: Apache-2.0
# The `small_mlp_program` entry of the x_sequence bindings (device and host):
# the argument words of bindings/small_mlp_native.mojo's calls, unpacked into
# native lists, then one call into autograd/small_mlp_program.mojo. No
# arithmetic here.
# The PACK half of the former `providers/python/small_mlp_program_pyapi.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `providers/python/small_mlp_program_pyapi_unpack.mojo`.
from std.python import Python, PythonObject
from sequence.exec_trait import Exec
from autograd.small_mlp_program import (
    SMALL_MLP_PROGRAM_OPEN, small_mlp_program_open, small_mlp_program_run, small_mlp_sessions_forward,
)


from providers.python.small_mlp_program_pyapi_unpack import (
    _floats,
    _words,
)


def small_mlp_program_open_py[E: Exec & Movable & Deinitable, KIND: Int](
    var ex: E, addresses: PythonObject, ints: PythonObject, floats: PythonObject,
) raises -> PythonObject:
    return PythonObject(small_mlp_program_open[E, KIND](ex^, _words(addresses), _words(ints), _floats(floats)))


def small_mlp_sessions_py[E: Exec & Movable & Deinitable, KIND: Int](
    var ex: E, addresses: PythonObject, ints: PythonObject,
) raises -> PythonObject:
    return PythonObject(small_mlp_sessions_forward[E, KIND](ex^, _words(addresses), _words(ints)))


def small_mlp_program_run_py[E: Exec & Movable & Deinitable, KIND: Int](
    op: Int, addresses: PythonObject, ints: PythonObject,
) raises -> PythonObject:
    return PythonObject(small_mlp_program_run[E, KIND](op, _words(addresses), _words(ints), List[Float32]()))
