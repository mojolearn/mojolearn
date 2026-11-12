# SPDX-License-Identifier: Apache-2.0
# PACK of the metric programs (lane metric-programs-2, 2026-11-12): arena
# words copied into a new result Array. One function for the copies that
# were `MxArena.words` (bindings/metrics_exp_common.mojo) and
# `mc_arena_f64_array` / `mc_arena_i64_array` (bindings/mcls_common.mojo).
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.arena_prog_run import MetricArenaOwner
from bindings.family_common_pack import family_array
from bindings.classical_arrays_door import classical_address


def metric_arena_array(arena: MetricArenaOwner, off: Int, words: Int, shape: PythonObject, typestr: String,
                       context: PythonObject) raises -> PythonObject:
    """PACK `Array._owned(prog.words(off, words, code), shape, typestr)`: a
    copy of `words` arena words as a new Array."""
    var result = family_array(shape, typestr, context)
    if words > 0:
        unsafe_memcpy(dest=MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(result))),
                      src=MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=arena.base() + 4 * off), count=words)
        _ = result  # keep the Python object alive through the native copy above
    _ = arena  # keep the arena store alive through the native copy above
    return result


def metric_f64_array(values: List[Float64], context: PythonObject) raises -> PythonObject:
    """PACK: binary64 values as a new 1-D `<f8` Array."""
    var result = family_array(Python.tuple(PythonObject(len(values))), "<f8", context)
    if len(values) > 0:
        unsafe_memcpy(dest=MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(result))),
                      src=MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(values.unsafe_ptr())),
                      count=len(values))
        _ = result  # keep the Python object alive through the native copy above
    _ = values
    return result
