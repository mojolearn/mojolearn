# SPDX-License-Identifier: Apache-2.0
"""Native Generator metadata and the inclusive exhausted next-stream sentinel."""
# The PACK half of the former `bindings/generator_state_native.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/generator_state_native_unpack.mojo`.
# The seed is admitted over the whole of [0, 2^64) as a Python integer (it does not fit a signed
# native word); the draw door splits it into its two 32-bit words. The counter rule is typed.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_setitem


from bindings.generator_state_native_unpack import (
    generator_fields,
)
from core.generator_draw_run import generator_stream_next


def generator_validate(seed: PythonObject, counter: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var s = py_call(bi.int, seed)
    var c = py_call(bi.int, counter)
    if Bool(py=s < 0) or Bool(py=s >= py_call(bi.int, "18446744073709551616")):
        raise Error("Generator: seed must be in [0, 2^64)")
    # Actual stream IDs are uint32. The next-stream counter also represents
    # exhaustion after the final valid draw, so that state can be saved.
    if Bool(py=c < 0) or Bool(py=c > PythonObject(4294967296)):
        raise Error("Generator: counter must be in [0, 2^32], including exhaustion")
    return Python.tuple(s, c)


def generator_init(value: PythonObject, seed: PythonObject, mode: PythonObject) raises -> PythonObject:
    var checked = generator_validate(seed, PythonObject(0))
    var fields = generator_fields(value)
    py_setitem(fields, "seed", checked[0])
    py_setitem(fields, "counter", checked[1])
    py_setitem(fields, "numeric_mode", mode)
    return PythonObject(None)


def generator_snapshot(value: PythonObject) raises -> PythonObject:
    var fields = generator_fields(value)
    var checked = generator_validate(fields["seed"], fields["counter"])
    var result = Python.dict()
    py_setitem(result, "seed", checked[0])
    py_setitem(result, "counter", checked[1])
    return result


def generator_restore(value: PythonObject, state: PythonObject) raises -> PythonObject:
    var checked = generator_validate(state["seed"], state["counter"])
    var fields = generator_fields(value)
    py_setitem(fields, "seed", checked[0])
    py_setitem(fields, "counter", checked[1])
    return PythonObject(None)


def generator_next(value: PythonObject) raises -> PythonObject:
    # UNPACK the counter word, RUN the typed rule (core/generator_draw_run.mojo),
    # PACK the next counter onto the Generator and the consumed stream id.
    var fields = generator_fields(value)
    var stream = generator_stream_next(Int(py=fields["counter"]))
    py_setitem(fields, "counter", PythonObject(stream + 1))
    return PythonObject(stream)
