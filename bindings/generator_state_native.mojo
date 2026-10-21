# SPDX-License-Identifier: Apache-2.0
"""Native Generator metadata and the inclusive exhausted next-stream sentinel."""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_setitem


def generator_fields(value: PythonObject) raises -> PythonObject:
    return py_call(Python.import_module("builtins").object.__getattribute__, value, "__dict__")


def generator_validate(seed: PythonObject, counter: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
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
    var fields = generator_fields(value)
    var counter = Int(py=fields["counter"])
    if counter >= 4294967296:
        raise Error("RuntimeError: Generator: the stream counter is exhausted")
    if counter < 0:
        raise Error("Generator: counter must be nonnegative")
    py_setitem(fields, "counter", PythonObject(counter + 1))
    return PythonObject(counter)
