# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The learning-rate schedule entries of the core bindings: the Python door
of sequence/lr_table.mojo (the binary64 table, its float32 decision, the
exact route over a run of steps, and `lr_spec_fill`, the one native call of
every public schedule method). A shell (lane neural-optimizers,
2026-10-21): each entry unpacks its argument words, calls one typed kernel
entry and converts the result back; the arithmetic moved word for word to
sequence/lr_table.mojo."""
from std.python import PythonObject

from std.python import Python
from std.python._cpython import PyObjectPtr
from bindings.optimizer_public_words import optimizer_python_error
from bindings.py_balanced import py_call
from sequence.lr_exact import LrOut, onecycle_exact, pow_exact, schedule_exact
from sequence.lr_table import (
    F32P, F64P, LR_BEYOND, LR_ONE_BASED, LrSpec, decide_many, lr_spec_fill, onecycle_fill,
    pow_values, schedule_block_values,
)


def _f64p(addr: PythonObject) raises -> F64P:
    var a = Int(py=addr)
    if a == 0:
        raise Error("lr table: null float64 buffer")
    return F64P(unsafe_from_address=a)


# TOMBSTONE: encode_labels_* exports (their Mojo bodies stay live), lr_schedule_exact_block, next_combination_i64 (dead native entries: no live caller) deleted 2026-10-26 by e9a91ed29; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_core.patch


# ----------------------------------------------------------- the exact route
# The steps the table cannot decide, and the schedules without a table,
# answered on the big rationals. Each returns [status, value]: status 0 ok,
# 1 overflow (Python's OverflowError), 2 a 64-term straddle
# (ArithmeticError), 3 a bad progress.


def _pair(r: LrOut) raises -> PythonObject:
    # Python.list owns its items once; list.append from Mojo would retain each.
    return Python.list(PythonObject(r.status), PythonObject(Float64(r.value)))


def lr_schedule_exact_binding(ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """The exact route of one `_Schedule` step: ip = [kind (0 constant, 1
    linear, 2 cosine), warmup, total (-1: none), t >= 1]; fp = [peak,
    min_lr]."""
    return _pair(schedule_exact(Int(py=ip[0]), Float64(py=fp[0]), Float64(py=fp[1]), Int(py=ip[1]),
                                Int(py=ip[2]), Int(py=ip[3])))


# ------------------------------------------------- the public schedule methods


def _spec(ip: PythonObject, fp: PythonObject) raises -> LrSpec:
    return LrSpec(
        Int(py=ip[0]), Int(py=ip[1]), Int(py=ip[2]), Int(py=ip[3]),
        Float64(py=fp[0]), Float64(py=fp[1]), Float64(py=fp[2]), Float64(py=fp[3]),
    )


# TOMBSTONE: lr_schedule_fill (dead native entry: no live caller; OneCycleLR._lr_at_slow calls lr_schedule_at with the exact-only kind) deleted 2026-10-26 by a98f13f57; code recoverable at 485ba9cc1.
# Restore: git apply experiments/removed/lr_schedule_fill.patch


# --------------------------------------- `lr_at(t)` and `lr_values(n)`, one call each
# (lane optimizers-thin-glue, 2026-10-21): the value AND its refusal by name.
# `style` names whose refusal messages a schedule raises: 0 the training
# schedules (`_LrTable`), 1 the sequence schedules (`_Sched`), 2 OneCycleLR.


def _lr_refusal(status: Int, t: Int, style: Int, total_steps: Int) -> Error:
    """The exception a schedule raised for a nonzero status at step t."""
    if style == 2 and status == LR_BEYOND:
        return Error("ValueError: OneCycleLR: step " + String(t) + " is beyond total_steps " + String(total_steps)
                     + " + 1 (torch refuses it too)")
    if status == LR_ONE_BASED:
        if style == 0:
            return Error("ValueError: mojolearn schedule: step t is ONE-BASED, got " + String(t))
        return Error("ValueError: schedule step t is ONE-BASED")
    if status == 1:
        return Error("OverflowError: mojolearn: schedule value overflows float32")
    if status == 2:
        return Error("ArithmeticError: mojolearn: the exact schedule value straddles a float32 rounding "
                     "boundary within 1e-60; refusing to guess a bit")
    if status == 3:
        return Error("ValueError: mojolearn: cosine progress must be in [0, 1]")
    return Error("KeyError: " + String(status))


def lr_schedule_at_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """lr_schedule_at(ip, fp, t, style): the float32 learning rate of
    ONE-BASED step `int(t)` as a float (sequence/lr_table.mojo
    `lr_spec_fill`), or the schedule's refusal."""
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) != 4:
            raise Error("TypeError: lr_schedule_at expects four arguments")
        var bi = Python.import_module("builtins")
        var ip = args[0]
        var t = Int(py=py_call(bi.int, args[2]))
        var style = Int(py=args[3])
        var store = List[Float32](length=1, fill=Float32(0.0))
        var r = lr_spec_fill(
            _spec(ip, args[1]), t, 1, store.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
        )
        if r[0] != 0:
            raise _lr_refusal(r[0], t, style, Int(py=ip[3]))
        var value = Float64(store[0])
        _ = store^
        return PythonObject(value).steal_data()
    except error:
        return optimizer_python_error(error)


def lr_schedule_series_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """lr_schedule_series(ip, fp, n, style): lr_at(1) .. lr_at(n) as an
    array('f'), or the refusal of the first failing step."""
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) != 4:
            raise Error("TypeError: lr_schedule_series expects four arguments")
        var bi = Python.import_module("builtins")
        var ip = args[0]
        var n = max(Int(py=py_call(bi.int, args[2])), 0)
        var style = Int(py=args[3])
        var out = py_call(Python.import_module("array").array, PythonObject("f"), py_call(bi.bytes, PythonObject(4 * n)))
        var r = lr_spec_fill(_spec(ip, args[1]), 1, n, F32P(unsafe_from_address=Int(py=out.buffer_info()[0])))
        if r[0] != 0:
            raise _lr_refusal(r[0], 1 + r[1], style, Int(py=ip[3]))
        return out.steal_data()
    except error:
        return optimizer_python_error(error)
