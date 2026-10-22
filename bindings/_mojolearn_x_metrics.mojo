# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE METRICS LANE'S GPU BINDING (the evaluation metrics and the
model_selection helpers the expansion added). One entry runs a program of
units on the device (x_metrics/common.mojo); the host binding runs the same
units on the CPU."""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from std.memory import bitcast
from x_metrics.common import PY2MOJO_CORE_ON
from x_metrics.cls_epi import IDN_CLS_EPI
from x_metrics.epilogue import expected_mi, row_sum_range
from x_metrics.epilogue import scatter_rows, first_rows_i32
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_metrics.device import (
    run_program_device, run_program_device_out, run_program_device_ranges, metrics_ctx, X_METRICS_STORE,
)


def run_binding(arena_addr: PythonObject, arena_len: PythonObject, prog_addr: PythonObject,
                stages: PythonObject) raises -> PythonObject:
    var fa = Int(py=arena_addr)
    var n = Int(py=arena_len)
    var qa = Int(py=prog_addr)
    var s = Int(py=stages)
    if fa == 0 or qa == 0 or n < 0 or s < 0:
        raise Error("x_metrics: invalid program buffers")
    with GILReleased(Python()):
        run_program_device(fa, n, qa, s)
    return PythonObject(s)


def run_out_binding(arena_addr: PythonObject, arena_len: PythonObject, prog_addr: PythonObject,
                    stages: PythonObject, outs_addr: PythonObject, nouts: PythonObject) raises -> PythonObject:
    """`x_metrics_run` that downloads only the `nouts` output ranges at
    `outs_addr` (Int32 quads [lo, hi, CNT, mult], x_metrics/device.mojo
    run_program_device_out; lane metrics-apple2)."""
    var fa = Int(py=arena_addr)
    var n = Int(py=arena_len)
    var qa = Int(py=prog_addr)
    var s = Int(py=stages)
    var oa = Int(py=outs_addr)
    var no = Int(py=nouts)
    if fa == 0 or qa == 0 or oa == 0 or n < 0 or s < 0 or no < 0:
        raise Error("x_metrics: invalid program buffers")
    with GILReleased(Python()):
        run_program_device_out(fa, n, qa, s, oa, no)
    return PythonObject(s)


def run_ranges_binding(arena_addr: PythonObject, prog_addr: PythonObject, sizes: PythonObject,
                       ins_addr: PythonObject, outs_addr: PythonObject) raises -> PythonObject:
    """`x_metrics_run_out` that also uploads only the input ranges (lane
    py-shared, core/arena_io.mojo). sizes = (arena_len, stages, nins,
    nouts); `nins` Int32 triples [lo, hi, src] at ins_addr, `nouts` quads
    [lo, hi, CNT, mult] at outs_addr."""
    var fa = Int(py=arena_addr)
    var qa = Int(py=prog_addr)
    var n = Int(py=sizes[0])
    var s = Int(py=sizes[1])
    var ni = Int(py=sizes[2])
    var no = Int(py=sizes[3])
    var ia = Int(py=ins_addr)
    var oa = Int(py=outs_addr)
    if fa == 0 or qa == 0 or n < 0 or s < 0 or ni < 0 or no < 0:
        raise Error("x_metrics: invalid program buffers")
    with GILReleased(Python()):
        run_program_device_ranges(fa, n, qa, s, ia, ni, oa, no)
    return PythonObject(s)


def dev_put_binding(addr: PythonObject, n_words: PythonObject) raises -> PythonObject:
    """A resident copy of n_words host words (core/device_store.mojo); its id."""
    var a = Int(py=addr)
    var n = Int(py=n_words)
    var id: Int
    with GILReleased(Python()):
        id = X_METRICS_STORE.get_or_create_ptr()[].put(metrics_ctx(), a, n)
    return PythonObject(id)


def dev_free_binding(id: PythonObject) raises -> PythonObject:
    var i = Int(py=id)
    with GILReleased(Python()):
        X_METRICS_STORE.get_or_create_ptr()[].free(metrics_ctx(), i)
    return PythonObject(None)


def dev_take_rows_binding(
    src_id: PythonObject, row_words: PythonObject, idx_addr: PythonObject, n_idx: PythonObject
) raises -> PythonObject:
    """A new resident slot: rows `idx` (n_idx host int64 words) of slot
    src_id, gathered on the device (core/device_store.mojo `take_rows`,
    lane cpu4-misc device-rows input); its id."""
    var s = Int(py=src_id)
    var w = Int(py=row_words)
    var a = Int(py=idx_addr)
    var n = Int(py=n_idx)
    var id: Int
    with GILReleased(Python()):
        id = X_METRICS_STORE.get_or_create_ptr()[].take_rows(metrics_ctx(), s, w, a, n)
    return PythonObject(id)


def dev_live_binding() raises -> PythonObject:
    return PythonObject(X_METRICS_STORE.get_or_create_ptr()[].live)


def first_rows_binding(codes: PythonObject, n: PythonObject, k: PythonObject, out_addr: PythonObject) raises -> PythonObject:
    """x_metrics/epilogue.mojo first_rows_i32 (lane metrics-apple3)."""
    first_rows_i32(Int(py=codes), Int(py=n), Int(py=k), Int(py=out_addr))
    return PythonObject(0)


# lane py-misc-metrics: the rest of DEVIATION 6106's O(n) epilogues
# (x_metrics/epilogue.mojo); addresses and counts in, a Float64 or a
# length out. Each raises on any case its Python fallback decides.


def numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def vendor_binding() raises -> PythonObject:
    return PythonObject(String(COMPILED_VENDOR))


def py2mojo_core_binding() raises -> PythonObject:
    """1 when the label layouts run in the binding (lane apple-fast-py2mojo-core),
    0 under -D MOJOLEARN_PY2MOJO_core_OFF (the Python layouts, the A/B arm)."""
    comptime if PY2MOJO_CORE_ON:
        return PythonObject(1)
    else:
        return PythonObject(0)


def idn_fam2_binding() raises -> PythonObject:
    """Lane fam2-prep-metrics: the bits of the IDENTICAL metric switches this
    binding was built with (1 IDN_CLS_EPI: op 55, the set-wise classification
    epilogue, x_metrics/cls_epi.mojo), read by
    python/mojolearn/_expansion_metrics.py `_idn_fam2`."""
    return PythonObject(1 if IDN_CLS_EPI else 0)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_x_metrics.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_x_metrics() -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_metrics")
        m.def_function[run_binding]("x_metrics_run")
        m.def_function[run_out_binding]("x_metrics_run_out")
        m.def_function[run_ranges_binding]("x_metrics_run_ranges")
        m.def_function[dev_put_binding]("x_metrics_dev_put")
        m.def_function[dev_free_binding]("x_metrics_dev_free")
        m.def_function[dev_live_binding]("x_metrics_dev_live")
        m.def_function[dev_take_rows_binding]("x_metrics_dev_take_rows")
        m.def_function[first_rows_binding]("x_metrics_first_rows")
        m.def_function[numeric_mode_binding]("x_metrics_numeric_mode")
        m.def_function[vendor_binding]("x_metrics_vendor")
        m.def_function[py2mojo_core_binding]("x_metrics_py2mojo_core")
        m.def_function[idn_fam2_binding]("x_metrics_idn_fam2")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_metrics: ", e))
