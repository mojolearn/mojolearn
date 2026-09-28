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
from x_metrics.epilogue import binary_auc, binary_ap, roc_arrays, expected_mi, row_sum_range
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


def dev_live_binding() raises -> PythonObject:
    return PythonObject(X_METRICS_STORE.get_or_create_ptr()[].live)


def curve_auc_binding(arena: PythonObject, fps: PythonObject, tps: PythonObject, keep: PythonObject,
                      c: PythonObject, max_fpr_bits: PythonObject) raises -> PythonObject:
    """x_metrics/epilogue.mojo binary_auc (lane metrics-apple2)."""
    var mf = bitcast[DType.float64](Int64(Int(py=max_fpr_bits)))
    return PythonObject(binary_auc(Int(py=arena), Int(py=fps), Int(py=tps), Int(py=keep), Int(py=c), mf))


def curve_ap_binding(arena: PythonObject, fps: PythonObject, tps: PythonObject, c: PythonObject) raises -> PythonObject:
    """x_metrics/epilogue.mojo binary_ap (lane metrics-apple2)."""
    return PythonObject(binary_ap(Int(py=arena), Int(py=fps), Int(py=tps), Int(py=c)))


def curve_roc_binding(arena: PythonObject, offs: PythonObject, c: PythonObject, drop: PythonObject,
                      outs: PythonObject) raises -> PythonObject:
    """x_metrics/epilogue.mojo roc_arrays (lane metrics-apple2): offs =
    (fps, tps, thr, keep), outs = the three Float64 buffer addresses."""
    return PythonObject(roc_arrays(
        Int(py=arena), Int(py=offs[0]), Int(py=offs[1]), Int(py=offs[2]), Int(py=offs[3]), Int(py=c),
        Int(py=drop) != 0, Int(py=outs[0]), Int(py=outs[1]), Int(py=outs[2]),
    ))


def expected_mi_binding(a: PythonObject, na: PythonObject, b: PythonObject, nb: PythonObject,
                        n: PythonObject) raises -> PythonObject:
    """x_metrics/epilogue.mojo expected_mi (lane metrics-apple2)."""
    return PythonObject(expected_mi(Int(py=a), Int(py=na), Int(py=b), Int(py=nb), Int(py=n)))


def row_sum_range_binding(s: PythonObject, n: PythonObject, k: PythonObject, out_addr: PythonObject) raises -> PythonObject:
    """x_metrics/epilogue.mojo row_sum_range (lane metrics-apple2)."""
    row_sum_range(Int(py=s), Int(py=n), Int(py=k), Int(py=out_addr))
    return PythonObject(0)


def numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def vendor_binding() raises -> PythonObject:
    return PythonObject(String(COMPILED_VENDOR))


@export
def PyInit__mojolearn_x_metrics() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_metrics")
        m.def_function[run_binding]("x_metrics_run")
        m.def_function[run_out_binding]("x_metrics_run_out")
        m.def_function[run_ranges_binding]("x_metrics_run_ranges")
        m.def_function[dev_put_binding]("x_metrics_dev_put")
        m.def_function[dev_free_binding]("x_metrics_dev_free")
        m.def_function[dev_live_binding]("x_metrics_dev_live")
        m.def_function[curve_auc_binding]("x_metrics_curve_auc")
        m.def_function[curve_ap_binding]("x_metrics_curve_ap")
        m.def_function[curve_roc_binding]("x_metrics_curve_roc")
        m.def_function[expected_mi_binding]("x_metrics_expected_mi")
        m.def_function[row_sum_range_binding]("x_metrics_row_sum_range")
        m.def_function[numeric_mode_binding]("x_metrics_numeric_mode")
        m.def_function[vendor_binding]("x_metrics_vendor")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_metrics: ", e))
