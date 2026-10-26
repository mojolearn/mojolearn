# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for `_mojolearn_x_metrics`. HOST ONLY: the same units as the
device, run in a loop on the caller's arena (x_metrics/host/program.mojo), with
the GPU binding's export names and address contract."""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from std.memory import bitcast
from x_metrics.epilogue import expected_mi, row_sum_range
from x_metrics.epilogue import scatter_rows, encode_small_i64, first_rows_i32
from x_metrics.epilogue import expected_mi_tasks, row_sum_range_tasks
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_metrics.common import X_METRICS_HOST_SABOTAGE, IP
from x_metrics.host.program import run_program_host
from x_metrics.host.out_ranges_impl import check_out_ranges


def run_binding(arena_addr: PythonObject, arena_len: PythonObject, prog_addr: PythonObject,
                stages: PythonObject) raises -> PythonObject:
    var fa = Int(py=arena_addr)
    var n = Int(py=arena_len)
    var qa = Int(py=prog_addr)
    var s = Int(py=stages)
    if fa == 0 or qa == 0 or n < 0 or s < 0:
        raise Error("x_metrics: invalid program buffers")
    with GILReleased(Python()):
        run_program_host(fa, n, qa, s)
    return PythonObject(s)


def run_out_binding(arena_addr: PythonObject, arena_len: PythonObject, prog_addr: PythonObject,
                    stages: PythonObject, outs_addr: PythonObject, nouts: PythonObject) raises -> PythonObject:
    """The device binding's `x_metrics_run_out` (lane metrics-apple2): the
    host runs in the caller's arena, so every word is already there and the
    output ranges only need checking."""
    var fa = Int(py=arena_addr)
    var n = Int(py=arena_len)
    var qa = Int(py=prog_addr)
    var s = Int(py=stages)
    var oa = Int(py=outs_addr)
    var no = Int(py=nouts)
    if fa == 0 or qa == 0 or oa == 0 or n < 0 or s < 0 or no < 0:
        raise Error("x_metrics: invalid program buffers")
    check_out_ranges(oa, no, n)
    with GILReleased(Python()):
        run_program_host(fa, n, qa, s)
    return PythonObject(s)


def first_rows_binding(codes: PythonObject, n: PythonObject, k: PythonObject, out_addr: PythonObject) raises -> PythonObject:
    """x_metrics/epilogue.mojo first_rows_i32 (lane metrics-apple3)."""
    first_rows_i32(Int(py=codes), Int(py=n), Int(py=k), Int(py=out_addr))
    return PythonObject(0)


# lane py-misc-metrics: the rest of DEVIATION 6106's O(n) epilogues
# (x_metrics/epilogue.mojo); addresses and counts in, a Float64 or a
# length out. Each raises on any case its Python fallback decides.


def x_metrics_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def x_metrics_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def x_metrics_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, "x_metrics host: pass -D MOJOLEARN_COLUMN_CPU"
    return PythonObject(column_name(TARGET_COLUMN))


def x_metrics_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(X_METRICS_HOST_SABOTAGE)


def x_metrics_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def x_metrics_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


# TOMBSTONE: x_metrics_dev_put/_free/_live/_take_rows, x_metrics_py2mojo_core, x_metrics_idn_fam2 (dead native entries: no public call reaches them) deleted 2026-10-26 by @@SHA@@; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_x_metrics.patch


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_x_metrics_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_x_metrics_host() -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_metrics_host")
        m.def_function[x_metrics_host_numeric_mode_binding]("x_metrics_host_numeric_mode")
        m.def_function[x_metrics_host_vendor_binding]("x_metrics_host_vendor")
        m.def_function[x_metrics_host_column_binding]("x_metrics_host_column")
        m.def_function[x_metrics_host_sabotage_binding]("x_metrics_host_sabotage")
        m.def_function[run_binding]("x_metrics_run")
        m.def_function[run_out_binding]("x_metrics_run_out")
        m.def_function[first_rows_binding]("x_metrics_first_rows")
        m.def_function[x_metrics_numeric_mode_binding]("x_metrics_numeric_mode")
        m.def_function[x_metrics_vendor_binding]("x_metrics_vendor")
        # TOMBSTONE: x_metrics_dev_put/_free/_live/_take_rows, x_metrics_py2mojo_core, x_metrics_idn_fam2 (dead native entries: no public call reaches them) deleted 2026-10-26 by @@SHA@@; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_metrics.patch
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_metrics_host: ", e))
