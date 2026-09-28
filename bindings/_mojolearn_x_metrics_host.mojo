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
from x_metrics.epilogue import binary_auc, binary_ap, roc_arrays, expected_mi, row_sum_range
from x_metrics.epilogue import scatter_rows, encode_small_i64, first_rows_i32, ovo_pair
from x_metrics.epilogue import (
    pr_arrays, det_arrays, ndcg_mean, class_sums, auc_xy, mi_contingency, centroids_f32, ch_extra,
    db_score,
)
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_metrics.common import X_METRICS_HOST_SABOTAGE, IP
from x_metrics.host.program import run_program_host


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
    for k in range(no):
        var lo = Int(IP(unsafe_from_address=oa).unsafe_load(4 * k))
        var hi = Int(IP(unsafe_from_address=oa).unsafe_load(4 * k + 1))
        var cn = Int(IP(unsafe_from_address=oa).unsafe_load(4 * k + 2))
        if lo < 0 or hi < lo or hi > n or cn >= n:
            raise Error("x_metrics: output range outside the arena")
    with GILReleased(Python()):
        run_program_host(fa, n, qa, s)
    return PythonObject(s)


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


def encode_small_binding(src: PythonObject, n: PythonObject, classes: PythonObject, max_classes: PythonObject,
                         codes: PythonObject) raises -> PythonObject:
    """x_metrics/epilogue.mojo encode_small_i64 (lane metrics-apple3)."""
    return PythonObject(encode_small_i64(Int(py=src), Int(py=n), Int(py=classes), Int(py=max_classes), Int(py=codes)))


def first_rows_binding(codes: PythonObject, n: PythonObject, k: PythonObject, out_addr: PythonObject) raises -> PythonObject:
    """x_metrics/epilogue.mojo first_rows_i32 (lane metrics-apple3)."""
    first_rows_i32(Int(py=codes), Int(py=n), Int(py=k), Int(py=out_addr))
    return PythonObject(0)


def ovo_pair_binding(codes: PythonObject, scores: PythonObject, dims: PythonObject,
                     outs: PythonObject) raises -> PythonObject:
    """x_metrics/epilogue.mojo ovo_pair (lane metrics-apple3): dims = (n, k,
    a, b, m), outs = (scores a, scores b, flags a, flags b) addresses."""
    return PythonObject(ovo_pair(
        Int(py=codes), Int(py=scores), Int(py=dims[0]), Int(py=dims[1]), Int(py=dims[2]), Int(py=dims[3]),
        Int(py=outs[0]), Int(py=outs[1]), Int(py=outs[2]), Int(py=outs[3]), Int(py=dims[4]),
    ))


def scatter_rows_binding(src: PythonObject, dst: PythonObject, idx: PythonObject, n: PythonObject,
                         n_dst: PythonObject, row_bytes: PythonObject) raises -> PythonObject:
    """x_metrics/epilogue.mojo scatter_rows (lane metrics-apple3)."""
    scatter_rows(Int(py=src), Int(py=dst), Int(py=idx), Int(py=n), Int(py=n_dst), Int(py=row_bytes))
    return PythonObject(0)


# lane py-misc-metrics: the rest of DEVIATION 6106's O(n) epilogues
# (x_metrics/epilogue.mojo); addresses and counts in, a Float64 or a
# length out. Each raises on any case its Python fallback decides.


def curve_pr_binding(arena: PythonObject, offs: PythonObject, c: PythonObject, drop: PythonObject,
                     outs: PythonObject) raises -> PythonObject:
    """pr_arrays: offs = (fps, tps, thr), outs = (precision, recall, thresholds) Float64 addresses."""
    return PythonObject(pr_arrays(
        Int(py=arena), Int(py=offs[0]), Int(py=offs[1]), Int(py=offs[2]), Int(py=c), Int(py=drop) != 0,
        Int(py=outs[0]), Int(py=outs[1]), Int(py=outs[2]),
    ))


def curve_det_binding(arena: PythonObject, offs: PythonObject, c: PythonObject, drop: PythonObject,
                      outs: PythonObject) raises -> PythonObject:
    """det_arrays: offs = (fps, tps, thr), outs = (fpr, fnr, thresholds) Float64 addresses."""
    return PythonObject(det_arrays(
        Int(py=arena), Int(py=offs[0]), Int(py=offs[1]), Int(py=offs[2]), Int(py=c), Int(py=drop) != 0,
        Int(py=outs[0]), Int(py=outs[1]), Int(py=outs[2]),
    ))


def ndcg_mean_binding(arena: PythonObject, gain: PythonObject, ideal: PythonObject, n: PythonObject,
                      w: PythonObject) raises -> PythonObject:
    return PythonObject(ndcg_mean(Int(py=arena), Int(py=gain), Int(py=ideal), Int(py=n), Int(py=w)))


def class_sums_binding(codes: PythonObject, w: PythonObject, n: PythonObject, k: PythonObject,
                       out_addr: PythonObject) raises -> PythonObject:
    class_sums(Int(py=codes), Int(py=w), Int(py=n), Int(py=k), Int(py=out_addr))
    return PythonObject(0)


def auc_xy_binding(x: PythonObject, y: PythonObject, n: PythonObject) raises -> PythonObject:
    return PythonObject(auc_xy(Int(py=x), Int(py=y), Int(py=n)))


def mi_contingency_binding(c: PythonObject, ka: PythonObject, kb: PythonObject) raises -> PythonObject:
    return PythonObject(mi_contingency(Int(py=c), Int(py=ka), Int(py=kb)))


def centroids_binding(arena: PythonObject, sums: PythonObject, counts: PythonObject, k: PythonObject,
                      d: PythonObject, out_addr: PythonObject) raises -> PythonObject:
    centroids_f32(Int(py=arena), Int(py=sums), Int(py=counts), Int(py=k), Int(py=d), Int(py=out_addr))
    return PythonObject(0)


def ch_extra_binding(arena: PythonObject, sums: PythonObject, gsum: PythonObject, counts: PythonObject,
                     k: PythonObject, d: PythonObject, n: PythonObject) raises -> PythonObject:
    return PythonObject(ch_extra(Int(py=arena), Int(py=sums), Int(py=gsum), Int(py=counts), Int(py=k),
                                 Int(py=d), Int(py=n)))


def db_score_binding(arena: PythonObject, sums: PythonObject, counts: PythonObject, k: PythonObject,
                     d: PythonObject, per: PythonObject) raises -> PythonObject:
    return PythonObject(db_score(Int(py=arena), Int(py=sums), Int(py=counts), Int(py=k), Int(py=d),
                                 Int(py=per)))


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


@export
def PyInit__mojolearn_x_metrics_host() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_metrics_host")
        m.def_function[x_metrics_host_numeric_mode_binding]("x_metrics_host_numeric_mode")
        m.def_function[x_metrics_host_vendor_binding]("x_metrics_host_vendor")
        m.def_function[x_metrics_host_column_binding]("x_metrics_host_column")
        m.def_function[x_metrics_host_sabotage_binding]("x_metrics_host_sabotage")
        m.def_function[run_binding]("x_metrics_run")
        m.def_function[run_out_binding]("x_metrics_run_out")
        m.def_function[curve_auc_binding]("x_metrics_curve_auc")
        m.def_function[curve_ap_binding]("x_metrics_curve_ap")
        m.def_function[curve_roc_binding]("x_metrics_curve_roc")
        m.def_function[expected_mi_binding]("x_metrics_expected_mi")
        m.def_function[row_sum_range_binding]("x_metrics_row_sum_range")
        m.def_function[scatter_rows_binding]("x_metrics_scatter_rows")
        m.def_function[encode_small_binding]("x_metrics_encode_small_i64")
        m.def_function[first_rows_binding]("x_metrics_first_rows")
        m.def_function[ovo_pair_binding]("x_metrics_ovo_pair")
        m.def_function[curve_pr_binding]("x_metrics_curve_pr")
        m.def_function[curve_det_binding]("x_metrics_curve_det")
        m.def_function[ndcg_mean_binding]("x_metrics_ndcg_mean")
        m.def_function[class_sums_binding]("x_metrics_class_sums")
        m.def_function[auc_xy_binding]("x_metrics_auc_xy")
        m.def_function[mi_contingency_binding]("x_metrics_mi_contingency")
        m.def_function[centroids_binding]("x_metrics_centroids")
        m.def_function[ch_extra_binding]("x_metrics_ch_extra")
        m.def_function[db_score_binding]("x_metrics_db_score")
        m.def_function[x_metrics_numeric_mode_binding]("x_metrics_numeric_mode")
        m.def_function[x_metrics_vendor_binding]("x_metrics_vendor")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_metrics_host: ", e))
