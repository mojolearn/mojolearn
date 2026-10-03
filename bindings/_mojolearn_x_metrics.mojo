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
from x_metrics.epilogue import roc_arrays, expected_mi, row_sum_range
from x_metrics.epilogue import scatter_rows, first_rows_i32, ovo_pair
from x_metrics.epilogue import (
    pr_arrays, det_arrays, ndcg_mean, class_sums, auc_xy, mi_contingency, centroids_f32, ch_extra,
    db_score, contingency_stats,
)
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


def contingency_stats_binding(o: PythonObject, dims: PythonObject, eps: PythonObject,
                              outs: PythonObject) raises -> PythonObject:
    """x_metrics/epilogue.mojo contingency_stats (lane pyglue-sweep): dims =
    (ka, kb, kk), outs = (C, C + eps or 0, rows, cols, pairs, entropies)."""
    contingency_stats(Int(py=o), Int(py=dims[0]), Int(py=dims[1]), Int(py=dims[2]), Float64(py=eps),
                      Int(py=outs[0]), Int(py=outs[1]), Int(py=outs[2]), Int(py=outs[3]), Int(py=outs[4]),
                      Int(py=outs[5]))
    return PythonObject(0)


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
        m.def_function[curve_roc_binding]("x_metrics_curve_roc")
        m.def_function[expected_mi_binding]("x_metrics_expected_mi")
        m.def_function[row_sum_range_binding]("x_metrics_row_sum_range")
        m.def_function[scatter_rows_binding]("x_metrics_scatter_rows")
        m.def_function[first_rows_binding]("x_metrics_first_rows")
        m.def_function[ovo_pair_binding]("x_metrics_ovo_pair")
        m.def_function[curve_pr_binding]("x_metrics_curve_pr")
        m.def_function[curve_det_binding]("x_metrics_curve_det")
        m.def_function[ndcg_mean_binding]("x_metrics_ndcg_mean")
        m.def_function[class_sums_binding]("x_metrics_class_sums")
        m.def_function[auc_xy_binding]("x_metrics_auc_xy")
        m.def_function[mi_contingency_binding]("x_metrics_mi_contingency")
        m.def_function[contingency_stats_binding]("x_metrics_contingency_stats")
        m.def_function[centroids_binding]("x_metrics_centroids")
        m.def_function[ch_extra_binding]("x_metrics_ch_extra")
        m.def_function[db_score_binding]("x_metrics_db_score")
        m.def_function[numeric_mode_binding]("x_metrics_numeric_mode")
        m.def_function[vendor_binding]("x_metrics_vendor")
        m.def_function[py2mojo_core_binding]("x_metrics_py2mojo_core")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_metrics: ", e))
