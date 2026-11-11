# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Lane kmeans-linear-speed stage probe (2026-09-17) for OLS and PCA. Not a
# check and not a timing row: how much of OUR IDENTICAL `ols_fit_host` and
# `pca_fit_host` is NOT the eigensolver. Each rep times the whole host entry,
# then the same staging on its own with a synchronize at every boundary
# (device allocation, the upload from a PINNED block, the Gram or covariance,
# `A^T b`); `rest_ms` is the entry minus those, which is the device Jacobi,
# the back-substitution and the readback.
#
#   pixi run mojo build -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 glm/ols_pca_stage_probe_main.mojo -o olspca_probe
#   ./olspca_probe <ols|pca> <prefix> <rows> <cols> <reps>
#
# <prefix>_X.bin, _Xc.bin, _yc.bin from
# bench/results/linear_cluster_istella_2026-09-11/probe_bins.py.
from std.memory import unsafe_memcpy
from std.sys import argv
from std.time import perf_counter_ns
from std.collections.string import atol
from max.gpu.host import DeviceBuffer, DeviceContext

from core.column_stats import STATS_TPB, xty_kernel
from core.gemm import gemm_tn
from decomposition.estimator import pca_fit_host
from decomposition.impl.linalg.detail.pca import compute_covariance
from glm.estimator import ols_fit_host


def _load(path: String, n: Int) raises -> List[UInt8]:
    var f = open(path, "r")
    var b = f.read_bytes()
    f.close()
    if len(b) != n * 4:
        raise Error(path + " holds " + String(len(b)) + " bytes, expected " + String(n * 4))
    return b^


def _ms(t0: Int) -> Float64:
    return Float64(Int(perf_counter_ns()) - t0) / 1.0e6


def _now() -> Int:
    return Int(perf_counter_ns())


def main() raises:
    var args = argv()
    if len(args) != 6:
        raise Error("usage: olspca_probe <ols|pca> <prefix> <rows> <cols> <reps>")
    var mode = String(args[1])
    var prefix = String(args[2])
    var rows = Int(atol(args[3]))
    var cols = Int(atol(args[4]))
    var reps = Int(atol(args[5]))
    var cells = rows * cols
    var ctx = DeviceContext()
    var suffix = String("_Xc.bin") if mode == "ols" else String("_X.bin")
    var bx = _load(prefix + suffix, cells)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](cells)
    var hy = ctx.enqueue_create_host_buffer[DType.float32](rows)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](cols * cols + 5 * cols)
    ctx.synchronize()
    unsafe_memcpy(dest=hx.unsafe_ptr().bitcast[UInt8](), src=bx.unsafe_ptr(), count=cells * 4)
    _ = bx^
    if mode == "ols":
        var by = _load(prefix + "_yc.bin", rows)
        unsafe_memcpy(dest=hy.unsafe_ptr().bitcast[UInt8](), src=by.unsafe_ptr(), count=rows * 4)
        _ = by^
    for rep in range(reps):
        var t0 = _now()
        if mode == "ols":
            ols_fit_host(ctx, hx.unsafe_ptr(), hy.unsafe_ptr(), hw.unsafe_ptr(), rows, cols)
        else:
            var p = hw.unsafe_ptr()
            _ = pca_fit_host(
                ctx, hx.unsafe_ptr(), p, p.unsafe_offset(cols * cols),
                p.unsafe_offset(cols * cols + cols),
                p.unsafe_offset(cols * cols + 2 * cols),
                p.unsafe_offset(cols * cols + 3 * cols),
                rows, cols, min(8, cols),
            )
        var t_total = _ms(t0)
        t0 = _now()
        var x = ctx.enqueue_create_buffer[DType.float32](cells)
        var y = ctx.enqueue_create_buffer[DType.float32](rows)
        var xa = ctx.enqueue_create_buffer[DType.float32](cells)
        var xa2 = ctx.enqueue_create_buffer[DType.float32](cells)
        var cov = ctx.enqueue_create_buffer[DType.float32](cols * cols)
        var mu = ctx.enqueue_create_buffer[DType.float32](cols)
        var ab = ctx.enqueue_create_buffer[DType.float32](cols)
        ctx.synchronize()
        var t_alloc = _ms(t0)
        t0 = _now()
        ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=y, src_ptr=hy.unsafe_ptr())
        ctx.synchronize()
        var t_up = _ms(t0)
        t0 = _now()
        if mode == "ols":
            gemm_tn(ctx, cov, x, xa, xa2, cols, cols, rows)
        else:
            compute_covariance(ctx, x, xa, xa2, mu, cov, rows, cols, True)
        ctx.synchronize()
        var t_gram = _ms(t0)
        t0 = _now()
        if mode == "ols":
            ctx.enqueue_function[xty_kernel](
                ab.unsafe_ptr(), x.unsafe_ptr(), y.unsafe_ptr(), Int32(rows), Int32(cols),
                grid_dim=(cols, 1, 1), block_dim=(STATS_TPB, 1, 1),
            )
        ctx.synchronize()
        var t_xty = _ms(t0)
        print(
            mode + " rep=" + String(rep) + " entry_total_ms=" + String(t_total)
            + " dev_alloc_ms=" + String(t_alloc) + " upload_pinned_ms=" + String(t_up)
            + " gram_or_cov_ms=" + String(t_gram) + " xty_ms=" + String(t_xty)
            + " rest_ms=" + String(t_total - t_alloc - t_up - t_gram - t_xty)
        )
        _ = x^
        _ = y^
        _ = xa^
        _ = xa2^
        _ = cov^
        _ = mu^
        _ = ab^
    _ = hx^
    _ = hy^
    _ = hw^
