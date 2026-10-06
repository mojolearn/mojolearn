# SPDX-License-Identifier: Apache-2.0
"""Measure complete production KMeans fits, with one in-process warmup.
Fixture generation, upload and initial-center restoration stay outside timing.
Identity was validated separately before this measurement-only campaign.
"""
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from gemm.checks.gemm_step_arms import gemm_step_env_int
from cluster.impl.detail.kmeans import kmeans_fit_main
from cluster.impl.kmeans_params import INIT_ARRAY, KMeansParams


def main() raises:
    var ctx = DeviceContext()
    var n = gemm_step_env_int("AB_ROWS", 100000)
    var d = gemm_step_env_int("AB_FEATURES", 32)
    var k = gemm_step_env_int("AB_CLUSTERS", 32)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n*d)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hc = ctx.enqueue_create_host_buffer[DType.float32](k*d)
    var x = ctx.enqueue_create_buffer[DType.float32](n*d)
    var w = ctx.enqueue_create_buffer[DType.float32](n)
    var cent = ctx.enqueue_create_buffer[DType.float32](k*d)
    var labels = ctx.enqueue_create_buffer[DType.uint32](n)
    ctx.synchronize()
    for i in range(n):
        hw.unsafe_ptr().unsafe_store(i, Float32(1.0))
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(i*d+f, Float32((i*17+f*31)%997)/Float32(997.0))
    for j in range(k):
        for f in range(d):
            hc.unsafe_ptr().unsafe_store(j*d+f, Float32((j*997//k+f*31)%997)/Float32(997.0))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=w, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()
    var params = KMeansParams.default()
    params.n_clusters = k
    params.init = INIT_ARRAY
    params.max_iter = 30
    params.n_init = 1
    params.seed = 7
    for phase in range(2):
        ctx.enqueue_copy(dst_buf=cent, src_ptr=hc.unsafe_ptr())
        ctx.synchronize()
        var begin = perf_counter_ns()
        _ = kmeans_fit_main(ctx, x, w, cent, labels, params, n, d, Float32(4096.0), Float32(4096.0))
        ctx.synchronize()
        var elapsed = perf_counter_ns()-begin
        print("MEASURE id=A08 phase="+String(phase)+" rows="+String(n)+" features="+String(d)+" clusters="+String(k)+" elapsed_ns="+String(elapsed))
