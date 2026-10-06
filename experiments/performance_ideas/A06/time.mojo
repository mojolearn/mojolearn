# SPDX-License-Identifier: Apache-2.0
"""Full public KNN search timing, with identical inputs and runtime batch arms."""
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from gemm.checks.gemm_step_arms import gemm_step_env_int
from neighbors.estimator import knn_search, DEFAULT_QUERY_TILE
from bench.knn_smallk_dispatch_fixture import _coordinate


def main() raises:
    var ctx = DeviceContext()
    var n = gemm_step_env_int("AB_ROWS", 100000)
    var q = gemm_step_env_int("AB_QUERIES", 2000)
    var d = gemm_step_env_int("AB_FEATURES", 8)
    var k = gemm_step_env_int("AB_K", 8)
    var index = ctx.enqueue_create_host_buffer[DType.float32](n*d)
    var queries = ctx.enqueue_create_host_buffer[DType.float32](q*d)
    var distances = ctx.enqueue_create_host_buffer[DType.float32](q*k)
    var indices = ctx.enqueue_create_host_buffer[DType.uint32](q*k)
    ctx.synchronize()
    for row in range(n):
        for f in range(d):
            index.unsafe_ptr().unsafe_store(row*d+f, _coordinate(row,f,0))
    for row in range(q):
        for f in range(d):
            queries.unsafe_ptr().unsafe_store(row*d+f, _coordinate(row,f,593))
    for arm in range(2):
        var tile = 256 if arm == 0 else DEFAULT_QUERY_TILE
        for phase in range(2):
            var begin = perf_counter_ns()
            var used = knn_search(ctx,index.unsafe_ptr(),n,queries.unsafe_ptr(),q,d,k,distances.unsafe_ptr(),indices.unsafe_ptr(),requested_query_tile=tile)
            ctx.synchronize()
            var elapsed = perf_counter_ns()-begin
            print("MEASURE id=A06 arm="+String(arm)+" phase="+String(phase)+" rows="+String(n)+" queries="+String(q)+" features="+String(d)+" k="+String(k)+" used_tile="+String(used)+" elapsed_ns="+String(elapsed))
