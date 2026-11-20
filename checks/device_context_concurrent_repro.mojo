# SPDX-License-Identifier: Apache-2.0
"""Concurrent shared-DeviceContext submission, with no Mojolearn imports.
Independent owners use all allocated host workers; each has distinct buffers.
The tiny kernel and input shape reproduce an ownership/submission failure,
not a benchmark. All owners remain live through synchronization and checks.
"""
from max.gpu.host import DeviceContext
from max.algorithm import sync_parallelize
from max.gpu import block_dim, block_idx, thread_idx
from std.sys.info import num_logical_cores
from std.os import getenv

comptime FP = MutPointer[Float32, MutAnyOrigin]

def scale(src: FP, dst: FP, count: Int64):
    var index = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if index < Int(count):
        dst[index] = src[index] * 2


def operation(ctx: DeviceContext, unit: Int) raises:
    var source = List[Float32](length=19, fill=Float32(unit + 1))
    var result = List[Float32](length=19, fill=-1)
    var src = ctx.enqueue_create_buffer[DType.float32](19)
    var dst = ctx.enqueue_create_buffer[DType.float32](19)
    ctx.enqueue_copy(dst_buf=src, src_ptr=source.unsafe_ptr())
    ctx.enqueue_function[scale](src.unsafe_ptr(), dst.unsafe_ptr(), Int64(19),
                               grid_dim=1, block_dim=128)
    ctx.enqueue_copy(dst_ptr=result.unsafe_ptr(), src_buf=dst)
    ctx.synchronize()
    for index in range(19):
        if result[index] != Float32(2 * (unit + 1)):
            raise Error("distinct owner output mismatch")
    _ = source^
    _ = result^
    _ = src^
    _ = dst^
    _ = ctx


def main() raises:
    var configured = getenv("MOJOLEARN_NATIVE_TEST_WORKERS")
    var workers = num_logical_cores() if configured == "" else Int(atol(configured))
    if workers < 2:
        raise Error("at least two allocated workers required")
    var ctx = DeviceContext()
    var failures = List[Int](length=workers, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm ctx, imm fp}:
        try:
            operation(ctx, unit)
        except error:
            print("FAIL owner", unit, String(error))
            fp.unsafe_store(unit, 1)
    print("START shared DeviceContext submission; workers", workers)
    sync_parallelize(task, workers)
    var failed = 0
    for value in failures:
        failed += value
    _ = failures^
    _ = ctx^
    if failed:
        raise Error("one or more concurrent owners failed")
    print("PASS shared DeviceContext submission; workers", workers)
