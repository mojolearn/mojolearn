# SPDX-License-Identifier: Apache-2.0
"""REMOTE GPU ONLY: contract 11.2 shipped shape, ~4.4 GB live device buffers."""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceContext
from embedding.checks.embedding_identical import identical_embedding_backward_into, EMB_TPB
from embedding.checks.embedding_sort import PLAN_SCAN, PLAN_SORT
from embedding.checks.embedding_oracle import EmbConfig
from embedding.checks.embedding_check import _upload_i32, _download_i32, compare_i32

comptime V = 128256
comptime D = 4096
comptime T = 4096
comptime CELLS = V * D
comptime CHUNK = 4096
comptime CHECKS = (CELLS + CHUNK - 1) // CHUNK


def fill_dy(dy: MutPointer[Float32, MutAnyOrigin]):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i < T * D:
        dy.unsafe_store(i, Float32(i % 31 - 15) * Float32(0.125))


def compare_bits(a: MutPointer[Float32, MutAnyOrigin], b: MutPointer[Float32, MutAnyOrigin], errors: MutPointer[Int32, MutAnyOrigin]):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= CHECKS:
        return
    var bad = Int32(0)
    for offset in range(CHUNK):
        var cell = i * CHUNK + offset
        if cell < CELLS:
            if bitcast[DType.uint32](a.unsafe_load(cell)) != bitcast[DType.uint32](b.unsafe_load(cell)):
                bad += 1
    errors.unsafe_store(i, bad)


def main() raises:
    var ctx = DeviceContext()
    var cfg = EmbConfig(V, D, V - 1, False)
    var h_ids = List[Int32]()
    for i in range(T):
        h_ids.append(Int32(V - 1 if i % 17 == 0 else ((i % 1024) * 127) % V))
    var ids = _upload_i32(ctx, h_ids)
    var dy = ctx.enqueue_create_buffer[DType.float32](T * D)
    var baseline = ctx.enqueue_create_buffer[DType.float32](CELLS)
    var candidate = ctx.enqueue_create_buffer[DType.float32](CELLS)
    var counts = ctx.enqueue_create_buffer[DType.int32](V)
    var begin = ctx.enqueue_create_buffer[DType.int32](V + 1)
    var perm = ctx.enqueue_create_buffer[DType.int32](T)
    var errors = ctx.enqueue_create_buffer[DType.int32](CHECKS)
    ctx.enqueue_function[fill_dy](dy.unsafe_ptr(), grid_dim=((T * D + 255) // 256, 1, 1), block_dim=(256, 1, 1))
    identical_embedding_backward_into(ctx, baseline, dy, ids, counts, begin, perm, T, cfg, PLAN_SCAN, EMB_TPB)
    ctx.synchronize()
    var ref_counts = _download_i32(ctx, counts, V)
    var ref_begin = _download_i32(ctx, begin, V + 1)
    var used = Int(ref_begin[V])
    var ref_perm = _download_i32(ctx, perm, used)
    var geometries: List[Int] = [32, 96, 160]
    for plan in range(2):
        for g in range(len(geometries)):
            identical_embedding_backward_into(ctx, candidate, dy, ids, counts, begin, perm, T, cfg, plan, geometries[g])
            ctx.enqueue_function[compare_bits](baseline.unsafe_ptr(), candidate.unsafe_ptr(), errors.unsafe_ptr(), grid_dim=((CHECKS + 127) // 128, 1, 1), block_dim=(128, 1, 1))
            ctx.synchronize()
            var failures = _download_i32(ctx, errors, CHECKS)
            var bad = 0
            for i in range(CHECKS):
                bad += Int(failures[i])
            var got_counts = _download_i32(ctx, counts, V)
            var got_begin = _download_i32(ctx, begin, V + 1)
            var got_perm = _download_i32(ctx, perm, used)
            if bad != 0 or compare_i32("counts", ref_counts, got_counts, True).n_diff != 0 or compare_i32("begin", ref_begin, got_begin, True).n_diff != 0 or compare_i32("perm", ref_perm, got_perm, True).n_diff != 0:
                raise Error("shipped embedding plan/geometry mismatch")
            print("shipped PASS plan=", plan, "threads=", geometries[g], "FP32 cells=", CELLS, "used perm=", used)
    _ = ids^
    _ = dy^
    _ = baseline^
    _ = candidate^
    _ = counts^
    _ = begin^
    _ = perm^
    _ = errors^
    print("PLAN_SORT shipped gate PASS: V128256 D4096 T4096, both plans, three geometries, every output bit")
