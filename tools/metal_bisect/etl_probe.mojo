# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Compile-only Metal probe for the ET device-loop kernels.

`mojo build -D ETL_PROBE=<n> -I . tools/metal_bisect/etl_probe.mojo` makes the
compiler emit the metallib for kernel group `n` only (never run it). A
"failed to compile metallib" names the group. n=0 builds every group.
"""
from std.sys.compile import is_defined
from std.sys import get_defined_int
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from extratrees.impl.decisiontree.flatnode import SparseTreeNode
from extratrees.impl.decisiontree.batched_levelalgo.split import Split
from extratrees.impl.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    InstanceRange,
    NodeWorkItem,
    WorkloadInfo,
)
from extratrees.impl.decisiontree.batched_levelalgo.kernels.et_loop_kernels import *
from extratrees.impl.decisiontree.batched_levelalgo.builder import (
    FrontierRecord,
    et_bf_admit_kernel,
    et_bf_pop_kernel,
    et_bf_expand_kernel,
)

comptime N = get_defined_int["ETL_PROBE", 0]()


def mini_split_copy(dst: MutPointer[Split, MutAnyOrigin], src: MutPointer[Split, MutAnyOrigin]):
    dst[unsafe_offset=0] = src[unsafe_offset=1]


def mini_wl_store(d: MutPointer[WorkloadInfo, MutAnyOrigin]):
    d[unsafe_offset=0] = WorkloadInfo(Int32(-1), Int32(-1), Int32(0), Int32(0))


def mini_i32_copy(dst: MutPointer[Int32, MutAnyOrigin], src: MutPointer[Int32, MutAnyOrigin]):
    dst[unsafe_offset=0] = src[unsafe_offset=1]


def mini_i32_gid(dst: MutPointer[Int32, MutAnyOrigin], n: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        dst[unsafe_offset=i] = Int32(i)


def mini_item_store(d: MutPointer[NodeWorkItem, MutAnyOrigin]):
    d[unsafe_offset=0] = NodeWorkItem(Int32(0), Int32(0), InstanceRange(Int32(0), Int32(0)))


def mini_item_copy(dst: MutPointer[NodeWorkItem, MutAnyOrigin], src: MutPointer[NodeWorkItem, MutAnyOrigin]):
    dst[unsafe_offset=0] = src[unsafe_offset=1]


def mini_split_fields(dst: MutPointer[Split, MutAnyOrigin], src: MutPointer[Split, MutAnyOrigin]):
    var q = src[unsafe_offset=1].quesval
    var c = src[unsafe_offset=1].colid
    var b = src[unsafe_offset=1].best_metric_val
    var n = src[unsafe_offset=1].n_left
    dst[unsafe_offset=0] = Split(q, c, b, n)


def mini_split_load_use(dst: MutPointer[Int32, MutAnyOrigin], src: MutPointer[Split, MutAnyOrigin]):
    var sp = src[unsafe_offset=1]
    dst[unsafe_offset=0] = sp.n_left


def mini_split_default(dst: MutPointer[Split, MutAnyOrigin]):
    dst[unsafe_offset=0] = Split()


@fieldwise_init
struct PlainS(ImplicitlyCopyable, Movable):
    var a: Float32
    var b: Int32


@fieldwise_init
struct TrivS(TrivialRegisterPassable):
    var a: Float32
    var b: Int32


@fieldwise_init
struct InlS(ImplicitlyCopyable, Movable):
    var a: Float32
    var b: Int32

    @always_inline
    def __init__(out self, *, copy: Self):
        self.a = copy.a
        self.b = copy.b


def mini_plain(dst: MutPointer[PlainS, MutAnyOrigin], src: MutPointer[PlainS, MutAnyOrigin]):
    dst[unsafe_offset=0] = src[unsafe_offset=1]


def mini_triv(dst: MutPointer[TrivS, MutAnyOrigin], src: MutPointer[TrivS, MutAnyOrigin]):
    dst[unsafe_offset=0] = src[unsafe_offset=1]


def mini_inl(dst: MutPointer[InlS, MutAnyOrigin], src: MutPointer[InlS, MutAnyOrigin]):
    dst[unsafe_offset=0] = src[unsafe_offset=1]


def P[T: AnyType](raw: DeviceBuffer[DType.uint8]) -> MutPointer[T, MutAnyOrigin]:
    return raw.unsafe_ptr().unsafe_bitcast[T]().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()


def main() raises:
    var ctx = DeviceContext()
    var raw = ctx.enqueue_create_buffer[DType.uint8](64)
    var z = Int32(0)
    var f = Float32(0)
    comptime if N == 0 or N == 1:
        ctx.enqueue_function[etl_init_kernel](
            P[Int32](raw), P[Int32](raw), P[SparseTreeNode[DType.float32]](raw), P[Int32](raw), P[Int32](raw), P[NodeWorkItem](raw), P[Int32](raw), P[Int32](raw), z, z, z, z, z,
            grid_dim=1, block_dim=ETL_TPB,
        )
    comptime if N == 0 or N == 2:
        ctx.enqueue_function[etl_pop_kernel[ETL_TPB]](
            P[Int32](raw), P[Int32](raw), P[NodeWorkItem](raw), P[Int32](raw), P[Int32](raw), z, z, z, z, z,
            grid_dim=1, block_dim=ETL_TPB,
        )
    comptime if N == 0 or N == 3:
        ctx.enqueue_function[etl_stage_kernel[ETL_TPB]](
            P[Int32](raw), z, P[NodeWorkItem](raw), P[Int32](raw), P[Int32](raw), P[NodeWorkItem](raw), P[Int32](raw), P[UInt32](raw), P[Int32](raw), P[Int32](raw), P[Int32](raw), P[Int32](raw), P[Int32](raw), P[Int32](raw),
            z, z, z, z, z, z,
            grid_dim=1, block_dim=ETL_TPB,
        )
    comptime if N == 0 or N == 4:
        ctx.enqueue_function[etl_map_kernel](
            P[Int32](raw), P[Int32](raw), P[Int32](raw), P[Int32](raw), P[WorkloadInfo](raw), z, z, grid_dim=1, block_dim=ETL_TPB,
        )
        ctx.enqueue_function[etl_copy_splits_kernel](
            P[Split](raw), P[Split](raw), z, grid_dim=1, block_dim=ETL_TPB,
        )
    comptime if N == 0 or N == 5:
        ctx.enqueue_function[etl_retry_kernel[ETL_TPB]](
            P[Int32](raw), P[NodeWorkItem](raw), P[Int32](raw), P[Split](raw), P[Int32](raw), P[NodeWorkItem](raw), P[Int32](raw), P[Int32](raw), z,
            grid_dim=1, block_dim=ETL_TPB,
        )
    comptime if N == 0 or N == 6:
        ctx.enqueue_function[etl_merge_kernel](
            P[Int32](raw), P[Split](raw), P[Split](raw), P[Int32](raw), P[Int32](raw), z, grid_dim=1, block_dim=ETL_TPB,
        )
    comptime if N == 0 or N == 7:
        ctx.enqueue_function[etl_push_rank_kernel[ETL_TPB]](
            P[Int32](raw), P[NodeWorkItem](raw), P[Int32](raw), P[Split](raw), P[Int32](raw), P[Int32](raw), P[Int32](raw), z, z, f, z,
            grid_dim=1, block_dim=ETL_TPB,
        )
    comptime if N == 0 or N == 8:
        ctx.enqueue_function[etl_push_slot_kernel](
            P[Int32](raw), P[Int32](raw), P[Int32](raw), z, z, z, grid_dim=1, block_dim=ETL_TPB,
        )
    comptime if N == 0 or N == 9:
        ctx.enqueue_function[etl_push_mark_kernel[ETL_TPB]](
            P[NodeWorkItem](raw), P[Int32](raw), P[Split](raw), P[Int32](raw), P[Int32](raw), P[Int32](raw), P[Int32](raw), P[Int32](raw), P[Int32](raw), P[Int32](raw), P[Int32](raw), P[Int32](raw), P[Int32](raw),
            z, z, z, z, z,
            grid_dim=1, block_dim=ETL_TPB,
        )
    comptime if N == 0 or N == 10:
        ctx.enqueue_function[etl_push_commit_kernel[ETL_TPB]](
            P[Int32](raw), P[Int32](raw), z, z, z, grid_dim=1, block_dim=ETL_TPB,
        )
    comptime if N == 0 or N == 11:
        ctx.enqueue_function[etl_push_write_kernel[ETL_TPB]](
            P[NodeWorkItem](raw), P[Int32](raw), P[Int32](raw), P[Split](raw), P[Int32](raw), P[Int32](raw), P[Int32](raw), P[Int32](raw), P[Int32](raw), P[SparseTreeNode[DType.float32]](raw), P[Int32](raw), P[Int32](raw), z, z, z,
            grid_dim=1, block_dim=ETL_TPB,
        )
    comptime if N == 0 or N == 12:
        ctx.enqueue_function[etl_tree_base_kernel[ETL_TPB]](
            P[Int32](raw), P[Int32](raw), z, grid_dim=1, block_dim=ETL_TPB,
        )
        ctx.enqueue_function[etl_scatter_kernel](
            P[SparseTreeNode[DType.float32]](raw), P[Int32](raw), P[Int32](raw), P[SparseTreeNode[DType.float32]](raw), P[InstanceRange](raw), z, grid_dim=1, block_dim=ETL_TPB,
        )
    comptime if N == 0 or N == 13:
        ctx.enqueue_function[et_bf_admit_kernel](
            P[Int32](raw), P[NodeWorkItem](raw), P[Int32](raw), P[Int32](raw), P[Split](raw), P[Int32](raw), P[Int32](raw), P[FrontierRecord](raw), P[Int32](raw), z, z, z, f, z,
            grid_dim=1, block_dim=ETL_TPB,
        )
    comptime if N == 0 or N == 14:
        ctx.enqueue_function[et_bf_pop_kernel[ETL_TPB]](
            P[Int32](raw), P[Int32](raw), P[FrontierRecord](raw), P[Int32](raw), P[NodeWorkItem](raw), P[Int32](raw), P[Int32](raw), P[Split](raw), z, z, z,
            grid_dim=1, block_dim=ETL_TPB,
        )
    comptime if N == 0 or N == 15:
        ctx.enqueue_function[et_bf_expand_kernel](
            P[Int32](raw), P[Int32](raw), P[NodeWorkItem](raw), P[Int32](raw), P[Split](raw), P[SparseTreeNode[DType.float32]](raw), P[Int32](raw), P[NodeWorkItem](raw), P[Int32](raw), P[Int32](raw), z, z, z, z, z, z, z,
            grid_dim=1, block_dim=ETL_TPB,
        )
    comptime if N == 41:
        ctx.enqueue_function[etl_map_kernel](
            P[Int32](raw), P[Int32](raw), P[Int32](raw), P[Int32](raw), P[WorkloadInfo](raw), z, z, grid_dim=1, block_dim=ETL_TPB,
        )
    comptime if N == 42:
        ctx.enqueue_function[etl_copy_splits_kernel](
            P[Split](raw), P[Split](raw), z, grid_dim=1, block_dim=ETL_TPB,
        )
    comptime if N == 43:
        ctx.enqueue_function[mini_split_copy](P[Split](raw), P[Split](raw), grid_dim=1, block_dim=1)
    comptime if N == 44:
        ctx.enqueue_function[mini_wl_store](P[WorkloadInfo](raw), grid_dim=1, block_dim=1)
    comptime if N == 45:
        ctx.enqueue_function[mini_i32_copy](P[Int32](raw), P[Int32](raw), grid_dim=1, block_dim=1)
    comptime if N == 46:
        ctx.enqueue_function[mini_i32_gid](P[Int32](raw), z, grid_dim=1, block_dim=1)
    comptime if N == 47:
        ctx.enqueue_function[mini_item_store](P[NodeWorkItem](raw), grid_dim=1, block_dim=1)
    comptime if N == 48:
        ctx.enqueue_function[mini_item_copy](P[NodeWorkItem](raw), P[NodeWorkItem](raw), grid_dim=1, block_dim=1)
    comptime if N == 49:
        ctx.enqueue_function[mini_split_fields](P[Split](raw), P[Split](raw), grid_dim=1, block_dim=1)
    comptime if N == 50:
        ctx.enqueue_function[mini_split_load_use](P[Int32](raw), P[Split](raw), grid_dim=1, block_dim=1)
    comptime if N == 51:
        ctx.enqueue_function[mini_split_default](P[Split](raw), grid_dim=1, block_dim=1)
    comptime if N == 60:
        ctx.enqueue_function[mini_plain](P[PlainS](raw), P[PlainS](raw), grid_dim=1, block_dim=1)
    comptime if N == 61:
        ctx.enqueue_function[mini_triv](P[TrivS](raw), P[TrivS](raw), grid_dim=1, block_dim=1)
    comptime if N == 62:
        ctx.enqueue_function[mini_inl](P[InlS](raw), P[InlS](raw), grid_dim=1, block_dim=1)
    ctx.synchronize()
