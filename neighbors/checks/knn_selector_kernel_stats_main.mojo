# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/knn-selector-speed: the compiled footprint of the tiled k-NN arm's
kernels on this box (registers, local bytes, shared bytes, and how many
256-thread blocks one multiprocessor holds), read from the runtime with
`DeviceFunction.get_attribute` (the route of DEVIATION 2519's
`tools/knn_selector_kernel_stats.sh`). Says nothing about speed. The
instantiations follow the build's defines (`-D MOJOLEARN_EXPERIMENTAL_KNN_TOPK_KEY32=1`,
the bounded kernel's diagnostic arms), so one run per define set."""
from std.sys.compile import is_defined
from max.gpu.host import Attribute, DeviceContext
from neighbors.checks.smem_distance_tile import smem_distance_tile_kernel
from neighbors.checks.knn_selector_bound_compact import (
    SBC_DEPTH,
    bound_compact_lists_kernel,
    bound_compact_select_kernel,
)


def main() raises:
    var ctx = DeviceContext()
    comptime k_matrix = smem_distance_tile_kernel[False, True, False]
    comptime k_topk = smem_distance_tile_kernel[True, True, False]
    comptime k_bounded = smem_distance_tile_kernel[True, True, True]
    comptime k_select = bound_compact_select_kernel[SBC_DEPTH]
    var f0 = ctx.compile_function[k_matrix]()
    print("KNN_KERNEL_STATS label=smem_matrix regs=", f0.get_attribute(Attribute.NUM_REGS), " local=", f0.get_attribute(Attribute.LOCAL_SIZE_BYTES), " shared=", f0.get_attribute(Attribute.SHARED_SIZE_BYTES), " blocks_per_sm_256=", f0.occupancy_max_active_blocks_per_multiprocessor(256, 0), sep="")
    print("KNN_KERNEL_ASM_BEGIN smem_topk")
    var f1 = ctx.compile_function[k_topk, dump_asm=is_defined["MOJOLEARN_KNN_STATS_DUMP_ASM"]()]()
    print("KNN_KERNEL_ASM_END smem_topk")
    print("KNN_KERNEL_STATS label=smem_topk regs=", f1.get_attribute(Attribute.NUM_REGS), " local=", f1.get_attribute(Attribute.LOCAL_SIZE_BYTES), " shared=", f1.get_attribute(Attribute.SHARED_SIZE_BYTES), " blocks_per_sm_256=", f1.occupancy_max_active_blocks_per_multiprocessor(256, 0), sep="")
    var f2 = ctx.compile_function[k_bounded]()
    print("KNN_KERNEL_STATS label=smem_topk_bounded regs=", f2.get_attribute(Attribute.NUM_REGS), " local=", f2.get_attribute(Attribute.LOCAL_SIZE_BYTES), " shared=", f2.get_attribute(Attribute.SHARED_SIZE_BYTES), " blocks_per_sm_256=", f2.occupancy_max_active_blocks_per_multiprocessor(256, 0), sep="")
    var f3 = ctx.compile_function[k_select]()
    print("KNN_KERNEL_STATS label=bound_compact_select regs=", f3.get_attribute(Attribute.NUM_REGS), " local=", f3.get_attribute(Attribute.LOCAL_SIZE_BYTES), " shared=", f3.get_attribute(Attribute.SHARED_SIZE_BYTES), " blocks_per_sm_256=", f3.occupancy_max_active_blocks_per_multiprocessor(256, 0), sep="")
    var f4 = ctx.compile_function[bound_compact_lists_kernel]()
    print("KNN_KERNEL_STATS label=bound_compact_lists regs=", f4.get_attribute(Attribute.NUM_REGS), " local=", f4.get_attribute(Attribute.LOCAL_SIZE_BYTES), " shared=", f4.get_attribute(Attribute.SHARED_SIZE_BYTES), " blocks_per_sm_256=", f4.occupancy_max_active_blocks_per_multiprocessor(256, 0), sep="")
