# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The kernel matrix rows of the SMO block solve.

These rows lived in checks/kernel_matrix.mojo until 2026-09-25 and moved here
unchanged, so an edit to them no longer changes the source closure (and the
release reuse identity) of every binding that imports the kernel matrix: only
svm/impl/smoblocksolve.mojo imports this file."""

from std.sys.compile import is_defined

from checks.kernel_matrix import (
    COLUMN_APPLE,
    COLUMN_CPU,
    COLUMN_NVIDIA,
)


def svm_block_solve_warp_folds_for[column: Int, width: Int]() -> Bool:
    """SCHEDULING row (2026-09-11, DEVIATION 2623): whether `svm/impl/smoblocksolve.mojo::smo_block_solve_kernel[width]` folds its three arg-reductions with `block_argext` warp butterflies (DEVIATION 2491) instead of the halving trees with a one-slot thread ballot that preceded it. Both select the same (value, key) element under a total order with unique keys, so no column's bits depend on this row. NVIDIA refuses the warp kernel at width 1024 (H100 80GB HBM3, driver 580.126.09, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, so every SVC fit above 512 training rows failed from 2491 through 0.8.2) and launches it at 512; the tree kernel launches at 1024 there. Fusing two of the warp folds or dropping the WSIZE threadgroup diagonal did not make the warp kernel launch. The Apple M4 and the AMD MI300X launch the warp kernel at 1024. `-D MOJOLEARN_SVM_TREE_FOLDS` takes the tree schedule on every column for an A/B."""
    comptime if is_defined["MOJOLEARN_SVM_TREE_FOLDS"]():
        return False
    if column == COLUMN_CPU:
        return True  # the every-width reading; the oracle spells the fold serially
    return not (column == COLUMN_NVIDIA and width > 512)


comptime SVM_SCHED_TREE = 0
comptime SVM_SCHED_WARP = 1
comptime SVM_SCHED_WARP_LANE0 = 2
comptime SVM_SCHED_FUSED_TREE = 3
comptime SVM_SCHED_RARY_TREE = 4


def svm_block_solve_tree_arity_for[column: Int, width: Int]() -> Int:
    """SCHEDULING row (2026-09-11, DEVIATION 2628): the arity R of SVM_SCHED_RARY_TREE's threadgroup tree (a power of two; levels = ceil(log_R(width))). Selection under a total order, so no bits depend on it. `-D MOJOLEARN_SVM_ARITY_16` / `_64` for an A/B; default 32."""
    comptime if is_defined["MOJOLEARN_SVM_ARITY_16"]():
        return 16
    comptime if is_defined["MOJOLEARN_SVM_ARITY_64"]():
        return 64
    return 32


def svm_block_solve_schedule_for[column: Int, width: Int]() -> Int:
    """SCHEDULING row (2026-09-11, DEVIATIONS 2627 and 2628): which schedule folds the three arg-reductions of `svm/impl/smoblocksolve.mojo::smo_block_solve_kernel[width]`. SVM_SCHED_TREE (0) is the pre-2491 halving trees with a thread ballot for `u` and `l` (42 barriers per inner iteration at width 1024); SVM_SCHED_WARP (1) is DEVIATION 2491's `block_argext` warp butterflies (8); SVM_SCHED_WARP_LANE0 (2) is `block_argext_lane0`, the same butterflies with the cross-warp fold on lane 0 in a runtime loop and a warp broadcast (DEVIATION 2627, 5); SVM_SCHED_FUSED_TREE (3) is one halving tree carrying the argmin, its thread and the argmax together plus a thread-carrying tree for `l`, no ballot (DEVIATION 2628, 26); SVM_SCHED_RARY_TREE (4) carries the same selections on a threadgroup tree of arity `svm_block_solve_tree_arity_for` (DEVIATION 2628's second shape, about 10). All five select the same (value, key) element under a total order with unique keys, so no column's bits depend on this row. Default: `svm_block_solve_warp_folds_for` (DEVIATION 2623) picks WARP or TREE. `-D MOJOLEARN_SVM_SCHED_TREE`, `_WARP`, `_WARP_LANE0`, `_FUSED_TREE` or `_RARY_TREE` forces one schedule on every column for an A/B (the `_RARY_TREE` define was missing from this row at 48f92b19, so that commit's R-ary kernel was unreachable)."""
    comptime if is_defined["MOJOLEARN_SVM_SCHED_TREE"]():
        return SVM_SCHED_TREE
    comptime if is_defined["MOJOLEARN_SVM_SCHED_WARP"]():
        return SVM_SCHED_WARP
    comptime if is_defined["MOJOLEARN_SVM_SCHED_WARP_LANE0"]():
        return SVM_SCHED_WARP_LANE0
    comptime if is_defined["MOJOLEARN_SVM_SCHED_FUSED_TREE"]():
        return SVM_SCHED_FUSED_TREE
    comptime if is_defined["MOJOLEARN_SVM_SCHED_RARY_TREE"]():
        return SVM_SCHED_RARY_TREE
    # APPLE (2026-09-25): the lane-0 cross-warp fold (5 barriers per inner
    # iteration against WARP's 8). Apple M4 SVC taxi 20k, alternating: WARP
    # 6,171 ms, WARP_LANE0 5,821 ms (0.943), RARY_TREE 5,950 ms; the same
    # fit in every arm (selection under a total order).
    if column == COLUMN_APPLE and svm_block_solve_warp_folds_for[column, width]():
        return SVM_SCHED_WARP_LANE0
    if svm_block_solve_warp_folds_for[column, width]():
        return SVM_SCHED_WARP
    # DEVIATION 2666 (2026-09-11): the column DEVIATION 2623 sends to the
    # halving trees -- NVIDIA above width 512, where CUDA refuses the warp
    # kernel -- takes the FUSED_TREE schedule instead. Measured on an NVIDIA
    # H200 (RunPod 4oih8bhjepzlmm, driver 570.211.01, ptxas 12.9.86), taxi
    # 10,000 x 11, five fits each: FUSED_TREE 771.1 ms, TREE 866.9 ms,
    # RARY_TREE at arity 16 1,324.4 ms and at 32 1,945.6 ms (1,937.3 ms
    # without its trailing and second update barriers), every arm giving the
    # same fits (n=400/600/2000 457e29b82bca9df9, 733a383c5699f427,
    # 2b66bc991a9c9ed0; taxi b0f91a7958162936) from five different binaries.
    # NVIDIA ONLY: Metal refuses the fused kernel's width-1024 pipeline
    # (threadgroup memory 36872 > 32768, Apple M4 gate 2026-09-11), so this
    # stays a row and never a global default. `-D MOJOLEARN_SVM_TREE_FOLDS`
    # still takes the pre-2491 trees on every column for an A/B.
    comptime if not is_defined["MOJOLEARN_SVM_TREE_FOLDS"]():
        comptime if column == COLUMN_NVIDIA:
            return SVM_SCHED_FUSED_TREE
    return SVM_SCHED_TREE
