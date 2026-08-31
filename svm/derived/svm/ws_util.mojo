# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""`set_unavailable`, `set_upper`, `set_lower`: the flag kernels of the
working-set selection.

PORT OF `cuml/cpp/src/svm/ws_util.cuh` at cuML v26.08.00. `update_priority`
is NOT ported: it serves `PrioritySelect`, which `FIFO_strategy = true`
never calls ("only FIFO is tested so far", `workingset.h:36`); see
`svm/NOT_IMPLEMENTED.tsv`.

The flags are `UInt8` where theirs are `bool`, because the scan that
compacts them (`gbdt/gpu_util/kernel/reorder_one_bit.mojo::
block_scan_flags_kernel`) reads a byte array. Same values, 0 and 1.
"""

from std.gpu import block_dim, block_idx, thread_idx

from svm.derived.svm.smo_sets import in_lower, in_upper


#: `int TPB = 256` (`workingset.h:166`).
comptime WS_TPB = 256


def set_unavailable_kernel(
    available: MutPointer[UInt8, MutAnyOrigin],
    n_rows_in: Int32,
    idx: MutPointer[Int32, MutAnyOrigin],
    n_selected_in: Int32,
):
    """`if (tid < n_selected) available[idx[tid]] = false;`"""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid < Int(n_selected_in):
        available.unsafe_store(Int(idx.unsafe_load(tid)), UInt8(0))


def set_upper_kernel(
    available: MutPointer[UInt8, MutAnyOrigin],
    n_in: Int32,
    alpha: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    C: MutPointer[Float32, MutAnyOrigin],
):
    """`if (tid < n) available[tid] = in_upper(alpha[tid], y[tid], C[tid]);`"""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid < Int(n_in):
        var v = in_upper(
            alpha.unsafe_load(tid), y.unsafe_load(tid), C.unsafe_load(tid)
        )
        available.unsafe_store(tid, UInt8(1) if v else UInt8(0))


def set_lower_kernel(
    available: MutPointer[UInt8, MutAnyOrigin],
    n_in: Int32,
    alpha: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    C: MutPointer[Float32, MutAnyOrigin],
):
    """`if (tid < n) available[tid] = in_lower(alpha[tid], y[tid], C[tid]);`"""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid < Int(n_in):
        var v = in_lower(
            alpha.unsafe_load(tid), y.unsafe_load(tid), C.unsafe_load(tid)
        )
        available.unsafe_store(tid, UInt8(1) if v else UInt8(0))
