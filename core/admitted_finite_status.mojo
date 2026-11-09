# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The finite status word of an admitted input, tested where the kernel
entry already holds its words (lane a5-admit-core-followup, 2026-11-08; plan
docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md section 3.2 step 3;
docs/plans/REGRESSIONS_0866_VS_0855.md row R1).

`admit_x(..., defer_finite=True)` (bindings/admitted_input.mojo) borrows a
float32 C-order X with NO host pass over it and returns `finite = False`:
the finiteness test moves into the kernel entry that reads X anyway.

  GPU entry   after the upload of X it makes in any case, enqueue
              `enqueue_finite_status(ctx, d_x, n, d_status, slot)` (no new
              synchronisation: one grid-stride kernel over the resident
              words) and copy `d_status` back with the result it already
              downloads. Word `slot` of the status buffer is 1 when any
              word is NaN or infinite, else it stays at the 0 the entry
              filled it with (`enqueue_fill`).
  host entry  the CPU column tests X before it reads it, with
              core/host_finite_scan.mojo `host_all_finite_f32` or its status
              word form `host_finite_status(address, n)` (the parallel SIMD
              scan; this module stays device-only: GPU code imports it, and
              the host-route fence refuses host threads reached from GPU code).

The entry returns the status word(s) to its door, which refuses through
`admitted_finite_refuse` (scikit-learn's wording, unchanged). A door that
does not pass `defer_finite=True` keeps the scan inside admit_x.

The test is on BITS (`|bits| >= 0x7F800000`), never a float compare
(Metal flushes compare operands), so a subnormal passes and NaN, +inf and
-inf fail, on NVIDIA, AMD, Apple and the host alike. Each thread stores the
constant 1 into its own input's word: no atomics, no order, no 64-bit word
(Apple has no 64-bit atomic). Numerics: none; no output bit changes.
"""
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceContext

comptime AFS_TPB = 256
comptime AFS_REDUCE_BLOCKS = 1024
#: The largest word count of one launch (the kernel's count is an Int32).
comptime AFS_MAX_N = 2147483000
#: Status word values.
comptime AFS_FINITE: Int32 = 0
comptime AFS_NONFINITE: Int32 = 1

comptime _AFS_F32 = MutPointer[Float32, MutAnyOrigin]
comptime _AFS_I32 = MutPointer[Int32, MutAnyOrigin]


def _afs_nonfinite_kernel(src: _AFS_F32, n_: Int32, status: _AFS_I32, slot_: Int32):
    """status[slot] = 1 when any float32 word of src[0:n] is NaN or infinite."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var n = Int(n_)
    var hit = False
    while i < n:
        if (bitcast[DType.uint32](src.unsafe_load(i)) & UInt32(0x7FFFFFFF)) >= UInt32(0x7F800000):
            hit = True
            break
        i += stride
    if hit:
        status.unsafe_store(Int(slot_), AFS_NONFINITE)


def enqueue_finite_status[wo: MutOrigin, so: MutOrigin, //](
    ctx: DeviceContext, d_words: MutPointer[Float32, wo], n: Int, d_status: MutPointer[Int32, so], slot: Int,
) raises:
    """Enqueue the finite test of the `n` resident float32 words at device
    pointer `d_words`; word `slot` of the device status buffer `d_status`
    becomes 1 when any of them is NaN or infinite. The caller fills the
    status buffer with 0 first and reads it back with its result: this adds
    no synchronisation. Spans past one launch's Int32 count go in
    `AFS_MAX_N`-word launches. Either pointer may carry its buffer's origin
    (a `DeviceBuffer.unsafe_ptr()`); the kernel takes untracked views."""
    var off = 0
    while off < n:  # driver-loop: over Int32-sized launches of one span
        var m = min(AFS_MAX_N, n - off)
        var grid = max(1, min(AFS_REDUCE_BLOCKS, (m + AFS_TPB - 1) // AFS_TPB))
        ctx.enqueue_function[_afs_nonfinite_kernel](
            d_words.unsafe_origin_cast[MutAnyOrigin]() + off, Int32(m), d_status.unsafe_origin_cast[MutAnyOrigin](), Int32(slot),
            grid_dim=grid, block_dim=AFS_TPB,
        )
        off += m


# TOMBSTONE: host_finite_status (the host status word, a wrapper of core/host_finite_scan host_all_finite_f32) deleted 2026-11-09 by lane a1b-doors-opt-in-trees-linear: no caller, and its host-thread import made this device module unreachable from GPU code; the CPU column calls host_all_finite_f32 directly (bindings/_mojolearn_x_linear_host.mojo `_x_status`). Never reached main: recoverable at f59d913b0b.
# host_finite_status (the CPU column's status word) moved to
# core/host_finite_scan.mojo by lane a6: this module is imported by GPU
# kernel entries, and a host scan in it made every such entry reach host
# threads (the no-host-routes fence).
# integ-w3: the merge keeps lane a6's move: host_finite_status lives in core/host_finite_scan.mojo
# (caller bindings/_mojolearn_estimators_host.mojo); only the copy in this module is gone.
