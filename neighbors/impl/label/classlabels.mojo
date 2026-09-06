# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`getUniquelabels` and `make_monotonic` for ARBITRARY int32 class labels.

PORT OF `raft/label/detail/classlabels.cuh::getUniquelabels` (`:50-82`),
`map_label_kernel` (`:122-140`) and `make_monotonic` (`:164-200`) at RAFT
`661a3b8`. Transliterated except where the DEVIATION BLOCK below says so. Do
not improve.

WHY THIS IS A SECOND PORT OF THE SAME HEADER, AND NOT A REUSE
-------------------------------------------------------------
`dbscan/impl/label/classlabels.mojo` ports `make_monotonic` too, with the
unique step REPLACED by a mark-and-scan (its DEVIATION 33). That replacement
is the same function ONLY under a precondition -- every label is an integer
in `1..N` or `MAX_LABEL` -- which `weak_cc` guarantees and a CLASSIFICATION
target does not: a caller's `y` may be `{-1, 7, 1000000}`. So the k-NN
classifier needs the GENERAL route, which is what RAFT's header is, and
this file is that route. The two files port the same upstream symbols at
the same pin and differ exactly where the DBSCAN file's DEVIATION 33 says
it differs.

WHERE THIS IS CALLED FROM
-------------------------
`neighbors/impl/knn/knn.mojo` (`ML::knn_classify` / `knn_class_proba`,
`knn.cu:344,379`) calls `getUniquelabels` once per output column;
`neighbors/impl/selection/knn.mojo` (`MLCommon::Selection::class_probs`,
`knn.cuh:194`) calls `make_monotonic`.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer


comptime LABEL_TPB = 256
"""`classlabels.cuh:167`: `static const size_t TPB_X = 256`."""


# ---------------------------------------------------------------------------
# DEVIATION BLOCK 541: `getUniquelabels` IS A HOST SORT + UNIQUE
#
# THEIRS: `cub::DeviceRadixSort::SortKeys` over all `n` labels into a
#   workspace, then `cub::DeviceSelect::Unique` over the sorted workspace,
#   `d_num_selected` read back, `unique` resized and copied
#   (`classlabels.cuh:59-81`). Two CUB device primitives and one
#   device-to-host scalar.
# OURS: the labels are read to the host, sorted (insertion into a sorted
#   list with binary search, O(n log u) for u distinct labels), and the
#   distinct values returned ascending. The unique array goes back to the
#   device where the kernels below read it.
# REASON: neither CUB primitive has a counterpart on this stack
#   (`archive/reference/VENDOR_LIBS.md`: CUB/Thrust are OPEN) and a device radix SORT is not
#   a thing this tree has ported -- `select_radix` is a top-k, not a sort.
#   The function computed is the same one: the ascending distinct values of
#   an int32 array, which is a pure function of the multiset and carries no
#   float, so it is bit-identical on every column by construction. The
#   cost is one host round trip over `n` int32, once per output column per
#   `predict`, against the k-NN search that precedes it. When a device sort
#   is ported, this is the first call to move onto it.
# ---------------------------------------------------------------------------


def getUniquelabels(
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.int32],
    n: Int,
) raises -> List[Int32]:
    """`classlabels.cuh:50`: the sorted distinct values of `y[0:n]`.

    Returned as a host list rather than a resized device vector; the caller
    uploads it where a kernel needs it (`make_monotonic`, `class_vote`).
    """
    var host = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=y)
    ctx.synchronize()
    var uniq = List[Int32]()
    for i in range(n):
        var v = host.unsafe_ptr().unsafe_load(i)
        # Binary search for the insertion point; skip if present.
        var lo = 0
        var hi = len(uniq)
        while lo < hi:
            var mid = (lo + hi) // 2
            if uniq[mid] < v:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(uniq) and uniq[lo] == v:
            continue
        uniq.insert(lo, v)
    # `[[mojo-buffer-freed-at-last-use]]`: keep the host buffer alive past
    # its last read.
    _ = host^
    return uniq^


def map_label_kernel(
    map_ids: MutPointer[Int32, MutAnyOrigin],
    n_labels_in: Int32,
    inp: MutPointer[Int32, MutAnyOrigin],
    out_p: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    zero_based_in: Int32,
):
    """`map_label_kernel` (`classlabels.cuh:122-140`), `filter_op` the
    `const_op(false)` of the unfiltered overload (`:199`): every element is
    mapped. `out[tid] = i + !zero_based` for the FIRST `i` with
    `in[tid] == map_ids[i]`; linear scan, theirs.

    Integer only. Nothing here is a numeric row.
    """
    var tid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    if tid < Int(n_in):
        var v = inp.unsafe_load(tid)
        var n_labels = Int(n_labels_in)
        for i in range(n_labels):
            if v == map_ids.unsafe_load(i):
                var base = 0 if zero_based_in != 0 else 1
                out_p.unsafe_store(tid, Int32(i + base))
                break


def make_monotonic(
    ctx: DeviceContext,
    mut out_buf: DeviceBuffer[DType.int32],
    mut inp: DeviceBuffer[DType.int32],
    n: Int,
    zero_based: Bool = False,
) raises:
    """`make_monotonic(out, in, N, stream, zero_based)` (`classlabels.cuh:197`
    -> `:164`): `getUniquelabels` over `in`, then `map_label_kernel`.

    `out` may alias `in`, as it does for DBSCAN's `final_relabel`; the k-NN
    caller passes two buffers.
    """
    var uniq = getUniquelabels(ctx, inp, n)
    var n_uniq = len(uniq)
    var map_ids = ctx.enqueue_create_buffer[DType.int32](n_uniq)
    var h_map = ctx.enqueue_create_host_buffer[DType.int32](n_uniq)
    ctx.synchronize()
    for i in range(n_uniq):
        h_map.unsafe_ptr().unsafe_store(i, uniq[i])
    ctx.enqueue_copy(dst_buf=map_ids, src_ptr=h_map.unsafe_ptr())
    ctx.synchronize()
    var blocks = (n + LABEL_TPB - 1) // LABEL_TPB
    ctx.enqueue_function[map_label_kernel](
        map_ids.unsafe_ptr(),
        Int32(n_uniq),
        inp.unsafe_ptr(),
        out_buf.unsafe_ptr(),
        Int32(n),
        Int32(1 if zero_based else 0),
        grid_dim=(blocks, 1, 1),
        block_dim=(LABEL_TPB, 1, 1),
    )
    ctx.synchronize()
    _ = map_ids^
    _ = h_map^
