# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Every vendor primitive this repo uses or recommends, run against an
independent host oracle.

THE RULE THIS FILE EXISTS TO ENFORCE
------------------------------------
**A signature check proves the GPU is REACHABLE. Only a run against an
independent oracle proves the answer is RIGHT.**

`archive/reference/VENDOR_LIBRARIES.md` listed `nn.argsort.argsort` as GPU because it takes a
`ctx` and a `target`. It does take both. It also returns a well-formed
permutation that is not sorted above 256 elements, raises nothing, and would
have shipped a DBSCAN that silently drops real neighbours. That was caught by
a check asserting an INDEX INVARIANT, not by anything in the signature.

HOW EVERY CHECK BELOW IS BUILT, AND WHY EACH RULE IS THERE
----------------------------------------------------------
1. **The oracle is computed independently on the host.** Not by rearranging
   the output of the same call. For a sort that means a host radix sort over
   the key bit patterns; for a matmul, a Float64 triple loop with the direct
   formula.
2. **The fixture is SCATTERED and HASHED** — splitmix64 over `(index, salt)`.
   A fixture that is uniform, sorted, or affine in the index verifies a total
   and nothing about placement. This repository has twice had a check pass a
   wrong reduction for exactly that reason.
3. **Sizes STRADDLE the block.** The `argsort` bug is invisible at 256 and at
   every size below it. Every size-parameterized check here runs 1, 2, 255,
   256, 257, 512, 513, 1023, 1024, 1025, 4096 and 100003 — the last a prime,
   so it is a multiple of no plausible block width.
4. **Selection primitives are compared on VALUES strictly** and on indices
   only up to ties, because equal keys have no defined winner.

`checks/vendor_main.mojo` runs this and prints the table. It exits non-zero if any
primitive WIRED into this tree tests WRONG.
"""

from std.math import sqrt
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import TileTensor
from layout.tile_layout import col_major, row_major

from nn.argsort import argsort
from linalg.matmul import matmul
from linalg.gemv import gemv_gpu
from layout.tensor_engine import DefaultEngine
from nn.topk import top_k
from max.gpu import block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.primitives.block import sum as block_sum
from max.gpu.primitives.block import max as block_max
from max.gpu.primitives.block import min as block_min
from max.gpu.primitives.block import prefix_sum as block_prefix_sum
from max.gpu.primitives.warp import (
    lane_group_min,
    shuffle_idx,
    shuffle_xor,
    vote,
)
from max.gpu.primitives.warp import prefix_sum as warp_prefix_sum
from max.gpu.primitives.warp import sum as warp_sum
from max.gpu.primitives.id import lane_id
from checks.kernel_matrix import TARGET_COLUMN, lib_lane_width_for
from std.utils import IndexList
from nn.toppminp_gpu import DoubleBuffer, run_radix_sort_pairs_gpu
from nn.argmaxmin_gpu import argmaxmin_gpu
from nn.gather_scatter import gather
from linalg.bmm import batched_matmul
from linalg.transpose import transpose
from nn.cumsum import cumsum
from core.gemm import gemm_nt, gemm_tn, gemm_tn_via_transpose
from core.gram_splitk import gram_splitk_applies
from checks.native_tests.kit import download_host_into, upload_host_into


# ---------------------------------------------------------------------------
# The verdict table.
# ---------------------------------------------------------------------------

comptime V_NOT_FOUND = "NOT FOUND"
comptime V_CPU_ONLY = "CPU ONLY"
comptime V_UNCHECKED = "GPU, UNCHECKED"
comptime V_CORRECT = "GPU, CORRECT"
comptime V_WRONG = "GPU, WRONG"


@fieldwise_init
struct Verdict(Copyable, Movable):
    """One row of the printed table."""

    var symbol: String
    var counterpart: String
    var verdict: String
    var sizes: String
    var wired: Bool
    """True when this tree CALLS it today. A WRONG verdict on a wired
    primitive is what makes `vendor_main` exit non-zero."""


# ---------------------------------------------------------------------------
# The fixture. splitmix64 over (index, salt).
# ---------------------------------------------------------------------------


def _mix(i: Int, salt: Int) -> UInt64:
    """splitmix64. Adjacent indices land nowhere near each other."""
    var z = UInt64(i + 1) * 0x9E3779B97F4A7C15 + UInt64(salt + 1) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _key_f32(i: Int, salt: Int) -> Float32:
    """A NON-NEGATIVE integer-valued Float32 below 2^23.

    Integer-valued because every such value is exact in Float32, so the host
    oracle and the device see literally the same number and a mismatch cannot
    be blamed on rounding. Non-negative because the host radix sort below
    orders IEEE bit patterns directly, which is monotone in the float value
    only for non-negative floats.
    """
    return Float32(Int(_mix(i, salt) % 8388593))


def _val_f32(i: Int, salt: Int) -> Float32:
    """A signed value in [-1, 1) for the arithmetic checks."""
    return Float32(Int(_mix(i, salt) % 2000001) - 1000000) * Float32(1.0e-6)


# ---------------------------------------------------------------------------
# The host oracle for every sort. LSD radix over IEEE bit patterns.
#
# Written here rather than reused from anywhere in this tree on purpose: an
# oracle that shares code with the thing it judges is not an oracle. Four
# 8-bit counting passes, O(n), and it is the ORDER that is being asserted, so
# it never touches the device call's output.
# ---------------------------------------------------------------------------


def _host_sort_u32(src: List[UInt32]) -> List[UInt32]:
    """Ascending sort of non-negative-float bit patterns."""
    var n = len(src)
    var a = List[UInt32](capacity=n)
    var b = List[UInt32](capacity=n)
    for i in range(n):
        a.append(src[i])
        b.append(UInt32(0))
    var shift = 0
    while shift < 32:
        var count = List[Int](capacity=257)
        for _ in range(257):
            count.append(0)
        for i in range(n):
            count[Int((a[i] >> UInt32(shift)) & 255) + 1] += 1
        for d in range(1, 257):
            count[d] += count[d - 1]
        for i in range(n):
            var d = Int((a[i] >> UInt32(shift)) & 255)
            b[count[d]] = a[i]
            count[d] += 1
        for i in range(n):
            a[i] = b[i]
        shift += 8
    return a^


# ---------------------------------------------------------------------------
# nn.argsort.argsort
# ---------------------------------------------------------------------------


def _sizes() -> List[Int]:
    """Every size-parameterized check runs all of these.

    256 is the block width that `argsort` turns out to sort within and never
    merge across, so 255/256/257 is the discriminating triple. 100003 is
    prime, so it is a multiple of no plausible block width (32, 64, 128, 256,
    512, 1024) and a kernel that assumes a whole number of tiles has nowhere
    to hide.
    """
    var s = List[Int]()
    s.append(1)
    s.append(2)
    s.append(255)
    s.append(256)
    s.append(257)
    s.append(512)
    s.append(513)
    s.append(1023)
    s.append(1024)
    s.append(1025)
    s.append(4096)
    s.append(100003)
    return s^


# ---------------------------------------------------------------------------
# The table.
# ---------------------------------------------------------------------------


def _pad(s: String, width: Int) raises -> String:
    var out = s + " "
    while out.byte_length() < width:
        out += " "
    return out^


# ---------------------------------------------------------------------------
# linalg.matmul.matmul  --  WIRED (core/gemm.mojo::gemm_nt, and gemm_tn
# through it, which is every distance matrix, every Gram matrix, PCA, OLS,
# truncated SVD and the k-NN fallback).
#
# The oracle is a Float64 triple loop with the direct formula. The device
# accumulates in Float32, so they are NOT expected to agree bit for bit; the
# tolerance is relative to the sum of |a*b| for that cell, which is the size
# of the cancellation the accumulation actually has to survive.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# linalg.gemv.gemv_gpu  --  WIRED (core/gemm.mojo::gemv_n, which is OLS's
# `w <- covA . Ab` and every matrix-vector product in glm/).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# nn.topk.top_k  --  WIRED (neighbors/.../knn_brute_force.mojo behind
# `use_vendor_topk`), and this is a SELECTION primitive, so ties are handled
# explicitly: the VALUES are compared strictly against the host oracle, and
# the INDICES only up to ties (the value AT the returned index must equal the
# value the oracle expected in that slot).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# BLOCK SCOPE: max.gpu.primitives.block.{sum, max, min, prefix_sum}
#
# WIRED, and more widely than anything else here: `core/row_norms.mojo`,
# `core/column_stats.mojo`, `dbscan/vertexdeg`, `dbscan/adjgraph`,
# `cluster/plus_plus`, `decomposition/pca` sign-flip, `neighbors/ball_cover`
# scan, `neighbors/select_radix`, `gbdt/gpu_util/partitions_reduce`.
#
# The fixture is HASHED per thread and INTEGER-VALUED, so the host oracle is
# exact and a float-association argument cannot explain away a mismatch. The
# grid is many blocks, and every block's expected answer is DIFFERENT, so a
# reduction that leaked across blocks or read the wrong block's slice shows
# up as a value no other block would have produced.
#
# Only thread 0 is checked for the reductions, because that is the contract
# the callers in this tree rely on (`pca.mojo:299` publishes through shared
# memory rather than assuming a broadcast). Whether the other threads also
# receive it is REPORTED but not asserted.
# ---------------------------------------------------------------------------


def block_check_kernel[
    TPB: Int
](
    out_sum: MutPointer[Float32, MutAnyOrigin],
    out_max: MutPointer[Float32, MutAnyOrigin],
    out_min: MutPointer[Float32, MutAnyOrigin],
    out_inc: MutPointer[Float32, MutAnyOrigin],
    out_exc: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
):
    var tid = Int(thread_idx.x)
    var base = Int(block_idx.x) * TPB
    var v = a.unsafe_load(base + tid)
    var s = block_sum[block_size=TPB](v)
    barrier()
    var mx = block_max[block_size=TPB](v)
    barrier()
    var mn = block_min[block_size=TPB](v)
    barrier()
    var inc = block_prefix_sum[block_size=TPB](v)
    barrier()
    var exc = block_prefix_sum[block_size=TPB, exclusive=True](v)
    out_sum.unsafe_store(base + tid, s)
    out_max.unsafe_store(base + tid, mx)
    out_min.unsafe_store(base + tid, mn)
    out_inc.unsafe_store(base + tid, inc)
    out_exc.unsafe_store(base + tid, exc)


def _block_one[
    TPB: Int
](ctx: DeviceContext, n_blocks: Int, mut broadcast_note: String) raises -> String:
    var n = n_blocks * TPB
    var a = ctx.enqueue_create_buffer[DType.float32](n)
    var o_sum = ctx.enqueue_create_buffer[DType.float32](n)
    var o_max = ctx.enqueue_create_buffer[DType.float32](n)
    var o_min = ctx.enqueue_create_buffer[DType.float32](n)
    var o_inc = ctx.enqueue_create_buffer[DType.float32](n)
    var o_exc = ctx.enqueue_create_buffer[DType.float32](n)
    var ha = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(n):
        # Integer-valued and under 1000, so a block of 1024 sums to under
        # 2^24 and every host expectation is EXACT in Float32.
        ha.unsafe_ptr().unsafe_store(i, Float32(Int(_mix(i, 51) % 1000)))
    upload_host_into[DType.float32](ctx, a, ha)
    ctx.synchronize()

    ctx.enqueue_function[block_check_kernel[TPB]](
        o_sum.unsafe_ptr(),
        o_max.unsafe_ptr(),
        o_min.unsafe_ptr(),
        o_inc.unsafe_ptr(),
        o_exc.unsafe_ptr(),
        a.unsafe_ptr(),
        grid_dim=(n_blocks, 1, 1),
        block_dim=(TPB, 1, 1),
    )
    ctx.synchronize()

    var hs = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hm = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hn = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hi = ctx.enqueue_create_host_buffer[DType.float32](n)
    var he = ctx.enqueue_create_host_buffer[DType.float32](n)
    download_host_into[DType.float32](ctx, o_sum, hs)
    download_host_into[DType.float32](ctx, o_max, hm)
    download_host_into[DType.float32](ctx, o_min, hn)
    download_host_into[DType.float32](ctx, o_inc, hi)
    download_host_into[DType.float32](ctx, o_exc, he)
    ctx.synchronize()

    var all_threads_get_reduction = True
    for b in range(n_blocks):
        var base = b * TPB
        var want_sum = Float32(0.0)
        var want_max = ha.unsafe_ptr().unsafe_load(base)
        var want_min = want_max
        var running = Float32(0.0)
        for t in range(TPB):
            var v = ha.unsafe_ptr().unsafe_load(base + t)
            want_sum += v
            if v > want_max:
                want_max = v
            if v < want_min:
                want_min = v
            var want_exc = running
            running += v
            var want_inc = running
            if hi.unsafe_ptr().unsafe_load(base + t) != want_inc:
                return (
                    "prefix_sum INCLUSIVE block "
                    + String(b)
                    + " lane "
                    + String(t)
                    + " device "
                    + String(hi.unsafe_ptr().unsafe_load(base + t))
                    + " host "
                    + String(want_inc)
                )
            if he.unsafe_ptr().unsafe_load(base + t) != want_exc:
                return (
                    "prefix_sum EXCLUSIVE block "
                    + String(b)
                    + " lane "
                    + String(t)
                    + " device "
                    + String(he.unsafe_ptr().unsafe_load(base + t))
                    + " host "
                    + String(want_exc)
                )
        if hs.unsafe_ptr().unsafe_load(base) != want_sum:
            return (
                "block.sum block "
                + String(b)
                + " device "
                + String(hs.unsafe_ptr().unsafe_load(base))
                + " host "
                + String(want_sum)
            )
        if hm.unsafe_ptr().unsafe_load(base) != want_max:
            return "block.max block " + String(b) + " wrong"
        if hn.unsafe_ptr().unsafe_load(base) != want_min:
            return "block.min block " + String(b) + " wrong"
        for t in range(TPB):
            if (
                hs.unsafe_ptr().unsafe_load(base + t) != want_sum
                or hm.unsafe_ptr().unsafe_load(base + t) != want_max
                or hn.unsafe_ptr().unsafe_load(base + t) != want_min
            ):
                all_threads_get_reduction = False
    if all_threads_get_reduction:
        broadcast_note = String("broadcast to all threads")
    else:
        broadcast_note = String("thread 0 only")
    return String("")


def check_block(mut rows: List[Verdict]) raises:
    var ctx = DeviceContext()
    var bad = String("")
    var note = String("")
    var n1 = String("")
    var e = _block_one[32](ctx, 37, n1)
    print("  block TPB = 32 ", "ok (" + n1 + ")" if e == "" else "FAIL: " + e)
    if e != "" and bad == "":
        bad = "TPB=32"
    note = n1
    e = _block_one[64](ctx, 37, n1)
    print("  block TPB = 64 ", "ok (" + n1 + ")" if e == "" else "FAIL: " + e)
    if e != "" and bad == "":
        bad = "TPB=64"
    e = _block_one[128](ctx, 37, n1)
    print("  block TPB = 128", "ok (" + n1 + ")" if e == "" else "FAIL: " + e)
    if e != "" and bad == "":
        bad = "TPB=128"
    e = _block_one[256](ctx, 37, n1)
    print("  block TPB = 256", "ok (" + n1 + ")" if e == "" else "FAIL: " + e)
    if e != "" and bad == "":
        bad = "TPB=256"
    e = _block_one[512](ctx, 37, n1)
    print("  block TPB = 512", "ok (" + n1 + ")" if e == "" else "FAIL: " + e)
    if e != "" and bad == "":
        bad = "TPB=512"
    e = _block_one[1024](ctx, 37, n1)
    print("  block TPB = 1024", "ok (" + n1 + ")" if e == "" else "FAIL: " + e)
    if e != "" and bad == "":
        bad = "TPB=1024"

    var verdict = V_CORRECT if bad == "" else V_WRONG
    var ev = (
        "TPB 32/64/128/256/512/1024, 37 blocks; " + note
    ) if bad == "" else ("smallest failing " + bad)
    rows.append(
        Verdict("max.gpu.primitives.block.sum", "cub::BlockReduce", verdict, ev, True)
    )
    rows.append(
        Verdict("max.gpu.primitives.block.max", "cub::BlockReduce(Max)", verdict, ev, True)
    )
    rows.append(
        Verdict("max.gpu.primitives.block.min", "cub::BlockReduce(Min)", verdict, ev, True)
    )
    rows.append(
        Verdict(
            "max.gpu.primitives.block.prefix_sum",
            "cub::BlockScan",
            verdict,
            ev + " (inclusive AND exclusive)",
            True,
        )
    )


# ---------------------------------------------------------------------------
# WARP SCOPE: std.gpu.primitives.warp.{shuffle_xor, shuffle_idx, sum,
# prefix_sum, vote} and std.gpu.primitives.id.lane_id
#
# WIRED: `shuffle_xor` in `unfused_distance_nn` and the fused SIMT kernel;
# `shuffle_idx` / `vote` / `warp.sum` / `lane_id` throughout
# `neighbors/.../ball_cover/registers.mojo`; `prefix_sum` and `warp.max` in
# `select_warpsort`.
#
# LANE WIDTH IS READ FROM THE KERNEL MATRIX, not typed. It is 32 on Apple and
# NVIDIA and 64 on AMD, and a check that hardcoded 32 would silently test the
# wrong half of a wavefront on the vendor where it matters most.
#
# `lane_group_min` is here even though nothing calls it, because
# `archive/reference/VENDOR_LIBRARIES.md` C7 recommends it as the answer to CUDA's `width`
# modulo on the strength of its documentation alone. It is checked at
# `num_lanes = 8` against a host oracle that folds within 8-lane groups.
# ---------------------------------------------------------------------------


comptime WARP_LANES = lib_lane_width_for[TARGET_COLUMN]()


# ---------------------------------------------------------------------------
# nn.toppminp_gpu.run_radix_sort_pairs_gpu
#
# NOT WIRED. `archive/reference/VENDOR_LIBRARIES.md` Appendix A2 item 1 recommends it as the
# missing `cub::DeviceRadixSort::SortPairs` AND, because it sorts each batch
# row independently, as `cub::DeviceSegmentedRadixSort` for equal-length
# segments. That recommendation rests on the signature and on one import
# probe. `nn.argsort` had exactly that much evidence behind it.
#
# The three caveats the appendix says probing is the only way to settle are
# settled here: `DoubleBuffer` takes (current, alternate, size) and its
# pointers must be rebound to `MutUntrackedOrigin`; `skip_sort` is a per-batch
# device `Bool` array and all-False means "sort every row"; and which half
# holds the result is REPORTED below rather than guessed.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# nn.argmaxmin_gpu.argmaxmin_gpu
#
# NOT WIRED. Appendix A2 item 6 recommends it for the k-means assignment step
# and for the reduction half of `unfused_distance_nn`, on the signature and
# one import probe.
#
# Ties: an argmin has no defined winner among equal values, so the VALUE at
# the returned index is compared against the host minimum, and the index
# itself only where the row minimum is unique. The hashed fixture makes the
# minimum unique in practice and the check says so when it is not.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# nn.gather_scatter.gather
#
# NOT WIRED. Listed under "Also confirmed available, unused so far".
# The index fixture is a HASHED permutation-free scatter: every row of the
# output pulls a different, non-monotone source row, so a gather that dropped
# the indices and copied straight through would be caught. That is exactly
# the failure a sorted or affine index fixture cannot see.
# ---------------------------------------------------------------------------


def _gather_one(ctx: DeviceContext, n_rows: Int, n_cols: Int, n_out: Int) raises -> String:
    var inp = ctx.enqueue_create_buffer[DType.float32](n_rows * n_cols)
    var idx = ctx.enqueue_create_buffer[DType.int32](n_out)
    var out = ctx.enqueue_create_buffer[DType.float32](n_out * n_cols)
    var hin = ctx.enqueue_create_host_buffer[DType.float32](n_rows * n_cols)
    var hidx = ctx.enqueue_create_host_buffer[DType.int32](n_out)
    var hout = ctx.enqueue_create_host_buffer[DType.float32](n_out * n_cols)
    ctx.synchronize()
    for i in range(n_rows * n_cols):
        hin.unsafe_ptr().unsafe_store(i, _key_f32(i, 91))
    for i in range(n_out):
        hidx.unsafe_ptr().unsafe_store(i, Int32(Int(_mix(i, 92) % UInt64(n_rows))))
    for i in range(n_out * n_cols):
        hout.unsafe_ptr().unsafe_store(i, Float32(-987654.0))
    upload_host_into[DType.float32](ctx, inp, hin)
    upload_host_into[DType.int32](ctx, idx, hidx)
    upload_host_into[DType.float32](ctx, out, hout)
    ctx.synchronize()

    gather[axis=0, target="gpu"](
        TileTensor(out, row_major(n_out, n_cols)),
        TileTensor(inp, row_major(n_rows, n_cols)),
        TileTensor(idx, row_major(n_out)),
        context=ctx,
    )
    ctx.synchronize()
    download_host_into[DType.float32](ctx, out, hout)
    ctx.synchronize()

    for i in range(n_out):
        var src = Int(hidx.unsafe_ptr().unsafe_load(i))
        for c in range(n_cols):
            var want = hin.unsafe_ptr().unsafe_load(src * n_cols + c)
            var got = hout.unsafe_ptr().unsafe_load(i * n_cols + c)
            if got != want:
                return (
                    "out ("
                    + String(i)
                    + ", "
                    + String(c)
                    + ") from source row "
                    + String(src)
                    + ": device "
                    + String(got)
                    + " host "
                    + String(want)
                )
    return String("")


def check_gather(mut rows: List[Verdict]) raises:
    var ctx = DeviceContext()
    var sizes = _sizes()
    var bad = String("")
    var raised = String("")
    for si in range(len(sizes)):
        var n = sizes[si]
        var err = String("")
        try:
            err = _gather_one(ctx, n, 3, n)
        except e:
            err = String(e)
            if raised == "":
                raised = String(e)
        if err != "":
            print("  gather rows =", n, "FAIL:", err)
            if bad == "":
                bad = "rows=" + String(n)
        else:
            print("  gather rows =", n, "ok")
    if bad == "":
        rows.append(
            Verdict(
                "nn.gather_scatter.gather[axis=0]",
                "thrust::gather",
                V_CORRECT,
                "rows in {1..100003} x 3 cols, hashed non-monotone indices",
                False,
            )
        )
    elif raised != "":
        rows.append(
            Verdict(
                "nn.gather_scatter.gather[axis=0]",
                "thrust::gather",
                V_UNCHECKED,
                "RAISES at " + bad + ": " + raised,
                False,
            )
        )
    else:
        rows.append(
            Verdict(
                "nn.gather_scatter.gather[axis=0]",
                "thrust::gather",
                V_WRONG,
                "smallest failing " + bad,
                False,
            )
        )


# ---------------------------------------------------------------------------
# linalg.bmm.batched_matmul
#
# NOT WIRED. Appendix A2 item 9 lists it, unprobed, as the per-segment Gram
# or distance block in one launch. Same Float64 host oracle as `matmul`, and
# the batch dimension is checked per BATCH, so a kernel that computed batch 0
# and broadcast it would fail.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# linalg.transpose.transpose  --  the standing reminder.
#
# `archive/reference/VENDOR_LIBRARIES.md` records it as signalling at runtime inside
# `_copy_with_strides` when handed device pointers, which is why
# `core/column_stats.mojo::transpose_kernel` exists. Re-run here so the claim
# has a date and an error string rather than a memory.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# nn.cumsum.cumsum  --  CPU ONLY, re-confirmed cheaply and moved on from.
#
# `archive/reference/VENDOR_LIBRARIES.md` already records this and the record is right: the
# signature carries neither `ctx: DeviceContext` nor `target`. What this adds
# is a RUN, so the row is not resting on a signature read: the call is made
# on HOST buffers and its answer is compared against a host oracle, and a
# sibling compile probe (see the lane file) shows that adding `target="gpu"`
# does not typecheck. Inclusive and exclusive are both exercised, because the
# file's own correction C5 records an earlier probe that concluded "inclusive
# only" from a guessed symbol name.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# COMPILE-ESTABLISHED ROWS.
#
# A program cannot contain a call that fails to compile, so these rows come
# from SIBLING probe files, one per row, each built with `mojo build` and each
# reproduced verbatim in
# `bench/results/LANE_vendor-correctness_2026-08-19.md`. They are here so the
# printed table is the whole picture rather than the runnable half of it.
#
# NOT FOUND and CPU ONLY are settled by the compiler and need no run.
# GPU, UNCHECKED rows say what stopped the run; none of them is a
# recommendation.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# THE n = 1 COLUMN IS REACHABLE FROM THIS TREE. This is the check that turns
# the `matmul` n=1 row from a curiosity into a WIRED defect.
#
# `core/gemm.mojo::gemm_nt` is `matmul[transpose_b=True]` and nothing else,
# and its `n` is a user-facing count at four call sites:
#
#   cluster/gbdt/cluster/detail/min_cluster_distance_compute.mojo:197
#       gemm_nt(ctx, dist_buf, x_tile, c_tile, ns, nc, n_features)
#       -- `nc` is n_clusters. k-means with ONE cluster is n = 1.
#   glm/gbdt/linalg/detail/lstsq.mojo:120 and :193
#       gemm_tn(ctx, cov_a, a, ..., n_cols, n_cols, n_rows)
#       -- `n_cols` is the feature count. Simple linear regression on ONE
#          predictor is n = 1, and `gemm_tn` finishes by calling `gemm_nt`.
#   decomposition/gbdt/linalg/detail/pca.mojo:202, tsvd.mojo:89
#       same shape, `n_cols` again.
#   neighbors/gbdt/neighbors/detail/knn_brute_force.mojo:154
#       gemm_nt(ctx, dist_tile, q_tile, index, rows, n_index, n_features)
#       -- `n_index` is the number of index points.
#
# This calls the REAL `core/gemm.mojo::gemm_nt`, not a local copy, so a fix
# there flips this row without an edit here.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# linalg.matmul with a COL-MAJOR first operand  --  the covariance dead end.
#
# The T-N Gram shape (`raft::stats::cov` cov.cuh:65-66, `lstsqEig`'s first
# gemm lstsq.cuh:293-305) begs for a zero-copy transposed view:
# `matmul[transpose_a=True]` is refused at compile time, but a
# `col_major(m, k)` TileTensor over X's buffer IS `X^T`, and cuBLAS's OP_T is
# exactly this trick -- same bytes, swapped strides.
#
# PROBED 2026-08-19 AND UNWIREABLE: the dispatcher honors the view's strides
# on SOME arms and silently IGNORES them on others, writing plausible wrong
# numbers. Correct at 32x32x100003, 33x17x255, 8x8x8, 129x127x513; WRONG at
# EVERY CELL across the whole m=n in {4..64} x k in {64..2048} band
# (8x8x512, 32x32x2048, 64x64x64, ...), at n=1 with m>1 (8x1x33), and in
# `gemv_gpu` at every output. The ok/wrong boundary zigzags with shape and
# matches no predicate worth trusting across a toolchain update, so
# `core/gemm.mojo::gemm_tn` stays on the materialized-transpose route and
# these rows exist so nobody re-tries the view without re-running this.
# ---------------------------------------------------------------------------

