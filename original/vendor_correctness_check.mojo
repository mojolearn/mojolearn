# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Every vendor primitive this repo uses or recommends, run against an
independent host oracle.

THE RULE THIS FILE EXISTS TO ENFORCE
------------------------------------
**A signature check proves the GPU is REACHABLE. Only a run against an
independent oracle proves the answer is RIGHT.**

`VENDOR_LIBRARIES.md` listed `nn.argsort.argsort` as GPU because it takes a
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

`vendor_main.mojo` runs this and prints the table. It exits non-zero if any
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
from nn.topk import top_k
from std.gpu import block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.primitives.block import sum as block_sum
from max.gpu.primitives.block import max as block_max
from max.gpu.primitives.block import min as block_min
from max.gpu.primitives.block import prefix_sum as block_prefix_sum
from std.gpu.primitives.warp import (
    lane_group_min,
    shuffle_idx,
    shuffle_xor,
    vote,
)
from std.gpu.primitives.warp import prefix_sum as warp_prefix_sum
from std.gpu.primitives.warp import sum as warp_sum
from std.gpu.primitives.id import lane_id
from original.kernel_matrix import TARGET_COLUMN, lib_lane_width_for
from std.utils import IndexList
from nn.toppminp_gpu import DoubleBuffer, run_radix_sort_pairs_gpu
from nn.argmaxmin_gpu import argmaxmin_gpu
from nn.gather_scatter import gather
from linalg.bmm import batched_matmul
from linalg.transpose import transpose
from nn.cumsum import cumsum
from core.gemm import gemm_nt, gemm_tn, gemm_tn_via_transpose
from core.gram_splitk import gram_splitk_applies


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


def _argsort_one(ctx: DeviceContext, n: Int, salt: Int) raises -> String:
    """One size. Returns "" on success, or the failure text.

    Three assertions, and none of them subsumes the others:

    * the output is a PERMUTATION of `0..n-1` (a sort that duplicates an
      index is broken even if the values it lands on happen to be ordered);
    * `keys[out[j]]` is non-decreasing in `j` (the ORDER, which is the thing
      a caller buys);
    * that same sequence equals the host-sorted key multiset element by
      element (catches a monotone output that lost or invented a value).
    """
    var keys = ctx.enqueue_create_buffer[DType.float32](n)
    var out = ctx.enqueue_create_buffer[DType.int64](n)
    var hk = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(n):
        hk.unsafe_ptr().unsafe_store(i, _key_f32(i, salt))
    ctx.enqueue_copy(dst_buf=keys, src_ptr=hk.unsafe_ptr())
    ctx.synchronize()

    argsort[target="gpu"](
        TileTensor(out, row_major(n)), TileTensor(keys, row_major(n)), ctx
    )
    ctx.synchronize()

    var ho = ctx.enqueue_create_host_buffer[DType.int64](n)
    ctx.enqueue_copy(dst_ptr=ho.unsafe_ptr(), src_buf=out)
    ctx.synchronize()

    # 1. permutation
    var seen = List[Bool](capacity=n)
    for _ in range(n):
        seen.append(False)
    for j in range(n):
        var p = Int(ho.unsafe_ptr().unsafe_load(j))
        if p < 0 or p >= n:
            return "index " + String(p) + " out of range at output " + String(j)
        if seen[p]:
            return "index " + String(p) + " appears twice, at output " + String(j)
        seen[p] = True

    # 2. monotone
    for j in range(1, n):
        var a = hk.unsafe_ptr().unsafe_load(
            Int(ho.unsafe_ptr().unsafe_load(j - 1))
        )
        var b = hk.unsafe_ptr().unsafe_load(Int(ho.unsafe_ptr().unsafe_load(j)))
        if b < a:
            return "NOT MONOTONE, first inversion at output index " + String(j)

    # 3. multiset, against the independent host radix sort
    var bits = List[UInt32](capacity=n)
    for i in range(n):
        bits.append(bitcast[DType.uint32](hk.unsafe_ptr().unsafe_load(i)))
    var sorted_bits = _host_sort_u32(bits)
    for j in range(n):
        var got = bitcast[DType.uint32](
            hk.unsafe_ptr().unsafe_load(Int(ho.unsafe_ptr().unsafe_load(j)))
        )
        if got != sorted_bits[j]:
            return (
                "multiset mismatch at output index "
                + String(j)
                + " (device key "
                + String(bitcast[DType.float32](got))
                + ", host "
                + String(bitcast[DType.float32](sorted_bits[j]))
                + ")"
            )
    return String("")


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


def check_argsort(mut rows: List[Verdict]) raises:
    var ctx = DeviceContext()
    var first_bad = -1
    var detail = String("")
    var sizes = _sizes()
    for si in range(len(sizes)):
        var n = sizes[si]
        var err = _argsort_one(ctx, n, 11)
        if err != "":
            print("  argsort n =", n, "FAIL:", err)
            if first_bad < 0:
                first_bad = n
                detail = err
        else:
            print("  argsort n =", n, "ok")
    if first_bad < 0:
        rows.append(
            Verdict(
                "nn.argsort.argsort",
                "cub::DeviceRadixSort::SortKeys",
                V_CORRECT,
                "1..100003",
                False,
            )
        )
    else:
        rows.append(
            Verdict(
                "nn.argsort.argsort",
                "cub::DeviceRadixSort::SortKeys",
                V_WRONG,
                "correct <= 256; smallest failing n = " + String(first_bad),
                False,
            )
        )
        print("  argsort SMALLEST FAILING SIZE:", first_bad, "-", detail)


# ---------------------------------------------------------------------------
# The table.
# ---------------------------------------------------------------------------


def _pad(s: String, width: Int) raises -> String:
    var out = s + " "
    while out.byte_length() < width:
        out += " "
    return out^


def run_vendor_correctness() raises -> Int:
    """Runs every check and prints the table. Returns the number of WIRED
    primitives that tested WRONG."""
    var rows = List[Verdict]()

    print("=== running vendor correctness checks ===")
    check_argsort(rows)
    check_matmul(rows)
    check_wired_gemm_at_n1(rows)
    check_matmul_colmajor(rows)
    check_gemv(rows)
    check_topk(rows)
    check_block(rows)
    check_warp(rows)
    check_radix_sort_pairs(rows)
    check_argmaxmin(rows)
    check_gather(rows)
    check_bmm(rows)
    check_cumsum(rows)
    check_compile_established(rows)

    # `linalg.transpose` is NOT called from here. It does not raise, it
    # ABORTS the process, so a probe of it inside this binary would take the
    # table with it. Run `vendor_main --transpose` to see the abort. The row
    # below carries the evidence that run produces.
    rows.append(
        Verdict(
            "linalg.transpose.transpose",
            "raft::linalg::transpose",
            V_WRONG,
            (
                "257x129 device buffers: ABORTS (not a catchable raise) in"
                " linalg::transpose::_copy_with_strides rank=2 dtype=f32 --"
                " 'enqueue_cpu_range is only supported on CPU"
                " DeviceContexts'. Run `vendor_main --transpose` to"
                " reproduce."
            ),
            False,
        )
    )

    # Ordered so the dangerous rows are read first: WIRED and WRONG at the
    # top, then WRONG anywhere, then everything else in the order it ran.
    var order = List[Int]()
    for i in range(len(rows)):
        if rows[i].wired and rows[i].verdict == V_WRONG:
            order.append(i)
    for i in range(len(rows)):
        if not rows[i].wired and rows[i].verdict == V_WRONG:
            order.append(i)
    for i in range(len(rows)):
        if rows[i].verdict != V_WRONG:
            order.append(i)

    print("")
    print("=== VENDOR CORRECTNESS TABLE ===")
    print(
        _pad(String("symbol"), 56)
        + _pad(String("counterpart"), 36)
        + _pad(String("verdict"), 16)
        + _pad(String("wired"), 7)
        + "sizes / evidence"
    )
    var wired_wrong = 0
    for oi in range(len(order)):
        var r = rows[order[oi]].copy()
        print(
            _pad(r.symbol, 56)
            + _pad(r.counterpart, 36)
            + _pad(r.verdict, 16)
            + _pad(String("YES") if r.wired else String("no"), 7)
            + r.sizes
        )
        if r.wired and r.verdict == V_WRONG:
            wired_wrong += 1
    print("")
    if wired_wrong != 0:
        print(
            "FAIL:",
            wired_wrong,
            "primitive(s) WIRED into this tree tested WRONG.",
        )
    else:
        print("no WIRED primitive tested WRONG")
    return wired_wrong


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


def _matmul_one[
    TB: Bool = True
](ctx: DeviceContext, m: Int, n: Int, k: Int) raises -> String:
    """`z[m x n] = x[m x k] . y[n x k]^T`, the one shape this repo asks for.

    `TB=False` runs the same product through the N-N form (`y` laid out
    `k x n`) instead, which is how the n=1 failure gets pinned to
    `transpose_b=True` rather than to n=1 itself.
    """
    var x = ctx.enqueue_create_buffer[DType.float32](m * k)
    var y = ctx.enqueue_create_buffer[DType.float32](n * k)
    var z = ctx.enqueue_create_buffer[DType.float32](m * n)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](m * k)
    var hy = ctx.enqueue_create_host_buffer[DType.float32](n * k)
    var hz = ctx.enqueue_create_host_buffer[DType.float32](m * n)
    ctx.synchronize()
    for i in range(m * k):
        hx.unsafe_ptr().unsafe_store(i, _val_f32(i, 21))
    for i in range(n * k):
        hy.unsafe_ptr().unsafe_store(i, _val_f32(i, 22))
    # A poison pattern in the output. `matmul` at n=1 returns zeros for some
    # outputs and raises nothing, so an output buffer that STARTS at zero
    # cannot tell "written zero" from "never written".
    for i in range(m * n):
        hz.unsafe_ptr().unsafe_store(i, Float32(-987654.0))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=y, src_ptr=hy.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=z, src_ptr=hz.unsafe_ptr())
    ctx.synchronize()

    comptime if TB:
        matmul[transpose_b=True, target="gpu"](
            TileTensor(z, row_major(m, n)),
            TileTensor(x, row_major(m, k)),
            TileTensor(y, row_major(n, k)),
            ctx,
        )
    else:
        matmul[transpose_b=False, target="gpu"](
            TileTensor(z, row_major(m, n)),
            TileTensor(x, row_major(m, k)),
            TileTensor(y, row_major(k, n)),
            ctx,
        )
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hz.unsafe_ptr(), src_buf=z)
    ctx.synchronize()

    for i in range(m):
        for j in range(n):
            var acc = Float64(0.0)
            var mag = Float64(0.0)
            for t in range(k):
                var a = Float64(hx.unsafe_ptr().unsafe_load(i * k + t))
                var b = Float64(0.0)

                comptime if TB:
                    b = Float64(hy.unsafe_ptr().unsafe_load(j * k + t))
                else:
                    b = Float64(hy.unsafe_ptr().unsafe_load(t * n + j))
                acc += a * b
                mag += abs(a * b)
            var got = Float64(hz.unsafe_ptr().unsafe_load(i * n + j))
            if got == Float64(-987654.0):
                return (
                    "cell ("
                    + String(i)
                    + ", "
                    + String(j)
                    + ") was NEVER WRITTEN (poison survived)"
                )
            var tol = mag * 2.0e-6 + 1.0e-6
            if abs(got - acc) > tol:
                return (
                    "cell ("
                    + String(i)
                    + ", "
                    + String(j)
                    + ") device "
                    + String(got)
                    + " host "
                    + String(acc)
                    + " tol "
                    + String(tol)
                )
    return String("")


def check_matmul(mut rows: List[Verdict]) raises:
    var ctx = DeviceContext()
    # m, n, k triples.
    var shapes: List[Int] = [
        1, 1, 1,
        1, 8, 3,
        255, 255, 33,
        256, 256, 64,
        257, 257, 65,
        513, 129, 127,
        1024, 512, 32,
        100003, 4, 8,
    ]
    var bad = String("")
    var s = 0
    while s < len(shapes):
        var m = shapes[s]
        var n = shapes[s + 1]
        var k = shapes[s + 2]
        var err = _matmul_one(ctx, m, n, k)
        if err != "":
            print("  matmul", m, "x", n, "x", k, "FAIL:", err)
            if bad == "":
                bad = String(m) + "x" + String(n) + "x" + String(k)
        else:
            print("  matmul", m, "x", n, "x", k, "ok")
        s += 3
    if bad == "":
        rows.append(
            Verdict(
                "linalg.matmul.matmul[transpose_b]",
                "cublasGemmEx (N-T)",
                V_CORRECT,
                "m,n,k up to 100003x4x8; 1x1x1, 255/256/257, 513x129x127",
                True,
            )
        )
    else:
        rows.append(
            Verdict(
                "linalg.matmul.matmul[transpose_b]",
                "cublasGemmEx (N-T)",
                V_WRONG,
                "smallest failing shape " + bad,
                True,
            )
        )

    # THE n = 1 COLUMN. `VENDOR_LIBRARIES.md` records this as "returns zeros
    # for some outputs, no error", and nothing had re-run it. Two things are
    # wrong with that sentence. It does not return zeros: it does not WRITE,
    # which a caller reusing a buffer sees as stale data rather than as zero.
    # And it is not a property of n = 1: the N-N form at the same n = 1 is
    # CORRECT, so the failure belongs to `transpose_b=True`.
    var err_nn = _matmul_one[False](ctx, 64, 1, 32)
    if err_nn == "":
        print("  matmul 64 x 1 x 32 transpose_b=FALSE ok")
    else:
        print("  matmul 64 x 1 x 32 transpose_b=FALSE FAIL:", err_nn)
    var err1 = _matmul_one[True](ctx, 64, 1, 32)
    if err1 == "":
        print("  matmul 64 x 1 x 32 transpose_b=TRUE ok")
        rows.append(
            Verdict(
                "linalg.matmul.matmul[transpose_b] at n = 1",
                "cublasGemmEx (degenerate)",
                V_CORRECT,
                "64x1x32",
                False,
            )
        )
    else:
        print("  matmul 64 x 1 x 32 transpose_b=TRUE FAIL:", err1)
        rows.append(
            Verdict(
                "linalg.matmul.matmul[transpose_b] at n = 1",
                "cublasGemmEx (degenerate)",
                V_WRONG,
                (
                    "64x1x32: "
                    + err1
                    + " -- the SAME product with transpose_b=False is "
                    + ("CORRECT" if err_nn == "" else "also wrong")
                ),
                False,
            )
        )


# ---------------------------------------------------------------------------
# linalg.gemv.gemv_gpu  --  WIRED (core/gemm.mojo::gemv_n, which is OLS's
# `w <- covA . Ab` and every matrix-vector product in glm/).
# ---------------------------------------------------------------------------


def _gemv_one(ctx: DeviceContext, m: Int, k: Int) raises -> String:
    var x = ctx.enqueue_create_buffer[DType.float32](m * k)
    var y = ctx.enqueue_create_buffer[DType.float32](k)
    var z = ctx.enqueue_create_buffer[DType.float32](m)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](m * k)
    var hy = ctx.enqueue_create_host_buffer[DType.float32](k)
    var hz = ctx.enqueue_create_host_buffer[DType.float32](m)
    ctx.synchronize()
    for i in range(m * k):
        hx.unsafe_ptr().unsafe_store(i, _val_f32(i, 31))
    for i in range(k):
        hy.unsafe_ptr().unsafe_store(i, _val_f32(i, 32))
    for i in range(m):
        hz.unsafe_ptr().unsafe_store(i, Float32(-987654.0))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=y, src_ptr=hy.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=z, src_ptr=hz.unsafe_ptr())
    ctx.synchronize()

    gemv_gpu[transpose_b=False](
        TileTensor(z, row_major(m, Int(1))),
        TileTensor(x, row_major(m, k)),
        TileTensor(y, row_major(k, Int(1))),
        ctx,
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hz.unsafe_ptr(), src_buf=z)
    ctx.synchronize()

    for i in range(m):
        var acc = Float64(0.0)
        var mag = Float64(0.0)
        for t in range(k):
            var a = Float64(hx.unsafe_ptr().unsafe_load(i * k + t))
            var b = Float64(hy.unsafe_ptr().unsafe_load(t))
            acc += a * b
            mag += abs(a * b)
        var got = Float64(hz.unsafe_ptr().unsafe_load(i))
        if got == Float64(-987654.0):
            return "row " + String(i) + " was NEVER WRITTEN (poison survived)"
        var tol = mag * 2.0e-6 + 1.0e-6
        if abs(got - acc) > tol:
            return (
                "row "
                + String(i)
                + " device "
                + String(got)
                + " host "
                + String(acc)
                + " tol "
                + String(tol)
            )
    return String("")


def check_gemv(mut rows: List[Verdict]) raises:
    var ctx = DeviceContext()
    var sizes = _sizes()
    var bad = String("")
    for si in range(len(sizes)):
        var m = sizes[si]
        var k = 3 + (m % 7)
        var err = _gemv_one(ctx, m, k)
        if err != "":
            print("  gemv_gpu m =", m, "k =", k, "FAIL:", err)
            if bad == "":
                bad = "m=" + String(m) + " k=" + String(k)
        else:
            print("  gemv_gpu m =", m, "k =", k, "ok")
    # A wide one too: the sweep above keeps k small, and k is the reduction
    # axis, which is where a split-K path would live.
    var errw = _gemv_one(ctx, 1000, 1025)
    if errw != "":
        print("  gemv_gpu m = 1000 k = 1025 FAIL:", errw)
        if bad == "":
            bad = "m=1000 k=1025"
    else:
        print("  gemv_gpu m = 1000 k = 1025 ok")
    if bad == "":
        rows.append(
            Verdict(
                "linalg.gemv.gemv_gpu[transpose_b=False]",
                "raft::linalg::gemv / cublasSgemv",
                V_CORRECT,
                "m in {1..100003}, k in {3..9}; plus m=1000 k=1025",
                True,
            )
        )
    else:
        rows.append(
            Verdict(
                "linalg.gemv.gemv_gpu[transpose_b=False]",
                "raft::linalg::gemv / cublasSgemv",
                V_WRONG,
                "smallest failing " + bad,
                True,
            )
        )


# ---------------------------------------------------------------------------
# nn.topk.top_k  --  WIRED (neighbors/.../knn_brute_force.mojo behind
# `use_vendor_topk`), and this is a SELECTION primitive, so ties are handled
# explicitly: the VALUES are compared strictly against the host oracle, and
# the INDICES only up to ties (the value AT the returned index must equal the
# value the oracle expected in that slot).
# ---------------------------------------------------------------------------


def _topk_one(ctx: DeviceContext, batch: Int, n: Int, k: Int) raises -> String:
    var vals = ctx.enqueue_create_buffer[DType.float32](batch * n)
    var ov = ctx.enqueue_create_buffer[DType.float32](batch * k)
    var oi = ctx.enqueue_create_buffer[DType.int32](batch * k)
    var hv = ctx.enqueue_create_host_buffer[DType.float32](batch * n)
    var hov = ctx.enqueue_create_host_buffer[DType.float32](batch * k)
    var hoi = ctx.enqueue_create_host_buffer[DType.int32](batch * k)
    ctx.synchronize()
    for i in range(batch * n):
        hv.unsafe_ptr().unsafe_store(i, _key_f32(i, 41))
    ctx.enqueue_copy(dst_buf=vals, src_ptr=hv.unsafe_ptr())
    ctx.synchronize()

    top_k[largest=False, target="gpu"](
        TileTensor(vals, row_major(batch, n)),
        k,
        1,
        TileTensor(ov, row_major(batch, k)),
        TileTensor(oi, row_major(batch, k)),
        False,
        ctx,
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hov.unsafe_ptr(), src_buf=ov)
    ctx.enqueue_copy(dst_ptr=hoi.unsafe_ptr(), src_buf=oi)
    ctx.synchronize()

    for b in range(batch):
        # Independent host oracle: full ascending sort of the row, take k.
        var bits = List[UInt32](capacity=n)
        for i in range(n):
            bits.append(
                bitcast[DType.uint32](hv.unsafe_ptr().unsafe_load(b * n + i))
            )
        var srt = _host_sort_u32(bits)
        for slot in range(k):
            var want = bitcast[DType.float32](srt[slot])
            var got = hov.unsafe_ptr().unsafe_load(b * k + slot)
            if got != want:
                return (
                    "row "
                    + String(b)
                    + " slot "
                    + String(slot)
                    + " VALUE device "
                    + String(got)
                    + " host "
                    + String(want)
                )
            var idx = Int(hoi.unsafe_ptr().unsafe_load(b * k + slot))
            if idx < 0 or idx >= n:
                return (
                    "row "
                    + String(b)
                    + " slot "
                    + String(slot)
                    + " index "
                    + String(idx)
                    + " out of range"
                )
            # Indices only up to ties: the value at the index must be the
            # value the oracle put in this slot. Equal keys have no defined
            # winner, so demanding a particular index would fail a correct
            # implementation.
            if hv.unsafe_ptr().unsafe_load(b * n + idx) != want:
                return (
                    "row "
                    + String(b)
                    + " slot "
                    + String(slot)
                    + " index "
                    + String(idx)
                    + " points at "
                    + String(hv.unsafe_ptr().unsafe_load(b * n + idx))
                    + ", expected "
                    + String(want)
                )
    return String("")


def check_topk(mut rows: List[Verdict]) raises:
    var ctx = DeviceContext()
    var sizes = _sizes()
    var bad = String("")
    for si in range(len(sizes)):
        var n = sizes[si]
        var k = 1 if n < 4 else (32 if n > 64 else 4)
        var err = _topk_one(ctx, 3, n, k)
        if err != "":
            print("  top_k batch=3 n =", n, "k =", k, "FAIL:", err)
            if bad == "":
                bad = "n=" + String(n) + " k=" + String(k)
        else:
            print("  top_k batch=3 n =", n, "k =", k, "ok")
    if bad == "":
        rows.append(
            Verdict(
                "nn.topk.top_k[largest=False]",
                "cub::DeviceSelect / raft select_k",
                V_CORRECT,
                "batch=3, n in {1..100003}, k in {1,4,32}",
                True,
            )
        )
    else:
        rows.append(
            Verdict(
                "nn.topk.top_k[largest=False]",
                "cub::DeviceSelect / raft select_k",
                V_WRONG,
                "smallest failing " + bad,
                True,
            )
        )


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
    ctx.enqueue_copy(dst_buf=a, src_ptr=ha.unsafe_ptr())
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
    ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=o_sum)
    ctx.enqueue_copy(dst_ptr=hm.unsafe_ptr(), src_buf=o_max)
    ctx.enqueue_copy(dst_ptr=hn.unsafe_ptr(), src_buf=o_min)
    ctx.enqueue_copy(dst_ptr=hi.unsafe_ptr(), src_buf=o_inc)
    ctx.enqueue_copy(dst_ptr=he.unsafe_ptr(), src_buf=o_exc)
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
# `VENDOR_LIBRARIES.md` C7 recommends it as the answer to CUDA's `width`
# modulo on the strength of its documentation alone. It is checked at
# `num_lanes = 8` against a host oracle that folds within 8-lane groups.
# ---------------------------------------------------------------------------


comptime WARP_LANES = lib_lane_width_for[TARGET_COLUMN]()


def warp_check_kernel(
    out_lane: MutPointer[Int32, MutAnyOrigin],
    out_sum: MutPointer[Float32, MutAnyOrigin],
    out_scan: MutPointer[Float32, MutAnyOrigin],
    out_bfly: MutPointer[Float32, MutAnyOrigin],
    out_bcast: MutPointer[Float32, MutAnyOrigin],
    out_vote: MutPointer[UInt32, MutAnyOrigin],
    out_lg8: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    tpb: Int32,
):
    var tid = Int(thread_idx.x)
    var g = Int(block_idx.x) * Int(tpb) + tid
    var v = a.unsafe_load(g)

    out_lane.unsafe_store(g, Int32(lane_id()))
    out_sum.unsafe_store(g, warp_sum(v))
    out_scan.unsafe_store(g, warp_prefix_sum(v))

    # The butterfly min every ported RAFT reducer uses: XOR over the whole
    # lane group, which is a total-order idempotent fold, so the answer is
    # the group minimum in EVERY lane.
    var m = v
    var off = 1
    while off < WARP_LANES:
        var other = shuffle_xor(m, UInt32(off))
        if other < m:
            m = other
        off *= 2
    out_bfly.unsafe_store(g, m)

    # Lane 0's value, broadcast. `shuffle_idx` has no width parameter, so
    # this is a whole-warp broadcast and nothing else.
    out_bcast.unsafe_store(g, shuffle_idx(v, UInt32(0)))

    out_vote.unsafe_store(g, vote[DType.uint32](v > Float32(500.0)))

    var lg = lane_group_min[num_lanes=8](v)
    out_lg8.unsafe_store(g, lg)


def check_warp(mut rows: List[Verdict]) raises:
    var ctx = DeviceContext()
    var tpb = 8 * WARP_LANES
    var n_blocks = 13
    var n = n_blocks * tpb
    var a = ctx.enqueue_create_buffer[DType.float32](n)
    var o_lane = ctx.enqueue_create_buffer[DType.int32](n)
    var o_sum = ctx.enqueue_create_buffer[DType.float32](n)
    var o_scan = ctx.enqueue_create_buffer[DType.float32](n)
    var o_bfly = ctx.enqueue_create_buffer[DType.float32](n)
    var o_bcast = ctx.enqueue_create_buffer[DType.float32](n)
    var o_vote = ctx.enqueue_create_buffer[DType.uint32](n)
    var o_lg8 = ctx.enqueue_create_buffer[DType.float32](n)
    var ha = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(n):
        ha.unsafe_ptr().unsafe_store(i, Float32(Int(_mix(i, 61) % 1000)))
    ctx.enqueue_copy(dst_buf=a, src_ptr=ha.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[warp_check_kernel](
        o_lane.unsafe_ptr(),
        o_sum.unsafe_ptr(),
        o_scan.unsafe_ptr(),
        o_bfly.unsafe_ptr(),
        o_bcast.unsafe_ptr(),
        o_vote.unsafe_ptr(),
        o_lg8.unsafe_ptr(),
        a.unsafe_ptr(),
        Int32(tpb),
        grid_dim=(n_blocks, 1, 1),
        block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()

    var hl = ctx.enqueue_create_host_buffer[DType.int32](n)
    var hs = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hc = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hb = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hd = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hv = ctx.enqueue_create_host_buffer[DType.uint32](n)
    var hg = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=hl.unsafe_ptr(), src_buf=o_lane)
    ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=o_sum)
    ctx.enqueue_copy(dst_ptr=hc.unsafe_ptr(), src_buf=o_scan)
    ctx.enqueue_copy(dst_ptr=hb.unsafe_ptr(), src_buf=o_bfly)
    ctx.enqueue_copy(dst_ptr=hd.unsafe_ptr(), src_buf=o_bcast)
    ctx.enqueue_copy(dst_ptr=hv.unsafe_ptr(), src_buf=o_vote)
    ctx.enqueue_copy(dst_ptr=hg.unsafe_ptr(), src_buf=o_lg8)
    ctx.synchronize()

    var bad_lane = String("")
    var bad_sum = String("")
    var bad_scan = String("")
    var bad_bfly = String("")
    var bad_bcast = String("")
    var bad_vote = String("")
    var bad_lg8 = String("")
    var scan_is_inclusive = True

    var n_warps = n // WARP_LANES
    for w in range(n_warps):
        var base = w * WARP_LANES
        var want_sum = Float32(0.0)
        var want_min = ha.unsafe_ptr().unsafe_load(base)
        var want_mask = UInt32(0)
        for t in range(WARP_LANES):
            var v = ha.unsafe_ptr().unsafe_load(base + t)
            want_sum += v
            if v < want_min:
                want_min = v
            if v > Float32(500.0):
                want_mask = want_mask | (UInt32(1) << UInt32(t))
        var running = Float32(0.0)
        for t in range(WARP_LANES):
            var v = ha.unsafe_ptr().unsafe_load(base + t)
            var exc = running
            running += v
            if hc.unsafe_ptr().unsafe_load(base + t) != running:
                scan_is_inclusive = False
                if hc.unsafe_ptr().unsafe_load(base + t) != exc:
                    if bad_scan == "":
                        bad_scan = (
                            "warp "
                            + String(w)
                            + " lane "
                            + String(t)
                            + " device "
                            + String(hc.unsafe_ptr().unsafe_load(base + t))
                            + " host inclusive "
                            + String(running)
                            + " host exclusive "
                            + String(exc)
                        )
            if Int(hl.unsafe_ptr().unsafe_load(base + t)) != t and bad_lane == "":
                bad_lane = (
                    "global "
                    + String(base + t)
                    + " reports lane "
                    + String(hl.unsafe_ptr().unsafe_load(base + t))
                    + ", expected "
                    + String(t)
                )
            if hs.unsafe_ptr().unsafe_load(base + t) != want_sum and bad_sum == "":
                bad_sum = (
                    "warp "
                    + String(w)
                    + " lane "
                    + String(t)
                    + " device "
                    + String(hs.unsafe_ptr().unsafe_load(base + t))
                    + " host "
                    + String(want_sum)
                )
            if hb.unsafe_ptr().unsafe_load(base + t) != want_min and bad_bfly == "":
                bad_bfly = (
                    "warp "
                    + String(w)
                    + " lane "
                    + String(t)
                    + " device "
                    + String(hb.unsafe_ptr().unsafe_load(base + t))
                    + " host "
                    + String(want_min)
                )
            if (
                hd.unsafe_ptr().unsafe_load(base + t)
                != ha.unsafe_ptr().unsafe_load(base)
                and bad_bcast == ""
            ):
                bad_bcast = (
                    "warp "
                    + String(w)
                    + " lane "
                    + String(t)
                    + " got "
                    + String(hd.unsafe_ptr().unsafe_load(base + t))
                    + ", lane 0 holds "
                    + String(ha.unsafe_ptr().unsafe_load(base))
                )
            if hv.unsafe_ptr().unsafe_load(base + t) != want_mask and bad_vote == "":
                bad_vote = (
                    "warp "
                    + String(w)
                    + " lane "
                    + String(t)
                    + " mask "
                    + String(hv.unsafe_ptr().unsafe_load(base + t))
                    + ", host "
                    + String(want_mask)
                )
        # lane_group_min[num_lanes=8]: the fold is within each aligned group
        # of 8 lanes.
        var grp = 0
        while grp < WARP_LANES:
            var gmin = ha.unsafe_ptr().unsafe_load(base + grp)
            for t in range(grp, grp + 8):
                var v = ha.unsafe_ptr().unsafe_load(base + t)
                if v < gmin:
                    gmin = v
            for t in range(grp, grp + 8):
                if hg.unsafe_ptr().unsafe_load(base + t) != gmin and bad_lg8 == "":
                    bad_lg8 = (
                        "warp "
                        + String(w)
                        + " lane "
                        + String(t)
                        + " group-of-8 device "
                        + String(hg.unsafe_ptr().unsafe_load(base + t))
                        + " host "
                        + String(gmin)
                    )
            grp += 8

    var lanes = String(WARP_LANES) + " lanes, " + String(n_warps) + " warps"

    def _emit(
        mut rows: List[Verdict], sym: String, cp: String, err: String, ev: String
    ) raises:
        if err == "":
            print("  " + sym + " ok")
            rows.append(Verdict(sym, cp, V_CORRECT, ev, True))
        else:
            print("  " + sym + " FAIL: " + err)
            rows.append(Verdict(sym, cp, V_WRONG, err, True))

    _emit(rows, String("std.gpu.primitives.id.lane_id"), String("raft::laneId"), bad_lane, lanes)
    _emit(rows, String("std.gpu.primitives.warp.sum"), String("cub::WarpReduce"), bad_sum, lanes)
    _emit(
        rows,
        String("std.gpu.primitives.warp.prefix_sum"),
        String("cub::WarpScan"),
        bad_scan,
        lanes + (", INCLUSIVE" if scan_is_inclusive else ", EXCLUSIVE"),
    )
    _emit(
        rows,
        String("std.gpu.primitives.warp.shuffle_xor"),
        String("cub::ShuffleIndex / raft::shfl_xor"),
        bad_bfly,
        lanes + ", butterfly min fold",
    )
    _emit(
        rows,
        String("std.gpu.primitives.warp.shuffle_idx"),
        String("raft::shfl"),
        bad_bcast,
        lanes + ", lane-0 broadcast",
    )
    _emit(rows, String("std.gpu.primitives.warp.vote"), String("__ballot_sync"), bad_vote, lanes)
    _emit(
        rows,
        String("std.gpu.primitives.warp.lane_group_min[8]"),
        String("__shfl_xor_sync(..., width=8)"),
        bad_lg8,
        lanes + ", num_lanes=8",
    )


# ---------------------------------------------------------------------------
# nn.toppminp_gpu.run_radix_sort_pairs_gpu
#
# NOT WIRED. `VENDOR_LIBRARIES.md` Appendix A2 item 1 recommends it as the
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


def _radix_pairs_one[
    BLOCK: Int
](ctx: DeviceContext, batch: Int, n: Int, mut which: String) raises -> String:
    var total = batch * n
    var ka = ctx.enqueue_create_buffer[DType.float32](total)
    var kb = ctx.enqueue_create_buffer[DType.float32](total)
    var ia = ctx.enqueue_create_buffer[DType.int32](total)
    var ib = ctx.enqueue_create_buffer[DType.int32](total)
    var skip = ctx.enqueue_create_buffer[DType.bool](batch)
    var hk = ctx.enqueue_create_host_buffer[DType.float32](total)
    var hi = ctx.enqueue_create_host_buffer[DType.int32](total)
    var hs = ctx.enqueue_create_host_buffer[DType.bool](batch)
    ctx.synchronize()
    for b in range(batch):
        hs.unsafe_ptr().unsafe_store(b, False)
        for i in range(n):
            hk.unsafe_ptr().unsafe_store(b * n + i, _key_f32(b * n + i, 71))
            hi.unsafe_ptr().unsafe_store(b * n + i, Int32(i))
    ctx.enqueue_copy(dst_buf=ka, src_ptr=hk.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=ia, src_ptr=hi.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=skip, src_ptr=hs.unsafe_ptr())
    ctx.synchronize()

    var keys = DoubleBuffer[DType.float32](
        rebind[Pointer[Scalar[DType.float32], MutUntrackedOrigin]](
            ka.unsafe_ptr()
        ),
        rebind[Pointer[Scalar[DType.float32], MutUntrackedOrigin]](
            kb.unsafe_ptr()
        ),
        total,
    )
    var ids = DoubleBuffer[DType.int32](
        rebind[Pointer[Scalar[DType.int32], MutUntrackedOrigin]](
            ia.unsafe_ptr()
        ),
        rebind[Pointer[Scalar[DType.int32], MutUntrackedOrigin]](
            ib.unsafe_ptr()
        ),
        total,
    )
    try:
        run_radix_sort_pairs_gpu[ascending=True, BLOCK_SIZE=BLOCK](
            ctx,
            keys,
            ids,
            rebind[Pointer[Scalar[DType.bool], MutUntrackedOrigin]](
                skip.unsafe_ptr()
            ),
            IndexList[2](batch, n),
        )
        ctx.synchronize()
    except e:
        which = String("RAISED")
        return String(e)

    # Which half holds the answer is undocumented, so read BOTH and say.
    var ha = ctx.enqueue_create_host_buffer[DType.float32](total)
    var hbb = ctx.enqueue_create_host_buffer[DType.float32](total)
    var hia = ctx.enqueue_create_host_buffer[DType.int32](total)
    var hib = ctx.enqueue_create_host_buffer[DType.int32](total)
    ctx.enqueue_copy(dst_ptr=ha.unsafe_ptr(), src_buf=ka)
    ctx.enqueue_copy(dst_ptr=hbb.unsafe_ptr(), src_buf=kb)
    ctx.enqueue_copy(dst_ptr=hia.unsafe_ptr(), src_buf=ia)
    ctx.enqueue_copy(dst_ptr=hib.unsafe_ptr(), src_buf=ib)
    ctx.synchronize()

    var half_a_sorted = True
    var half_b_sorted = True
    for b in range(batch):
        for j in range(1, n):
            if ha.unsafe_ptr().unsafe_load(b * n + j) < ha.unsafe_ptr().unsafe_load(
                b * n + j - 1
            ):
                half_a_sorted = False
            if hbb.unsafe_ptr().unsafe_load(
                b * n + j
            ) < hbb.unsafe_ptr().unsafe_load(b * n + j - 1):
                half_b_sorted = False
    if half_a_sorted and not half_b_sorted:
        which = String("result in the CURRENT half (the input buffer)")
    elif half_b_sorted and not half_a_sorted:
        which = String("result in the ALTERNATE half")
    elif half_a_sorted and half_b_sorted:
        which = String(
            "verified in the CURRENT half (the input buffer); the alternate"
            " half came back monotone too"
        )
    else:
        which = String("NEITHER half is monotone")
        return "neither half of the double buffer came back monotone"

    var vals = ha if half_a_sorted else hbb
    var idxs = hia if half_a_sorted else hib

    for b in range(batch):
        var bits = List[UInt32](capacity=n)
        for i in range(n):
            bits.append(
                bitcast[DType.uint32](hk.unsafe_ptr().unsafe_load(b * n + i))
            )
        var srt = _host_sort_u32(bits)
        var seen = List[Bool](capacity=n)
        for _ in range(n):
            seen.append(False)
        for j in range(n):
            var want = bitcast[DType.float32](srt[j])
            var got = vals.unsafe_ptr().unsafe_load(b * n + j)
            if got != want:
                return (
                    "row "
                    + String(b)
                    + " position "
                    + String(j)
                    + " key device "
                    + String(got)
                    + " host "
                    + String(want)
                )
            var p = Int(idxs.unsafe_ptr().unsafe_load(b * n + j))
            if p < 0 or p >= n:
                return (
                    "row "
                    + String(b)
                    + " position "
                    + String(j)
                    + " payload index "
                    + String(p)
                    + " out of range"
                )
            if seen[p]:
                return (
                    "row "
                    + String(b)
                    + " payload index "
                    + String(p)
                    + " appears twice"
                )
            seen[p] = True
            # Payload only up to ties, same rule as top_k.
            if hk.unsafe_ptr().unsafe_load(b * n + p) != want:
                return (
                    "row "
                    + String(b)
                    + " position "
                    + String(j)
                    + " payload "
                    + String(p)
                    + " points at key "
                    + String(hk.unsafe_ptr().unsafe_load(b * n + p))
                    + ", expected "
                    + String(want)
                )
    return String("")


def _radix_sweep[
    BLOCK: Int
](mut rows: List[Verdict], label: String) raises:
    var sizes = _sizes()
    var bad = String("")
    var raised = String("")
    var which = String("")
    var note = String("")
    for si in range(len(sizes)):
        var n = sizes[si]
        # A fresh context per size: this call writes through raw untracked
        # pointers into two buffers it does not own, and leaked ping-pong
        # state between sizes would be indistinguishable from a sort bug.
        var ctx = DeviceContext()
        var err = _radix_pairs_one[BLOCK](ctx, 3, n, which)
        if err != "":
            print("  " + label + " batch=3 n =", n, "FAIL:", err)
            if bad == "":
                bad = "n=" + String(n)
            if which == "RAISED" and raised == "":
                raised = err
        else:
            print("  " + label + " batch=3 n =", n, "ok (" + which + ")")
            if note == "":
                note = which
    if bad == "":
        rows.append(
            Verdict(
                "nn.toppminp_gpu.run_radix_sort_pairs_gpu" + label,
                "cub::DeviceRadixSort::SortPairs",
                V_CORRECT,
                "batch=3, n in {1..100003}; " + note,
                False,
            )
        )
    elif raised != "":
        rows.append(
            Verdict(
                "nn.toppminp_gpu.run_radix_sort_pairs_gpu" + label,
                "cub::DeviceRadixSort::SortPairs",
                V_UNCHECKED,
                "RAISES at " + bad + ": " + raised,
                False,
            )
        )
    else:
        rows.append(
            Verdict(
                "nn.toppminp_gpu.run_radix_sort_pairs_gpu" + label,
                "cub::DeviceRadixSort::SortPairs",
                V_WRONG,
                "smallest failing " + bad,
                False,
            )
        )


def check_radix_sort_pairs(mut rows: List[Verdict]) raises:
    _radix_sweep[256](rows, String(" [BLOCK_SIZE=256, the default]"))
    _radix_sweep[128](rows, String(" [BLOCK_SIZE=128]"))
    _radix_sweep[64](rows, String(" [BLOCK_SIZE=64]"))


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


def _argmin_one(ctx: DeviceContext, batch: Int, n: Int) raises -> String:
    var inp = ctx.enqueue_create_buffer[DType.float32](batch * n)
    var out = ctx.enqueue_create_buffer[DType.int32](batch)
    var hin = ctx.enqueue_create_host_buffer[DType.float32](batch * n)
    var hout = ctx.enqueue_create_host_buffer[DType.int32](batch)
    ctx.synchronize()
    for i in range(batch * n):
        hin.unsafe_ptr().unsafe_store(i, _key_f32(i, 81))
    for b in range(batch):
        hout.unsafe_ptr().unsafe_store(b, Int32(-424242))
    ctx.enqueue_copy(dst_buf=inp, src_ptr=hin.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=out, src_ptr=hout.unsafe_ptr())
    ctx.synchronize()

    argmaxmin_gpu[largest=False](
        ctx,
        TileTensor(inp, row_major(batch, n)),
        TileTensor(out, row_major(batch, Int(1))),
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hout.unsafe_ptr(), src_buf=out)
    ctx.synchronize()

    for b in range(batch):
        var want = hin.unsafe_ptr().unsafe_load(b * n)
        var ties = 1
        for i in range(1, n):
            var v = hin.unsafe_ptr().unsafe_load(b * n + i)
            if v < want:
                want = v
                ties = 1
            elif v == want:
                ties += 1
        var got = Int(hout.unsafe_ptr().unsafe_load(b))
        if got == -424242:
            return "row " + String(b) + " was NEVER WRITTEN (poison survived)"
        if got < 0 or got >= n:
            return "row " + String(b) + " index " + String(got) + " out of range"
        if hin.unsafe_ptr().unsafe_load(b * n + got) != want:
            return (
                "row "
                + String(b)
                + " index "
                + String(got)
                + " points at "
                + String(hin.unsafe_ptr().unsafe_load(b * n + got))
                + ", host minimum is "
                + String(want)
                + " ("
                + String(ties)
                + " ties)"
            )
    return String("")


def check_argmaxmin(mut rows: List[Verdict]) raises:
    var ctx = DeviceContext()
    var sizes = _sizes()
    var bad = String("")
    var raised = String("")
    for si in range(len(sizes)):
        var n = sizes[si]
        var err = String("")
        try:
            err = _argmin_one(ctx, 5, n)
        except e:
            err = String(e)
            if raised == "":
                raised = String(e)
        if err != "":
            print("  argmaxmin_gpu batch=5 n =", n, "FAIL:", err)
            if bad == "":
                bad = "n=" + String(n)
        else:
            print("  argmaxmin_gpu batch=5 n =", n, "ok")
    if bad == "":
        rows.append(
            Verdict(
                "nn.argmaxmin_gpu.argmaxmin_gpu[largest=False]",
                "cub::ArgMin / raft KVP argmin",
                V_CORRECT,
                "batch=5, n in {1..100003}",
                False,
            )
        )
    elif raised != "":
        rows.append(
            Verdict(
                "nn.argmaxmin_gpu.argmaxmin_gpu[largest=False]",
                "cub::ArgMin / raft KVP argmin",
                V_UNCHECKED,
                "RAISES at " + bad + ": " + raised,
                False,
            )
        )
    else:
        rows.append(
            Verdict(
                "nn.argmaxmin_gpu.argmaxmin_gpu[largest=False]",
                "cub::ArgMin / raft KVP argmin",
                V_WRONG,
                "smallest failing " + bad,
                False,
            )
        )


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
    ctx.enqueue_copy(dst_buf=inp, src_ptr=hin.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=idx, src_ptr=hidx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=out, src_ptr=hout.unsafe_ptr())
    ctx.synchronize()

    gather[axis=0, target="gpu"](
        TileTensor(out, row_major(n_out, n_cols)),
        TileTensor(inp, row_major(n_rows, n_cols)),
        TileTensor(idx, row_major(n_out)),
        context=ctx,
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hout.unsafe_ptr(), src_buf=out)
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


def _bmm_one(ctx: DeviceContext, batch: Int, m: Int, n: Int, k: Int) raises -> String:
    var a = ctx.enqueue_create_buffer[DType.float32](batch * m * k)
    var b = ctx.enqueue_create_buffer[DType.float32](batch * n * k)
    var c = ctx.enqueue_create_buffer[DType.float32](batch * m * n)
    var ha = ctx.enqueue_create_host_buffer[DType.float32](batch * m * k)
    var hb = ctx.enqueue_create_host_buffer[DType.float32](batch * n * k)
    var hc = ctx.enqueue_create_host_buffer[DType.float32](batch * m * n)
    ctx.synchronize()
    for i in range(batch * m * k):
        ha.unsafe_ptr().unsafe_store(i, _val_f32(i, 101))
    for i in range(batch * n * k):
        hb.unsafe_ptr().unsafe_store(i, _val_f32(i, 102))
    for i in range(batch * m * n):
        hc.unsafe_ptr().unsafe_store(i, Float32(-987654.0))
    ctx.enqueue_copy(dst_buf=a, src_ptr=ha.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=b, src_ptr=hb.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=c, src_ptr=hc.unsafe_ptr())
    ctx.synchronize()

    batched_matmul[transpose_b=True, target="gpu"](
        TileTensor(c, row_major(batch, m, n)),
        TileTensor(a, row_major(batch, m, k)),
        TileTensor(b, row_major(batch, n, k)),
        context=ctx,
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hc.unsafe_ptr(), src_buf=c)
    ctx.synchronize()

    for bt in range(batch):
        for i in range(m):
            for j in range(n):
                var acc = Float64(0.0)
                var mag = Float64(0.0)
                for t in range(k):
                    var av = Float64(
                        ha.unsafe_ptr().unsafe_load(bt * m * k + i * k + t)
                    )
                    var bv = Float64(
                        hb.unsafe_ptr().unsafe_load(bt * n * k + j * k + t)
                    )
                    acc += av * bv
                    mag += abs(av * bv)
                var got = Float64(
                    hc.unsafe_ptr().unsafe_load(bt * m * n + i * n + j)
                )
                if got == Float64(-987654.0):
                    return (
                        "batch "
                        + String(bt)
                        + " cell ("
                        + String(i)
                        + ", "
                        + String(j)
                        + ") was NEVER WRITTEN (poison survived)"
                    )
                var tol = mag * 2.0e-6 + 1.0e-6
                if abs(got - acc) > tol:
                    return (
                        "batch "
                        + String(bt)
                        + " cell ("
                        + String(i)
                        + ", "
                        + String(j)
                        + ") device "
                        + String(got)
                        + " host "
                        + String(acc)
                    )
    return String("")


def check_bmm(mut rows: List[Verdict]) raises:
    var ctx = DeviceContext()
    var shapes: List[Int] = [
        1, 1, 1, 1,
        3, 255, 33, 17,
        3, 256, 64, 32,
        3, 257, 65, 33,
        7, 129, 127, 65,
        2, 1025, 32, 16,
    ]
    var bad = String("")
    var raised = String("")
    var s = 0
    while s < len(shapes):
        var bt = shapes[s]
        var m = shapes[s + 1]
        var n = shapes[s + 2]
        var k = shapes[s + 3]
        var err = String("")
        try:
            err = _bmm_one(ctx, bt, m, n, k)
        except e:
            err = String(e)
            if raised == "":
                raised = String(e)
        if err != "":
            print("  batched_matmul", bt, "x", m, "x", n, "x", k, "FAIL:", err)
            if bad == "":
                bad = String(bt) + "x" + String(m) + "x" + String(n) + "x" + String(k)
        else:
            print("  batched_matmul", bt, "x", m, "x", n, "x", k, "ok")
        s += 4
    if bad == "":
        rows.append(
            Verdict(
                "linalg.bmm.batched_matmul[transpose_b]",
                "cublasGemmStridedBatchedEx",
                V_CORRECT,
                "batch 1..7, m/n/k straddling 255/256/257 and 1025",
                False,
            )
        )
    elif raised != "":
        rows.append(
            Verdict(
                "linalg.bmm.batched_matmul[transpose_b]",
                "cublasGemmStridedBatchedEx",
                V_UNCHECKED,
                "RAISES at " + bad + ": " + raised,
                False,
            )
        )
    else:
        rows.append(
            Verdict(
                "linalg.bmm.batched_matmul[transpose_b]",
                "cublasGemmStridedBatchedEx",
                V_WRONG,
                "smallest failing " + bad,
                False,
            )
        )


# ---------------------------------------------------------------------------
# linalg.transpose.transpose  --  the standing reminder.
#
# `VENDOR_LIBRARIES.md` records it as signalling at runtime inside
# `_copy_with_strides` when handed device pointers, which is why
# `core/column_stats.mojo::transpose_kernel` exists. Re-run here so the claim
# has a date and an error string rather than a memory.
# ---------------------------------------------------------------------------


def check_transpose_aborts() raises:
    """NOT part of the table run: this ABORTS the process on the M4.

    `VENDOR_LIBRARIES.md` recorded `linalg.transpose` as "SIGNALS at runtime",
    which reads like something a caller could catch and fall back from. It is
    not. It is a hard abort inside `_copy_with_strides`, so a `try` around it
    does not help and a program that reaches it dies. That is the difference
    between an unusable call and a dangerous one.
    """
    var rows = List[Verdict]()
    _check_transpose(rows)


def _check_transpose(mut rows: List[Verdict]) raises:
    var ctx = DeviceContext()
    var m = 257
    var n = 129
    var inp = ctx.enqueue_create_buffer[DType.float32](m * n)
    var out = ctx.enqueue_create_buffer[DType.float32](m * n)
    var hin = ctx.enqueue_create_host_buffer[DType.float32](m * n)
    var hout = ctx.enqueue_create_host_buffer[DType.float32](m * n)
    ctx.synchronize()
    for i in range(m * n):
        hin.unsafe_ptr().unsafe_store(i, _key_f32(i, 111))
        hout.unsafe_ptr().unsafe_store(i, Float32(-987654.0))
    ctx.enqueue_copy(dst_buf=inp, src_ptr=hin.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=out, src_ptr=hout.unsafe_ptr())
    ctx.synchronize()

    var err = String("")
    var perms: List[Int] = [1, 0]
    try:
        transpose(
            TileTensor(out, row_major(n, m)),
            TileTensor(inp, row_major(m, n)),
            perms.unsafe_ptr(),
            ctx,
        )
        ctx.synchronize()
        ctx.enqueue_copy(dst_ptr=hout.unsafe_ptr(), src_buf=out)
        ctx.synchronize()
        for i in range(m):
            for j in range(n):
                var want = hin.unsafe_ptr().unsafe_load(i * n + j)
                var got = hout.unsafe_ptr().unsafe_load(j * m + i)
                if got != want:
                    err = (
                        "cell ("
                        + String(j)
                        + ", "
                        + String(i)
                        + ") device "
                        + String(got)
                        + " host "
                        + String(want)
                    )
                    break
            if err != "":
                break
    except e:
        print("  linalg.transpose on device buffers RAISED:", String(e))
        rows.append(
            Verdict(
                "linalg.transpose.transpose",
                "raft transpose",
                V_UNCHECKED,
                "RAISES on device buffers: " + String(e),
                False,
            )
        )
        return
    if err == "":
        print("  linalg.transpose 257 x 129 ok")
        rows.append(
            Verdict(
                "linalg.transpose.transpose",
                "raft transpose",
                V_CORRECT,
                "257x129 device buffers",
                False,
            )
        )
    else:
        print("  linalg.transpose 257 x 129 FAIL:", err)
        rows.append(
            Verdict(
                "linalg.transpose.transpose",
                "raft transpose",
                V_WRONG,
                "257x129: " + err,
                False,
            )
        )


# ---------------------------------------------------------------------------
# nn.cumsum.cumsum  --  CPU ONLY, re-confirmed cheaply and moved on from.
#
# `VENDOR_LIBRARIES.md` already records this and the record is right: the
# signature carries neither `ctx: DeviceContext` nor `target`. What this adds
# is a RUN, so the row is not resting on a signature read: the call is made
# on HOST buffers and its answer is compared against a host oracle, and a
# sibling compile probe (see the lane file) shows that adding `target="gpu"`
# does not typecheck. Inclusive and exclusive are both exercised, because the
# file's own correction C5 records an earlier probe that concluded "inclusive
# only" from a guessed symbol name.
# ---------------------------------------------------------------------------


def check_cumsum(mut rows: List[Verdict]) raises:
    var ctx = DeviceContext()
    var n = 4099
    var hin = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hinc = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hexc = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(n):
        hin.unsafe_ptr().unsafe_store(i, Float32(Int(_mix(i, 121) % 1000)))
    cumsum[exclusive=False, reverse=False, axis=0](
        TileTensor(hinc, row_major(n)), TileTensor(hin, row_major(n))
    )
    cumsum[exclusive=True, reverse=False, axis=0](
        TileTensor(hexc, row_major(n)), TileTensor(hin, row_major(n))
    )
    var running = Float32(0.0)
    var bad = String("")
    for i in range(n):
        var v = hin.unsafe_ptr().unsafe_load(i)
        var exc = running
        running += v
        if hinc.unsafe_ptr().unsafe_load(i) != running and bad == "":
            bad = "inclusive at " + String(i)
        if hexc.unsafe_ptr().unsafe_load(i) != exc and bad == "":
            bad = "exclusive at " + String(i)
    if bad == "":
        print("  cumsum (host) inclusive and exclusive ok, n = 4099")
    else:
        print("  cumsum (host) FAIL:", bad)
    rows.append(
        Verdict(
            "nn.cumsum.cumsum",
            "cub::DeviceScan::{Inclusive,Exclusive}Sum",
            V_CPU_ONLY,
            (
                "no ctx, no target in the signature; runs and is correct on"
                " HOST buffers at n=4099, inclusive and exclusive"
            )
            if bad == ""
            else ("host answer WRONG: " + bad),
            False,
        )
    )


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


def check_compile_established(mut rows: List[Verdict]) raises:
    rows.append(
        Verdict(
            "algorithm.reductions.reduce_{sum,max,min,mean,...}",
            "cub::DeviceReduce",
            V_NOT_FOUND,
            (
                "`from algorithm.reductions import reduce_sum` ->"
                " \"unable to locate module 'reductions'\". THE WHOLE MODULE"
                " IS FICTIONAL. `algorithm.reduce_op` and"
                " `algorithm.rowwise.launch` are the real symbols and both"
                " compile."
            ),
            False,
        )
    )
    rows.append(
        Verdict(
            "linalg.matmul.matmul[transpose_a=True]",
            "cublasGemmEx (T-N)",
            V_NOT_FOUND,
            (
                "refused at COMPILE time:"
                " max/kernels/src/linalg/matmul/__init__.mojo:110:9"
                " constraint failed: transpose_a not yet supported"
            ),
            False,
        )
    )
    rows.append(
        Verdict(
            "nn.argsort.argsort on rank 2",
            "cub::DeviceSegmentedRadixSort",
            V_NOT_FOUND,
            (
                "refused at COMPILE time: max/kernels/src/nn/argsort.mojo:547"
                " constraint failed. A (1, n) batched shape is NOT a"
                " workaround for the rank-1 bug."
            ),
            False,
        )
    )
    rows.append(
        Verdict(
            "linalg.qr_factorization.qr_factorization",
            "cuSOLVER geqrf",
            V_CPU_ONLY,
            "passing a DeviceContext: invalid call, unexpected argument",
            False,
        )
    )
    rows.append(
        Verdict(
            "nn.arg_nonzero.arg_nonzero",
            "cub::DeviceSelect::Flagged",
            V_CPU_ONLY,
            "passing a DeviceContext: invalid call, unexpected argument",
            False,
        )
    )
    rows.append(
        Verdict(
            "nn.softmax.softmax",
            "(no counterpart in this port)",
            V_UNCHECKED,
            (
                "imports; takes an InputFn closure rather than an input"
                " tensor. No call site in this tree and none in cuVS/cuML's"
                " dispatch for k-means, DBSCAN, PCA, k-NN or OLS, so nothing"
                " here would consume a verdict."
            ),
            False,
        )
    )
    rows.append(
        Verdict(
            "nn.concat.concat",
            "(no counterpart in this port)",
            V_UNCHECKED,
            (
                "imports; same reason as softmax. Listed here only because"
                " VENDOR_LIBRARIES.md had it under 'confirmed available'."
            ),
            False,
        )
    )
    rows.append(
        Verdict(
            "nn.moe.moe_create_indices",
            "cub bucket sort + CSR offsets",
            V_UNCHECKED,
            (
                "imports. Would need a fixture shaped like MoE routing"
                " (topk_ids, expert_usage_stats) to exercise; its shared"
                " cache is sized by `expected_count` and it launches one"
                " block per expert, so DBSCAN-CSR-sized bucket counts are the"
                " open question, not correctness at 8 buckets."
            ),
            False,
        )
    )
    rows.append(
        Verdict(
            "algorithm.rowwise.launch + algorithm.reduce_op.ArgMin",
            "cub::BlockReduce<KeyValuePair>",
            V_UNCHECKED,
            (
                "both import. Needs a user ReduceOp struct and a body"
                " closure; not written this round. This is the one row worth"
                " promoting next, because it is the only shipped"
                " key-carrying reduction."
            ),
            False,
        )
    )
    rows.append(
        Verdict(
            "max.algorithm.reduction.sum (shape-taking overload)",
            "cub::DeviceReduce",
            V_UNCHECKED,
            "imports; not run. `algorithm.reductions` (plural) does NOT exist.",
            False,
        )
    )


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


def check_wired_gemm_at_n1(mut rows: List[Verdict]) raises:
    var ctx = DeviceContext()
    var m = 64
    var n = 1
    var k = 32
    var x = ctx.enqueue_create_buffer[DType.float32](m * k)
    var y = ctx.enqueue_create_buffer[DType.float32](n * k)
    var z = ctx.enqueue_create_buffer[DType.float32](m * n)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](m * k)
    var hy = ctx.enqueue_create_host_buffer[DType.float32](n * k)
    var hz = ctx.enqueue_create_host_buffer[DType.float32](m * n)
    ctx.synchronize()
    for i in range(m * k):
        hx.unsafe_ptr().unsafe_store(i, _val_f32(i, 131))
    for i in range(n * k):
        hy.unsafe_ptr().unsafe_store(i, _val_f32(i, 132))
    for i in range(m * n):
        hz.unsafe_ptr().unsafe_store(i, Float32(-987654.0))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=y, src_ptr=hy.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=z, src_ptr=hz.unsafe_ptr())
    ctx.synchronize()

    gemm_nt(ctx, z, x, y, m, n, k)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hz.unsafe_ptr(), src_buf=z)
    ctx.synchronize()

    var unwritten = 0
    var wrong = 0
    for i in range(m):
        var acc = Float64(0.0)
        var mag = Float64(0.0)
        for t in range(k):
            var a = Float64(hx.unsafe_ptr().unsafe_load(i * k + t))
            var b = Float64(hy.unsafe_ptr().unsafe_load(t))
            acc += a * b
            mag += abs(a * b)
        var got = Float64(hz.unsafe_ptr().unsafe_load(i))
        if got == Float64(-987654.0):
            unwritten += 1
        elif abs(got - acc) > mag * 2.0e-6 + 1.0e-6:
            wrong += 1
    if unwritten == 0 and wrong == 0:
        print("  core/gemm.mojo::gemm_nt at n = 1 ok")
        rows.append(
            Verdict(
                "core/gemm.mojo::gemm_nt at n = 1 (WIRED PATH)",
                "cublasGemmEx (degenerate)",
                V_CORRECT,
                "m=64 n=1 k=32 through the real wrapper",
                True,
            )
        )
    else:
        print(
            "  core/gemm.mojo::gemm_nt at n = 1 FAIL:",
            unwritten,
            "of",
            m,
            "outputs NEVER WRITTEN,",
            wrong,
            "written wrong",
        )
        rows.append(
            Verdict(
                "core/gemm.mojo::gemm_nt at n = 1 (WIRED PATH)",
                "cublasGemmEx (degenerate)",
                V_WRONG,
                (
                    "m=64 n=1 k=32: "
                    + String(unwritten)
                    + " of "
                    + String(m)
                    + " outputs NEVER WRITTEN, "
                    + String(wrong)
                    + " written wrong. REACHABLE: k-means with n_clusters=1"
                    " (min_cluster_distance_compute.mojo:197), OLS/PCA/TSVD"
                    " with one feature (lstsq.mojo:120,193; pca.mojo:202;"
                    " tsvd.mojo:89), k-NN with one index point"
                    " (knn_brute_force.mojo:154)"
                ),
                True,
            )
        )


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


def _matmul_colmajor_one(
    ctx: DeviceContext, m: Int, n: Int, k: Int
) raises -> String:
    """a stored row-major (k x m), viewed `col_major(m, k)`; b stored
    row-major (k x n); z poisoned; every cell against a Float64 oracle."""
    var a = ctx.enqueue_create_buffer[DType.float32](k * m)
    var b = ctx.enqueue_create_buffer[DType.float32](k * n)
    var z = ctx.enqueue_create_buffer[DType.float32](m * n)
    var ha = ctx.enqueue_create_host_buffer[DType.float32](k * m)
    var hb = ctx.enqueue_create_host_buffer[DType.float32](k * n)
    var hz = ctx.enqueue_create_host_buffer[DType.float32](m * n)
    ctx.synchronize()
    for i in range(k * m):
        ha.unsafe_ptr().unsafe_store(i, _val_f32(i, 71))
    for i in range(k * n):
        hb.unsafe_ptr().unsafe_store(i, _val_f32(i, 72))
    for i in range(m * n):
        hz.unsafe_ptr().unsafe_store(i, Float32(-987654.0))
    ctx.enqueue_copy(dst_buf=a, src_ptr=ha.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=b, src_ptr=hb.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=z, src_ptr=hz.unsafe_ptr())
    ctx.synchronize()

    matmul[transpose_b=False, target="gpu"](
        TileTensor(z, row_major(m, n)),
        TileTensor(a, col_major(m, k)),
        TileTensor(b, row_major(k, n)),
        ctx,
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hz.unsafe_ptr(), src_buf=z)
    ctx.synchronize()

    for i in range(m):
        for j in range(n):
            var acc = Float64(0.0)
            var mag = Float64(0.0)
            for t in range(k):
                var av = Float64(ha.unsafe_ptr().unsafe_load(t * m + i))
                var bv = Float64(hb.unsafe_ptr().unsafe_load(t * n + j))
                acc += av * bv
                mag += abs(av * bv)
            var got = Float64(hz.unsafe_ptr().unsafe_load(i * n + j))
            if got == Float64(-987654.0):
                return (
                    "cell ("
                    + String(i)
                    + ", "
                    + String(j)
                    + ") was NEVER WRITTEN (poison survived)"
                )
            var tol = mag * 1.0e-5 + 1.0e-6
            if abs(got - acc) > tol:
                return (
                    "cell ("
                    + String(i)
                    + ", "
                    + String(j)
                    + ") device "
                    + String(got)
                    + " host "
                    + String(acc)
                    + " tol "
                    + String(tol)
                )
    return String("")


def check_matmul_colmajor(mut rows: List[Verdict]) raises:
    var ctx = DeviceContext()

    # One shape the dispatcher gets RIGHT and two from the wrong band, plus
    # the n=1 arm. If every one of these ever comes back correct, the
    # toolchain changed underneath us -- the verdict flips to CORRECT and the
    # view becomes worth re-probing ACROSS THE WHOLE SWEEP before any wiring.
    var ok_large = _matmul_colmajor_one(ctx, 32, 32, 100003)
    var band1 = _matmul_colmajor_one(ctx, 8, 8, 512)
    var band2 = _matmul_colmajor_one(ctx, 32, 32, 2048)
    var n1 = _matmul_colmajor_one(ctx, 8, 1, 33)
    print(
        "  matmul colmajor-A 32x32x100003:",
        ("ok" if ok_large == "" else "FAIL " + ok_large),
    )
    print(
        "  matmul colmajor-A 8x8x512:",
        ("ok" if band1 == "" else "WRONG, as recorded"),
    )
    print(
        "  matmul colmajor-A 32x32x2048:",
        ("ok" if band2 == "" else "WRONG, as recorded"),
    )
    print(
        "  matmul colmajor-A 8x1x33:",
        ("ok" if n1 == "" else "WRONG, as recorded"),
    )
    if band1 == "" and band2 == "" and n1 == "" and ok_large == "":
        rows.append(
            Verdict(
                "linalg.matmul[A col_major view] (T-N)",
                "cublasGemmEx (OP_T, strides)",
                V_CORRECT,
                "HAZARD GONE at 4 probe shapes -- re-run the full sweep"
                " (LANE_covariance-unblock) before wiring",
                False,
            )
        )
    else:
        rows.append(
            Verdict(
                "linalg.matmul[A col_major view] (T-N)",
                "cublasGemmEx (OP_T, strides)",
                V_WRONG,
                "arm-dependent: ok at 32x32x100003 yet EVERY cell wrong"
                " across m=n 4..64 x k 64..2048 and at n=1; UNWIREABLE",
                False,
            )
        )

    # gemv_gpu on a col-major view: stride-blind at every output. This is
    # why the zero-copy X^T y has NO vendor route and
    # `column_stats.mojo::xty_kernel` stays hand-written.
    var kb = 4001
    var mb = 8
    var a2 = ctx.enqueue_create_buffer[DType.float32](kb * mb)
    var b2 = ctx.enqueue_create_buffer[DType.float32](kb)
    var c2 = ctx.enqueue_create_buffer[DType.float32](mb)
    var ha2 = ctx.enqueue_create_host_buffer[DType.float32](kb * mb)
    var hb2 = ctx.enqueue_create_host_buffer[DType.float32](kb)
    var hc2 = ctx.enqueue_create_host_buffer[DType.float32](mb)
    ctx.synchronize()
    for i in range(kb * mb):
        ha2.unsafe_ptr().unsafe_store(i, _val_f32(i, 92))
    for i in range(kb):
        hb2.unsafe_ptr().unsafe_store(i, _val_f32(i, 93))
    for i in range(mb):
        hc2.unsafe_ptr().unsafe_store(i, Float32(-987654.0))
    ctx.enqueue_copy(dst_buf=a2, src_ptr=ha2.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=b2, src_ptr=hb2.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=c2, src_ptr=hc2.unsafe_ptr())
    ctx.synchronize()
    gemv_gpu[transpose_b=False](
        TileTensor(c2, row_major(mb, Int(1))),
        TileTensor(a2, col_major(mb, kb)),
        TileTensor(b2, row_major(kb, Int(1))),
        ctx,
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hc2.unsafe_ptr(), src_buf=c2)
    ctx.synchronize()
    var gemv_bad = 0
    for i in range(mb):
        var acc = Float64(0.0)
        var mag = Float64(0.0)
        for t in range(kb):
            acc += Float64(ha2.unsafe_ptr().unsafe_load(t * mb + i)) * Float64(
                hb2.unsafe_ptr().unsafe_load(t)
            )
            mag += abs(
                Float64(ha2.unsafe_ptr().unsafe_load(t * mb + i))
            ) * abs(Float64(hb2.unsafe_ptr().unsafe_load(t)))
        var got = Float64(hc2.unsafe_ptr().unsafe_load(i))
        if got == Float64(-987654.0) or abs(got - acc) > mag * 2.0e-6 + 1.0e-6:
            gemv_bad += 1
    if gemv_bad == 0:
        print("  gemv_gpu colmajor-A ok (HAZARD GONE)")
        rows.append(
            Verdict(
                "linalg.gemv.gemv_gpu[A col_major view]",
                "cublasgemv (OP_T)",
                V_CORRECT,
                "8 outputs, k=4001",
                False,
            )
        )
    else:
        print(
            "  gemv_gpu colmajor-A WRONG at",
            gemv_bad,
            "of",
            mb,
            "outputs, as recorded",
        )
        rows.append(
            Verdict(
                "linalg.gemv.gemv_gpu[A col_major view]",
                "cublasgemv (OP_T)",
                V_WRONG,
                "wrong at "
                + String(gemv_bad)
                + " of 8 outputs, k=4001; strides ignored -- xty_kernel"
                " stays hand-written",
                False,
            )
        )

    # THE WIRED PATH: gemm_tn through the real wrapper. At 32x32x10007 the
    # dispatch takes the SPLIT-K arm (core/gram_splitk.mojo) -- printed
    # below from the same predicate -- and the vendor arm
    # (transpose_kernel x2 + gemm_nt) is then run EXPLICITLY at the same
    # shape, because a dispatched wrapper run covers one arm only
    # (PORTING_RULES.md 8). Operands are the SAME matrix, so the diagonal is
    # a same-sign sum and the tolerance must cover plain accumulation-order
    # spread: measured worst 4.09e-6 relative at 32x32x10007 on two
    # independent routes, so the budget is 1e-5, not the 2e-6 the
    # cancellation-shaped checks use. Output AND both alias buffers
    # pre-poisoned: on the vendor arm a transpose that does not run cannot
    # pass (the split-K arm never touches the aliases, by design).
    var km = 10007
    var mm = 32
    var x = ctx.enqueue_create_buffer[DType.float32](km * mm)
    var xa = ctx.enqueue_create_buffer[DType.float32](km * mm)
    var xa2 = ctx.enqueue_create_buffer[DType.float32](km * mm)
    var z = ctx.enqueue_create_buffer[DType.float32](mm * mm)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](km * mm)
    var hz = ctx.enqueue_create_host_buffer[DType.float32](mm * mm)
    var hpois = ctx.enqueue_create_host_buffer[DType.float32](km * mm)
    ctx.synchronize()
    for i in range(km * mm):
        hx.unsafe_ptr().unsafe_store(i, _val_f32(i, 91))
        hpois.unsafe_ptr().unsafe_store(i, Float32(-123456.0))
    for i in range(mm * mm):
        hz.unsafe_ptr().unsafe_store(i, Float32(-987654.0))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=z, src_ptr=hz.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=xa, src_ptr=hpois.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=xa2, src_ptr=hpois.unsafe_ptr())
    ctx.synchronize()

    var wrapper_arm = String("split-K") if gram_splitk_applies(
        mm, mm, km
    ) else String("transpose+matmul")

    gemm_tn(ctx, z, x, xa, xa2, mm, mm, km)
    ctx.enqueue_copy(dst_ptr=hz.unsafe_ptr(), src_buf=z)
    ctx.synchronize()

    var tn_bad = 0
    for i in range(mm):
        for j in range(mm):
            var acc = Float64(0.0)
            var mag = Float64(0.0)
            for t in range(km):
                var av = Float64(hx.unsafe_ptr().unsafe_load(t * mm + i))
                var bv = Float64(hx.unsafe_ptr().unsafe_load(t * mm + j))
                acc += av * bv
                mag += abs(av * bv)
            var got = Float64(hz.unsafe_ptr().unsafe_load(i * mm + j))
            if got == Float64(-987654.0):
                tn_bad += 1
            elif abs(got - acc) > mag * 1.0e-5 + 1.0e-6:
                tn_bad += 1
    if tn_bad == 0:
        print("  core/gemm.mojo::gemm_tn (arm: " + wrapper_arm + ") ok")
        rows.append(
            Verdict(
                "core/gemm.mojo::gemm_tn (WIRED PATH)",
                "raft::stats::cov / lstsqEig gemm (OP_T)",
                V_CORRECT,
                "32x32x10007 through the real wrapper, arm: " + wrapper_arm,
                True,
            )
        )
    else:
        print(
            "  core/gemm.mojo::gemm_tn (arm: " + wrapper_arm + ") FAIL at",
            tn_bad,
            "of",
            mm * mm,
            "cells",
        )
        rows.append(
            Verdict(
                "core/gemm.mojo::gemm_tn (WIRED PATH)",
                "raft::stats::cov / lstsqEig gemm (OP_T)",
                V_WRONG,
                String(tn_bad)
                + " of "
                + String(mm * mm)
                + " cells wrong, arm: "
                + wrapper_arm,
                True,
            )
        )

    # The VENDOR ARM, explicitly: re-poison output and aliases, then call
    # gemm_tn_via_transpose by name. This is the row that keeps the
    # transpose_kernel x2 + gemm_nt route covered now that the wrapper
    # dispatches this shape to split-K.
    ctx.enqueue_copy(dst_buf=xa, src_ptr=hpois.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=xa2, src_ptr=hpois.unsafe_ptr())
    for i in range(mm * mm):
        hz.unsafe_ptr().unsafe_store(i, Float32(-987654.0))
    ctx.enqueue_copy(dst_buf=z, src_ptr=hz.unsafe_ptr())
    ctx.synchronize()

    gemm_tn_via_transpose(ctx, z, x, xa, xa2, mm, mm, km)
    ctx.enqueue_copy(dst_ptr=hz.unsafe_ptr(), src_buf=z)
    ctx.synchronize()

    var arm_bad = 0
    for i in range(mm):
        for j in range(mm):
            var acc = Float64(0.0)
            var mag = Float64(0.0)
            for t in range(km):
                var av = Float64(hx.unsafe_ptr().unsafe_load(t * mm + i))
                var bv = Float64(hx.unsafe_ptr().unsafe_load(t * mm + j))
                acc += av * bv
                mag += abs(av * bv)
            var got = Float64(hz.unsafe_ptr().unsafe_load(i * mm + j))
            if got == Float64(-987654.0):
                arm_bad += 1
            elif abs(got - acc) > mag * 1.0e-5 + 1.0e-6:
                arm_bad += 1
    if arm_bad == 0:
        print("  core/gemm.mojo::gemm_tn_via_transpose (vendor arm) ok")
        rows.append(
            Verdict(
                "core/gemm.mojo::gemm_tn_via_transpose (vendor arm)",
                "raft::stats::cov / lstsqEig gemm (OP_T)",
                V_CORRECT,
                "32x32x10007 called by name, aliases pre-poisoned",
                True,
            )
        )
    else:
        print(
            "  core/gemm.mojo::gemm_tn_via_transpose FAIL at",
            arm_bad,
            "of",
            mm * mm,
            "cells",
        )
        rows.append(
            Verdict(
                "core/gemm.mojo::gemm_tn_via_transpose (vendor arm)",
                "raft::stats::cov / lstsqEig gemm (OP_T)",
                V_WRONG,
                String(arm_bad) + " of " + String(mm * mm) + " cells wrong",
                True,
            )
        )
