# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Launch RAFT's warp-sort top-k, check it against the ported radix select,
and sabotage it.

NO RAFT COUNTERPART. Same discipline as `neighbors/checks/knn_check.mojo`:
a kernel is not ported until it has been enqueued (`archive/reference/PORTING.md 9`), and a
correct answer is not by itself evidence that the kernel ran.

WHY THIS FILE EXISTS
--------------------
`neighbors/gbdt/matrix/detail/select_warpsort.mojo` was committed in a state
where `mojo build` CRASHED at any launch site, so it had never executed. The
crash is fixed (see the loop-update note in `block_sort_done`); this file is
the proof that the fix reaches the device and computes RAFT's answer.

WHAT IS COMPARED, AND HOW STRICTLY
-----------------------------------
Three-way, on one scattered fixture:

  * warpsort   -- `warpsort_topk_block_kernel`, this round's new path
  * radix      -- `radix_topk_one_block_kernel`, the incumbent ported select
  * host oracle -- a Float64 sort on the host, INDEPENDENT of both

VALUES are compared strictly, as a sorted multiset per row. INDICES are
compared exactly only on the distinct-value fixture, where no tie exists to
resolve. There is a second fixture that is deliberately full of ties, and on
that one only the values are compared, because `select_radix.mojo` documents
that RAFT's radix places tied outputs with `atomicAdd` and their identity is
therefore not reproducible run to run. Warpsort's ties are stable for a fixed
launch geometry but not equal to radix's, so an index comparison there would
be asserting something neither library promises.

THE FIXTURE IS SCATTERED AND HASHED, ON PURPOSE
------------------------------------------------
A fixture whose cells all hold the same value verifies the total and nothing
about placement, and has already passed two real bugs in this repo at the
exact failing parameters. Every cell here is a distinct hashed float, mixed
sign, so a wrong reduction, a wrong lane mapping and a wrong payload are all
visible.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_idx, thread_idx

from neighbors.impl.matrix.detail.select_radix import (
    SELECT_BLOCK,
    radix_topk_one_block_kernel,
)
from neighbors.impl.matrix.detail.select_warpsort import (
    warpsort_topk_block_kernel,
)
from neighbors.impl.neighbors.topk.warp_topk import WarpSelect


comptime WS_ROWS = 8
comptime WS_LEN = 4096


def _cell(row: Int, col: Int, salt: Int) -> Float32:
    """A distinct, scattered, signed value per cell. Not monotone in either
    index, so a kernel that reduced along the wrong axis cannot pass."""
    var h = (row * 2654435761 + col * 40503 + salt * 2246822519) % 1000003
    var v = Float32(h) / Float32(1000003) - Float32(0.5)
    # Spread the exponent too, so twiddle_in/twiddle_out are exercised over
    # more than one binade.
    if (h % 7) == 0:
        v = v * Float32(1024.0)
    if (h % 11) == 0:
        v = v * Float32(0.0009765625)
    return v


def _run_warpsort[
    capacity: Int, block_warps: Int
](
    ctx: DeviceContext,
    mut in_val: DeviceBuffer[DType.float32],
    mut in_idx: DeviceBuffer[DType.uint32],
    mut out_val: DeviceBuffer[DType.float32],
    mut out_idx: DeviceBuffer[DType.uint32],
    rows: Int,
    length: Int,
    k: Int,
) raises:
    """LAUNCH GEOMETRY per the module docstring of `select_warpsort.mojo`.

    `warp_width = min(capacity, 32)`, `block_dim.x = block_warps * warp_width`
    and it MUST be a multiple of 32, `grid = (num_blocks, batch, 1)`. This
    check uses `num_blocks == 1`, the single-pass form, which writes the final
    answer and needs no second merging launch.
    """
    comptime WW = capacity if capacity < 32 else 32
    comptime BDX = block_warps * WW
    ctx.enqueue_function[warpsort_topk_block_kernel[capacity, True, block_warps]](
        in_val.unsafe_ptr(),
        in_idx.unsafe_ptr(),
        out_val.unsafe_ptr(),
        out_idx.unsafe_ptr(),
        Int32(length),
        Int32(k),
        Int32(0),
        grid_dim=(1, rows, 1),
        block_dim=(BDX, 1, 1),
    )
    ctx.synchronize()


def _run_radix(
    ctx: DeviceContext,
    mut in_val: DeviceBuffer[DType.float32],
    mut buf_val: DeviceBuffer[DType.float32],
    mut buf_idx: DeviceBuffer[DType.uint32],
    mut out_val: DeviceBuffer[DType.float32],
    mut out_idx: DeviceBuffer[DType.uint32],
    rows: Int,
    length: Int,
    k: Int,
    buf_len: Int,
) raises:
    ctx.enqueue_function[radix_topk_one_block_kernel](
        in_val.unsafe_ptr(),
        out_val.unsafe_ptr(),
        out_idx.unsafe_ptr(),
        buf_val.unsafe_ptr(),
        buf_idx.unsafe_ptr(),
        Int32(length),
        Int32(k),
        Int32(buf_len),
        Int32(1),
        grid_dim=(rows, 1, 1),
        block_dim=(SELECT_BLOCK, 1, 1),
    )
    ctx.synchronize()


def _sorted_copy(mut v: List[Float32]):
    """Insertion sort. The lists are at most `k <= 256` long."""
    for i in range(1, len(v)):
        var x = v[i]
        var j = i - 1
        while j >= 0 and v[j] > x:
            v[j + 1] = v[j]
            j -= 1
        v[j + 1] = x


def _one_case[
    capacity: Int, block_warps: Int
](
    ctx: DeviceContext,
    rows: Int,
    length: Int,
    k: Int,
    salt: Int,
    tied: Bool,
    check_indices: Bool,
) raises -> Int:
    """Run both selects on one fixture and compare. Returns the mismatch
    count so the caller can report every case, not just the first failure."""
    var eff_k = k
    if eff_k > length:
        eff_k = length

    var buf_len = length // 8
    if buf_len < eff_k:
        buf_len = eff_k

    var d_in = ctx.enqueue_create_buffer[DType.float32](rows * length)
    var d_inidx = ctx.enqueue_create_buffer[DType.uint32](rows * length)
    var d_bufv = ctx.enqueue_create_buffer[DType.float32](rows * 2 * buf_len)
    var d_bufi = ctx.enqueue_create_buffer[DType.uint32](rows * 2 * buf_len)
    var w_val = ctx.enqueue_create_buffer[DType.float32](rows * eff_k)
    var w_idx = ctx.enqueue_create_buffer[DType.uint32](rows * eff_k)
    var r_val = ctx.enqueue_create_buffer[DType.float32](rows * eff_k)
    var r_idx = ctx.enqueue_create_buffer[DType.uint32](rows * eff_k)
    ctx.synchronize()

    var host = ctx.enqueue_create_host_buffer[DType.float32](rows * length)
    for r in range(rows):
        for c in range(length):
            var v = _cell(r, c, salt)
            if tied:
                # Deliberate ties: collapse onto a coarse grid so many cells
                # share the k-th value and the tie handling is exercised.
                v = Float32(Int(v * Float32(8.0))) / Float32(8.0)
            host.unsafe_ptr().unsafe_store(r * length + c, v)
    ctx.enqueue_copy(dst_buf=d_in, src_ptr=host.unsafe_ptr())
    ctx.synchronize()

    _run_warpsort[capacity, block_warps](
        ctx, d_in, d_inidx, w_val, w_idx, rows, length, eff_k
    )
    _run_radix(
        ctx, d_in, d_bufv, d_bufi, r_val, r_idx, rows, length, eff_k, buf_len
    )

    var hw = ctx.enqueue_create_host_buffer[DType.float32](rows * eff_k)
    var hwi = ctx.enqueue_create_host_buffer[DType.uint32](rows * eff_k)
    var hr = ctx.enqueue_create_host_buffer[DType.float32](rows * eff_k)
    ctx.enqueue_copy(dst_ptr=hw.unsafe_ptr(), src_buf=w_val)
    ctx.enqueue_copy(dst_ptr=hwi.unsafe_ptr(), src_buf=w_idx)
    ctx.enqueue_copy(dst_ptr=hr.unsafe_ptr(), src_buf=r_val)
    ctx.synchronize()

    var bad = 0
    for r in range(rows):
        # --- the HOST ORACLE, independent of both kernels -----------------
        var all_v = List[Float32]()
        for c in range(length):
            all_v.append(host.unsafe_ptr().unsafe_load(r * length + c))
        _sorted_copy(all_v)

        var wv = List[Float32]()
        var rv = List[Float32]()
        for t in range(eff_k):
            wv.append(hw.unsafe_ptr().unsafe_load(r * eff_k + t))
            rv.append(hr.unsafe_ptr().unsafe_load(r * eff_k + t))
        _sorted_copy(wv)
        _sorted_copy(rv)

        for t in range(eff_k):
            if wv[t] != all_v[t]:
                bad += 1
            if rv[t] != all_v[t]:
                bad += 1

        # --- INDICES, only where the fixture has no ties ------------------
        if check_indices:
            for t in range(eff_k):
                var ix = Int(hwi.unsafe_ptr().unsafe_load(r * eff_k + t))
                if ix < 0 or ix >= length:
                    bad += 1
                elif (
                    host.unsafe_ptr().unsafe_load(r * length + ix)
                    != hw.unsafe_ptr().unsafe_load(r * eff_k + t)
                ):
                    bad += 1
    return bad


def check_warpsort_matches_radix() raises:
    """Values against the host oracle AND against the ported radix select, at
    every k RAFT's own dispatch would send here, plus `k > n`.

    `capacity = bound_by_power_of_two(k)` and `block_warps` is chosen so that
    `block_warps * min(capacity, 32) == 256`, which is the multiple-of-32
    contract the kernel's main loop depends on.
    """
    var ctx = DeviceContext()
    var total = 0

    total += _one_case[1, 256](ctx, WS_ROWS, WS_LEN, 1, 0, False, True)
    total += _one_case[2, 128](ctx, WS_ROWS, WS_LEN, 2, 1, False, True)
    total += _one_case[8, 32](ctx, WS_ROWS, WS_LEN, 8, 2, False, True)
    total += _one_case[32, 8](ctx, WS_ROWS, WS_LEN, 32, 3, False, True)
    total += _one_case[64, 8](ctx, WS_ROWS, WS_LEN, 64, 4, False, True)
    total += _one_case[128, 8](ctx, WS_ROWS, WS_LEN, 100, 5, False, True)
    total += _one_case[256, 8](ctx, WS_ROWS, WS_LEN, 256, 6, False, True)

    # k > n. `block_kernel:777` trims k to the row length; the caller gets
    # `length` results and the remaining slots are never written.
    total += _one_case[8, 32](ctx, WS_ROWS, 5, 8, 7, False, True)

    # TIES. Values only -- see the module docstring.
    total += _one_case[32, 8](ctx, WS_ROWS, WS_LEN, 32, 8, True, False)

    if total != 0:
        raise Error("warpsort/radix/oracle disagreement count " + String(total))
    print("check_warpsort_matches_radix: OK")


def check_warpsort_reach_by_sabotage() raises:
    """A digest cannot tell a working kernel from a no-op.

    Sabotage: drive ONE cell of ONE row to a value far below every other cell
    in the fixture. PREDICTED SHAPE: that row's top-k must now contain that
    value at rank 0 and carry that column as its payload, and no other row's
    answer may change at all. A no-op kernel fails the first half; a kernel
    that ignores its payload fails the second; a kernel that reduces across
    rows fails the third.
    """
    var ctx = DeviceContext()
    comptime CAP = 32
    comptime BW = 8
    comptime K = 32
    var rows = WS_ROWS
    var length = WS_LEN

    var d_in = ctx.enqueue_create_buffer[DType.float32](rows * length)
    var d_inidx = ctx.enqueue_create_buffer[DType.uint32](rows * length)
    var w_val = ctx.enqueue_create_buffer[DType.float32](rows * K)
    var w_idx = ctx.enqueue_create_buffer[DType.uint32](rows * K)
    ctx.synchronize()

    var host = ctx.enqueue_create_host_buffer[DType.float32](rows * length)
    for r in range(rows):
        for c in range(length):
            host.unsafe_ptr().unsafe_store(r * length + c, _cell(r, c, 21))
    ctx.enqueue_copy(dst_buf=d_in, src_ptr=host.unsafe_ptr())
    ctx.synchronize()

    _run_warpsort[CAP, BW](ctx, d_in, d_inidx, w_val, w_idx, rows, length, K)
    var base_v = ctx.enqueue_create_host_buffer[DType.float32](rows * K)
    var base_i = ctx.enqueue_create_host_buffer[DType.uint32](rows * K)
    ctx.enqueue_copy(dst_ptr=base_v.unsafe_ptr(), src_buf=w_val)
    ctx.enqueue_copy(dst_ptr=base_i.unsafe_ptr(), src_buf=w_idx)
    ctx.synchronize()

    # The window: big enough that the planted value MUST come back first,
    # small enough that the other k-1 answers of that row are untouched.
    comptime VICTIM_ROW = 3
    comptime VICTIM_COL = 2711
    comptime PLANTED = Float32(-9999.0)
    host.unsafe_ptr().unsafe_store(
        VICTIM_ROW * length + VICTIM_COL, PLANTED
    )
    ctx.enqueue_copy(dst_buf=d_in, src_ptr=host.unsafe_ptr())
    ctx.synchronize()

    _run_warpsort[CAP, BW](ctx, d_in, d_inidx, w_val, w_idx, rows, length, K)
    var sab_v = ctx.enqueue_create_host_buffer[DType.float32](rows * K)
    var sab_i = ctx.enqueue_create_host_buffer[DType.uint32](rows * K)
    ctx.enqueue_copy(dst_ptr=sab_v.unsafe_ptr(), src_buf=w_val)
    ctx.enqueue_copy(dst_ptr=sab_i.unsafe_ptr(), src_buf=w_idx)
    ctx.synchronize()

    # 1. the planted value is in the victim row, with the right payload
    var found = False
    for t in range(K):
        if sab_v.unsafe_ptr().unsafe_load(VICTIM_ROW * K + t) == PLANTED:
            if (
                Int(sab_i.unsafe_ptr().unsafe_load(VICTIM_ROW * K + t))
                == VICTIM_COL
            ):
                found = True
    if not found:
        raise Error(
            "sabotage: planted value absent from the victim row's top-k, or"
            " came back with the wrong payload -- the kernel did not run"
        )

    # 2. the victim row otherwise kept k-1 of its old answers
    var kept = 0
    for t in range(K):
        var old = base_v.unsafe_ptr().unsafe_load(VICTIM_ROW * K + t)
        for u in range(K):
            if sab_v.unsafe_ptr().unsafe_load(VICTIM_ROW * K + u) == old:
                kept += 1
                break
    if kept != K - 1:
        raise Error(
            "sabotage: victim row kept "
            + String(kept)
            + " of its old answers, expected "
            + String(K - 1)
        )

    # 3. no other row moved at all
    for r in range(rows):
        if r == VICTIM_ROW:
            continue
        for t in range(K):
            if base_v.unsafe_ptr().unsafe_load(r * K + t) != sab_v.unsafe_ptr(
            ).unsafe_load(r * K + t):
                raise Error(
                    "sabotage: row " + String(r) + " moved; the selection is"
                    " leaking across rows"
                )
            if base_i.unsafe_ptr().unsafe_load(r * K + t) != sab_i.unsafe_ptr(
            ).unsafe_load(r * K + t):
                raise Error(
                    "sabotage: row " + String(r) + " payloads moved"
                )

    print("check_warpsort_reach_by_sabotage: OK")


# =========================================================================
# `neighbors/impl/neighbors/topk`'s `WarpSelect`, the REGISTER-RESIDENT queue.
#
# Different file, different implementation, different check. This one has no
# shared memory and no block phase: one warp owns one row and the answer
# never leaves registers until `write_out`. It is what makes the fused
# distance+select kernel possible, so it is checked here on its own before
# any fused kernel exists to carry it.
#
# THE CALL CONTRACT, and it is not optional: every lane of the warp must
# call `add` the SAME number of times, because `check_thread_q` contains a
# warp vote and a merge full of shuffles. The probe kernel below rounds the
# row length up to a multiple of 32 and feeds the sentinel key for the tail,
# which is what a fused kernel must also do.
# =========================================================================


def _warpselect_probe_kernel[
    num_warp_q: Int, num_thread_q: Int
](
    inp: MutPointer[Float32, MutAnyOrigin],
    out_k: MutPointer[Float32, MutAnyOrigin],
    out_v: MutPointer[UInt32, MutAnyOrigin],
    length: Int32,
    k_in: Int32,
):
    var row = Int(block_idx.x)
    var n = Int(length)
    var k = Int(k_in)
    var base = inp.unsafe_offset(row * n)

    var q = WarpSelect[num_warp_q, num_thread_q, False](
        Float32(3.4028234663852886e38), UInt32(0xFFFFFFFF), k
    )

    # Uniform trip count across the warp; the tail feeds the sentinel.
    var lim = ((n + 31) // 32) * 32
    var i = Int(thread_idx.x)
    while i < lim:
        var key = Float32(3.4028234663852886e38)
        var val = UInt32(0xFFFFFFFF)
        if i < n:
            key = base.unsafe_load(i)
            val = UInt32(i)
        q.add(key, val)
        i += 32

    q.reduce()
    q.write_out(out_k.unsafe_offset(row * k), out_v.unsafe_offset(row * k), k)


def _one_warpselect_case[
    num_warp_q: Int, num_thread_q: Int
](ctx: DeviceContext, rows: Int, length: Int, k: Int, salt: Int) raises -> Int:
    var d_in = ctx.enqueue_create_buffer[DType.float32](rows * length)
    var d_ok = ctx.enqueue_create_buffer[DType.float32](rows * k)
    var d_ov = ctx.enqueue_create_buffer[DType.uint32](rows * k)
    ctx.synchronize()

    var host = ctx.enqueue_create_host_buffer[DType.float32](rows * length)
    for r in range(rows):
        for c in range(length):
            host.unsafe_ptr().unsafe_store(r * length + c, _cell(r, c, salt))
    ctx.enqueue_copy(dst_buf=d_in, src_ptr=host.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[
        _warpselect_probe_kernel[num_warp_q, num_thread_q]
    ](
        d_in.unsafe_ptr(),
        d_ok.unsafe_ptr(),
        d_ov.unsafe_ptr(),
        Int32(length),
        Int32(k),
        grid_dim=(rows, 1, 1),
        block_dim=(32, 1, 1),
    )
    ctx.synchronize()

    var hk = ctx.enqueue_create_host_buffer[DType.float32](rows * k)
    var hv = ctx.enqueue_create_host_buffer[DType.uint32](rows * k)
    ctx.enqueue_copy(dst_ptr=hk.unsafe_ptr(), src_buf=d_ok)
    ctx.enqueue_copy(dst_ptr=hv.unsafe_ptr(), src_buf=d_ov)
    ctx.synchronize()

    var bad = 0
    for r in range(rows):
        var all_v = List[Float32]()
        for c in range(length):
            all_v.append(host.unsafe_ptr().unsafe_load(r * length + c))
        _sorted_copy(all_v)

        var got = List[Float32]()
        for t in range(k):
            got.append(hk.unsafe_ptr().unsafe_load(r * k + t))
        _sorted_copy(got)

        for t in range(k):
            if got[t] != all_v[t]:
                bad += 1

        # The payload must name the cell the value came from. The fixture
        # has distinct values, so there is no tie to excuse a mismatch.
        for t in range(k):
            var ix = Int(hv.unsafe_ptr().unsafe_load(r * k + t))
            if ix < 0 or ix >= length:
                bad += 1
            elif (
                host.unsafe_ptr().unsafe_load(r * length + ix)
                != hk.unsafe_ptr().unsafe_load(r * k + t)
            ):
                bad += 1
    return bad


def check_warpselect_matches_oracle() raises:
    """Both instantiations `fused_l2_knn.cuh:743-771` makes, and nothing
    else: `NumWarpQ=32, NumThreadQ=2` for k <= 32 and `NumWarpQ=64,
    NumThreadQ=3` for k <= 64. `NumThreadQ = 3` is the odd size that forces
    the padded-SIMD deviation, so it is not a corner case here, it is half
    the shipped configuration."""
    var ctx = DeviceContext()
    var total = 0
    total += _one_warpselect_case[32, 2](ctx, WS_ROWS, WS_LEN, 1, 30)
    total += _one_warpselect_case[32, 2](ctx, WS_ROWS, WS_LEN, 8, 31)
    total += _one_warpselect_case[32, 2](ctx, WS_ROWS, WS_LEN, 32, 32)
    total += _one_warpselect_case[64, 3](ctx, WS_ROWS, WS_LEN, 33, 33)
    total += _one_warpselect_case[64, 3](ctx, WS_ROWS, WS_LEN, 50, 34)
    total += _one_warpselect_case[64, 3](ctx, WS_ROWS, WS_LEN, 64, 35)
    # A row length that is NOT a multiple of the warp size, to exercise the
    # sentinel tail.
    total += _one_warpselect_case[64, 3](ctx, WS_ROWS, 1013, 64, 36)
    if total != 0:
        raise Error(
            "WarpSelect/oracle disagreement count " + String(total)
        )
    print("check_warpselect_matches_oracle: OK")


def check_warpselect_reach_by_sabotage() raises:
    """Same discipline as the block kernel's sabotage: plant one value far
    below the rest of one row and require it back, with its column as the
    payload, and require no other row to move."""
    var ctx = DeviceContext()
    comptime K = 32
    var rows = WS_ROWS
    var length = WS_LEN

    var d_in = ctx.enqueue_create_buffer[DType.float32](rows * length)
    var d_ok = ctx.enqueue_create_buffer[DType.float32](rows * K)
    var d_ov = ctx.enqueue_create_buffer[DType.uint32](rows * K)
    ctx.synchronize()

    var host = ctx.enqueue_create_host_buffer[DType.float32](rows * length)
    for r in range(rows):
        for c in range(length):
            host.unsafe_ptr().unsafe_store(r * length + c, _cell(r, c, 41))
    ctx.enqueue_copy(dst_buf=d_in, src_ptr=host.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[_warpselect_probe_kernel[32, 2]](
        d_in.unsafe_ptr(), d_ok.unsafe_ptr(), d_ov.unsafe_ptr(),
        Int32(length), Int32(K), grid_dim=(rows, 1, 1), block_dim=(32, 1, 1),
    )
    ctx.synchronize()
    var base_k = ctx.enqueue_create_host_buffer[DType.float32](rows * K)
    ctx.enqueue_copy(dst_ptr=base_k.unsafe_ptr(), src_buf=d_ok)
    ctx.synchronize()

    comptime VICTIM_ROW = 5
    comptime VICTIM_COL = 1777
    comptime PLANTED = Float32(-9999.0)
    host.unsafe_ptr().unsafe_store(VICTIM_ROW * length + VICTIM_COL, PLANTED)
    ctx.enqueue_copy(dst_buf=d_in, src_ptr=host.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[_warpselect_probe_kernel[32, 2]](
        d_in.unsafe_ptr(), d_ok.unsafe_ptr(), d_ov.unsafe_ptr(),
        Int32(length), Int32(K), grid_dim=(rows, 1, 1), block_dim=(32, 1, 1),
    )
    ctx.synchronize()
    var sab_k = ctx.enqueue_create_host_buffer[DType.float32](rows * K)
    var sab_v = ctx.enqueue_create_host_buffer[DType.uint32](rows * K)
    ctx.enqueue_copy(dst_ptr=sab_k.unsafe_ptr(), src_buf=d_ok)
    ctx.enqueue_copy(dst_ptr=sab_v.unsafe_ptr(), src_buf=d_ov)
    ctx.synchronize()

    var found = False
    for t in range(K):
        if sab_k.unsafe_ptr().unsafe_load(VICTIM_ROW * K + t) == PLANTED:
            if (
                Int(sab_v.unsafe_ptr().unsafe_load(VICTIM_ROW * K + t))
                == VICTIM_COL
            ):
                found = True
    if not found:
        raise Error(
            "sabotage: WarpSelect did not return the planted value with its"
            " own column as payload -- the queue did not run"
        )

    var kept = 0
    for t in range(K):
        var old = base_k.unsafe_ptr().unsafe_load(VICTIM_ROW * K + t)
        for u in range(K):
            if sab_k.unsafe_ptr().unsafe_load(VICTIM_ROW * K + u) == old:
                kept += 1
                break
    if kept != K - 1:
        raise Error(
            "sabotage: WarpSelect victim row kept "
            + String(kept)
            + " of its old answers, expected "
            + String(K - 1)
        )

    for r in range(rows):
        if r == VICTIM_ROW:
            continue
        for t in range(K):
            if base_k.unsafe_ptr().unsafe_load(
                r * K + t
            ) != sab_k.unsafe_ptr().unsafe_load(r * K + t):
                raise Error(
                    "sabotage: WarpSelect row "
                    + String(r)
                    + " moved; the queue is leaking across rows"
                )

    print("check_warpselect_reach_by_sabotage: OK")
