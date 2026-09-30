# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SGD IN BOUNDED LAUNCHES, SAME BITS (lane/linfit-speed, 2026-09-29).

x_linear/device.mojo launches an SGD fit as ONE kernel that runs every epoch
of every row. At the board's shapes that launch holds the GPU for minutes,
and macOS aborts a Metal launch that holds the GPU for seconds, leaving its
output partly written with no error (the 2026-09-29 M2 Pro finding). On
Apple the fit therefore runs here, cut into launches that each do a bounded
amount of work:

  * ORDER: each epoch's order is the one-launch fit's (Fisher-Yates, i
    descending, j = draw mod (i + 1), draw k of problem c's stream the k-th
    splitmix64 word from seed + 1000003 c), applied to the previous epoch's
    order in slices of at most SGD_SWAPS swaps, one thread per problem.
  * ROWS: each launch runs rows [r0, r1) of one epoch of every problem, at
    most SGD_ROW_STEPS row terms, from the state the previous launch stored
    (x_linear/sgd.mojo SgdSpan): the same rows in the same order, the same
    operations, the words carried exactly, so the words are the one-launch
    fit's.
  * CHECKED: every launch's outputs are POISONED first (a quiet NaN word no
    computation here produces) and read back whole: the weights, the L1
    history and the state of every problem, whose last word each launch
    stores is a done mark. The shuffle permutes in place, so each slice
    stores a done mark per problem after its last swap, and after every
    epoch's shuffle the order is read back and checked to be a permutation.
    A stale word raises by name.

Elsewhere the one-launch fit stays (no watchdog on a compute GPU);
MOJOLEARN_X_LINEAR_SGD_BOUNDED=1 runs this path there too (the gate uses it
to prove the words equal), MOJOLEARN_X_LINEAR_SGD_STEPS / _SWAPS set the
budgets and MOJOLEARN_X_LINEAR_SGD_THREAD=1 takes the one-thread form.
"""
from std.gpu import block_idx, block_dim, thread_idx, WARP_SIZE
from std.os import getenv
from std.memory import bitcast
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from x_linear.ops import FP, IP, ld, st, ldi, sti, i2f
from x_linear.team import team_at
from x_linear.sgd import (
    SgdSpan, SGD_ST, ST_INTERCEPT, ST_EPOCHS, ST_STOP, ST_DONE, SGD_DONE_WORD, SGD_WARP_MAX_CHUNKS,
    sgd_one, sgd_one_warp, _splitmix_at,
)

#: Row terms (rows x max(d, WARP_SIZE)) per row launch. SCHEDULING only.
comptime SGD_ROW_STEPS = 1 << 20
#: Swaps per shuffle launch. SCHEDULING only.
comptime SGD_SWAPS = 1 << 16
comptime SB_TPB = 256
#: Threads of a row launch's block: one warp or wavefront runs the problem
#: (64 covers AMD's wavefront; the lanes past the first warp return).
comptime SB_ROW_TPB = 64
#: The warp form holds d <= SGD_WARP_MAX_CHUNKS * WARP_SIZE; 32 is the
#: smallest warp of any column, so d <= 256 takes it everywhere.
comptime SB_WARP_MAX_D = SGD_WARP_MAX_CHUNKS * 32
comptime SB_POISON = UInt32(0x7FC0DEAD)


def _env_int(name: String, default: Int) -> Int:
    var s = getenv(name, "")
    if s == "":
        return default
    try:
        var v = Int(s)
        return v if v > 0 else default
    except:
        return default


def use_sgd_bounded() -> Bool:
    comptime if has_apple_gpu_accelerator():
        return True
    return getenv("MOJOLEARN_X_LINEAR_SGD_BOUNDED", "") == "1"


def _poison() -> Float32:
    return bitcast[DType.float32](SB_POISON)


# ------------------------------------------------------------------ kernels

def sb_order_init_kernel(order: IP, total: Int32, n: Int32):
    """order[c * n + i] = i."""
    var e = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if e < Int(total):
        sti(order, e, e % Int(n))


def sb_targets_kernel(y: FP, ys: FP, total: Int32, n: Int32, k: Int32):
    """The one-thread form's targets (x_linear/sgd.mojo sgd_fit): problem c
    of k classes, row i."""
    var e = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if e >= Int(total):
        return
    var nn = Int(n)
    var c = e // nn
    var i = e % nn
    var kk = Int(k)
    var v = ld(y, i)
    var t: Float32
    if kk == 0:
        t = v
    elif kk == 1:
        t = Float32(1)
    elif kk == 2:
        t = Float32(1) if v == Float32(1) else Float32(-1)
    else:
        t = Float32(1) if v == i2f(c) else Float32(-1)
    st(ys, e, t)


def sb_shuffle_kernel(
    order: IP, n_in: Int32, seed_lo: Int32, seed_hi: Int32, epoch: Int32, tt0: Int32, tt1: Int32,
    problems: Int32, done: FP,
):
    """Problem c's swaps tt in [tt0, tt1) of epoch `epoch` (ops.shuffle's:
    i = n - 1 - tt, j = draw mod (i + 1), draw number epoch * (n - 1) + tt + 1
    of the problem's stream), then its done mark."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c >= Int(problems):
        return
    var n = Int(n_in)
    var seed = (UInt64(UInt32(seed_hi)) << 32) | UInt64(UInt32(seed_lo))
    seed = seed + UInt64(1000003) * UInt64(c)
    var idx = order + c * n
    var base = Int(epoch) * (n - 1)
    for tt in range(Int(tt0), Int(tt1)):
        var i = n - 1 - tt
        var j = Int(_splitmix_at(seed, base + tt + 1) % UInt64(i + 1))
        var a = ldi(idx, i)
        sti(idx, i, ldi(idx, j))
        sti(idx, j, a)
    st(done, c, Float32(Int(epoch) + 1))


def sb_rows_kernel(
    x: FP, y: FP, ys: FP, n_in: Int32, d_in: Int32, ip: IP, fp: FP, order: IP,
    w_in: FP, w_out: FP, q_in: FP, q_out: FP, s_in: FP, s_out: FP,
    epoch: Int32, r0: Int32, r1: Int32, form: Int32,
):
    """Rows [r0, r1) of epoch `epoch` of problem c = block_idx.x. form 0:
    the warp form (one warp a block, K by d), 1: the one-thread form."""
    var n = Int(n_in)
    var d = Int(d_in)
    var c = Int(block_idx.x)
    var k = ldi(ip, 0)
    var loss = ldi(ip, 1)
    var penalty = ldi(ip, 2)
    var lr = ldi(ip, 3)
    var fit_intercept = ldi(ip, 4) != 0
    var max_iter = ldi(ip, 5)
    var nic = ldi(ip, 6)
    var do_shuffle = ldi(ip, 7) != 0
    var seed = (UInt64(UInt32(ip.unsafe_load(9))) << 32) | UInt64(UInt32(ip.unsafe_load(8)))
    var alpha = ld(fp, 0)
    var l1r = ld(fp, 1)
    var eta0 = ld(fp, 2)
    var power_t = ld(fp, 3)
    var eps = ld(fp, 4)
    var tol = ld(fp, 5)
    var has_sw = ldi(ip, 10) != 0
    var has_cw = ldi(ip, 11) != 0
    var swp = y + n
    var problems = k if k > 2 else 1
    var cw_pos = ld(fp, 6 + c) if has_cw else Float32(1)
    var cw_neg = ld(fp, 6 + problems + c) if has_cw else Float32(1)
    var span = SgdSpan.make(
        True, Int(epoch), Int(r0), Int(r1), w_in + c * d, q_in + c * d, s_in + c * SGD_ST, s_out + c * SGD_ST,
    )
    var pseed = seed + UInt64(1000003) * UInt64(c)
    var idx = order + c * n
    var sb = s_out + c * SGD_ST
    if Int(form) == 1:
        if Int(thread_idx.x) == 0:
            _ = sgd_one(
                x, ys + c * n, n, d, loss, penalty, alpha, l1r, lr, eta0, power_t, eps,
                fit_intercept, max_iter, tol, nic, do_shuffle, pseed, k == 1,
                w_out, c * d, sb, ST_INTERCEPT, q_out + c * d, idx,
                swp, has_sw, cw_pos, cw_neg, has_cw, span,
            )
        return
    var lane = Int(thread_idx.x)
    if lane >= WARP_SIZE:
        return
    var tm = team_at(lane, WARP_SIZE, s_out, 0, 0, 0)
    var chunks = (d + WARP_SIZE - 1) // WARP_SIZE
    if chunks <= 1:
        _ = sgd_one_warp[1](
            lane, x, y, k, c, n, d, loss, penalty, alpha, l1r, lr, eta0, power_t, eps,
            fit_intercept, max_iter, tol, nic, do_shuffle, pseed, k == 1,
            w_out, c * d, sb, ST_INTERCEPT, idx, swp, has_sw, cw_pos, cw_neg, has_cw,
            False, tm, 0, idx, idx, idx, span, q_out + c * d,
        )
    elif chunks <= 2:
        _ = sgd_one_warp[2](
            lane, x, y, k, c, n, d, loss, penalty, alpha, l1r, lr, eta0, power_t, eps,
            fit_intercept, max_iter, tol, nic, do_shuffle, pseed, k == 1,
            w_out, c * d, sb, ST_INTERCEPT, idx, swp, has_sw, cw_pos, cw_neg, has_cw,
            False, tm, 0, idx, idx, idx, span, q_out + c * d,
        )
    elif chunks <= 4:
        _ = sgd_one_warp[4](
            lane, x, y, k, c, n, d, loss, penalty, alpha, l1r, lr, eta0, power_t, eps,
            fit_intercept, max_iter, tol, nic, do_shuffle, pseed, k == 1,
            w_out, c * d, sb, ST_INTERCEPT, idx, swp, has_sw, cw_pos, cw_neg, has_cw,
            False, tm, 0, idx, idx, idx, span, q_out + c * d,
        )
    else:
        _ = sgd_one_warp[SGD_WARP_MAX_CHUNKS](
            lane, x, y, k, c, n, d, loss, penalty, alpha, l1r, lr, eta0, power_t, eps,
            fit_intercept, max_iter, tol, nic, do_shuffle, pseed, k == 1,
            w_out, c * d, sb, ST_INTERCEPT, idx, swp, has_sw, cw_pos, cw_neg, has_cw,
            False, tm, 0, idx, idx, idx, span, q_out + c * d,
        )


# ------------------------------------------------------------------ host

def _check(p: FP, count: Int, what: String) raises:
    for i in range(count):
        if bitcast[DType.uint32](ld(p, i)) == SB_POISON:
            raise Error(
                "x_linear SGD: a bounded launch (" + what + ") left output word " + String(i)
                + " unwritten (on Apple: a launch the system aborted); the fit is refused"
            )


def _host(b: HostBuffer[DType.float32]) -> FP:
    return FP(unsafe_from_address=Int(b.unsafe_ptr()))


def _host_i(b: HostBuffer[DType.int32]) -> IP:
    return IP(unsafe_from_address=Int(b.unsafe_ptr()))


def _check_perm(o: IP, problems: Int, n: Int, mut seen: List[Bool], epoch: Int) raises:
    for c in range(problems):
        for i in range(n):
            seen[i] = False
        for i in range(n):
            var v = ldi(o, c * n + i)
            if v < 0 or v >= n or seen[v]:
                raise Error(
                    "x_linear SGD: epoch " + String(epoch) + "'s order of problem " + String(c)
                    + " is not a permutation (a shuffle launch the system aborted); the fit is refused"
                )
            seen[v] = True


def sgd_fit_bounded(
    ctx: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], n_out: Int, res: FP,
) raises:
    """x_linear/sgd.mojo `sgd_fit` in bounded launches (see the header).
    res (host): coef P*d, intercept P, n_iter, status."""
    var k = Int(ip[0])
    var max_iter = Int(ip[5])
    var do_shuffle = Int(ip[7]) != 0
    var problems = k if k > 2 else 1
    var row_steps = _env_int("MOJOLEARN_X_LINEAR_SGD_STEPS", SGD_ROW_STEPS)
    var swaps = _env_int("MOJOLEARN_X_LINEAR_SGD_SWAPS", SGD_SWAPS)
    var form = 1 if (d > SB_WARP_MAX_D or getenv("MOJOLEARN_X_LINEAR_SGD_THREAD", "") == "1") else 0
    var pd = problems * d
    var ps = problems * SGD_ST
    if max_iter < 1:
        # no epoch: sgd_fit's result is its initial state
        for e in range(pd):
            st(res, e, Float32(0))
        for c in range(problems):
            st(res, pd + c, Float32(1) if k == 1 else Float32(0))
        st(res, pd + problems, Float32(0))
        st(res, pd + problems + 1, Float32(0))
        return
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var dy = ctx.enqueue_create_buffer[DType.float32](max(n_y, 1))
    var dip = ctx.enqueue_create_buffer[DType.int32](max(len(ip), 1))
    var dfp = ctx.enqueue_create_buffer[DType.float32](max(len(fp), 1))
    var hip = ip.copy()
    var hfp = fp.copy()
    var order = ctx.enqueue_create_buffer[DType.int32](problems * n)
    var hord = ctx.enqueue_create_host_buffer[DType.int32](problems * n)
    var ys = ctx.enqueue_create_buffer[DType.float32](problems * n if form == 1 else 1)
    var hys = ctx.enqueue_create_host_buffer[DType.float32](problems * n if form == 1 else 1)
    var wa = ctx.enqueue_create_buffer[DType.float32](pd)
    var wb = ctx.enqueue_create_buffer[DType.float32](pd)
    var qa = ctx.enqueue_create_buffer[DType.float32](pd)
    var qb = ctx.enqueue_create_buffer[DType.float32](pd)
    var sa = ctx.enqueue_create_buffer[DType.float32](ps)
    var sbuf = ctx.enqueue_create_buffer[DType.float32](ps)
    var done = ctx.enqueue_create_buffer[DType.float32](problems)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](pd)
    var hq = ctx.enqueue_create_host_buffer[DType.float32](pd)
    var hs = ctx.enqueue_create_host_buffer[DType.float32](ps)
    var hdone = ctx.enqueue_create_host_buffer[DType.float32](problems)
    if n_x > 0:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    if n_y > 0:
        ctx.enqueue_copy(dst_buf=dy, src_ptr=y)
    ctx.enqueue_copy(dst_buf=dip, src_ptr=hip.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dfp, src_ptr=hfp.unsafe_ptr())
    # the identity order (checked whole), the one-thread form's targets
    order.enqueue_fill(Int32(-1))
    ctx.enqueue_function[sb_order_init_kernel](
        order.unsafe_ptr(), Int32(problems * n), Int32(n),
        grid_dim=(problems * n + SB_TPB - 1) // SB_TPB, block_dim=SB_TPB,
    )
    ctx.enqueue_copy(dst_buf=hord, src_buf=order)
    if form == 1:
        ys.enqueue_fill(_poison())
        ctx.enqueue_function[sb_targets_kernel](
            dy.unsafe_ptr(), ys.unsafe_ptr(), Int32(problems * n), Int32(n), Int32(k),
            grid_dim=(problems * n + SB_TPB - 1) // SB_TPB, block_dim=SB_TPB,
        )
        ctx.enqueue_copy(dst_buf=hys, src_buf=ys)
    ctx.synchronize()
    var ho = _host_i(hord)
    for e in range(problems * n):
        if ldi(ho, e) != e % n:
            raise Error("x_linear SGD: the identity order launch left word " + String(e) + " unwritten; the fit is refused")
    if form == 1:
        _check(_host(hys), problems * n, "targets")
    var seen = List[Bool](capacity=n)
    for _ in range(n):
        seen.append(False)
    var rows_per = max(1, row_steps // max(d, 32))
    var into_b = True  # the next launch writes the b buffers
    var first = True
    for epoch in range(max_iter):
        if do_shuffle and n > 1:
            var tt0 = 0
            while tt0 < n - 1:
                var tt1 = min(n - 1, tt0 + swaps)
                done.enqueue_fill(_poison())
                ctx.enqueue_function[sb_shuffle_kernel](
                    order.unsafe_ptr(), Int32(n), hip[8], hip[9], Int32(epoch), Int32(tt0), Int32(tt1),
                    Int32(problems), done.unsafe_ptr(),
                    grid_dim=(problems + SB_TPB - 1) // SB_TPB, block_dim=SB_TPB,
                )
                ctx.enqueue_copy(dst_buf=hdone, src_buf=done)
                ctx.synchronize()
                var hd = _host(hdone)
                for c in range(problems):
                    if ld(hd, c) != Float32(epoch + 1):
                        raise Error(
                            "x_linear SGD: the shuffle launch of epoch " + String(epoch) + " (swaps "
                            + String(tt0) + ".." + String(tt1) + ") did not finish problem " + String(c)
                            + " (on Apple: a launch the system aborted); the fit is refused"
                        )
                tt0 = tt1
            ctx.enqueue_copy(dst_buf=hord, src_buf=order)
            ctx.synchronize()
            _check_perm(_host_i(hord), problems, n, seen, epoch)
        var r0 = 0
        while r0 < n:
            var r1 = min(n, r0 + rows_per)
            if into_b:
                wb.enqueue_fill(_poison())
                qb.enqueue_fill(_poison())
                sbuf.enqueue_fill(_poison())
                ctx.enqueue_function[sb_rows_kernel](
                    dx.unsafe_ptr(), dy.unsafe_ptr(), ys.unsafe_ptr(), Int32(n), Int32(d), dip.unsafe_ptr(),
                    dfp.unsafe_ptr(), order.unsafe_ptr(), wa.unsafe_ptr(), wb.unsafe_ptr(), qa.unsafe_ptr(),
                    qb.unsafe_ptr(), sa.unsafe_ptr(), sbuf.unsafe_ptr(), Int32(epoch), Int32(r0), Int32(r1),
                    Int32(form), grid_dim=problems, block_dim=SB_ROW_TPB,
                )
                ctx.enqueue_copy(dst_buf=hw, src_buf=wb)
                ctx.enqueue_copy(dst_buf=hq, src_buf=qb)
                ctx.enqueue_copy(dst_buf=hs, src_buf=sbuf)
            else:
                wa.enqueue_fill(_poison())
                qa.enqueue_fill(_poison())
                sa.enqueue_fill(_poison())
                ctx.enqueue_function[sb_rows_kernel](
                    dx.unsafe_ptr(), dy.unsafe_ptr(), ys.unsafe_ptr(), Int32(n), Int32(d), dip.unsafe_ptr(),
                    dfp.unsafe_ptr(), order.unsafe_ptr(), wb.unsafe_ptr(), wa.unsafe_ptr(), qb.unsafe_ptr(),
                    qa.unsafe_ptr(), sbuf.unsafe_ptr(), sa.unsafe_ptr(), Int32(epoch), Int32(r0), Int32(r1),
                    Int32(form), grid_dim=problems, block_dim=SB_ROW_TPB,
                )
                ctx.enqueue_copy(dst_buf=hw, src_buf=wa)
                ctx.enqueue_copy(dst_buf=hq, src_buf=qa)
                ctx.enqueue_copy(dst_buf=hs, src_buf=sa)
            ctx.synchronize()
            _check(_host(hw), pd, "weights")
            _check(_host(hq), pd, "L1 history")
            _check(_host(hs), ps, "state")
            for c in range(problems):
                if ld(_host(hs), c * SGD_ST + ST_DONE) != SGD_DONE_WORD:
                    raise Error("x_linear SGD: a bounded launch did not finish problem " + String(c) + "; the fit is refused")
            into_b = not into_b
            first = False
            r0 = r1
        var all_stopped = True
        for c in range(problems):
            if Int(bitcast[DType.int32](ld(_host(hs), c * SGD_ST + ST_STOP))) == 0:
                all_stopped = False
        if all_stopped:
            break
    # the result, as sgd_fit leaves it: coef, intercepts, epochs, status
    var hwp = _host(hw)
    var hsp = _host(hs)
    for e in range(pd):
        st(res, e, ld(hwp, e))
    var max_epochs = 0
    var status = 0
    for c in range(problems):
        var stop = Int(bitcast[DType.int32](ld(hsp, c * SGD_ST + ST_STOP)))
        st(res, pd + c, ld(hsp, c * SGD_ST + ST_INTERCEPT))
        if stop == 2:
            status = -1
        else:
            var ep = Int(bitcast[DType.int32](ld(hsp, c * SGD_ST + ST_EPOCHS)))
            if ep > max_epochs:
                max_epochs = ep
    st(res, pd + problems, i2f(max_epochs))
    st(res, pd + problems + 1, i2f(status))
    _ = hip^
    _ = hfp^
    _ = seen^
