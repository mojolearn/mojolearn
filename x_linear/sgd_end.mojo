# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SGD's epoch end on the device (lane cgr4-device-optim, 2026-10-03).

Before this lane every epoch brought the weights home (d words a problem)
and the host ran sgd_one's / sgd_mb_one's epoch-end statements: the
floating-point overflow check over the weights, the mean objective with
the penalty, the no-improvement count, the adaptive rate and the stop.
Here one block a problem runs them on the device, from a stop state
double-buffered by epoch parity (the kernel reads parity `par`, writes
1 - par: a Metal witness rerun is idempotent), and writes the next epoch's
rate where the epoch kernels read it. Per epoch the host reads two words
(mini-batch: stop, failed) or one (per-sample: the live problem count).
The finite check is an OR over the weights (order-free); the mini-batch
penalty is `mb_penalty_t`, the vfold order the host column folds too.

State words, mini-batch (one problem): parity p at 4p: best, no-improve
(bits), eta, 0; flags at 8: stop, failed.
Per-sample (P problems): parity p, problem c at (p P + c) * SGD_END_ST:
best, no-improve (bits), eta, active (bits), epochs (bits), failed (bits).
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.memory import bitcast, stack_allocation
from std.atomic import Atomic
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from x_linear.ops import FP, IP, ld, st, fa, fs, fm, fd, fabs, i2f, perm_at, perm_key
from x_linear.team import team_at, team_barrier
from x_linear.sgd import mb_penalty_t, ws_mul, oc_offset, LR_ADAPTIVE
from x_linear.witness import witness_end

comptime SGD_END_TPB = 256
comptime SGD_END_ST = 6
comptime SGD_MB_FLAGS = 8
comptime SGD_MB_WORDS = 10


@always_inline
def _bits(v: Int) -> Float32:
    return bitcast[DType.float32](Int32(v))


@always_inline
def _int(v: Float32) -> Int:
    return Int(bitcast[DType.int32](v))


@always_inline
def _ok(v: Float32) -> Bool:
    return v == v and fabs(v) < Float32(3.0e38)


def sgd_mb_end_kernel(
    w: FP, bias: FP, obj: FP, stt: FP, cf: FP, parts: FP, d: Int32, n: Int32, alpha: Float32, l1r: Float32,
    penalty: Int32, tol: Float32, nic: Int32, lr: Int32, need_obj: Int32, one_class: Int32, par: Int32,
    wf: IP, woff: Int32, nonce: Int32,
):
    """One block: sgd_mb_one's epoch end. cf[4] gets the next epoch's rate."""
    var tid = Int(thread_idx.x)
    var t = team_at(tid, Int(block_dim.x), parts, 0, 0, 0)
    var dd = Int(d)
    var bad = stack_allocation[1, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    if tid == 0:
        bad[0] = 0
    barrier()
    for j in range(tid, dd, Int(block_dim.x)):
        if not _ok(ld(w, j)):
            bad[0] = 1
    barrier()
    var a = 4 * Int(par)
    var bo = 4 * (1 - Int(par))
    var best = ld(stt, a)
    var no_improve = _int(ld(stt, a + 1))
    var eta = ld(stt, a + 2)
    var b = ld(bias, 0)
    var stop = 0
    var failed = 0
    if not (_ok(b) and bad[0] == 0):
        failed = 1
        stop = 1
    elif need_obj != 0:
        var pen = mb_penalty_t(t, w, 0, dd, alpha, l1r, Int(penalty), parts)
        var mean_obj = fa(fd(ld(obj, 0), i2f(Int(n))), pen)
        if one_class != 0:
            mean_obj = fa(mean_obj, fm(alpha, b))
        if mean_obj > fs(best, tol):
            no_improve += 1
        else:
            no_improve = 0
        if mean_obj < best:
            best = mean_obj
        if no_improve >= Int(nic):
            if Int(lr) == LR_ADAPTIVE and eta > Float32(1e-6):
                eta = fd(eta, Float32(5))
                no_improve = 0
            else:
                stop = 1
    if tid == 0:
        st(stt, bo, best)
        st(stt, bo + 1, _bits(no_improve))
        st(stt, bo + 2, eta)
        st(cf, 4, eta)
        st(stt, SGD_MB_FLAGS, Float32(stop))
        st(stt, SGD_MB_FLAGS + 1, Float32(failed))
    witness_end(wf, woff, nonce)


def sgd_mb_res_kernel(w: FP, bias: FP, res: FP, c: Int32, d: Int32, problems: Int32, one_class: Int32,
                      failed: Int32):
    """Problem c's result words: coef row c, the intercept (one-class: the
    offset 1 - intercept); zeros on a non-finite fit."""
    var j = Int(block_idx.x) * SGD_END_TPB + Int(thread_idx.x)
    var dd = Int(d)
    var cc = Int(c)
    if j < dd:
        st(res, cc * dd + j, Float32(0) if failed != 0 else ld(w, j))
    elif j == dd:
        var b = ld(bias, 0)
        var v = (fs(Float32(1), b) if one_class != 0 else b) if failed == 0 else Float32(0)
        st(res, Int(problems) * dd + cc, v)


def sgd_ps_end_kernel(
    w: FP, ps: FP, act: IP, pf: FP, stt: FP, live: IP, d: Int32, n: Int32, problems: Int32, tol: Float32,
    nic: Int32, lr: Int32, need_obj: Int32, par: Int32, epoch: Int32, ps_st: Int32,
    wf: IP, woff: Int32, nonce: Int32,
):
    """Block c: sgd_one's epoch end for problem c. pf[3c] gets its next rate,
    act[c] its activity; a live problem adds one to live[0]."""
    var tid = Int(thread_idx.x)
    var c = Int(block_idx.x)
    var dd = Int(d)
    var pp = Int(problems)
    var sst = Int(ps_st)
    var a = (Int(par) * pp + c) * SGD_END_ST
    var bo = ((1 - Int(par)) * pp + c) * SGD_END_ST
    var active = _int(ld(stt, a + 3))
    var bad = stack_allocation[1, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    if tid == 0:
        bad[0] = 0
    barrier()
    if active != 0:
        var hi = ld(ps, sst * c + 4)
        var lo = ld(ps, sst * c + 5)
        for j in range(tid, dd, Int(block_dim.x)):
            if not _ok(ws_mul(ld(w, c * dd + j), hi, lo)):
                bad[0] = 1
    barrier()
    var best = ld(stt, a)
    var no_imp = _int(ld(stt, a + 1))
    var eta = ld(stt, a + 2)
    var epochs = _int(ld(stt, a + 4))
    var failed = _int(ld(stt, a + 5))
    if active != 0:
        epochs = Int(epoch) + 1
        if not (_ok(ld(ps, sst * c)) and bad[0] == 0):
            failed = 1
            active = 0
        else:
            var mean_obj = fd(ld(ps, sst * c + 2), i2f(Int(n)))
            if need_obj != 0 and mean_obj > fs(best, tol):
                no_imp += 1
            else:
                no_imp = 0
            if mean_obj < best:
                best = mean_obj
            if no_imp >= Int(nic):
                if Int(lr) == LR_ADAPTIVE and eta > Float32(1e-6):
                    eta = fd(eta, Float32(5))
                    no_imp = 0
                else:
                    active = 0
    if tid == 0:
        st(stt, bo, best)
        st(stt, bo + 1, _bits(no_imp))
        st(stt, bo + 2, eta)
        st(stt, bo + 3, _bits(active))
        st(stt, bo + 4, _bits(epochs))
        st(stt, bo + 5, _bits(failed))
        st(pf, 3 * c, eta)
        act.unsafe_store(c, Int32(active))
        if active != 0:
            _ = Atomic[DType.int32].fetch_add(live, Int32(1))
    witness_end(wf, woff, nonce)


def sgd_ps_res_kernel(w: FP, ps: FP, stt: FP, res: FP, d: Int32, problems: Int32, one_class: Int32, par: Int32,
                      ps_st: Int32):
    """The result words from the final state (parity par): coef = wscale * v,
    the intercept (one-class: `oc_offset`), zeros for a failed problem; the
    epochs maximum and the status by thread 0 over the P problems."""
    var q = Int(block_idx.x) * SGD_END_TPB + Int(thread_idx.x)
    var dd = Int(d)
    var pp = Int(problems)
    var sst = Int(ps_st)
    if q < pp * dd:
        var c = q // dd
        var f = _int(ld(stt, (Int(par) * pp + c) * SGD_END_ST + 5))
        st(res, q, Float32(0) if f != 0 else ws_mul(ld(w, q), ld(ps, sst * c + 4), ld(ps, sst * c + 5)))
    elif q < pp * dd + pp:
        var c = q - pp * dd
        var f = _int(ld(stt, (Int(par) * pp + c) * SGD_END_ST + 5))
        var v = Float32(0)
        if f == 0:
            v = oc_offset(ld(ps, sst * c), ld(ps, sst * c + 3)) if one_class != 0 else ld(ps, sst * c)
        st(res, q, v)
    elif q == pp * dd + pp:
        var mx = 0
        var status = 0
        for c in range(pp):
            var base = (Int(par) * pp + c) * SGD_END_ST
            if _int(ld(stt, base + 5)) != 0:
                status = -1
            else:
                mx = max(mx, _int(ld(stt, base + 4)))
        st(res, q, i2f(mx))
        st(res, q + 1, i2f(status))


# ------------------------------------------------ the epoch order and the targets on the device
def sgd_ys_kernel(y: FP, ys: FP, idx: IP, n: Int32, k: Int32, c: Int32, one_class: Int32):
    """Problem c's +-1 targets (sgd_fit's statements) and the identity order,
    a thread a row."""
    var i = Int(block_idx.x) * SGD_END_TPB + Int(thread_idx.x)
    if i < Int(n):
        var v: Float32
        if one_class != 0:
            v = Float32(1)  # sgd_fit's one-class target (y is not read)
        else:
            var yv = ld(y, i)
            if Int(k) == 0:
                v = yv
            elif Int(k) == 2:
                v = Float32(1) if yv == Float32(1) else Float32(-1)
            else:
                v = Float32(1) if yv == i2f(Int(c)) else Float32(-1)
        st(ys, i, v)
        idx.unsafe_store(i, Int32(i))


def sgd_iota_kernel(idx: IP, n: Int32, problems: Int32):
    """Every problem's identity order: idx[c n + i] = i."""
    var q = Int(block_idx.x) * SGD_END_TPB + Int(thread_idx.x)
    var nn = Int(n)
    if q < nn * Int(problems):
        idx.unsafe_store(q, Int32(q % nn))


def sgd_perm_kernel(idx: IP, n: Int32, seed_lo: Int32, seed_hi: Int32, epoch: Int32, c0: Int32, problems: Int32):
    """The epoch order of problems c0 .. c0 + problems - 1 (seed + 1000003 c),
    a thread a row: `perm_at` (x_linear/ops.mojo, DEVIATION 5004 revised),
    the host column's `perm_fill` order."""
    var q = Int(block_idx.x) * SGD_END_TPB + Int(thread_idx.x)
    var nn = Int(n)
    if q < nn * Int(problems):
        var cl = q // nn
        var i = q - cl * nn
        var seed = (UInt64(UInt32(seed_hi)) << 32) | UInt64(UInt32(seed_lo))
        var sc = seed + UInt64(1000003) * UInt64(Int(c0) + cl)
        idx.unsafe_store(q, Int32(perm_at(i, nn, perm_key(sc, Int(epoch)))))
