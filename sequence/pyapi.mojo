# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE ADDRESS CONTRACT of the sequence lane's bindings, written once over
`Exec`: `bindings/_mojolearn_x_sequence.mojo` calls these with a
`DeviceExec`, `bindings/_mojolearn_x_sequence_host.mojo` with a `HostExec`.
Every buffer is a caller-owned C-contiguous array handed over by address;
nothing is retained after the call."""
from std.python import PythonObject

from std.math import sqrt
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_div, identical_mul, identical_mul_add, identical_pow64, identical_sqrt
from sequence.exec import Exec
from sequence.ops import FP, OP_STL, OP_AF_ALPHA, OP_AF_BLK_SUMSQ, OP_AF_ROW, OP_AF_COL, OP_AF_RMEAN, OP_AF_UPDATE_MAT, OP_AF_VEC, OP_AF_DENOM, OP_AF_APPLY, OP_SEG_SUMSQ, OP_CHUNK_SUMSQ, OP_LAMB_UPD, OP_LAMB_RATIO, OP_LAMB_APPLY, OP_LN_FWD, OP_LN_BWD_X, OP_LN_BWD_W, OP_THETA, OP_CROSTON, OP_ETS, OP_GARCH, OP_PROPHET_FEATURES, OP_PROPHET_FIT, OP_PROPHET_PREDICT, OP_PROPHET_FG_PART, OP_PROPHET_FG_SUM, OP_MOE_ROUTE, OP_MOE_HIDDEN, OP_MOE_OUT, OP_DIVS, OP_FILL, OP_VAR_DESIGN, OP_COLSCALE, OP_CHOLSOLVE, OP_ROWSCALE, OP_VAR_FORECAST, OP_SUB, OP_SCALE, Args, OPT_ADAGRAD, OPT_ADAM, OPT_ADAMW, OPT_RMSPROP, OPT_SGD, OPT_LION, OPT_SK_ADAM, OPT_SK_SGD, OPT_NADAM
from sequence.recurrent import gemm
from sequence.mlp_fit import MLPNet, mlp_fit, mlp_predict
from sequence.recurrent import TASK_CE, TASK_MSE, Net, OptConfig, OptState, opt_scalars, opt_step, rnn_fit, rnn_predict
from sequence.ets import ets_scratch
from sequence.garch import GARCH_SNAP
from sequence.moe_tiled import TILE_P, TILE_Q
from sequence.prophet import MEM, ProphetData, _dot, _fg_prior
from sequence.prophet import div as _pdiv
from sequence.ops import add as p_add, fma3 as p_fma3, ld as p_ld, mul as p_mul, st as p_st, sub as p_sub
from sequence.ops import sumsq_fold as _sumsq_fold_host
from sequence.adafactor import AF_NORM_BLOCK, af_alpha_tail, af_denom_tail
from std.sys.compile import is_defined
from std.os import getenv as _getenv_seq


def fptr(addr: PythonObject, what: String) raises -> FP:
    var a = Int(py=addr)
    if a == 0:
        raise Error("sequence: null float32 buffer for " + what)
    return FP(unsafe_from_address=a)


def iptr(addr: PythonObject, what: String) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    var a = Int(py=addr)
    if a == 0:
        raise Error("sequence: null int32 buffer for " + what)
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=a)


def ival(p: PythonObject, i: Int) raises -> Int:
    return Int(py=p[i])


def fval(p: PythonObject, i: Int) raises -> Float32:
    return Float32(Float64(py=p[i]))


def net_of(ip: PythonObject) raises -> Net:
    """ip[0:5] = cell, D, H, L, O."""
    var net = Net(ival(ip, 0), ival(ip, 1), ival(ip, 2), ival(ip, 3), ival(ip, 4))
    if net.cell < 0 or net.cell > 3:
        raise Error("sequence: cell must be 0 (rnn tanh), 1 (rnn relu), 2 (lstm) or 3 (gru)")
    if net.D < 1 or net.H < 1 or net.L < 1 or net.O < 1:
        raise Error("sequence: input size, hidden size, layers and outputs must be >= 1")
    return net


def opt_of(ip: PythonObject, at: Int, fp: PythonObject, fat: Int) raises -> OptConfig:
    var kind = ival(ip, at)
    if kind < OPT_SGD or kind > OPT_NADAM or kind == OPT_SK_ADAM or kind == OPT_SK_SGD:
        raise Error("sequence: unknown optimizer kind " + String(kind))
    return OptConfig(kind, ival(ip, at + 1), fval(fp, fat), fval(fp, fat + 1), fval(fp, fat + 2),
                     fval(fp, fat + 3), fval(fp, fat + 4))


def rnn_fit_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """addrs = [X, y, order (int32), steps (int32), params (in/out), losses, lrs];
    ip = [cell, D, H, L, O, task, N, T, n_order, n_steps, opt_kind, opt_flags];
    fp = [f1, f2, eps, weight_decay, f7, initial_accumulator]."""
    if len(addrs) != 7 or len(ip) != 12 or len(fp) != 6:
        raise Error("rnn_fit: requires 7 addresses, 12 integer and 6 float parameters")
    var x_addr = addrs[0]
    var y_addr = addrs[1]
    var order_addr = addrs[2]
    var steps_addr = addrs[3]
    var p_addr = addrs[4]
    var losses_addr = addrs[5]
    var lrs_addr = addrs[6]
    var net = net_of(ip)
    var task = ival(ip, 5)
    var N = ival(ip, 6)
    var T = ival(ip, 7)
    var n_order = ival(ip, 8)
    var n_steps = ival(ip, 9)
    if task != TASK_MSE and task != TASK_CE:
        raise Error("rnn_fit: task must be 0 (mse) or 1 (cross-entropy)")
    if task == TASK_CE and net.O < 2:
        raise Error("rnn_fit: cross-entropy needs at least two classes")
    # The order arrives as int32 (it was float32, which capped the whole
    # schedule, epochs x N, below 2^24: 16 epochs at 1M rows). Its length is
    # bounded only by the int32 step offsets; the sample indices it holds
    # stay below N < 2^24, so the float32 copy the device gathers with is
    # exact and the bits are the float32 order's.
    if N < 1 or T < 1 or n_steps < 1 or n_order < 1 or N >= 16777216 or n_order >= 2147483647:
        raise Error("rnn_fit: N, T, steps and order must be >= 1, N < 2^24 and order < 2^31 - 1")
    var cfg = opt_of(ip, 10, fp, 0)
    var steps = iptr(steps_addr, "steps")
    for k in range(n_steps):
        var off = Int(steps.unsafe_load(2 * k))
        var cnt = Int(steps.unsafe_load(2 * k + 1))
        if cnt < 1 or off < 0 or off + cnt > n_order:
            raise Error("rnn_fit: step " + String(k) + " reads outside the order")
    var order_i = iptr(order_addr, "order")
    var order_f = List[Float32](capacity=n_order)
    for i in range(n_order):
        var v = Int(order_i.unsafe_load(i))
        if v < 0 or v >= N:
            raise Error("rnn_fit: order holds a value that is not a sample index")
        order_f.append(Float32(v))
    var order = FP(unsafe_from_address=Int(order_f.unsafe_ptr()))
    if task == TASK_CE:
        var y = fptr(y_addr, "y")
        for i in range(N):
            var v = Int(y.unsafe_load(i))
            if v < 0 or v >= net.O or Float32(v) != y.unsafe_load(i):
                raise Error("rnn_fit: a label is not a class index below the class count")
    rnn_fit(ex, net, task, fptr(x_addr, "X"), fptr(y_addr, "y"), N, T, order, n_order,
            steps, n_steps, fptr(p_addr, "params"), fptr(losses_addr, "losses"),
            fptr(lrs_addr, "lrs"), cfg, fval(fp, 5))
    _ = order_f^
    return PythonObject(net.n_params())


def rnn_predict_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """addrs = [X, params, out, seq (ignored unless want_seq)];
    ip = [cell, D, H, L, O, task, N, T, chunk, want_seq]."""
    if len(addrs) != 4 or len(ip) != 10:
        raise Error("rnn_predict: requires 4 addresses and 10 integer parameters")
    var x_addr = addrs[0]
    var p_addr = addrs[1]
    var out_addr = addrs[2]
    var seq_addr = addrs[3]
    var net = net_of(ip)
    var task = ival(ip, 5)
    var N = ival(ip, 6)
    var T = ival(ip, 7)
    var chunk = ival(ip, 8)
    var want = ival(ip, 9) != 0
    if N < 1 or T < 1 or chunk < 1 or N >= 16777216:
        raise Error("rnn_predict: N, T and chunk must be >= 1 and N < 2^24")
    var seq = fptr(seq_addr, "seq") if want else fptr(out_addr, "out")
    rnn_predict(ex, net, task, fptr(x_addr, "X"), N, T, fptr(p_addr, "params"),
                fptr(out_addr, "out"), seq, want, chunk)
    return PythonObject(N * net.O)


def rnn_n_params_py(ip: PythonObject) raises -> PythonObject:
    return PythonObject(net_of(ip).n_params())


def opt_slots(cfg: OptConfig) -> Tuple[Bool, Bool, Bool]:
    """Which of state1, state2, state3 `sequence/ops.mojo::op_opt` reads or
    writes for this configuration; the others are neither uploaded nor
    downloaded (lane py-sequence)."""
    var k = cfg.kind
    if k == OPT_SGD:
        return (cfg.f1 != Float32(0.0), False, False)
    if k == OPT_RMSPROP:
        return (cfg.f7 > Float32(0.0), True, (cfg.flags & 1) != 0)
    if k == OPT_ADAGRAD:
        return (False, True, False)
    if k == OPT_LION:
        return (True, False, False)
    # Adam, AdamW, Adamax, NAdam
    return (True, True, False)


def opt_step_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """One optimizer step over a flat float32 buffer, in place.
    addrs = [params, grads, state1, state2, state3 (, scalars)];
    ip = [n, kind, flags, t (, t0)]; fp = [lr, f1, f2, eps, weight_decay, f7].
    `scalars` (float32[3]: beta1^t0, beta2^t0, NAdam's mu product) is the
    host scalars' running state after step t0 (t0 = 0: not yet run, the
    buffer is ignored); the call advances it through step t and writes it
    back, so a caller that passes it every step pays O(1) per step. Without
    it, or with t0 < t - 1, the steps t0 + 1 .. t - 1 are replayed (the
    state is a function of t alone, so both give the same bits). Only the
    state slots the kind uses cross the host-device boundary."""
    if len(addrs) < 5 or len(addrs) > 6 or len(ip) != len(addrs) - 1 or len(fp) != 6:
        raise Error("optimizer_step: requires 5 (or 6) addresses, 4 (or 5) integer and 6 float parameters")
    var p_addr = addrs[0]
    var g_addr = addrs[1]
    var n = ival(ip, 0)
    var t = ival(ip, 3)
    if n < 1 or t < 1:
        raise Error("optimizer_step: n and the one-based step t must be >= 1")
    var cfg = opt_of(ip, 1, fp, 1)
    var st = OptState()
    var k0 = 1
    var carry = len(addrs) == 6
    var sc = fptr(addrs[5], "scalars") if carry else fptr(p_addr, "params")
    if carry:
        var t0 = ival(ip, 4)
        if t0 < 0 or t0 >= t:
            raise Error("optimizer_step: the scalars' step t0 must satisfy 0 <= t0 < t")
        if t0 > 0:
            st.pw1 = sc.unsafe_load(0)
            st.pw2 = sc.unsafe_load(1)
            st.mu_prod = sc.unsafe_load(2)
            k0 = t0 + 1
    for k in range(k0, t):
        _ = opt_scalars(cfg, st, k, fval(fp, 0))
    var used = opt_slots(cfg)
    var hp = fptr(p_addr, "params")
    # `bind`: the host column updates the caller's arrays in place (OP_OPT
    # never writes the gradient slot); the device uploads copies. A slot
    # the kind never touches is not bound: the element body never
    # dereferences it, so it is handed the params buffer.
    var P = ex.bind(hp, n)
    var G = ex.bind(fptr(g_addr, "grads"), n)
    var h1 = fptr(addrs[2], "state1") if used[0] else hp
    var h2 = fptr(addrs[3], "state2") if used[1] else hp
    var h3 = fptr(addrs[4], "state3") if used[2] else hp
    var s1 = ex.bind(h1, n) if used[0] else P
    var s2 = ex.bind(h2, n) if used[1] else P
    var s3 = ex.bind(h3, n) if used[2] else P
    opt_step(ex, cfg, st, t, fval(fp, 0), P, G, s1, s2, s3, n)
    # the copies share one wait (apple2)
    ex.download_async(hp, P, n)
    if used[0]:
        ex.download_async(h1, s1, n)
    if used[1]:
        ex.download_async(h2, s2, n)
    if used[2]:
        ex.download_async(h3, s3, n)
    ex.sync()
    if carry:
        sc.unsafe_store(0, st.pw1)
        sc.unsafe_store(1, st.pw2)
        sc.unsafe_store(2, st.mu_prod)
    return PythonObject(n)


def stl_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """STL over a batch of series (`sequence/stl.mojo`).
    addrs = [y (B, n), season, trend, weights, resid (each B, n, written)];
    ip = [B, n, period, seasonal, trend, low_pass, degrees (s + 2 t + 4 l),
    seasonal_jump, trend_jump, low_pass_jump, inner_iter, outer_iter]."""
    if len(addrs) != 5 or len(ip) != 12:
        raise Error("stl: requires 5 addresses and 12 integer parameters")
    var B = ival(ip, 0)
    var n = ival(ip, 1)
    var np_ = ival(ip, 2)
    if B < 1 or n < 1:
        raise Error("stl: at least one series of at least one observation")
    if np_ < 2:
        raise Error("stl: period must be a positive integer >= 2")
    for k in range(3, 6):
        var v = ival(ip, k)
        if v < 3 or v % 2 == 0:
            raise Error("stl: seasonal, trend and low_pass must be odd integers >= 3")
    if ival(ip, 4) <= np_ or ival(ip, 5) <= np_:
        raise Error("stl: trend and low_pass must exceed the period")
    var degs = ival(ip, 6)
    if degs < 0 or degs > 7:
        raise Error("stl: every degree must be 0 or 1")
    for k in range(7, 10):
        if ival(ip, k) < 1:
            raise Error("stl: every jump must be a positive integer")
    if ival(ip, 10) < 1 or ival(ip, 11) < 0:
        raise Error("stl: inner_iter must be >= 1 and outer_iter >= 0")
    var n2 = n + 2 * np_
    var y = ex.alloc(B * n)
    ex.upload(y, fptr(addrs[0], "y"), B * n)
    var outs = List[FP]()
    for _ in range(4):
        outs.append(ex.alloc(B * n))
    var work = ex.alloc(B * 5 * n2)
    var sortbuf = ex.alloc(B * n)
    var a = Args()
    a.p0 = y
    a.p1 = outs[0]
    a.p2 = outs[1]
    a.p3 = outs[2]
    a.p4 = outs[3]
    a.p5 = work
    a.p6 = sortbuf
    a.i0 = n
    a.i1 = np_
    a.i2 = ival(ip, 3)
    a.i3 = ival(ip, 4)
    a.i4 = ival(ip, 5)
    a.i5 = degs
    a.i6 = ival(ip, 7)
    a.i7 = ival(ip, 8)
    a.i8 = ival(ip, 9)
    a.i9 = ival(ip, 10)
    a.i10 = ival(ip, 11)
    ex.launch[OP_STL](a, B)
    ex.sync()
    for k in range(4):
        ex.download(fptr(addrs[k + 1], "output"), outs[k], B * n)
    return PythonObject(B * n)


def var_fit_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """VAR(p) by OLS (`sequence/vecar.mojo`). addrs = [y (n, K), params (m, K),
    sigma_u (K, K), resid (n - p, K)], each output written; ip = [n, K, p,
    k_trend]. Returns 0, or 1 + the design column whose Cholesky pivot was
    not positive (nothing is written then)."""
    if len(addrs) != 4 or len(ip) != 4:
        raise Error("var_fit: requires 4 addresses and 4 integer parameters")
    var n = ival(ip, 0)
    var K = ival(ip, 1)
    var p = ival(ip, 2)
    var kt = ival(ip, 3)
    if K < 1 or p < 1 or kt < 0 or kt > 1:
        raise Error("var_fit: K >= 1, p >= 1 and k_trend 0 or 1")
    var R = n - p
    var m = kt + K * p
    if R - m < 1:
        raise Error("var_fit: too few observations for the lag order (need n - p > k_trend + K p)")
    var y = ex.alloc(n * K)
    ex.upload(y, fptr(addrs[0], "y"), n * K)
    var Z = ex.alloc(R * m)
    var Ys = ex.alloc(R * K)
    var sc = ex.alloc(m)
    var G = ex.alloc(m * m)
    var Bm = ex.alloc(m * K)
    var F = ex.alloc(R * K)
    var Rs = ex.alloc(R * K)
    var S = ex.alloc(K * K)
    var status = ex.alloc(1)
    var a = Args()
    a.p0 = y
    a.p1 = Z
    a.p2 = Ys
    a.i0 = K
    a.i1 = p
    a.i2 = kt
    a.i3 = m
    ex.launch[OP_VAR_DESIGN](a, R * m)
    var b = Args()
    b.p0 = Z
    b.p1 = sc
    b.i0 = R
    b.i1 = m
    ex.launch[OP_COLSCALE](b, m)
    gemm(ex, Z, Z, G, m, m, R, 1, m, m, 1, False, m)
    gemm(ex, Z, Ys, Bm, m, K, R, 1, m, K, 1, False, K)
    var c = Args()
    c.p0 = G
    c.p1 = Bm
    c.p2 = status
    c.i0 = m
    c.i1 = K
    ex.launch[OP_CHOLSOLVE](c, 1)
    ex.sync()
    var st = List[Float32](length=1, fill=Float32(0.0))
    ex.download(FP(unsafe_from_address=Int(st.unsafe_ptr())), status, 1)
    var code = Int(st[0])
    if code != 0:
        return PythonObject(code)
    gemm(ex, Z, Bm, F, R, K, m, m, 1, K, 1, False, K)
    var d = Args()
    d.p0 = Ys
    d.p1 = F
    d.p2 = Rs
    ex.launch[OP_SUB](d, R * K)
    gemm(ex, Rs, Rs, S, K, K, R, 1, K, K, 1, False, K)
    var e = Args()
    e.p0 = S
    e.f0 = Float32(1.0) / Float32(R - m)
    ex.launch[OP_SCALE](e, K * K)
    var f = Args()
    f.p0 = Bm
    f.p1 = sc
    f.i1 = K
    ex.launch[OP_ROWSCALE](f, m * K)
    ex.sync()
    ex.download(fptr(addrs[1], "params"), Bm, m * K)
    ex.download(fptr(addrs[2], "sigma_u"), S, K * K)
    ex.download(fptr(addrs[3], "resid"), Rs, R * K)
    return PythonObject(0)


def var_forecast_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """addrs = [y_last (p, K), params (m, K), out (h, K)]; ip = [K, p, k_trend, h]."""
    if len(addrs) != 3 or len(ip) != 4:
        raise Error("var_forecast: requires 3 addresses and 4 integer parameters")
    var K = ival(ip, 0)
    var p = ival(ip, 1)
    var kt = ival(ip, 2)
    var h = ival(ip, 3)
    if K < 1 or p < 1 or kt < 0 or kt > 1 or h < 1:
        raise Error("var_forecast: K, p, h >= 1 and k_trend 0 or 1")
    var m = kt + K * p
    var y = ex.alloc(p * K)
    ex.upload(y, fptr(addrs[0], "y"), p * K)
    var P = ex.alloc(m * K)
    ex.upload(P, fptr(addrs[1], "params"), m * K)
    var out = ex.alloc(h * K)
    var a = Args()
    a.p0 = y
    a.p1 = P
    a.p2 = out
    a.i0 = K
    a.i1 = p
    a.i2 = kt
    a.i3 = h
    ex.launch[OP_VAR_FORECAST](a, 1)
    ex.sync()
    ex.download(fptr(addrs[2], "out"), out, h * K)
    return PythonObject(h * K)


def _mlp_net(ip: PythonObject, at: Int, D: Int, O: Int, act: Int, out_act: Int) raises -> MLPNet:
    var nh = ival(ip, at)
    var sizes = List[Int]()
    sizes.append(D)
    for k in range(nh):
        var h = ival(ip, at + 1 + k)
        if h < 1:
            raise Error("mlp: every hidden layer needs at least one unit")
        sizes.append(h)
    sizes.append(O)
    if act < 0 or act > 3 or out_act < 0 or out_act > 4:
        raise Error("mlp: unknown activation code")
    return MLPNet(sizes^, act, out_act)


def mlp_fit_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """addrs = [X (N, D), Y (N, O), params (in/out), loss_curve (max_iter, out)];
    ip = [N, D, O, act, out_act, loss, solver, lr_schedule, nesterov, batch,
    max_iter, shuffle, seed, n_iter_no_change, n_hidden, h_1, ..., h_n];
    fp = [lr, beta1, beta2, eps, momentum, power_t, alpha, tol]. Returns n_iter."""
    if len(addrs) != 4 or len(fp) != 8 or len(ip) < 15:
        raise Error("mlp_fit: requires 4 addresses, >= 15 integer and 8 float parameters")
    var N = ival(ip, 0)
    var D = ival(ip, 1)
    var O = ival(ip, 2)
    if N < 1 or D < 1 or O < 1 or N >= 16777216:
        raise Error("mlp_fit: N, D, O >= 1 and N < 2^24")
    var net = _mlp_net(ip, 14, D, O, ival(ip, 3), ival(ip, 4))
    if len(ip) != 15 + len(net.sizes) - 2:
        raise Error("mlp_fit: the hidden layer count does not match the sizes given")
    var batch = ival(ip, 9)
    var max_iter = ival(ip, 10)
    if batch < 1 or max_iter < 1:
        raise Error("mlp_fit: batch_size and max_iter must be >= 1")
    var n_iter = mlp_fit(ex, net, fptr(addrs[0], "X"), fptr(addrs[1], "Y"), N, fptr(addrs[2], "params"),
                         fptr(addrs[3], "loss_curve"), ival(ip, 5), ival(ip, 6), ival(ip, 7), ival(ip, 8) != 0,
                         batch, max_iter, ival(ip, 11) != 0, UInt64(ival(ip, 12)), ival(ip, 13),
                         fval(fp, 0), fval(fp, 1), fval(fp, 2), fval(fp, 3), fval(fp, 4),
                         Float64(py=fp[5]), fval(fp, 6), Float64(py=fp[7]))
    return PythonObject(n_iter)


def mlp_predict_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """addrs = [X (N, D), params, out (N, O)]; ip = [N, D, O, act, out_act,
    chunk, n_hidden, h_1, ..., h_n]."""
    if len(addrs) != 3 or len(ip) < 7:
        raise Error("mlp_predict: requires 3 addresses and >= 7 integer parameters")
    var N = ival(ip, 0)
    var D = ival(ip, 1)
    var O = ival(ip, 2)
    var chunk = ival(ip, 5)
    if N < 1 or D < 1 or O < 1 or chunk < 1:
        raise Error("mlp_predict: N, D, O, chunk >= 1")
    var net = _mlp_net(ip, 6, D, O, ival(ip, 3), ival(ip, 4))
    mlp_predict(ex, net, fptr(addrs[0], "X"), N, fptr(addrs[1], "params"), fptr(addrs[2], "out"), chunk)
    return PythonObject(N * O)


#: FAST's two-pass norms (sequence/adafactor.mojo, op_chunk_sumsq) start at
#: this many elements; below it the one-thread fold is as quick.
comptime FAST_NORM_MIN = 65536
comptime _PARTS = 4096      # = sequence.adafactor.SUMSQ_THREADS


def _fast_norms() -> Bool:
    return GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL


#: lane/neural-net-experiment (2026-09-30): under IDENTICAL, Adafactor's two
#: whole-tensor norms are ONE THREAD's ascending chain of fmas (`sumsq_fold`),
#: which a GPU thread walks at memory latency: the bench board's adafactor
#: lane (one tensor of 16,777,216 values, 10 steps) took 10.7 s on an
#: MI325X against torch eager's 5.8 ms, about a second per step, most of it
#: those two chains. From HOST_FOLD_MIN values the SAME chain (the same
#: `sumsq_fold`, the same order, the same fmas, so the same bits: the host
#: binding runs this very code) is folded on the host: the parameter norm
#: from the caller's own bytes before they are uploaded, the update norm
#: from the update downloaded once. MOJOLEARN_SEQ_HOST_FOLD=0 keeps the
#: device chains (the A/B and the digest check). FAST keeps its partials.
comptime HOST_FOLD_MIN = 1 << 16


#: lane hr-adafactor (2026-10-02): IDENTICAL's whole-tensor norms are the
#: blocked order on the device (sequence/adafactor.mojo, AF_NORM_BLOCK):
#: block partials, one GPU thread per block, then the partials added
#: ascending; nothing leaves the device and no host fold runs. The A/B
#: define -D MOJOLEARN_SEQ_AF_SERIAL_NORM=1 restores main's order (the one
#: ascending chain, folded as before) for the measurement only; the final
#: commit of the lane deletes it with the old route.
comptime AF_SERIAL_NORM = is_defined["MOJOLEARN_SEQ_AF_SERIAL_NORM"]()


def _blocked_norms() -> Bool:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL or AF_SERIAL_NORM:
        return False
    return True


def _blk_sumsq[E: Exec](mut ex: E, src: FP, n: Int, parts: FP) raises -> Int:
    """IDENTICAL: the ceil(n / AF_NORM_BLOCK) block partials of src[0:n]'s
    sum of squares into parts[0:]; returns how many."""
    var nb = (n + AF_NORM_BLOCK - 1) // AF_NORM_BLOCK
    var c = Args()
    c.p0 = src
    c.p1 = parts
    c.i0 = n
    c.i1 = AF_NORM_BLOCK
    ex.launch[OP_AF_BLK_SUMSQ](c, nb)
    return nb


def _host_folds() -> Bool:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        return False
    comptime if not AF_SERIAL_NORM:
        return False
    return String(_getenv_seq("MOJOLEARN_SEQ_HOST_FOLD")) != "0"


def _chunk_sumsq[E: Exec](mut ex: E, src: FP, start: Int, n: Int, parts: FP, slot: Int) raises -> Int:
    """FAST: min(_PARTS, n) strided partial sums of squares of src[start:start+n]
    into parts[slot:]; returns how many."""
    var T = min(_PARTS, n)
    var c = Args()
    c.p0 = src
    c.p1 = parts
    c.i0 = n
    c.i1 = T
    c.i2 = slot
    c.i3 = start
    ex.launch[OP_CHUNK_SUMSQ](c, T)
    return T


def adafactor_step_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """One Adafactor step of ONE tensor, in place (`sequence/adafactor.mojo`).
    addrs = [param, grad, row_var (R) or variance (n), col_var (C; ignored for
    a vector)]; ip = [R, C, t] with C = 0 for a vector of R values;
    fp = [lr, beta2_decay, eps1, eps2, d, weight_decay]."""
    if len(addrs) != 4 or len(ip) != 3 or len(fp) != 6:
        raise Error("adafactor_step: requires 4 addresses, 3 integer and 6 float parameters")
    var R = ival(ip, 0)
    var C = ival(ip, 1)
    var t = ival(ip, 2)
    if R < 1 or C < 0 or t < 1:
        raise Error("adafactor_step: R >= 1, C >= 0 and the one-based step t >= 1")
    var n = R * C if C > 0 else R
    var lr = Float64(py=fp[0])
    var w = Float32(identical_pow64(Float64(t), Float64(py=fp[1])))
    var rho = Float32(min(lr, Float64(1.0) / sqrt(Float64(t))))
    var eps1 = fval(fp, 2)
    var eps1sq = ftz(identical_mul(eps1, eps1))
    var wd = fval(fp, 5)
    var P = ex.alloc(n)
    var G = ex.alloc(n)
    var S1 = ex.alloc(n if C == 0 else R)
    var S2 = ex.alloc(C if C > 0 else 1)
    var U = ex.alloc(n)
    var sc = ex.alloc(4)
    var hp = fptr(addrs[0], "param")
    var h1 = fptr(addrs[2], "row_var / variance")
    ex.upload(P, hp, n)
    ex.upload(G, fptr(addrs[1], "grad"), n)
    ex.upload(S1, h1, n if C == 0 else R)
    if C > 0:
        ex.upload(S2, fptr(addrs[3], "col_var"), C)
    var fast = _fast_norms() and n >= FAST_NORM_MIN
    var host_fold = _host_folds() and n >= HOST_FOLD_MIN
    var blocked = _blocked_norms() and n > AF_NORM_BLOCK
    var parts = ex.alloc(_PARTS if fast else ((n + AF_NORM_BLOCK - 1) // AF_NORM_BLOCK if blocked else 1))
    # the four scalars, mirrored on the host when the folds run there:
    # [unused, alpha, rmean (device only), denom]
    var hsc = List[Float32](length=4, fill=Float32(0.0))
    var a = Args()
    a.p0 = P
    a.p1 = sc
    a.i0 = n
    a.f0 = fval(fp, 3)
    a.f1 = rho
    if host_fold:
        # ||p||^2 from the caller's bytes (the same bytes P was uploaded
        # from), the same chain as op_af_alpha's; then its tail, into
        # hsc[1], and up into sc[1] before anything reads sc
        var ah = a
        ah.p1 = FP(unsafe_from_address=Int(hsc.unsafe_ptr()))
        af_alpha_tail(ah, _sumsq_fold_host(hp, 0, n, 1))
        ex.upload(sc + 1, FP(unsafe_from_address=Int(hsc.unsafe_ptr())) + 1, 1)
    else:
        if fast:
            a.p2 = parts
            a.i2 = _chunk_sumsq(ex, P, 0, n, parts, 0)
        elif blocked:
            a.p2 = parts
            a.i2 = _blk_sumsq(ex, P, n, parts)
        ex.launch[OP_AF_ALPHA](a, 1)
    if wd != Float32(0.0):
        var s = Args()
        s.p0 = P
        s.f0 = Float32(1.0) - ftz(identical_mul(Float32(lr), wd))
        ex.launch[OP_SCALE](s, n)
    if C > 0:
        var r = Args()
        r.p0 = G
        r.p1 = S1
        r.i0 = C
        r.f0 = w
        ex.launch[OP_AF_ROW](r, R)
        var c = Args()
        c.p0 = G
        c.p1 = S2
        c.i0 = R
        c.i1 = C
        c.f0 = w
        ex.launch[OP_AF_COL](c, C)
        var m = Args()
        m.p0 = S1
        m.p1 = sc
        m.i0 = R
        m.f0 = eps1
        ex.launch[OP_AF_RMEAN](m, 1)
        var u = Args()
        u.p0 = G
        u.p1 = S1
        u.p2 = S2
        u.p3 = sc
        u.p4 = U
        u.i0 = C
        u.f0 = eps1sq
        ex.launch[OP_AF_UPDATE_MAT](u, n)
    else:
        var v = Args()
        v.p0 = G
        v.p1 = S1
        v.p2 = U
        v.f0 = w
        v.f1 = eps1sq
        ex.launch[OP_AF_VEC](v, n)
    var d = Args()
    d.p0 = U
    d.p1 = sc
    d.i0 = n
    d.f0 = fval(fp, 4)
    if host_fold:
        # ||u||^2 from the update downloaded once (one wait), the same
        # chain as op_af_denom's; its tail reads alpha (hsc[1], the value
        # sc[1] holds) and writes hsc[3], uploaded into sc[3]
        var hu = List[Float32](length=n, fill=Float32(0.0))
        var hup = FP(unsafe_from_address=Int(hu.unsafe_ptr()))
        ex.download(hup, U, n)
        var dh = d
        dh.p1 = FP(unsafe_from_address=Int(hsc.unsafe_ptr()))
        af_denom_tail(dh, _sumsq_fold_host(hup, 0, n, 1))
        ex.upload(sc + 3, FP(unsafe_from_address=Int(hsc.unsafe_ptr())) + 3, 1)
        _ = hu^
    else:
        if fast:
            d.p2 = parts
            d.i2 = _chunk_sumsq(ex, U, 0, n, parts, 0)
        elif blocked:
            d.p2 = parts
            d.i2 = _blk_sumsq(ex, U, n, parts)
        ex.launch[OP_AF_DENOM](d, 1)
    var ap = Args()
    ap.p0 = P
    ap.p1 = U
    ap.p2 = sc
    ex.launch[OP_AF_APPLY](ap, n)
    ex.download_async(hp, P, n)
    ex.download_async(h1, S1, n if C == 0 else R)
    if C > 0:
        ex.download_async(fptr(addrs[3], "col_var"), S2, C)
    ex.sync()
    _ = hsc^
    return PythonObject(n)


def lamb_step_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """One LAMB step over packed tensors, in place (`sequence/adafactor.mojo`,
    timm's Lamb). addrs = [params, grads, exp_avg, exp_avg_sq] (flat);
    ip = [n_tensors, t, flags, off_0, ..., off_n] with flags bit0 trust_clip,
    bit1 always_adapt, bit2 grad_averaging, bit3 bias_correction, bit4 the
    global gradient-norm clip; fp = [lr, beta1, beta2, eps, weight_decay,
    max_grad_norm (, t0)]. An optional fifth address, float32[2], carries
    (beta1^t0, beta2^t0) after step t0 = fp[6] (0: not yet run); the call
    advances it through step t and writes it back, so the bias corrections
    cost O(1) per step instead of a replay of t products (lane py-sequence;
    the same products in the same order, so the same bits)."""
    if len(addrs) < 4 or len(addrs) > 5 or len(fp) != len(addrs) + 2 or len(ip) < 5:
        raise Error("lamb_step: requires 4 (or 5) addresses, >= 5 integer and 6 (or 7) float parameters")
    var nt = ival(ip, 0)
    var t = ival(ip, 1)
    var flags = ival(ip, 2)
    if nt < 1 or t < 1 or len(ip) != 4 + nt:
        raise Error("lamb_step: n_tensors >= 1, t >= 1 and n_tensors + 1 offsets")
    var offs = List[Float32]()
    for k in range(nt + 1):
        var o = ival(ip, 3 + k)
        if (k == 0 and o != 0) or (k > 0 and o <= Int(offs[k - 1])) or o >= 16777216:
            raise Error("lamb_step: offsets must rise strictly from 0, below 2^24")
        offs.append(Float32(o))
    var n = Int(offs[nt])
    var lr = fval(fp, 0)
    var b1 = fval(fp, 1)
    var b2 = fval(fp, 2)
    var wd = fval(fp, 4)
    var P = ex.alloc(n)
    var G = ex.alloc(n)
    var M = ex.alloc(n)
    var V = ex.alloc(n)
    var U = ex.alloc(n)
    var O = ex.alloc(nt + 1)
    var nrm = ex.alloc(nt)
    var hp = fptr(addrs[0], "params")
    var hm = fptr(addrs[2], "exp_avg")
    var hv = fptr(addrs[3], "exp_avg_sq")
    ex.upload(P, hp, n)
    ex.upload(G, fptr(addrs[1], "grads"), n)
    ex.upload(M, hm, n)
    ex.upload(V, hv, n)
    ex.upload(O, FP(unsafe_from_address=Int(offs.unsafe_ptr())), nt + 1)
    var fast = _fast_norms() and n >= FAST_NORM_MIN
    var partsA = ex.alloc(nt * _PARTS if fast else 1)
    var partsB = ex.alloc(nt * _PARTS if fast else 1)
    if (flags & 16) != 0:
        var q = Args()
        q.p0 = G
        q.p1 = O
        q.p2 = nrm
        if fast:
            for k in range(nt):
                _ = _chunk_sumsq(ex, G, Int(offs[k]), Int(offs[k + 1]) - Int(offs[k]), partsA, k * _PARTS)
            q.p3 = partsA
            q.i0 = 1
        ex.launch[OP_SEG_SUMSQ](q, nt)
        ex.sync()
        var h = List[Float32](length=nt, fill=Float32(0.0))
        ex.download(FP(unsafe_from_address=Int(h.unsafe_ptr())), nrm, nt)
        var gs = Float32(0.0)
        for k in range(nt):
            var nk = ftz(identical_sqrt(h[k]))
            gs = ftz(identical_mul_add(nk, nk, gs))
        var clip = ftz(identical_div(ftz(identical_sqrt(gs)), fval(fp, 5)))
        if clip > Float32(1.0):
            var d = Args()
            d.p0 = G
            d.f0 = clip
            ex.launch[OP_DIVS](d, n)
    var bc1 = Float32(1.0)
    var bc2 = Float32(1.0)
    var pw1 = Float32(1.0)
    var pw2 = Float32(1.0)
    if (flags & 8) != 0:
        var k0 = 0
        if len(addrs) == 5:
            var t0 = Int(Float64(py=fp[6]))
            if t0 < 0 or t0 >= t or Float64(t0) != Float64(py=fp[6]):
                raise Error("lamb_step: the scalars' step t0 must be an integer with 0 <= t0 < t")
            if t0 > 0:
                var sc = fptr(addrs[4], "scalars")
                pw1 = sc.unsafe_load(0)
                pw2 = sc.unsafe_load(1)
                k0 = t0
        for _ in range(k0, t):
            pw1 = ftz(identical_mul(pw1, b1))
            pw2 = ftz(identical_mul(pw2, b2))
        bc1 = Float32(1.0) - pw1
        bc2 = Float32(1.0) - pw2
    var u = Args()
    u.p0 = P
    u.p1 = G
    u.p2 = M
    u.p3 = V
    u.p4 = U
    u.f1 = b1
    u.f2 = b2
    u.f3 = fval(fp, 3)
    u.f4 = wd
    u.f5 = bc1
    u.f6 = ftz(identical_sqrt(bc2))
    u.f7 = (Float32(1.0) - b1) if (flags & 4) != 0 else Float32(1.0)
    ex.launch[OP_LAMB_UPD](u, n)
    var ratio = ex.alloc(nt)
    var r = Args()
    r.p0 = P
    r.p1 = U
    r.p2 = O
    r.p3 = ratio
    r.i0 = flags & 1
    if wd != Float32(0.0) or (flags & 2) != 0:
        if fast:
            for k in range(nt):
                var s = Int(offs[k])
                var e = Int(offs[k + 1])
                _ = _chunk_sumsq(ex, P, s, e - s, partsA, k * _PARTS)
                _ = _chunk_sumsq(ex, U, s, e - s, partsB, k * _PARTS)
            r.p4 = partsA
            r.p5 = partsB
            r.i1 = 1
        ex.launch[OP_LAMB_RATIO](r, nt)
    else:
        var f = Args()
        f.p0 = ratio
        f.f0 = Float32(1.0)
        ex.launch[OP_FILL](f, nt)
    for k in range(nt):
        var s = Int(offs[k])
        var e = Int(offs[k + 1])
        var ap = Args()
        ap.p0 = P + s
        ap.p1 = U + s
        ap.p2 = ratio
        ap.i0 = k
        ap.f0 = lr
        ex.launch[OP_LAMB_APPLY](ap, e - s)
    ex.download_async(hp, P, n)
    ex.download_async(hm, M, n)
    ex.download_async(hv, V, n)
    ex.sync()
    _ = offs^
    if len(addrs) == 5 and (flags & 8) != 0:
        var sc = fptr(addrs[4], "scalars")
        sc.unsafe_store(0, pw1)
        sc.unsafe_store(1, pw2)
    return PythonObject(n)


def layer_norm_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """LayerNorm forward and, with dy, backward (`sequence/layernorm.mojo`).
    addrs = [x (M, D), weight (D) or 0, bias (D) or 0, y (M, D) out (0 allowed
    with backward), dy (M, D) or 0, dx out or 0, dweight out or 0, dbias out or 0];
    ip = [M, D, has_weight, has_bias, backward]; fp = [eps]."""
    if len(addrs) != 8 or len(ip) != 5 or len(fp) != 1:
        raise Error("layer_norm: requires 8 addresses, 5 integer and 1 float parameters")
    var M = ival(ip, 0)
    var D = ival(ip, 1)
    var hw = ival(ip, 2) != 0
    var hb = ival(ip, 3) != 0
    var bwd = ival(ip, 4) != 0
    if M < 1 or D < 1:
        raise Error("layer_norm: M and D must be >= 1")
    var X = ex.alloc(M * D)
    ex.upload(X, fptr(addrs[0], "x"), M * D)
    var W = ex.alloc(D)
    var Bb = ex.alloc(D)
    if hw:
        ex.upload(W, fptr(addrs[1], "weight"), D)
    if hb:
        ex.upload(Bb, fptr(addrs[2], "bias"), D)
    var Y = ex.alloc(M * D)
    var mean = ex.alloc(M)
    var rstd = ex.alloc(M)
    var a = Args()
    a.p0 = X
    a.p1 = W
    a.p2 = Bb
    a.p3 = Y
    a.p4 = mean
    a.p5 = rstd
    a.i0 = D
    a.i1 = 1 if hw else 0
    a.i2 = 1 if hb else 0
    a.f0 = fval(fp, 0)
    ex.launch[OP_LN_FWD](a, M)
    if bwd:
        var DY = ex.alloc(M * D)
        ex.upload(DY, fptr(addrs[4], "dy"), M * D)
        var DX = ex.alloc(M * D)
        var DW = ex.alloc(D)
        var DB = ex.alloc(D)
        var b = Args()
        b.p0 = DY
        b.p1 = X
        b.p2 = W
        b.p3 = DX
        b.p4 = mean
        b.p5 = rstd
        b.i0 = D
        b.i1 = 1 if hw else 0
        ex.launch[OP_LN_BWD_X](b, M)
        var c = Args()
        c.p0 = DY
        c.p1 = X
        c.p2 = DW
        c.p3 = DB
        c.p4 = mean
        c.p5 = rstd
        c.i0 = D
        c.i1 = M
        # FAST: the column folds split over row blocks (about 8192 threads,
        # at least 1024 rows each), then one ordered sum of the S partials
        var S = min(max(8192 // D, 1), M // 1024) if _fast_norms() else 0
        if S > 1:
            var RS = (M + S - 1) // S
            S = (M + RS - 1) // RS
            var PW = ex.alloc(S * D)
            var PB = ex.alloc(S * D)
            c.p6 = PW
            c.p7 = PB
            c.i2 = S
            c.i3 = RS
            ex.launch[OP_LN_BWD_W](c, S * D)
            c.i4 = 1
            ex.launch[OP_LN_BWD_W](c, D)
        else:
            ex.launch[OP_LN_BWD_W](c, D)
        ex.download_async(fptr(addrs[5], "dx"), DX, M * D)
        if hw:
            ex.download_async(fptr(addrs[6], "dweight"), DW, D)
        if hb:
            ex.download_async(fptr(addrs[7], "dbias"), DB, D)
    # a backward call may pass y = 0: y is then recomputed on the device (for
    # mean / rstd) but never crosses back (lane py-sequence)
    if not bwd or Int(py=addrs[3]) != 0:
        ex.download_async(fptr(addrs[3], "y"), Y, M * D)
    ex.sync()
    return PythonObject(M * D)


def theta_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """statsforecast's Theta family over a batch of series (`sequence/theta.mojo`).
    addrs = [y (B, n), forecast (B, h) out, info (B, 8) out];
    ip = [B, n, h, season_length, model (-1 auto, 0 STM, 1 OTM, 2 DSTM,
    3 DOTM), decomposition (0 multiplicative, 1 additive), fixed mask];
    fp = [initial_smoothed, alpha, theta] (read where fixed)."""
    if len(addrs) != 3 or len(ip) != 7 or len(fp) != 3:
        raise Error("theta: requires 3 addresses, 7 integer and 3 float parameters")
    var B = ival(ip, 0)
    var n = ival(ip, 1)
    var h = ival(ip, 2)
    var m = ival(ip, 3)
    var model = ival(ip, 4)
    if B < 1 or n <= 3 or h < 1 or m < 1 or model < -1 or model > 3:
        raise Error("theta: B >= 1, n > 3 (the reference refuses tiny series), h >= 1, season_length >= 1")
    var stride = n + n + m + 5 * (n + h) + n + 64 + 12 + h
    var Y = ex.alloc(B * n)
    ex.upload(Y, fptr(addrs[0], "y"), B * n)
    var F = ex.alloc(B * h)
    var I = ex.alloc(B * 8)
    var S = ex.alloc(B * stride)
    var a = Args()
    a.p0 = Y
    a.p1 = F
    a.p2 = I
    a.p3 = S
    a.i0 = n
    a.i1 = h
    a.i2 = m
    a.i3 = model
    a.i4 = ival(ip, 5)
    a.i5 = ival(ip, 6)
    a.i6 = stride
    a.f0 = fval(fp, 0)
    a.f1 = fval(fp, 1)
    a.f2 = fval(fp, 2)
    ex.launch[OP_THETA](a, B)
    ex.sync()
    ex.download(fptr(addrs[1], "forecast"), F, B * h)
    ex.download(fptr(addrs[2], "info"), I, B * 8)
    return PythonObject(B * h)


def croston_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """addrs = [y (B, n), mean (B) out]; ip = [B, n, variant]
    (`sequence/croston.mojo`)."""
    if len(addrs) != 2 or len(ip) != 3:
        raise Error("croston: requires 2 addresses and 3 integer parameters")
    var B = ival(ip, 0)
    var n = ival(ip, 1)
    var v = ival(ip, 2)
    if B < 1 or n < 1 or v < 0 or v > 2:
        raise Error("croston: B, n >= 1 and variant 0 (classic), 1 (optimized) or 2 (SBA)")
    var Y = ex.alloc(B * n)
    ex.upload(Y, fptr(addrs[0], "y"), B * n)
    var M = ex.alloc(B)
    var S = ex.alloc(B * 2 * n)
    var a = Args()
    a.p0 = Y
    a.p1 = M
    a.p2 = S
    a.i0 = n
    a.i1 = v
    ex.launch[OP_CROSTON](a, B)
    ex.sync()
    ex.download(fptr(addrs[1], "mean"), M, B)
    return PythonObject(B)


#: ETS's FAST Nelder-Mead stall stop (sequence/nm.mojo): the fit ends once
#: its best value has not dropped by more than ETS_FAST_STALL_REL |best| for
#: ETS_FAST_STALL_ITERS iterations. FAST only; chosen by the paired quality
#: sweep in tools/sequence_quality.py:
#: ETS's simplex still gains more than 1e-6 |best| every 50 to 300 iterations
#: at the 1000 cap, so no stall stop ends it early; the stop stays opt-in.
comptime ETS_FAST_STALL_ITERS = 0   # off: the sweep found no stop that saves time (below)
comptime ETS_FAST_STALL_REL = Float32(1e-6)


def ets_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """ETS over a batch of series (`sequence/ets.mojo`).
    addrs = [y (B, n), forecast (B, h) out, info (B, 10) out, seasonal states (B, m) out];
    ip = [B, n, h, error (0 A, 1 M), trend (0 N, 1 A), damped, fixed mask, season (0 N, 1 A, 2 M), m];
    fp = [alpha, beta, phi, gamma] (read where fixed). FAST: an optional
    10th integer and 5th float override the Nelder-Mead stall stop
    (ETS_FAST_STALL_ITERS / ETS_FAST_STALL_REL; 0 iterations: off)."""
    if len(addrs) != 4 or (len(ip) != 9 and len(ip) != 10) or (len(fp) != 4 and len(fp) != 5):
        raise Error("ets: requires 4 addresses, 9 (or 10) integer and 4 (or 5) float parameters")
    var B = ival(ip, 0)
    var n = ival(ip, 1)
    var h = ival(ip, 2)
    var season = ival(ip, 7)
    var m = ival(ip, 8) if season != 0 else 1
    if B < 1 or n < 4 or h < 1:
        raise Error("ets: B >= 1, n >= 4 and h >= 1")
    if season < 0 or season > 2 or (season != 0 and (m < 2 or m > 58 or n <= m)):
        raise Error("ets: a season needs 2 <= m <= 58 and n > m")
    var stride = ets_scratch(n, m)
    var Y = ex.alloc(B * n)
    ex.upload(Y, fptr(addrs[0], "y"), B * n)
    var F = ex.alloc(B * h)
    var I = ex.alloc(B * 10)
    var S = ex.alloc(B * stride)
    var SS = ex.alloc(B * m)
    var a = Args()
    a.p0 = Y
    a.p1 = F
    a.p2 = I
    a.p3 = S
    a.p4 = SS
    a.i0 = n
    a.i1 = h
    a.i2 = ival(ip, 3)
    a.i3 = ival(ip, 4)
    a.i4 = ival(ip, 5)
    a.i5 = ival(ip, 6)
    a.i6 = season
    a.i7 = m
    a.i8 = stride
    a.f0 = fval(fp, 0)
    a.f1 = fval(fp, 1)
    a.f2 = fval(fp, 2)
    a.f3 = fval(fp, 3)
    a.i9 = ival(ip, 9) if len(ip) == 10 else ETS_FAST_STALL_ITERS
    a.f4 = fval(fp, 4) if len(fp) == 5 else ETS_FAST_STALL_REL
    ex.launch[OP_ETS](a, B)
    ex.sync()
    ex.download(fptr(addrs[1], "forecast"), F, B * h)
    ex.download(fptr(addrs[2], "info"), I, B * 10)
    if season != 0:
        ex.download(fptr(addrs[3], "seasonal states"), SS, B * m)
    return PythonObject(B * h)


#: GARCH's FAST Nelder-Mead stall stop (as ETS_FAST_STALL_*): float32 never
#: meets the reference's tol_std 1e-6 at -loglik ~ 100, so a capped simplex
#: ran both 2000-iteration runs; FAST ends a run whose best value has stopped
#: moving. Chosen by the paired quality sweep (tools/sequence_quality.py).
comptime GARCH_FAST_STALL_ITERS = 50
comptime GARCH_FAST_STALL_REL = Float32(1e-5)


def garch_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """GARCH(p, o, q) over a batch of series (`sequence/garch.mojo`).
    addrs = [y (B, n), params (B, 1 + 1 + p + o + q) out, info (B, 4) out,
    sigma (B, n) out, variance forecast (B, h) out];
    ip = [B, n, h, p, o, q, constant mean]. FAST: an optional 8th and 9th
    override the Nelder-Mead stall stop (iterations, relative drop in units
    of 1e-9; GARCH_FAST_STALL_*; 0 iterations: off)."""
    if len(addrs) != 5 or (len(ip) != 7 and len(ip) != 9):
        raise Error("garch: requires 5 addresses and 7 (or 9) integer parameters")
    var B = ival(ip, 0)
    var n = ival(ip, 1)
    var h = ival(ip, 2)
    var p = ival(ip, 3)
    var o = ival(ip, 4)
    var q = ival(ip, 5)
    var cm = ival(ip, 6)
    if B < 1 or n < 10 or h < 1 or p < 0 or o < 0 or q < 0 or p + o + q < 1 or 1 + p + o + q + cm > 8:
        raise Error("garch: B >= 1, n >= 10, h >= 1, p + o + q >= 1 and at most 8 parameters")
    var m = max(p, max(o, q))
    var stride = 4 * n + 64 + max(128, 3 * (m + h)) + GARCH_SNAP
    var Y = ex.alloc(B * n)
    ex.upload(Y, fptr(addrs[0], "y"), B * n)
    var P = ex.alloc(B * (1 + 1 + p + o + q))
    var I = ex.alloc(B * 4)
    var Sg = ex.alloc(B * n)
    var F = ex.alloc(B * h)
    var S = ex.alloc(B * stride)
    var a = Args()
    a.p0 = Y
    a.p1 = P
    a.p2 = I
    a.p3 = Sg
    a.p4 = F
    a.p5 = S
    a.i0 = n
    a.i1 = h
    a.i2 = p
    a.i3 = o
    a.i4 = q
    a.i5 = cm
    a.i6 = stride
    a.i7 = ival(ip, 7) if len(ip) == 9 else GARCH_FAST_STALL_ITERS
    a.f0 = Float32(Float64(ival(ip, 8)) * 1e-9) if len(ip) == 9 else GARCH_FAST_STALL_REL
    ex.launch[OP_GARCH](a, B)
    ex.sync()
    ex.download(fptr(addrs[1], "params"), P, B * (1 + 1 + p + o + q))
    ex.download(fptr(addrs[2], "info"), I, B * 4)
    ex.download(fptr(addrs[3], "sigma"), Sg, B * n)
    ex.download(fptr(addrs[4], "forecast"), F, B * h)
    return PythonObject(B)


def _prophet_X[E: Exec](mut ex: E, frac_addr: PythonObject, orders_addr: PythonObject, h_addr: PythonObject,
                        N: Int, ns: Int, nh: Int, K: Int) raises -> FP:
    var X = ex.alloc(N * K)
    var Fr = ex.alloc(N * max(ns, 1))
    var Od = ex.alloc(max(ns, 1))
    var H = ex.alloc(N * max(nh, 1))
    if ns > 0:
        ex.upload(Fr, fptr(frac_addr, "frac"), N * ns)
        ex.upload(Od, fptr(orders_addr, "orders"), ns)
    if nh > 0:
        ex.upload(H, fptr(h_addr, "holidays"), N * nh)
    var a = Args()
    a.p0 = Fr
    a.p1 = Od
    a.p2 = H
    a.p3 = X
    a.i0 = ns
    a.i1 = nh
    a.i2 = K
    if K > 0:
        ex.launch[OP_PROPHET_FEATURES](a, N)
    return X


#: FAST's prophet fit runs its likelihood over point chunks on the device
#: (apple2) from this many points; a smaller series keeps a thread each.
comptime PROPHET_FAST_MIN_N = 16384


def _prophet_fg_dev[E: Exec](mut ex: E, pa: Args, C: Int, thd: FP, outd: FP, d: ProphetData,
                             th: FP, g: FP) raises -> Float32:
    """prophet_fg at host th into host g: the likelihood's C chunks and their
    ordered sums on the device (op_prophet_fg_part / _sum), the priors here."""
    var P = 3 + d.S + d.K
    ex.upload(thd, th, P)
    ex.launch[OP_PROPHET_FG_PART](pa, C)
    var s = Args()
    s.p6 = pa.p6
    s.p7 = outd
    s.i4 = C
    s.i5 = P + 1
    ex.launch[OP_PROPHET_FG_SUM](s, P + 1)
    var sse = List[Float32](length=1, fill=Float32(0.0))
    ex.download_async(g, outd, P)
    ex.download_async(FP(unsafe_from_address=Int(sse.unsafe_ptr())), outd + P, 1)
    ex.sync()
    var r = _fg_prior(d, th, g, sse[0])
    _ = sse^
    return r


def lbfgs_prophet_host[E: Exec](mut ex: E, pa: Args, C: Int, thd: FP, outd: FP,
                                d: ProphetData, th: FP, w: FP, max_iter: Int) raises -> Tuple[Float32, Int]:
    """FAST (apple2): sequence/prophet.mojo::lbfgs_prophet line for line, on
    the host, each objective the device's chunked likelihood
    (`_prophet_fg_dev`). th and w are host memory."""
    var P = 3 + d.S + d.K
    var g = w
    var dvec = g + P
    var thn = dvec + P
    var gn = thn + P
    var q = gn + P
    var sm = q + P
    var ym = sm + MEM * P
    var rho = ym + MEM * P
    var al = rho + MEM
    var f = _prophet_fg_dev(ex, pa, C, thd, outd, d, th, g)
    var npairs = 0
    var head = 0
    var small = 0
    var it = 0
    while it < max_iter:
        # two-loop recursion: q = g; newest to oldest, then oldest to newest
        for i in range(P):
            p_st(q, i, p_ld(g, i))
        var idx = head
        for _ in range(npairs):
            idx = (idx - 1 + MEM) % MEM
            var aa = p_mul(p_ld(rho, idx), _dot(sm + idx * P, q, P))
            p_st(al, idx, aa)
            for i in range(P):
                p_st(q, i, p_sub(p_ld(q, i), p_mul(aa, p_ld(ym + idx * P, i))))
        var gamma = Float32(1.0)
        if npairs > 0:
            var last = (head - 1 + MEM) % MEM
            var yy = _dot(ym + last * P, ym + last * P, P)
            if yy > Float32(0.0):
                gamma = _pdiv(_dot(sm + last * P, ym + last * P, P), yy)
        else:
            var gg = ftz(identical_sqrt(_dot(g, g, P)))
            if gg > Float32(1.0):
                gamma = _pdiv(Float32(1.0), gg)
        for i in range(P):
            p_st(q, i, p_mul(gamma, p_ld(q, i)))
        var start = (head - npairs + MEM) % MEM
        idx = start
        for _ in range(npairs):
            var bb = p_mul(p_ld(rho, idx), _dot(ym + idx * P, q, P))
            var coef = p_sub(p_ld(al, idx), bb)
            for i in range(P):
                p_st(q, i, p_fma3(coef, p_ld(sm + idx * P, i), p_ld(q, i)))
            idx = (idx + 1) % MEM
        for i in range(P):
            p_st(dvec, i, -p_ld(q, i))
        var gd = _dot(g, dvec, P)
        if not (gd < Float32(0.0)):
            for i in range(P):
                p_st(dvec, i, -p_ld(g, i))
            gd = -_dot(g, g, P)
            npairs = 0
        # backtracking Armijo
        var step = Float32(1.0)
        var fnew = Float32(0.0)
        var ok = False
        for _ in range(40):
            for i in range(P):
                p_st(thn, i, p_fma3(step, p_ld(dvec, i), p_ld(th, i)))
            fnew = _prophet_fg_dev(ex, pa, C, thd, outd, d, thn, gn)
            if fnew <= p_fma3(p_mul(Float32(1e-4), step), gd, f):
                ok = True
                break
            step = p_mul(step, Float32(0.5))
        it += 1
        if not ok:
            break
        # the new pair, kept only with positive curvature (q and dvec are
        # free here; the slot is written only when the pair is kept)
        for i in range(P):
            p_st(q, i, p_sub(p_ld(thn, i), p_ld(th, i)))
            p_st(dvec, i, p_sub(p_ld(gn, i), p_ld(g, i)))
        var sy = _dot(q, dvec, P)
        if sy > Float32(1e-12):
            var slot = head
            for i in range(P):
                p_st(sm + slot * P, i, p_ld(q, i))
                p_st(ym + slot * P, i, p_ld(dvec, i))
            p_st(rho, slot, _pdiv(Float32(1.0), sy))
            head = (head + 1) % MEM
            if npairs < MEM:
                npairs += 1
        var fscale = abs(f)
        if abs(fnew) > fscale:
            fscale = abs(fnew)
        if fscale < Float32(1.0):
            fscale = Float32(1.0)
        var df = p_sub(f, fnew)
        for i in range(P):
            p_st(th, i, p_ld(thn, i))
            p_st(g, i, p_ld(gn, i))
        f = fnew
        var gmax = Float32(0.0)
        for i in range(P):
            var v = abs(p_ld(g, i))
            if v > gmax:
                gmax = v
        if gmax < Float32(1e-5):
            break
        if df <= p_mul(Float32(1e-7), fscale):
            small += 1
            if small >= 3:
                break
        else:
            small = 0
    return (f, it)


def prophet_fit_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """The Prophet-style fit over a batch of series sharing t
    (`sequence/prophet.mojo`). addrs = [y (B, N), t (N, scaled), frac (N, ns),
    orders (ns, as floats), holidays (N, nh), changepoints (S, scaled),
    prior scales (K), params (B, 3 + S + K) out, info (B, 4) out];
    ip = [B, N, ns, nh, K, S, multiplicative, max_iter]; fp = [tau]."""
    if len(addrs) != 9 or len(ip) != 8 or len(fp) != 1:
        raise Error("prophet_fit: requires 9 addresses, 8 integer and 1 float parameters")
    var B = ival(ip, 0)
    var N = ival(ip, 1)
    var ns = ival(ip, 2)
    var nh = ival(ip, 3)
    var K = ival(ip, 4)
    var S = ival(ip, 5)
    if B < 1 or N < 2 or ns < 0 or nh < 0 or K < 0 or S < 0:
        raise Error("prophet_fit: B >= 1, N >= 2 and nonnegative counts")
    var P = 3 + S + K
    var stride = N + P + (6 + 2 * 5) * P + 2 * 5
    var X = _prophet_X(ex, addrs[2], addrs[3], addrs[4], N, ns, nh, K)
    var Y = ex.alloc(B * N)
    ex.upload(Y, fptr(addrs[0], "y"), B * N)
    var T = ex.alloc(N)
    ex.upload(T, fptr(addrs[1], "t"), N)
    var C_ = ex.alloc(max(S, 1))
    if S > 0:
        ex.upload(C_, fptr(addrs[5], "changepoints"), S)
    var Sg = ex.alloc(max(K, 1))
    if K > 0:
        ex.upload(Sg, fptr(addrs[6], "prior scales"), K)
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        if N >= PROPHET_FAST_MIN_N:
            # FAST (apple2): one series at a time, its likelihood over point
            # chunks on the device; L-BFGS and the priors on the host
            var C = min(4096, (N + 255) // 256)
            var parts = ex.alloc(C * (P + 1))
            var thd = ex.alloc(P)
            var outd = ex.alloc(P + 1)
            var ysd = ex.alloc(N)
            var hy = fptr(addrs[0], "y")
            var hpar = fptr(addrs[7], "params")
            var hinf = fptr(addrs[8], "info")
            var dh = ProphetData(FP(unsafe_from_address=64), FP(unsafe_from_address=64),
                                 fptr(addrs[5], "changepoints") if S > 0 else FP(unsafe_from_address=64),
                                 fptr(addrs[6], "prior scales") if K > 0 else FP(unsafe_from_address=64),
                                 N, K, S, fval(fp, 0), ival(ip, 6) != 0)
            var ht = fptr(addrs[1], "t")
            var ys = List[Float32](length=N, fill=Float32(0.0))
            var ws = List[Float32](length=(6 + 2 * 5) * P + 2 * 5 + P, fill=Float32(0.0))
            var pys = FP(unsafe_from_address=Int(ys.unsafe_ptr()))
            var pw = FP(unsafe_from_address=Int(ws.unsafe_ptr()))
            var th = pw + (6 + 2 * 5) * P + 2 * 5
            var pa = Args()
            pa.p0 = ysd
            pa.p1 = T
            pa.p2 = X
            pa.p3 = C_
            pa.p4 = Sg
            pa.p5 = thd
            pa.p6 = parts
            pa.i0 = N
            pa.i1 = K
            pa.i2 = S
            pa.i3 = ival(ip, 6)
            pa.i4 = C
            pa.f0 = fval(fp, 0)
            for b in range(B):
                var y = hy + b * N
                var scale = Float32(0.0)
                for i in range(N):
                    var v = abs(y[i])
                    if v > scale:
                        scale = v
                if scale == Float32(0.0):
                    scale = Float32(1.0)
                for i in range(N):
                    pys[i] = ftz(identical_div(y[i], scale))
                ex.upload(ysd, pys, N)
                var t0 = ht[0]
                var t1 = ht[N - 1]
                var k = ftz(identical_div(pys[N - 1] - pys[0], t1 - t0)) if t1 != t0 else Float32(0.0)
                th[0] = k
                th[1] = pys[0] - ftz(identical_mul(k, t0))
                for j in range(S + 1 + K):
                    th[2 + j] = Float32(0.0)
                var r = lbfgs_prophet_host(ex, pa, C, thd, outd, dh, th, pw, ival(ip, 7))
                for j in range(P):
                    hpar[b * P + j] = th[j]
                hinf[b * 4] = scale
                hinf[b * 4 + 1] = r[0]
                hinf[b * 4 + 2] = Float32(r[1])
                hinf[b * 4 + 3] = Float32(0.0)
            _ = ys^
            _ = ws^
            return PythonObject(B)
    var Pm = ex.alloc(B * P)
    var I = ex.alloc(B * 4)
    var W = ex.alloc(B * stride)
    var a = Args()
    a.p0 = Y
    a.p1 = T
    a.p2 = X
    a.p3 = C_
    a.p4 = Sg
    a.p5 = Pm
    a.p6 = I
    a.p7 = W
    a.i0 = N
    a.i1 = K
    a.i2 = S
    a.i3 = ival(ip, 6)
    a.i4 = stride
    a.i5 = ival(ip, 7)
    a.f0 = fval(fp, 0)
    ex.launch[OP_PROPHET_FIT](a, B)
    ex.sync()
    ex.download(fptr(addrs[7], "params"), Pm, B * P)
    ex.download(fptr(addrs[8], "info"), I, B * 4)
    return PythonObject(B)


def prophet_predict_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """addrs = [params (B, P), info (B, 4), t (M, scaled), frac (M, ns),
    orders (ns), holidays (M, nh), changepoints (S), yhat (B, M) out,
    trend (B, M) out]; ip = [B, M, ns, nh, K, S, multiplicative]."""
    if len(addrs) != 9 or len(ip) != 7:
        raise Error("prophet_predict: requires 9 addresses and 7 integer parameters")
    var B = ival(ip, 0)
    var M = ival(ip, 1)
    var ns = ival(ip, 2)
    var nh = ival(ip, 3)
    var K = ival(ip, 4)
    var S = ival(ip, 5)
    if B < 1 or M < 1:
        raise Error("prophet_predict: B, M >= 1")
    var P = 3 + S + K
    var X = _prophet_X(ex, addrs[3], addrs[4], addrs[5], M, ns, nh, K)
    var Pm = ex.alloc(B * P)
    ex.upload(Pm, fptr(addrs[0], "params"), B * P)
    var I = ex.alloc(B * 4)
    ex.upload(I, fptr(addrs[1], "info"), B * 4)
    var T = ex.alloc(M)
    ex.upload(T, fptr(addrs[2], "t"), M)
    var C = ex.alloc(max(S, 1))
    if S > 0:
        ex.upload(C, fptr(addrs[6], "changepoints"), S)
    var Yh = ex.alloc(B * M)
    var Tr = ex.alloc(B * M)
    var a = Args()
    a.p0 = Pm
    a.p1 = I
    a.p2 = T
    a.p3 = X
    a.p4 = C
    a.p5 = Yh
    a.p6 = Tr
    a.i0 = M
    a.i1 = K
    a.i2 = S
    a.i3 = ival(ip, 6)
    ex.launch[OP_PROPHET_PREDICT](a, B * M)
    ex.sync()
    ex.download(fptr(addrs[7], "yhat"), Yh, B * M)
    ex.download(fptr(addrs[8], "trend"), Tr, B * M)
    return PythonObject(B * M)


def fptr_of(xs: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(xs.unsafe_ptr()))


def moe_forward_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """The Mixtral sparse MoE block forward (`sequence/moe.mojo`).
    addrs = [x (T, D), router (E, D), gate_up (E, 2F, D), down (E, D, F),
    y (T, D) out, logits (T, E) out, selected (T, k) out as floats,
    weights (T, k) out]; ip = [T, D, F, E, k, renormalise]."""
    if len(addrs) != 8 or len(ip) != 6:
        raise Error("moe_forward: requires 8 addresses and 6 integer parameters")
    var T = ival(ip, 0)
    var D = ival(ip, 1)
    var F = ival(ip, 2)
    var En = ival(ip, 3)
    var k = ival(ip, 4)
    if T < 1 or D < 1 or F < 1 or En < 1 or k < 1 or k > En:
        raise Error("moe_forward: T, D, F, E >= 1 and 1 <= k <= E")
    var X = ex.alloc(T * D)
    ex.upload(X, fptr(addrs[0], "x"), T * D)
    var Wg = ex.alloc(En * D)
    ex.upload(Wg, fptr(addrs[1], "router"), En * D)
    var Gu = ex.alloc(En * 2 * F * D)
    ex.upload(Gu, fptr(addrs[2], "gate_up_proj"), En * 2 * F * D)
    var Dn = ex.alloc(En * D * F)
    ex.upload(Dn, fptr(addrs[3], "down_proj"), En * D * F)
    var Y = ex.alloc(T * D)
    var L = ex.alloc(T * En)
    var Sel = ex.alloc(T * k)
    var W = ex.alloc(T * k)
    var Pr = ex.alloc(T * En)
    var H = ex.alloc(T * k * F)
    var a = Args()
    a.p0 = X
    a.p1 = Wg
    a.p2 = L
    a.p3 = Sel
    a.p4 = W
    a.p5 = Pr
    a.i0 = D
    a.i1 = En
    a.i2 = k
    a.i3 = ival(ip, 5)
    ex.launch[OP_MOE_ROUTE](a, T)
    # The pairs (token, pick) grouped by expert for the tiled device
    # products (lane neural-pass29, sequence/moe_tiled.mojo): `order` the
    # pair indices sorted by expert (stable), `poff` each expert's first
    # pair, `boff` each expert's first block of TILE_P pairs x TILE_Q
    # outputs. The host executor runs the items and reads none of these.
    ex.sync()
    var sel_host = List[Float32](length=T * k, fill=Float32(0.0))
    ex.download(fptr_of(sel_host), Sel, T * k)
    var counts = List[Int](length=En + 1, fill=0)
    for i in range(T * k):
        counts[Int(sel_host[i]) + 1] += 1
    var poff_host = List[Float32](length=En + 1, fill=Float32(0.0))
    var running = 0
    for e in range(En):
        poff_host[e] = Float32(running)
        running += counts[e + 1]
    poff_host[En] = Float32(running)
    var cursor = List[Int](length=En, fill=0)
    for e in range(En):
        cursor[e] = Int(poff_host[e])
    var order_host = List[Float32](length=T * k, fill=Float32(0.0))
    for i in range(T * k):
        var e = Int(sel_host[i])
        order_host[cursor[e]] = Float32(i)
        cursor[e] += 1
    var n_ftiles = (F + TILE_Q - 1) // TILE_Q
    var n_dtiles = (D + TILE_Q - 1) // TILE_Q
    var boff_h = List[Float32](length=En + 1, fill=Float32(0.0))
    var boff_o = List[Float32](length=En + 1, fill=Float32(0.0))
    var blocks_h = 0
    var blocks_o = 0
    for e in range(En):
        boff_h[e] = Float32(blocks_h)
        boff_o[e] = Float32(blocks_o)
        var ptiles = (counts[e + 1] + TILE_P - 1) // TILE_P
        blocks_h += ptiles * n_ftiles
        blocks_o += ptiles * n_dtiles
    boff_h[En] = Float32(blocks_h)
    boff_o[En] = Float32(blocks_o)
    var Order = ex.alloc(T * k)
    ex.upload(Order, fptr_of(order_host), T * k)
    var Poff = ex.alloc(En + 1)
    ex.upload(Poff, fptr_of(poff_host), En + 1)
    var BoffH = ex.alloc(En + 1)
    ex.upload(BoffH, fptr_of(boff_h), En + 1)
    var BoffO = ex.alloc(En + 1)
    ex.upload(BoffO, fptr_of(boff_o), En + 1)
    var S = ex.alloc(T * k * D)
    var b = Args()
    b.p0 = X
    b.p1 = Gu
    b.p2 = Sel
    b.p3 = H
    b.p4 = Order
    b.p5 = Poff
    b.p6 = BoffH
    b.i0 = D
    b.i1 = F
    b.i2 = k
    b.i3 = En
    b.i4 = blocks_h
    b.i5 = n_ftiles
    ex.launch[OP_MOE_HIDDEN](b, T * k * F)
    var c = Args()
    c.p0 = H
    c.p1 = Dn
    c.p2 = Sel
    c.p3 = W
    c.p4 = Y
    c.p5 = S
    c.p6 = Order
    c.p7 = Poff
    c.p8 = BoffO
    c.i0 = D
    c.i1 = F
    c.i2 = k
    c.i3 = En
    c.i4 = blocks_o
    c.i5 = n_dtiles
    ex.launch[OP_MOE_OUT](c, T * D)
    _ = sel_host^
    _ = order_host^
    _ = poff_host^
    _ = boff_h^
    _ = boff_o^
    _ = counts^
    _ = cursor^
    ex.sync()
    ex.download(fptr(addrs[4], "y"), Y, T * D)
    ex.download(fptr(addrs[5], "logits"), L, T * En)
    ex.download(fptr(addrs[6], "selected"), Sel, T * k)
    ex.download(fptr(addrs[7], "weights"), W, T * k)
    return PythonObject(T * D)
