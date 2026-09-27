# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE ADDRESS CONTRACT of the sequence lane's bindings, written once over
`Exec`: `bindings/_mojolearn_x_sequence.mojo` calls these with a
`DeviceExec`, `bindings/_mojolearn_x_sequence_host.mojo` with a `HostExec`.
Every buffer is a caller-owned C-contiguous array handed over by address;
nothing is retained after the call."""
from std.python import PythonObject

from checks.numerics import ftz, identical_mul
from sequence.exec import Exec
from sequence.ops import FP, OP_STL, OP_VAR_DESIGN, OP_COLSCALE, OP_CHOLSOLVE, OP_ROWSCALE, OP_VAR_FORECAST, OP_SUB, OP_SCALE, Args, OPT_ADAGRAD, OPT_ADAM, OPT_ADAMW, OPT_RMSPROP, OPT_SGD
from sequence.recurrent import gemm
from sequence.recurrent import TASK_CE, TASK_MSE, Net, OptConfig, OptState, opt_step, rnn_fit, rnn_predict


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
    if kind < OPT_SGD or kind > OPT_ADAGRAD:
        raise Error("sequence: unknown optimizer kind " + String(kind))
    return OptConfig(kind, ival(ip, at + 1), fval(fp, fat), fval(fp, fat + 1), fval(fp, fat + 2),
                     fval(fp, fat + 3), fval(fp, fat + 4))


def rnn_fit_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """addrs = [X, y, order, steps (int32), params (in/out), losses, lrs];
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
    if N < 1 or T < 1 or n_steps < 1 or n_order < 1 or N >= 16777216 or n_order >= 16777216:
        raise Error("rnn_fit: N, T, steps and order must be >= 1 and N, order < 2^24")
    var cfg = opt_of(ip, 10, fp, 0)
    var steps = iptr(steps_addr, "steps")
    for k in range(n_steps):
        var off = Int(steps.unsafe_load(2 * k))
        var cnt = Int(steps.unsafe_load(2 * k + 1))
        if cnt < 1 or off < 0 or off + cnt > n_order:
            raise Error("rnn_fit: step " + String(k) + " reads outside the order")
    var order = fptr(order_addr, "order")
    for i in range(n_order):
        var v = Int(order.unsafe_load(i))
        if v < 0 or v >= N or Float32(v) != order.unsafe_load(i):
            raise Error("rnn_fit: order holds a value that is not a sample index")
    if task == TASK_CE:
        var y = fptr(y_addr, "y")
        for i in range(N):
            var v = Int(y.unsafe_load(i))
            if v < 0 or v >= net.O or Float32(v) != y.unsafe_load(i):
                raise Error("rnn_fit: a label is not a class index below the class count")
    rnn_fit(ex, net, task, fptr(x_addr, "X"), fptr(y_addr, "y"), N, T, order, n_order,
            steps, n_steps, fptr(p_addr, "params"), fptr(losses_addr, "losses"),
            fptr(lrs_addr, "lrs"), cfg, fval(fp, 5))
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


def opt_step_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """One optimizer step over a flat float32 buffer, in place.
    addrs = [params, grads, state1, state2, state3];
    ip = [n, kind, flags, t]; fp = [lr, f1, f2, eps, weight_decay, f7]."""
    if len(addrs) != 5 or len(ip) != 4 or len(fp) != 6:
        raise Error("optimizer_step: requires 5 addresses, 4 integer and 6 float parameters")
    var p_addr = addrs[0]
    var g_addr = addrs[1]
    var s1_addr = addrs[2]
    var s2_addr = addrs[3]
    var s3_addr = addrs[4]
    var n = ival(ip, 0)
    var t = ival(ip, 3)
    if n < 1 or t < 1:
        raise Error("optimizer_step: n and the one-based step t must be >= 1")
    var cfg = opt_of(ip, 1, fp, 1)
    var st = OptState()
    # the running powers are a function of t alone: replay them
    for _ in range(t - 1):
        _ = opt_advance(cfg, st)
    var P = ex.alloc(n)
    var G = ex.alloc(n)
    var s1 = ex.alloc(n)
    var s2 = ex.alloc(n)
    var s3 = ex.alloc(n)
    var hp = fptr(p_addr, "params")
    var h1 = fptr(s1_addr, "state1")
    var h2 = fptr(s2_addr, "state2")
    var h3 = fptr(s3_addr, "state3")
    ex.upload(P, hp, n)
    ex.upload(G, fptr(g_addr, "grads"), n)
    ex.upload(s1, h1, n)
    ex.upload(s2, h2, n)
    ex.upload(s3, h3, n)
    opt_step(ex, cfg, st, t, fval(fp, 0), P, G, s1, s2, s3, n)
    ex.sync()
    ex.download(hp, P, n)
    ex.download(h1, s1, n)
    ex.download(h2, s2, n)
    ex.download(h3, s3, n)
    return PythonObject(n)


def opt_advance(cfg: OptConfig, mut st: OptState) -> Int:
    if cfg.kind == OPT_ADAM or cfg.kind == OPT_ADAMW:
        st.pw1 = ftz(identical_mul(st.pw1, cfg.f1))
        st.pw2 = ftz(identical_mul(st.pw2, cfg.f2))
    return 0


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
