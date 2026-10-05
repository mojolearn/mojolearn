# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ProphetForecaster's input preparation (lane cgr4-py-compute: it was numpy
in `_x_sequence_prophet.py`): the time axis in float64 days, its checks
(NaN, ascending), span and least spacing, the scaled time, the seasonal
phase (t mod P) / P by an EXACT fmod, and the changepoints (the caller's,
sorted; or Prophet's `np.linspace(0, hist - 1, n + 1)` rows, rounded half
to even, the first dropped), scaled.

Float64 throughout and one rounding to float32 per output, so it runs in
Mojo on the host in both bindings (the Apple GPU has no float64): the same
code and the same bits on every column. O(N) per series set, no data
dependence across rows beyond the running checks."""
from std.memory import bitcast
from std.builtin.sort import sort
from std.python import PythonObject
from sequence.exec_trait import Exec
from sequence.ops import Args, OP_PROPHET_PREP

comptime F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime I64P = MutPointer[Int64, MutUntrackedOrigin]


def _f64p(addr: PythonObject, what: String) raises -> F64P:
    var a = Int(py=addr)
    if a == 0:
        raise Error("prophet_prep: null buffer for " + what)
    return F64P(unsafe_from_address=a)


def _f32p(addr: PythonObject, what: String) raises -> F32P:
    var a = Int(py=addr)
    if a == 0:
        raise Error("prophet_prep: null buffer for " + what)
    return F32P(unsafe_from_address=a)


def fmod_exact64(a: Float64, b: Float64) -> Float64:
    """C fmod(a, b) for finite b > 0, exact (the sign of a), by integer long
    division of the significands; a NaN or infinite a gives NaN."""
    var ua = bitcast[DType.uint64](a)
    var ub = bitcast[DType.uint64](b)
    var ea = Int((ua >> 52) & 0x7FF)
    var eb = Int((ub >> 52) & 0x7FF)
    if ea == 0x7FF:
        return bitcast[DType.float64](UInt64(0x7FF8000000000000))
    if abs(a) < b:
        return a
    var ma = ua & UInt64(0xFFFFFFFFFFFFF)
    var mb = ub & UInt64(0xFFFFFFFFFFFFF)
    if ea == 0:
        ea = 1
    else:
        ma |= UInt64(1) << 52
    if eb == 0:
        eb = 1
    else:
        mb |= UInt64(1) << 52
    var r = ma % mb
    for _ in range(ea - eb):
        r = (r << 1) % mb
    var bits = UInt64(0)
    if r != 0:
        # value = r * 2^(eb - 1075); normalize r to bit 52, exactly
        var s = 0
        while (r << UInt64(s)) < (UInt64(1) << 52):
            s += 1
        var m = r << UInt64(s)
        var E = eb - s
        if E >= 1:
            bits = (UInt64(E) << 52) | (m & UInt64(0xFFFFFFFFFFFFF))
        else:
            bits = m >> UInt64(1 - E)
    var v = bitcast[DType.float64](bits)
    return -v if a < 0 else v


def _round_half_even(x: Float64) -> Int:
    var f = Int(x)  # x >= 0 here
    var d = x - Float64(f)
    if d > 0.5:
        return f + 1
    if d < 0.5:
        return f
    return f + (f & 1)


def prophet_days_py(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """addrs = [src, days (N) float64 out, info (3) float64 out]; ip = [N,
    kind]: kind 0 float64 days, 1 int64 nanoseconds since 1970 (days =
    ns / 86400e9). info = [status (0 ok, 1 a NaN, 2 not ascending), span
    days[-1] - days[0], least spacing (0 for N < 2)]."""
    if len(addrs) != 3 or len(ip) != 2:
        raise Error("prophet_days: requires 3 addresses and 2 integer parameters")
    var N = Int(py=ip[0])
    var kind = Int(py=ip[1])
    if N < 1:
        raise Error("prophet_days: N >= 1")
    var dst = _f64p(addrs[1], "days")
    var info = _f64p(addrs[2], "info")
    if kind == 1:
        var src = I64P(unsafe_from_address=Int(py=addrs[0]))
        for i in range(N):
            dst[i] = Float64(src[i]) / 86400e9
    elif kind == 0:
        var src = _f64p(addrs[0], "t")
        for i in range(N):
            dst[i] = src[i]
    else:
        raise Error("prophet_days: kind must be 0 or 1")
    var status = 0
    var dmin = Float64(0)
    for i in range(N):
        if dst[i] != dst[i]:
            status = 1
            break
    if status == 0:
        for i in range(1, N):
            var g = dst[i] - dst[i - 1]
            if not (g >= 0):
                status = 2
                break
            if i == 1 or g < dmin:
                dmin = g
    info[0] = Float64(status)
    info[1] = dst[N - 1] - dst[0]
    info[2] = dmin
    return PythonObject(status)


def prophet_features_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """addrs = [days (N) float64, periods (ns) float64, t (N) float32 out,
    frac (N, max(ns, 1)) float32 out]; ip = [N, ns]; fp = [start, t_scale].
    t = (days - start) / t_scale; frac[i, s] = fmod(days, P) / P, + 1 when
    negative; one rounding to float32 each. ns == 0 writes frac zeros.
    Lane cpu4-python: on the executor (`op_prophet_prep`, soft binary64 and
    the exact integer fmod: the host float64 unit's IEEE results, the same
    bits on every column), not a host loop over the rows."""
    if len(addrs) != 4 or len(ip) != 2 or len(fp) != 2:
        raise Error("prophet_features: requires 4 addresses, 2 integer and 2 float parameters")
    var N = Int(py=ip[0])
    var ns = Int(py=ip[1])
    if N < 1 or ns < 0:
        raise Error("prophet_features: N >= 1, ns >= 0")
    var start = bitcast[DType.uint64](Float64(py=fp[0]))
    var scale = bitcast[DType.uint64](Float64(py=fp[1]))
    var days = _f64p(addrs[0], "days")
    var tsc = _f32p(addrs[2], "t")
    var frac = _f32p(addrs[3], "frac")
    var cols = max(ns, 1)
    var dper = ex.alloc(2 * cols)
    if ns > 0:
        var per = _f64p(addrs[1], "periods")
        for s in range(ns):  # small-loop(ns: user seasonalities): checks each period once, no row data
            if not (per[s] > 0):
                raise Error("prophet_features: periods must be positive")
        ex.upload(dper, F32P(unsafe_from_address=Int(py=addrs[1])), 2 * ns)
    var ddays = ex.alloc(2 * N)
    ex.upload(ddays, F32P(unsafe_from_address=Int(py=addrs[0])), 2 * N)
    var dt = ex.alloc(N)
    var df = ex.alloc(N * cols)
    var a = Args()
    a.p0 = ddays
    a.p1 = dper
    a.p2 = dt
    a.p3 = df
    a.i0 = N
    a.i1 = ns
    a.i2 = Int(start & UInt64(0xFFFF))
    a.i3 = Int((start >> 16) & UInt64(0xFFFF))
    a.i4 = Int((start >> 32) & UInt64(0xFFFF))
    a.i5 = Int((start >> 48) & UInt64(0xFFFF))
    a.i6 = Int(scale & UInt64(0xFFFF))
    a.i7 = Int((scale >> 16) & UInt64(0xFFFF))
    a.i8 = Int((scale >> 32) & UInt64(0xFFFF))
    a.i9 = Int((scale >> 48) & UInt64(0xFFFF))
    ex.launch[OP_PROPHET_PREP](a, N * (1 + cols))
    ex.sync()
    ex.download(tsc, dt, N)
    ex.download(frac, df, N * cols)
    return PythonObject(0)


def prophet_changepoints_py(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """addrs = [days (N) float64, user (n_user) float64 days (sorted here in
    place), out (max(S, 1)) float32]; ip = [N, n_user (-1: none given), hist,
    n_cp]; fp = [start, t_scale]. The caller's changepoints sorted, or
    days[round(linspace(0, hist - 1, n_cp + 1))][1:]; scaled like t; none
    gives Prophet's dummy [0]. Returns S (>= 1), or -1 when a caller's
    changepoint falls outside the data (or is NaN)."""
    if len(addrs) != 3 or len(ip) != 4 or len(fp) != 2:
        raise Error("prophet_changepoints: requires 3 addresses, 4 integer and 2 float parameters")
    var N = Int(py=ip[0])
    var n_user = Int(py=ip[1])
    var hist = Int(py=ip[2])
    var n_cp = Int(py=ip[3])
    var start = Float64(py=fp[0])
    var scale = Float64(py=fp[1])
    var days = _f64p(addrs[0], "days")
    var out = _f32p(addrs[2], "changepoints")
    var cp = List[Float64]()
    if n_user >= 0:
        if n_user > 0:
            var u = _f64p(addrs[1], "user changepoints")
            for i in range(n_user):
                var v = u[i]
                if v != v:
                    return PythonObject(-1)
                cp.append(v)
            sort(cp)
            if cp[0] < days[0] or cp[n_user - 1] > days[N - 1]:
                return PythonObject(-1)
    elif n_cp > 0:
        if hist < 1 or hist > N:
            raise Error("prophet_changepoints: hist must be in [1, N]")
        var step = Float64(hist - 1) / Float64(n_cp)
        for i in range(1, n_cp + 1):
            var idx = hist - 1 if i == n_cp else _round_half_even(Float64(i) * step)
            cp.append(days[idx])
    if len(cp) == 0:
        out[0] = 0
        return PythonObject(1)
    for i in range(len(cp)):
        out[i] = Float32((cp[i] - start) / scale)
    return PythonObject(len(cp))
