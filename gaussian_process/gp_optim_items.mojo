# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into the gp GPU binding and the gp CPU host binding; product, not only a check.
"""The kernel hyperparameter optimizer's state machine, ONE definition for the
device (`gaussian_process/gp_optim.mojo`, one block of GP_OPT_TPB threads per
stage, the state resident in device buffers) and the host column
(`bindings/_mojolearn_gp_host.mojo`, the same items called with tid = 0,
nt = 1). cgr4-device-optim-gp (2026-10-03): the optimizer was pure Python
(`_gp_optimizer.py`) calling the binding once per evaluation.

DEVIATION 2881: A PROJECTED L-BFGS WITH EXPLICIT RULES, NOT SCIPY'S L-BFGS-B.
The answer is the same bits on every column (Apple, NVIDIA, AMD and the CPU
column), not SciPy's bits. Every rule below is a function of the objective's
bits alone:

  state        FLOAT32 (Metal has no float64): theta, the gradient, the
               direction, the history and every scalar. Every product is
               `identical_mul`, every multiply-add `identical_mul_add`, every
               quotient `identical_div`, every stored value through `ftz`.
  dots         every theta-sized dot is folded blocked-then-tree: chunks of
               GP_OPT_CHUNK = 32 entries, each an ascending fused
               multiply-add chain from +0.0; then the chunk partials folded
               pairwise level by level, `part[i] = part[2i] + part[2i+1]`,
               an odd last one carried to the next level.
  objective    f = -lml and g = -grad, the float32 values the device
               likelihood leaves in its buffers. A failed factorization or a
               non-finite value is f = +inf, g = 0 (scikit-learn returns -inf
               likelihood, `_gpr.py:593`).
  theta        `x`, the natural log of every free hyperparameter; the
               hyperparameter that runs is `ftz(identical_exp(x))`, written
               into the device parameter table by the stage that moved x.
  start        run 0: `identical_log` of the kernel's float32 values; run
               r > 0: `lo + u (hi - lo)` (one `identical_mul_add`), `u` the
               restart draw `gp_restart_uniform32` (Philox, 24 bits); then
               clipped into the bounds, `lo`/`hi` = `identical_log` of the
               bounds.
  history      M = 10 pairs (SciPy's `maxcor`), a ring, newest replaces
               oldest. A pair is stored only when `s.y > EPS * y.y`,
               EPS = 2**-23 (float32's epsilon; it was 2**-52 in float64).
  active set   entry i is ACTIVE when `x_i <= lo_i and g_i > 0` or
               `x_i >= hi_i and g_i < 0`; its direction entry is zero.
  direction    the two-loop recursion over the history applied to g with the
               active entries zeroed (`q = fma(-a, y, q)`, then `q *= s.y/y.y`
               of the newest pair, then `q = fma(a - b, s, q)`), negated,
               active entries zeroed. If `g.d >= 0` the history is cleared
               and `d = -g` (active entries zeroed).
  line search  backtracking on the PROJECTED path `x(t) = clip(fma(t, d, x))`:
               t = 1, except the very first step, t = min(1, 1 / max|d_i|);
               accept when f(x(t)) is finite and
               `f(x(t)) <= fma(C1, g.(x(t) - x), f)`, C1 = 1e-4; otherwise
               t = t / 2, at most MAX_LS = 20 trials. No curvature (Wolfe)
               condition.
  stops, in    `pg <= PGTOL` (1e-5, SciPy's `pgtol`) where
  this order   `pg = max_i |clip(x_i - g_i) - x_i|`, tested before a step;
               no descent direction; the line search exhausting MAX_LS or
               reaching a trial equal to x; after an accepted step,
               `f_old - f <= FTOL * max(|f_old|, |f|, 1)` with
               FTOL = 1e7 * 2**-52 (SciPy's `factr` times float64 epsilon,
               rounded to float32); MAX_ITER = 200 accepted steps.
  residency    theta, g, d, the history and every scalar live in device
               buffers; the likelihood and its gradient are evaluated in
               place by the device kernels at the parameter table these
               stages write. The only words that come home during a run are
               the factorization's `info` (the Cholesky's own read) and this
               file's stop word, once per evaluation.
  restarts     run sequentially; the smallest f wins and a tie goes to the
               earlier run (`np.argmin`).

How it differs from SciPy's L-BFGS-B (Byrd, Lu, Nocedal and Zhu 1995): SciPy
finds a generalized Cauchy point along the projected steepest descent path and
then minimizes the quadratic model over the free variables, uses the compact
limited-memory matrix, and runs the More-Thuente line search with the strong
Wolfe conditions (`dcsrch`), all in float64. This optimizer keeps only the
two-loop direction with an active-set mask, an Armijo backtracking search on
the projected path and the stops above, in float32. On a smooth likelihood
both approach the same local maximum; the optimized theta and likelihood agree
with scikit-learn's to the precision the float32 likelihood allows, and can
differ where the likelihood has several maxima.

THE BARRIER. Each stage is written once for `nt` threads of one block: the
elementwise steps stride over theta (`p = tid, tid + nt, ...`), a fold's
chunks stride the same way, and thread 0 alone writes every scalar and folds
every chunk tree, between `_osync` barriers (a block barrier that also orders
device memory: `llvm.air.wg.barrier(3, 1)` on Apple, `barrier()` elsewhere;
nothing on the host). Every branch reads a word thread 0 wrote before a
barrier, so the block takes it together. With tid = 0, nt = 1 (the host
column) the same statements run in the same order.
"""

from std.memory import bitcast
from std.sys import llvm_intrinsic
from std.sys.info import is_apple_gpu, is_gpu
from max.gpu.sync import barrier

from checks.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_log,
    identical_mul,
    identical_mul_add,
)
from core.philox import philox4x32_10

comptime _P = MutPointer[Float32, MutAnyOrigin]
comptime _I = MutPointer[Int32, MutAnyOrigin]

#: threads of the one block every stage launches with
comptime GP_OPT_TPB = 128
#: entries per chunk of a theta-sized fold
comptime GP_OPT_CHUNK = 32
#: history pairs (SciPy's maxcor)
comptime GP_OPT_M = 10
comptime GP_OPT_MAX_LS = 20
comptime GP_OPT_MAX_ITER = 200

# the stop word, `si[GP_OPT_SI_STOP]`
comptime GP_STOP_RUNNING = 0
comptime GP_STOP_PGTOL = 1
comptime GP_STOP_NO_DESCENT = 2
comptime GP_STOP_LINE_SEARCH = 3
comptime GP_STOP_FTOL = 4
comptime GP_STOP_MAX_ITER = 5
comptime GP_STOP_NONFINITE_START = 6

# the int32 state words
comptime _SI_NITER = 0
comptime _SI_NEVAL = 1
comptime _SI_FIRST = 2
comptime _SI_K = 3
comptime _SI_HEAD = 4
comptime GP_OPT_SI_STOP = 5
comptime _SI_PHASE = 6
comptime _SI_LSCNT = 7
comptime GP_OPT_SI_BEST = 8
comptime _SI_BC = 9
comptime GP_OPT_SI_LEN = 10

# the float32 theta-sized regions, in units of T
comptime _R_X = 0
comptime _R_G = 1
comptime _R_D = 2
comptime _R_XT = 3
comptime _R_GT = 4
comptime _R_Q = 5
comptime _R_LO = 6
comptime _R_HI = 7
comptime _R_ACT = 8
comptime _R_YS = 9
comptime _R_BX = 10
comptime _R_COUNT = 11

# the float32 scalars, after the parts
comptime _SF_F = 0
comptime _SF_FT = 1
comptime _SF_T = 2
comptime _SF_BC = 3
comptime _SF_BEST = 4
comptime _SF_COUNT = 5

#: `ctr[3]` of the restart draws, ASCII "GPOR".
comptime GP_OPT_RESTART_TAG: UInt32 = 0x47504F52


@always_inline
def _osync():
    """A block barrier that also orders DEVICE memory; nothing on the host."""
    comptime if is_gpu():
        comptime if is_apple_gpu():
            llvm_intrinsic["llvm.air.wg.barrier", NoneType](Int32(3), Int32(1))
        else:
            barrier()


@always_inline
def gp_opt_chunks(t: Int) -> Int:
    return (t + GP_OPT_CHUNK - 1) // GP_OPT_CHUNK if t > 0 else 1


@always_inline
def _o(t: Int, region: Int) -> Int:
    return region * t


@always_inline
def _o_s(t: Int) -> Int:
    return _R_COUNT * t


@always_inline
def _o_y(t: Int) -> Int:
    return _R_COUNT * t + GP_OPT_M * t


@always_inline
def _o_rho(t: Int) -> Int:
    return _R_COUNT * t + 2 * GP_OPT_M * t


@always_inline
def _o_gam(t: Int) -> Int:
    return _o_rho(t) + GP_OPT_M


@always_inline
def _o_a(t: Int) -> Int:
    return _o_gam(t) + GP_OPT_M


@always_inline
def _o_parts(t: Int) -> Int:
    return _o_a(t) + GP_OPT_M


@always_inline
def _o_sc(t: Int) -> Int:
    return _o_parts(t) + gp_opt_chunks(t)


def gp_opt_st_len(t: Int) -> Int:
    """Floats of the float32 state buffer for `t` theta entries."""
    return _o_sc(t) + _SF_COUNT


@always_inline
def _inf32() -> Float32:
    return bitcast[DType.float32](UInt32(0x7F800000))


@always_inline
def _finite(v: Float32) -> Bool:
    return (bitcast[DType.uint32](v) & UInt32(0x7F800000)) != UInt32(0x7F800000)


@always_inline
def _abs(v: Float32) -> Float32:
    return -v if v < Float32(0.0) else v


@always_inline
def _clip(v: Float32, lo: Float32, hi: Float32) -> Float32:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


@always_inline
def _pgtol() -> Float32:
    return Float32(1.0e-5)


@always_inline
def _c1() -> Float32:
    return Float32(1.0e-4)


@always_inline
def _ftol() -> Float32:
    # 1e7 * 2**-52
    return Float32(2.220446049250313e-09)


@always_inline
def _eps() -> Float32:
    # 2**-23
    return bitcast[DType.float32](UInt32(0x34000000))


def gp_restart_uniform32(seed_lo: UInt32, seed_hi: UInt32, restart: Int, dim: Int) -> Float32:
    """The restart draw in [0, 1): `philox4x32_10(ctr = (restart, dim, 0,
    "GPOR"), key = (seed low, seed high))`, `u = (w0 >> 8) * 2^-24`, exact."""
    var ctr = SIMD[DType.uint32, 4](
        UInt32(restart & 0xFFFFFFFF), UInt32(dim & 0xFFFFFFFF), UInt32(0), GP_OPT_RESTART_TAG
    )
    var key = SIMD[DType.uint32, 2](seed_lo, seed_hi)
    var w = philox4x32_10(ctr, key)
    # 2^-24 as float32 bits
    return (w[0] >> UInt32(8)).cast[DType.float32]() * bitcast[DType.float32](UInt32(0x33800000))


def gp_opt_theta_map(
    kinds: List[Int32], ls_off: List[Int32], ls_len: List[Int32], free: List[Int32]
) -> List[Int32]:
    """Theta entry p's home: `m >= 0` is node m's parameter (CONST, WHITE),
    `m < 0` is length-scale table entry `-m - 1` (RBF, MATERN), in the
    postfix leaf order theta uses (DEVIATION 2880). Structural: integers of
    the kernel spec, no data."""
    var out = List[Int32]()
    for t in range(len(kinds)):
        if Int(free[t]) == 0:
            continue
        var k = Int(kinds[t])
        if k == 2 or k == 3:
            for j in range(Int(ls_len[t])):
                out.append(Int32(-(Int(ls_off[t]) + j) - 1))
        elif k == 0 or k == 1:
            out.append(Int32(t))
    return out^


@always_inline
def _param_load(tmap: _I, p: Int, dpar: _P, dls: _P) -> Float32:
    var m = Int(tmap.unsafe_load(p))
    if m >= 0:
        return dpar.unsafe_load(m)
    return dls.unsafe_load(-m - 1)


@always_inline
def _param_store(tmap: _I, p: Int, dpar: _P, dls: _P, v: Float32):
    var m = Int(tmap.unsafe_load(p))
    if m >= 0:
        dpar.unsafe_store(m, v)
    else:
        dls.unsafe_store(-m - 1, v)


@always_inline
def _bcast(tid: Int, si: _I, v: Int) -> Int:
    """Thread 0's `v` to every thread of the block."""
    if tid == 0:
        si.unsafe_store(_SI_BC, Int32(v))
    _osync()
    var r = Int(si.unsafe_load(_SI_BC))
    _osync()
    return r


def _tree(st: _P, po: Int, nch: Int):
    """The chunk partials folded pairwise level by level into `st[po]`."""
    var m = nch
    while m > 1:
        var h = m // 2
        for i in range(h):
            st.unsafe_store(po + i, ftz(st.unsafe_load(po + 2 * i) + st.unsafe_load(po + 2 * i + 1)))
        var odd = m - 2 * h
        if odd == 1:
            st.unsafe_store(po + h, st.unsafe_load(po + m - 1))
        m = h + odd


def _dot(tid: Int, nt: Int, t: Int, st: _P, ao: Int, bo: Int) -> Float32:
    """`a . b` over theta, blocked-then-tree. Every thread returns it."""
    _osync()
    var nch = gp_opt_chunks(t)
    var po = _o_parts(t)
    var c = tid
    while c < nch:
        var acc = Float32(0.0)
        var lo = c * GP_OPT_CHUNK
        var hi = min(lo + GP_OPT_CHUNK, t)
        for i in range(lo, hi):
            acc = ftz(identical_mul_add(st.unsafe_load(ao + i), st.unsafe_load(bo + i), acc))
        st.unsafe_store(po + c, acc)
        c += nt
    _osync()
    var sc = _o_sc(t)
    if tid == 0:
        _tree(st, po, nch)
        st.unsafe_store(sc + _SF_BC, st.unsafe_load(po))
    _osync()
    var r = st.unsafe_load(sc + _SF_BC)
    _osync()
    return r


def _parts_max(st: _P, po: Int, nch: Int) -> Float32:
    var m = Float32(0.0)
    for c in range(nch):
        var v = st.unsafe_load(po + c)
        if v > m:
            m = v
    return m


def _parts_sum(st: _P, po: Int, nch: Int) -> Float32:
    var s = Float32(0.0)
    for c in range(nch):
        s = s + st.unsafe_load(po + c)
    return s


def gp_opt_init_item(
    tid: Int, nt: Int, t: Int, st: _P, si: _I, tmap: _I, bnd: _P, dpar: _P, dls: _P,
    run: Int, seed_lo: UInt32, seed_hi: UInt32,
):
    """Run `run`'s start: bounds, the clipped start in XT, its parameters
    written, the counters cleared. The next evaluation is the start's."""
    var p = tid
    while p < t:
        var lo = ftz(identical_log(ftz(bnd.unsafe_load(2 * p))))
        var hi = ftz(identical_log(ftz(bnd.unsafe_load(2 * p + 1))))
        st.unsafe_store(_o(t, _R_LO) + p, lo)
        st.unsafe_store(_o(t, _R_HI) + p, hi)
        var x0 = Float32(0.0)
        if run == 0:
            x0 = ftz(identical_log(ftz(_param_load(tmap, p, dpar, dls))))
        else:
            var u = gp_restart_uniform32(seed_lo, seed_hi, run - 1, p)
            x0 = ftz(identical_mul_add(u, ftz(hi - lo), lo))
        var xt = _clip(x0, lo, hi)
        st.unsafe_store(_o(t, _R_XT) + p, xt)
        _param_store(tmap, p, dpar, dls, ftz(identical_exp(xt)))
        p += nt
    if tid == 0:
        si.unsafe_store(_SI_NITER, Int32(0))
        si.unsafe_store(_SI_NEVAL, Int32(0))
        si.unsafe_store(_SI_FIRST, Int32(1))
        si.unsafe_store(_SI_K, Int32(0))
        si.unsafe_store(_SI_HEAD, Int32(0))
        si.unsafe_store(GP_OPT_SI_STOP, Int32(GP_STOP_RUNNING))
        si.unsafe_store(_SI_PHASE, Int32(0))
        si.unsafe_store(_SI_LSCNT, Int32(0))
        var sc = _o_sc(t)
        st.unsafe_store(sc + _SF_F, Float32(0.0))
        st.unsafe_store(sc + _SF_FT, Float32(0.0))
        st.unsafe_store(sc + _SF_T, Float32(1.0))


def _trial(tid: Int, nt: Int, t: Int, st: _P, si: _I, tmap: _I, dpar: _P, dls: _P):
    """`xt = clip(fma(t, d, x))`; a trial equal to x stops the run
    (line-search); otherwise its parameters are written."""
    var ts = st.unsafe_load(_o_sc(t) + _SF_T)
    var nch = gp_opt_chunks(t)
    var po = _o_parts(t)
    var c = tid
    while c < nch:
        var cnt = Float32(0.0)
        var lo = c * GP_OPT_CHUNK
        var hi = min(lo + GP_OPT_CHUNK, t)
        for i in range(lo, hi):
            var xi = st.unsafe_load(_o(t, _R_X) + i)
            var v = ftz(identical_mul_add(ts, st.unsafe_load(_o(t, _R_D) + i), xi))
            var xt = _clip(v, st.unsafe_load(_o(t, _R_LO) + i), st.unsafe_load(_o(t, _R_HI) + i))
            st.unsafe_store(_o(t, _R_XT) + i, xt)
            if xt != xi:
                cnt = cnt + Float32(1.0)
        st.unsafe_store(po + c, cnt)
        c += nt
    _osync()
    var sv = 0
    if tid == 0:
        if _parts_sum(st, po, nch) == Float32(0.0):
            sv = GP_STOP_LINE_SEARCH
            si.unsafe_store(GP_OPT_SI_STOP, Int32(sv))
    if _bcast(tid, si, sv) != 0:
        return
    var p = tid
    while p < t:
        _param_store(tmap, p, dpar, dls, ftz(identical_exp(st.unsafe_load(_o(t, _R_XT) + p))))
        p += nt


def _prepare(tid: Int, nt: Int, t: Int, st: _P, si: _I, tmap: _I, dpar: _P, dls: _P):
    """The top of an iteration: the max-iter and pgtol stops, the active set,
    the two-loop direction, the first step length, the first trial."""
    var sv = 0
    if tid == 0 and Int(si.unsafe_load(_SI_NITER)) >= GP_OPT_MAX_ITER:
        sv = GP_STOP_MAX_ITER
        si.unsafe_store(GP_OPT_SI_STOP, Int32(sv))
    if _bcast(tid, si, sv) != 0:
        return
    var nch = gp_opt_chunks(t)
    var po = _o_parts(t)
    var ox = _o(t, _R_X)
    var og = _o(t, _R_G)
    var od = _o(t, _R_D)
    var oq = _o(t, _R_Q)
    var olo = _o(t, _R_LO)
    var ohi = _o(t, _R_HI)
    var oact = _o(t, _R_ACT)
    # pg = max |clip(x - g) - x|
    var c = tid
    while c < nch:
        var m = Float32(0.0)
        var lo = c * GP_OPT_CHUNK
        var hi = min(lo + GP_OPT_CHUNK, t)
        for i in range(lo, hi):
            var xi = st.unsafe_load(ox + i)
            var cl = _clip(ftz(xi - st.unsafe_load(og + i)), st.unsafe_load(olo + i), st.unsafe_load(ohi + i))
            var v = _abs(ftz(cl - xi))
            if v > m:
                m = v
        st.unsafe_store(po + c, m)
        c += nt
    _osync()
    sv = 0
    if tid == 0:
        if _parts_max(st, po, nch) <= _pgtol():
            sv = GP_STOP_PGTOL
            si.unsafe_store(GP_OPT_SI_STOP, Int32(sv))
    if _bcast(tid, si, sv) != 0:
        return
    # the active set and q = g with the active entries zeroed
    var p = tid
    while p < t:
        var xi = st.unsafe_load(ox + p)
        var gi = st.unsafe_load(og + p)
        var act = (xi <= st.unsafe_load(olo + p) and gi > Float32(0.0)) or (
            xi >= st.unsafe_load(ohi + p) and gi < Float32(0.0)
        )
        st.unsafe_store(oact + p, Float32(1.0) if act else Float32(0.0))
        st.unsafe_store(oq + p, Float32(0.0) if act else gi)
        p += nt
    var k = Int(si.unsafe_load(_SI_K))
    var head = Int(si.unsafe_load(_SI_HEAD))
    var os = _o_s(t)
    var oy = _o_y(t)
    var orho = _o_rho(t)
    var oa = _o_a(t)
    for pp in range(k - 1, -1, -1):
        var slot = (head + pp) % GP_OPT_M
        var a = ftz(identical_mul(st.unsafe_load(orho + slot), _dot(tid, nt, t, st, os + slot * t, oq)))
        if tid == 0:
            st.unsafe_store(oa + pp, a)
        p = tid
        while p < t:
            st.unsafe_store(oq + p, ftz(identical_mul_add(-a, st.unsafe_load(oy + slot * t + p), st.unsafe_load(oq + p))))
            p += nt
    if k > 0:
        var gam = st.unsafe_load(_o_gam(t) + (head + k - 1) % GP_OPT_M)
        p = tid
        while p < t:
            st.unsafe_store(oq + p, ftz(identical_mul(gam, st.unsafe_load(oq + p))))
            p += nt
    for pp in range(k):
        var slot = (head + pp) % GP_OPT_M
        var b = ftz(identical_mul(st.unsafe_load(orho + slot), _dot(tid, nt, t, st, oy + slot * t, oq)))
        var coef = ftz(st.unsafe_load(oa + pp) - b)
        p = tid
        while p < t:
            st.unsafe_store(oq + p, ftz(identical_mul_add(coef, st.unsafe_load(os + slot * t + p), st.unsafe_load(oq + p))))
            p += nt
    p = tid
    while p < t:
        var act = st.unsafe_load(oact + p) != Float32(0.0)
        st.unsafe_store(od + p, Float32(0.0) if act else ftz(-st.unsafe_load(oq + p)))
        p += nt
    var gd = _dot(tid, nt, t, st, og, od)
    if not (gd < Float32(0.0)):
        if tid == 0:
            si.unsafe_store(_SI_K, Int32(0))
            si.unsafe_store(_SI_HEAD, Int32(0))
        p = tid
        while p < t:
            var act = st.unsafe_load(oact + p) != Float32(0.0)
            st.unsafe_store(od + p, Float32(0.0) if act else ftz(-st.unsafe_load(og + p)))
            p += nt
        gd = _dot(tid, nt, t, st, og, od)
        if not (gd < Float32(0.0)):
            sv = 0
            if tid == 0:
                sv = GP_STOP_NO_DESCENT
                si.unsafe_store(GP_OPT_SI_STOP, Int32(sv))
            _ = _bcast(tid, si, sv)
            return
    var first = Int(si.unsafe_load(_SI_FIRST))
    var step = Float32(1.0)
    if first != 0:
        c = tid
        while c < nch:
            var m = Float32(0.0)
            var lo = c * GP_OPT_CHUNK
            var hi = min(lo + GP_OPT_CHUNK, t)
            for i in range(lo, hi):
                var v = _abs(st.unsafe_load(od + i))
                if v > m:
                    m = v
            st.unsafe_store(po + c, m)
            c += nt
        _osync()
        if tid == 0:
            var dmax = _parts_max(st, po, nch)
            if dmax > Float32(1.0):
                step = ftz(identical_div(Float32(1.0), dmax))
    if tid == 0:
        st.unsafe_store(_o_sc(t) + _SF_T, step)
        si.unsafe_store(_SI_LSCNT, Int32(0))
    _osync()
    _trial(tid, nt, t, st, si, tmap, dpar, dls)


def gp_opt_step_item(
    tid: Int, nt: Int, t: Int, st: _P, si: _I, tmap: _I, dpar: _P, dls: _P,
    lml: _P, graw: _P, info: Int,
):
    """After an evaluation at XT (`lml`, `graw` the likelihood and its
    gradient there, `info` the factorization's): the start's bookkeeping or
    the Armijo test, then the next direction or the next trial. On return
    `si[GP_OPT_SI_STOP]` is the stop word (0: evaluate at the new XT)."""
    var nch = gp_opt_chunks(t)
    var po = _o_parts(t)
    var sc = _o_sc(t)
    var ox = _o(t, _R_X)
    var og = _o(t, _R_G)
    var oxt = _o(t, _R_XT)
    var ogt = _o(t, _R_GT)
    var oq = _o(t, _R_Q)
    var oys = _o(t, _R_YS)
    var c = tid
    while c < nch:
        var cnt = Float32(0.0)
        var lo = c * GP_OPT_CHUNK
        var hi = min(lo + GP_OPT_CHUNK, t)
        if info == 0:
            for i in range(lo, hi):
                if not _finite(graw.unsafe_load(i)):
                    cnt = cnt + Float32(1.0)
        st.unsafe_store(po + c, cnt)
        c += nt
    _osync()
    var okv = 0
    if tid == 0:
        if info == 0 and _finite(lml.unsafe_load(0)) and _parts_sum(st, po, nch) == Float32(0.0):
            okv = 1
    var ok = _bcast(tid, si, okv) != 0
    var p = tid
    while p < t:
        st.unsafe_store(ogt + p, ftz(-graw.unsafe_load(p)) if ok else Float32(0.0))
        p += nt
    var phv = 0
    if tid == 0:
        st.unsafe_store(sc + _SF_FT, ftz(-lml.unsafe_load(0)) if ok else _inf32())
        si.unsafe_store(_SI_NEVAL, si.unsafe_load(_SI_NEVAL) + Int32(1))
        phv = Int(si.unsafe_load(_SI_PHASE))
    var phase = _bcast(tid, si, phv)
    if phase == 0:
        p = tid
        while p < t:
            st.unsafe_store(ox + p, st.unsafe_load(oxt + p))
            st.unsafe_store(og + p, st.unsafe_load(ogt + p))
            p += nt
        var sv0 = 0
        if tid == 0:
            var f0 = st.unsafe_load(sc + _SF_FT)
            st.unsafe_store(sc + _SF_F, f0)
            si.unsafe_store(_SI_PHASE, Int32(1))
            if not _finite(f0):
                sv0 = GP_STOP_NONFINITE_START
                si.unsafe_store(GP_OPT_SI_STOP, Int32(sv0))
        if _bcast(tid, si, sv0) == 0:
            _prepare(tid, nt, t, st, si, tmap, dpar, dls)
        return
    # the Armijo test on the projected path
    p = tid
    while p < t:
        st.unsafe_store(oq + p, ftz(st.unsafe_load(oxt + p) - st.unsafe_load(ox + p)))
        p += nt
    var gs = _dot(tid, nt, t, st, og, oq)
    var accv = 0
    if tid == 0:
        var ft = st.unsafe_load(sc + _SF_FT)
        var f = st.unsafe_load(sc + _SF_F)
        if _finite(ft) and ft <= ftz(identical_mul_add(_c1(), gs, f)):
            accv = 1
    if _bcast(tid, si, accv) != 0:
        p = tid
        while p < t:
            st.unsafe_store(oys + p, ftz(st.unsafe_load(ogt + p) - st.unsafe_load(og + p)))
            p += nt
        var sy = _dot(tid, nt, t, st, oq, oys)
        var yy = _dot(tid, nt, t, st, oys, oys)
        var slotv = -1
        if tid == 0:
            if sy > ftz(identical_mul(_eps(), yy)):
                var k = Int(si.unsafe_load(_SI_K))
                var head = Int(si.unsafe_load(_SI_HEAD))
                var sl = 0
                if k < GP_OPT_M:
                    sl = (head + k) % GP_OPT_M
                    si.unsafe_store(_SI_K, Int32(k + 1))
                else:
                    sl = head
                    si.unsafe_store(_SI_HEAD, Int32((head + 1) % GP_OPT_M))
                st.unsafe_store(_o_rho(t) + sl, ftz(identical_div(Float32(1.0), sy)))
                st.unsafe_store(_o_gam(t) + sl, ftz(identical_div(sy, yy)))
                slotv = sl
        var slot_all = _bcast(tid, si, slotv)
        p = tid
        while p < t:
            if slot_all >= 0:
                st.unsafe_store(_o_s(t) + slot_all * t + p, st.unsafe_load(oq + p))
                st.unsafe_store(_o_y(t) + slot_all * t + p, st.unsafe_load(oys + p))
            st.unsafe_store(ox + p, st.unsafe_load(oxt + p))
            st.unsafe_store(og + p, st.unsafe_load(ogt + p))
            p += nt
        var sv1 = 0
        if tid == 0:
            var f_old = st.unsafe_load(sc + _SF_F)
            var f = st.unsafe_load(sc + _SF_FT)
            st.unsafe_store(sc + _SF_F, f)
            si.unsafe_store(_SI_NITER, si.unsafe_load(_SI_NITER) + Int32(1))
            si.unsafe_store(_SI_FIRST, Int32(0))
            var m = _abs(f_old)
            if _abs(f) > m:
                m = _abs(f)
            if Float32(1.0) > m:
                m = Float32(1.0)
            if ftz(f_old - f) <= ftz(identical_mul(_ftol(), m)):
                sv1 = GP_STOP_FTOL
                si.unsafe_store(GP_OPT_SI_STOP, Int32(sv1))
        if _bcast(tid, si, sv1) == 0:
            _prepare(tid, nt, t, st, si, tmap, dpar, dls)
        return
    var sv2 = 0
    if tid == 0:
        st.unsafe_store(sc + _SF_T, ftz(identical_mul(Float32(0.5), st.unsafe_load(sc + _SF_T))))
        var n = Int(si.unsafe_load(_SI_LSCNT)) + 1
        si.unsafe_store(_SI_LSCNT, Int32(n))
        if n >= GP_OPT_MAX_LS:
            sv2 = GP_STOP_LINE_SEARCH
            si.unsafe_store(GP_OPT_SI_STOP, Int32(sv2))
    if _bcast(tid, si, sv2) == 0:
        _trial(tid, nt, t, st, si, tmap, dpar, dls)


def gp_opt_run_end_item(tid: Int, nt: Int, t: Int, st: _P, si: _I, rec: _P, run: Int):
    """Run `run`'s record (n_iter, n_eval, stop, f) and the running best:
    the smallest f, a tie to the earlier run."""
    var sc = _o_sc(t)
    var bv = 0
    if tid == 0:
        var f = st.unsafe_load(sc + _SF_F)
        rec.unsafe_store(run * 4 + 0, Float32(Int(si.unsafe_load(_SI_NITER))))
        rec.unsafe_store(run * 4 + 1, Float32(Int(si.unsafe_load(_SI_NEVAL))))
        rec.unsafe_store(run * 4 + 2, Float32(Int(si.unsafe_load(GP_OPT_SI_STOP))))
        rec.unsafe_store(run * 4 + 3, f)
        if run == 0 or f < st.unsafe_load(sc + _SF_BEST):
            st.unsafe_store(sc + _SF_BEST, f)
            si.unsafe_store(GP_OPT_SI_BEST, Int32(run))
            bv = 1
    if _bcast(tid, si, bv) != 0:
        var p = tid
        while p < t:
            st.unsafe_store(_o(t, _R_BX) + p, st.unsafe_load(_o(t, _R_X) + p))
            p += nt


def gp_opt_final_item(tid: Int, nt: Int, t: Int, st: _P, tmap: _I, dpar: _P, dls: _P, out: _P):
    """The winner: `out[p]` its theta, `out[t + p]` the float32
    hyperparameter that runs there, also written into the parameter table."""
    var p = tid
    while p < t:
        var x = st.unsafe_load(_o(t, _R_BX) + p)
        var v = ftz(identical_exp(x))
        _param_store(tmap, p, dpar, dls, v)
        out.unsafe_store(p, x)
        out.unsafe_store(t + p, v)
        p += nt
