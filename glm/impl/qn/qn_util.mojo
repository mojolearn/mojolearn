# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`LBFGSParam`, the return codes, `check_convergence`, `lbfgs_search_dir`.

Reference: `cuml/cpp/src/glm/qn/qn_util.cuh` (cuML `00094f7`). WHOLE FILE
since 2026-09-01: `project_orth`, `get_pseudo_grad`, `op_project` and
`op_pseudo_grad` -- OWL-QN's four operators -- are implemented at the bottom,
with the two device kernels that apply them elementwise and their host
wrappers.

WHY THE KERNELS ARE HERE AND NOT IN `simple_mat/dense.mojo`. In the reference these
are functors passed to `SimpleVec::assign_binary`, so the arithmetic lives
in `qn_util.cuh` and the loop lives in `dense.hpp`. Here the loop is a
launch, and a launch that calls `get_pseudo_grad` has to be in a module that
can see it. `qn_util.mojo` already imports `simple_mat/dense.mojo`, so
putting the kernels in `dense.mojo` would close an import cycle. They are
placed beside the operator they apply, which is also where a reader looking
for `op_pseudo_grad` will look.

THE HOST SCALARS ARE FLOAT32 AND EVERY ONE OF THEM STEERS A BRANCH
-----------------------------------------------------------------
`T = float` on the Python float32 path, so `ys`, `yy`, `alpha[j]`, `beta`,
`step`, `fmag` are HOST Float32 and the comparisons below -- the skipping
test `ys <= eps * yy`, the convergence test `gnorm <= epsilon * fmag`, the
insufficient-change test -- are branches on them. The inputs to every one
are device reductions that `simple_mat/dense.mojo` pins (DEVIATION 547);
the host arithmetic between them is single IEEE operations (`/`, `*`, `-`,
`<=`, `max`, `abs`), which round the same on every host, plus one
contraction candidate handled in `qn_linesearch.mojo`. So under IDENTICAL
the branch sequence -- and therefore the ITERATION COUNT and the L-BFGS
history -- is a function of the inputs alone, and the count is recorded on
the card (`qn.n_iter`) as the certificate's integer stage.

`std::numeric_limits<T>::epsilon()` for float is `2^-23`; written as the
literal below and not derived.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.sys.info import is_apple_gpu
from std.memory import bitcast

from core.column_stats import STATS_TPB
from core.pinned_reduce import pinned_block_sum
from glm.impl.qn.simple_mat.dense import (
    VEC_ELEM_TPB,
    ax,
    ax_inplace,
    axpy_inplace,
    copy_vec,
    dot,
    dot_kernel,
    dot_self_kernel,
    ieee_div_f32,
    ieee_sub_f32,
    read_scalars,
    squared_norm,
)
from glm.impl.linear_model.qn import QNParams
from checks.numerics import ftz, identical_mul_add


# `LINE_SEARCH_ALGORITHM`, `qn_util.cuh:30-35`
comptime LBFGS_LS_BT_ARMIJO = 1
comptime LBFGS_LS_BT = 2
comptime LBFGS_LS_BT_WOLFE = 2
comptime LBFGS_LS_BT_STRONG_WOLFE = 3

# `LINE_SEARCH_RETCODE`, `:37-44`
comptime LS_SUCCESS = 0
comptime LS_INVALID_STEP_MIN = 1
comptime LS_INVALID_STEP_MAX = 2
comptime LS_MAX_ITERS_REACHED = 3
comptime LS_INVALID_DIR = 4
comptime LS_INVALID_STEP = 5

# `OPT_RETCODE`, `:46-52`
comptime OPT_SUCCESS = 0
comptime OPT_NUMERIC_ERROR = 1
comptime OPT_LS_FAILED = 2
comptime OPT_MAX_ITERS_REACHED = 3
comptime OPT_INVALID_ARGS = 4

#: `std::numeric_limits<float>::epsilon()`
comptime FLOAT_EPSILON = Float32(1.1920928955078125e-7)


@fieldwise_init
struct LBFGSParam(ImplicitlyCopyable, Copyable, Movable):
    """`LBFGSParam<T>` (`qn_util.cuh:54-122`) with T = float."""

    var m: Int
    var epsilon: Float32
    var past: Int
    var delta: Float32
    var max_iterations: Int
    var linesearch: Int
    var max_linesearch: Int
    var min_step: Float32
    var max_step: Float32
    var ftol: Float32
    var wolfe: Float32
    var ls_dec: Float32
    var ls_inc: Float32

    @staticmethod
    def defaults() -> Self:
        """The default constructor, `qn_util.cuh:71-86`. NOTE `linesearch =
        LBFGS_LS_BT_ARMIJO`: cuML's shipped line search is backtracking
        Armijo; the Wolfe arms of `ls_success` exist and are not reached
        from the Python door, which never sets this field."""
        return Self(
            6, Float32(1e-5), 0, Float32(0.0), 0, LBFGS_LS_BT_ARMIJO, 20,
            Float32(1e-20), Float32(1e20), Float32(1e-4), Float32(0.9),
            Float32(0.5), Float32(2.1),
        )

    @staticmethod
    def from_params(pams: QNParams) -> Self:
        """`explicit LBFGSParam(const qn_params&)`, `qn_util.cuh:88-98`:
        `m = lbfgs_memory`, `epsilon = grad_tol`, `past = change_tol > 0 ?
        10 : 0`, `delta = change_tol`, `max_iterations = max_iter`,
        `max_linesearch = linesearch_max_iter`, `ftol = change_tol > 0 ?
        change_tol * 0.1 : 1e-4`. The products are double then narrowed,
        as `T(pams.change_tol * 0.1)` is."""
        var p = Self.defaults()
        p.m = pams.lbfgs_memory
        p.epsilon = Float32(pams.grad_tol)
        p.past = 10 if pams.change_tol > 0.0 else 0
        p.delta = Float32(pams.change_tol)
        p.max_iterations = pams.max_iter
        p.max_linesearch = pams.linesearch_max_iter
        p.ftol = Float32(pams.change_tol * 0.1) if pams.change_tol > 0.0 else Float32(1e-4)
        return p^

    def check_param(self) -> Int:
        """`check_param`, `qn_util.cuh:100-121`: 0 if valid, else the 1-based
        index of the first failing test."""
        var ret = 1
        if self.m <= 0:
            return ret
        ret += 1
        if self.epsilon <= Float32(0.0):
            return ret
        ret += 1
        if self.past < 0:
            return ret
        ret += 1
        if self.delta < Float32(0.0):
            return ret
        ret += 1
        if self.max_iterations < 0:
            return ret
        ret += 1
        if self.linesearch < LBFGS_LS_BT_ARMIJO or self.linesearch > LBFGS_LS_BT_STRONG_WOLFE:
            return ret
        ret += 1
        if self.max_linesearch <= 0:
            return ret
        ret += 1
        if self.min_step < Float32(0.0):
            return ret
        ret += 1
        if self.max_step < self.min_step:
            return ret
        ret += 1
        if self.ftol <= Float32(0.0) or self.ftol >= Float32(0.5):
            return ret
        ret += 1
        if self.wolfe <= self.ftol or self.wolfe >= Float32(1.0):
            return ret
        ret += 1
        return 0


def check_convergence(
    param: LBFGSParam,
    k: Int,
    fx: Float32,
    gnorm: Float32,
    mut fx_hist: List[Float32],
) -> Bool:
    """`check_convergence`, `qn_util.cuh:147-169`."""
    var fmag = max(fx, param.epsilon)
    if gnorm <= param.epsilon * fmag:
        return True
    if param.past > 0:
        if k >= param.past and abs(fx_hist[k % param.past] - fx) <= param.delta * fmag:
            return True
        fx_hist[k % param.past] = fx
    return False


@always_inline
def _block_dot_bcast(
    u: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    n: Int,
    tid: Int,
) -> Float32:
    """`dense.dot_kernel`'s value, character for character (the same
    strided `identical_mul_add` partials over STATS_TPB threads, the same
    pinned fold, the same ftz), handed to every thread of the block through
    one threadgroup word."""
    var acc = Float32(0.0)
    var i = tid
    while i < n:
        acc = identical_mul_add(u.unsafe_load(i), v.unsafe_load(i), acc)
        i += STATS_TPB
    var s0 = ftz(pinned_block_sum[STATS_TPB](acc))
    var slot = stack_allocation[
        1, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    if tid == 0:
        slot[0] = s0
    barrier()
    var r = slot[0]
    barrier()
    return r


def host_le_eps_times(ys: Float32, yy: Float32) -> Bool:
    """The host's `ys <= FLOAT_EPSILON * yy` (FLOAT_EPSILON = 2^-23) on any
    target, BY BITS. The product is formed as the host rounds it (an
    exponent shift while it stays normal; below that, `yy`'s value in
    2^-149 quanta shifted right by 23 and rounded to nearest even), and the
    comparison is the IEEE one on the two words (NaN compares false, -0 ==
    +0). `ys` is an ftz'd reduction, never subnormal; `yy` a squared norm."""
    var bs = bitcast[DType.uint32](ys)
    var by = bitcast[DType.uint32](yy)
    if (bs & UInt32(0x7FFFFFFF)) > UInt32(0x7F800000) or (by & UInt32(0x7FFFFFFF)) > UInt32(0x7F800000):
        return False
    var sign = by & UInt32(0x80000000)
    var ey = Int((by >> 23) & UInt32(0xFF))
    var mag: UInt32
    if ey == 0xFF:
        mag = UInt32(0x7F800000)
    elif ey > 23:
        mag = (by & UInt32(0x7FFFFFFF)) - (UInt32(23) << 23)
    else:
        var q: UInt64
        var f = (by & UInt32(0x7FFFFF)).cast[DType.uint64]()
        if ey == 0:
            q = f
        else:
            q = (f | UInt64(0x800000)) << UInt64(ey - 1)
        var m = q >> 23
        var rem = q & UInt64(0x7FFFFF)
        if rem > UInt64(0x400000) or (rem == UInt64(0x400000) and (m & UInt64(1)) == UInt64(1)):
            m += 1
        mag = m.cast[DType.uint32]()
    var bp = sign | mag
    if (bs & UInt32(0x7FFFFFFF)) == UInt32(0):
        bs = UInt32(0)
    if (bp & UInt32(0x7FFFFFFF)) == UInt32(0):
        bp = UInt32(0)
    # IEEE order on non-NaN words: the usual sign-flip key
    var ks = (bs ^ UInt32(0x80000000)) if (bs & UInt32(0x80000000)) == UInt32(0) else ~bs
    var kp = (bp ^ UInt32(0x80000000)) if (bp & UInt32(0x80000000)) == UInt32(0) else ~bp
    return ks <= kp


#: Largest L-BFGS memory the fused direction supports (its alpha history
#: lives in threadgroup memory).
comptime LBFGS_FUSED_MAX_M = 256


@always_inline
def _dev_barrier():
    """The block barrier of `lbfgs_dir_kernel`. It orders THREADGROUP memory
    only on Apple (`threadgroup_barrier(mem_threadgroup)`), and that is all
    the kernel needs: no thread reads a DEVICE word another thread of the
    launch wrote. Every elementwise pass and every dot's strided partial use
    the same partition (index i belongs to thread i mod STATS_TPB), so `drt`,
    `S`, `Y`, `x`, `xp`, `grad`, `gradp` are only ever read by the thread that
    wrote them; the dots cross threads through `pinned_block_sum`'s
    threadgroup memory; `alpha` is threadgroup memory; `yhist[end]` written
    here is read from the register `ys`. (A release/acquire fence spelling of
    a device barrier, 33cd0d549, made Apple's Metal compiler service crash
    creating the pipeline: XPC_ERROR_CONNECTION_INTERRUPTED, M3 Ultra.)"""
    barrier()


def lbfgs_dir_kernel(
    drt: MutPointer[Float32, MutAnyOrigin],
    pseudo: MutPointer[Float32, MutAnyOrigin],
    s_all: MutPointer[Float32, MutAnyOrigin],
    y_all: MutPointer[Float32, MutAnyOrigin],
    yhist: MutPointer[Float32, MutAnyOrigin],
    alpha_unused: MutPointer[Float32, MutAnyOrigin],
    verdict: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    xp: MutPointer[Float32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    gradp: MutPointer[Float32, MutAnyOrigin],
    dg_out: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    m_in: Int32,
    n_vec_in: Int32,
    end_in: Int32,
    neg_one: Float32,
    use_pseudo: Int32,
    do_dot: Int32,
):
    """lane/linear-apple: ALL of `lbfgs_search_dir` in ONE launch of ONE
    block of STATS_TPB threads (grid 1), with no synchronize.

    `ys = dot(S[end], Y[end])` and `yy = squaredNorm(Y[end])` are
    `dot_kernel` / `dot_self_kernel`'s values (`_block_dot_bcast`); the
    skipping test is `host_le_eps_times`, the host's comparison by bits;
    `verdict[0]` gets 1.0 (skipped: `drt` untouched, as the host returned
    before touching it) or 0.0, and `verdict[1]` gets `ys`, for the host's
    bookkeeping (`lbfgs_search_dir_resolve`). Then the two loops:
    `drt = neg_one * g` is `ax_kernel`; each dot is `dot_kernel`; each
    update is `axpy_inplace_kernel`'s `ftz(identical_mul_add(a, x, drt))`;
    `drt *= ys / yy` is `ax_inplace_kernel`. The host scalars `ys / yy`,
    `alpha[j] = dot / yhist[j]`, `-alpha[j]`, `beta = dot / yhist[j]` and
    `alpha[j] - beta` are `ieee_div_f32` / `ieee_sub_f32` / negation, the
    host's IEEE words. A barrier separates every write of `drt` from the
    next read of it by another thread. The skip decision is uniform (every
    thread holds the same `ys`, `yy`), so every barrier is reached by all.

    lane/linear-apple, launch fusion (Metal pays per launch): the kernel
    first does the solver's two history updates, `S[end] = -1 * xp + x` and
    `Y[end] = -1 * gradp + grad` (`axpy_kernel`'s `ftz(identical_mul_add)`),
    and the NEXT iteration's two saves `xp = x`, `gradp = grad` (each thread
    its own indices, after reading them). The direction's source `g` is
    `pseudo` when `use_pseudo` (OWL-QN) else `grad`. With `do_dot` (L-BFGS)
    it ends with the line search's `dg_init = dot(grad, drt)`
    (`dot_kernel`'s value) into `dg_out[0]`, skipped pair or not, as the
    reference computes it either way."""
    var n = Int(n_in)
    var m = Int(m_in)
    var tid = Int(thread_idx.x)
    var end_prev = Int(end_in)
    var g = pseudo if Int(use_pseudo) != 0 else grad
    var sw = s_all + end_prev * n
    var yw = y_all + end_prev * n
    var i0 = tid
    while i0 < n:
        var xi = x.unsafe_load(i0)
        var gi = grad.unsafe_load(i0)
        sw.unsafe_store(i0, ftz(identical_mul_add(neg_one, xp.unsafe_load(i0), xi)))
        yw.unsafe_store(i0, ftz(identical_mul_add(neg_one, gradp.unsafe_load(i0), gi)))
        xp.unsafe_store(i0, xi)
        gradp.unsafe_store(i0, gi)
        i0 += STATS_TPB
    _dev_barrier()
    var ys = _block_dot_bcast(s_all + end_prev * n, y_all + end_prev * n, n, tid)
    var yv = y_all + end_prev * n
    var yy = _block_dot_bcast(yv, yv, n, tid)
    if host_le_eps_times(ys, yy):
        if tid == 0:
            verdict.unsafe_store(0, Float32(1.0))
            verdict.unsafe_store(1, ys)
    else:
        _two_loop(drt, g, s_all, y_all, yhist, verdict, n, m, tid,
                  end_prev, Int(n_vec_in), neg_one, ys, yy)
    if Int(do_dot) != 0:
        var dg = _block_dot_bcast(grad, drt, n, tid)
        if tid == 0:
            dg_out.unsafe_store(0, dg)


@always_inline
def _two_loop(
    drt: MutPointer[Float32, MutAnyOrigin],
    g: MutPointer[Float32, MutAnyOrigin],
    s_all: MutPointer[Float32, MutAnyOrigin],
    y_all: MutPointer[Float32, MutAnyOrigin],
    yhist: MutPointer[Float32, MutAnyOrigin],
    verdict: MutPointer[Float32, MutAnyOrigin],
    n: Int,
    m: Int,
    tid: Int,
    end_prev: Int,
    n_vec_prev: Int,
    neg_one: Float32,
    ys: Float32,
    yy: Float32,
):
    """`lbfgs_dir_kernel`'s not-skipped branch (see there)."""
    if tid == 0:
        verdict.unsafe_store(0, Float32(0.0))
        verdict.unsafe_store(1, ys)
        yhist.unsafe_store(end_prev, ys)
    var bound = min(m, n_vec_prev + 1)
    var scale = ieee_div_f32(ys, yy)
    var alpha = stack_allocation[
        LBFGS_FUSED_MAX_M, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var i = tid
    while i < n:
        drt.unsafe_store(i, ftz(neg_one * g.unsafe_load(i)))
        i += STATS_TPB
    _dev_barrier()
    var j = (end_prev + 1) % m
    for _ in range(bound):
        j = (j + m - 1) % m
        var d = _block_dot_bcast(s_all + j * n, drt, n, tid)
        var yh = ys if j == end_prev else yhist.unsafe_load(j)
        var a = ieee_div_f32(d, yh)
        if tid == 0:
            alpha[j] = a
        var na = -a
        var yj = y_all + j * n
        i = tid
        while i < n:
            drt.unsafe_store(
                i, ftz(identical_mul_add(na, yj.unsafe_load(i), drt.unsafe_load(i)))
            )
            i += STATS_TPB
        _dev_barrier()
    i = tid
    while i < n:
        drt.unsafe_store(i, ftz(scale * drt.unsafe_load(i)))
        i += STATS_TPB
    _dev_barrier()
    for _ in range(bound):
        var d = _block_dot_bcast(y_all + j * n, drt, n, tid)
        var yh = ys if j == end_prev else yhist.unsafe_load(j)
        var beta = ieee_div_f32(d, yh)
        var c = ieee_sub_f32(alpha[j], beta)
        var sj = s_all + j * n
        i = tid
        while i < n:
            drt.unsafe_store(
                i, ftz(identical_mul_add(c, sj.unsafe_load(i), drt.unsafe_load(i)))
            )
            i += STATS_TPB
        _dev_barrier()
        j = (j + 1) % m


def lbfgs_search_dir_enqueue(
    ctx: DeviceContext,
    param: LBFGSParam,
    n_vec: Int,
    end_prev: Int,
    mut s_all: DeviceBuffer[DType.float32],
    mut y_all: DeviceBuffer[DType.float32],
    mut hist: DeviceBuffer[DType.float32],
    mut pseudo: DeviceBuffer[DType.float32],
    use_pseudo: Bool,
    mut drt: DeviceBuffer[DType.float32],
    n: Int,
    mut scalar: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut xp: DeviceBuffer[DType.float32],
    mut grad: DeviceBuffer[DType.float32],
    mut gradp: DeviceBuffer[DType.float32],
    do_dot: Bool,
) raises:
    """`lbfgs_search_dir`, `qn_util.cuh:176-241` (`drt = -H g` by the
    two-loop recursion over the `S`, `Y` history), enqueued as ONE launch
    with no synchronize (lane/linear-apple, 2026-09-28). The host learns
    whether the pair was skipped, and `ys`, from `scalar` words 2 and 3,
    which come home with the next read of `scalar` (the line search's
    `dg_init`); `lbfgs_search_dir_resolve` then does the host's bookkeeping.
    Before this the function synchronized 2 + 2 * min(m, n_vec) times per
    iteration. `S[j]`/`Y[j]` are rows j of `s_all`/`y_all` (m x n); `hist`
    holds the device yhist (words 0..m-1) and alpha (m..2m-1)."""
    if param.m > LBFGS_FUSED_MAX_M:
        raise Error(
            "qn: lbfgs_memory " + String(param.m) + " exceeds the fused "
            "direction's " + String(LBFGS_FUSED_MAX_M)
        )
    var hist_alpha = hist.create_sub_buffer[DType.float32](param.m, param.m)
    var verdict = scalar.create_sub_buffer[DType.float32](2, 2)
    # The launch also does the solver's S/Y updates, the next iteration's
    # xp / gradp saves and (do_dot) the line search's dg_init into scalar
    # word 0: see lbfgs_dir_kernel. `pseudo` is the direction's source when
    # use_pseudo (OWL-QN), else any distinct buffer (unread).
    ctx.enqueue_function[lbfgs_dir_kernel](
        drt.unsafe_ptr(), pseudo.unsafe_ptr(), s_all.unsafe_ptr(), y_all.unsafe_ptr(),
        hist.unsafe_ptr(), hist_alpha.unsafe_ptr(), verdict.unsafe_ptr(),
        x.unsafe_ptr(), xp.unsafe_ptr(), grad.unsafe_ptr(), gradp.unsafe_ptr(),
        scalar.unsafe_ptr(),
        Int32(n), Int32(param.m), Int32(n_vec), Int32(end_prev), Float32(-1.0),
        Int32(1) if use_pseudo else Int32(0), Int32(1) if do_dot else Int32(0),
        grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
    )
    _ = hist_alpha^
    _ = verdict^


def lbfgs_search_dir_resolve(
    param: LBFGSParam,
    mut n_vec: Int,
    end_prev: Int,
    mut yhist: List[Float32],
    stage: HostBuffer[DType.float32],
) -> Int:
    """The host half of `lbfgs_search_dir` after `stage` holds `scalar`'s
    words 0..3: on a skip `end` stays; otherwise `n_vec += 1`,
    `yhist[end] = ys` and `end` advances. Returns the new `end`."""
    var skipped = stage.unsafe_ptr().unsafe_load(2) != Float32(0.0)
    if skipped:
        return end_prev
    n_vec += 1
    yhist[end_prev] = stage.unsafe_ptr().unsafe_load(3)
    return (end_prev + 1) % param.m


# ===========================================================================
# OWL-QN'S OPERATORS (`qn_util.cuh:131-134`, `:237-261`)
# ===========================================================================
#
# The l1 solver never differentiates `|w|`. It works with a PSEUDO-GRADIENT,
# which is the subgradient of the l1 term chosen to point downhill, and it
# keeps every step inside the orthant the current point is in by projecting.
# Those are the two operators below and they are the whole of what makes
# OWL-QN different from L-BFGS. DEVIATION 552.
#
# BOTH ARE DISCRETE FUNCTIONS OF A FLOAT, which is IDENTITY_PATHS row 32's
# class and not a rounding: `project_orth` returns exactly 0 or exactly `x`
# on a sign test, and `get_pseudo_grad` picks one of four expressions on
# `x != 0`, `dmins > 0`, `dplus < 0`. A last-bit disagreement at one of
# those boundaries does not move a coefficient's fifth decimal, it moves
# WHICH ORTHANT the iterate is in and therefore how many entries of the
# solution are exactly zero -- the sparsity pattern, which is the thing an
# l1 fit is asked for. That is why every operand is flushed through `ftz`
# before it is compared and why the l1 arm's identity claim is about a
# SPARSITY PATTERN as well as about bits.


@always_inline
def project_orth(x: Float32, y: Float32) -> Float32:
    """`project_orth(x, y)`, `qn_util.cuh:131-134`: `x * y <= 0 ? 0 : x`.

    Note `<=`, so a product of exactly zero projects to zero, and note that
    a NaN product fails the test and returns `x` unchanged, which is theirs.
    """
    return Float32(0.0) if ftz(x * y) <= Float32(0.0) else x


@always_inline
def get_pseudo_grad(x: Float32, dlossx: Float32, c: Float32) -> Float32:
    """`get_pseudo_grad(x, dlossx, C)`, `qn_util.cuh:236-245`.

        if (x != 0) return dlossx + sgn(x) * C;
        dplus = dlossx + C; dmins = dlossx - C;
        if (dmins > 0) return dmins;
        if (dplus < 0) return dplus;
        return 0;

    `raft::sgn` returns an **int** (`raft/core/math.hpp:706-710`,
    `(T(0) < val) - (val < T(0))`), so `sgn(x) * C` is an exact +C or -C and
    the add is one rounding. It is written that way here rather than as a
    copysign because `sgn` of a NaN is 0 on their side, which makes the
    first branch `dlossx + 0`, and a copysign would not.
    """
    if x != Float32(0.0):
        var sgn = (
            Float32(1.0) if Float32(0.0) < x
            else (Float32(-1.0) if x < Float32(0.0) else Float32(0.0))
        )
        return ftz(dlossx + ftz(sgn * c))
    var dplus = ftz(dlossx + c)
    var dmins = ftz(dlossx - c)
    if dmins > Float32(0.0):
        return dmins
    if dplus < Float32(0.0):
        return dplus
    return Float32(0.0)


def pseudo_grad_kernel(
    pseudo: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    l1: Float32,
):
    """`op_pseudo_grad` under `assign_binary` (`qn_util.cuh:255-261`,
    `dense.hpp`): `pseudo[i] = get_pseudo_grad(x[i], grad[i], l1)`.

    One thread per entry, no fold, so the block width is SCHEDULING."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        pseudo.unsafe_store(
            i, ftz(get_pseudo_grad(x.unsafe_load(i), grad.unsafe_load(i), l1))
        )


def project_neg_kernel(
    drt: MutPointer[Float32, MutAnyOrigin],
    pseudo: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`drt.assign_binary(drt, pseudo, op_project(-1.0))`
    (`qn_solvers.cuh:393`, `qn_util.cuh:247-253`):
    `drt[i] = project_orth(drt[i], -1 * pseudo[i])`.

    In place on `drt`, which is what their call site is (`drt` is both the
    first operand and the destination). The `-1 *` is exact."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var y = ftz(Float32(-1.0) * pseudo.unsafe_load(i))
        drt.unsafe_store(i, project_orth(drt.unsafe_load(i), y))


def update_pseudo(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut grad: DeviceBuffer[DType.float32],
    l1: Float32,
    pg_limit: Int,
    mut pseudo: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """`update_pseudo`, `qn_solvers.cuh:245-259`.

        if (grad.len > pg_limit) { pseudo = grad; mask(pg_limit) = op(x, grad) }
        else                     { pseudo = op(x, grad) over all n }

    THE BRANCH IS NOT AN OPTIMIZATION, IT IS WHERE THE BIAS ESCAPES THE
    PENALTY. `pg_limit` is `loss.D * loss.C` (`qn_solvers.cuh:447`), the
    weight block, while `n` is `n_param = (D + fit_intercept) * C`. With an
    intercept the two differ by one entry per class, and that entry gets the
    RAW loss gradient copied straight through rather than a pseudo-gradient
    -- i.e. the intercept is not l1-penalized, matching
    `glm_regularizer.cuh:45`, where it is not l2-penalized either.
    """
    if n > pg_limit:
        copy_vec(ctx, pseudo, grad)
        ctx.enqueue_function[pseudo_grad_kernel](
            pseudo.unsafe_ptr(), x.unsafe_ptr(), grad.unsafe_ptr(),
            Int32(pg_limit), l1,
            grid_dim=((pg_limit + VEC_ELEM_TPB - 1) // VEC_ELEM_TPB, 1, 1),
            block_dim=(VEC_ELEM_TPB, 1, 1),
        )
    else:
        ctx.enqueue_function[pseudo_grad_kernel](
            pseudo.unsafe_ptr(), x.unsafe_ptr(), grad.unsafe_ptr(),
            Int32(n), l1,
            grid_dim=((n + VEC_ELEM_TPB - 1) // VEC_ELEM_TPB, 1, 1),
            block_dim=(VEC_ELEM_TPB, 1, 1),
        )
    # lane/linear-apple: no synchronize. Nothing on the host reads `pseudo`;
    # the next launch on this context is ordered after this one.


def project_direction(
    ctx: DeviceContext,
    mut drt: DeviceBuffer[DType.float32],
    mut pseudo: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """"Project drt onto orthant of -pseudog", `qn_solvers.cuh:392-393`."""
    ctx.enqueue_function[project_neg_kernel](
        drt.unsafe_ptr(), pseudo.unsafe_ptr(), Int32(n),
        grid_dim=((n + VEC_ELEM_TPB - 1) // VEC_ELEM_TPB, 1, 1),
        block_dim=(VEC_ELEM_TPB, 1, 1),
    )
    # lane/linear-apple: no synchronize (nothing on the host reads `drt`).
