"""The host oracle: the SAME SMO, serial, in Float32 through the identical
helpers (and in Float64 as a sanity reference).

NOT A PORT. cuML has one GPU backend and checks nothing against a host
solver. This file exists because the device port claims BIT identity with
something, and that something has to be spelled where a reader can follow
every rounding: one thread, one loop, ascending, through `ftz`,
`identical_mul_add`, `identical_exp` and `gemm_oracle_cell` (the normative
v1 GEMM cell), with the same working-set rule, the same block-solve logic
and the same tie-breaks as `svm/ported/svm/*.mojo`. The device gate in
`svc_check.mojo` is device == oracle, bitwise, on alphas, on `f` after every
outer iteration, on the working-set sequence, on `b`, and on the decision
function.

Generic over `dt` so the Float64 reference is the same code: under
`DType.float32` every seam goes through the identical helpers; under
`DType.float64` it is plain arithmetic with the stdlib `exp` (the FLOAT64
reference is for TOLERANCE sanity -- does the Float32 solver land near a
better-conditioned solve of the same problem -- not for bit comparison).

WHERE EACH PIECE OF THEIRS IS IN THIS FILE

    smosolver.cuh::Solve / UpdateF / SvcInit / CheckStoppingCondition
                                    -> smo_oracle_fit
    workingset.cuh::Select / SimpleSelect / GatherAvailable
                                    -> _select_ws
    smoblocksolve.cuh::SmoBlockSolve -> _block_solve
    results.cuh::Get / CalcB        -> _results
    kernel_matrices.cu::linear/rbf  -> _kernel_cell, _row_norm
    svc_impl.cuh::svcPredict        -> smo_oracle_decision
"""

from std.builtin.sort import sort
from std.math import exp, fma, inf, isnan
from std.memory import bitcast

from gemm.mojo_only.gemm_oracle import (
    contract_leaf_size,
    fold_balanced_tree,
    leaf_begin,
    leaf_count,
    leaf_end,
)
from mojo_only.numerics import ftz, identical_exp, identical_mul_add
from svm.ported.svm.smosolver import fold_order_for, hash_f32_list
from svm.ported.svm.svm_parameter import (
    KERNEL_LINEAR,
    KERNEL_RBF,
    KernelParams,
    SvmParameter,
)


comptime ORACLE_WS_SIZE = 1024
comptime ORACLE_MAX_INNER = 10000
comptime ORACLE_ETA_EPS = 1.0e-12


# ---------------------------------------------------------------------------
# the arithmetic seams, per dtype
# ---------------------------------------------------------------------------


@always_inline
def _flush[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    comptime if dt == DType.float32:
        return rebind[Scalar[dt]](ftz(rebind[Float32](x)))
    else:
        return x


@always_inline
def _mad[dt: DType](a: Scalar[dt], b: Scalar[dt], c: Scalar[dt]) -> Scalar[dt]:
    comptime if dt == DType.float32:
        return rebind[Scalar[dt]](
            identical_mul_add(rebind[Float32](a), rebind[Float32](b), rebind[Float32](c))
        )
    else:
        return a * b + c


@always_inline
def _exp[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    comptime if dt == DType.float32:
        return rebind[Scalar[dt]](identical_exp(rebind[Float32](x)))
    else:
        return rebind[Scalar[dt]](exp(rebind[Float64](x)))


@always_inline
def _inf[dt: DType]() -> Scalar[dt]:
    comptime if dt == DType.float32:
        return rebind[Scalar[dt]](inf[DType.float32]())
    else:
        return rebind[Scalar[dt]](inf[DType.float64]())


@always_inline
def _isnan[dt: DType](x: Scalar[dt]) -> Bool:
    comptime if dt == DType.float32:
        return isnan(rebind[Float32](x))
    else:
        return isnan(rebind[Float64](x))


def in_upper_g[dt: DType](a: Scalar[dt], y: Scalar[dt], C: Scalar[dt]) -> Bool:
    return (y < Scalar[dt](0) and a > Scalar[dt](0)) or (y > Scalar[dt](0) and a < C)


def in_lower_g[dt: DType](a: Scalar[dt], y: Scalar[dt], C: Scalar[dt]) -> Bool:
    return (y < Scalar[dt](0) and a < C) or (y > Scalar[dt](0) and a > Scalar[dt](0))


# ---------------------------------------------------------------------------
# the kernel matrix, per cell
# ---------------------------------------------------------------------------


def _row_norm[dt: DType](x: List[Scalar[dt]], i: Int, k: Int) -> Scalar[dt]:
    """`row_norm_l2sq_kernel`: ascending serial chain."""
    var acc = Scalar[dt](0)
    for c in range(k):
        var v = _flush[dt](x[i * k + c])
        acc = _flush[dt](_mad[dt](v, v, acc))
    return _flush[dt](acc)


def _dot[dt: DType](
    xa: List[Scalar[dt]], ia: Int, xb: List[Scalar[dt]], ib: Int, na: Int, nb: Int, k: Int
) -> Scalar[dt]:
    """Float32: the v1 GEMM cell (`gemm/IDENTICAL_FP32_CONTRACT.md`): leaf
    partials at `contract_leaf_size(k)`, each an ascending chain of
    `ftz(fma(ftz(a), ftz(b), acc))` seeded `+0.0`, combined by
    `gemm_oracle.fold_balanced_tree` (IMPORTED; the leaf loop is spelled
    here because `gemm_oracle_cell` takes `List[Float32]` and this file is
    generic -- the device-vs-oracle gate on F7 (k = 200) is what holds the
    two leaf loops together). Float64: a plain ascending chain."""
    comptime if dt == DType.float32:
        var leaf = contract_leaf_size(k)
        var pcount = leaf_count(k, leaf)
        var partials = List[Float32]()
        for t in range(pcount):
            var acc = Scalar[dt](0)
            for c in range(leaf_begin(t, leaf), leaf_end(t, leaf, k)):
                acc = _flush[dt](
                    _mad[dt](_flush[dt](xa[ia * k + c]), _flush[dt](xb[ib * k + c]), acc)
                )
            partials.append(rebind[Float32](_flush[dt](acc)))
        return rebind[Scalar[dt]](fold_balanced_tree(partials))
    else:
        var acc = Scalar[dt](0)
        for c in range(k):
            acc = acc + xa[ia * k + c] * xb[ib * k + c]
        return acc


def _kernel_cell[
    dt: DType
](
    kp: KernelParams,
    xa: List[Scalar[dt]],
    norm_a: List[Scalar[dt]],
    ia: Int,
    xb: List[Scalar[dt]],
    norm_b: List[Scalar[dt]],
    ib: Int,
    na: Int,
    nb: Int,
    k: Int,
) -> Scalar[dt]:
    """`kernel_op` for one cell: linear, or the RBF expansion in THEIR
    association (`kernel_matrices.mojo`)."""
    var dot = _dot[dt](xa, ia, xb, ib, na, nb, k)
    if kp.kernel == KERNEL_LINEAR:
        return dot
    var gain = Scalar[dt](kp.gamma)
    var s = _flush[dt](
        _flush[dt](_flush[dt](norm_a[ia]) + _flush[dt](norm_b[ib]))
        - _flush[dt](Scalar[dt](2) * _flush[dt](dot))
    )
    var e = _flush[dt]((-gain) * s)
    return _flush[dt](_exp[dt](e))


# ---------------------------------------------------------------------------
# the result record
# ---------------------------------------------------------------------------


struct OracleResult[dt: DType](Movable):
    var alpha: List[Scalar[Self.dt]]
    var f: List[Scalar[Self.dt]]
    var b: Scalar[Self.dt]
    var dual_coefs: List[Scalar[Self.dt]]
    var support_idx: List[Int32]
    var n_iter: Int
    var n_outer_iter: Int
    var ws_seq: List[List[Int32]]
    var alpha_hash_seq: List[UInt64]
    var f_hash_seq: List[UInt64]
    var diff_seq: List[Scalar[Self.dt]]
    var inner_iter_seq: List[Int]
    var nnz_seq: List[Int]
    var y: List[Scalar[Self.dt]]
    var objective_seq: List[Float64]
    """The dual objective W(alpha) after each outer iteration, Float64,
    only when the caller asks (it is O(n^2 k) per iteration)."""

    def __init__(out self):
        self.alpha = List[Scalar[Self.dt]]()
        self.f = List[Scalar[Self.dt]]()
        self.b = Scalar[Self.dt](0)
        self.dual_coefs = List[Scalar[Self.dt]]()
        self.support_idx = List[Int32]()
        self.n_iter = 0
        self.n_outer_iter = 0
        self.ws_seq = List[List[Int32]]()
        self.alpha_hash_seq = List[UInt64]()
        self.f_hash_seq = List[UInt64]()
        self.diff_seq = List[Scalar[Self.dt]]()
        self.inner_iter_seq = List[Int]()
        self.nnz_seq = List[Int]()
        self.y = List[Scalar[Self.dt]]()
        self.objective_seq = List[Float64]()


def _twiddle32(x: Float32) -> UInt32:
    var bits = bitcast[DType.uint32](x)
    if bits & UInt32(0x80000000) != UInt32(0):
        return ~bits
    return bits | UInt32(0x80000000)


def _twiddle64(x: Float64) -> UInt64:
    var bits = bitcast[DType.uint64](x)
    if bits & UInt64(0x8000000000000000) != UInt64(0):
        return ~bits
    return bits | UInt64(0x8000000000000000)


def _sorted_by_f[dt: DType](f: List[Scalar[dt]]) -> List[Int32]:
    """`cub::DeviceRadixSort::SortPairs(f, f_idx)`: indices in ascending
    `(twiddled f bits, index)` order. Float32: one UInt64 key `(key << 32)
    | idx`. Float64: `(key, idx)` pairs -- a 64-bit key does not leave
    room for the index, so the pair sort is spelled as key-then-idx with
    an insertion of ties in index order."""
    var n = len(f)
    var out = List[Int32]()
    comptime if dt == DType.float32:
        var keys = List[UInt64]()
        for i in range(n):
            keys.append((UInt64(_twiddle32(rebind[Float32](f[i]))) << 32) | UInt64(i))
        sort(keys)
        for i in range(n):
            out.append(Int32(Int(keys[i] & UInt64(0xFFFFFFFF))))
    else:
        # sort keys, then indices: two passes of an integer sort over
        # (key, idx) is the same total order. Use a stable merge: sort idx
        # by key with ties by idx -- spelled as sorting a list of key, then
        # for equal keys the ascending-idx order is recovered by scanning.
        var keys = List[UInt64]()
        var idx_of = List[Int]()
        for i in range(n):
            keys.append(_twiddle64(rebind[Float64](f[i])))
        var order = List[Int]()
        for i in range(n):
            order.append(i)
        # insertion sort on (key, idx); n is small for the Float64 arm
        for i in range(1, n):
            var j = i
            while j > 0:
                var a = order[j - 1]
                var bb = order[j]
                if keys[a] > keys[bb] or (keys[a] == keys[bb] and a > bb):
                    order[j - 1] = bb
                    order[j] = a
                    j -= 1
                else:
                    break
        for i in range(n):
            out.append(Int32(order[i]))
    return out^


# ---------------------------------------------------------------------------
# the solver
# ---------------------------------------------------------------------------


struct _WsState(Movable):
    var firstcall: Bool
    var idx: List[Int32]
    var ws_idx_save: List[Int32]
    var n_ws: Int
    var n_train: Int

    def __init__(out self, n_train: Int, n_ws: Int):
        self.firstcall = True
        self.n_train = n_train
        self.n_ws = n_ws
        self.idx = List[Int32]()
        self.ws_idx_save = List[Int32]()
        for i in range(n_ws):
            self.idx.append(Int32(i))
            self.ws_idx_save.append(Int32(i))


def _gather_available[
    dt: DType
](
    mut ws: _WsState,
    sorted_idx: List[Int32],
    mut available: List[Bool],
    n_already_selected: Int,
    n_needed: Int,
    copy_front: Bool,
) -> Int:
    """`GatherAvailable`."""
    if n_already_selected > 0:
        for p in range(n_already_selected):
            available[Int(ws.idx[p])] = False
    var sel = List[Int32]()
    for p in range(len(sorted_idx)):
        if available[Int(sorted_idx[p])]:
            sel.append(sorted_idx[p])
    var n_selected = len(sel)
    var n_copy = n_needed if n_selected > n_needed else n_selected
    var src_off = 0 if copy_front else n_selected - n_copy
    for c in range(n_copy):
        ws.idx[n_already_selected + c] = sel[src_off + c]
    return n_copy


def _select_ws[
    dt: DType
](
    mut ws: _WsState,
    f: List[Scalar[dt]],
    alpha: List[Scalar[dt]],
    y: List[Scalar[dt]],
    C: Scalar[dt],
):
    """`WorkingSet::Select` (FIFO) + `SimpleSelect`."""
    var n_train = ws.n_train
    var n_ws = ws.n_ws
    if n_ws >= n_train:
        return
    var nc = n_ws // 4
    var n_selected = 0
    if ws.firstcall:
        if nc >= 1:
            ws.firstcall = False
    else:
        for p in range(2 * nc):
            ws.idx[p] = ws.ws_idx_save[2 * nc + p]
        n_selected = nc * 2
    # SimpleSelect
    var n_already = n_selected
    var n_needed = n_ws - n_already
    var sorted_idx = _sorted_by_f[dt](f)
    var available = List[Bool]()
    for i in range(n_train):
        available.append(in_upper_g[dt](alpha[i], y[i], C))
    n_already += _gather_available[dt](ws, sorted_idx, available, n_already, n_needed // 2, True)
    for i in range(n_train):
        available[i] = in_lower_g[dt](alpha[i], y[i], C)
    n_already += _gather_available[dt](ws, sorted_idx, available, n_already, n_ws - n_already, False)
    if n_already < n_ws:
        for i in range(n_train):
            available[i] = True
        n_already += _gather_available[dt](ws, sorted_idx, available, n_already, n_ws - n_already, True)
    for p in range(n_ws):
        ws.ws_idx_save[p] = ws.idx[p]


def _block_solve[
    dt: DType
](
    ws_idx: List[Int32],
    n_ws: Int,
    y_array: List[Scalar[dt]],
    mut alpha: List[Scalar[dt]],
    f_array: List[Scalar[dt]],
    tile: List[Scalar[dt]],
    C_: Scalar[dt],
    eps: Scalar[dt],
    max_iter: Int,
    mut delta_alpha: List[Scalar[dt]],
) -> Tuple[Scalar[dt], Int]:
    """`SmoBlockSolve`, serialized: the per-thread registers are arrays
    indexed by `tid`, the three reductions are serial scans with the
    DEVIATION 633 / 635 tie-break (smaller training index; the scans
    compare with `<`/`>`/`==`, never a hardware `min`/`max`, row 39).
    Returns `(return_buff[0], n_iter)`."""
    var pos_inf = _inf[dt]()
    var neg_inf = -_inf[dt]()
    var y = List[Scalar[dt]]()
    var f = List[Scalar[dt]]()
    var a = List[Scalar[dt]]()
    var a_save = List[Scalar[dt]]()
    var Kd = List[Scalar[dt]]()
    var key = List[Int]()
    for t in range(n_ws):
        var idx = Int(ws_idx[t])
        y.append(y_array[idx])
        f.append(f_array[idx])
        a.append(alpha[idx])
        a_save.append(alpha[idx])
        Kd.append(tile[t + t * n_ws])
        key.append(idx)
    var C = C_
    var eta_eps = Scalar[dt](ORACLE_ETA_EPS)
    var diff0 = Scalar[dt](0)
    var diff_end = Scalar[dt](0)
    var n_iter = 0
    while n_iter < max_iter:
        # argmin over upper of (f, key)
        var f_u = pos_inf
        var u = -1
        var u_key = 2147483647
        for t in range(n_ws):
            var v = f[t] if in_upper_g[dt](a[t], y[t], C) else pos_inf
            if v < f_u or (v == f_u and key[t] < u_key):
                f_u = v
                u = t
                u_key = key[t]
        if u < 0:
            u = 0
        # f_max over lower: DEVIATION 635, the same key-tied argmax as `u`
        # (the value of the smallest training index among the maximal f;
        # only a +0.0/-0.0 pair can tie with different bits, row 39)
        var f_max = neg_inf
        var fmax_key = 2147483647
        for t in range(n_ws):
            var v = f[t] if in_lower_g[dt](a[t], y[t], C) else neg_inf
            if v > f_max or (v == f_max and key[t] < fmax_key):
                f_max = v
                fmax_key = key[t]
        var diff = _flush[dt](f_max - f_u)
        if n_iter == 0:
            diff0 = diff
            var d10 = _flush[dt](Scalar[dt](0.1) * diff)
            diff_end = eps if eps > d10 else d10
        if diff < diff_end:
            break
        # argmax over lower of (f_u - f)^2 / eta
        var best = neg_inf
        var l = -1
        var l_key = 2147483647
        for t in range(n_ws):
            var v = neg_inf
            if f_u < f[t] and in_lower_g[dt](a[t], y[t], C):
                var Kui = tile[u * n_ws + t]
                var eta_ui = _flush[dt](_flush[dt](Kd[t] + Kd[u]) - _flush[dt](Scalar[dt](2) * Kui))
                if eta_ui < eta_eps:
                    eta_ui = eta_eps
                var d = _flush[dt](f_u - f[t])
                v = _flush[dt](_flush[dt](d * d) / eta_ui)
            if v > best or (v == best and key[t] < l_key):
                best = v
                l = t
                l_key = key[t]
        if l < 0:
            l = 0
        # update
        var tmp_u = C - a[u] if y[u] > Scalar[dt](0) else a[u]
        var tmp_l = a[l] if y[l] > Scalar[dt](0) else C - a[l]
        var Kul = tile[u * n_ws + l]
        var eta_ul = _flush[dt](_flush[dt](Kd[u] + Kd[l]) - _flush[dt](Scalar[dt](2) * Kul))
        if eta_ul < eta_eps:
            eta_ul = eta_eps
        var q_l = _flush[dt](_flush[dt](f[l] - f_u) / eta_ul)
        if q_l < tmp_l:
            tmp_l = q_l
        var q = tmp_u if tmp_u < tmp_l else tmp_l
        a[u] = _flush[dt](a[u] + q * y[u])
        a[l] = _flush[dt](a[l] - q * y[l])
        for t in range(n_ws):
            var Kui = tile[u * n_ws + t]
            var Kli = tile[l * n_ws + t]
            f[t] = _flush[dt](_mad[dt](q, _flush[dt](Kui - Kli), f[t]))
        if q == Scalar[dt](0):
            break
        n_iter += 1
    for t in range(n_ws):
        var idx = Int(ws_idx[t])
        alpha[idx] = a[t]
        delta_alpha[t] = _flush[dt](_flush[dt](a[t] - a_save[t]) * y[t])
    return (diff0, n_iter)


def smo_oracle_fit[
    dt: DType
](
    x: List[Scalar[dt]],
    y: List[Scalar[dt]],
    n_rows: Int,
    n_cols: Int,
    param: SvmParameter,
    kp: KernelParams,
    record_objective: Bool = False,
) raises -> OracleResult[dt]:
    """`SmoSolver::Solve` + `Results::Get`, serial. `y` is +-1 already
    (the caller does `getOvrlabels`; `svc_check` shares one helper)."""
    var res = OracleResult[dt]()
    var n_train = n_rows
    var k = n_cols
    var C = Scalar[dt](param.C)
    var tol = Scalar[dt](param.tol)
    var n_ws = ORACLE_WS_SIZE
    if n_ws > n_train:
        n_ws = n_train
    # Initialize
    var alpha = List[Scalar[dt]]()
    var f = List[Scalar[dt]]()
    for i in range(n_train):
        alpha.append(Scalar[dt](0))
        f.append(-y[i])
    var norms = List[Scalar[dt]]()
    if kp.kernel == KERNEL_RBF:
        for i in range(n_rows):
            norms.append(_row_norm[dt](x, i, k))
    var ws = _WsState(n_train, n_ws)
    # counters
    var max_outer_iter = param.max_outer_iter
    if max_outer_iter == -1:
        max_outer_iter = n_train * 100
        if max_outer_iter < 100000:
            max_outer_iter = 100000
    var max_iter = param.max_iter
    var n_outer_iter = 0
    var n_iter = 0
    var diff_prev = Scalar[dt](0)
    var n_small_diff = 0
    var n_increased_diff = 0
    var keep_going = True
    var delta_alpha = List[Scalar[dt]]()
    for _ in range(n_ws):
        delta_alpha.append(Scalar[dt](0))
    var tile = List[Scalar[dt]]()
    for _ in range(n_ws * n_ws):
        tile.append(Scalar[dt](0))

    while keep_going:
        for t in range(n_ws):
            delta_alpha[t] = Scalar[dt](0)
        _select_ws[dt](ws, f, alpha, y, C)
        # square tile K(ws, ws)
        for u in range(n_ws):
            var iu = Int(ws.idx[u])
            for t in range(n_ws):
                var it = Int(ws.idx[t])
                tile[u * n_ws + t] = _kernel_cell[dt](
                    kp, x, norms, iu, x, norms, it, n_rows, n_rows, k
                )
        var max_iter_this_block = ORACLE_MAX_INNER
        if max_iter != -1:
            var rem = max_iter - n_iter
            if rem < max_iter_this_block:
                max_iter_this_block = rem
        var r = _block_solve[dt](
            ws.idx, n_ws, y, alpha, f, tile, C, tol, max_iter_this_block, delta_alpha
        )
        var diff = r[0]
        var inner = r[1]
        # GetNonzeroDeltaAlpha
        var nz_idx = List[Int32]()
        var nz_da = List[Scalar[dt]]()
        for t in range(n_ws):
            if delta_alpha[t] != Scalar[dt](0):
                nz_idx.append(ws.idx[t])
                nz_da.append(delta_alpha[t])
        var nnz = len(nz_idx)
        if nnz > 0:
            var order = fold_order_for(nz_idx)
            for i in range(n_train):
                var acc = Scalar[dt](0)
                for rr in range(nnz):
                    var j = Int(order[rr])
                    var kij = _kernel_cell[dt](
                        kp, x, norms, Int(nz_idx[j]), x, norms, i, n_rows, n_rows, k
                    )
                    acc = _flush[dt](_mad[dt](kij, nz_da[j], acc))
                f[i] = _flush[dt](f[i] + acc)
        # CheckStoppingCondition
        if Float64(diff) > Float64(diff_prev) * 1.5 and n_outer_iter > 0:
            n_increased_diff += 1
        keep_going = True
        var dd = abs(diff - diff_prev)
        if Float64(dd) < 0.001 * Float64(tol):
            n_small_diff += 1
        else:
            diff_prev = diff
            n_small_diff = 0
        if n_small_diff > param.nochange_steps:
            keep_going = False
        if diff < tol:
            keep_going = False
        if _isnan[dt](diff):
            raise Error("SMO error: NaN found during fitting (oracle).")
        n_iter += inner
        n_outer_iter += 1
        if (max_iter != -1 and n_iter >= max_iter) or n_outer_iter >= max_outer_iter:
            keep_going = False
        # record
        var wsc = List[Int32]()
        for t in range(n_ws):
            wsc.append(ws.idx[t])
        res.ws_seq.append(wsc^)
        comptime if dt == DType.float32:
            res.alpha_hash_seq.append(hash_f32_list(rebind[List[Float32]](alpha)))
            res.f_hash_seq.append(hash_f32_list(rebind[List[Float32]](f)))
        res.diff_seq.append(diff)
        res.inner_iter_seq.append(inner)
        res.nnz_seq.append(nnz)
        if record_objective:
            res.objective_seq.append(
                dual_objective[dt](x, y, alpha, norms, n_rows, k, kp)
            )

    # Results
    var coef = List[Scalar[dt]]()
    for i in range(n_train):
        coef.append(alpha[i] * y[i])
    for i in range(n_train):
        if coef[i] != Scalar[dt](0):
            res.dual_coefs.append(coef[i])
            res.support_idx.append(Int32(i))
    var n_support = len(res.dual_coefs)
    if n_support == 0:
        var s = Scalar[dt](0)
        for i in range(n_train):
            s = _flush[dt](s + _flush[dt](f[i]))
        res.b = _flush[dt](-s / Scalar[dt](n_train))
    else:
        var n_free = 0
        var s = Scalar[dt](0)
        for i in range(n_train):
            if Scalar[dt](0) < alpha[i] and alpha[i] < C:
                s = _flush[dt](s + _flush[dt](f[i]))
                n_free += 1
        if n_free > 0:
            res.b = _flush[dt](-s / Scalar[dt](n_free))
        else:
            # strict `<`/`>` ascending: the FIRST index wins a +0.0/-0.0
            # tie, as `serial_min/max_f32_kernel` do over the compaction
            # (row 39); no hardware min/max
            var b_up = _inf[dt]()
            var b_low = -_inf[dt]()
            var nu = 0
            var nl = 0
            for i in range(n_train):
                if in_upper_g[dt](alpha[i], y[i], C):
                    nu += 1
                    if f[i] < b_up:
                        b_up = f[i]
                if in_lower_g[dt](alpha[i], y[i], C):
                    nl += 1
                    if f[i] > b_low:
                        b_low = f[i]
            if nu == 0 or nl == 0:
                raise Error("Incorrect training: cannot calculate the constant (oracle)")
            res.b = _flush[dt](-_flush[dt](b_up + b_low) / Scalar[dt](2))
    res.alpha = alpha^
    res.f = f^
    res.n_iter = n_iter
    res.n_outer_iter = n_outer_iter
    res.y = y.copy()
    return res^


def dual_objective[
    dt: DType
](
    x: List[Scalar[dt]],
    y: List[Scalar[dt]],
    alpha: List[Scalar[dt]],
    norms: List[Scalar[dt]],
    n: Int,
    k: Int,
    kp: KernelParams,
) -> Float64:
    """`W(alpha) = -sum alpha + 1/2 sum alpha_i alpha_j y_i y_j K_ij`, the
    quantity their header says SMO MINIMIZES (`smoblocksolve.cuh:30`),
    in Float64 over the Float32 kernel cells. For the "objective decreases"
    gate only."""
    var lin = Float64(0)
    var quad = Float64(0)
    for i in range(n):
        var ai = Float64(alpha[i])
        if ai == 0.0:
            continue
        lin += ai
        for j in range(n):
            var aj = Float64(alpha[j])
            if aj == 0.0:
                continue
            var kij = Float64(_kernel_cell[dt](kp, x, norms, i, x, norms, j, n, n, k))
            quad += ai * aj * Float64(y[i]) * Float64(y[j]) * kij
    return -lin + 0.5 * quad


def smo_oracle_decision[
    dt: DType
](
    res: OracleResult[dt],
    x_train: List[Scalar[dt]],
    n_train: Int,
    xq: List[Scalar[dt]],
    nq: Int,
    k: Int,
    kp: KernelParams,
) -> List[Scalar[dt]]:
    """`svcPredict(..., predict_class = false)`: `sum_j K(x_q, sv_j) *
    dual_j` ascending over the support vectors, plus `b`."""
    var out = List[Scalar[dt]]()
    var sv_rows = List[Scalar[dt]]()
    var ns = len(res.support_idx)
    for j in range(ns):
        var r = Int(res.support_idx[j])
        for c in range(k):
            sv_rows.append(x_train[r * k + c])
    var norm_q = List[Scalar[dt]]()
    var norm_sv = List[Scalar[dt]]()
    if kp.kernel == KERNEL_RBF:
        for i in range(nq):
            norm_q.append(_row_norm[dt](xq, i, k))
        for j in range(ns):
            norm_sv.append(_row_norm[dt](sv_rows, j, k))
    for i in range(nq):
        var acc = Scalar[dt](0)
        for j in range(ns):
            var kij = _kernel_cell[dt](kp, xq, norm_q, i, sv_rows, norm_sv, j, nq, ns, k)
            acc = _flush[dt](_mad[dt](kij, res.dual_coefs[j], acc))
        out.append(_flush[dt](acc + res.b))
    return out^


def global_kkt_gap[
    dt: DType
](res: OracleResult[dt], C: Scalar[dt]) -> Scalar[dt]:
    """`max{f_i : i in lower} - min{f_i : i in upper}` over the WHOLE
    training set, the quantity the block solve drives below `tol` on the
    working set. The KKT gate."""
    var b_up = _inf[dt]()
    var b_low = -_inf[dt]()
    for i in range(len(res.alpha)):
        if in_upper_g[dt](res.alpha[i], res.y[i], C):
            if res.f[i] < b_up:
                b_up = res.f[i]
        if in_lower_g[dt](res.alpha[i], res.y[i], C):
            if res.f[i] > b_low:
                b_low = res.f[i]
    return b_low - b_up
