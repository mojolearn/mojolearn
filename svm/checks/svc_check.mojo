# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The SVC gates: oracle properties, device == oracle bitwise, launch
invariance, and the sabotage hooks.

NOT A PORT. Run with:

    pixi run mojo run -I . svm/checks/svc_check.mojo                 # host oracle only
    tools/with_build_lock.sh     pixi run mojo run -I . svm/svc_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . svm/svc_main.mojo

Every line carries the mode the binary COMPILED in. Under IDENTICAL the
device-vs-oracle comparisons ASSERT; under FAST they REPORT (the vendor
matmul and the stdlib exp are the FAST arms and are not the oracle's bits).

FIXTURES (all HASHED, non-uniform, 13-bit significands so every product
is inexact; a uniform fixture hides a permutation):

    F1 blobs     n=300  k=8   linear  C=1       labels 3 / 7, separable
    F2 xor       n=240  k=2   rbf g=0.5 C=10    labels 0 / 1, sign(x*y)
    F3 dup       n=1280 k=8   linear  C=1       640 hashed rows (sep 0.45), EACH TWICE
                                                -> exactly equal f, AND n > 1024
                                                so the working-set sort runs
    F4 c_small   F2 with C=0.05                 every alpha bound
    F5 c_large   F2 with C=100
    F6 big       n=1500 k=8   linear  C=1       overlapping blobs (sep 0.3); selection +
                                                FIFO; the oracle's Float64 twin
    F7 wide_k    n=96   k=200 linear  C=1       the v1 GEMM leaf path (k > 128)

REGRESSION FIXTURES (EPSILON_SVR; `n_train = 2n`, so the solver domain is
twice the row count and `n_ws = min(1024, 2n)`):

    R1 line      n=200  k=4   linear  eps=0.1  C=1    2n=400  <= n_ws
    R2 rbf       n=160  k=2   rbf 0.5 eps=0.05 C=10   2n=320  <= n_ws
    R3 tube      n=240  k=3   linear  eps=0.30 C=1    2n=480  <= n_ws; eps wide
                                                      vs the target spread, so
                                                      MOST rows are non-SVs
    R4 dup       n=300  k=4   linear  eps=0.1  C=1    2n=600; 150 rows EACH
                                                      TWICE
    R5 big       n=1030 k=3   linear  eps=0.1  C=1    2n=2060 > SMO_WS_SIZE:
                                                      FIFO + the radix sort run
    R6 zero      n=600  k=3   linear  eps=0.0  C=1    2n=1200 > SMO_WS_SIZE;
                                                      targets non-negative with
                                                      +0.0 AND -0.0 planted, so
                                                      `f` carries BOTH ZERO
                                                      SIGNS from ordinary input

    Z  signed zero  n_ws=8 planted at the BLOCK-SOLVE ENTRY (row 39): every
                    f in the working set is a zero, +0.0 and -0.0 mixed,
                    keys a permutation, upper/lower sets split by y; two
                    orders (A, B); block 32 and 1024
    N  nan          refusals of non-finite X / labels / C / tol / gamma, and
                    a finite input whose linear kernel OVERFLOWS (DEVIATION 637)

SABOTAGES (each a `-D` define; the README table records the failing line):

    MOJOLEARN_SVM_SABOTAGE_WS_TIE       workingset.mojo   must FAIL on F3
    MOJOLEARN_SVM_SABOTAGE_FOLD_ROTATE  smosolver.mojo    must FAIL on F1, F3, F6,
                                        R4, R5, R6 -- and is BIT-INERT on F2, F4,
                                        F5, F7 and R1-R3, which this line used to
                                        claim as "(F1..F7)". `start = block_idx.x
                                        % nnz` can only move a fold when there is
                                        MORE THAN ONE BLOCK, and `update_f_kernel`
                                        launches `ceil(n_rows / SEL_TPB)` = 1 block
                                        for every fixture with n_rows <= 256. The
                                        README's own recorded evidence names F1 and
                                        F6 only; the claim was never measured on
                                        the short fixtures. Corrected 2026-08-31
                                        while building the regression lane, whose
                                        `check_svr_sabotage_reach` prints the block
                                        count per fixture for exactly this reason.
    MOJOLEARN_SVM_SABOTAGE_STD_EXP      kernel_matrices   must FAIL on rbf (IDENTICAL)
    MOJOLEARN_SVM_SABOTAGE_NO_FTZ       smosolver.mojo    REPORT (bit-inert on FTZ hw)
    MOJOLEARN_SVM_SABOTAGE_ARG_TIE_HIGH pinned_argreduce  must FAIL on F3 and Z
    MOJOLEARN_SVM_SABOTAGE_FMAX_NOKEY   smoblocksolve     must FAIL on Z order A
    MOJOLEARN_SVM_SABOTAGE_FMAX_HWMAX   smoblocksolve     Apple-INERT on Z (row 39); predicted FAIL on NVIDIA/AMD
    MOJOLEARN_SVM_SABOTAGE_FMAX_HWMAX_SWAP smoblocksolve  must FAIL on Z order A on Apple
"""

from std.math import inf
from std.memory import bitcast
from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace, first_divergence
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name
from svm.checks.device_select import read_f32, upload_f32, upload_i32
from svm.checks.smo_oracle import (
    OracleResult,
    global_kkt_gap,
    smo_oracle_decision,
    smo_oracle_fit,
    svr_dual_objective,
    svr_gradient_reference,
    svr_kkt_gap,
    svr_tube_bound,
    _block_solve,
    _kernel_cell,
    _row_norm,
    _vec_index,
)
from svm.impl.svm.smoblocksolve import SMO_WS_SIZE
from svm.impl.svm.smosolver import (
    SmoSolver,
    SmoTrace,
    fold_order_for,
    launch_block_solve,
)
from svm.impl.svm.svc_impl import (
    svc_fit,
    svc_predict,
    unique_labels_sorted,
)
from svm.impl.svm.svm_parameter import (
    EPSILON_SVR,
    KERNEL_LINEAR,
    KERNEL_RBF,
    KernelParams,
    SvmModel,
    SvmParameter,
    check_finite_list,
    check_rung1_scope,
)


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29. This used to
    be a local two-way `IDENTICAL`-or-`FAST`, written when there were
    two tiers, and it answered "FAST" for a DETERMINISTIC build -- so
    a driver run under the middle tier printed the wrong arm onto
    every line it produced. A correctly-labelled measurement of the
    wrong arm is the failure this tree has been bitten by repeatedly,
    and forty-four copies of a mode label is how it happens.
    """
    return numeric_mode_name()


def _is_identical() -> Bool:
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return True
    return False


def _bits(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


def _show(x: Float32) -> String:
    return String(x) + "/" + hex(_bits(x))


# ===========================================================================
# FIXTURES
# ===========================================================================


def _hash64(i: Int, salt: Int) -> UInt64:
    var h = UInt64(i) * UInt64(0x9E3779B97F4A7C15) + UInt64(salt + 1) * UInt64(
        0xBF58476D1CE4E5B9
    )
    h = h ^ (h >> UInt64(29))
    h = h * UInt64(0x94D049BB133111EB)
    return h ^ (h >> UInt64(32))


def _unit(i: Int, salt: Int) -> Float32:
    """A hashed value in [-1, 1) with a 13-bit significand."""
    var h = _hash64(i, salt)
    var m = Int(h & UInt64(0x1FFF))  # 13 bits
    var v = Float32(m) / Float32(8192.0)
    if (h >> UInt64(20)) & UInt64(1) == UInt64(1):
        return -v
    return v


struct Fixture(Movable):
    var name: String
    var x: List[Float32]
    var labels: List[Float32]
    var n: Int
    var k: Int
    var kp: KernelParams
    var param: SvmParameter
    var min_accuracy: Float64

    def __init__(
        out self,
        name: String,
        var x: List[Float32],
        var labels: List[Float32],
        n: Int,
        k: Int,
        kp: KernelParams,
        param: SvmParameter,
        min_accuracy: Float64,
    ):
        self.name = name
        self.x = x^
        self.labels = labels^
        self.n = n
        self.k = k
        self.kp = kp.copy()
        self.param = param.copy()
        self.min_accuracy = min_accuracy


def _blobs(n: Int, k: Int, sep: Float32, salt: Int, lab0: Float32, lab1: Float32) -> Tuple[List[Float32], List[Float32]]:
    var x = List[Float32]()
    var labels = List[Float32]()
    for i in range(n):
        var cls = Int(_hash64(i, salt + 77) & UInt64(1))
        var sign = Float32(1.0) if cls == 1 else Float32(-1.0)
        for c in range(k):
            x.append(sign * sep + _unit(i * k + c, salt))
        labels.append(lab1 if cls == 1 else lab0)
    return (x^, labels^)


def fixture_blobs() -> Fixture:
    var d = _blobs(300, 8, Float32(1.5), 11, Float32(3.0), Float32(7.0))
    var p = SvmParameter.default()
    return Fixture("F1.blobs", d[0].copy(), d[1].copy(), 300, 8, KernelParams.linear(), p, 0.99)


def fixture_xor(C: Float64, name: String) -> Fixture:
    var n = 240
    var x = List[Float32]()
    var labels = List[Float32]()
    for i in range(n):
        var a = Float32(2.0) * _unit(2 * i, 23)
        var b = Float32(2.0) * _unit(2 * i + 1, 23)
        # keep away from the axes so the planted labeling is learnable
        if a > Float32(0.0) and a < Float32(0.25):
            a = a + Float32(0.25)
        if a < Float32(0.0) and a > Float32(-0.25):
            a = a - Float32(0.25)
        if b > Float32(0.0) and b < Float32(0.25):
            b = b + Float32(0.25)
        if b < Float32(0.0) and b > Float32(-0.25):
            b = b - Float32(0.25)
        x.append(a)
        x.append(b)
        labels.append(Float32(1.0) if a * b > Float32(0.0) else Float32(0.0))
    var p = SvmParameter.default()
    p.C = C
    return Fixture(name, x^, labels^, n, 2, KernelParams.rbf(0.5), p, 0.9)


def fixture_dup() -> Fixture:
    var base = _blobs(640, 8, Float32(0.45), 31, Float32(-1.0), Float32(1.0))
    var x = List[Float32]()
    var labels = List[Float32]()
    for i in range(640):
        for rep in range(2):
            for c in range(8):
                x.append(base[0][i * 8 + c])
            labels.append(base[1][i])
    var p = SvmParameter.default()
    return Fixture("F3.dup", x^, labels^, 1280, 8, KernelParams.linear(), p, 0.7)


def fixture_big() -> Fixture:
    var d = _blobs(1500, 8, Float32(0.3), 41, Float32(0.0), Float32(1.0))
    var p = SvmParameter.default()
    return Fixture("F6.big", d[0].copy(), d[1].copy(), 1500, 8, KernelParams.linear(), p, 0.6)


def fixture_wide_k() -> Fixture:
    var d = _blobs(96, 200, Float32(0.4), 53, Float32(0.0), Float32(1.0))
    var p = SvmParameter.default()
    return Fixture("F7.wide_k", d[0].copy(), d[1].copy(), 96, 200, KernelParams.linear(), p, 0.95)


def all_fixtures() -> List[Fixture]:
    var out = List[Fixture]()
    out.append(fixture_blobs())
    out.append(fixture_xor(10.0, "F2.xor"))
    out.append(fixture_dup())
    out.append(fixture_xor(0.05, "F4.c_small"))
    out.append(fixture_xor(100.0, "F5.c_large"))
    out.append(fixture_big())
    out.append(fixture_wide_k())
    return out^


def ovr_y(labels: List[Float32]) raises -> List[Float32]:
    """`getOvrlabels(idx = 1)`: +1 where label == the LARGER unique label."""
    var u = unique_labels_sorted(labels)
    if len(u) != 2:
        raise Error("fixture is not binary")
    var y = List[Float32]()
    for v in labels:
        y.append(Float32(1.0) if v == u[1] else Float32(-1.0))
    return y^


def _same_i32(a: List[Int32], b: List[Int32]) -> Bool:
    if len(a) != len(b):
        return False
    for p in range(len(a)):
        if a[p] != b[p]:
            return False
    return True


def _to_f64(v: List[Float32]) -> List[Float64]:
    var out = List[Float64]()
    for x in v:
        out.append(Float64(x))
    return out^


# ===========================================================================
# THE ORACLE GATES (host only)
# ===========================================================================


def check_oracle_kkt_and_accuracy(fx: Fixture) raises:
    """KKT gap within tol (asserted when the working set is the whole set;
    reported otherwise, because the solver only drives the GAP ON THE
    WORKING SET below tol), and predict accuracy on the planted labels."""
    var y = ovr_y(fx.labels)
    var res = smo_oracle_fit[DType.float32](fx.x, y, fx.n, fx.k, fx.param, fx.kp)
    var gap = global_kkt_gap[DType.float32](res, Float32(fx.param.C))
    var dec = smo_oracle_decision[DType.float32](res, fx.x, fx.n, fx.x, fx.n, fx.k, fx.kp)
    var ok = 0
    for i in range(fx.n):
        var pred = Float32(1.0) if dec[i] >= Float32(0.0) else Float32(-1.0)
        if pred == y[i]:
            ok += 1
    var acc = Float64(ok) / Float64(fx.n)
    print(
        "  " + fx.name + " [" + _mode_name() + "] oracle f32: n_support="
        + String(len(res.support_idx)) + " outer=" + String(res.n_outer_iter)
        + " inner=" + String(res.n_iter) + " b=" + _show(res.b)
        + " kkt_gap=" + _show(gap) + " acc=" + String(acc)
    )
    if fx.n <= 1024:
        if not (gap < Float32(fx.param.tol)):
            raise Error(fx.name + ": KKT gap " + String(gap) + " not below tol " + String(fx.param.tol))
    else:
        print("    (n > n_ws: the global gap is REPORTED; the stop rule is on the working set)")
    if acc < fx.min_accuracy:
        raise Error(fx.name + ": accuracy " + String(acc) + " below " + String(fx.min_accuracy))


def check_oracle_objective_decreases(fx: Fixture) raises:
    """`W(alpha)` after every outer iteration is non-increasing, up to
    Float32 round-off in the Float64 evaluation (a rise of 1e-5 relative
    is round-off; a rise of 1e-2 is a sign error)."""
    var y = ovr_y(fx.labels)
    var res = smo_oracle_fit[DType.float32](fx.x, y, fx.n, fx.k, fx.param, fx.kp, True)
    var worst = 0.0
    for t in range(1, len(res.objective_seq)):
        var prev = res.objective_seq[t - 1]
        var cur = res.objective_seq[t]
        var rise = cur - prev
        var scale = abs(prev)
        if scale < 1.0:
            scale = 1.0
        if rise / scale > worst:
            worst = rise / scale
    print(
        "  " + fx.name + " objective: first=" + String(res.objective_seq[0])
        + " last=" + String(res.objective_seq[len(res.objective_seq) - 1])
        + " worst relative rise=" + String(worst)
    )
    if worst > 1.0e-4:
        raise Error(fx.name + ": objective rose by " + String(worst) + " relative")


def check_oracle_f32_matches_f64_reference(fx: Fixture) raises:
    """The Float32 solve lands where the Float64 solve of the same problem
    lands: same support-vector SET on the separable fixture, `b` within
    1e-3 relative, decision function within 1e-3 of the Float64 one."""
    var y = ovr_y(fx.labels)
    var r32 = smo_oracle_fit[DType.float32](fx.x, y, fx.n, fx.k, fx.param, fx.kp)
    var x64 = _to_f64(fx.x)
    var y64 = _to_f64(y)
    var r64 = smo_oracle_fit[DType.float64](x64, y64, fx.n, fx.k, fx.param, fx.kp)
    # SV sets
    var only32 = 0
    var only64 = 0
    var in64 = List[Bool]()
    for _ in range(fx.n):
        in64.append(False)
    for j in range(len(r64.support_idx)):
        in64[Int(r64.support_idx[j])] = True
    var in32 = List[Bool]()
    for _ in range(fx.n):
        in32.append(False)
    for j in range(len(r32.support_idx)):
        in32[Int(r32.support_idx[j])] = True
    for i in range(fx.n):
        if in32[i] and not in64[i]:
            only32 += 1
        if in64[i] and not in32[i]:
            only64 += 1
    var d32 = smo_oracle_decision[DType.float32](r32, fx.x, fx.n, fx.x, fx.n, fx.k, fx.kp)
    var d64 = smo_oracle_decision[DType.float64](r64, x64, fx.n, x64, fx.n, fx.k, fx.kp)
    var maxd = 0.0
    for i in range(fx.n):
        var d = abs(Float64(d32[i]) - d64[i])
        if d > maxd:
            maxd = d
    var bdiff = abs(Float64(r32.b) - r64.b)
    print(
        "  " + fx.name + " f32 vs f64 reference: n_sv " + String(len(r32.support_idx))
        + " vs " + String(len(r64.support_idx)) + " (only32=" + String(only32)
        + ", only64=" + String(only64) + ") b " + String(r32.b) + " vs "
        + String(r64.b) + " max|decision diff|=" + String(maxd)
        + " outer " + String(r32.n_outer_iter) + " vs " + String(r64.n_outer_iter)
    )
    if only32 != 0 or only64 != 0:
        raise Error(fx.name + ": support-vector set differs from the Float64 reference")
    if bdiff > 1.0e-3 * (abs(r64.b) + 1.0):
        raise Error(fx.name + ": b differs from the Float64 reference by " + String(bdiff))
    if maxd > 1.0e-2:
        raise Error(fx.name + ": decision function differs from Float64 by " + String(maxd))


def check_rbf_float_vs_double_reference() raises:
    """DEVIATION 630's measurement: the RBF cell in Float32 (ours) against
    THEIR spelling's double exp of the same Float32 argument, rounded to
    Float32. Prints the max ULP distance; asserts it is small (<= 2)."""
    var fx = fixture_xor(10.0, "F2.xor")
    var norms = List[Float32]()
    for i in range(fx.n):
        norms.append(_row_norm[DType.float32](fx.x, i, fx.k))
    var max_ulp = 0
    var n_diff = 0
    var n_cells = 0
    from std.math import exp
    for i in range(fx.n):
        for j in range(fx.n):
            var k32 = _kernel_cell[DType.float32](fx.kp, fx.x, norms, i, fx.x, norms, j, fx.n, fx.n, fx.k)
            # THEIR arm: the float expansion, then exp in DOUBLE of
            # (-1.0 * gain * s), stored to float.
            var dot = Float32(0.0)
            for c in range(fx.k):
                dot = dot + fx.x[i * fx.k + c] * fx.x[j * fx.k + c]
            var s = norms[i] + norms[j] - dot * Float32(2.0)
            var e64 = exp(-1.0 * Float64(Float32(fx.kp.gamma)) * Float64(s))
            var kref = Float32(e64)
            var u = Int(_bits(k32)) - Int(_bits(kref))
            if u < 0:
                u = -u
            if u > max_ulp:
                max_ulp = u
            if u != 0:
                n_diff += 1
            n_cells += 1
    print(
        "  DEVIATION 630 measurement: rbf float32 vs their double-exp spelling:"
        + " cells=" + String(n_cells) + " differing=" + String(n_diff)
        + " max_ulp=" + String(max_ulp)
    )
    if max_ulp > 2:
        raise Error("rbf float arm is more than 2 ulp from the double reference")


# ===========================================================================
# THE DEVICE GATES
# ===========================================================================


struct DeviceRun(Movable):
    var model: SvmModel
    var trace: SmoTrace
    var decision: List[Float32]
    var classes: List[Float32]

    def __init__(out self, var model: SvmModel, var trace: SmoTrace, var decision: List[Float32], var classes: List[Float32]):
        self.model = model^
        self.trace = trace^
        self.decision = decision^
        self.classes = classes^


def _queries(fx: Fixture) -> List[Float32]:
    """The training rows plus 37 fresh hashed rows, the decision-function
    query set."""
    var q = fx.x.copy()
    for i in range(37):
        for c in range(fx.k):
            q.append(Float32(2.0) * _unit(i * fx.k + c, 97))
    return q^


def _run_device(
    ctx: DeviceContext,
    fx: Fixture,
    block_solve_threads: Int,
    kernel_tile_byte_limit: Int,
    predict_buffer_mib: Float64,
    scratch_pad: Int,
    scratch_poison: Float32,
    mut card: IdentityTrace,
) raises -> DeviceRun:
    var trace = SmoTrace()
    var model = svc_fit(
        ctx, fx.x, fx.labels, fx.n, fx.k, fx.param, fx.kp, card, trace, False,
        kernel_tile_byte_limit, block_solve_threads, True, scratch_pad, scratch_poison,
    )
    var q = _queries(fx)
    var nq = fx.n + 37
    var dec = svc_predict(ctx, model, q, nq, fx.k, fx.kp, predict_buffer_mib, False, card)
    var cls = svc_predict(ctx, model, q, nq, fx.k, fx.kp, predict_buffer_mib, True, card)
    return DeviceRun(model^, trace^, dec^, cls^)


def _compare_device_to_oracle(fx: Fixture, run: DeviceRun, strict: Bool) raises -> Int:
    """Device vs oracle: ws sequence, alpha/f hash per outer iteration, b,
    dual coefs, support idx, decision function. Returns the number of
    divergences; raises on the first when `strict`."""
    var y = ovr_y(fx.labels)
    var res = smo_oracle_fit[DType.float32](fx.x, y, fx.n, fx.k, fx.param, fx.kp)
    var q = _queries(fx)
    var nq = fx.n + 37
    var odec = smo_oracle_decision[DType.float32](res, fx.x, fx.n, q, nq, fx.k, fx.kp)
    var n_div = 0
    var first = String("")

    def note(mut n_div: Int, mut first: String, msg: String):
        n_div += 1
        if first == "":
            first = msg

    if len(run.trace.ws_seq) != len(res.ws_seq):
        note(n_div, first, "outer iteration count device " + String(len(run.trace.ws_seq)) + " vs oracle " + String(len(res.ws_seq)))
    var n_it = len(run.trace.ws_seq)
    if len(res.ws_seq) < n_it:
        n_it = len(res.ws_seq)
    for t in range(n_it):
        if not _same_i32(run.trace.ws_seq[t], res.ws_seq[t]):
            note(n_div, first, "ws sequence differs at outer iteration " + String(t))
        if run.trace.alpha_hash_seq[t] != res.alpha_hash_seq[t]:
            note(n_div, first, "alpha hash differs at outer iteration " + String(t))
        if run.trace.f_hash_seq[t] != res.f_hash_seq[t]:
            note(n_div, first, "f hash differs at outer iteration " + String(t))
        if _bits(run.trace.diff_seq[t]) != _bits(res.diff_seq[t]):
            note(n_div, first, "diff differs at outer iteration " + String(t) + ": " + _show(run.trace.diff_seq[t]) + " vs " + _show(res.diff_seq[t]))
        if run.trace.inner_iter_seq[t] != res.inner_iter_seq[t]:
            note(n_div, first, "inner iteration count differs at outer iteration " + String(t))
    if _bits(run.model.b) != _bits(res.b):
        note(n_div, first, "b differs: device " + _show(run.model.b) + " oracle " + _show(res.b))
    if len(run.model.dual_coefs) != len(res.dual_coefs):
        note(n_div, first, "n_support differs: device " + String(len(run.model.dual_coefs)) + " oracle " + String(len(res.dual_coefs)))
    else:
        for j in range(len(res.dual_coefs)):
            if _bits(run.model.dual_coefs[j]) != _bits(res.dual_coefs[j]) or run.model.support_idx[j] != res.support_idx[j]:
                note(n_div, first, "dual coef / support idx differs at " + String(j))
                break
    for i in range(nq):
        if _bits(run.decision[i]) != _bits(odec[i]):
            note(n_div, first, "decision function differs at query " + String(i) + ": " + _show(run.decision[i]) + " vs " + _show(odec[i]))
            break
    var verdict = "IDENTICAL" if n_div == 0 else ("DIVERGES (" + String(n_div) + ": " + first + ")")
    print(
        "  " + fx.name + " [" + _mode_name() + "] device vs oracle: " + verdict
        + "  outer=" + String(len(run.trace.ws_seq)) + " n_support=" + String(run.model.n_support)
        + " b=" + _show(run.model.b)
    )
    if strict and n_div != 0:
        raise Error(fx.name + ": device diverges from the oracle: " + first)
    return n_div


def check_device_matches_oracle(ctx: DeviceContext, fixtures: List[Fixture]) raises:
    """THE GATE. Under IDENTICAL every fixture must be bitwise the oracle;
    under FAST the divergence count is reported."""
    var card = IdentityTrace.disabled()
    var total = 0
    for i in range(len(fixtures)):
        var run = _run_device(ctx, fixtures[i], 0, 1 << 30, 200.0, 0, Float32(0.0), card)
        total += _compare_device_to_oracle(fixtures[i], run, _is_identical())
    if not _is_identical():
        print("  [FAST] device-vs-oracle is a REPORT in this mode: " + String(total) + " divergences over " + String(len(fixtures)) + " fixtures")


def check_ws_sequence_is_pure_in_f_and_index(ctx: DeviceContext) raises:
    """DEVIATION 631's gate: on F3 (every row twice, n > n_ws) the device
    working-set sequence equals the oracle's `(twiddle(f) << 32 | idx)`
    order, position for position. Asserted in BOTH modes -- the selection
    is integer work and the sort is exact; under FAST the f values
    themselves may differ from the oracle's (vendor GEMM), so only the
    FIRST working set (f = -y, no arithmetic) is asserted there, the rest
    reported."""
    var fx = fixture_dup()
    var card = IdentityTrace.disabled()
    var run = _run_device(ctx, fx, 0, 1 << 30, 200.0, 0, Float32(0.0), card)
    var y = ovr_y(fx.labels)
    var res = smo_oracle_fit[DType.float32](fx.x, y, fx.n, fx.k, fx.param, fx.kp)
    var n_it = len(run.trace.ws_seq)
    if len(res.ws_seq) < n_it:
        n_it = len(res.ws_seq)
    var first_bad = -1
    for t in range(n_it):
        if not _same_i32(run.trace.ws_seq[t], res.ws_seq[t]) and first_bad < 0:
            first_bad = t
    # the fixture must actually carry ties: count equal adjacent f in the
    # oracle's final f (duplicates are adjacent indices 2i, 2i+1)
    var ties = 0
    for i in range(0, fx.n, 2):
        if _bits(res.f[i]) == _bits(res.f[i + 1]):
            ties += 1
    print(
        "  F3.dup ws sequence: " + String(n_it) + " outer iterations, first divergence at "
        + ("none" if first_bad < 0 else String(first_bad)) + "; equal-f duplicate pairs in final f: "
        + String(ties) + " of " + String(fx.n // 2)
    )
    if ties < fx.n // 4:
        raise Error("F3.dup: the fixture does not carry enough equal f values to test the tie order")
    if _is_identical():
        if first_bad >= 0:
            raise Error("F3.dup: working-set sequence diverges from the (f bits, index) order at iteration " + String(first_bad))
    else:
        if first_bad == 0:
            raise Error("F3.dup: the FIRST working set (no arithmetic yet) diverges from the (f bits, index) order")


def _same_run(a: DeviceRun, b: DeviceRun) -> String:
    """'' when every byte agrees, else the first difference."""
    if len(a.trace.ws_seq) != len(b.trace.ws_seq):
        return "outer iteration count"
    for t in range(len(a.trace.ws_seq)):
        if a.trace.alpha_hash_seq[t] != b.trace.alpha_hash_seq[t]:
            return "alpha hash at outer iteration " + String(t)
        if a.trace.f_hash_seq[t] != b.trace.f_hash_seq[t]:
            return "f hash at outer iteration " + String(t)
        if not _same_i32(a.trace.ws_seq[t], b.trace.ws_seq[t]):
            return "ws at outer iteration " + String(t)
    if _bits(a.model.b) != _bits(b.model.b):
        return "b"
    if len(a.model.dual_coefs) != len(b.model.dual_coefs):
        return "n_support"
    for j in range(len(a.model.dual_coefs)):
        if _bits(a.model.dual_coefs[j]) != _bits(b.model.dual_coefs[j]):
            return "dual coef " + String(j)
        if a.model.support_idx[j] != b.model.support_idx[j]:
            return "support idx " + String(j)
    for i in range(len(a.decision)):
        if _bits(a.decision[i]) != _bits(b.decision[i]):
            return "decision at " + String(i)
        if _bits(a.classes[i]) != _bits(b.classes[i]):
            return "class at " + String(i)
    return ""


def check_device_is_launch_invariant(ctx: DeviceContext) raises:
    """THE HEADLINE: the bytes do not move across the block-solve block
    size (theirs 1024 vs the smallest power of two), the n_rows batch of
    the full-tile computation (one batch vs 256-row batches vs 100-row
    batches), the predict batch (one vs many), and two scratch paddings
    with two poisons. F2 (rbf, n_ws == n) and F6 (linear, n > n_ws:
    selection + FIFO + sort) and F7 (k > 128, the v1 leaf path). Asserted
    under IDENTICAL; a REPORT under FAST (the vendor matmul is free to
    fold per shape there)."""
    var card = IdentityTrace.disabled()
    for i in range(3):
        var fx: Fixture
        if i == 0:
            fx = fixture_xor(10.0, "F2.xor")
        elif i == 1:
            fx = fixture_big()
        else:
            fx = fixture_wide_k()
        var n_ws = fx.n if fx.n < 1024 else 1024
        var base = _run_device(ctx, fx, 0, 1 << 30, 200.0, 0, Float32(0.0), card)
        # (a) block size: theirs, 1024
        var arm_a = _run_device(ctx, fx, 1024, 1 << 30, 200.0, 0, Float32(0.0), card)
        var da = _same_run(base, arm_a)
        # (b) full-tile batches of 256 rows and of 100 rows
        var arm_b = _run_device(ctx, fx, 0, 256 * n_ws * 4, 200.0, 0, Float32(0.0), card)
        var db = _same_run(base, arm_b)
        var arm_b2 = _run_device(ctx, fx, 0, 100 * n_ws * 4, 200.0, 0, Float32(0.0), card)
        var db2 = _same_run(base, arm_b2)
        # (c) predict batches: a 0.001 MiB buffer forces tiny batches
        var arm_c = _run_device(ctx, fx, 0, 1 << 30, 0.001, 0, Float32(0.0), card)
        var dc = _same_run(base, arm_c)
        # (d) two paddings and two poisons
        var arm_d = _run_device(ctx, fx, 0, 1 << 30, 200.0, 37, Float32(-7.25e20), card)
        var dd = _same_run(base, arm_d)
        var arm_e = _run_device(ctx, fx, 0, 1 << 30, 200.0, 1029, Float32(3.0e-39), card)
        var de = _same_run(base, arm_e)
        # (e) everything at once
        var arm_f = _run_device(ctx, fx, 1024, 100 * n_ws * 4, 0.001, 37, Float32(-7.25e20), card)
        var df = _same_run(base, arm_f)
        var verdict = String("")
        if da != "":
            verdict += " block1024:" + da
        if db != "":
            verdict += " batch256:" + db
        if db2 != "":
            verdict += " batch100:" + db2
        if dc != "":
            verdict += " predict_batch:" + dc
        if dd != "":
            verdict += " pad37:" + dd
        if de != "":
            verdict += " pad1029:" + de
        if df != "":
            verdict += " all:" + df
        print(
            "  " + fx.name + " [" + _mode_name() + "] launch invariance over 7 arms: "
            + ("INVARIANT" if verdict == "" else "MOVES:" + verdict)
            + "  (outer=" + String(len(base.trace.ws_seq)) + ")"
        )
        if verdict != "":
            if _is_identical():
                raise Error(fx.name + ": the bytes moved across launches:" + verdict)
            print("    [FAST] REPORTED, not asserted: the vendor matmul routes n == 1 to a GEMV and folds as it likes")


def _try_refusal(
    ctx: DeviceContext, fx: Fixture, p: SvmParameter, kp: KernelParams, sw: Bool, expect: String
) raises -> Int:
    var card = IdentityTrace.disabled()
    var trace = SmoTrace()
    var raised = False
    var msg = String("")
    try:
        _ = svc_fit(ctx, fx.x, fx.labels, fx.n, fx.k, p, kp, card, trace, sw)
    except e:
        raised = True
        msg = String(e)
    if not raised:
        raise Error("refusal missing for " + expect)
    if msg.find(expect) < 0:
        raise Error("refusal for " + expect + " does not name it: " + msg)
    return 1


def check_refusals(ctx: DeviceContext) raises:
    """Every unported parameter raises BY NAME."""
    var fx = fixture_blobs()
    var card = IdentityTrace.disabled()
    var trace = SmoTrace()
    var hits = 0
    var p1 = SvmParameter.default()
    p1.cache_size = 200.0
    hits += _try_refusal(ctx, fx, p1, KernelParams.linear(), False, "cache_size")
    # THE svmType REFUSAL MOVED, AND THIS BLOCK IS WHAT NOTICED.
    #
    # It used to read `p2.svmType = 2` (EPSILON_SVR) and expect the word
    # "svmType" out of `check_rung1_scope`. On 2026-08-31 the scope check
    # began ADMITTING EPSILON_SVR, so that assertion stopped being about a
    # refusal: the fit ran on into a 2n-domain solve and this gate HUNG
    # rather than failed, twice, for seventeen minutes each. A refusal test
    # whose subject stops refusing does not fail loudly, it stops being a
    # test, and that is the whole reason the four cases below are separate.
    #
    # 3 is NU_SVR, which is unported UPSTREAM too and still comes out of the
    # scope check by name.
    var p2 = SvmParameter.default()
    p2.svmType = 3
    hits += _try_refusal(ctx, fx, p2, KernelParams.linear(), False, "svmType")
    # THE `UNGATED` CLAUSE IS GONE, 2026-08-31, AND ITS ABSENCE IS THE POINT.
    #
    # It read `p2b.svmType = 2` and asserted that `SmoSolver.solve` refuses
    # EPSILON_SVR with the word UNGATED. That assertion was correct exactly
    # as long as no regression gate existed. The gates below now exist --
    # `check_svr_objective_decreases`, `check_svr_kkt`,
    # `check_svr_eps_tube`, `check_svr_device_matches_oracle`,
    # `check_svr_sabotage_reach` -- so the clause is deleted rather than left
    # to pass on a message that happens to still match.
    #
    # `SmoSolver.solve`'s raise is DELIBERATELY still in the tree and is not
    # this file's to remove: until it goes, `check_svr_device_matches_oracle`
    # reports BLOCKED and FAILS, which is the loud state a lane wants while
    # its device arm cannot run. The three host gates pass without it.
    #
    # A NEGATIVE epsilon on a regressor is still refused below and still
    # comes out of `check_rung1_scope`, which is the boundary that has to
    # stay named.
    # `epsilon` on a CLASSIFIER, where upstream ignores it.
    var p3 = SvmParameter.default()
    p3.epsilon = 0.1
    hits += _try_refusal(ctx, fx, p3, KernelParams.linear(), False, "epsilon")
    # and a NEGATIVE epsilon on a regressor, which is DEVIATION 636's family
    # and must be caught BEFORE the UNGATED refusal above, or the scope
    # check is not the thing rejecting it.
    var p3b = SvmParameter.default()
    p3b.svmType = 2
    p3b.epsilon = -1.0
    hits += _try_refusal(
        ctx, fx, p3b, KernelParams.linear(), False, "epsilon must be non-negative"
    )
    hits += _try_refusal(ctx, fx, SvmParameter.default(), KernelParams(1, 3, 1.0, 0.0), False, "POLYNOMIAL")
    hits += _try_refusal(ctx, fx, SvmParameter.default(), KernelParams(3, 3, 1.0, 0.0), False, "TANH")
    hits += _try_refusal(ctx, fx, SvmParameter.default(), KernelParams(4, 3, 1.0, 0.0), False, "PRECOMPUTED")
    hits += _try_refusal(ctx, fx, SvmParameter.default(), KernelParams.linear(), True, "sample_weight")
    # multiclass
    var lab3 = fx.labels.copy()
    lab3[0] = Float32(99.0)
    var raised = False
    try:
        _ = svc_fit(ctx, fx.x, lab3, fx.n, fx.k, SvmParameter.default(), KernelParams.linear(), card, trace, False)
    except e:
        raised = True
        if String(e).find("binary") < 0:
            raise Error("multiclass refusal does not say binary: " + String(e))
    if not raised:
        raise Error("multiclass was not refused")
    print(
        "  refusals by name: " + String(hits + 1)
        + " (cache_size, svmType=NU_SVR, epsilon on C_SVC, negative epsilon"
        " on SVR, POLYNOMIAL, TANH, PRECOMPUTED, sample_weight, multiclass)"
    )


def check_card_is_emitted(ctx: DeviceContext, path_a: String, path_b: String) raises:
    """Two traced fits of F2 into two files agree stage for stage, and the
    card has the stages the README names."""
    var fx = fixture_xor(10.0, "F2.xor")
    var card_a = IdentityTrace.to_path(path_a)
    var card_b = IdentityTrace.to_path(path_b)
    _ = _run_device(ctx, fx, 0, 1 << 30, 200.0, 0, Float32(0.0), card_a)
    _ = _run_device(ctx, fx, 1024, 100 * 240 * 4, 0.001, 37, Float32(-7.25e20), card_b)
    var d = first_divergence(path_a, path_b)
    print("  card: two traced fits (different launches) first divergence: " + (d if d != "" else "none"))
    if d != "":
        # FACT 3 (row 39 audit): the two launches differ in the full-tile
        # BATCH and the predict batch, and under FAST the kernel rows come
        # from the vendor matmul, which may fold per shape; the agreement
        # is only pinned under IDENTICAL.
        if _is_identical():
            raise Error("the two cards diverge at " + d)
        print("    RECORDED [FAST]: the two cards diverge at " + d + " (vendor matmul per batch shape; asserted under IDENTICAL only)")


# ===========================================================================
# ROW 39: SIGNED ZERO AT THE BLOCK-SOLVE REDUCTIONS, AND THE NaN AUDIT
# ===========================================================================


struct ZeroFixture(Movable):
    """A working set of 8 whose every f is a zero of planted sign, at the
    block-solve ENTRY (the real kernel, the real oracle function). Why not
    through `svc_fit`: `f = -0.0` arises only from a negative subnormal
    flushed at the f seam (row 10), which no O(1)-scale input with 13-bit
    significands can produce, and `f = +0.0` needs a sample exactly on the
    margin; planting at the entry is the only way to put BOTH in one set.
    Keys (`ws_idx`) are a permutation so that tree position, working-set
    position and training index all differ."""

    var name: String
    var ws_idx: List[Int32]
    var y: List[Float32]
    var alpha: List[Float32]
    var f: List[Float32]
    var tile: List[Float32]
    var expect_diff_bits: UInt32

    def __init__(
        out self, name: String, var ws_idx: List[Int32], var y: List[Float32],
        var alpha: List[Float32], var f: List[Float32], var tile: List[Float32],
        expect_diff_bits: UInt32,
    ):
        self.name = name
        self.ws_idx = ws_idx^
        self.y = y^
        self.alpha = alpha^
        self.f = f^
        self.tile = tile^
        self.expect_diff_bits = expect_diff_bits


def _zero_fixture(order_b: Bool) -> ZeroFixture:
    """Order A: by POSITION t = 0..7 the keys are [6,1,7,0,3,5,2,4]; the
    UPPER set (y = +1, alpha = 0) is positions {0, 2, 5} = keys {6, 7, 5},
    the LOWER set (y = -1, alpha = 0) is positions {1, 3, 4, 6, 7} = keys
    {1, 0, 3, 2, 4}. f by KEY: [-0, +0, -0, +0, -0, +0, -0, -0]. The
    smallest upper key (5) holds +0.0, so f_u = +0.0; the smallest lower
    key (0) holds -0.0, so f_max = -0.0 under DEVIATION 635; diff = -0.0 -
    (+0.0) = -0.0 = 0x80000000, and the solve stops at n_iter = 0 (diff <
    eps). The pre-635 strict-`>` tree over the lower lanes (upper lanes
    masked to -inf: [-inf, +0, -inf, -0, +0, -inf, -0, -0] by position)
    never updates on a tie and its survivor is +0.0 (hand trace in the
    README), so SAB_FMAX_NOKEY gives +0.0 - (+0.0) = +0.0 and FAILS; the
    hardware-max tree `max(mine, other)` on Apple (second operand) lands on
    -0.0 and is INERT here, while NVIDIA/AMD would give +0.0 (FAIL); the
    swapped `max(other, mine)` on Apple gives +0.0 (FAIL). Order B flips
    every sign: f_u = -0.0, f_max = +0.0, diff = +0.0 = 0x00000000, and
    NOKEY is inert (which is why both orders run)."""
    var keys: List[Int32] = [6, 1, 7, 0, 3, 5, 2, 4]
    var upper_pos: List[Int] = [0, 2, 5]
    var y = List[Float32]()
    var alpha = List[Float32]()
    var f = List[Float32]()
    for _ in range(8):
        y.append(Float32(-1.0))
        alpha.append(Float32(0.0))
        f.append(Float32(0.0))
    for t in upper_pos:
        y[Int(keys[t])] = Float32(1.0)
    # f by key, order A
    var neg = List[Bool]()
    var signs_a: List[Bool] = [True, False, True, False, True, False, True, True]
    for k in range(8):
        var is_neg = signs_a[k]
        if order_b:
            is_neg = not is_neg
        neg.append(is_neg)
        f[k] = bitcast[DType.float32](UInt32(0x80000000)) if is_neg else Float32(0.0)
    # a hashed symmetric PSD-ish tile (unused at n_iter = 0, must be real memory)
    var tile = List[Float32]()
    for i in range(8):
        for j in range(8):
            var v = _unit(i * 8 + j, 5) if i <= j else _unit(j * 8 + i, 5)
            if i == j:
                v = Float32(2.0) + v
            tile.append(v)
    var expect = UInt32(0x00000000) if order_b else UInt32(0x80000000)
    return ZeroFixture(
        "Z.order_" + ("B" if order_b else "A"), keys^, y^, alpha^, f^, tile^, expect
    )


def _run_zero_fixture_device(
    ctx: DeviceContext, zf: ZeroFixture, threads: Int
) raises -> Tuple[Float32, Int, List[Float32], List[Float32]]:
    """One `SmoBlockSolve` launch on the planted set: returns (diff,
    n_iter, alpha after, delta_alpha)."""
    var y = upload_f32(ctx, zf.y)
    var alpha = upload_f32(ctx, zf.alpha)
    var f = upload_f32(ctx, zf.f)
    var tile = upload_f32(ctx, zf.tile)
    var ws_idx = upload_i32(ctx, zf.ws_idx)
    var C_list = List[Float32]()
    var da0 = List[Float32]()
    for _ in range(8):
        C_list.append(Float32(1.0))
        da0.append(Float32(0.0))
    var C_vec = upload_f32(ctx, C_list)
    var delta_alpha = upload_f32(ctx, da0)
    var rb0: List[Float32] = [Float32(0.0), Float32(0.0)]
    var return_buff = upload_f32(ctx, rb0)
    launch_block_solve(
        ctx, threads, y, 8, alpha, 8, delta_alpha, f, tile, ws_idx, C_vec,
        Float32(1.0e-3), return_buff, 10000,
    )
    ctx.synchronize()
    var rb = read_f32(ctx, return_buff, 2)
    var a_out = read_f32(ctx, alpha, 8)
    var da_out = read_f32(ctx, delta_alpha, 8)
    _ = y^
    _ = f^
    _ = tile^
    _ = ws_idx^
    _ = C_vec^
    _ = alpha^
    _ = delta_alpha^
    _ = return_buff^
    return (rb[0], Int(rb[1]), a_out^, da_out^)


def check_block_solve_signed_zero_tie(ctx: DeviceContext) raises:
    """ROW 39, THE -0.0 FIXTURE: both zeros in one working set, both
    orders, block 32 and 1024; device == oracle BITWISE on `diff` (the one
    recorded bit a zero's sign can reach), n_iter, alpha and delta_alpha,
    and `diff` equals the value the smallest-key rule predicts.

    Asserted in BOTH modes: no arithmetic runs (n_iter = 0; the one
    operation is `(+-0) - (+-0)`, exact and sign-defined by IEEE 754 on
    every vendor), the folds are integer keys plus IEEE compares, and no
    library call or hardware `max`/`min` is on the path. That is the
    property DEVIATIONS 633/635 claim, and it is not vendor-shaped."""
    var bad = String("")
    for order in range(2):
        var zf = _zero_fixture(order == 1)
        # the oracle
        var o_alpha = zf.alpha.copy()
        var o_da = List[Float32]()
        for _ in range(8):
            o_da.append(Float32(0.0))
        var o = _block_solve[DType.float32](
            zf.ws_idx, 8, zf.y, o_alpha, zf.f, zf.tile, Float32(1.0), Float32(1.0e-3),
            10000, o_da,
        )
        var o_diff = o[0]
        var o_iter = o[1]
        for which in range(2):
            var threads = 32 if which == 0 else 1024
            var d = _run_zero_fixture_device(ctx, zf, threads)
            var d_diff = d[0]
            var d_iter = d[1]
            var same = _bits(d_diff) == _bits(o_diff) and d_iter == o_iter
            for i in range(8):
                if _bits(d[2][i]) != _bits(o_alpha[i]) or _bits(d[3][i]) != _bits(o_da[i]):
                    same = False
            var expected_ok = _bits(d_diff) == zf.expect_diff_bits
            print(
                "  " + zf.name + " [" + _mode_name() + "] block=" + String(threads)
                + " device diff=" + _show(d_diff) + " n_iter=" + String(d_iter)
                + " | oracle diff=" + _show(o_diff) + " n_iter=" + String(o_iter)
                + " | expected diff bits " + hex(zf.expect_diff_bits)
                + (" OK" if (same and expected_ok) else " MISMATCH")
            )
            if not same and bad == "":
                bad = zf.name + " block=" + String(threads) + ": device diff " + _show(d_diff) + " vs oracle " + _show(o_diff)
            if not expected_ok and bad == "":
                bad = zf.name + " block=" + String(threads) + ": diff " + _show(d_diff) + " is not the smallest-key answer " + hex(zf.expect_diff_bits)
    if bad != "":
        raise Error("signed-zero tie: " + bad)


def _expect_raise_naming(what: String, name: String, e_msg: String, raised: Bool) raises:
    if not raised:
        raise Error(what + ": no raise (expected one naming " + name + ")")
    if e_msg.find(name) < 0:
        raise Error(what + ": the raise does not name " + name + ": " + e_msg)


def check_nan_never_recorded(ctx: DeviceContext) raises:
    """FACT 2 (row 39): no NaN can reach a recorded stage. (1) DEVIATION
    636: non-finite X, labels, C, tol, gamma and predict-X are refused BY
    NAME before any stage; (2) DEVIATION 637: a FINITE input whose linear
    kernel overflows (x ~ 1.5e19, k = 2: every K is +inf, eta = inf - inf)
    raises the NaN sentence before the iteration's record. (1) is host
    refusal, asserted in both modes; (2) is asserted under IDENTICAL and
    RECORDED under FAST (a fast-math build may fold `x != x` to false)."""
    var fx = fixture_blobs()
    var card = IdentityTrace.disabled()
    var trace = SmoTrace()
    var n_ok = 0

    # (1a) X
    var x_nan = fx.x.copy()
    x_nan[17] = bitcast[DType.float32](UInt32(0x7FC00000))
    var raised = False
    var msg = String("")
    try:
        _ = svc_fit(ctx, x_nan, fx.labels, fx.n, fx.k, fx.param, fx.kp, card, trace, False)
    except e:
        raised = True
        msg = String(e)
    _expect_raise_naming("X NaN", "X contains a non-finite value at flat index 17", msg, raised)
    n_ok += 1
    # (1b) X inf
    var x_inf = fx.x.copy()
    x_inf[3] = inf[DType.float32]()
    raised = False
    msg = ""
    try:
        _ = svc_fit(ctx, x_inf, fx.labels, fx.n, fx.k, fx.param, fx.kp, card, trace, False)
    except e:
        raised = True
        msg = String(e)
    _expect_raise_naming("X inf", "X contains a non-finite value at flat index 3", msg, raised)
    n_ok += 1
    # (1c) labels
    var lab = fx.labels.copy()
    lab[5] = bitcast[DType.float32](UInt32(0xFFC00000))
    raised = False
    msg = ""
    try:
        _ = svc_fit(ctx, fx.x, lab, fx.n, fx.k, fx.param, fx.kp, card, trace, False)
    except e:
        raised = True
        msg = String(e)
    _expect_raise_naming("labels NaN", "labels contains a non-finite value at flat index 5", msg, raised)
    n_ok += 1
    # (1d) C = inf
    var p_c = SvmParameter.default()
    p_c.C = inf[DType.float64]()
    raised = False
    msg = ""
    try:
        _ = svc_fit(ctx, fx.x, fx.labels, fx.n, fx.k, p_c, fx.kp, card, trace, False)
    except e:
        raised = True
        msg = String(e)
    _expect_raise_naming("C inf", "C must be finite", msg, raised)
    n_ok += 1
    # (1e) tol = NaN (passes `tol > 0` and would never stop)
    var p_t = SvmParameter.default()
    p_t.tol = Float64(bitcast[DType.float32](UInt32(0x7FC00000)))
    raised = False
    msg = ""
    try:
        _ = svc_fit(ctx, fx.x, fx.labels, fx.n, fx.k, p_t, fx.kp, card, trace, False)
    except e:
        raised = True
        msg = String(e)
    _expect_raise_naming("tol NaN", "tol must be finite", msg, raised)
    n_ok += 1
    # (1f) gamma = inf (NaN on the kernel diagonal) and gamma < 0
    raised = False
    msg = ""
    try:
        _ = svc_fit(ctx, fx.x, fx.labels, fx.n, fx.k, fx.param, KernelParams.rbf(inf[DType.float64]()), card, trace, False)
    except e:
        raised = True
        msg = String(e)
    _expect_raise_naming("gamma inf", "gamma must be finite and >= 0", msg, raised)
    n_ok += 1
    raised = False
    msg = ""
    try:
        _ = svc_fit(ctx, fx.x, fx.labels, fx.n, fx.k, fx.param, KernelParams.rbf(-0.5), card, trace, False)
    except e:
        raised = True
        msg = String(e)
    _expect_raise_naming("gamma negative", "gamma must be finite and >= 0", msg, raised)
    n_ok += 1
    # (1g) predict X
    var model = svc_fit(ctx, fx.x, fx.labels, fx.n, fx.k, fx.param, fx.kp, card, trace, False)
    var q = fx.x.copy()
    q[0] = bitcast[DType.float32](UInt32(0x7FC00000))
    raised = False
    msg = ""
    try:
        _ = svc_predict(ctx, model, q, fx.n, fx.k, fx.kp, 200.0, False, card)
    except e:
        raised = True
        msg = String(e)
    _expect_raise_naming("predict X NaN", "predict X contains a non-finite value", msg, raised)
    n_ok += 1
    print("  NaN refusals by name (DEVIATION 636): " + String(n_ok) + " (X NaN, X inf, labels, C, tol, gamma inf, gamma < 0, predict X)")

    # (2) the overflow trap: finite input, every kernel cell +inf
    var n = 8
    var k = 2
    var xo = List[Float32]()
    var labo = List[Float32]()
    for i in range(n):
        for c in range(k):
            xo.append(Float32(1.5e19) * (Float32(1.0) + Float32(0.25) * abs(_unit(i * k + c, 61))))
        labo.append(Float32(1.0) if (i & 1) == 1 else Float32(0.0))
    var po = SvmParameter.default()
    raised = False
    msg = ""
    try:
        _ = svc_fit(ctx, xo, labo, n, k, po, KernelParams.linear(), card, trace, False)
    except e:
        raised = True
        msg = String(e)
    var trap_ok = raised and msg.find("NaN found during fitting") >= 0
    print(
        "  overflow fixture (linear, |x| ~ 1.5e19, K = +inf everywhere) [" + _mode_name() + "]: "
        + ("RAISED: " + msg if raised else "NO RAISE (fit completed)")
    )
    if not trap_ok:
        if _is_identical():
            raise Error("DEVIATION 637: the overflowing fit did not raise the NaN sentence: " + (msg if raised else "no raise"))
        print("    RECORDED [FAST]: the NaN trap did not fire (fast-math may fold x != x); asserted under IDENTICAL only")


# ===========================================================================
# EPSILON_SVR: THE REGRESSION FIXTURES
# ===========================================================================
#
# Same construction as F1-F7: every coordinate is `_unit`, a hashed value in
# [-1, 1) with a 13-BIT SIGNIFICAND, so every product in the kernel is
# inexact and a permutation of the fold cannot hide. The TARGETS are then
# full-precision combinations of those coordinates, which is deliberate: a
# 13-bit target would make `epsilon - y` exact in a way ordinary regression
# data is not, and the eps-tube gate wants ordinary data.
#
# WHAT `n_train = 2n` CHANGES ABOUT FIXTURE DESIGN, and it is not cosmetic:
#
#   * `n_ws = min(1024, 2n)`, so the working-set SORT and the FIFO carry-over
#     need `n > 512`, not `n > 1024`. R5 and R6 are the two that reach them;
#     R1-R4 have `n_ws == n_train` and take `WorkingSet::Select`'s early
#     return, exactly as F1/F2 do.
#   * `ws_idx % n_rows` COLLIDES. Whenever `i` and `i + n` are both in the
#     working set with nonzero delta_alpha, `GetNonzeroDeltaAlpha` hands
#     `fold_order_for` the SAME index twice. On R1-R4 the working set is the
#     whole domain, so EVERY row collides; on R5/R6 a subset does. See
#     `check_svr_fold_order`.
#   * `f = +-epsilon - y` can carry BOTH ZERO SIGNS. See R6 and
#     `check_svr_signed_zero`.


struct RegFixture(Movable):
    """A regression fixture. `yr` holds TARGETS, not labels, and no
    `getOvrlabels` runs on them: `svrFit` (`svr_impl.cuh:68`) hands `y`
    straight to `SmoSolver::Solve`."""

    var name: String
    var x: List[Float32]
    var yr: List[Float32]
    var n: Int
    var k: Int
    var kp: KernelParams
    var param: SvmParameter
    var min_r2: Float64
    """Assert the training R^2 is at least this. `-1.0` means DO NOT ASSERT,
    which is honest for R3 (targets independent of X: there is nothing to
    fit) and for the iteration-capped R5/R6."""
    var converges: Bool
    """True when the fixture is expected to reach `diff < tol` inside its
    outer-iteration budget AND the whole domain is the working set, so the
    GLOBAL KKT gap is asserted rather than reported. The solver's stop rule
    is on the working set only; that is why this is a per-fixture fact and
    not a formula."""
    var do_objective: Bool
    """Record `W(alpha)` per outer iteration. O(n^2 k) EVERY iteration, so
    only the three small fixtures carry it."""
    var do_tube: Bool
    """Assert the eps tube. Needs a converged fit AND a real population of
    rows with both multipliers at zero."""

    def __init__(
        out self,
        name: String,
        var x: List[Float32],
        var yr: List[Float32],
        n: Int,
        k: Int,
        kp: KernelParams,
        param: SvmParameter,
        min_r2: Float64,
        converges: Bool,
        do_objective: Bool,
        do_tube: Bool,
    ):
        self.name = name
        self.x = x^
        self.yr = yr^
        self.n = n
        self.k = k
        self.kp = kp.copy()
        self.param = param.copy()
        self.min_r2 = min_r2
        self.converges = converges
        self.do_objective = do_objective
        self.do_tube = do_tube


def _svr_param(C: Float64, epsilon: Float64, max_outer: Int) -> SvmParameter:
    var p = SvmParameter.default()
    p.svmType = EPSILON_SVR
    p.C = C
    p.epsilon = epsilon
    p.max_outer_iter = max_outer
    return p^


def _reg_x(n: Int, k: Int, scale: Float32, salt: Int) -> List[Float32]:
    var x = List[Float32]()
    for i in range(n):
        for c in range(k):
            x.append(scale * _unit(i * k + c, salt))
    return x^


def _linear_targets(
    x: List[Float32], n: Int, k: Int, salt: Int, noise: Float32
) -> List[Float32]:
    """`yr_i = sum_c w_c x_ic + noise * u_i` with `w` hashed: a target a
    LINEAR kernel can actually fit, so R^2 means something."""
    var w = List[Float32]()
    for c in range(k):
        w.append(_unit(c, salt + 5))
    var yr = List[Float32]()
    for i in range(n):
        var acc = Float32(0.0)
        for c in range(k):
            acc = acc + w[c] * x[i * k + c]
        yr.append(acc + noise * _unit(i, salt + 9))
    return yr^


def fixture_svr_line() -> RegFixture:
    """R1: the plain, well-conditioned one. `2n = 400 <= 1024`, so the
    working set is the WHOLE domain, one block solve per outer iteration and
    no selection: the global KKT gap is therefore the quantity the solver
    actually drives below `tol`, and it is ASSERTED here."""
    var x = _reg_x(200, 4, Float32(1.0), 101)
    var yr = _linear_targets(x, 200, 4, 101, Float32(0.03))
    return RegFixture(
        "R1.line", x^, yr^, 200, 4, KernelParams.linear(),
        _svr_param(1.0, 0.1, 60), 0.85, True, True, True,
    )


def fixture_svr_rbf() -> RegFixture:
    """R2: the RBF arm of the regression path, so `identical_exp` and the
    expanded `(nx + ny) - 2 dot` spelling are on the SVR gate too and not
    only on the classifier's. Targets are a smooth quadratic of `x`, which
    an RBF can fit and a linear kernel cannot."""
    var n = 160
    var x = _reg_x(n, 2, Float32(2.0), 113)
    var yr = List[Float32]()
    for i in range(n):
        var a = x[2 * i]
        var b = x[2 * i + 1]
        yr.append(
            Float32(0.15) * a * b + Float32(0.1) * (a * a - b * b)
            + Float32(0.02) * _unit(i, 117)
        )
    return RegFixture(
        "R2.rbf", x^, yr^, n, 2, KernelParams.rbf(0.5),
        _svr_param(10.0, 0.05, 60), 0.5, True, True, False,
    )


def fixture_svr_tube() -> RegFixture:
    """R3: THE EPS-TUBE GATE'S SUBJECT. The targets are hashed noise with no
    relation to `X` (so nothing is learnable and `min_r2` is not asserted)
    bounded to `[-0.5, 0.5)`, and `epsilon = 0.30`. The optimum therefore
    leaves most rows strictly inside the tube with `alpha_i = alpha*_i = 0`,
    which is exactly the population `svr_tube_bound` makes a claim about. A
    fixture where every row is a support vector would let the tube gate pass
    vacuously, and `check_svr_eps_tube` refuses to pass on fewer than `n/5`
    subjects."""
    var n = 240
    var x = _reg_x(n, 3, Float32(1.0), 127)
    var yr = List[Float32]()
    for i in range(n):
        yr.append(Float32(0.5) * _unit(i, 131))
    return RegFixture(
        "R3.tube", x^, yr^, n, 3, KernelParams.linear(),
        _svr_param(1.0, 0.30, 60), -1.0, True, True, True,
    )


def fixture_svr_dup() -> RegFixture:
    """R4: 150 hashed rows, EACH TWICE, with the target duplicated too, so
    two rows of `X` are bitwise equal and their kernel columns are bitwise
    equal. Two things this reaches that R1 does not:

      * `fold_order_for` sees the same PROJECTED index twice (as it does on
        every SVR fixture whose working set is the whole domain) AND, on top
        of that, two DIFFERENT projected indices whose kernel rows are
        identical, so the fold's order decides a float sum between equal
        multiplicands;
      * the block solve's three arg-reductions see genuinely equal `f`
        values, which is what `SAB_ARG_TIE_HIGH` moves.

    The QP is degenerate here (a duplicated row makes the solution
    non-unique). That is the POINT: determinism, not uniqueness, is what
    this lane claims, and both arms follow one tie rule."""
    var base_n = 150
    var k = 4
    var bx = _reg_x(base_n, k, Float32(1.0), 137)
    var by = _linear_targets(bx, base_n, k, 137, Float32(0.05))
    var x = List[Float32]()
    var yr = List[Float32]()
    for i in range(base_n):
        for _ in range(2):
            for c in range(k):
                x.append(bx[i * k + c])
            yr.append(by[i])
    return RegFixture(
        "R4.dup", x^, yr^, 2 * base_n, k, KernelParams.linear(),
        _svr_param(1.0, 0.1, 60), -1.0, True, False, False,
    )


def fixture_svr_big() -> RegFixture:
    """R5: `n = 1030`, so `n_train = 2060 > SMO_WS_SIZE` and the working set
    is a SELECTION: `WorkingSet::Select`'s FIFO carry-over of `2 * n_ws/4`
    from the previous set, `SimpleSelect`'s two `GatherAvailable` halves and
    the radix sort over the doubled domain all run. The projected indices
    collide only PARTIALLY here, which is the harder case for
    `fold_order_for` (a mix of distinct and repeated keys in one list).

    Capped at 8 outer iterations: the oracle is one host thread over a
    1024 x 1024 tile per iteration. The global KKT gap is REPORTED, not
    asserted, for the same reason F6.big's is -- the stop rule is on the
    working set."""
    var x = _reg_x(1030, 3, Float32(1.0), 149)
    var yr = _linear_targets(x, 1030, 3, 149, Float32(0.05))
    return RegFixture(
        "R5.big", x^, yr^, 1030, 3, KernelParams.linear(),
        _svr_param(1.0, 0.1, 8), -1.0, False, False, False,
    )


def fixture_svr_zero() -> RegFixture:
    """R6: THE SIGNED-ZERO FIXTURE, AND IT IS NOT PLANTED AT A KERNEL ENTRY.

    `epsilon = 0`, every target NON-NEGATIVE, and two planted families:
    rows `i % 47 == 0` have target `+0.0` and rows `i % 47 == 11` have target
    `-0.0`. Both are ordinary float32 inputs; `check_finite_list` accepts
    them and no other gate in the lane looks at a target's sign bit.

    WHAT THAT PUTS IN `f`, from `SvrInit` and IEEE 754 alone:

        target   f[i] = eps - y        f[i + n] = (-eps) - y
        -------  --------------------  ---------------------
        y > 0    0.0 - y   = -y        -0.0 - y  = -y          (equal bits)
        y = +0.0 0.0 - 0.0 = +0.0      -0.0 - 0.0 = -0.0
        y = -0.0 0.0 + 0.0 = +0.0      -0.0 + 0.0 = +0.0

    so the SECOND half of `f` carries `-0.0` at one planted family and
    `+0.0` at the other. Under C_SVC a `-0.0` in `f` needed a negative
    subnormal flushed at the f seam, which no 13-bit-significand input of
    O(1) scale can produce; here it falls out of the data.

    NON-NEGATIVE TARGETS ARE WHY THE ZEROS MATTER RATHER THAN JUST EXISTING.
    At `alpha = 0` the lower set is exactly the second half, where
    `f = -y <= 0` with equality only at the planted rows -- so the zeros ARE
    the argmax of `f` over the lower set, which is `f_max`'s tie and
    DEVIATION 635's rule, reached from data. `GatherAvailable`'s second call
    copies from the BACK of the f-order, so they are selected.

    `n = 600` puts `n_train = 1200` above `SMO_WS_SIZE`, so the twiddle keys
    and the radix sort see both signs too (`-0.0` twiddles to 0x7FFFFFFF and
    `+0.0` to 0x80000000, so `-0.0` sorts STRICTLY BEFORE `+0.0`).

    WHAT IT DOES NOT REACH, and this is stated because a gate that overclaims
    is worse than none: the recorded `diff` bits. `diff = f_max - f_u`, and
    `(+0.0) - c` and `(-0.0) - c` are the same float for every non-zero `c`,
    so the zero's sign is invisible there unless `f_u` is ALSO a zero -- and
    `f_u` is the min of `-y` over the first half, which is a zero only if
    every target is zero. The recorded stages the sign DOES reach are
    `svm.init.f` and every `svm.iterNNN.f` (FNV-1a over raw bytes: `+0.0`
    and `-0.0` hash differently) and `svm.iterNNN.ws_idx` (the twiddle
    order). The planted Z fixture is still the only thing that moves `diff`.
    """
    var n = 600
    var x = _reg_x(n, 3, Float32(1.0), 163)
    var yr = List[Float32]()
    var minus_zero = bitcast[DType.float32](UInt32(0x80000000))
    for i in range(n):
        # NON-NEGATIVE, so the lower set's f_max is a zero (see the paragraph
        # above); `-0.0` is written by BIT PATTERN, as `_zero_fixture` writes
        # it, so the fixture's content cannot depend on how a build folds a
        # unary minus on a zero literal. The NEGATION that matters is the one
        # inside `svr_init_kernel` and its oracle twin, and that is exactly
        # what this fixture is here to gate.
        var v = abs(_unit(i, 167))
        if i % 47 == 0:
            v = Float32(0.0)
        elif i % 47 == 11:
            v = minus_zero
        yr.append(v)
    return RegFixture(
        "R6.zero", x^, yr^, n, 3, KernelParams.linear(),
        _svr_param(1.0, 0.0, 6), -1.0, False, False, False,
    )


def all_reg_fixtures() -> List[RegFixture]:
    var out = List[RegFixture]()
    out.append(fixture_svr_line())
    out.append(fixture_svr_rbf())
    out.append(fixture_svr_tube())
    out.append(fixture_svr_dup())
    out.append(fixture_svr_big())
    out.append(fixture_svr_zero())
    return out^


def _reg_norms(rfx: RegFixture) -> List[Float32]:
    var norms = List[Float32]()
    if rfx.kp.kernel == KERNEL_RBF:
        for i in range(rfx.n):
            norms.append(_row_norm[DType.float32](rfx.x, i, rfx.k))
    return norms^


def _svr_init_f(rfx: RegFixture) -> List[Float32]:
    """`SvrInit`'s `f`, spelled here a second time ON PURPOSE. This one is
    not checking that anybody agrees with anybody: it is a statement about
    what the FIXTURE contains, used by `check_svr_signed_zero`."""
    var eps = Float32(rfx.param.epsilon)
    var neg_eps = -eps
    var f = List[Float32]()
    for i in range(rfx.n):
        f.append(eps - rfx.yr[i])
    for i in range(rfx.n):
        f.append(neg_eps - rfx.yr[i])
    return f^


# ===========================================================================
# THE SVR PROPERTY GATES (host only)
#
# These are the load-bearing part of the regression lane and the reason
# `SmoSolver.solve`'s refusal can come out at all. A device-versus-oracle
# comparison proves that `svm/impl/svm/*.mojo` and `svm/checks/smo_oracle.mojo`
# agree; it cannot prove that either is SOLVING AN SVR, because both were
# written from one reading of `smosolver.cuh` and a shared misreading of
# `SvrInit`'s signs or of `CombineCoefs`' direction would sit in both. Each
# gate below is derived from the eps-insensitive formulation written out in
# `smo_oracle.mojo`'s "THE REGRESSION PROPERTIES" header and reads no
# quantity the solver produced except `alpha`, `b` and the support set.
#
#   (1) check_svr_objective_decreases
#           W(alpha) = 1/2 sum_ij c_i c_j K_ij + eps sum_p alpha_p
#                      - sum_i yr_i c_i,     c_i = alpha_i - alpha_{i+n}
#       NON-INCREASING across outer iterations. What it catches: swapping
#       SvrInit's two lines solves the eps -> -eps problem, and the two W's
#       differ by 2 eps sum_p alpha_p, which GROWS as SMO lifts alpha off
#       zero -- so the descent of one is the ascent of the other.
#
#   (2) check_svr_kkt
#           g_i        = sum_j c_j K_ij                  (Float64, from alpha)
#           f_ref[i]   = g_i + eps - yr_i
#           f_ref[i+n] = g_i - eps - yr_i
#           gap = max{f_ref : I_low} - min{f_ref : I_up}   <  tol
#       The gradient is RECOMPUTED, so SvrInit, UpdateF and the solver's own
#       `f` are nowhere on the path; `max|f_ref - f|` is printed as the
#       loudest single signal in the lane (a sign error moves it by 2 eps).
#
#   (3) check_svr_eps_tube
#           for every row with alpha_i == 0 AND alpha*_i == 0:
#               |yr_i - decision(x_i)|  <=  eps + max(gap, 0)
#       Cheap and independent of (1) and (2): the only gate that touches
#       `b`, the FOLDED dual coefficients and the support set, so it is the
#       one a wrong CombineCoefs direction cannot survive.
#
# Each takes an OracleResult the caller already computed, so one fixture is
# solved ONCE for all of its gates. `svc_main.mojo` is the driver.
# ===========================================================================


def svr_oracle_fit(rfx: RegFixture) raises -> OracleResult[DType.float32]:
    """One host solve of a regression fixture. `smo_oracle_fit` dispatches on
    `param.svmType`; `rfx.yr` are TARGETS and the +-1 label vector is built
    inside, as `SvrInit` builds `y_label`."""
    return smo_oracle_fit[DType.float32](
        rfx.x, rfx.yr, rfx.n, rfx.k, rfx.param, rfx.kp, rfx.do_objective
    )


def _svr_r2(rfx: RegFixture, pred: List[Float32]) -> Float64:
    var mean = Float64(0)
    for i in range(rfx.n):
        mean += Float64(rfx.yr[i])
    mean /= Float64(rfx.n)
    var ss_res = Float64(0)
    var ss_tot = Float64(0)
    for i in range(rfx.n):
        var d = Float64(rfx.yr[i]) - Float64(pred[i])
        ss_res += d * d
        var t = Float64(rfx.yr[i]) - mean
        ss_tot += t * t
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def check_svr_objective_decreases(
    rfx: RegFixture, res: OracleResult[DType.float32]
) raises:
    """GATE (1). Non-increasing `W`, up to Float32 round-off in the Float64
    evaluation -- the same 1e-4 relative bar `check_oracle_objective_decreases`
    uses. A rise of 1e-2 is a sign error, not round-off."""
    if not rfx.do_objective:
        raise Error(rfx.name + ": fitted without record_objective")
    if len(res.objective_seq) < 2:
        raise Error(
            rfx.name + ": only " + String(len(res.objective_seq))
            + " outer iteration(s); the monotone gate needs at least 2"
        )
    var worst = 0.0
    var worst_at = 0
    for t in range(1, len(res.objective_seq)):
        var prev = res.objective_seq[t - 1]
        var cur = res.objective_seq[t]
        var rise = cur - prev
        var scale = abs(prev)
        if scale < 1.0:
            scale = 1.0
        if rise / scale > worst:
            worst = rise / scale
            worst_at = t
    print(
        "  " + rfx.name + " [" + _mode_name() + "] eps-insensitive dual W:"
        + " first=" + String(res.objective_seq[0]) + " last="
        + String(res.objective_seq[len(res.objective_seq) - 1]) + " over "
        + String(len(res.objective_seq)) + " outer iterations; worst relative"
        " rise=" + String(worst) + " at " + String(worst_at)
    )
    if worst > 1.0e-4:
        raise Error(
            rfx.name + ": the eps-insensitive dual ROSE by " + String(worst)
            + " relative at outer iteration " + String(worst_at)
            + "; SMO minimizes it, so this is a sign error, not round-off"
        )


def check_svr_kkt(rfx: RegFixture, res: OracleResult[DType.float32]) raises:
    """GATE (2). The SVR KKT gap over the 2n domain, off the INDEPENDENT
    gradient, below `tol` at convergence.

    `max|f_ref - f|` is asserted at `1e-3 * scale`, a Float64-vs-Float32 bar
    rather than a tight one; a SvrInit sign swap moves it by `2 * epsilon`,
    which is 1e-1 on these fixtures. The gap itself is asserted only when
    the whole domain is the working set (`converges`), because the solver's
    stop rule is the gap ON THE WORKING SET -- reported otherwise, exactly as
    `check_oracle_kkt_and_accuracy` reports it for F3 and F6."""
    var norms = _reg_norms(rfx)
    var f_ref = svr_gradient_reference[DType.float32](
        rfx.x, rfx.yr, res.alpha, norms, rfx.n, rfx.k, rfx.kp, rfx.param.epsilon
    )
    var max_df = 0.0
    var scale = 1.0
    for p in range(2 * rfx.n):
        var d = abs(f_ref[p] - Float64(res.f[p]))
        if d > max_df:
            max_df = d
        if abs(f_ref[p]) > scale:
            scale = abs(f_ref[p])
    var gap = svr_kkt_gap[DType.float32](f_ref, res.alpha, rfx.n, rfx.param.C)
    var dec = smo_oracle_decision[DType.float32](
        res, rfx.x, rfx.n, rfx.x, rfx.n, rfx.k, rfx.kp
    )
    var r2 = _svr_r2(rfx, dec)
    print(
        "  " + rfx.name + " [" + _mode_name() + "] SVR oracle: n_train="
        + String(2 * rfx.n) + " n_ws=" + String(2 * rfx.n if 2 * rfx.n < SMO_WS_SIZE else SMO_WS_SIZE)
        + " n_support=" + String(len(res.support_idx))
        + " outer=" + String(res.n_outer_iter) + " inner=" + String(res.n_iter)
        + " b=" + _show(res.b) + " kkt_gap(independent)=" + String(gap)
        + " max|f_ref-f|=" + String(max_df) + " R2=" + String(r2)
    )
    if max_df > 1.0e-3 * scale:
        raise Error(
            rfx.name + ": the eps-insensitive gradient recomputed FROM ALPHA"
            " differs from the solver's f by " + String(max_df) + " (scale "
            + String(scale) + "): SvrInit or UpdateF is wrong"
        )
    if rfx.converges:
        if not (gap < rfx.param.tol):
            raise Error(
                rfx.name + ": SVR KKT gap " + String(gap) + " not below tol "
                + String(rfx.param.tol) + " over the 2n domain"
            )
    else:
        print(
            "    (2n > n_ws or the outer-iteration budget is capped: the"
            " global gap is REPORTED; the stop rule is on the working set)"
        )
    if rfx.min_r2 > -1.0:
        if r2 < rfx.min_r2:
            raise Error(
                rfx.name + ": training R2 " + String(r2) + " below "
                + String(rfx.min_r2)
            )


def check_svr_eps_tube(rfx: RegFixture, res: OracleResult[DType.float32]) raises:
    """GATE (3). THE TUBE. For a converged fit, every row whose BOTH
    multipliers are zero has its residual inside `+-epsilon`.

    `svr_tube_bound` derives the exact allowance `epsilon + max(gap, 0)`;
    the Float32 slack on top is `1e-3 * (1 + max|yr|)`, a round-off allowance
    for a length-`n_support` Float32 chain and two orders below the
    `2 * epsilon` a fold-direction error produces.

    The subject population is `alpha_i == 0 AND alpha*_i == 0`, NOT `the
    folded coefficient is zero`: those differ when both multipliers are equal
    and nonzero, and the derivation covers only the former. The size of the
    difference is reported. Fewer than `n/5` subjects FAILS rather than
    passing vacuously."""
    var norms = _reg_norms(rfx)
    var f_ref = svr_gradient_reference[DType.float32](
        rfx.x, rfx.yr, res.alpha, norms, rfx.n, rfx.k, rfx.kp, rfx.param.epsilon
    )
    var gap = svr_kkt_gap[DType.float32](f_ref, res.alpha, rfx.n, rfx.param.C)
    var bound = svr_tube_bound(rfx.param.epsilon, gap)
    var dec = smo_oracle_decision[DType.float32](
        res, rfx.x, rfx.n, rfx.x, rfx.n, rfx.k, rfx.kp
    )
    var max_abs_y = 0.0
    for i in range(rfx.n):
        if abs(Float64(rfx.yr[i])) > max_abs_y:
            max_abs_y = abs(Float64(rfx.yr[i]))
    var slack = 1.0e-3 * (1.0 + max_abs_y)
    var n_free_rows = 0
    var n_folded_zero_but_bound = 0
    var worst = 0.0
    var worst_at = -1
    for i in range(rfx.n):
        var a = Float64(res.alpha[i])
        var astar = Float64(res.alpha[i + rfx.n])
        if a == 0.0 and astar == 0.0:
            n_free_rows += 1
            var r = abs(Float64(rfx.yr[i]) - Float64(dec[i]))
            if r > worst:
                worst = r
                worst_at = i
        elif a - astar == 0.0:
            n_folded_zero_but_bound += 1
    print(
        "  " + rfx.name + " [" + _mode_name() + "] eps tube: epsilon="
        + String(rfx.param.epsilon) + " gap=" + String(gap) + " bound="
        + String(bound) + " (+slack " + String(slack) + "); rows with BOTH"
        " multipliers zero=" + String(n_free_rows) + " of " + String(rfx.n)
        + " (folded-zero-but-bound=" + String(n_folded_zero_but_bound)
        + "); worst |residual| among them=" + String(worst)
        + (" at row " + String(worst_at) if worst_at >= 0 else "")
    )
    if n_free_rows * 5 < rfx.n:
        raise Error(
            rfx.name + ": only " + String(n_free_rows) + " of " + String(rfx.n)
            + " rows have both multipliers at zero, so the tube gate would"
            " pass vacuously. Widen epsilon or lower C on this fixture."
        )
    if worst > bound + slack:
        raise Error(
            rfx.name + ": row " + String(worst_at) + " is a non-support-vector"
            " whose |residual| " + String(worst) + " is OUTSIDE the tube bound "
            + String(bound) + " + slack " + String(slack)
        )


# ===========================================================================
# THE TWO TRAPS THE BRIEF NAMED, MEASURED RATHER THAN ASSUMED
# ===========================================================================


def check_svr_fold_order(rfx: RegFixture, res: OracleResult[DType.float32]) raises:
    """`fold_order_for`'s docstring says working-set indices are distinct.
    THAT IS FALSE UNDER SVR, and this measures how false it is here.

    NOT A FIX: the function is deliberately untouched (out of this lane's
    scope, and the docstring is the orchestrator's to correct). What is
    checked instead is that its ORDER STAYS TOTAL, which it does for a reason
    the docstring does not state: the sort key is `(index << 32) | position`
    and the POSITION is unique, so equal indices are broken by their position
    in the nonzero list -- which is the compaction's order, ascending working-
    set position. The permutation is therefore well defined and launch-
    invariant even though the stated premise fails. THE CLAIM IS WRONG; THE
    CODE IS RIGHT, and the device-vs-oracle gate on these fixtures is what
    holds the two spellings of that order together.

    Asserts (a) `fold_order_for` returns a permutation ascending in
    `(projected index, position)` over the WORST working set this fit
    produced, and (b) collisions actually occur -- but (b) ONLY WHERE THEY
    ARE PROVABLE, which is not everywhere:

      * `2n <= SMO_WS_SIZE` (R1-R4): the working set is the whole domain, so
        every row appears exactly twice. n_collide == n_rows.
      * `n < SMO_WS_SIZE` (R6, n = 600): `SimpleSelect` takes `n_ws/2 = 512`
        rows from the upper set and 512 from the lower set, and at
        `alpha = 0` those sets are the two halves, so the two 512-element
        row-index sets must overlap in at least `1024 - n` rows.
      * `n >= SMO_WS_SIZE` (R5, n = 1030): NOTHING IS GUARANTEED. 512 + 512
        row indices fit inside 1030 without touching, so iteration 0 can be
        collision-free, and whether the FIFO carry-over produces one later
        is a property of the data, not of the shapes. REPORTED there, and
        that is why R6 exists beside it rather than R5 alone."""
    if len(res.ws_seq) == 0:
        raise Error(rfx.name + ": no outer iteration recorded")
    # the worst working set over the whole fit, not just the first
    var best_t = 0
    var n_collide = -1
    for t in range(len(res.ws_seq)):
        var seen_t = List[Int]()
        for _ in range(rfx.n):
            seen_t.append(0)
        var c = 0
        for q in range(len(res.ws_seq[t])):
            var v = _vec_index(Int(res.ws_seq[t][q]), rfx.n, True)
            seen_t[v] = seen_t[v] + 1
            if seen_t[v] == 2:
                c += 1
        if c > n_collide:
            n_collide = c
            best_t = t
    var ws0 = res.ws_seq[best_t].copy()
    var proj = List[Int32]()
    for t in range(len(ws0)):
        proj.append(Int32(_vec_index(Int(ws0[t]), rfx.n, True)))
    var order = fold_order_for(proj)
    var used = List[Bool]()
    for _ in range(len(proj)):
        used.append(False)
    var bad = String("")
    for r in range(len(order)):
        var j = Int(order[r])
        if j < 0 or j >= len(proj) or used[j]:
            bad = "fold_order_for is not a permutation, rank " + String(r)
            break
        used[j] = True
        if r > 0:
            var jp = Int(order[r - 1])
            var ok = (proj[jp] < proj[j]) or (proj[jp] == proj[j] and jp < j)
            if not ok:
                bad = (
                    "fold_order_for is not ascending in (index, position) at"
                    " rank " + String(r)
                )
                break
    var must_collide = rfx.n < SMO_WS_SIZE
    print(
        "  " + rfx.name + " [" + _mode_name() + "] fold-order collisions:"
        " worst working set is iteration " + String(best_t) + ", holding "
        + String(len(ws0)) + " of " + String(2 * rfx.n) + " domain points over "
        + String(rfx.n) + " rows; " + String(n_collide) + " rows appear TWICE"
        " (ws_idx % n_rows collides, and fold_order_for's docstring says they"
        " cannot); the order is "
        + ("TOTAL and ascending in (index, position)" if bad == "" else bad)
        + ("; collisions ASSERTED" if must_collide else "; collisions REPORTED (n >= SMO_WS_SIZE: not provable from the shapes)")
    )
    if bad != "":
        raise Error(rfx.name + ": " + bad)
    if must_collide and n_collide == 0:
        raise Error(
            rfx.name + ": no projected-index collision in ANY working set,"
            " though n < SMO_WS_SIZE makes one unavoidable; the selection or"
            " _vec_index is wrong"
        )


def check_svr_signed_zero(rfx: RegFixture, res: OracleResult[DType.float32]) raises:
    """R6: BOTH ZERO SIGNS IN `f` FROM ORDINARY INPUT, and an honest account
    of what they do and do not reach.

    ASSERTS, on the fixture's own initial `f`:
      * at least 8 entries are `+0.0` and at least 8 are `-0.0`;
      * every `-0.0` lies in the SECOND half, because `(-eps) - y` is the
        only line of `SvrInit` that can produce one;
      * `+0.0` and `-0.0` are BOTH present in the LOWER set at `alpha = 0`
        (the second half is exactly that set), which is `f_max`'s domain and
        DEVIATION 635's tie;
      * both are the MAXIMUM over that set, so the tie is decisive and not
        merely present;
      * the working set the selection ACTUALLY produced at outer iteration 0
        holds both signs.

    REPORTS, without asserting, the negative result, because a gate that
    overclaims is worse than no gate: the signs do NOT reach the recorded
    `diff` bits. `diff = f_max - f_u`, and `(+0.0) - c` and `(-0.0) - c` are
    the same float for every non-zero `c`; making `f_u` a zero as well needs
    EVERY target to be zero, which is not a regression. The stages the sign
    DOES reach, and which are bitwise gated device-vs-oracle, are `svm.init.f`
    and every `svm.iterNNN.f` (FNV-1a over raw bytes: the two zeros hash
    differently) and `svm.iterNNN.ws_idx` (twiddle: `-0.0` -> 0x7fffffff
    sorts strictly before `+0.0` -> 0x80000000). The PLANTED Z fixture
    remains the only thing in this lane that moves `diff` by a zero's sign,
    and the README's signed-zero section still holds for `diff`."""
    var f0 = _svr_init_f(rfx)
    var n_pos = 0
    var n_neg = 0
    var neg_in_first_half = 0
    for p in range(2 * rfx.n):
        if _bits(f0[p]) == UInt32(0x00000000):
            n_pos += 1
        elif _bits(f0[p]) == UInt32(0x80000000):
            n_neg += 1
            if p < rfx.n:
                neg_in_first_half += 1
    var f_max = f0[rfx.n]
    for p in range(rfx.n, 2 * rfx.n):
        if f0[p] > f_max:
            f_max = f0[p]
    var n_max_pos = 0
    var n_max_neg = 0
    for p in range(rfx.n, 2 * rfx.n):
        if f0[p] == f_max:
            if _bits(f0[p]) == UInt32(0x00000000):
                n_max_pos += 1
            elif _bits(f0[p]) == UInt32(0x80000000):
                n_max_neg += 1
    var ws0 = res.ws_seq[0].copy()
    var ws_pos = 0
    var ws_neg = 0
    for t in range(len(ws0)):
        var b = _bits(f0[Int(ws0[t])])
        if b == UInt32(0x00000000):
            ws_pos += 1
        elif b == UInt32(0x80000000):
            ws_neg += 1
    print(
        "  " + rfx.name + " [" + _mode_name() + "] signed zeros in SvrInit's f:"
        " +0.0 x " + String(n_pos) + ", -0.0 x " + String(n_neg)
        + " (every -0.0 in the second half: "
        + ("yes" if neg_in_first_half == 0 else "NO") + "); f_max over the"
        " lower set = " + _show(f_max) + " tied by " + String(n_max_pos)
        + " x +0.0 and " + String(n_max_neg) + " x -0.0; the selected ws0 ("
        + String(len(ws0)) + " entries) holds " + String(ws_pos) + " x +0.0"
        " and " + String(ws_neg) + " x -0.0"
    )
    print(
        "    REACHED: the twiddle keys and the radix sort (-0.0 -> 0x7fffffff"
        " sorts strictly before +0.0 -> 0x80000000), the f_max argmax tie"
        " (DEVIATION 635), svm.init.f and every svm.iterNNN.f hash."
    )
    print(
        "    NOT REACHED, recorded deliberately: the `diff` bits. diff ="
        " f_max - f_u and (+-0) - c is one float for every non-zero c; f_u is"
        " a zero only if every target is zero. The planted Z fixture remains"
        " the only thing that moves diff by a zero's sign."
    )
    if n_pos < 8 or n_neg < 8:
        raise Error(
            rfx.name + ": f carries " + String(n_pos) + " +0.0 and "
            + String(n_neg) + " -0.0; the fixture was built for at least 8 of"
            " each"
        )
    if neg_in_first_half != 0:
        raise Error(
            rfx.name + ": " + String(neg_in_first_half) + " -0.0 entries sit in"
            " the FIRST half of f, which `eps - y` cannot produce"
        )
    if n_max_pos == 0 or n_max_neg == 0:
        raise Error(
            rfx.name + ": f_max over the lower set is not tied by BOTH zero"
            " signs (+0.0 x " + String(n_max_pos) + ", -0.0 x "
            + String(n_max_neg) + "); DEVIATION 635's tie is not reached from"
            " ordinary input here"
        )
    if ws_pos == 0 or ws_neg == 0:
        raise Error(
            rfx.name + ": the working set at outer iteration 0 does not hold"
            " both zero signs (+0.0 x " + String(ws_pos) + ", -0.0 x "
            + String(ws_neg) + ")"
        )


def check_svr_sabotage_reach(
    rfx: RegFixture, res: OracleResult[DType.float32]
) raises:
    """WHAT EACH EXISTING SABOTAGE IS PREDICTED TO DO TO THIS FIXTURE, and
    whether the fixture REACHES its branch at all. `verify reach, not output`:
    a sabotage that a fixture cannot reach passes, and a pass then means
    nothing.

    The sabotages are compile-time `-D` defines, so this function cannot turn
    them on. What it does is measure, from the oracle's own run, the
    PRECONDITION each one needs, and print the prediction beside it. The
    orchestrator then rebuilds under each define and checks the printed
    prediction against the printed result.

      SAB_FOLD_ROTATE   `start = block_idx.x % nnz` in `update_f_kernel`.
                        Needs nnz >= 2 AND MORE THAN ONE BLOCK, i.e.
                        n_rows > SEL_TPB = 256 -- with one block `block_idx.x`
                        is 0 and the rotation is the identity. R1/R2/R3 are
                        INERT for that reason alone; R4/R5/R6 reach it.
      SAB_WS_TIE        reversed position into the stable sort. Needs the
                        sort to RUN (n_train > n_ws, so n > 512) and equal f
                        values. R6 is the strong case: with epsilon = 0,
                        `0 - y` and `-0.0 - y` are the same float for every
                        non-zero target, so the halves are pairwise tied.
      SAB_ARG_TIE_HIGH  both arg-reductions tie to the HIGHER key. Needs two
                        working-set entries with bitwise-equal f in the SAME
                        membership set. R4 (duplicated rows) and R6 (the
                        +0.0 family) reach it; on the others it depends on a
                        hash collision and is reported, not predicted.
      SAB_FMAX_*        decide which index wins the f_max tie. REACHED on R6
                        and PREDICTED BIT-INERT THERE ANYWAY: f_max's only
                        consumer is `diff = f_max - f_u`, so a tie between
                        equal values cannot move a bit unless the values
                        differ in their SIGN OF ZERO and f_u is not a zero.
                        See `check_svr_signed_zero`. The planted Z fixture
                        stays the arm that bites.
      SAB_STD_EXP       RBF only: R2, and only under IDENTICAL.
      SAB_NO_FTZ        reached by every fixture at the f seam; bit-inert on
                        an FTZ backend (Apple), REPORTED there."""
    var f0 = _svr_init_f(rfx)
    var blocks = (rfx.n + 256 - 1) // 256
    var max_nnz = 0
    for t in range(len(res.nnz_seq)):
        if res.nnz_seq[t] > max_nnz:
            max_nnz = res.nnz_seq[t]
    var n_train = 2 * rfx.n
    var n_ws = n_train if n_train < SMO_WS_SIZE else SMO_WS_SIZE
    var sort_runs = n_ws < n_train
    # How many domain points share their f BITS with another point of the
    # SAME membership set, at alpha = 0 where the first half is exactly I_up
    # and the second half exactly I_low. That is the precondition
    # SAB_ARG_TIE_HIGH needs: two candidates the reduction can order two
    # ways. Counted over the whole half rather than over the selected
    # working set, so on R5/R6 (where the working set is a subset) this is
    # an UPPER BOUND on reach and is reported as such, not asserted.
    var ties_up = 0
    var ties_low = 0
    var half = rfx.n
    for a in range(half):
        for b in range(a + 1, half):
            if _bits(f0[a]) == _bits(f0[b]):
                ties_up += 1
                break
    for a in range(half, 2 * half):
        for b in range(a + 1, 2 * half):
            if _bits(f0[a]) == _bits(f0[b]):
                ties_low += 1
                break
    print(
        "  " + rfx.name + " [" + _mode_name() + "] sabotage reach:"
        " update_f blocks=" + String(blocks) + " (FOLD_ROTATE needs > 1),"
        " max nnz=" + String(max_nnz) + " (needs >= 2) -> FOLD_ROTATE "
        + ("REACHED" if (blocks > 1 and max_nnz >= 2) else "INERT")
        + " | ws sort runs=" + ("yes" if sort_runs else "no")
        + " -> WS_TIE " + ("REACHED" if (sort_runs and ties_up + ties_low > 0) else "INERT")
        + " | f-bit ties at iter 0: upper=" + String(ties_up) + " lower="
        + String(ties_low) + " -> ARG_TIE_HIGH "
        + ("REACHED" if ties_up + ties_low > 0 else "INERT")
        + " | kernel=" + ("rbf -> STD_EXP REACHED" if rfx.kp.kernel == KERNEL_RBF else "linear -> STD_EXP INERT")
    )
    print(
        "    FMAX_NOKEY / FMAX_HWMAX / FMAX_HWMAX_SWAP: PREDICTED BIT-INERT on"
        " every regression fixture. f_max feeds only `diff = f_max - f_u`, so"
        " a tie among equal values cannot move a recorded bit, and the one"
        " tie whose members differ in bits (+0.0 vs -0.0, R6) is subtracted"
        " from a non-zero f_u. The planted Z fixture is the arm that bites."
    )


# ===========================================================================
# THE SVR DEVICE GATE
#
# There is no `svr_fit` on the estimator surface, and this file does not add
# one -- it spells `svrFitX` (`svr_impl.cuh:37-83`) inline, which is four
# lines: validate, upload, `SmoSolver::Solve`, `model.n_cols = n_cols`. The
# ONE thing upstream does that this cannot is skip the label machinery,
# because `svcFit` computes `getUniquelabels` and asserts two classes and a
# regression target has neither. `svrFitX` never calls it either; the
# regression path reaches `Solve` directly, and so does this.
#
# `model.unique_labels` is filled with a dummy pair. `decision_kernel` takes
# `label0`/`label1` as arguments in every build but reads them only when
# `predict_class_in != 0`, and SVR prediction is `svcPredict(...,
# predict_class = false)` -- upstream's `svrPredict` IS that call. So the
# dummies change no number, and `predict_class = true` is never asked for
# here.
# ===========================================================================


def _reg_queries(rfx: RegFixture) -> List[Float32]:
    """The training rows plus 37 fresh hashed rows, as `_queries` does."""
    var q = rfx.x.copy()
    for i in range(37):
        for c in range(rfx.k):
            q.append(Float32(2.0) * _unit(i * rfx.k + c, 97))
    return q^


def _run_svr_device(
    ctx: DeviceContext,
    rfx: RegFixture,
    block_solve_threads: Int,
    kernel_tile_byte_limit: Int,
    predict_buffer_mib: Float64,
    scratch_pad: Int,
    scratch_poison: Float32,
    mut card: IdentityTrace,
) raises -> DeviceRun:
    """`svrFitX` + `svrPredict`, dense FP32. `classes` comes back EMPTY:
    there is no class arm on a regressor."""
    if rfx.k <= 0:
        raise Error("Parameter n_cols: number of columns cannot be less than one")
    if rfx.n <= 0:
        raise Error("Parameter n_rows: number of rows cannot be less than one")
    if len(rfx.x) != rfx.n * rfx.k or len(rfx.yr) != rfx.n:
        raise Error("svr fit: x / y sizes do not match n_rows x n_cols")
    check_rung1_scope(rfx.param, rfx.kp, False)
    # DEVIATION 636 applies to the regression path too: theirs validates
    # nothing, ours refuses a non-finite target for the same reason it
    # refuses a non-finite label -- a NaN would land in `svm.init.f` with a
    # vendor-specific payload on the very first recorded stage.
    check_finite_list(rfx.x, "X")
    check_finite_list(rfx.yr, "labels")

    var model = SvmModel()
    model.n_classes = 0
    var dummy_labels: List[Float32] = [Float32(-1.0), Float32(1.0)]
    model.unique_labels = dummy_labels^
    card.header(
        "svm.svr_fit n_rows=" + String(rfx.n) + " n_cols=" + String(rfx.k)
        + " kernel=" + String(rfx.kp.kernel) + " C=" + String(rfx.param.C)
        + " epsilon=" + String(rfx.param.epsilon)
        + " tol=" + String(rfx.param.tol)
    )
    var x = upload_f32(ctx, rfx.x)
    var yv = upload_f32(ctx, rfx.yr)
    ctx.synchronize()
    card.record_device[DType.float32](ctx, "svm.input.x", x, rfx.n * rfx.k)
    card.record_device[DType.float32](ctx, "svm.input.y", yv, rfx.n)

    var smo = SmoSolver(
        ctx, rfx.param, rfx.kp, rfx.n, rfx.k, kernel_tile_byte_limit,
        block_solve_threads, True, scratch_pad, scratch_poison,
    )
    smo.solve(
        ctx, x, yv, model, card, rfx.param.max_iter, rfx.param.max_outer_iter
    )
    model.n_cols = rfx.k
    ctx.synchronize()
    var trace = smo.trace^
    smo.trace = SmoTrace()
    _ = smo^
    _ = x^
    _ = yv^

    var q = _reg_queries(rfx)
    var nq = rfx.n + 37
    var dec = svc_predict(
        ctx, model, q, nq, rfx.k, rfx.kp, predict_buffer_mib, False, card
    )
    return DeviceRun(model^, trace^, dec^, List[Float32]())


def _svr_blocked(msg: String) -> Bool:
    """True when the failure is `SmoSolver.solve`'s standing UNGATED refusal
    rather than a divergence. That clause is NOT this file's to delete -- it
    comes out by hand once these gates have been seen to pass."""
    return msg.find("UNGATED") >= 0


def _compare_svr_device_to_oracle(
    rfx: RegFixture, run: DeviceRun, res: OracleResult[DType.float32], strict: Bool
) raises -> Int:
    """Device vs oracle on a regression fit: the working-set sequence, the
    alpha and f hashes over the 2n domain per outer iteration, `diff`, the
    inner-iteration count, `b`, the FOLDED dual coefficients, the support
    indices and the decision function on n + 37 queries. Same comparison as
    `_compare_device_to_oracle`, minus the class arm, which a regressor has
    no counterpart for."""
    var q = _reg_queries(rfx)
    var nq = rfx.n + 37
    var odec = smo_oracle_decision[DType.float32](
        res, rfx.x, rfx.n, q, nq, rfx.k, rfx.kp
    )
    var n_div = 0
    var first = String("")
    if len(run.trace.ws_seq) != len(res.ws_seq):
        n_div += 1
        first = (
            "outer iteration count device " + String(len(run.trace.ws_seq))
            + " vs oracle " + String(len(res.ws_seq))
        )
    var n_it = len(run.trace.ws_seq)
    if len(res.ws_seq) < n_it:
        n_it = len(res.ws_seq)
    for t in range(n_it):
        var m = String("")
        if not _same_i32(run.trace.ws_seq[t], res.ws_seq[t]):
            m = "ws sequence differs at outer iteration " + String(t)
        elif run.trace.alpha_hash_seq[t] != res.alpha_hash_seq[t]:
            m = "alpha hash differs at outer iteration " + String(t)
        elif run.trace.f_hash_seq[t] != res.f_hash_seq[t]:
            m = "f hash differs at outer iteration " + String(t)
        elif _bits(run.trace.diff_seq[t]) != _bits(res.diff_seq[t]):
            m = (
                "diff differs at outer iteration " + String(t) + ": "
                + _show(run.trace.diff_seq[t]) + " vs " + _show(res.diff_seq[t])
            )
        elif run.trace.inner_iter_seq[t] != res.inner_iter_seq[t]:
            m = "inner iteration count differs at outer iteration " + String(t)
        elif run.trace.nnz_seq[t] != res.nnz_seq[t]:
            m = "nonzero delta_alpha count differs at outer iteration " + String(t)
        if m != "":
            n_div += 1
            if first == "":
                first = m
    if _bits(run.model.b) != _bits(res.b):
        n_div += 1
        if first == "":
            first = "b differs: device " + _show(run.model.b) + " oracle " + _show(res.b)
    if len(run.model.dual_coefs) != len(res.dual_coefs):
        n_div += 1
        if first == "":
            first = (
                "n_support differs: device " + String(len(run.model.dual_coefs))
                + " oracle " + String(len(res.dual_coefs))
            )
    else:
        for j in range(len(res.dual_coefs)):
            if (
                _bits(run.model.dual_coefs[j]) != _bits(res.dual_coefs[j])
                or run.model.support_idx[j] != res.support_idx[j]
            ):
                n_div += 1
                if first == "":
                    first = "folded dual coef / support idx differs at " + String(j)
                break
    for i in range(nq):
        if _bits(run.decision[i]) != _bits(odec[i]):
            n_div += 1
            if first == "":
                first = (
                    "decision function differs at query " + String(i) + ": "
                    + _show(run.decision[i]) + " vs " + _show(odec[i])
                )
            break
    var verdict = "IDENTICAL" if n_div == 0 else ("DIVERGES (" + String(n_div) + ": " + first + ")")
    print(
        "  " + rfx.name + " [" + _mode_name() + "] SVR device vs oracle: "
        + verdict + "  outer=" + String(len(run.trace.ws_seq)) + " n_support="
        + String(run.model.n_support) + " b=" + _show(run.model.b)
    )
    if strict and n_div != 0:
        raise Error(rfx.name + ": SVR device diverges from the oracle: " + first)
    return n_div


def check_svr_device_matches_oracle(
    ctx: DeviceContext, fixtures: List[RegFixture]
) raises:
    """THE SVR GATE. Under IDENTICAL every regression fixture must be bitwise
    the oracle over the 2n domain; under FAST the divergence count is
    reported.

    WHILE `SmoSolver.solve`'s UNGATED clause IS STILL IN THE TREE THIS GATE
    FAILS, and that is the honest state: nothing was compared. It prints
    BLOCKED and names the clause. It is not this file's to remove; the three
    host property gates above pass without the device, which is what makes
    removing it safe to do by hand."""
    var card = IdentityTrace.disabled()
    var total = 0
    var blocked = 0
    for i in range(len(fixtures)):
        var res = svr_oracle_fit(fixtures[i])
        var msg = String("")
        var run_ok = False
        var n_div = 0
        try:
            var run = _run_svr_device(
                ctx, fixtures[i], 0, 1 << 30, 200.0, 0, Float32(0.0), card
            )
            n_div = _compare_svr_device_to_oracle(
                fixtures[i], run, res, _is_identical()
            )
            run_ok = True
        except e:
            msg = String(e)
        if run_ok:
            total += n_div
        elif _svr_blocked(msg):
            blocked += 1
            print(
                "  " + fixtures[i].name + " [" + _mode_name() + "] SVR device"
                " vs oracle: BLOCKED -- SmoSolver.solve still refuses"
                " EPSILON_SVR as UNGATED, so NOTHING WAS COMPARED."
            )
        else:
            raise Error(fixtures[i].name + ": " + msg)
    if blocked > 0:
        raise Error(
            String(blocked) + " of " + String(len(fixtures)) + " regression"
            " fixtures were BLOCKED by SmoSolver.solve's UNGATED refusal."
            " The host property gates (objective, KKT, tube) run without the"
            " device and are what that clause was waiting on; delete the"
            " `if self.svmType == EPSILON_SVR: raise` block in"
            " svm/impl/svm/smosolver.mojo::solve and re-run."
        )
    if not _is_identical():
        print(
            "  [FAST] SVR device-vs-oracle is a REPORT in this mode: "
            + String(total) + " divergences over " + String(len(fixtures))
            + " fixtures"
        )


def _same_run_reg(a: DeviceRun, b: DeviceRun) -> String:
    """`_same_run` without the class arm."""
    if len(a.trace.ws_seq) != len(b.trace.ws_seq):
        return "outer iteration count"
    for t in range(len(a.trace.ws_seq)):
        if a.trace.alpha_hash_seq[t] != b.trace.alpha_hash_seq[t]:
            return "alpha hash at outer iteration " + String(t)
        if a.trace.f_hash_seq[t] != b.trace.f_hash_seq[t]:
            return "f hash at outer iteration " + String(t)
        if not _same_i32(a.trace.ws_seq[t], b.trace.ws_seq[t]):
            return "ws at outer iteration " + String(t)
    if _bits(a.model.b) != _bits(b.model.b):
        return "b"
    if len(a.model.dual_coefs) != len(b.model.dual_coefs):
        return "n_support"
    for j in range(len(a.model.dual_coefs)):
        if _bits(a.model.dual_coefs[j]) != _bits(b.model.dual_coefs[j]):
            return "dual coef " + String(j)
        if a.model.support_idx[j] != b.model.support_idx[j]:
            return "support idx " + String(j)
    for i in range(len(a.decision)):
        if _bits(a.decision[i]) != _bits(b.decision[i]):
            return "decision at " + String(i)
    return ""


def check_svr_device_is_launch_invariant(ctx: DeviceContext) raises:
    """The launch-invariance headline on the REGRESSION path. It is a
    separate gate from the classifier's because SVR adds a launch the
    classifier does not have: `UpdateF` runs TWICE per batch, on `f` and on
    `f + n_rows`, over the same tile and the same `delta_alpha`. Batching the
    full tile therefore interleaves twice as many launches, and the
    invariance claim has to be made again over that shape.

    R4.dup (n = 300, two update_f blocks) and R6.zero (n = 600, the sort and
    both zero signs). Asserted under IDENTICAL, reported under FAST."""
    var card = IdentityTrace.disabled()
    var blocked = 0
    for i in range(2):
        var rfx: RegFixture
        if i == 0:
            rfx = fixture_svr_dup()
        else:
            rfx = fixture_svr_zero()
        var n_train = 2 * rfx.n
        var n_ws = n_train if n_train < SMO_WS_SIZE else SMO_WS_SIZE
        var msg = String("")
        var verdict = String("")
        try:
            var base = _run_svr_device(ctx, rfx, 0, 1 << 30, 200.0, 0, Float32(0.0), card)
            var arm_a = _run_svr_device(ctx, rfx, 1024, 1 << 30, 200.0, 0, Float32(0.0), card)
            var da = _same_run_reg(base, arm_a)
            var arm_b = _run_svr_device(ctx, rfx, 0, 256 * n_ws * 4, 200.0, 0, Float32(0.0), card)
            var db = _same_run_reg(base, arm_b)
            var arm_c = _run_svr_device(ctx, rfx, 0, 1 << 30, 0.001, 0, Float32(0.0), card)
            var dc = _same_run_reg(base, arm_c)
            var arm_d = _run_svr_device(ctx, rfx, 0, 1 << 30, 200.0, 37, Float32(-7.25e20), card)
            var dd = _same_run_reg(base, arm_d)
            var arm_e = _run_svr_device(ctx, rfx, 1024, 100 * n_ws * 4, 0.001, 1029, Float32(3.0e-39), card)
            var de = _same_run_reg(base, arm_e)
            if da != "":
                verdict += " block1024:" + da
            if db != "":
                verdict += " batch256:" + db
            if dc != "":
                verdict += " predict_batch:" + dc
            if dd != "":
                verdict += " pad37:" + dd
            if de != "":
                verdict += " all:" + de
            print(
                "  " + rfx.name + " [" + _mode_name() + "] SVR launch"
                " invariance over 5 arms: "
                + ("INVARIANT" if verdict == "" else "MOVES:" + verdict)
                + "  (outer=" + String(len(base.trace.ws_seq)) + ")"
            )
        except e:
            msg = String(e)
            if _svr_blocked(msg):
                blocked += 1
                print(
                    "  " + rfx.name + " [" + _mode_name() + "] SVR launch"
                    " invariance: BLOCKED by the UNGATED refusal"
                )
            else:
                raise Error(rfx.name + ": " + msg)
        if verdict != "":
            if _is_identical():
                raise Error(rfx.name + ": the bytes moved across launches:" + verdict)
            print(
                "    [FAST] REPORTED, not asserted: the vendor matmul folds"
                " per shape"
            )
    if blocked > 0:
        raise Error(
            "SVR launch invariance BLOCKED on " + String(blocked)
            + " fixture(s) by SmoSolver.solve's UNGATED refusal"
        )
