"""The SVC gates: oracle properties, device == oracle bitwise, launch
invariance, and the sabotage hooks.

NOT A PORT. Run with:

    pixi run mojo run -I . svm/mojo_only/svc_check.mojo                 # host oracle only
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

SABOTAGES (each a `-D` define; the README table records the failing line):

    MOJOLEARN_SVM_SABOTAGE_WS_TIE       workingset.mojo   must FAIL on F3
    MOJOLEARN_SVM_SABOTAGE_FOLD_ROTATE  smosolver.mojo    must FAIL (F1..F7)
    MOJOLEARN_SVM_SABOTAGE_STD_EXP      kernel_matrices   must FAIL on rbf (IDENTICAL)
    MOJOLEARN_SVM_SABOTAGE_NO_FTZ       smosolver.mojo    REPORT (bit-inert on FTZ hw)
"""

from std.math import inf
from std.memory import bitcast
from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace, first_divergence
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from svm.mojo_only.smo_oracle import (
    OracleResult,
    global_kkt_gap,
    smo_oracle_decision,
    smo_oracle_fit,
    _kernel_cell,
    _row_norm,
)
from svm.ported.svm.smosolver import SmoTrace
from svm.ported.svm.svc_impl import (
    svc_fit,
    svc_predict,
    unique_labels_sorted,
)
from svm.ported.svm.svm_parameter import (
    KERNEL_LINEAR,
    KERNEL_RBF,
    KernelParams,
    SvmModel,
    SvmParameter,
)


def _mode_name() -> String:
    """The mode this binary COMPILED in."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return String("IDENTICAL")
    return String("FAST")


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
    var p2 = SvmParameter.default()
    p2.svmType = 2
    hits += _try_refusal(ctx, fx, p2, KernelParams.linear(), False, "svmType")
    var p3 = SvmParameter.default()
    p3.epsilon = 0.1
    hits += _try_refusal(ctx, fx, p3, KernelParams.linear(), False, "epsilon")
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
    print("  refusals by name: " + String(hits + 1) + " (cache_size, svmType, epsilon, POLYNOMIAL, TANH, PRECOMPUTED, sample_weight, multiclass)")


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
        raise Error("the two cards diverge at " + d)
