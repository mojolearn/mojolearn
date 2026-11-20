# SPDX-License-Identifier: Apache-2.0
"""lu_factor reconstructs P A and repeats; lu_solve answers A x = b and
A^T x = b and repeats (ported from
python/mojolearn/tests/test_native_smallfam_public.py
`test_lu_factor_reconstructs_and_repeats` and
`test_lu_solve_answers_the_system`).

The original system (`_system()`: `np.random.default_rng(5)`, an 8 x 8
`uniform + 8 I`, a vector b and a 3-column B) is NumPy PCG64 output, stored
row-major as raw float32 words in `fixtures/metrics/lu-*` by
`tools/tsa_metrics_native_fixtures.py`. The factor is the x_decomp host
executor's `lu` (the XD_LU operation the CPU column's kit runs: partial
pivoting, LAPACK swap order) and the solve `lu_solve_run` (XD_LU_SOLVE,
trans 0 is A, 1 and 2 are A^T for real A); `trans` 2 is passed as the
public call passes it.

    LU_UNIT 0       L U == A[perm] (perm from the pivots' successive row
                    swaps) to rtol = atol = 1e-5 in Float64, and a second
                    factor is the same words
    LU_UNIT 1 .. 6  trans in (0, 1, 2) x right-hand side in (b, B): M x ==
                    rhs to rtol = atol = 1e-4 in Float64 (M = A or A^T), and
                    a repeat solve is the same words

The dtype and shape of the public outputs, the plain-tuple host route, the
refusals and the singular warning stay in the Python file (public
boundary); so do the score and fit_transform compositions, which compare
public methods with each other. Importable; no `main()`.
"""
from std.memory import bitcast
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.host import HostExec as XdHost
from x_decomp.api import lu_solve_run
from checks.native_tests.tsa_metrics_fixture_words import METRICS_FIXTURE_DIR, fixture_f32
from core.host_parallel import host_parallelize
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime LU_N = 8
comptime LU_COLS = 3
comptime LU_UNITS = 7


def system_a() raises -> List[Float32]:
    return fixture_f32(METRICS_FIXTURE_DIR + "lu-a.f32", LU_N * LU_N)


def factor(a: List[Float32]) raises -> Tuple[List[Float32], List[Int32]]:
    var lu = a.copy()
    var piv = List[Int32](length=LU_N, fill=Int32(-1))
    var info = List[Float32](length=1, fill=Float32(9))
    XdHost.lu(F32Ptr(unsafe_from_address=Int(lu.unsafe_ptr())), I32Ptr(unsafe_from_address=Int(piv.unsafe_ptr())),
              F32Ptr(unsafe_from_address=Int(info.unsafe_ptr())), LU_N)
    require(info[0] == 0.0, "lu_factor reported a singular factor of a diagonally dominant system")
    return (lu^, piv^)


def same_words(a: List[Float32], b: List[Float32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):  # small-loop(one small factor or solution)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            return False
    return True


def factor_unit() raises:
    var a = system_a()
    var first = factor(a)
    var lu = first[0].copy()
    var piv = first[1].copy()
    var perm = List[Int](length=LU_N, fill=0)
    for i in range(LU_N):  # small-loop(eight rows)
        perm[i] = i
    for i in range(LU_N):  # Dependent successive row swaps (the pivot order).
        var p = Int(piv[i])
        require(p >= i and p < LU_N, "lu_factor pivot out of range")
        var held = perm[i]
        perm[i] = perm[p]
        perm[p] = held
    for i in range(LU_N):  # small-loop(8 x 8 reconstruction)
        for j in range(LU_N):
            var total = Float64(0)
            for k in range(min(i, j) + 1):
                var l = Float64(1) if k == i else Float64(lu[i * LU_N + k])
                total += l * Float64(lu[k * LU_N + j])
            var want = Float64(a[perm[i] * LU_N + j])
            require(abs(total - want) <= 1e-5 + 1e-5 * abs(want),
                    "L U differs from A[perm] at (" + String(i) + ", " + String(j) + ")")
    var second = factor(a)
    require(same_words(second[0], lu), "a second lu_factor moved a factor word")
    for i in range(LU_N):  # small-loop(eight pivots)
        require(second[1][i] == piv[i], "a second lu_factor moved a pivot")


def solve(lu: List[Float32], piv: List[Int32], rhs: List[Float32], nrhs: Int, trans: Int) raises -> List[Float32]:
    var x = rhs.copy()
    lu_solve_run[XdHost](F32Ptr(unsafe_from_address=Int(lu.unsafe_ptr())),
                         I32Ptr(unsafe_from_address=Int(piv.unsafe_ptr())),
                         F32Ptr(unsafe_from_address=Int(x.unsafe_ptr())), LU_N, nrhs, trans)
    return x^


def solve_unit(unit: Int) raises:
    var trans = (unit - 1) // 2
    var nrhs = 1 if (unit - 1) % 2 == 0 else LU_COLS
    var rhs = fixture_f32(METRICS_FIXTURE_DIR + ("lu-b.f32" if nrhs == 1 else "lu-bm.f32"), LU_N * nrhs)
    var a = system_a()
    var pair = factor(a)
    var x = solve(pair[0], pair[1], rhs, nrhs, trans)
    for i in range(LU_N):  # small-loop(8 x nrhs residual)
        for c in range(nrhs):
            var total = Float64(0)
            for k in range(LU_N):
                var m = a[i * LU_N + k] if trans == 0 else a[k * LU_N + i]
                total += Float64(m) * Float64(x[k * nrhs + c])
            var want = Float64(rhs[i * nrhs + c])
            require(abs(total - want) <= 1e-4 + 1e-4 * abs(want),
                    "lu_solve trans " + String(trans) + " does not answer the system at (" + String(i) + ", "
                    + String(c) + ")")
    var again = solve(pair[0], pair[1], rhs, nrhs, trans)
    require(same_words(again, x), "a repeat lu_solve moved a word, trans " + String(trans))


def lu_unit(unit: Int) raises:
    if unit == 0:
        factor_unit()
    else:
        solve_unit(unit)


# ---- The host column (was metrics_smallfam_public_cpu_suite.mojo) ----
# lu_factor / lu_solve on the x_decomp host executor.
#
# The seven units (the factor, then six trans x right-hand-side solves, each
# on its own factor) are independent and run across host cores. The device
# factor and solve are gated by the decomposition kit's own device checks;
# the original ran wherever the public call ran, and its host column is
# this executor.
def _run_cpu(mut report: Report) raises:
    var cases = report.cases(LU_UNITS)
    def task(unit: Int) {imm cases}:
        try:
            lu_unit(unit)
        except error:
            cases.fail(unit, error)
    report.parallel(task, LU_UNITS)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("metrics_smallfam_public", cpu=True, gpu=False, serial="", run=run_cases)
