# SPDX-License-Identifier: Apache-2.0
"""The pinned standard-normal helpers the product runs natively
(bindings/normal_dist_helpers.mojo: erfc64, normal_cdf64, normal_inv_cdf64),
held to the references their deleted Python twins in `_portable_math` were
checked against in tests/test_py_bugs.py (lane pyout-runtime-boundary,
2026-11-11): erfc within 2 ulps of mpmath at 200 bits, the infinities exact,
and the quantile the inverse of the CDF. Pure scalar functions: the cases
are independent and run across the native host worker pool.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from bindings.normal_dist_helpers import erfc64, normal_cdf64, normal_inv_cdf64

comptime PORTABLE_MATH_CASES = 3


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _ulps(a: Float64, b: Float64) -> Int:
    var ia = Int(bitcast[DType.int64](a))
    var ib = Int(bitcast[DType.int64](b))
    return abs(ia - ib)


def erfc_matches_mpmath() raises:
    # mpmath.erfc at 200 bits, rounded to binary64 (the former Python table).
    var xs: List[Float64] = [-5.5, -3.0, -1.1, -0.5, -0.001, 0.0, 1e-20, 0.2, 0.3, 0.6, 0.9, 1.0,
                             1.2, 1.5, 2.0, 2.8, 2.9, 3.5, 5.0, 5.9, 6.5, 10.0, 20.0, 27.5]
    var refs: List[Float64] = [
        1.9999999999999927, 1.9999779095030015, 1.8802050695740817, 1.5204998778130465,
        1.0011283787909693, 1.0, 1.0, 0.7772974107895215, 0.6713732405408726, 0.3961439091520741,
        0.20309178757716786, 0.15729920705028513, 0.08968602177036464, 0.033894853524689274,
        0.004677734981047266, 7.501319466545911e-05, 4.109787809945886e-05, 7.430983723414128e-07,
        1.537459794428035e-12, 7.190409783550478e-17, 3.8421483271206475e-20, 2.088487583762545e-45,
        5.395865611607901e-176, 0.0,
    ]
    for i in range(len(xs)):  # driver-loop: over the fixed reference table
        require(_ulps(erfc64(xs[i]), refs[i]) <= 2, "erfc64 off by more than 2 ulps at " + String(xs[i]))
    var inf = Float64.MAX * 2.0
    require(erfc64(inf) == 0.0, "erfc64(+inf) must be 0")
    require(erfc64(-inf) == 2.0, "erfc64(-inf) must be 2")


def normal_cdf_is_half_erfc() raises:
    require(normal_cdf64(0.0) == 0.5, "normal_cdf64(0) must be 0.5")
    var xs: List[Float64] = [-3.0, -1.0, 0.5, 2.0]
    for i in range(len(xs)):  # driver-loop: over the fixed reference points
        var s = normal_cdf64(xs[i]) + normal_cdf64(-xs[i])
        require(abs(s - 1.0) <= 4.0e-16, "normal_cdf64 is not symmetric at " + String(xs[i]))


def normal_inv_cdf_inverts_the_cdf() raises:
    require(normal_inv_cdf64(0.5) == 0.0, "normal_inv_cdf64(0.5) must be 0")
    require(abs(normal_inv_cdf64(0.975) - 1.959963984540054) <= 4.0e-15,
            "normal_inv_cdf64(0.975) is not 1.959963984540054")
    var ps: List[Float64] = [1e-10, 0.001, 0.02, 0.3, 0.7, 0.98, 0.999]
    for i in range(len(ps)):  # driver-loop: over the fixed reference points (both AS241 regions)
        var p = ps[i]
        var back = normal_cdf64(normal_inv_cdf64(p))
        require(abs(back - p) <= 1.0e-13 * p + 1.0e-300, "normal_inv_cdf64 does not invert the CDF at " + String(p))


def _case(index: Int) raises:
    if index == 0:
        erfc_matches_mpmath()
    elif index == 1:
        normal_cdf_is_half_erfc()
    else:
        normal_inv_cdf_inverts_the_cdf()


def run_checks() raises:
    var failures = List[Int](length=PORTABLE_MATH_CASES, fill=0)
    var fp = failures.unsafe_ptr()

    def task(unit: Int) {imm fp}:
        try:
            _case(unit)
        except problem:
            fp.unsafe_store(unit, 1)
            print("FAIL portable_math case", unit, String(problem))

    host_parallelize(task, PORTABLE_MATH_CASES)
    var failed = 0
    for i in range(PORTABLE_MATH_CASES):  # driver-loop: over the case results
        failed += failures[i]
    _ = failures^
    if failed:
        raise Error("portable math: " + String(failed) + " case(s) failed")
