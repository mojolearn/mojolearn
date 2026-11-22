# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""sparse_encode on an orthonormal dictionary, moved from
test_public_verifier_gap_methods.py (2026-11-14), through x_decomp's typed
encode driver (x_decomp/dictl.mojo sparse_encode on the host Kit,
x_decomp/dictl_dev.mojo sparse_encode_dev on the device DKit) with the
public door's words (bindings/dict_learning_native_unpack.mojo
_encode_words: alpha 0.125 and max_iter 100 for the lasso routes, no init,
positive off).

rows = [[2, -0.75, 0.0625], [-0.5, 1, -3]], dictionary = I(3):
  0..2  threshold, lasso_cd, lasso_lars equal the soft threshold
        sign(x) max(|x| - 0.125, 0) within atol 1e-6 (rtol 0)
  3     omp with one nonzero keeps the largest atom: [[2, 0, 0], [0, 0, -3]]
        exactly
The Theta batch forecasts, the residual dropout mask and the refusals stay
in the Python test.
"""
from std.sys.compile import is_defined
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, CPU_COLUMN_BUILD, require
from x_decomp.dictl import ENC_LASSO_CD, ENC_LASSO_LARS, ENC_OMP, ENC_THRESHOLD, sparse_encode
from x_decomp.kit import Mat, Kit
from x_decomp.host import HostExec
from x_decomp.dictl_dev import sparse_encode_dev
from x_decomp.kit_device import DKit, DMat


# The column this binary was built for (`-D MOJOLEARN_COLUMN_CPU=1` builds the host column).


comptime VERIFIER_GAP_FIXTURES = 4


def encode_algo(unit: Int) -> Int:
    var algos: List[Int] = [ENC_THRESHOLD, ENC_LASSO_CD, ENC_LASSO_LARS, ENC_OMP]
    return algos[unit]


def encode_nnz(unit: Int) -> Int:
    return 1 if unit == 3 else 0


def rows_mat() -> Mat:
    var x = Mat(2, 3)
    var v: List[Float32] = [2.0, -0.75, 0.0625, -0.5, 1.0, -3.0]
    for i in range(6):
        x.d[i] = v[i]
    return x^


def eye_mat() -> Mat:
    var d = Mat(3, 3)
    for i in range(3):
        d.d[4 * i] = 1.0
    return d^


def check_code(unit: Int, code: Mat) raises:
    var tag = "sparse_encode fixture " + String(unit)
    require(code.r == 2 and code.c == 3, tag + ": code shape")
    var x = rows_mat()
    if unit == 3:
        var want: List[Float32] = [2.0, 0.0, 0.0, 0.0, 0.0, -3.0]
        for i in range(6):
            require(code.d[i] == want[i], tag + ": omp keeps the largest atom at " + String(i))
        return
    for i in range(6):
        var v = x.d[i]
        var mag = max(abs(v) - Float32(0.125), Float32(0))
        var want = mag if v > 0 else (-mag if v < 0 else Float32(0))
        require(abs(Float64(code.d[i]) - Float64(want)) <= 1e-6,
                tag + ": soft threshold at " + String(i) + " got " + String(code.d[i]) + ", expected " + String(want))


# ---- The host column (was core_verifier_gap_methods_cpu_suite.mojo) ----
# sparse_encode on the host Kit; the four independent fixtures run across
# host cores, each with its own kit and failure slot.
def check_unit(unit: Int) raises:
    var k = Kit[HostExec]()
    var code = sparse_encode(k, rows_mat(), eye_mat(), encode_algo(unit), 0.125, encode_nnz(unit), Mat(0, 0),
                             False, 100, False)
    check_code(unit, code)


def test_sparse_encode() raises:
    comptime if CPU_COLUMN_BUILD:
        check_cases("sparse_encode", VERIFIER_GAP_FIXTURES, check_unit)
    else:
        # DKit enqueues on x_decomp's one shared context (x_decomp/kit_device.mojo xd_ctx), so concurrent kits would
        # race that context's ownership; the fixtures run one after another, each kernel keeps its parallelism.
        for unit in range(VERIFIER_GAP_FIXTURES):  # driver-loop: ordered on x_decomp's shared context
            try:
                var k = DKit()
                var x = k.upload(rows_mat())
                var d = k.upload(eye_mat())
                var none = DMat(0, 0)
                var code = sparse_encode_dev(k, x, d, encode_algo(unit), 0.125, encode_nnz(unit), none, False, 100, False)
                var host = k.get(code)
                k.sync()
                check_code(unit, host)
            except problem:
                raise Error("sparse_encode row " + String(unit) + ": " + String(problem))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
