# SPDX-License-Identifier: Apache-2.0
"""sparse_encode on the device DKit, in order: DKit enqueues on x_decomp's
one shared context (x_decomp/kit_device.mojo xd_ctx), so concurrent kits
would race that context's ownership; the 2 x 3 fixtures run one after
another and each kernel keeps its own parallelism."""
from checks.native_tests.core_verifier_gap_methods_cases import (
    VERIFIER_GAP_FIXTURES, require, encode_algo, encode_nnz, rows_mat, eye_mat, check_code,
)
from x_decomp.dictl_dev import sparse_encode_dev
from x_decomp.kit_device import DKit, DMat


def run_checks() raises:
    var failed = 0
    for unit in range(VERIFIER_GAP_FIXTURES):
        try:
            var k = DKit()
            var x = k.upload(rows_mat())
            var d = k.upload(eye_mat())
            var none = DMat(0, 0)
            var code = sparse_encode_dev(k, x, d, encode_algo(unit), 0.125, encode_nnz(unit), none, False, 100, False)
            var host = k.get(code)
            k.sync()
            check_code(unit, host)
        except error:
            failed += 1
            print("FAIL core_verifier_gap_methods_gpu fixture", unit, String(error))
    require(failed == 0, "core_verifier_gap_methods_gpu failed " + String(failed) + " of " + String(VERIFIER_GAP_FIXTURES))
    print("PASS core_verifier_gap_methods_gpu:", VERIFIER_GAP_FIXTURES, "sparse_encode fixtures, ordered on one device")
