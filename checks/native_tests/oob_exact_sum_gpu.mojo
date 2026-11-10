# SPDX-License-Identifier: Apache-2.0
"""Direct OOB GPU entry, without a Python pointer-list binding.

The product entry owns a process-wide xtrees.ops_device context. Submit
fixtures in order to respect its stream/allocation ownership; the existing
row/limb kernels provide parallel work inside each operation. This is not
concurrent-session or cross-vendor identity coverage.
"""
from checks.native_tests.oob_exact_sum_cases import OOB_EXACT_FIXTURES, require, run_oob_fixture


def main() raises:
    var failed = 0
    for fixture_id in range(OOB_EXACT_FIXTURES):
        try:
            run_oob_fixture[True](fixture_id)
        except error:
            print("FAIL oob_exact_sum_gpu fixture", fixture_id, String(error))
            failed += 1
    require(failed == 0, "oob_exact_sum_gpu failed " + String(failed) + " fixtures")
    print("PASS oob_exact_sum_gpu:", OOB_EXACT_FIXTURES, "native fixtures; shared-context submissions ordered")
