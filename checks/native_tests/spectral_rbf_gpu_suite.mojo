# SPDX-License-Identifier: Apache-2.0
"""One RBF graph at a time on the physical GPU, with existing parallel kernels.

Both existing immediate and resident cell sequences run. The separately
recorded shared-context concurrent-submission failure remains open.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_decomp.descriptor_device import x_decomp_device_v1
from checks.native_tests.spectral_rbf_cases import SPECTRAL_RBF_NUMERICAL_FIXTURES, numerical, require

comptime SPECTRAL_RBF_GPU_FIXTURES = 2*SPECTRAL_RBF_NUMERICAL_FIXTURES


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failed = 0
    for unit in range(SPECTRAL_RBF_GPU_FIXTURES):
        try:
            numerical(x_decomp_device_v1,unit%2,unit>=2)
        except error:
            failed += 1
            print("FAIL spectral_rbf_gpu fixture",unit,String(error))
    require(failed == 0,"spectral_rbf_gpu failed " + String(failed) + " fixtures")
    print("PASS spectral_rbf_gpu:",SPECTRAL_RBF_GPU_FIXTURES,"supplementary ordered RBF formula graphs")
