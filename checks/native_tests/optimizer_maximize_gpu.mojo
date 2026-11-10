# SPDX-License-Identifier: Apache-2.0
"""Native GPU maximize; ordered submission for shared neural context/pool.

The update kernels and independent native input cells remain parallel. This
retains automatic resident choice and proves no concurrent-session claim.
"""
from core.neural_context import neural_ctx
from training.optimizer_kernels_gpu import GpuOptKernels, _NEURAL_CTX
from training.optimizer_public import resident_close
from training.optimizer_public_common import PublicStep, PublicStepOut
from checks.native_tests.optimizer_maximize_cases import MAXIMIZE_FIXTURES, OptimizerTestKernels, require, run_maximize_fixture


struct GpuHarness(OptimizerTestKernels):
    @staticmethod
    def step(req: PublicStep) raises -> PublicStepOut:
        return GpuOptKernels.step(req)

    @staticmethod
    def download(handle: Int, m_addr: Int, v_addr: Int, n: Int) raises -> Int:
        return GpuOptKernels.download(handle, m_addr, v_addr, n)

    @staticmethod
    def close(handle: Int) raises -> Int:
        var ctx = neural_ctx[_NEURAL_CTX]()
        return resident_close(ctx, handle)


def main() raises:
    var failed = 0
    for fixture_id in range(MAXIMIZE_FIXTURES):
        try:
            run_maximize_fixture[GpuHarness](fixture_id)
        except error:
            print("FAIL optimizer_maximize_gpu fixture", fixture_id, String(error))
            failed += 1
    require(failed == 0, "optimizer_maximize_gpu failed " + String(failed) + " fixtures")
    print("PASS optimizer_maximize_gpu:", MAXIMIZE_FIXTURES, "signed-zero/ownership fixtures; shared-context ordered")
