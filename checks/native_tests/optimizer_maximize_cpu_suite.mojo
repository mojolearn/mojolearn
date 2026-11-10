# SPDX-License-Identifier: Apache-2.0
"""Parallel native maximize fixtures; original automatic state ownership."""
from core.host_parallel import host_parallelize
from training.host.optimizer_kernels import HostOptKernels
from training.host.optimizer_public_host import host_close
from training.optimizer_public_common import PublicStep, PublicStepOut
from checks.native_tests.optimizer_maximize_cases import MAXIMIZE_FIXTURES, OptimizerTestKernels, require, run_maximize_fixture


struct HostHarness(OptimizerTestKernels):
    @staticmethod
    def step(req: PublicStep) raises -> PublicStepOut:
        return HostOptKernels.step(req)

    @staticmethod
    def download(handle: Int, m_addr: Int, v_addr: Int, n: Int) raises -> Int:
        return HostOptKernels.download(handle, m_addr, v_addr, n)

    @staticmethod
    def close(handle: Int) raises -> Int:
        return host_close(handle)


def run_checks() raises:
    var failures = List[Int](length=MAXIMIZE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            run_maximize_fixture[HostHarness](fixture_id)
        except error:
            print("FAIL optimizer_maximize_cpu fixture", fixture_id, String(error))
            fp.unsafe_store(fixture_id, 1)
    host_parallelize(task, MAXIMIZE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "optimizer_maximize_cpu failed " + String(failed) + " fixtures")
    print("PASS optimizer_maximize_cpu:", MAXIMIZE_FIXTURES, "parallel signed-zero/ownership fixtures")
