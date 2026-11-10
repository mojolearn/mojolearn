# SPDX-License-Identifier: Apache-2.0
"""Importable native CPU SVC calibration/shuffle/coupling contracts."""
from core.host_parallel import host_parallelize
from svm.host.svc_proba import host_splitmix_perm, host_platt_train
from svm.impl.svc_rows import U64P, sp_sigmoid_predict, multiclass_probability
from checks.native_tests.svc_probability_cases import SVC_PROBABILITY_FIXTURES, SvcProbabilityCalls, fixture, require


struct HostSvcProbability(SvcProbabilityCalls):
    @staticmethod
    def shuffle(dst: Int, count: Int, seed: Int) raises:
        host_splitmix_perm(dst, count, seed, 0)

    @staticmethod
    def platt(decisions: Int, labels: Int, count: Int, dst: Int) raises:
        host_platt_train(decisions, labels, count, dst)

    @staticmethod
    def sigmoid(calibration: Int, dst: Int) raises:
        var ab = U64P(unsafe_from_address=calibration)
        var p = U64P(unsafe_from_address=dst)
        def point(index: Int) {imm ab, imm p}:
            var decision = UInt64(0x4000000000000000) if index == 0 else UInt64(0xC000000000000000)
            p[index] = sp_sigmoid_predict(decision, ab[0], ab[1])
        host_parallelize(point, 2)

    @staticmethod
    def couple(matrix: Int, dst: Int, scratch: Int) raises -> Int:
        var q = U64P(unsafe_from_address=scratch)
        return multiclass_probability(3, U64P(unsafe_from_address=matrix),
            U64P(unsafe_from_address=dst), q, q + 9)


def run_checks() raises:
    var failures = List[Int](length=SVC_PROBABILITY_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            fixture[HostSvcProbability](fixture_id)
        except error:
            fp.unsafe_store(fixture_id, 1)
            print("FAIL svc_probability_cpu fixture", fixture_id, String(error))
    host_parallelize(task, SVC_PROBABILITY_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "svc_probability_cpu failed " + String(failed) + " fixtures")
    print("PASS svc_probability_cpu:", SVC_PROBABILITY_FIXTURES, "native probability contracts")
