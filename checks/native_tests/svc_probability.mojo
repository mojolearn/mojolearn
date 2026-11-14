# SPDX-License-Identifier: Apache-2.0
"""SVC probability contracts using actual native arithmetic, no Python oracle.

Preserve the original fifty-element permutation, eight calibration samples and
three-class coupling matrix. Expectations are independent permutation, monotone
orientation and probability-distribution properties, not copied algorithms.
"""
from core.host_parallel import host_parallelize
from svm.host.svc_proba import host_splitmix_perm, host_platt_train
from svm.impl.svc_rows import U64P, sp_sigmoid_predict, multiclass_probability, I32P
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from svm.estimator import _family_ctx
from svm.impl.svc_epilogue import device_splitmix_perm, device_platt_train
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime SVC_PROBABILITY_FIXTURES = 3
comptime SVC_F64_GUARD = Float64(27182.5)
comptime SVC_I32_GUARD = Int32(-1234567)


trait SvcProbabilityCalls:
    @staticmethod
    def shuffle(dst: Int, count: Int, seed: Int) raises: ...
    @staticmethod
    def platt(decisions: Int, labels: Int, count: Int, dst: Int) raises: ...
    @staticmethod
    def sigmoid(calibration: Int, dst: Int) raises: ...
    @staticmethod
    def couple(matrix: Int, dst: Int, scratch: Int) raises -> Int: ...


def same_words(actual: List[Float64], expected: List[Float64], label: String) raises:
    require(len(actual) == len(expected), label + " extent")
    var a = Pointer[UInt64, ImmUntrackedOrigin](unsafe_from_address=Int(actual.unsafe_ptr()))
    var b = Pointer[UInt64, ImmUntrackedOrigin](unsafe_from_address=Int(expected.unsafe_ptr()))
    var failures = List[Int](length=len(actual), fill=0)
    var fp = failures.unsafe_ptr()
    def cell(index: Int) {imm a, imm b, imm fp}:
        if a[index] != b[index]:
            fp.unsafe_store(index, 1)
    host_parallelize(cell, len(actual))
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, label + " was mutated")
    _ = actual
    _ = expected


def shuffle_fixture[A: SvcProbabilityCalls]() raises:
    var first = List[Int32](length=52, fill=SVC_I32_GUARD)
    var repeat = List[Int32](length=52, fill=SVC_I32_GUARD)
    var other = List[Int32](length=52, fill=SVC_I32_GUARD)
    A.shuffle(Int(first.unsafe_ptr()) + 4, 50, 3)
    A.shuffle(Int(repeat.unsafe_ptr()) + 4, 50, 3)
    A.shuffle(Int(other.unsafe_ptr()) + 4, 50, 4)
    var ap = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=Int(first.unsafe_ptr()) + 4)
    var bp = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=Int(repeat.unsafe_ptr()) + 4)
    var cp = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=Int(other.unsafe_ptr()) + 4)
    var failures = List[Int](length=50, fill=0)
    var differences = List[Int](length=50, fill=0)
    var fp = failures.unsafe_ptr()
    var dp = differences.unsafe_ptr()
    def index_contract(index: Int) {imm ap, imm bp, imm cp, imm fp, imm dp}:
        if ap[index] != bp[index]:
            fp.unsafe_store(index, 1)
        if ap[index] != cp[index]:
            dp.unsafe_store(index, 1)
        # Independent membership counts: each required index occurs once.
        # The inner fold is dependent; all fifty index checks run in parallel.
        var first_count = 0
        var other_count = 0
        var expected = Int32(index)
        for position in range(50):
            first_count += Int(ap[position] == expected)
            other_count += Int(cp[position] == expected)
        if first_count != 1 or other_count != 1:
            fp.unsafe_store(index, 1)
    host_parallelize(index_contract, 50)
    var failed = 0
    var different = 0
    for index in range(50):
        failed += failures[index]
        different += differences[index]
    require(failed == 0, "SVC shuffle lost permutation or same-seed contract")
    require(different > 0, "SVC shuffle did not distinguish seeds3 and4")
    require(first[0] == SVC_I32_GUARD and first[51] == SVC_I32_GUARD
        and repeat[0] == SVC_I32_GUARD and repeat[51] == SVC_I32_GUARD
        and other[0] == SVC_I32_GUARD and other[51] == SVC_I32_GUARD,
        "SVC shuffle output guards")


def platt_fixture[A: SvcProbabilityCalls]() raises:
    var decisions: List[Float64] = [SVC_F64_GUARD, -2, -1.5, -1, -0.2, 0.3, 1, 1.4, 2.2, SVC_F64_GUARD]
    var labels: List[Float64] = [SVC_F64_GUARD, -1, -1, -1, 1, -1, 1, 1, 1, SVC_F64_GUARD]
    var saved_decisions = decisions.copy()
    var saved_labels = labels.copy()
    var calibration = List[Float64](length=4, fill=SVC_F64_GUARD)
    A.platt(Int(decisions.unsafe_ptr()) + 8, Int(labels.unsafe_ptr()) + 8, 8,
        Int(calibration.unsafe_ptr()) + 8)
    require(calibration[1] < 0, "Platt calibration does not increase P(+1) with decision")
    require(calibration[0] == SVC_F64_GUARD and calibration[3] == SVC_F64_GUARD, "Platt output guards")
    var saved_calibration = calibration.copy()
    var probability = List[Float64](length=4, fill=SVC_F64_GUARD)
    A.sigmoid(Int(calibration.unsafe_ptr()) + 8, Int(probability.unsafe_ptr()) + 8)
    require(probability[1] > 0.5 and probability[2] < 0.5,
        "Platt positive/negative decision orientation")
    require(probability[1] <= 1 and probability[2] >= 0, "Platt sigmoid probability range")
    require(probability[0] == SVC_F64_GUARD and probability[3] == SVC_F64_GUARD, "sigmoid output guards")
    same_words(decisions, saved_decisions, "Platt decisions")
    same_words(labels, saved_labels, "Platt labels")
    same_words(calibration, saved_calibration, "sigmoid calibration")
    _ = probability


def coupling_fixture[A: SvcProbabilityCalls]() raises:
    var matrix: List[Float64] = [SVC_F64_GUARD,
        0, 0.7, 0.8, 0.3, 0, 0.6, 0.2, 0.4, 0, SVC_F64_GUARD]
    var saved_matrix = matrix.copy()
    var probabilities = List[Float64](length=5, fill=SVC_F64_GUARD)
    var scratch = List[Float64](length=14, fill=SVC_F64_GUARD)
    var status = A.couple(Int(matrix.unsafe_ptr()) + 8, Int(probabilities.unsafe_ptr()) + 8,
        Int(scratch.unsafe_ptr()) + 8)
    require(status == 0, "valid SVC coupling matrix was refused")
    require(abs(probabilities[1] + probabilities[2] + probabilities[3] - 1) < 1e-9,
        "SVC coupled probabilities do not sum to one")
    require(probabilities[1] > probabilities[2] and probabilities[2] > probabilities[3],
        "SVC coupled class ordering")
    require(probabilities[1] <= 1 and probabilities[3] >= 0, "SVC coupled probability range")
    require(probabilities[0] == SVC_F64_GUARD and probabilities[4] == SVC_F64_GUARD
        and scratch[0] == SVC_F64_GUARD and scratch[13] == SVC_F64_GUARD,
        "SVC coupling output/scratch guards")
    same_words(matrix, saved_matrix, "SVC coupling input")
    _ = probabilities
    _ = scratch


def fixture[A: SvcProbabilityCalls](fixture_id: Int) raises:
    if fixture_id == 0:
        shuffle_fixture[A]()
    elif fixture_id == 1:
        platt_fixture[A]()
    else:
        coupling_fixture[A]()


# ---- The host column (was svc_probability_cpu_suite.mojo) ----
# Importable native CPU SVC calibration/shuffle/coupling contracts.
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


def _run_cpu(mut report: Report) raises:
    var cases = report.cases(SVC_PROBABILITY_FIXTURES)
    def task(fixture_id: Int) {imm cases}:
        try:
            fixture[HostSvcProbability](fixture_id)
        except error:
            cases.fail(fixture_id, error)
    report.parallel(task, SVC_PROBABILITY_FIXTURES)


# ---- The device column (was svc_probability_gpu_suite.mojo) ----
# Importable native GPU SVC probability contracts.
#
# The native SVM entries share _family_ctx, so fixture submissions are ordered.
# Thin test kernels call the actual sigmoid/coupling routines, with no duplicated
# algorithm. A coupling row is an inherently dependent three-class solve; the
# original fixture has one row. Each sigmoid point is an independent GPU thread.
def _sigmoid_contract(ab: U64P, dst: U64P):
    var index = Int(block_idx.x) * 32 + Int(thread_idx.x)
    if index < 2:
        var decision = UInt64(0x4000000000000000) if index == 0 else UInt64(0xC000000000000000)
        dst[index + 1] = sp_sigmoid_predict(decision, ab[1], ab[2])


def _coupling_contract(matrix: U64P, dst: U64P, scratch: U64P, status: I32P):
    var row = Int(block_idx.x) * 32 + Int(thread_idx.x)
    if row == 0:
        status[1] = Int32(multiclass_probability(3, matrix + 1, dst + 1, scratch + 1, scratch + 10))


def _upload(ctx: DeviceContext, address: Int, count: Int) raises -> DeviceBuffer[DType.uint64]:
    var buffer = ctx.enqueue_create_buffer[DType.uint64](count)
    ctx.enqueue_copy(dst_buf=buffer, src_ptr=U64P(unsafe_from_address=address))
    return buffer^


struct DeviceSvcProbability(SvcProbabilityCalls):
    @staticmethod
    def shuffle(dst: Int, count: Int, seed: Int) raises:
        device_splitmix_perm(dst, count, seed, 0)

    @staticmethod
    def platt(decisions: Int, labels: Int, count: Int, dst: Int) raises:
        device_platt_train(decisions, labels, count, dst)

    @staticmethod
    def sigmoid(calibration: Int, dst: Int) raises:
        var ctx = _family_ctx()
        var ab = _upload(ctx, calibration - 8, 4)
        var probabilities = _upload(ctx, dst - 8, 4)
        ctx.enqueue_function[_sigmoid_contract](ab.unsafe_ptr(), probabilities.unsafe_ptr(),
            grid_dim=1, block_dim=32)
        ctx.enqueue_copy(dst_ptr=U64P(unsafe_from_address=dst - 8), src_buf=probabilities)
        ctx.enqueue_copy(dst_ptr=U64P(unsafe_from_address=calibration - 8), src_buf=ab)
        ctx.synchronize()
        _ = ab^
        _ = probabilities^
        _ = ctx^

    @staticmethod
    def couple(matrix: Int, dst: Int, scratch: Int) raises -> Int:
        var ctx = _family_ctx()
        var r = _upload(ctx, matrix - 8, 11)
        var p = _upload(ctx, dst - 8, 5)
        var q = _upload(ctx, scratch - 8, 14)
        var status: List[Int32] = [1234567, -1, 1234567]
        var st = ctx.enqueue_create_buffer[DType.int32](3)
        var sp = I32P(unsafe_from_address=Int(status.unsafe_ptr()))
        ctx.enqueue_copy(dst_buf=st, src_ptr=sp)
        ctx.enqueue_function[_coupling_contract](r.unsafe_ptr(), p.unsafe_ptr(), q.unsafe_ptr(), st.unsafe_ptr(),
            grid_dim=1, block_dim=32)
        ctx.enqueue_copy(dst_ptr=U64P(unsafe_from_address=matrix - 8), src_buf=r)
        ctx.enqueue_copy(dst_ptr=U64P(unsafe_from_address=dst - 8), src_buf=p)
        ctx.enqueue_copy(dst_ptr=U64P(unsafe_from_address=scratch - 8), src_buf=q)
        ctx.enqueue_copy(dst_ptr=sp, src_buf=st)
        ctx.synchronize()
        require(status[0] == 1234567 and status[2] == 1234567, "SVC coupling status guards")
        var result = Int(status[1])
        _ = r^
        _ = p^
        _ = q^
        _ = st^
        _ = ctx^
        return result


def _run_gpu(mut report: Report) raises:
    var cases = report.cases(SVC_PROBABILITY_FIXTURES)
    for fixture_id in range(SVC_PROBABILITY_FIXTURES):
        try:
            fixture[DeviceSvcProbability](fixture_id)
        except error:
            cases.fail(fixture_id, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("svc_probability", cpu=True, gpu=True, serial="", run=run_cases)
