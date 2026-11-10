# SPDX-License-Identifier: Apache-2.0
"""SVC probability contracts using actual native arithmetic, no Python oracle.

Preserve the original fifty-element permutation, eight calibration samples and
three-class coupling matrix. Expectations are independent permutation, monotone
orientation and probability-distribution properties, not copied algorithms.
"""
from core.host_parallel import host_parallelize

comptime SVC_PROBABILITY_FIXTURES = 3
comptime SVC_F64_GUARD = Float64(27182.5)
comptime SVC_I32_GUARD = Int32(-1234567)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


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
        for position in range(50):
            first_count += Int(ap[position] == index)
            other_count += Int(cp[position] == index)
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
