# SPDX-License-Identifier: Apache-2.0
"""Native Logistic fitted-state PACK contracts, independent of QN arithmetic."""
from core.host_parallel import host_parallelize
from core.logistic_model_state import (
    LogisticFitLayout, logistic_target_count, logistic_fit_target_count,
    logistic_publish_coefficients,
)

comptime LOGISTIC_LAYOUT_FIXTURES = 22
comptime LAYOUT_GUARD = UInt32(0xCB123456)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def coefficient_word(index: Int) -> UInt32:
    # Deliberately include signed zeros and distinct quiet/signaling NaN
    # payloads: PACK must copy words without converting or canonicalizing.
    var special: InlineArray[UInt32, 4] = [0x00000000, 0x80000000, 0x7FC00042, 0x7F800123]
    if index % 17 < 4:
        return special[index % 17]
    return UInt32(0x3F000000) + UInt32((index * 97) % 0x007FFFFF)


def bias_word(target: Int) -> UInt32:
    return UInt32(0xBF000000) + UInt32(target * 19)


def packed_fixture(fixture_id: Int) raises:
    var shapes: List[Int] = [0, 1, 0, 3, 1, 1, 1, 1, 4, 3, 4, 3, 257, 1025, 257, 1025]
    var features = shapes[fixture_id * 2]
    var targets = shapes[fixture_id * 2 + 1]
    var intercept = fixture_id % 2 == 1
    var layout = LogisticFitLayout(features, targets, intercept)
    require(layout.parameters == features + Int(intercept) and layout.weights == (features + Int(intercept)) * targets
        and layout.coefficients == features * targets, "coefficient layout extents")
    var source = List[UInt32](length=layout.weights + 2, fill=LAYOUT_GUARD)
    var coefficients = List[UInt32](length=layout.coefficients + 2, fill=LAYOUT_GUARD)
    var biases = List[UInt32](length=targets + 2, fill=LAYOUT_GUARD)
    var wp = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(source.unsafe_ptr()) + 4)
    def input_word(index: Int) {imm wp, imm features, imm targets}:
        if index < features * targets:
            var feature = index // targets
            var target = index - feature * targets
            wp[index] = coefficient_word(target * features + feature)
        else:
            wp[index] = bias_word(index - features * targets)
    host_parallelize(input_word, layout.weights)
    var saved_source = source.copy()
    var source_address = Int(source.unsafe_ptr()) + 4 if layout.weights else 0
    var coefficient_address = Int(coefficients.unsafe_ptr()) + 4 if layout.coefficients else 0
    logistic_publish_coefficients(layout, source_address, coefficient_address, Int(biases.unsafe_ptr()) + 4)
    var cp = Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=Int(coefficients.unsafe_ptr()) + 4)
    var bp = Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=Int(biases.unsafe_ptr()) + 4)
    var actual_source = Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=Int(source.unsafe_ptr()))
    var expected_source = Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=Int(saved_source.unsafe_ptr()))
    var count = max(layout.coefficients + targets, layout.weights + 2)
    var failures = List[Int](length=count, fill=0)
    var fp = failures.unsafe_ptr()
    var coefficient_count = layout.coefficients
    var weight_count = layout.weights
    def output_word(index: Int) {imm cp, imm bp, imm actual_source, imm expected_source,
            imm coefficient_count, imm weight_count, imm targets, imm intercept, imm fp}:
        if index < coefficient_count:
            if cp[index] != coefficient_word(index):
                fp.unsafe_store(index, 1)
        elif index < coefficient_count + targets:
            var target = index - coefficient_count
            if bp[target] != (bias_word(target) if intercept else UInt32(0)):
                fp.unsafe_store(index, 1)
        if index < weight_count + 2 and actual_source[index] != expected_source[index]:
            fp.unsafe_store(index, 1)
    host_parallelize(output_word, count)
    # Raw pointers do not retain their owner. Keep the original source alive
    # through its immutability comparison, then deliberately release it below.
    _ = source
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "native coefficient/intercept layout or raw words differ")
    require(coefficients[0] == LAYOUT_GUARD and coefficients[len(coefficients) - 1] == LAYOUT_GUARD
        and biases[0] == LAYOUT_GUARD and biases[len(biases) - 1] == LAYOUT_GUARD, "PACK output guards")
    # Destroy the source owner after synchronous publication. Packed values
    # must remain in the independent caller-owned destination buffers.
    source = List[UInt32]()
    def retained_word(index: Int) {imm cp, imm bp, imm coefficient_count, imm intercept, imm fp}:
        var expected = coefficient_word(index) if index < coefficient_count else (
            bias_word(index - coefficient_count) if intercept else UInt32(0))
        var actual = cp[index] if index < coefficient_count else bp[index - coefficient_count]
        if actual != expected:
            fp.unsafe_store(index, 1)
    host_parallelize(retained_word, coefficient_count + targets)
    failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "PACK outputs depended on released source owner")
    _ = coefficients
    _ = biases
    _ = saved_source


def refusal_fixture(fixture_id: Int) raises:
    var weights: List[UInt32] = [LAYOUT_GUARD, 1, 2, 3, 4, 5, 6, LAYOUT_GUARD]
    var coefficients = List[UInt32](length=6, fill=LAYOUT_GUARD)
    var biases = List[UInt32](length=4, fill=LAYOUT_GUARD)
    var w = Int(weights.unsafe_ptr()) + 4
    var c = Int(coefficients.unsafe_ptr()) + 4
    var b = Int(biases.unsafe_ptr()) + 4
    if fixture_id == 0:
        w = 0
    elif fixture_id == 1:
        w += 1
    elif fixture_id == 2:
        c = 0
    elif fixture_id == 3:
        c += 1
    elif fixture_id == 4:
        b = 0
    elif fixture_id == 5:
        b += 1
    elif fixture_id == 6:
        w = 9223372036854775804
    else:
        b = 9223372036854775804
    var refused = False
    try:
        logistic_publish_coefficients(LogisticFitLayout(2, 2, True), w, c, b)
    except error:
        refused = String(error) == "ValueError: LogisticRegression coefficient storage must be an aligned float32 span"
    require(refused, "invalid coefficient storage was not refused by the native span check")
    for value in coefficients:  # Bounded six-word refusal sentinel, no data processing.
        require(value == LAYOUT_GUARD, "refused PACK changed coefficients")
    for value in biases:
        require(value == LAYOUT_GUARD, "refused PACK changed intercepts")
    require(weights[1] == 1 and weights[6] == 6 and weights[0] == LAYOUT_GUARD and weights[7] == LAYOUT_GUARD,
        "refused PACK changed source")


def metadata_fixture(fixture_id: Int) raises:
    if fixture_id == 0:
        require(logistic_target_count(2) == 1 and logistic_fit_target_count(2) == 1
            and logistic_fit_target_count(3) == 3 and logistic_target_count(17) == 17, "native class target count")
        return
    if fixture_id == 1:
        var refused = False
        try:
            _ = logistic_fit_target_count(1)
        except error:
            refused = String(error) == "mojolearn LogisticRegression: y has one class"
        require(refused, "one-class fit target was accepted")
        return
    var features = -1 if fixture_id == 2 else (2 if fixture_id == 3 else 2305843009213693951)
    var targets = 0 if fixture_id == 3 else (2 if fixture_id == 5 else 1)
    var refused = False
    try:
        _ = LogisticFitLayout(features, targets, fixture_id == 4)
    except error:
        refused = String(error) == "ValueError: LogisticRegression coefficient layout exceeds native storage"
    require(refused, "invalid/overflow coefficient layout was accepted")


def run_checks() raises:
    var failures = List[Int](length=LOGISTIC_LAYOUT_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            if fixture_id < 8:
                packed_fixture(fixture_id)
            elif fixture_id < 16:
                refusal_fixture(fixture_id - 8)
            else:
                metadata_fixture(fixture_id - 16)
        except error:
            fp.unsafe_store(fixture_id, 1)
            print("FAIL logistic_model_state fixture", fixture_id, String(error))
    host_parallelize(task, LOGISTIC_LAYOUT_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "logistic_model_state failed " + String(failed) + " fixtures")
    print("PASS logistic_model_state:", LOGISTIC_LAYOUT_FIXTURES, "native coefficient PACK fixtures")
