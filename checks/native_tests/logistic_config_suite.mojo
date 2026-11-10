# SPDX-License-Identifier: Apache-2.0
"""Native Logistic scalar contracts: analytic words plus independent NumPy plans.

The source reference data evaluates original NumPy expressions, never the Mojo
helper. No product kernels, Python calls or hardware identity checks occur here.
"""
from std.memory import bitcast
from std.math import isnan
from core.host_parallel import host_parallelize
from core.logistic_config import (
    LogisticScalar, LogisticPromotion, logistic_scalar_operation,
    logistic_strength_plan, logistic_strengths, logistic_change_tolerance,
    logistic_penalty, logistic_positive, logistic_ratio_valid,
)
from core.sequence_array_cast import SEQ_CAST_OVERFLOW, SEQ_CAST_UNDERFLOW, SEQ_CAST_INVALID
from checks.native_tests.logistic_config_reference import (
    plan_word, plan_width, plan_kind, tolerance_input_word, tolerance_word,
)

comptime LOGISTIC_CONFIG_FIXTURES = 54


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def real(word: UInt64) -> Float64:
    return bitcast[DType.float64](word)


def two(exponent: Int) -> Float64:
    return real(UInt64(exponent + 1023) << 52) if exponent >= -1022 else real(UInt64(1) << UInt64(exponent + 1074))


def scalar_fixture(fixture_id: Int) raises:
    var left = LogisticScalar(1.0, 0, 8)
    var right = LogisticScalar(3.0, 1, 2)
    var operation = 2
    var weak = False
    var expected = real(0x3FD5555555555555)
    var width = 8
    var kind = 1
    var flags = 0
    var cast_flags = 0
    var message = False
    var nan = False
    if fixture_id == 1 or fixture_id == 3:
        weak = True
        width = 2 if fixture_id == 1 else 4
        right.width = width
        expected = real(UInt64(0x3FD5540000000000 if width == 2 else 0x3FD5555560000000))
        message = True
    elif fixture_id == 2:
        right.width = 4
    elif 4 <= fixture_id <= 9:
        left = LogisticScalar(1.0, 1, 2)
        if fixture_id <= 6:
            right = LogisticScalar(3.0, 2, 1 << (fixture_id - 4))
            width = 2 if fixture_id == 4 else (4 if fixture_id == 5 else 8)
            message = fixture_id == 4
        elif fixture_id == 7:
            left.width = 4
            width = 4
            message = True
        else:
            left = LogisticScalar(1.0, 4 if fixture_id == 8 else 3, 8 if fixture_id == 8 else 1)
            width = 2
            message = True
        expected = real(UInt64(0x3FD5540000000000 if width == 2 else (0x3FD5555560000000 if width == 4 else 0x3FD5555555555555)))
    elif fixture_id == 10:
        right = LogisticScalar(3.0, 0, 8)
        kind = 0
    elif fixture_id == 11 or fixture_id == 12:
        weak = True
        operation = 0
        width = 2
        message = True
        if fixture_id == 11:
            left.value = 70000
            right.value = 0
            nan = True
            flags = SEQ_CAST_INVALID
            cast_flags = SEQ_CAST_OVERFLOW
        else:
            left.value = two(-100)
            right.value = 1
            expected = 0
    elif 13 <= fixture_id <= 19:
        operation = 0
        width = 2 if fixture_id <= 15 else (4 if fixture_id <= 17 else 8)
        message = True
        right = LogisticScalar(0.5, 1, width)
        var normal = -14 if width == 2 else (-126 if width == 4 else -1022)
        var last = -24 if width == 2 else (-149 if width == 4 else -1074)
        var exact = fixture_id == 13 or fixture_id == 16 or fixture_id == 18
        left = LogisticScalar(two(normal if exact else last), 1, width)
        expected = two(normal - 1) if exact else Float64(0)
        if not exact:
            flags = SEQ_CAST_UNDERFLOW
        if fixture_id == 15:
            left.value = 3 * two(-24)
            expected = two(-23)
    elif 20 <= fixture_id <= 23:
        width = 8 if fixture_id < 22 else 2
        message = True
        left = LogisticScalar(1.0, 1, width)
        right = LogisticScalar(two(1023) if fixture_id == 20 else (3 * two(1022) if fixture_id == 21 else (32768.0 if fixture_id == 22 else 49152.0)), 1, width)
        expected = two(-1023) if fixture_id == 20 else (real(0x0005555555555555) if fixture_id == 21 else (two(-15) if fixture_id == 22 else 341 * two(-24)))
        if fixture_id == 21 or fixture_id == 23:
            flags = SEQ_CAST_UNDERFLOW
    elif fixture_id == 24 or fixture_id == 25:
        operation = 0
        width = 2 if fixture_id == 24 else 4
        message = True
        left = LogisticScalar(65504.0 if fixture_id == 24 else Float64(bitcast[DType.float32](UInt32(0x7F7FFFFF))), 1, width)
        right = LogisticScalar(2.0, 1, width)
        expected = real(0x7FF0000000000000)
        flags = SEQ_CAST_OVERFLOW
    elif fixture_id == 26 or fixture_id == 27:
        operation = 0
        left = LogisticScalar(real(UInt64(0x7FF0000000000042 if fixture_id == 26 else 0x7FF8000000000042)), 1, 8, fixture_id == 26)
        right = LogisticScalar(1.0, 1, 8)
        message = True
        nan = True
        flags = SEQ_CAST_INVALID if fixture_id == 26 else 0
    elif fixture_id == 28 or fixture_id == 29:
        left = LogisticScalar(real(0x7FF0000000000000), 1, 8)
        right = LogisticScalar(real(0x7FF0000000000000) if fixture_id == 28 else Float64(0), 1, 8)
        operation = 1 if fixture_id == 28 else 0
        message = True
        nan = True
        flags = SEQ_CAST_INVALID
    elif fixture_id == 30:
        operation = 0
        left = LogisticScalar(real(0x7FEFFFFFFFFFFFFF), 0, 8)
        right = LogisticScalar(2.0, 0, 8)
        expected = real(0x7FF0000000000000)
        kind = 0
    elif fixture_id == 31:
        left = LogisticScalar(1.0, 1, 2)
        right = LogisticScalar(two(-24), 1, 2)
        width = 2
        message = True
        expected = real(0x7FF0000000000000)
        flags = SEQ_CAST_OVERFLOW
    elif fixture_id >= 32:
        left = LogisticScalar(1.0, 1, 4)
        right = LogisticScalar(3.0, 2, 2 if fixture_id == 32 else 4)
        width = 4 if fixture_id == 32 else 8
        expected = real(UInt64(0x3FD5555560000000 if fixture_id == 32 else 0x3FD5555555555555))
        message = fixture_id == 32
    var result = logistic_scalar_operation(left, right, operation, weak)
    require(result.scalar.width == width and result.scalar.kind == kind and not result.scalar.signaling,
        "Logistic scalar result promotion")
    require(result.status == flags and result.cast_status == cast_flags and result.operation == operation
        and result.scalar_message == message, "Logistic scalar arithmetic/error event")
    if nan:
        require(isnan(result.scalar.value), "Logistic invalid arithmetic did not return NaN")
    else:
        require(bitcast[DType.uint64](result.scalar.value) == bitcast[DType.uint64](expected),
            "Logistic scalar rounded value differs")


def strength_fixture(fixture_id: Int) raises:
    var penalty = 3
    var c = LogisticScalar(8.0, 0, 8)
    var ratio = LogisticScalar(0.25, 0, 8)
    var weak = False
    if fixture_id == 0:
        penalty = 0
        c = LogisticScalar(0.0, 1, 2)
        ratio.value = 0
        weak = True
    elif fixture_id == 1:
        penalty = 1
        ratio.value = 0
    elif fixture_id == 2:
        penalty = 2
        c = LogisticScalar(4.0, 1, 2)
        ratio.value = 0
        weak = True
    elif 4 <= fixture_id <= 7:
        var width = 2 if fixture_id == 4 or fixture_id == 6 else 4
        c = LogisticScalar(3.0, 1, width)
        ratio = LogisticScalar(0.300048828125 if width == 2 else Float64(bitcast[DType.float32](UInt32(0x3E99999A))), 1, width)
        weak = fixture_id < 6
    elif fixture_id >= 8:
        c = LogisticScalar(3.0, 1, 2)
        ratio = LogisticScalar(1.0, 2 if fixture_id == 8 else 4, 2 if fixture_id == 8 else 8)
        weak = True
    var plan = logistic_strength_plan(penalty, c, ratio, weak)
    var pair = logistic_strengths(penalty, c, ratio, weak)
    require(bitcast[DType.uint64](plan.l1) == plan_word(fixture_id, 0)
        and bitcast[DType.uint64](plan.l2) == plan_word(fixture_id, 1)
        and bitcast[DType.uint64](pair[0]) == plan_word(fixture_id, 0)
        and bitcast[DType.uint64](pair[1]) == plan_word(fixture_id, 1), "independent NumPy strength values differ")
    for index in range(4):  # Four ordered configuration expressions, not a data loop.
        var event = plan.event(index)
        var operation = 0 if fixture_id == 0 or (fixture_id < 3 and index > 0) else (2 if index == 0 else (1 if index == 2 else 0))
        var message = False
        if fixture_id == 2:
            message = index == 0
        elif fixture_id == 4 or fixture_id == 5:
            message = True
        elif fixture_id == 6 or fixture_id == 7:
            message = index == 1 or index == 3
        elif fixture_id == 8:
            message = index == 0 or index == 3
        elif fixture_id == 9:
            message = index != 2
        require(bitcast[DType.uint64](event.scalar.value) == plan_word(fixture_id, index + 2)
            and event.scalar.width == plan_width(fixture_id, index) and event.scalar.kind == plan_kind(fixture_id, index),
            "Logistic intermediate strength rounding/promotion differs")
        require(event.status == 0 and event.cast_status == 0 and event.operation == operation
            and event.scalar_message == message, "Logistic ordered strength event differs")


def tolerance_fixture(fixture_id: Int) raises:
    var width = 2 if fixture_id < 2 else (4 if fixture_id < 4 else 8)
    var result = logistic_change_tolerance(LogisticScalar(real(tolerance_input_word(fixture_id)), 1, width), fixture_id % 2 == 1)
    require(bitcast[DType.uint64](result) == tolerance_word(fixture_id), "independent NumPy tolerance rounding differs")


def admission_fixture(fixture_id: Int) raises:
    var nan = real(0x7FF8000000000042)
    var inf = real(0x7FF0000000000000)
    if fixture_id == 0:
        require(logistic_penalty("anything", True) == 0 and logistic_penalty("l1", False) == 1
            and logistic_penalty("l2", False) == 2 and logistic_penalty("elasticnet", False) == 3
            and logistic_penalty("L2", False) == -1 and logistic_penalty("none", False) == -1, "penalty names/admission")
    elif fixture_id == 1:
        require(logistic_positive(nan) and logistic_positive(inf) and logistic_positive(two(-1074))
            and not logistic_positive(0) and not logistic_positive(-0.0) and not logistic_positive(-inf), "positive scalar comparison policy")
    elif fixture_id == 2:
        require(logistic_ratio_valid(nan) and logistic_ratio_valid(0) and logistic_ratio_valid(1)
            and not logistic_ratio_valid(-two(-1074)) and not logistic_ratio_valid(real(0x3FF0000000000001))
            and not logistic_ratio_valid(inf), "ratio comparison policy")
    else:
        var promotion = LogisticPromotion()
        require(not promotion.captured and not promotion.weak, "promotion starts uncaptured")
        for kind in range(5):
            require(LogisticScalar(1.0, kind, 8).numpy() == (1 <= kind <= 3), "scalar origin classification")


def run_checks() raises:
    var failures = List[Int](length=LOGISTIC_CONFIG_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            if fixture_id < 34:
                scalar_fixture(fixture_id)
            elif fixture_id < 44:
                strength_fixture(fixture_id - 34)
            elif fixture_id < 50:
                tolerance_fixture(fixture_id - 44)
            else:
                admission_fixture(fixture_id - 50)
        except error:
            fp.unsafe_store(fixture_id, 1)
            print("FAIL logistic_config fixture", fixture_id, String(error))
    host_parallelize(task, LOGISTIC_CONFIG_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "logistic_config failed " + String(failed) + " fixtures")
    print("PASS logistic_config:", LOGISTIC_CONFIG_FIXTURES, "native promotion/rounding/event fixtures")
