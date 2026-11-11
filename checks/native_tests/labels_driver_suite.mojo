# SPDX-License-Identifier: Apache-2.0
"""Independent native label wire, kernel and ownership fixtures; no Python."""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.hotpath_impl import HpStat, HP_STAT_INT, HP_STAT_FLOAT, HP_STAT_TUPLE
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_fixed_call import native_descriptor_call_fixed
from core.labels_native_api import (
    LabelsNativeDriver, LabelsPrepareRequest, LabelsReduceRequest, LabelsArgmaxRequest,
    LabelsGatherRequest, LabelsRatioRequest, labels_native_dispatch, labels_stat_store, labels_stat_load,
)
from core.labels_native_host import LabelsHostKernels, labels_host_fsum


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _entry(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return labels_native_dispatch[LabelsHostKernels](op, aa, na, pp, np, ff, nf, result, context, sink)


def _encode[dt: DType](code: Int) raises:
    var source: List[Scalar[dt]] = [7, 2, 7, 0, 2]
    var classes = List[UInt64](length=5, fill=0)
    var codes = List[Int32](length=5, fill=-99)
    var driver = LabelsNativeDriver(_entry)
    var count = driver.prepare(LabelsPrepareRequest(Int(source.unsafe_ptr()), Int(classes.unsafe_ptr()),
        Int(codes.unsafe_ptr()), 0, code, 5, 0, 0))
    _require(count == 3, "three sorted distinct classes")
    var expected_classes: List[Int] = [0, 2, 7]
    var expected_codes: List[Int32] = [2, 1, 2, 0, 1]
    for i in range(3):
        var word = bitcast[DType.uint64](Float64(expected_classes[i])) if code < 2 else UInt64(expected_classes[i])
        _require(classes[i] == word, "class widening and order")
    for i in range(5):
        _require(codes[i] == expected_codes[i], "inverse codes")
    _ = source^


def _refusal(unit: Int) raises:
    var source: List[Float64] = [1.0]
    var classes = List[UInt64](length=1, fill=0)
    var codes = List[Int32](length=1, fill=0)
    var request = LabelsPrepareRequest(Int(source.unsafe_ptr()), Int(classes.unsafe_ptr()),
        Int(codes.unsafe_ptr()), 0, 1, 1, 0, 0)
    var expected = String()
    if unit == 0:
        request.count = -1
        expected = "prepare_targets: n must be non-negative, got -1"
    elif unit == 1:
        request.count = 0
        expected = "prepare_targets: labels need n >= 1"
    elif unit == 2 or unit == 3:
        if unit == 2:
            request.classes = 0
        else:
            request.codes = 0
        expected = "prepare_targets: null label output address"
    elif unit == 4:
        request.label_code = 6
        expected = "prepare_targets: label dtype code must be 0..5, got 6"
    elif unit == 5:
        source[0] = bitcast[DType.float64](UInt64(0x7FF8000000000042))
        request.weights = Int(source.unsafe_ptr())
        request.weight_code = 1
        expected = "mojolearn: y contains a NaN label; NaN is not a class"
    elif unit == 6:
        request.labels = 0
        request.weights = Int(source.unsafe_ptr())
        request.weight_code = 2
        expected = "prepare_targets: weight dtype code must be 0 or 1, got 2"
    elif unit == 7:
        expected = "argmax_rows_f32: n_rows must be non-negative and n_cols positive"
    elif unit == 8:
        expected = "argmax_rows_f64: n_rows must be non-negative and n_cols positive"
    elif unit == 9:
        expected = "gather_i32: n must be non-negative and the table non-empty"
    elif unit == 10:
        expected = "gather_i64: n must be non-negative and the table non-empty"
    else:
        expected = "class_ratio_f64: mode is 0 (balanced) or 1 (prior)"
    var driver = LabelsNativeDriver(_entry)
    var message = String()
    try:
        if unit <= 6:
            _ = driver.prepare(request)
        elif unit == 7 or unit == 8:
            driver.argmax(unit == 8, LabelsArgmaxRequest(0, 0, -1 if unit == 7 else 1, 0))
        elif unit == 9 or unit == 10:
            driver.gather(unit - 9, LabelsGatherRequest(0, 0, 0, 1, -1))
        else:
            _ = driver.ratio(LabelsRatioRequest(0, 0, 2, 5, 2))
    except problem:
        message = String(problem)
    _require(message == expected, "refusal order and text: " + message)
    _ = source^
    _ = classes^
    _ = codes^


def _weights(unit: Int) raises:
    var source: List[Float64] = [1, 2]
    var flags = 0
    var expected = 0
    var count = 2
    var code = 1
    if unit == 1 or unit == 2:
        source[0] = -1
        flags = 1 if unit == 2 else 0
        expected = 0 if unit == 2 else -4
    elif unit == 3 or unit == 4:
        source[0] = 0
        source[1] = 0
        flags = 2 if unit == 4 else 0
        expected = 0 if unit == 4 else -5
    elif unit == 5 or unit == 6:
        source[0] = bitcast[DType.float64](UInt64(0x7FF0000000000000 if unit == 5 else 0x7FF8000000000042))
        source[1] = -1
        flags = 3
        expected = -3
    elif unit == 7:
        count = 0
        code = 99  # Empty weight vectors never inspect an unused dtype.
    elif unit == 8:
        source[0] = bitcast[DType.float64](UInt64(0x8000000000000000))
        source[1] = 0
        expected = -5
    elif unit == 9:
        source[0] = -1
        source[1] = 0
        flags = 1
    var driver = LabelsNativeDriver(_entry)
    _require(driver.prepare(LabelsPrepareRequest(0, 0, 0, Int(source.unsafe_ptr()), 0, count, code, flags)) == expected,
        "weight finite/negative/all-zero status order")
    _ = source^


def _stat(unit: Int) raises:
    var driver = LabelsNativeDriver(_entry)
    if unit >= 5:
        var source: List[Int64] = [-9223372036854775807 - 1, -1, 9223372036854775807]
        if unit == 7:
            source[0] = 9223372036854775807
            source[1] = 9223372036854775807
            source[2] = -1
        var value = driver.reduce(LabelsReduceRequest(Int(source.unsafe_ptr()), 3, 3, 5 if unit == 7 else unit - 5))
        if unit == 7:
            _require(value.kind == HP_STAT_TUPLE and value.hi == 0 and value.mid == 4294967295 and value.low == 4294967293,
                "exact integer sum exceeds signed 64-bit without losing words")
        else:
            _require(value.kind == HP_STAT_INT and value.i == (-9223372036854775807 - 1 if unit == 5 else 9223372036854775807),
                "signed 64-bit reduction result")
        _ = source^
        return
    var source: List[Float64] = [1, 2]
    var operation = 4
    if unit == 0:
        source[0] = bitcast[DType.float64](UInt64(0x8000000000000000))
        source[1] = 0
        operation = 0
    elif unit == 1:
        source[0] = bitcast[DType.float64](UInt64(0x7FF8000000000042))
        operation = 1
    elif unit == 3:
        source[1] = 1.5
    elif unit == 4:
        source[1] = bitcast[DType.float64](UInt64(0x7FF0000000000000))
    var value = driver.reduce(LabelsReduceRequest(Int(source.unsafe_ptr()), 1, 2, operation))
    if unit == 0 or unit == 1:
        _require(value.kind == HP_STAT_FLOAT and bitcast[DType.uint64](value.f) == UInt64(0x8000000000000000 if unit == 0 else 0x7FF8000000000042),
            "reduction wire preserves signed zero/first NaN payload")
    else:
        _require(value.kind == HP_STAT_INT and value.i == (1 if unit == 2 else 0), "finite integral predicate")
    _ = source^


def _gather(unit: Int) raises:
    var driver = LabelsNativeDriver(_entry)
    if unit >= 6:
        driver.gather(0 if unit == 6 else 2, LabelsGatherRequest(0, 0, 0, 1, 0))
        return
    var kind = unit % 3
    var bad = unit >= 3
    var message = String()
    if kind == 0:
        var table: List[Int32] = [7, -9, 3]
        var codes: List[Int32] = [2, Int32(3 if bad else 0), 1]
        var dst = List[Int32](length=3, fill=-99)
        try:
            driver.gather(0, LabelsGatherRequest(Int(table.unsafe_ptr()), Int(codes.unsafe_ptr()), Int(dst.unsafe_ptr()), 3, 3))
        except problem:
            message = String(problem)
        if bad:
            _require(message == "gather_i32: code out of range" and dst[0] == -99 and dst[1] == -99 and dst[2] == -99, "gather refusal before writes")
        else:
            _require(message == "" and dst[0] == 3 and dst[1] == 7 and dst[2] == -9, "integer gather")
        _ = table^
        _ = codes^
        return
    var table: List[UInt64] = [0x8000000000000000, 0x7FF8000000000042, 0xFFFFFFFFFFFFFFFF]
    var codes: List[Int64] = [2, Int64(3 if bad else 0), 1]
    var dst = List[UInt64](length=3, fill=0xDEADBEEF)
    try:
        driver.gather(kind, LabelsGatherRequest(Int(table.unsafe_ptr()), Int(codes.unsafe_ptr()), Int(dst.unsafe_ptr()), 3, 3))
    except problem:
        message = String(problem)
    if bad:
        var name = String("gather_i64") if kind == 1 else String("gather_f64")
        _require(message == name + ": code out of range" and dst[0] == 0xDEADBEEF and dst[1] == 0xDEADBEEF and dst[2] == 0xDEADBEEF,
            "wide gather refusal before writes")
    else:
        _require(message == "" and dst[0] == table[2] and dst[1] == table[0] and dst[2] == table[1], "wide gather preserves raw words")
    _ = table^
    _ = codes^


def _argmax[dt: DType](unit: Int) raises:
    var driver = LabelsNativeDriver(_entry)
    if unit == 2:
        driver.argmax(dt == DType.float64, LabelsArgmaxRequest(0, 0, 0, 3))
        return
    var source: List[Scalar[dt]] = [1, 3, 3, 0, 5, 6]
    source[3] = bitcast[DType.float64](UInt64(0x7FF8000000000042)).cast[dt]()
    if unit == 1:
        source[0] = source[3]
        source[3] = 0
        source[4] = 0
        source[5] = 0
    var dst = List[Int64](length=2, fill=-1)
    driver.argmax(dt == DType.float64, LabelsArgmaxRequest(Int(source.unsafe_ptr()), Int(dst.unsafe_ptr()), 2, 3))
    _require(dst[0] == Int64(1 if unit == 0 else 0) and dst[1] == 0, "first max wins, including first NaN")
    _ = source^


def _ratio(unit: Int) raises:
    var counts: List[Int64] = [2, 3]
    var dst = List[Float64](length=2, fill=-1)
    var driver = LabelsNativeDriver(_entry)
    if unit == 3:
        _require(driver.ratio(LabelsRatioRequest(0, 0, 0, 0, 99)) == 0, "empty ratio skips unused mode")
        return
    if unit == 4:
        var message = String()
        try:
            _ = driver.ratio(LabelsRatioRequest(0, 0, 2, 0, 1))
        except problem:
            message = String(problem)
        _require(message == "class_ratio_f64: the prior of zero rows", "prior zero-row refusal")
        return
    if unit == 2:
        counts[1] = 0
    var status = driver.ratio(LabelsRatioRequest(Int(counts.unsafe_ptr()), Int(dst.unsafe_ptr()), 2, 5, 1 if unit == 1 else 0))
    _require(status == (1 if unit == 2 else 0), "zero-count ratio status")
    if unit == 1:
        _require(bitcast[DType.uint64](dst[0]) == 0x3FD999999999999A and bitcast[DType.uint64](dst[1]) == 0x3FE3333333333333,
            "prior ratio known words")
    else:
        _require(dst[0] == 1.25 and (dst[1] == 0 if unit == 2 else bitcast[DType.uint64](dst[1]) == 0x3FEAAAAAAAAAAAAB),
            "balanced ratio known words")
    _ = counts^


def _fsum(unit: Int) raises:
    var source: List[Float64] = [1e16, 1, -1e16]
    var dst: List[Float64] = [-1]
    if unit == 1:
        source[0] = bitcast[DType.float64](UInt64(0x7FF0000000000000))
        source[1] = bitcast[DType.float64](UInt64(0xFFF0000000000000))
    var status = labels_host_fsum(0 if unit == 2 else Int(source.unsafe_ptr()), 0 if unit == 2 else 3, Int(dst.unsafe_ptr()))
    _require(status == (1 if unit == 1 else 0), "exact fsum refusal status")
    if unit != 1:
        _require(dst[0] == (0.0 if unit == 2 else 1.0), "exact cancellation/empty fsum")
    _ = source^


def _large[dt: DType](code: Int) raises:
    var count = 4097  # One more distinct value than the established encoder cap.
    var source = List[Scalar[dt]](length=count, fill=0)
    var classes = List[UInt64](length=count, fill=0)
    var codes = List[Int32](length=count, fill=-1)
    var sp = source.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var cp = classes.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var ip = codes.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var tasks = host_predict_task_count_work(count, count)
    var chunk = host_predict_chunk(count, tasks)
    def fill(worker: Int) {imm sp, imm count, imm chunk}:
        for i in range(worker * chunk, min(count, (worker + 1) * chunk)):
            sp[i] = Scalar[dt](count - 1 - i)
    host_parallelize(fill, tasks)
    var driver = LabelsNativeDriver(_entry)
    _require(driver.prepare(LabelsPrepareRequest(Int(sp), Int(cp), Int(ip), 0, code, count, 0, 0)) == count,
        "uncapped fallback retains every class")
    var failed = List[Int](length=tasks, fill=0)
    var fp = failed.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    def verify(worker: Int) {imm cp, imm ip, imm fp, imm count, imm chunk, imm code}:
        for i in range(worker * chunk, min(count, (worker + 1) * chunk)):
            var expected = bitcast[DType.uint64](Float64(i)) if code < 2 else UInt64(i)
            if cp[i] != expected or ip[i] != Int32(count - 1 - i):
                fp[worker] = 1
    host_parallelize(verify, tasks)
    for flag in failed:
        _require(flag == 0, "uncapped class/order and inverse-code oracle")
    _ = source^
    _ = classes^
    _ = codes^


def _wire(unit: Int) raises:
    var words = List[Int64](length=3, fill=0)
    var address = Int(words.unsafe_ptr())
    if unit <= 2:
        var value = HpStat(unit, -77, bitcast[DType.float64](UInt64(0x8000000000000000)), -1, 0xABCD, 23)
        var tag = labels_stat_store(value, address)
        var decoded = labels_stat_load(tag, address)
        _require(decoded.kind == unit, "wire answer kind")
        if unit == 0:
            _require(decoded.i == -77, "wire signed integer")
        elif unit == 1:
            _require(bitcast[DType.uint64](decoded.f) == 0x8000000000000000, "wire signed zero")
        else:
            _require(decoded.hi == -1 and decoded.mid == 0xABCD and decoded.low == 23, "wire exact integer triple")
    else:
        var message = String()
        var expected = String()
        try:
            if unit == 3:
                expected = "labels native ABI: unknown reduce_stat answer kind 7"
                _ = labels_stat_load(7, address)
            elif unit == 4:
                expected = "labels native ABI: reduce_stat needs an output span"
                _ = labels_stat_store(HpStat(0, 0, 0, 0, 0, 0), 0)
            else:
                var a: Array[Int, 0] = []
                var p: Array[Int, 0] = []
                var f: Array[Float64, 0] = []
                if unit == 5:
                    expected = "labels native ABI: unknown operation"
                    _ = native_descriptor_call_fixed(_entry, 99, a, p, f)
                elif unit == 6:
                    expected = "labels native ABI: prepare_targets takes 4 addresses and 4 words"
                    _ = native_descriptor_call_fixed(_entry, 2, a, p, f)
                else:
                    expected = "labels native ABI: invalid result or scalar span"
                    var scalar: Array[Float64, 1] = [1]
                    _ = native_descriptor_call_fixed(_entry, 0, a, p, scalar)
        except problem:
            message = String(problem)
        _require(message == expected, "wire refusal text: " + message)
    _ = words^


def _concurrent(unit: Int) raises:
    var token = unit * 17 + 9
    var source: List[Int32] = [Int32(token + 1), Int32(token), Int32(token + 1)]
    var classes = List[UInt64](length=3, fill=0)
    var codes = List[Int32](length=3, fill=-1)
    var driver = LabelsNativeDriver(_entry)
    _require(driver.prepare(LabelsPrepareRequest(Int(source.unsafe_ptr()), Int(classes.unsafe_ptr()),
        Int(codes.unsafe_ptr()), 0, 2, 3, 0, 0)) == 2, "concurrent class count")
    _require(classes[0] == UInt64(token) and classes[1] == UInt64(token + 1)
        and codes[0] == 1 and codes[1] == 0 and codes[2] == 1, "concurrent owners and values")
    _require(driver.reduce(LabelsReduceRequest(Int(source.unsafe_ptr()), 2, 3, 0)).i == token,
        "concurrent native result storage")
    _ = source^


def run_checks() raises:
    comptime FIXTURES = 100
    var failures = List[Int](length=FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            if unit == 0:
                _encode[DType.float32](0)
            elif unit == 1:
                _encode[DType.float64](1)
            elif unit == 2:
                _encode[DType.int32](2)
            elif unit == 3:
                _encode[DType.int64](3)
            elif unit == 4:
                _encode[DType.uint32](4)
            elif unit == 5:
                _encode[DType.uint8](5)
            elif unit < 18:
                _refusal(unit - 6)
            elif unit < 28:
                _weights(unit - 18)
            elif unit < 36:
                _stat(unit - 28)
            elif unit < 44:
                _gather(unit - 36)
            elif unit < 47:
                _argmax[DType.float32](unit - 44)
            elif unit < 50:
                _argmax[DType.float64](unit - 47)
            elif unit < 55:
                _ratio(unit - 50)
            elif unit < 58:
                _fsum(unit - 55)
            elif unit == 58:
                _large[DType.float32](0)
            elif unit == 59:
                _large[DType.int64](3)
            elif unit < 68:
                _wire(unit - 60)
            else:
                _concurrent(unit - 68)
        except problem:
            print("FAIL labels_driver fixture", unit, String(problem))
            fp.unsafe_store(unit, 1)
    host_parallelize(task, FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    _require(failed == 0, "native label driver failed " + String(failed) + " of " + String(FIXTURES) + " fixtures")
    print("PASS labels_driver:", FIXTURES, "native fixtures, parallel")
