# SPDX-License-Identifier: Apache-2.0
"""Independent wire-contract oracles for local fixed descriptor requests.

The fake entries never dereference payload addresses. Distinct sentinels name
each ABI slot; only metadata and the real retained result/error owners are read.
Every fixture, including nested calls, owns all of its request storage.
"""
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_call, native_descriptor_report
from core.native_descriptor_fixed_call import native_descriptor_call_fixed
from core.logistic_native_api import (
    QnNativeDriver, QnFoldDriver, QnFitRequest, QnQueryRequest, QnLinkRequest, QnFoldStore,
)


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _ready(result: Int, context: Int) raises:
    _require(MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] == 0,
        "descriptor result must start reset")
    _require(len(MutPointer[List[UInt8], MutUntrackedOrigin](unsafe_from_address=context)[]) == 0,
        "descriptor error must start empty")


def _int_slots[N: Int](address: Int, expected: InlineArray[Int, N]) raises:
    var actual = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=address)
    comptime for i in range(N):
        _require(actual[i] == expected[i], "integer/address wire slot " + String(i))
    _ = expected


def _float_slots(address: Int) raises:
    var actual = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=address)
    _require(actual[0] == 0.125 and actual[1] == 0.25 and actual[2] == 0.5
        and actual[3] == 1.0 and actual[4] == 2.0, "floating-point wire slots")


def _dense_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        _ready(result, context)
        _require(aa > 0 and pp > 0, "dense request metadata exists")
        var token = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)[0]
        if operation == 0:
            _require(na == 6 and np == 9 and nf == 5 and ff > 0, "dense fit counts")
            var a: InlineArray[Int, 6] = [token, token + 1, token + 2, token + 3, token + 4, token + 5]
            var p: InlineArray[Int, 9] = [17, 19, 3, 29, 31, 7, 1, 0, 5]
            _int_slots(aa, a)
            _int_slots(pp, p)
            _float_slots(ff)
        elif 1 <= operation <= 4:
            _require(na == 4 and np == 5 and nf == 0 and ff == 0, "dense query counts/empty floats")
            var a: InlineArray[Int, 4] = [token, token + 2, token + 3, token + 5]
            var p: InlineArray[Int, 5] = [17, 19, 3, 1, 1]
            _int_slots(aa, a)
            _int_slots(pp, p)
        elif operation == 5 or operation == 6:
            _require(na == 2 and np == 2 and nf == 0 and ff == 0, "dense link counts/empty floats")
            var a: InlineArray[Int, 2] = [token, token + 3]
            var p: InlineArray[Int, 2] = [17, 3]
            _int_slots(aa, a)
            _int_slots(pp, p)
        else:
            _require(operation == 7 and na == 2 and np == 1 and nf == 0 and ff == 0, "dense log contract")
            var a: InlineArray[Int, 2] = [token, token + 3]
            var p: InlineArray[Int, 1] = [51]
            _int_slots(aa, a)
            _int_slots(pp, p)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = token + 97
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


def _fold_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        _ready(result, context)
        _require(aa > 0 and pp > 0, "fold request metadata exists")
        var token = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)[0]
        if operation == 0:
            token -= 1
            _require(na == 5 and np == 12 and nf == 5 and ff > 0, "fold fit counts")
            var a: InlineArray[Int, 5] = [token + 1, token + 2, token + 3, token + 4, token + 6]
            var p: InlineArray[Int, 12] = [53, 59, 61, 17, 19, 3, 29, 31, 7, 1, 0, 5]
            _int_slots(aa, a)
            _int_slots(pp, p)
            _float_slots(ff)
        else:
            token -= 2
            _require(1 <= operation <= 4 and na == 3 and np == 8 and nf == 0 and ff == 0, "fold query contract")
            var a: InlineArray[Int, 3] = [token + 2, token + 3, token + 6]
            var p: InlineArray[Int, 8] = [53, 59, 61, 17, 19, 3, 1, 1]
            _int_slots(aa, a)
            _int_slots(pp, p)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = token + 97
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


def _empty_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        _ready(result, context)
        _require(aa == 0 and pp == 0 and ff == 0 and na == 0 and np == 0 and nf == 0,
            "empty fixed spans must carry null addresses")
        if operation == 1:
            raise Error("nested refusal")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = 73
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


def _fixed_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        _ready(result, context)
        _require(na == 1 and np == 1 and nf == 1 and aa > 0 and pp > 0 and ff > 0, "fixed request spans")
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        var first = a[0]
        var second = p[0]
        var third = f[0]
        if operation == 1:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = -777
            # Multiple callback fragments exercise complete UTF-8 error capture.
            var prefix = String("fixed λ ")
            var suffix = String("refusal")
            var prefix_bytes = prefix.as_bytes()
            var suffix_bytes = suffix.as_bytes()
            sink(context, Int(prefix_bytes.unsafe_ptr()), len(prefix_bytes))
            sink(context, Int(suffix_bytes.unsafe_ptr()), len(suffix_bytes))
            _ = prefix_bytes
            _ = suffix_bytes
            _ = prefix
            _ = suffix
            return 1
        if operation == 2:
            return 0  # A successful entry may intentionally leave the result zero.
        if operation == 3 or operation == 4:
            var empty_i: InlineArray[Int, 0] = []
            var empty_f: InlineArray[Float64, 0] = []
            if operation == 3:
                _require(native_descriptor_call_fixed(_empty_entry, 0, empty_i, empty_i, empty_f) == 73,
                    "nested descriptor result")
            else:
                var message = String()
                try:
                    _ = native_descriptor_call_fixed(_empty_entry, 1, empty_i, empty_i, empty_f)
                except problem:
                    message = String(problem)
                _require(message == "nested refusal", "nested refusal must not replace outer error storage")
            _require(a[0] == first and p[0] == second and f[0] == third, "nested call must preserve outer fixed spans")
            _ready(result, context)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = first + second + Int(third)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


def _fit_request(token: Int) -> QnFitRequest:
    return QnFitRequest(token, token + 1, token + 2, token + 3, token + 4, token + 5,
        17, 19, 3, 29, 31, 7, True, False, 5, 0.125, 0.25, 0.5, 1.0, 2.0)


def logistic_driver_fixture(operation: Int, token: Int = 101) raises:
    var driver = QnNativeDriver(_dense_entry)
    if operation == 0:
        _require(driver.fit(_fit_request(token)) == token + 97, "fit iterations must cross the ABI")
    elif 1 <= operation <= 4:
        driver.query(operation, QnQueryRequest(token, token + 2, token + 3, token + 5, 17, 19, 3, True, True))
    elif operation == 5 or operation == 6:
        driver.link(operation, QnLinkRequest(token, token + 3, 17, 3))
    else:
        driver.logarithm(token, token + 3, 51)


def logistic_fold_fixture(operation: Int) raises:
    var token = 211
    var driver = QnFoldDriver(_fold_entry)
    var store = QnFoldStore(53, 59, 61)
    if operation == 0:
        _require(driver.fit(store, _fit_request(token), token + 6) == token + 97, "fold iterations must cross the ABI")
    else:
        driver.query(operation, store,
            QnQueryRequest(token, token + 2, token + 3, token + 5, 17, 19, 3, True, True), token + 6)


def logistic_fixed_fixture(unit: Int) raises:
    var a: InlineArray[Int, 1] = [37]
    var p: InlineArray[Int, 1] = [41]
    var f: InlineArray[Float64, 1] = [43.0]
    if unit == 1 or unit == 5:
        var message = String()
        try:
            _ = native_descriptor_call_fixed(_fixed_entry, 1, a, p, f)
        except problem:
            message = String(problem)
        _require(message == "fixed λ refusal", "full refusal text takes precedence over written result")
        if unit == 1:
            return
    var value = native_descriptor_call_fixed(_fixed_entry, 0 if unit == 5 else unit, a, p, f)
    _require(value == (0 if unit == 2 else 121), "fixed span result/reset")
    if unit == 5:
        var dynamic_a: List[Int] = [37]
        var dynamic_p: List[Int] = [41]
        var dynamic_f: List[Float64] = [43.0]
        _require(native_descriptor_call(_fixed_entry, 0, dynamic_a, dynamic_p, dynamic_f) == value,
            "unchanged dynamic and fixed helpers share result semantics")
    _require(a[0] == 37 and p[0] == 41 and f[0] == 43.0, "request storage remains owned after return")


def logistic_concurrent_fixture(unit: Int) raises:
    var token = 1001 + unit * 11
    logistic_driver_fixture(0, token)
    var a: InlineArray[Int, 1] = [token]
    var p: InlineArray[Int, 1] = [token + 1]
    var f: InlineArray[Float64, 1] = [Float64(token + 2)]
    _require(native_descriptor_call_fixed(_fixed_entry, 3, a, p, f) == 3 * token + 3,
        "concurrent/nested callers must retain private request and output storage")
    _require(a[0] == token and p[0] == token + 1 and f[0] == Float64(token + 2), "concurrent request owners")
