# SPDX-License-Identifier: Apache-2.0
"""Native KDE cache identity and take-once ownership, without device callbacks.

Fake handles are never released or queried. Pure state has no cleanup
callback; taking a handle transfers the value to its actual boundary owner.
Independent metadata fixtures run in parallel.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.kde_native_api import KDENativeDriver, KDE_MODE
from core.kde_native_state import KDEFitKey, KDEResidentState, KDEAdmissionWords

comptime KDE_STATE_FIXTURES = 30


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def state_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    if operation != KDE_MODE or result == 0 or na != 0 or np != 0 or nf != 0:
        native_descriptor_report(Error("KDE pure state only permits mode metadata"), context, sink)
        return 1
    MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = 1
    return 0


def key() -> KDEFitKey:
    return KDEFitKey(101, 37, 5, 103, String("gaussian"), String("euclidean"), 0.75)


def state(handle: Int = 107) -> KDEResidentState:
    return KDEResidentState(109, 113, key(), handle, KDENativeDriver(state_entry))


def move_state(var value: KDEResidentState) -> KDEResidentState:
    return value^


def move_words(var value: KDEAdmissionWords) -> KDEAdmissionWords:
    return value^


def fixture(unit: Int) raises:
    if unit <= 12:
        var first = key()
        var second = first.copy()
        var expected = unit == 0 or unit == 9 or unit == 11
        if unit == 1:
            second.train += 1
        elif unit == 2:
            second.rows += 1
        elif unit == 3:
            second.features += 1
        elif unit == 4:
            second.weights = 0
        elif unit == 5:
            second.kernel = "linear"
        elif unit == 6:
            second.metric = "manhattan"
        elif unit == 7:
            second.bandwidth = 1.25
        elif unit == 8:
            second.bandwidth = 0.7500000000000001
            require(Float32(first.bandwidth) == Float32(second.bandwidth), "KDE key fixture distinguishes Float64 from rounded kernel scalar")
        elif unit == 9:
            first.bandwidth = 0.0
            second.bandwidth = -0.0
        elif unit == 10:
            first.bandwidth = bitcast[DType.float64](UInt64(0x7FF8000000000000))
            second.bandwidth = first.bandwidth
        elif unit == 11:
            first.kernel = "kernel\0λ🙂"
            first.metric = "metric\0世界"
            second = first.copy()
        elif unit == 12:
            second.kernel = "changed"
            second.metric = "changed"
            require(first.kernel == "gaussian" and first.metric == "euclidean", "KDE copied key owns independent native strings")
        require(first.matches(second) == expected, "KDE exact complete fitted key")
        require(second.matches(first) == expected, "KDE key comparison is symmetric")
    elif unit <= 18:
        var handle = 0 if unit == 13 else (9007199254740993 if unit == 14 else (-1 if unit == 15 else 107))
        var value = state(handle)
        var candidate = key()
        if unit == 18:
            candidate.train += 1
        require(value.matches(110 if unit == 16 else 109, 114 if unit == 17 else 113, candidate)
            == (unit == 13 or unit == 14), "KDE owner/provider/key/valid-handle identity")
        require(value.handle == handle, "KDE pure matching never consumes the handle")
    elif unit <= 22:
        var handle = 0 if unit == 19 else (9007199254740993 if unit == 20 else (-1 if unit == 21 else 107))
        var value = state(handle)
        if unit == 22:
            var moved = move_state(value^)
            require(moved.matches(109, 113, key()) and moved.handle == 107, "KDE moved native owner retains identity")
            require(moved.driver.metadata(KDE_MODE) == 1, "KDE moved owner retains native driver")
            require(moved.take_handle() == 107 and moved.take_handle() == -1, "KDE moved handle is taken once")
        else:
            require(value.take_handle() == handle, "KDE exact integer handle transfer")
            require(value.handle == -1 and value.take_handle() == -1 and not value.matches(109, 113, key()),
                "KDE consumed state never matches or returns a second live handle")
            require(value.owner == 109 and value.provider == 113 and value.key.matches(key())
                and value.driver.metadata(KDE_MODE) == 1, "KDE take changes only native handle state")
    else:
        var words: List[Int] = []
        if unit >= 25:
            words = [11, 13, 17, 19, 23]
        var admission = KDEAdmissionWords(words.copy())
        var candidate = words.copy()
        if unit == 24 or unit == 26:
            candidate.append(29)
        elif unit == 27:
            candidate[4] = 29
        elif unit == 28:
            words[0] = 29
            require(admission.words[0] == 11, "KDE owned admission words do not alias caller copy")
        if unit == 29:
            var moved = move_words(admission^)
            require(moved.matches(candidate), "KDE moved admission owner retains metadata")
        else:
            require(admission.matches(candidate) == (unit == 23 or unit == 25 or unit == 28),
                "KDE complete admission metadata length and words")


def run_checks() raises:
    var failures = List[Int](length=KDE_STATE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL kde_state fixture", unit, String(error))
    host_parallelize(task, KDE_STATE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "kde_state failed " + String(failed) + " fixtures")
    print("PASS kde_state:", KDE_STATE_FIXTURES, "native key/owner/admission fixtures")
