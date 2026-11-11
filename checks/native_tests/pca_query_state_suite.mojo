# SPDX-License-Identifier: Apache-2.0
"""Native PCA cache settings and metadata-key ownership, without Python pins.

These cases specify admitted key identity, not a PCA numerical algorithm.
All independent fixtures run through the existing host parallel policy.
"""
from core.host_parallel import host_parallelize
from core.pca_query_state import PCAQueryState, PCAQueryIdentity

comptime PCA_QUERY_STATE_FIXTURES = 24


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def key(valid: Bool = True) -> PCAQueryIdentity:
    var result = PCAQueryIdentity(valid)
    result.words.append(101)
    result.words.append(0)  # An absent field occupies its admitted slot.
    result.words.append(303)
    return result^


def pair(left: PCAQueryIdentity, right: PCAQueryIdentity, expected: Bool) raises:
    require(left.same(right) == expected, "PCA key left/right admission identity")
    require(right.same(left) == expected, "PCA key right/left admission identity")


def surviving_copy() -> PCAQueryIdentity:
    var original = key()
    var copied = original.copy()
    original.words[0] = -17
    original.words.append(404)
    original.valid = False
    # The source owner is destroyed on return; copied native storage must
    # remain independent, with its original values and validity.
    return copied^


def fixture(unit: Int) raises:
    if unit <= 6:
        var original = PCAQueryState(47, 19, 7, unit == 1, 123456789)
        require(original.fit_rows == 47 and original.features == 19 and original.count == 7,
                "PCA native query dimensions retain distinct constructor positions")
        require(original.whiten == (unit == 1) and original.connection == 123456789,
                "PCA native query whitening/connection settings")
        if unit >= 2:
            var copied = original.copy()
            if unit == 2:
                copied.fit_rows = 91
            elif unit == 3:
                copied.features = 29
            elif unit == 4:
                copied.count = 11
            elif unit == 5:
                copied.whiten = True
            else:
                copied.connection = 987654321
            require(original.fit_rows == 47 and original.features == 19 and original.count == 7
                    and not original.whiten and original.connection == 123456789,
                    "PCA query-state copy does not alias its source")
            require(copied.fit_rows == (91 if unit == 2 else 47)
                    and copied.features == (29 if unit == 3 else 19)
                    and copied.count == (11 if unit == 4 else 7)
                    and copied.whiten == (unit == 5)
                    and copied.connection == (987654321 if unit == 6 else 123456789),
                    "PCA query-state mutation changes only the selected copied field")
    elif unit <= 10:
        var left_valid = ((unit - 7) & 1) != 0
        var right_valid = ((unit - 7) & 2) != 0
        var left = key(left_valid)
        var right = key(right_valid)
        pair(left, right, unit == 10)
    elif unit == 11:
        var left = PCAQueryIdentity()
        var right = PCAQueryIdentity()
        pair(left, right, True)
    elif unit == 12:
        var invalid = PCAQueryIdentity(False)
        var valid = PCAQueryIdentity()
        pair(invalid, valid, False)
        require(not invalid.same(invalid), "invalid empty key never admits cached state")
    elif unit == 13:
        var absent = PCAQueryIdentity()
        absent.words.append(0)
        var empty = PCAQueryIdentity()
        pair(absent, empty, False)
    elif unit == 14:
        var left = key()
        var reordered = PCAQueryIdentity()
        reordered.words.append(303)
        reordered.words.append(0)
        reordered.words.append(101)
        pair(left, reordered, False)
    elif unit == 15:
        var left = key()
        var duplicate = key()
        duplicate.words.append(303)
        pair(left, duplicate, False)
    elif unit == 16:
        var left = key()
        var moved_absent = PCAQueryIdentity()
        moved_absent.words.append(0)
        moved_absent.words.append(101)
        moved_absent.words.append(303)
        pair(left, moved_absent, False)
    elif unit <= 19:
        var left = key()
        var changed = key()
        changed.words[unit - 17] = 777
        pair(left, changed, False)
    elif unit == 20:
        var left = PCAQueryIdentity()
        left.words.append(-1)
        left.words.append(Int(0x1234567812345678))
        var right = left.copy()
        pair(left, right, True)
        right.words[1] = Int(0x2234567812345678)
        pair(left, right, False)
    elif unit == 21:
        var original = key()
        var copied = original.copy()
        copied.words[0] = 909
        copied.words.append(505)
        copied.valid = False
        var expected = key()
        pair(original, expected, True)
        require(len(copied.words) == 4 and copied.words[0] == 909 and not copied.valid,
                "PCA copied identity owns mutable native key storage")
        pair(original, copied, False)
    elif unit == 22:
        var copied = surviving_copy()
        var expected = key()
        pair(copied, expected, True)
        require(len(copied.words) == 3 and copied.valid,
                "PCA identity copy survives source destruction")
    else:
        var left = key()
        var right = key()
        require(Int(left.words.unsafe_ptr()) != Int(right.words.unsafe_ptr()),
                "PCA equivalent keys have separately owned storage")
        pair(left, right, True)
        _ = left^
        _ = right^


def run_checks() raises:
    var failures = List[Int](length=PCA_QUERY_STATE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL pca_query_state fixture", unit, String(error))
    host_parallelize(task, PCA_QUERY_STATE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "pca_query_state failed " + String(failed) + " fixtures")
    print("PASS pca_query_state:", PCA_QUERY_STATE_FIXTURES, "native state/key ownership fixtures")
