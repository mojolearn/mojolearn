# SPDX-License-Identifier: Apache-2.0
"""Native classification keys: exact ordering, duplicates and parallel mapping."""
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.labels_score_keys import (
    ScoreLabelKey, ScoreKeyIndex, score_key_compare, score_key_hash,
    score_integer_keys, score_key_join, score_key_codes,
    SCORE_INT, SCORE_BIGINT, SCORE_UTF8, SCORE_UCS4,
)

comptime SCORE_KEY_FIXTURES = 28


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def integer_index(unit: Int) raises:
    var sizes: List[Int] = [0, 1, 2, 3, 17, 31, 257, 1025, 65535, 65537, 131073, 262271]
    var count = sizes[unit]
    var classes = count if unit >= 10 else min(17, count)
    var words = List[Int64](unsafe_uninit_length=count)
    var wp = words.unsafe_ptr()
    var tasks = host_predict_task_count_work(max(1, count), max(1, count))
    var chunk = host_predict_chunk(max(1, count), tasks)
    def input(worker: Int) {imm chunk, imm count, imm wp, imm classes}:
        for i in range(worker * chunk, min((worker + 1) * chunk, count)):
            wp[i] = Int64((count - 1 - i) % classes - classes // 2)
    host_parallelize(input, tasks)
    var keys = score_integer_keys(Int(words.unsafe_ptr()), count)
    var index = ScoreKeyIndex(Int(keys.unsafe_ptr()), count)
    require(len(index.sorted_rows) == classes, "distinct integer class count")
    var output = List[Int32](length=count + 2, fill=-77)
    score_key_codes(index, keys, Int(output.unsafe_ptr()) + 4)
    var kp = keys.unsafe_ptr()
    var rp = index.sorted_rows.unsafe_ptr()
    var op = output.unsafe_ptr()
    var failures = List[Int](length=tasks, fill=0)
    var fp = failures.unsafe_ptr()
    def check(worker: Int) {imm chunk, imm count, imm op, imm classes, imm tasks, imm kp, imm rp, imm fp}:
        var failed = 0
        for i in range(worker * chunk, min((worker + 1) * chunk, count)):
            failed += Int(op[i + 1] != Int32((count - 1 - i) % classes))
        var part = host_predict_chunk(max(1, classes), tasks)
        for rank in range(worker * part, min((worker + 1) * part, classes)):
            failed += Int(kp[rp[rank]].number != Int64(rank - classes // 2))
            # Atomic representative selection must retain the first input row.
            failed += Int(rp[rank] != (count - 1 - rank) % classes)
        fp[worker] = failed
    host_parallelize(check, tasks)
    for failed in failures:
        require(failed == 0, "native integer order, first representative or inverse code")
    require(output[0] == -77 and output[count + 1] == -77, "code destination guards")
    require(index.lookup(ScoreLabelKey(SCORE_INT, -9223372036854775807, 0, 0, False)) == -1,
            "unknown native key must return -1")
    _ = index
    _ = keys^
    _ = words^
    _ = output^


def unicode_pair(unit: Int) raises:
    var strings: List[String] = ["", "a", "é", "😀", "é", "é", "z", "😀"]
    var points: List[UInt32] = [0, 97, 233, 128512, 101, 769, 0xD800, 0xDC00]
    var text = strings[unit].copy()
    var owned = List[UInt32]()
    var expected = 0
    if unit < 4:
        if unit != 0:
            owned.append(points[unit])
    elif unit == 4:
        owned.append(101)
        owned.append(769)
    elif unit == 5:
        owned.append(101)
        owned.append(769)
        expected = 1  # composed U+00E9 follows U+0065 U+0301
    elif unit == 6:
        owned.append(0xD800)
        expected = -1
    else:
        owned.append(0xDC00)
        expected = 1
    var a = ScoreLabelKey(SCORE_UTF8, 0, Int(text.unsafe_ptr()), text.byte_length(), False)
    var b = ScoreLabelKey(SCORE_UCS4, 0, Int(owned.unsafe_ptr()), len(owned), False)
    require(score_key_compare(a, b) == expected and score_key_compare(b, a) == -expected,
            "Unicode code-point order or cross-storage equality")
    if expected == 0:
        require(score_key_hash(a) == score_key_hash(b), "equal UTF-8/UCS4 keys need equal hash")
    var keys: List[ScoreLabelKey] = [a, b]
    var index = ScoreKeyIndex(Int(keys.unsafe_ptr()), len(keys))
    require(len(index.sorted_rows) == (1 if expected == 0 else 2), "Unicode distinct count")
    require(index.lookup(a) == (1 if expected > 0 else 0), "Unicode native lookup")
    _ = index
    _ = keys^
    _ = text^
    _ = owned^
    _ = strings^
    _ = points^


def bigint_index(unit: Int) raises:
    var lower = String("8000000000000001")
    var upper = String("10000000000000000")
    if unit == 1:
        lower = "ffffffffffffffff"
        upper = "10000000000000000"
    elif unit == 2:
        lower = "100000000000000000000000000000000000000000000000000"
        upper = "200000000000000000000000000000000000000000000000000"
    elif unit == 3:
        lower = "100000000000000000000000000000000000000000000000000"
        upper = "100000000000000000000000000000000000000000000000001"
    var a = ScoreLabelKey(SCORE_BIGINT, 0, Int(lower.unsafe_ptr()), lower.byte_length(), False)
    var b = ScoreLabelKey(SCORE_BIGINT, 0, Int(upper.unsafe_ptr()), upper.byte_length(), False)
    var na = a.copy()
    na.negative = True
    var nb = b.copy()
    nb.negative = True
    var keys: List[ScoreLabelKey] = [a, nb, ScoreLabelKey(SCORE_INT, 0, 0, 0, False), na, b, a]
    var index = ScoreKeyIndex(Int(keys.unsafe_ptr()), len(keys))
    var expected: List[Int] = [1, 3, 2, 0, 4]
    require(len(index.sorted_rows) == 5, "large integer union deduplication")
    for i in range(5):
        require(index.sorted_rows[i] == expected[i], "signed arbitrary integer numeric order")
    var last = ScoreKeyIndex(Int(keys.unsafe_ptr()), len(keys), keep_last=True, ordered=False)
    require(last.lookup(a, False) == 5 and last.lookup(nb, False) == 1,
            "rebound duplicate class order retains final dictionary assignment")
    _ = index
    _ = last
    _ = keys^
    _ = lower^
    _ = upper^


def join_codes(unit: Int) raises:
    var left: List[Int64] = [7, 1, 7, -3]
    var right: List[Int64] = [-3, 10, 1, 10]
    if unit == 1:
        left = List[Int64]()
    elif unit == 2:
        right = List[Int64]()
    elif unit == 3:
        left = List[Int64]()
        right = List[Int64]()
    var a = score_integer_keys(Int(left.unsafe_ptr()), len(left))
    var b = score_integer_keys(Int(right.unsafe_ptr()), len(right))
    var joined = score_key_join(a, b)
    require(len(joined) == len(left) + len(right), "joined source extent")
    for i in range(len(left)):
        require(joined[i].number == left[i], "left source order")
    for i in range(len(right)):
        require(joined[len(left) + i].number == right[i], "right source order")
    var index = ScoreKeyIndex(Int(joined.unsafe_ptr()), len(joined))
    var expected = 4 if unit == 0 else (0 if unit == 3 else 3)
    require(len(index.sorted_rows) == expected, "joined distinct set")
    var missing = ScoreLabelKey(SCORE_INT, 999, 0, 0, False)
    require(index.lookup(missing) == -1, "joined unknown class")
    _ = index
    _ = joined^
    _ = a^
    _ = b^
    _ = left^
    _ = right^


def run_checks() raises:
    var failures = List[Int](length=SCORE_KEY_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            if unit < 12:
                integer_index(unit)
            elif unit < 20:
                unicode_pair(unit - 12)
            elif unit < 24:
                bigint_index(unit - 20)
            else:
                join_codes(unit - 24)
        except error:
            fp[unit] = 1
            print("FAIL labels_score_keys fixture", unit, String(error))
    host_parallelize(task, SCORE_KEY_FIXTURES)
    var failed = 0
    for item in failures:
        failed += item
    require(failed == 0, "labels_score_keys failed " + String(failed) + " fixtures")
    print("PASS labels_score_keys:", SCORE_KEY_FIXTURES, "native key/index fixtures")
