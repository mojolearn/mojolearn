# SPDX-License-Identifier: Apache-2.0
"""Independent collision, Unicode-union and lifetime contracts for score keys.

The numeric and Unicode order tables are literal mathematical oracles, not
results copied from the implementation. Fixtures and row preparation/checks
use the shared host worker pool. No Python objects or numerical GPU kernels.
"""
from core.host_parallel import host_parallelize
from core.labels_score_keys import (
    ScoreLabelKey, ScoreKeyIndex, SCORE_INT, SCORE_UTF8, SCORE_UCS4,
    score_key_hash, score_integer_keys, score_key_join, score_key_codes,
)

comptime LABEL_SCORE_STRESS_FIXTURES = 16
comptime CODE_GUARD = Int32(0x3A4B5C6D)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def unicode_codes(count: Int, variant: Int, check_index: Bool) raises -> List[Int32]:
    # Sorted code-point order: empty,NUL,a,a+NUL,e+combiningacute,é,
    # lone high surrogate,U+10000,grinningface,fox. The surrogate has no
    # UTF-8 representation and is supplied through its admitted UCS4 form.
    var bytes: List[UInt8] = [0,97,97,0,101,204,129,195,169,
        240,144,128,128,240,159,152,128,240,159,166,138]
    var points: List[UInt32] = [0,97,97,0,101,769,233,0xD800,0x10000,0x1F600,0x1F98A]
    var byte_starts: List[Int] = [0,0,1,2,4,7,0,9,13,17]
    var byte_counts: List[Int] = [0,1,1,2,3,2,0,4,4,4]
    var point_starts: List[Int] = [0,0,1,2,4,6,7,8,9,10]
    var point_counts: List[Int] = [0,1,1,2,2,1,1,1,1,1]
    var first_rows: List[Int] = [5,8,1,4,7,0,3,6,9,2]
    var prototypes = List[ScoreLabelKey]()
    for label in range(10):  # Bounded literal alphabet metadata, not label rows.
        var utf8 = ScoreLabelKey(SCORE_UTF8, 0, Int(bytes.unsafe_ptr()) + byte_starts[label], byte_counts[label], False)
        var ucs4 = ScoreLabelKey(SCORE_UCS4, 0, Int(points.unsafe_ptr()) + point_starts[label] * 4, point_counts[label], False)
        prototypes.append(ucs4 if label == 6 else utf8)
        prototypes.append(ucs4)
    var split = count // 3
    var left = List[ScoreLabelKey](unsafe_uninit_length=split)
    var right = List[ScoreLabelKey](unsafe_uninit_length=count - split)
    var lp = left.unsafe_ptr()
    var rp = right.unsafe_ptr()
    var pp = prototypes.unsafe_ptr()
    def prepare(row: Int) {imm pp, imm lp, imm rp, imm split, imm variant}:
        var label = (row * 7 + 5) % 10
        var form = (row // 10 + variant) % 2
        var key = pp[label * 2 + form]
        if row < split:
            (lp + row).unsafe_write(key)
        else:
            (rp + row - split).unsafe_write(key)
    host_parallelize(prepare, count)
    _ = prototypes^
    var joined = score_key_join(left, right)
    var index = ScoreKeyIndex(Int(joined.unsafe_ptr()), len(joined))
    require(len(index.sorted_rows) == 10, "mixed Unicode storage failed union deduplication")
    if check_index:
        for label in range(10):  # Bounded known distinct alphabet.
            require(index.sorted_rows[label] == first_rows[label], "Unicode sorted representative order changed")
        var last = ScoreKeyIndex(Int(joined.unsafe_ptr()), len(joined), keep_last=True, ordered=False)
        for label in range(10):
            var first = first_rows[label]
            var expected = first + ((count - 1 - first) // 10) * 10
            require(last.lookup(joined[first], False) == expected, "cross-storage final duplicate was lost")
        _ = last^
    var result = List[Int32](length=count + 2, fill=CODE_GUARD)
    score_key_codes(index, joined, Int(result.unsafe_ptr()) + 4)
    require(result[0] == CODE_GUARD and result[count + 1] == CODE_GUARD,
        "Unicode code publication overwrote caller guards")
    _ = index^
    _ = joined^
    _ = left^
    _ = right^
    # Keep every byte/code-point owner alive through the last native lookup.
    _ = bytes^
    _ = points^
    return result^


def verify_unicode_codes(codes: List[Int32], count: Int) raises:
    var failures = List[Int](length=count, fill=0)
    var cp = codes.unsafe_ptr()
    var fp = failures.unsafe_ptr()
    def compare(row: Int) {imm cp, imm fp}:
        if cp[row + 1] != Int32((row * 7 + 5) % 10):
            fp.unsafe_store(row, 1)
    host_parallelize(compare, count)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "Unicode inverse codes differ from literal alphabet order")
    require(codes[0] == CODE_GUARD and codes[count + 1] == CODE_GUARD, "detached Unicode output guards changed")
    _ = codes


def unicode_union_fixture(index: Int) raises:
    var counts: List[Int] = [31,257,65537,262271]
    var count = counts[index]
    var result = unicode_codes(count, index, True)
    # All source owners and indexes have left scope. The returned code buffer
    # must be independent, including the parallel large-union path.
    verify_unicode_codes(result, count)


def collision_fixture(index: Int) raises:
    # Input selection only: these65ascending integers share the current hash's
    # low10bits. Correctness expectations below come solely from integer order.
    # The final integer is absent, so its lookup traverses the whole chain.
    var values: List[Int64] = [1346,2301,3354,4245,5426,6317,7132,8261,9204,10269,11148,12341,
        13220,14039,15228,16111,17172,18055,19244,20063,20942,22135,23014,24079,25022,26151,
        26966,27841,29038,29849,30982,31921,33865,34808,35937,36752,37945,38824,39643,40768,
        41715,42776,43659,44848,45731,46546,47739,48618,49683,50562,51755,52570,53461,54642,
        55533,56586,57477,58658,59485,60364,61557,62436,63501,64444,65979]
    var counts: List[Int] = [64,128,256,448]
    var count = counts[index]
    var source = List[Int64](unsafe_uninit_length=count)
    var vp = values.unsafe_ptr()
    var sp = source.unsafe_ptr()
    def prepare(row: Int) {imm vp, imm sp}:
        sp.unsafe_store(row, vp[(row * 17 + 3) % 64])
    host_parallelize(prepare, count)
    var keys = score_integer_keys(Int(source.unsafe_ptr()), count)
    # Integer keys own their value, so source storage may now be destroyed.
    _ = source^
    var native = ScoreKeyIndex(Int(keys.unsafe_ptr()), len(keys))
    var last = ScoreKeyIndex(Int(keys.unsafe_ptr()), len(keys), keep_last=True, ordered=False)
    require(len(native.sorted_rows) == 64, "collision chain lost distinct keys")
    var absent = ScoreLabelKey(SCORE_INT, values[64], 0, 0, False)
    require(native.lookup(absent) == -1 and last.lookup(absent, False) == -1,
        "missing colliding key failed to terminate at an empty slot")
    var mask = UInt64(len(native.slots) - 1)
    var bucket = score_key_hash(absent) & mask
    for label in range(64):  # Bounded collision alphabet; row work is parallel.
        var key = ScoreLabelKey(SCORE_INT, values[label], 0, 0, False)
        require((score_key_hash(key) & mask) == bucket, "collision fixture no longer exercises one probe chain")
        require(native.lookup(key) == label, "collision lookup changed sorted class code")
        var first = ((label - 3 + 64) * 49) % 64
        require(native.sorted_rows[label] == first, "colliding first representative changed")
        var final = first + ((count - 1 - first) // 64) * 64
        require(last.lookup(key, False) == final, "colliding last representative changed")
    var codes = List[Int32](length=count + 2, fill=CODE_GUARD)
    score_key_codes(native, keys, Int(codes.unsafe_ptr()) + 4)
    var failures = List[Int](length=count, fill=0)
    var cp = codes.unsafe_ptr()
    var fp = failures.unsafe_ptr()
    def compare(row: Int) {imm cp, imm fp}:
        if cp[row + 1] != Int32((row * 17 + 3) % 64):
            fp.unsafe_store(row, 1)
    host_parallelize(compare, count)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "colliding inverse-code batch differs from integer-order oracle")
    require(codes[0] == CODE_GUARD and codes[count + 1] == CODE_GUARD, "collision output guards changed")
    _ = native^
    _ = last^
    _ = keys^
    _ = values^
    _ = codes^


def detached_fixture(index: Int) raises:
    var counts: List[Int] = [10,37,257,65537]
    var count = counts[index]
    var result = unicode_codes(count, index + 1, False)
    # Exercise later work before checking the detached codes. No original
    # borrowed address, key list or index remains in this scope.
    var unrelated = List[UInt32](length=count, fill=0xDEADBEEF)
    verify_unicode_codes(result, count)
    require(len(unrelated) == count and unrelated[0] == UInt32(0xDEADBEEF), "independent owner changed")
    _ = unrelated^


def missing_codes_fixture(index: Int) raises:
    var source: List[Int64] = [7,-3,7,10]
    var query: List[Int64] = [10,7,999,-3]
    var keys = score_integer_keys(Int(source.unsafe_ptr()), len(source))
    var lookups = score_integer_keys(Int(query.unsafe_ptr()), len(query))
    _ = source^
    _ = query^
    var sorted = index < 2
    var last = index % 2 == 1
    var native = ScoreKeyIndex(Int(keys.unsafe_ptr()), len(keys), keep_last=last, ordered=sorted)
    var codes = List[Int32](length=6, fill=CODE_GUARD)
    score_key_codes(native, lookups, Int(codes.unsafe_ptr()) + 4, sorted_code=sorted)
    var expected: List[Int32] = [2,1,-1,0]
    if not sorted:
        expected = [3,Int32(2 if last else 0),-1,1]
    for i in range(4):  # Fixed requested-class order, including unknown sentinel.
        require(codes[i + 1] == expected[i], "sorted/requested class code or missing sentinel changed")
    require(codes[0] == CODE_GUARD and codes[5] == CODE_GUARD, "missing-code publication changed guards")
    _ = native^
    _ = keys^
    _ = lookups^


def fixture(index: Int) raises:
    if index < 4:
        unicode_union_fixture(index)
    elif index < 8:
        collision_fixture(index - 4)
    elif index < 12:
        detached_fixture(index - 8)
    else:
        missing_codes_fixture(index - 12)


def run_checks() raises:
    var failures = List[Int](length=LABEL_SCORE_STRESS_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(index: Int) {imm fp}:
        try:
            fixture(index)
        except error:
            fp.unsafe_store(index, 1)
            print("FAIL label_score_keys_stress fixture", index, String(error))
    host_parallelize(task, LABEL_SCORE_STRESS_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "label_score_keys_stress failed " + String(failed) + " fixtures")
    print("PASS label_score_keys_stress:", LABEL_SCORE_STRESS_FIXTURES, "native union/collision/lifetime fixtures")
