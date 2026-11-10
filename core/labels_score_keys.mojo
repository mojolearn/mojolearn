# SPDX-License-Identifier: Apache-2.0
"""Native immutable classification keys and parallel lookup preparation.

The boundary pins admitted scalar storage. Integer keys are signed Int64 or
canonical hexadecimal magnitudes; text keys borrow UTF-8 or owned UCS4 spans.
No Python hash, equality, sorting, dictionary or set operation occurs here.
One index contains either numeric keys or text keys; admission refuses mixing.
Big-integer keys are canonical hexadecimal magnitudes outside signed Int64.
"""
from std.algorithm import sort
from std.atomic import Atomic, Ordering
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk

comptime SCORE_INT = 0
comptime SCORE_BIGINT = 1
comptime SCORE_UTF8 = 2
comptime SCORE_UCS4 = 3


@fieldwise_init
struct ScoreLabelKey(ImplicitlyCopyable, Movable):
    var kind: Int
    var number: Int64
    var address: Int
    var count: Int
    var negative: Bool


comptime KeyPointer = MutPointer[ScoreLabelKey, MutUntrackedOrigin]
comptime WordPointer = MutPointer[Int, MutUntrackedOrigin]
comptime BytePointer = MutPointer[UInt8, MutUntrackedOrigin]
comptime U32Pointer = MutPointer[UInt32, MutUntrackedOrigin]


def _bytes_compare(a: Int, na: Int, b: Int, nb: Int) -> Int:
    var ap = BytePointer(unsafe_from_address=a)
    var bp = BytePointer(unsafe_from_address=b)
    for i in range(min(na, nb)):
        if ap[i] != bp[i]:
            return -1 if ap[i] < bp[i] else 1
    return 0 if na == nb else (-1 if na < nb else 1)


def _point(key: ScoreLabelKey, mut at: Int) -> UInt32:
    if key.kind == SCORE_UCS4:
        var point = U32Pointer(unsafe_from_address=key.address)[at]
        at += 1
        return point
    var bytes = BytePointer(unsafe_from_address=key.address)
    var first = UInt32(bytes[at])
    at += 1
    if first < 128:
        return first
    var extra = 1 if first < 224 else (2 if first < 240 else 3)
    var result = first & (UInt32(31) if extra == 1 else (UInt32(15) if extra == 2 else UInt32(7)))
    for _ in range(extra):
        result = (result << 6) | (UInt32(bytes[at]) & 63)
        at += 1
    return result


def score_key_compare(a: ScoreLabelKey, b: ScoreLabelKey) -> Int:
    if a.kind == SCORE_INT and b.kind == SCORE_INT:
        return 0 if a.number == b.number else (-1 if a.number < b.number else 1)
    if a.kind < SCORE_UTF8:
        if a.kind == SCORE_INT:
            return 1 if b.negative else -1
        if b.kind == SCORE_INT:
            return -1 if a.negative else 1
        if a.negative != b.negative:
            return -1 if a.negative else 1
        var cmp = -1 if a.count < b.count else 1
        if a.count == b.count:
            cmp = _bytes_compare(a.address, a.count, b.address, b.count)
        return -cmp if a.negative else cmp
    if a.kind == SCORE_UTF8 and b.kind == SCORE_UTF8:
        return _bytes_compare(a.address, a.count, b.address, b.count)
    var ia = 0
    var ib = 0
    while ia < a.count and ib < b.count:
        var pa = _point(a, ia)
        var pb = _point(b, ib)
        if pa != pb:
            return -1 if pa < pb else 1
    return 0 if ia == a.count and ib == b.count else (-1 if ia == a.count else 1)


def _hash_byte(mut value: UInt64, byte: UInt32):
    value = (value ^ UInt64(byte)) * UInt64(1099511628211)


def score_key_hash(key: ScoreLabelKey) -> UInt64:
    var result = UInt64(14695981039346656037)
    if key.kind == SCORE_INT:
        var word = UInt64(key.number)
        for _ in range(8):
            _hash_byte(result, UInt32(word & 255))
            word >>= 8
        return result
    if key.kind == SCORE_BIGINT:
        _hash_byte(result, UInt32(1 if key.negative else 0))
    if key.kind != SCORE_UCS4:
        var data = BytePointer(unsafe_from_address=key.address)
        for i in range(key.count):
            _hash_byte(result, UInt32(data[i]))
        return result
    var points = U32Pointer(unsafe_from_address=key.address)
    for i in range(key.count):
        var point = points[i]
        if point < 128:
            _hash_byte(result, point)
        elif point < 2048:
            _hash_byte(result, 192 | (point >> 6))
            _hash_byte(result, 128 | (point & 63))
        elif point < 65536:
            _hash_byte(result, 224 | (point >> 12))
            _hash_byte(result, 128 | ((point >> 6) & 63))
            _hash_byte(result, 128 | (point & 63))
        else:
            _hash_byte(result, 240 | (point >> 18))
            _hash_byte(result, 128 | ((point >> 12) & 63))
            _hash_byte(result, 128 | ((point >> 6) & 63))
            _hash_byte(result, 128 | (point & 63))
    return result


def _less(keys: KeyPointer, a: Int, b: Int) -> Bool:
    return score_key_compare(keys[a], keys[b]) < 0


def _merge_cut(keys: KeyPointer, data: WordPointer, left: Int, middle: Int, end: Int,
               diagonal: Int) -> Int:
    # Stable merge path: a partition starts where its diagonal intersects the
    # merge of the two sorted runs. Each task writes a disjoint output span.
    var lo = max(0, diagonal - (end - middle))
    var hi = min(diagonal, middle - left)
    while lo < hi:
        var x = (lo + hi) // 2
        var y = diagonal - x
        if y > 0 and x < middle - left and not _less(keys, data[middle + y - 1], data[left + x]):
            lo = x + 1
        else:
            hi = x
    return lo


def _sort_distinct(keys: KeyPointer, mut rows: List[Int]):
    var n = len(rows)
    if n < 2:
        return
    # Comparison sorting does at least one comparison per level. Feed that
    # work to the existing host policy rather than treating sort as one copy.
    var levels = 0
    var remaining = n
    while remaining > 1:
        levels += 1
        remaining = (remaining + 1) // 2
    var units = host_predict_task_count_work(n, n * levels)
    var chunk = host_predict_chunk(n, units)
    var rp = WordPointer(unsafe_from_address=Int(rows.unsafe_ptr()))
    @parameter
    def initial(unit: Int):
        var lo = unit * chunk
        var hi = min(lo + chunk, n)
        if lo >= hi:
            return
        var part = Span(unsafe_ptr=rp + lo, length=hi - lo)
        @parameter
        def less(a: Int, b: Int) -> Bool:
            return _less(keys, a, b)
        sort[cmp_fn=less](part)
    host_parallelize(initial, units)
    var scratch = List[Int](unsafe_uninit_length=n)
    var sp = WordPointer(unsafe_from_address=Int(scratch.unsafe_ptr()))
    var width = chunk
    while width < n:
        var source = rp
        var target = sp
        var run_width = width
        @parameter
        def merge(unit: Int):
            var lo = unit * chunk
            var hi = min(lo + chunk, n)
            while lo < hi:
                var left = (lo // (2 * run_width)) * (2 * run_width)
                var middle = min(left + run_width, n)
                var end = min(left + 2 * run_width, n)
                var stop = min(hi, end)
                var x = left + _merge_cut(keys, source, left, middle, end, lo - left)
                var y = middle + (lo - left) - (x - left)
                for position in range(lo, stop):
                    if x < middle and (y >= end or not _less(keys, source[y], source[x])):
                        target[position] = source[x]
                        x += 1
                    else:
                        target[position] = source[y]
                        y += 1
                lo = stop
        host_parallelize(merge, units)
        var swap = rp
        rp = sp
        sp = swap
        width *= 2
    if Int(rp) != Int(rows.unsafe_ptr()):
        var final = WordPointer(unsafe_from_address=Int(rows.unsafe_ptr()))
        var source = rp
        @parameter
        def copy(unit: Int):
            for i in range(unit * chunk, min((unit + 1) * chunk, n)):
                final[i] = source[i]
        host_parallelize(copy, units)
    _ = scratch^
    _ = rows


struct ScoreKeyIndex(Movable):
    var keys: Int
    var count: Int
    var slots: List[Int64]
    var sorted_rows: List[Int]
    var codes: List[Int32]

    def __init__(out self, keys: Int, count: Int, keep_last: Bool = False, ordered: Bool = True):
        self.keys = keys
        self.count = count
        var capacity = 16
        while capacity < 2 * count:
            capacity *= 2
        self.slots = List[Int64](unsafe_uninit_length=capacity)
        self.codes = List[Int32](unsafe_uninit_length=count if ordered else 0)
        self.sorted_rows = List[Int]()
        var kp = KeyPointer(unsafe_from_address=keys)
        var slots = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(self.slots.unsafe_ptr()))
        var mask = capacity - 1
        # Even an integer hashes eight bytes; use that minimum work estimate.
        var units = host_predict_task_count_work(max(1, count), max(1, count) * 8)
        var chunk = host_predict_chunk(max(1, count), units)
        var slot_chunk = host_predict_chunk(capacity, units)
        @parameter
        def clear(unit: Int):
            for i in range(unit * slot_chunk, min((unit + 1) * slot_chunk, capacity)):
                slots[i] = -1
        host_parallelize(clear, units)
        var row_slots = List[Int](unsafe_uninit_length=count if ordered else 0)
        var positions = WordPointer(unsafe_from_address=Int(row_slots.unsafe_ptr()))
        @parameter
        def insert(unit: Int):
            for i in range(unit * chunk, min((unit + 1) * chunk, count)):
                var slot = Int(score_key_hash(kp[i]) & UInt64(mask))
                while True:
                    var prior = Atomic.load[ordering=Ordering.RELAXED](slots + slot)
                    if prior < 0:
                        var expected = Int64(-1)
                        if Atomic.compare_exchange[success_ordering=Ordering.RELAXED, failure_ordering=Ordering.RELAXED](slots + slot, expected, Int64(i)):
                            break
                        prior = expected
                        if prior < 0:
                            continue
                    if score_key_compare(kp[Int(prior)], kp[i]) == 0:
                        if keep_last:
                            _ = Atomic[DType.int64].max(slots + slot, Int64(i))
                        else:
                            _ = Atomic[DType.int64].min(slots + slot, Int64(i))
                        break
                    slot = (slot + 1) & mask
                if ordered:
                    positions[i] = slot
        host_parallelize(insert, units)
        if not ordered:
            _ = row_slots^
            return
        var counts = List[Int](length=units, fill=0)
        var cp = WordPointer(unsafe_from_address=Int(counts.unsafe_ptr()))
        @parameter
        def tally(unit: Int):
            var total = 0
            for i in range(unit * chunk, min((unit + 1) * chunk, count)):
                total += Int(slots[positions[i]] == Int64(i))
            cp[unit] = total
        host_parallelize(tally, units)
        var distinct = 0
        for unit in range(units):  # task summaries, not label data
            var amount = counts[unit]
            counts[unit] = distinct
            distinct += amount
        self.sorted_rows = List[Int](unsafe_uninit_length=distinct)
        var rows = WordPointer(unsafe_from_address=Int(self.sorted_rows.unsafe_ptr()))
        @parameter
        def compact(unit: Int):
            var at = cp[unit]
            for i in range(unit * chunk, min((unit + 1) * chunk, count)):
                if slots[positions[i]] == Int64(i):
                    rows[at] = i
                    at += 1
        host_parallelize(compact, units)
        _sort_distinct(kp, self.sorted_rows)
        var sorted_rows = WordPointer(unsafe_from_address=Int(self.sorted_rows.unsafe_ptr()))
        var codes = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(self.codes.unsafe_ptr()))
        var code_chunk = host_predict_chunk(max(1, distinct), units)
        @parameter
        def number(unit: Int):
            for i in range(unit * code_chunk, min((unit + 1) * code_chunk, distinct)):
                codes[sorted_rows[i]] = Int32(i)
        host_parallelize(number, units)
        _ = row_slots^
        _ = counts^

    def lookup(self, key: ScoreLabelKey, sorted_code: Bool = True) -> Int:
        var keys = KeyPointer(unsafe_from_address=self.keys)
        var mask = len(self.slots) - 1
        var slot = Int(score_key_hash(key) & UInt64(mask))
        while self.slots[slot] >= 0:
            var row = Int(self.slots[slot])
            if score_key_compare(keys[row], key) == 0:
                return Int(self.codes[row]) if sorted_code else row
            slot = (slot + 1) & mask
        return -1


def score_integer_keys(source: Int, count: Int) -> List[ScoreLabelKey]:
    var result = List[ScoreLabelKey](unsafe_uninit_length=count)
    var dst = KeyPointer(unsafe_from_address=Int(result.unsafe_ptr()))
    var words = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=source)
    var units = host_predict_task_count_work(max(1, count), max(1, count))
    var chunk = host_predict_chunk(max(1, count), units)
    @parameter
    def task(unit: Int):
        for i in range(unit * chunk, min((unit + 1) * chunk, count)):
            (dst + i).unsafe_write(ScoreLabelKey(SCORE_INT, words[i], 0, 0, False))
    host_parallelize(task, units)
    return result^


def score_key_join(a: List[ScoreLabelKey], b: List[ScoreLabelKey]) -> List[ScoreLabelKey]:
    var split = len(a)
    var count = split + len(b)
    var result = List[ScoreLabelKey](unsafe_uninit_length=count)
    var dst = KeyPointer(unsafe_from_address=Int(result.unsafe_ptr()))
    var ap = KeyPointer(unsafe_from_address=Int(a.unsafe_ptr()))
    var bp = KeyPointer(unsafe_from_address=Int(b.unsafe_ptr()))
    var units = host_predict_task_count_work(max(1, count), max(1, count))
    var chunk = host_predict_chunk(max(1, count), units)
    @parameter
    def task(unit: Int):
        for i in range(unit * chunk, min((unit + 1) * chunk, count)):
            if i < split:
                (dst + i).unsafe_write(ap[i].copy())
            else:
                (dst + i).unsafe_write(bp[i - split].copy())
    host_parallelize(task, units)
    _ = a
    _ = b
    return result^


def score_key_codes(index: ScoreKeyIndex, keys: List[ScoreLabelKey], destination: Int, sorted_code: Bool = True):
    var count = len(keys)
    var source = KeyPointer(unsafe_from_address=Int(keys.unsafe_ptr()))
    var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=destination)
    # An immutable index is shared only across the synchronous worker join.
    var lookup = Pointer(to=index)
    var units = host_predict_task_count_work(max(1, count), max(1, count))
    var chunk = host_predict_chunk(max(1, count), units)
    @parameter
    def task(unit: Int):
        for i in range(unit * chunk, min((unit + 1) * chunk, count)):
            dst[i] = Int32(lookup[].lookup(source[i], sorted_code))
    host_parallelize(task, units)
    _ = index
    _ = keys
