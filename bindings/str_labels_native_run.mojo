# SPDX-License-Identifier: Apache-2.0
# RUN part of bindings/str_labels_native.mojo: typed Mojo only (no Python
# imports). The native code-point table of str labels, the distinct-label
# hashing and ordering and the lookup; the door reads the labels through the
# C API into a StrTable and releases the GIL around these calls.
from core.host_parallel import host_parallelize

# ---------------------------------------------------------------- distinct labels
# The ORDER RULE for str labels (the distinct values in code-point order, each
# class the FIRST label of its value, each label's code its class's rank) in
# O(n) hashing plus a sort of the k DISTINCT classes only (round2-fixes,
# 2026-11-11; perf audit 2 B): every label is hashed once across the host
# pool, each task keeps its row chunk's distinct labels (the earliest row of
# each), ONE merge of the per-task sets in chunk order keeps each class's
# global first row, the k classes are sorted, and the codes are written across
# the pool. The former route merge-sorted all n labels on one thread.

comptime _UNIQUE_CHUNK = 65536  # labels per hashing task: each task hashes and probes tens of thousands of labels, so the pool's dispatch cost is amortized whatever the label count
comptime _FNV_OFFSET = UInt64(0xCBF29CE484222325)
comptime _FNV_PRIME = UInt64(0x100000001B3)

comptime _LOOKUP_CHUNK = 4096  # labels per lookup task: each task runs thousands of binary searches, so the pool's dispatch cost is amortized whatever the label count


struct StrTable(Movable):
    """The code points of n labels, one contiguous UInt32 run each."""
    var points: List[UInt32]
    var starts: List[Int]
    var lengths: List[Int]

    def __init__(out self):
        self.points = List[UInt32]()
        self.starts = List[Int]()
        self.lengths = List[Int]()

    def compare(self, a: Int, b: Int) -> Int:
        return str_cross(self, a, self, b)


def str_cross(rows: StrTable, i: Int, known: StrTable, j: Int) -> Int:
    """Code-point order (Python's str order) of label i of `rows` against
    label j of `known`: -1, 0 or 1. The one comparison every str-label path
    uses."""
    var la = rows.lengths[i]
    var lb = known.lengths[j]
    var sa = rows.starts[i]
    var sb = known.starts[j]
    for q in range(min(la, lb)):  # native data loop, no CPython (not a driver loop): over one label's code points (native)
        var x = rows.points[sa + q]
        var y = known.points[sb + q]
        if x != y:
            return -1 if x < y else 1
    if la == lb:
        return 0
    return -1 if la < lb else 1


def _sorted(table: StrTable, n: Int) -> List[Int]:
    # A stable bottom-up merge sort of 0..n-1 by code-point order: equal
    # labels keep their input order, so each run starts at its first label.
    var rows = List[Int](capacity=n)
    for i in range(n):  # native data loop, no CPython (not a driver loop): over the label indices
        rows.append(i)
    # The identity rows: the sorted positions are the sorted label indices.
    return _sort_by_rows(table, rows)


def _sort_by_rows(table: StrTable, rows: List[Int]) -> List[Int]:
    """The positions 0..len(rows)-1 ordered by the code points of their rows
    of `table` (a permutation of positions, not rows): a stable bottom-up
    merge sort (equal labels keep their order)."""
    var n = len(rows)
    var order = List[Int](capacity=n)
    for i in range(n):  # native data loop, no CPython (not a driver loop): over the positions
        order.append(i)
    var buffer = List[Int](length=n, fill=0)
    var width = 1
    while width < n:  # driver-loop: over merge widths
        var lo = 0
        while lo < n:  # driver-loop: over merged runs
            var mid = min(lo + width, n)
            var hi = min(lo + 2 * width, n)
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:  # driver-loop: over one merge
                if table.compare(rows[order[j]], rows[order[i]]) < 0:
                    buffer[k] = order[j]
                    j += 1
                else:
                    buffer[k] = order[i]
                    i += 1
                k += 1
            while i < mid:  # driver-loop: over one merge's tail
                buffer[k] = order[i]
                i += 1
                k += 1
            while j < hi:  # driver-loop: over one merge's tail
                buffer[k] = order[j]
                j += 1
                k += 1
            lo = hi
        for t in range(n):  # native data loop, no CPython (not a driver loop): over the positions
            order[t] = buffer[t]
        width *= 2
    return order^


def _str_hash(table: StrTable, i: Int) -> UInt64:
    # FNV-1a over label i's code points and its length.
    var h = _FNV_OFFSET
    var s = table.starts[i]
    for q in range(table.lengths[i]):  # native data loop, no CPython (not a driver loop): over one label's code points (native)
        h = (h ^ UInt64(table.points[s + q])) * _FNV_PRIME
    return (h ^ UInt64(table.lengths[i])) * _FNV_PRIME


def _pow2_at_least(n: Int) -> Int:
    var c = 1
    while c < n:  # driver-loop: over powers of two (a size, not data)
        c *= 2
    return c


def _str_unique_native(table: StrTable, n: Int, codes_address: Int) -> List[Int]:
    """`str_unique_into`'s native work (no Python object touched)."""
    if n <= 0:
        return List[Int]()
    var units = (n + _UNIQUE_CHUNK - 1) // _UNIQUE_CHUNK
    var local_cap = _pow2_at_least(2 * min(n, _UNIQUE_CHUNK))  # load at most one half: a probe always ends
    var hashes = List[UInt64](length=n, fill=0)
    var local_rows = List[Int](length=units * local_cap, fill=-1)
    var tp = UnsafePointer(to=table)
    var hashes_address = Int(hashes.unsafe_ptr())
    var rows_address = Int(local_rows.unsafe_ptr())
    def dedupe(unit: Int) {imm tp, imm n, imm local_cap, imm hashes_address, imm rows_address}:
        var hp = MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=hashes_address)
        var slots = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=rows_address)
        var base = unit * local_cap
        var mask = local_cap - 1
        var stop = min(n, (unit + 1) * _UNIQUE_CHUNK)
        for i in range(unit * _UNIQUE_CHUNK, stop):  # native data loop, no CPython (not a driver loop): labels of one hashing chunk (native, no Python)
            var h = _str_hash(tp[], i)
            hp[i] = h
            var s = Int(h & UInt64(mask))
            while True:  # driver-loop: over one open-addressing probe (native)
                var r = slots[base + s]
                if r < 0:
                    slots[base + s] = i  # the chunk's earliest row of this label
                    break
                if hp[r] == h and tp[].compare(r, i) == 0:
                    break
                s = (s + 1) & mask
    host_parallelize(dedupe, units)
    _ = hashes  # written by every task above
    # ONE merge of the per-task sets, in chunk order: the first insertion of
    # a class comes from its earliest chunk, whose slot holds its earliest
    # row there, so each class keeps its global first row.
    var total = 0
    for u in range(units * local_cap):  # native data loop, no CPython (not a driver loop): over the per-task slots (native)
        if local_rows[u] >= 0:
            total += 1
    var cap = _pow2_at_least(2 * total + 2)
    var mask = cap - 1
    var global_rows = List[Int](length=cap, fill=-1)
    var global_ids = List[Int](length=cap, fill=-1)
    var reps = List[Int](capacity=total)
    for u in range(units * local_cap):  # native data loop, no CPython (not a driver loop): the one merge of the per-task distinct sets, in chunk order (native)
        var r = local_rows[u]
        if r < 0:
            continue
        var h = hashes[r]
        var g = Int(h & UInt64(mask))
        while True:  # driver-loop: over one open-addressing probe (native)
            var q = global_rows[g]
            if q < 0:
                global_rows[g] = r
                global_ids[g] = len(reps)
                reps.append(r)
                break
            if hashes[q] == h and table.compare(q, r) == 0:
                break
            g = (g + 1) & mask
    _ = local_rows^
    # Sort only the k distinct classes (ids are positions in `reps`);
    # rank[id] is the class index.
    var order = _sort_by_rows(table, reps)
    var k = len(reps)
    var firsts = List[Int](length=k, fill=0)
    var rank = List[Int](length=k, fill=0)
    for c in range(k):  # driver-loop: over the distinct classes (native)
        firsts[c] = reps[order[c]]
        rank[order[c]] = c
    var rows_at = Int(global_rows.unsafe_ptr())
    var ids_at = Int(global_ids.unsafe_ptr())
    var rank_at = Int(rank.unsafe_ptr())
    def assign(unit: Int) {imm tp, imm n, imm mask, imm hashes_address, imm rows_at, imm ids_at, imm rank_at,
                           imm codes_address}:
        var hp = MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=hashes_address)
        var gr = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=rows_at)
        var gi = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=ids_at)
        var rk = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=rank_at)
        var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=codes_address)
        var stop = min(n, (unit + 1) * _UNIQUE_CHUNK)
        for i in range(unit * _UNIQUE_CHUNK, stop):  # native data loop, no CPython (not a driver loop): labels of one code chunk (native, no Python)
            var h = hp[i]
            var g = Int(h & UInt64(mask))
            while True:  # driver-loop: over one open-addressing probe (native)
                var q = gr[g]
                if q < 0:
                    dst[i] = Int32(-1)  # unreachable: every label's class was merged
                    break
                if hp[q] == h and tp[].compare(q, i) == 0:
                    dst[i] = Int32(rk[gi[g]])
                    break
                g = (g + 1) & mask
    host_parallelize(assign, units)
    _ = hashes  # read by every task above
    _ = global_rows
    _ = global_ids
    _ = rank
    _ = table
    return firsts^


@fieldwise_init
struct StrClasses(Movable):
    """Fitted str classes for lookup: the distinct classes in code-point
    order (`distinct`, rows of the classes table) and the index each answers
    with (`index`: a duplicated class keeps its LAST index, as
    `{c: i for i, c in enumerate(classes)}` does)."""
    var distinct: List[Int]
    var index: List[Int]


def str_classes(known: StrTable, k: Int) -> StrClasses:
    """The lookup form of the k classes of `known` (`StrClasses`)."""
    var order = _sorted(known, k)
    var distinct = List[Int]()
    var index = List[Int]()
    for t in range(k):  # driver-loop: over the sorted classes
        if t == 0 or known.compare(order[t - 1], order[t]) != 0:
            distinct.append(order[t])
            index.append(order[t])
        else:
            index[len(index) - 1] = max(index[len(index) - 1], order[t])
    return StrClasses(distinct^, index^)


def str_find(rows: StrTable, i: Int, known: StrTable, classes: StrClasses) -> Int:
    """Label i of `rows` looked up among the classes of `known`: its class
    index, or -1. A binary search of the distinct sorted classes."""
    var lo = 0
    var hi = len(classes.distinct)
    while lo < hi:  # driver-loop: over a binary search of the classes
        var mid = (lo + hi) // 2
        var c = str_cross(rows, i, known, classes.distinct[mid])
        if c == 0:
            return classes.index[mid]
        if c > 0:
            lo = mid + 1
        else:
            hi = mid
    return -1


@fieldwise_init
struct StrUnseen(Movable):
    """The labels a lookup did not find: the rows of the first five (label
    order) and their count."""
    var rows: List[Int]
    var count: Int


def _str_codes_native(rows: StrTable, known: StrTable, classes: StrClasses, address: Int,
                      as_float: Bool) -> StrUnseen:
    var n = len(rows.starts)
    var tp = UnsafePointer(to=rows)
    var kp = UnsafePointer(to=known)
    var cp = UnsafePointer(to=classes)
    var units = (n + _LOOKUP_CHUNK - 1) // _LOOKUP_CHUNK
    def lookup(unit: Int) {imm tp, imm kp, imm cp, imm n, imm address, imm as_float}:
        var stop = min(n, (unit + 1) * _LOOKUP_CHUNK)
        for i in range(unit * _LOOKUP_CHUNK, stop):  # native data loop (not a driver loop): labels of one lookup chunk (native, no Python)
            var found = str_find(tp[], i, kp[], cp[])
            if as_float:
                MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)[i] = Float32(found)
            else:
                MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)[i] = Int32(found)
    host_parallelize(lookup, units)
    _ = rows  # read by every lookup task above
    _ = known
    _ = classes
    var first = List[Int]()
    var count = 0
    for i in range(n):  # native data loop (not a driver loop): native scan for unseen labels (no CPython call)
        var missing = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)[i] < 0 if as_float else MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)[i] < 0
        if missing:
            if count < 5:
                first.append(i)
            count += 1
    return StrUnseen(first^, count)
