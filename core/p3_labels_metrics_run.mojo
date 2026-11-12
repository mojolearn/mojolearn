# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN pieces of `bindings/p3_labels_metrics_native.mojo` (lane
p3-labels-metrics, 2026-11-06): the typed str-label passes over UTF-8 views
the door has already read in place (address, length per label): the FNV-1a
hash set of distinct labels, the byte-order merge sort of the k distinct
labels, the dense codes, and the class index of a lookup. Typed Mojo only,
no Python import (tools/python_free_dirs.sh). The door reads the views
through the CPython table and keeps the label list alive past every call
here; the bodies are moved unchanged, so codes and class order are unchanged
(lane rs-metrics-estimators, 2026-11-12)."""

comptime P3_TEXT_OK = 1
comptime P3_TEXT_NOT_STR = 0
comptime P3_TEXT_FALLBACK = -1


struct P3TextCodes(Movable):
    """`status` P3_TEXT_*; on P3_TEXT_OK `first[c]` is the row of the first
    label of class c (classes in Python's str order) and `codes[i]` the
    dense class of row i."""
    var status: Int
    var first: List[Int]
    var codes: List[Int32]

    def __init__(out self, status: Int):
        self.status = status
        self.first = List[Int]()
        self.codes = List[Int32]()


struct P3Utf8(Movable):
    """The UTF-8 bytes (address, length) of n str labels, read in place."""
    var addr: List[Int]
    var size: List[Int]

    def __init__(out self, n: Int):
        self.addr = List[Int](capacity=n)
        self.size = List[Int](capacity=n)


def _p3_bytes_cmp(a: Int, la: Int, b: Int, lb: Int) -> Int:
    var pa = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=a)
    var pb = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=b)
    for k in range(min(la, lb)):  # driver-loop: over one label's bytes
        var x = pa[k]
        var y = pb[k]
        if x != y:
            return -1 if x < y else 1
    if la == lb:
        return 0
    return -1 if la < lb else 1


def _p3_hash(a: Int, la: Int) -> UInt64:
    var p = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=a)
    var h = UInt64(14695981039346656037)
    for k in range(la):  # driver-loop: over one label's bytes
        h = (h ^ UInt64(p[k])) * UInt64(1099511628211)
    return h


struct P3TextSet(Movable):
    """Distinct labels by hash: `rows[d]` the first row of distinct label d,
    `slots` the open-addressing table of distinct ids (-1 empty)."""
    var rows: List[Int]
    var hashes: List[UInt64]
    var slots: List[Int]
    var mask: Int

    def __init__(out self, n: Int):
        var capacity = 16
        while capacity < 2 * n:  # driver-loop: power-of-two table size (log2 n steps)
            capacity *= 2
        self.rows = List[Int]()
        self.hashes = List[UInt64]()
        self.slots = List[Int](length=capacity, fill=-1)
        self.mask = capacity - 1

    def find_or_add(mut self, text: P3Utf8, row: Int, add: Bool) -> Int:
        """The distinct id of `text[row]`; a new id when unseen and `add`,
        else -1."""
        var a = text.addr[row]
        var la = text.size[row]
        var h = _p3_hash(a, la)
        var slot = Int(h & UInt64(self.mask))
        for _ in range(len(self.slots)):  # driver-loop: open-addressing probe (expected O(1); bounded by the table size)
            var d = self.slots[slot]
            if d < 0:
                if not add:
                    return -1
                d = len(self.rows)
                self.rows.append(row)
                self.hashes.append(h)
                self.slots[slot] = d
                return d
            if self.hashes[d] == h:
                var r = self.rows[d]
                if _p3_bytes_cmp(text.addr[r], text.size[r], a, la) == 0:
                    return d
            slot = (slot + 1) & self.mask
        return -1  # unreachable: the table is at most half full


def p3_sort_distinct(text: P3Utf8, rows: List[Int]) -> List[Int]:
    """The distinct ids 0..k-1 sorted by their bytes (bottom-up merge sort;
    the ids are distinct, so order ties cannot occur)."""
    var k = len(rows)
    var order = List[Int](capacity=k)
    for d in range(k):  # driver-loop: over the k distinct labels
        order.append(d)
    var buffer = List[Int](length=k, fill=0)
    var width = 1
    while width < k:  # driver-loop: over merge widths
        var lo = 0
        while lo < k:  # driver-loop: over merged runs
            var mid = min(lo + width, k)
            var hi = min(lo + 2 * width, k)
            var i = lo
            var j = mid
            var o = lo
            while i < mid and j < hi:  # driver-loop: over one merge
                var ri = rows[order[i]]
                var rj = rows[order[j]]
                if _p3_bytes_cmp(text.addr[rj], text.size[rj], text.addr[ri], text.size[ri]) < 0:
                    buffer[o] = order[j]
                    j += 1
                else:
                    buffer[o] = order[i]
                    i += 1
                o += 1
            while i < mid:  # driver-loop: over one merge's tail
                buffer[o] = order[i]
                i += 1
                o += 1
            while j < hi:  # driver-loop: over one merge's tail
                buffer[o] = order[j]
                j += 1
                o += 1
            lo = hi
        for s in range(k):  # driver-loop: over the k distinct labels
            order[s] = buffer[s]
        width *= 2
    return order^


def p3_text_codes_from_utf8(text: P3Utf8, n: Int) -> P3TextCodes:
    """`sorted(dict.fromkeys(labels))` and each label's dense code from the
    n labels' UTF-8 views, by hash then a sort of the k distinct labels."""
    var seen = P3TextSet(n)
    var ids = List[Int32](capacity=n)
    for i in range(n):  # driver-loop: one hash probe per label, native
        ids.append(Int32(seen.find_or_add(text, i, True)))
    var order = p3_sort_distinct(text, seen.rows)
    var k = len(order)
    var rank = List[Int32](length=k, fill=0)
    var out = P3TextCodes(P3_TEXT_OK)
    out.first = List[Int](capacity=k)
    for c in range(k):  # driver-loop: over the k sorted classes
        rank[order[c]] = Int32(c)
        out.first.append(seen.rows[order[c]])
    out.codes = List[Int32](capacity=n)
    for i in range(n):  # driver-loop: native code write per label
        out.codes.append(rank[Int(ids[i])])
    return out^


def p3_class_index(ctext: P3Utf8, k: Int, mut seen: P3TextSet, mut answer: List[Int]):
    """The distinct-class set of k str classes and, per distinct class, its
    index (a duplicated class keeps its LAST index)."""
    for c in range(k):  # driver-loop: over the k classes
        var d = seen.find_or_add(ctext, c, True)
        if d == len(answer):
            answer.append(c)
        else:
            answer[d] = c  # a duplicated class keeps its last index


def _p3_probe(seen: P3TextSet, ctext: P3Utf8, one: P3Utf8) -> Int:
    """The distinct class id of the single label in `one`, or -1."""
    var a = one.addr[0]
    var la = one.size[0]
    var h = _p3_hash(a, la)
    var slot = Int(h & UInt64(seen.mask))
    for _ in range(len(seen.slots)):  # driver-loop: open-addressing probe (expected O(1); bounded by the table size)
        var d = seen.slots[slot]
        if d < 0:
            return -1
        if seen.hashes[d] == h:
            var r = seen.rows[d]
            if _p3_bytes_cmp(ctext.addr[r], ctext.size[r], a, la) == 0:
                return d
        slot = (slot + 1) & seen.mask
    return -1
