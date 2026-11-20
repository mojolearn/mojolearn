# SPDX-License-Identifier: Apache-2.0
"""The existing classification metrics signed-label encoding entry.

DENSE-RANGE ENCODE (lane fb-metrics, 2026-11-17; full board d95f805cce vs
0.8.55 on the L40S: precision/recall/f1/confusion/cohen-kappa/contingency/
mutual-info rows 2-5x slower, +5-11 ms per call at 100k labels). The radix
route (`host_unique_inverse_cast`) widens every label to 64 bits, copies it,
and sorts (key, row) pairs: about 60 bytes of fresh scratch per label (a
6 MB first-touch per 100k-label vector), eight digit counts and a pool split
per stage. When the labels' value range is at most the label count (classes
are small integers: the common case for class labels, at any n), one table
of `range` slots answers the same ORDER RULE in two reads of the caller's
buffer at its own width: the bounds, the presence marks, one scan over the
table (class id = number of present values below), one gather per row. The
outputs are the same integers (sorted distinct values, dense int32 codes),
so no bit moves. Wider ranges keep the radix route. The size rule is cost
reasoning, not a shape: the table is O(range) <= O(n), the radix scratch
O(n) words. Time it on a neighbouring shape (e.g. 30k or 300k labels, k=2
and k=50) and on a non-board dataset (e.g. covertype's 7 classes).
VERDICT 2026-11-18 (A/B ab2, owed rows metrics-dense-labels and
-neighbour, NVIDIA L40S host c6dd9ebea3e3, driver 595.112.07, 14-core
cgroup, wheel d652a4b0a1 and its B set, base and arm back to back in one
job, one run per arm, small rows the median of 20 calls): the radix arm
`-D MOJOLEARN_METRICS_FAST_DENSE_LABELS_OFF` was slower on 68 of 72 cells,
arm/base geomean 3.62, median 4.16 (precision-score taxi 1.94 -> 7.91 ms,
confusion-matrix istella 1.55 -> 7.77, f1 istella 1.98 -> 8.26, brier
istella 0.93 -> 9.24, cohen-kappa istella 1.28 -> 8.49; neighbour shape
taxi 300,000 labels: f1 1.87 -> 6.96, confusion 1.50 -> 7.39); the four
others 0.85-1.10 (accuracy, silhouette: rows the encode does not dominate).
Digests and scores equal on all 72 cells. The dense table stays the one
route for in-range signed labels and the arm is deleted (docs/TOMBSTONES.md).
Limits: NVIDIA only (AGENTS.md part 8); covertype not run."""
from core.cast_in import CastIn, CAST_KIND_SIGNED
from core.label_cast import host_unique_inverse_cast
from core.host_parallel import host_parallelize

# TOMBSTONE: MOJOLEARN_METRICS_FAST_DENSE_LABELS_OFF (LOSER: radix route 3.62x slower geomean, 68 of 72 metric cells slower, NV L40S 2026-11-18) deleted 2026-11-18 by lane/fx-verdicts; code recoverable at 0c65d61fed.
# Restore: git apply experiments/removed/MOJOLEARN_METRICS_FAST_DENSE_LABELS_OFF.patch; record in docs/TOMBSTONES.md.
comptime _DENSE_CHUNK = 65536  # labels (or table slots) per host-pool task, as core/label_encode.mojo's _UI_CHUNK
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I16P = MutPointer[Int16, MutUntrackedOrigin]
comptime _I8P = MutPointer[Int8, MutUntrackedOrigin]
comptime _IP = MutPointer[Int, MutUntrackedOrigin]


@always_inline
def _signed_at(addr: Int, item: Int, i: Int) -> Int64:
    """Label `i` of a contiguous signed buffer of `item`-byte words."""
    if item == 8:
        return _I64P(unsafe_from_address=addr)[i]
    if item == 4:
        return Int64(_I32P(unsafe_from_address=addr)[i])
    if item == 2:
        return Int64(_I16P(unsafe_from_address=addr)[i])
    return Int64(_I8P(unsafe_from_address=addr)[i])


@always_inline
def _store_signed(addr: Int, item: Int, c: Int, v: Int64):
    """Class `c` written at the labels' own width (the radix route's narrow)."""
    if item == 8:
        _I64P(unsafe_from_address=addr)[c] = v
    elif item == 4:
        _I32P(unsafe_from_address=addr)[c] = Int32(v)
    elif item == 2:
        _I16P(unsafe_from_address=addr)[c] = Int16(v)
    else:
        _I8P(unsafe_from_address=addr)[c] = Int8(v)


def score_encode_dense(addr: Int, item: Int, count: Int, classes_addr: Int, codes_addr: Int) -> Int:
    """The ORDER RULE through a value table: returns k, or -1 when the value
    range exceeds `count` (the caller takes the radix route). Every stage
    splits its labels (or table slots) across the host pool."""
    if count <= 0 or not (item == 1 or item == 2 or item == 4 or item == 8):
        return -1
    var units = (count + _DENSE_CHUNK - 1) // _DENSE_CHUNK
    # 1. bounds, per run
    var lo = List[Int64](length=units, fill=0)
    var hi = List[Int64](length=units, fill=0)
    var lo_at = Int(lo.unsafe_ptr())
    var hi_at = Int(hi.unsafe_ptr())
    def bounds(unit: Int) {imm addr, imm item, imm count, imm lo_at, imm hi_at}:
        var start = unit * _DENSE_CHUNK
        var stop = min(count, start + _DENSE_CHUNK)
        var a = _signed_at(addr, item, start)
        var b = a
        for i in range(start + 1, stop):  # native data loop (not a driver loop): labels of one bounds run
            var v = _signed_at(addr, item, i)
            a = min(a, v)
            b = max(b, v)
        _I64P(unsafe_from_address=lo_at)[unit] = a
        _I64P(unsafe_from_address=hi_at)[unit] = b
    host_parallelize(bounds, units)
    var low = lo[0]
    var high = hi[0]
    for u in range(1, units):  # driver-loop: over the bounds runs
        low = min(low, lo[u])
        high = max(high, hi[u])
    # two's-complement difference: exact for any pair of int64 values
    var span = high.cast[DType.uint64]() - low.cast[DType.uint64]()
    if span >= UInt64(count):
        return -1
    var width = Int(span) + 1
    # 2. presence marks. Concurrent runs may store the same 1 into one slot;
    # every store writes the same value, so the table is the same whatever
    # the order.
    var table = List[Int32](length=width, fill=0)
    var table_at = Int(table.unsafe_ptr())
    def mark(unit: Int) {imm addr, imm item, imm count, imm low, imm table_at}:
        var tp = _I32P(unsafe_from_address=table_at)
        var stop = min(count, (unit + 1) * _DENSE_CHUNK)
        for i in range(unit * _DENSE_CHUNK, stop):  # native data loop (not a driver loop): labels of one mark run
            tp[Int(_signed_at(addr, item, i) - low)] = 1
    host_parallelize(mark, units)
    # 3. scan: class id of each present value, and the class table
    var slots = (width + _DENSE_CHUNK - 1) // _DENSE_CHUNK
    var base = List[Int](length=slots, fill=0)
    var base_at = Int(base.unsafe_ptr())
    def present(unit: Int) {imm width, imm table_at, imm base_at}:
        var tp = _I32P(unsafe_from_address=table_at)
        var c = 0
        var stop = min(width, (unit + 1) * _DENSE_CHUNK)
        for j in range(unit * _DENSE_CHUNK, stop):  # native data loop (not a driver loop): table slots of one count run
            c += Int(tp[j])
        _IP(unsafe_from_address=base_at)[unit] = c
    host_parallelize(present, slots)
    var k = 0
    for u in range(slots):  # driver-loop: over the table runs, exclusive prefix
        var c = base[u]
        base[u] = k
        k += c
    def ids(unit: Int) {imm width, imm item, imm low, imm table_at, imm base_at, imm classes_addr}:
        var tp = _I32P(unsafe_from_address=table_at)
        var running = _IP(unsafe_from_address=base_at)[unit]
        var stop = min(width, (unit + 1) * _DENSE_CHUNK)
        for j in range(unit * _DENSE_CHUNK, stop):  # native data loop (not a driver loop): table slots of one id run
            if tp[j] != 0:
                tp[j] = Int32(running)
                _store_signed(classes_addr, item, running, low + Int64(j))
                running += 1
    host_parallelize(ids, slots)
    # 4. one code per row
    def emit(unit: Int) {imm addr, imm item, imm count, imm low, imm table_at, imm codes_addr}:
        var tp = _I32P(unsafe_from_address=table_at)
        var cp = _I32P(unsafe_from_address=codes_addr)
        var stop = min(count, (unit + 1) * _DENSE_CHUNK)
        for i in range(unit * _DENSE_CHUNK, stop):  # native data loop (not a driver loop): labels of one emit run
            cp[i] = tp[Int(_signed_at(addr, item, i) - low)]
    host_parallelize(emit, units)
    _ = lo^  # the buffers above are read and written through their addresses
    _ = hi^
    _ = table^
    _ = base^
    return k


def score_encode_cast(src: CastIn, label_kind: Int, classes_addr: Int, codes_addr: Int) -> Int:
    """`host_unique_inverse_cast(src, label_kind, ...)` for the metric doors:
    a contiguous 1-D signed integer buffer (`label_kind` 1) whose value
    range is within its length takes the dense table, everything else
    (floats, unsigned, bool, strided, wide ranges) the radix route. Same
    contract and the same outputs: k, or -2 for a NaN label."""
    if (label_kind == 1 and src.kind == CAST_KIND_SIGNED and src.cols == 1
            and src.s0 == src.itemsize):
        var dense = score_encode_dense(src.addr, src.itemsize, src.rows, classes_addr, codes_addr)
        if dense >= 0:
            return dense
    return host_unique_inverse_cast(src, label_kind, classes_addr, codes_addr)


def score_encode_signed(addr: Int, item: Int, count: Int, classes_addr: Int, codes_addr: Int) raises -> Int:
    """The ORDER RULE over `count` signed little-endian integer labels of
    `item` bytes (contiguous): the sorted distinct values written to
    `classes_addr` (items of the same width) and one dense int32 code per
    row; returns k. The host label encoder called natively: no Python call
    into the base binding's `prepare_targets` entry, the same classes and
    codes (exact integers). A value range within the label count takes the
    dense table (module docstring); any other range the radix route."""
    var src = CastIn(addr, CAST_KIND_SIGNED, item, count, 1, item, item)
    var k = score_encode_cast(src, 1, classes_addr, codes_addr)
    if k < 0:
        raise Error("RuntimeError: prepare_targets refused integer labels")
    return k
