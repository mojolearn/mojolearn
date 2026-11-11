# SPDX-License-Identifier: Apache-2.0
"""Owned hierarchy COO preparation from admitted sparse storage.

This preserves stored entry order, duplicates, zeros and source-word casts.
Independent input/output spans use the existing host grain/core policy. Only
the bounded task summaries are folded by the calling thread. No Python values
or hierarchy numerical kernels participate in this storage conversion.
"""
from std.memory import bitcast
from core.classical_pandas_impl import pd_raw_at
from core.cast_in import cast_f32_bits
from core.host_sparse_impl import host_sparse_index
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk

comptime _Words = MutPointer[Float32, MutUntrackedOrigin]
comptime _Ints = MutPointer[Int, MutUntrackedOrigin]


@fieldwise_init
struct ConnectivityStorage(ImplicitlyCopyable, Movable):
    var address: Int
    var size0: Int
    var size1: Int
    var size2: Int
    var stride0: Int
    var stride1: Int
    var stride2: Int
    var width: Int
    var kind: Int
    var native: Bool

    def index(self, at: Int) -> Int:
        return host_sparse_index(self.address, self.stride0, self.width, self.native, at)

    def value(self, offset: Int) -> Float32:
        return bitcast[DType.float32](cast_f32_bits(pd_raw_at(self.address, offset, self.width, self.native), self.kind, self.width))


@fieldwise_init
struct ConnectivityEdge(ImplicitlyCopyable, Movable):
    var row: Float32
    var col: Float32
    var value: Float32


struct HierarchyConnectivityPlan(Movable):
    var words: List[Float32]
    var edges: Int

    def __init__(out self, edges: Int = 0):
        self.edges = edges
        self.words = List[Float32](unsafe_uninit_length=3 * edges)

    def address(self) -> Int:
        return Int(self.words.unsafe_ptr()) if self.edges else 0


def connectivity_edge(row: Int, col: Int, word: UInt64, kind: Int, width: Int, n: Int) raises -> ConnectivityEdge:
    # DOK/LIL unpack already checks coordinates and dtype conversion in source
    # order. This last scalar normalization is part of that admission; only
    # the later transposition of admitted records is independent work.
    if row < 0 or col < 0 or row >= n or col >= n:
        raise Error("AgglomerativeClustering: a connectivity edge is outside [0, n)")
    return ConnectivityEdge(Float32(row), Float32(col), bitcast[DType.float32](cast_f32_bits(word, kind, width)))


def connectivity_pack(records: List[ConnectivityEdge]) -> HierarchyConnectivityPlan:
    var count = len(records)
    var result = HierarchyConnectivityPlan(count)
    if count == 0:
        return result^
    var dst = _Words(unsafe_from_address=result.address())
    var src = Pointer[ConnectivityEdge, ImmUntrackedOrigin](unsafe_from_address=Int(records.unsafe_ptr()))
    var units = host_predict_task_count_work(count, 3 * count)
    var chunk = host_predict_chunk(count, units)
    def task(unit: Int) {imm count, imm dst, imm src, imm chunk}:
        for at in range(unit * chunk, min((unit + 1) * chunk, count)):
            dst[at] = src[at].row
            dst[count + at] = src[at].col
            dst[2 * count + at] = src[at].value
    host_parallelize(task, units)
    _ = records
    return result^


def _first_pointer_error(pointers: ConnectivityStorage, outer: Int, count: Int) -> Int:
    if outer == 0:
        return outer
    var units = host_predict_task_count_work(outer, outer)
    var chunk = host_predict_chunk(outer, units)
    var failures = List[Int](length=units, fill=outer)
    var fp = _Ints(unsafe_from_address=Int(failures.unsafe_ptr()))
    def task(unit: Int) {imm pointers, imm outer, imm count, imm chunk, imm fp}:
        for major in range(unit * chunk, min((unit + 1) * chunk, outer)):
            var start = pointers.index(major)
            var end = pointers.index(major + 1)
            if end < start or end > count:
                fp[unit] = major
                return
    host_parallelize(task, units)
    var first = outer
    for unit in range(units):  # driver-loop: at most one ordered summary per host task
        first = min(first, failures[unit])
    return first


def _major(pointers: ConnectivityStorage, outer: Int, entry: Int) -> Int:
    # One upper-bound search per task, then a linear cursor through that task's
    # pointer interval. Empty compressed rows keep their original positions.
    var lo = 0
    var hi = outer
    while lo < hi:
        var middle = (lo + hi) // 2
        if pointers.index(middle + 1) <= entry:
            lo = middle + 1
        else:
            hi = middle
    return lo


def connectivity_compressed(data: ConnectivityStorage, indices: ConnectivityStorage,
                            pointers: ConnectivityStorage, n: Int, column: Bool,
                            blocked: Bool) raises -> HierarchyConnectivityPlan:
    var br = data.size1 if blocked else 1
    var bc = data.size2 if blocked else 1
    if br < 1 or bc < 1 or n % br != 0 or n % bc != 0:
        raise Error("sparse connectivity blocks must evenly cover its shape")
    var outer = n // br
    var inner = n // bc
    if pointers.size0 != outer + 1 or pointers.index(0) != 0:
        raise Error("sparse connectivity pointer count or origin is invalid")
    var count = min(data.size0, indices.size0)
    var bad_major = _first_pointer_error(pointers, outer, count)
    # Baseline visits each row's pointer before that row's indices. A bad
    # index in an earlier complete row must win over a later bad pointer.
    # Only the proven prefix is read, so malformed pointers cannot escape
    # either source storage while that error priority is preserved.
    var complete = pointers.index(bad_major)
    var block_words = br * bc
    var edges = complete * block_words
    var result = HierarchyConnectivityPlan(edges)
    if edges == 0:
        if bad_major != outer:
            raise Error("sparse connectivity pointers exceed stored data or are not ordered")
        return result^
    var dst = _Words(unsafe_from_address=result.address())
    var units = host_predict_task_count_work(edges, 3 * edges)
    var chunk = host_predict_chunk(edges, units)
    var failures = List[Int](length=units, fill=0)
    var fp = _Ints(unsafe_from_address=Int(failures.unsafe_ptr()))
    def task(unit: Int) {imm data, imm indices, imm pointers, imm bad_major, imm inner, imm edges, imm br, imm bc, imm block_words, imm column, imm blocked, imm dst, imm chunk, imm fp}:
        var lo = unit * chunk
        var hi = min((unit + 1) * chunk, edges)
        if lo >= hi:
            return
        var major = _major(pointers, bad_major, lo // block_words)
        var active_entry = -1
        var minor = 0
        for at in range(lo, hi):
            var entry = at // block_words
            if entry != active_entry:
                while entry >= pointers.index(major + 1):
                    major += 1
                minor = indices.index(entry)
                if minor < 0 or minor >= inner:
                    fp[unit] = 1
                    return
                active_entry = entry
            var r = (at % block_words) // bc
            var c = at % bc
            var row = minor if column else major * br + r
            var col = major if column else minor * bc + c
            var offset = entry * data.stride0
            if blocked:
                offset += r * data.stride1 + c * data.stride2
            dst[at] = Float32(row)
            dst[edges + at] = Float32(col)
            dst[2 * edges + at] = data.value(offset)
    host_parallelize(task, units)
    for unit in range(units):  # driver-loop: one index refusal flag per host task
        if failures[unit]:
            raise Error("sparse connectivity index exceeds its shape")
    if bad_major != outer:
        raise Error("sparse connectivity pointers exceed stored data or are not ordered")
    return result^


def connectivity_coo(rows: ConnectivityStorage, cols: ConnectivityStorage,
                     data: ConnectivityStorage, n: Int) raises -> HierarchyConnectivityPlan:
    var count = data.size0
    if rows.size0 != count or cols.size0 != count:
        raise Error("sparse connectivity coordinate counts differ from stored data")
    var result = HierarchyConnectivityPlan(count)
    if count == 0:
        return result^
    var dst = _Words(unsafe_from_address=result.address())
    var units = host_predict_task_count_work(count, 3 * count)
    var chunk = host_predict_chunk(count, units)
    var failures = List[Int](length=units, fill=0)
    var fp = _Ints(unsafe_from_address=Int(failures.unsafe_ptr()))
    def task(unit: Int) {imm rows, imm cols, imm data, imm n, imm count, imm dst, imm chunk, imm fp}:
        for at in range(unit * chunk, min((unit + 1) * chunk, count)):
            var row = rows.index(at)
            var col = cols.index(at)
            if row < 0 or col < 0 or row >= n or col >= n:
                fp[unit] = 1
                return
            dst[at] = Float32(row)
            dst[count + at] = Float32(col)
            dst[2 * count + at] = data.value(at * data.stride0)
    host_parallelize(task, units)
    for unit in range(units):  # driver-loop: one refusal flag per host task
        if failures[unit]:
            raise Error("AgglomerativeClustering: a connectivity edge is outside [0, n)")
    return result^


def _diagonal_begin(shift: Int) -> Int:
    return max(0, shift)


def _diagonal_end(shift: Int, n: Int, columns: Int) -> Int:
    var end = min(n, columns)
    # Avoid overflow in n + shift for extreme signed index words. A negative
    # shift at or below -n has no square-matrix entries, as in the old loop.
    if shift < 0:
        return 0 if shift <= -n else min(end, n + shift)
    return end


def connectivity_dia(offsets: ConnectivityStorage, data: ConnectivityStorage,
                     n: Int) raises -> HierarchyConnectivityPlan:
    var count = offsets.size0
    if count != data.size0:
        raise Error("sparse connectivity diagonal count differs from stored data")
    if count == 0:
        return HierarchyConnectivityPlan()
    var units = host_predict_task_count_work(count, count)
    var chunk = host_predict_chunk(count, units)
    var starts = List[Int](unsafe_uninit_length=count + 1)
    var totals = List[Int](length=units, fill=0)
    var sp = _Ints(unsafe_from_address=Int(starts.unsafe_ptr()))
    var tp = _Ints(unsafe_from_address=Int(totals.unsafe_ptr()))
    def measure(unit: Int) {imm offsets, imm data, imm n, imm count, imm chunk, imm sp, imm tp}:
        var total = 0
        for diagonal in range(unit * chunk, min((unit + 1) * chunk, count)):
            var shift = offsets.index(diagonal)
            sp[diagonal] = total
            total += max(0, _diagonal_end(shift, n, data.size1) - _diagonal_begin(shift))
        tp[unit] = total
    host_parallelize(measure, units)
    var edges = 0
    for unit in range(units):  # driver-loop: ordered prefix of at most one count per task
        var amount = totals[unit]
        totals[unit] = edges
        edges += amount
    starts[count] = edges
    def prefix(unit: Int) {imm count, imm chunk, imm sp, imm tp}:
        for diagonal in range(unit * chunk, min((unit + 1) * chunk, count)):
            sp[diagonal] += tp[unit]
    host_parallelize(prefix, units)
    var result = HierarchyConnectivityPlan(edges)
    if edges == 0:
        return result^
    var dst = _Words(unsafe_from_address=result.address())
    var output_units = host_predict_task_count_work(edges, 3 * edges)
    var output_chunk = host_predict_chunk(edges, output_units)
    def convert(unit: Int) {imm offsets, imm data, imm count, imm edges, imm sp, imm dst, imm output_chunk}:
        var lo = unit * output_chunk
        var hi = min((unit + 1) * output_chunk, edges)
        if lo >= hi:
            return
        var left = 0
        var right = count
        while left < right:
            var middle = (left + right) // 2
            if sp[middle + 1] <= lo:
                left = middle + 1
            else:
                right = middle
        var diagonal = left
        for at in range(lo, hi):
            while at >= sp[diagonal + 1]:
                diagonal += 1
            var shift = offsets.index(diagonal)
            var col = _diagonal_begin(shift) + at - sp[diagonal]
            var row = col - shift
            dst[at] = Float32(row)
            dst[edges + at] = Float32(col)
            dst[2 * edges + at] = data.value(diagonal * data.stride0 + col * data.stride1)
    host_parallelize(convert, output_units)
    _ = starts^
    _ = totals^
    return result^
