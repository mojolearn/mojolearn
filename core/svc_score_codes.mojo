# SPDX-License-Identifier: Apache-2.0
"""Native validation of the existing Python-style class index range, and the
code-space helpers that keep SVC score's CPython work per used class instead
of per row (bindings/svc_score_boundary.mojo)."""
from std.builtin.sort import sort
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk


def svc_score_codes_valid(codes: List[Int64], classes: Int) -> Bool:
    var count = len(codes)
    if count == 0:
        return True
    var units = host_predict_task_count_work(count, count)
    var chunk = host_predict_chunk(count, units)
    var failures = List[Int](length=units, fill=0)
    var source = Pointer[Int64, ImmUntrackedOrigin](unsafe_from_address=Int(codes.unsafe_ptr()))
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm source, imm flags, imm count, imm classes, imm chunk}:
        var failed = False
        for row in range(unit * chunk, min((unit + 1) * chunk, count)):
            failed = failed or source[row] < -Int64(classes) or source[row] >= Int64(classes)
        flags[unit] = Int(failed)
    host_parallelize(task, units)
    var result = True
    for flag in failures:  # metadata: one status per independent partition
        result = result and flag == 0
    _ = codes
    return result


struct SvcUsedClasses(Movable):
    """The classes a prediction actually uses. `order` holds their
    (non-negative) class indices in order of first appearance among the rows;
    `slots` holds, per row, the position of that row's class in `order`."""
    var order: List[Int]
    var slots: List[Int32]

    def __init__(out self, rows: Int):
        self.order = List[Int]()
        self.slots = List[Int32](length=rows, fill=0)


def svc_used_classes(codes: List[Int64], classes: Int) raises -> SvcUsedClasses:
    """The used classes of `codes` (already admitted by
    `svc_score_codes_valid`; a negative code counts from the end, as Python
    indexing does). Rows split across the host pool; the per-class reduce is
    over the units' first rows."""
    var count = len(codes)
    var result = SvcUsedClasses(count)
    if count == 0:
        return result^
    if classes <= 0:
        raise Error("IndexError: list index out of range")
    if count > 2147483647 or count > (Int(1) << 62) // classes:
        raise Error("mojolearn SVC score: label dimensions exceed native indexing")
    var units = host_predict_task_count_work(count, count)
    var chunk = host_predict_chunk(count, units)
    # One first-row table per unit; `count` marks a class the unit never saw.
    var firsts = List[Int](length=units * classes, fill=count)
    var source = Pointer[Int64, ImmUntrackedOrigin](unsafe_from_address=Int(codes.unsafe_ptr()))
    var fp = firsts.unsafe_ptr()
    def scan(unit: Int) {imm source, imm fp, imm count, imm classes, imm chunk}:
        var base = unit * classes
        for row in range(unit * chunk, min((unit + 1) * chunk, count)):
            var c = Int(source[row])
            if c < 0:
                c += classes
            if fp[base + c] == count:
                fp[base + c] = row
    host_parallelize(scan, units)
    var first = List[Int](length=classes, fill=count)
    var first_p = first.unsafe_ptr()
    var class_units = host_predict_task_count_work(classes, classes * units)
    var class_chunk = host_predict_chunk(classes, class_units)
    def reduce(unit: Int) {imm fp, imm first_p, imm units, imm classes, imm class_chunk, imm count}:
        for c in range(unit * class_chunk, min((unit + 1) * class_chunk, classes)):
            var earliest = count
            for u in range(units):  # small-loop(host pool units: the earliest unit that saw the class)
                if fp[u * classes + c] < earliest:
                    earliest = fp[u * classes + c]
                    break
            first_p[c] = earliest
    host_parallelize(reduce, class_units)
    # Units own ascending row ranges, so the first unit that saw a class holds
    # its first row. Order the used classes by that row: the rows are
    # distinct, so `row * classes + class` sorts by row alone.
    var keys = List[Int]()
    for c in range(classes):  # metadata: collect the used classes (at most one word per class)
        if first[c] < count:
            keys.append(first[c] * classes + c)
    sort(keys)
    var position = List[Int32](length=classes, fill=-1)
    for i in range(len(keys)):  # metadata: one position per used class
        var c = keys[i] % classes
        result.order.append(c)
        position[c] = Int32(i)
    var pp = position.unsafe_ptr()
    var sp = result.slots.unsafe_ptr()
    def place(unit: Int) {imm source, imm pp, imm sp, imm count, imm classes, imm chunk}:
        for row in range(unit * chunk, min((unit + 1) * chunk, count)):
            var c = Int(source[row])
            if c < 0:
                c += classes
            sp[row] = pp[c]
    host_parallelize(place, units)
    _ = codes
    _ = firsts
    _ = first
    _ = position
    return result^


def svc_gather_words(class_words: List[UInt64], slots: List[Int32]) raises -> List[UInt64]:
    """One word per row: `class_words[slots[row]]` (the rows split across the
    host pool). `slots` come from `svc_used_classes`, so every slot indexes
    `class_words`."""
    var count = len(slots)
    var result = List[UInt64](length=count, fill=0)
    if count == 0:
        return result^
    var units = host_predict_task_count_work(count, count)
    var chunk = host_predict_chunk(count, units)
    var wp = Pointer[UInt64, ImmUntrackedOrigin](unsafe_from_address=Int(class_words.unsafe_ptr()))
    var sp = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=Int(slots.unsafe_ptr()))
    var dst = result.unsafe_ptr()
    def gather(unit: Int) {imm wp, imm sp, imm dst, imm count, imm chunk}:
        for row in range(unit * chunk, min((unit + 1) * chunk, count)):
            dst[row] = wp[Int(sp[row])]
    host_parallelize(gather, units)
    _ = class_words
    _ = slots
    return result^
