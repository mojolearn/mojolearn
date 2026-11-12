# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# The "already admitted in this operation" registry: the TYPED table (RUN;
# no Python). Split from bindings/admitted_registry.mojo (lane
# resident-pin-tables, 2026-11-12): the held matrices and their parents are
# pinned in the door-side store of bindings/py_admitted_registry.mojo, in
# lists aligned row for row with this table; here each row keeps only words:
# the matrix's id() word, its parent's id() word (0: none), the data address,
# the shape, the holding thread and the finite verdict, plus the open-scope
# depth of each thread.
#
# A row is listed only while the door pins its matrix (and parent), so an
# id() word compared here can never belong to another object. Matching is by
# identity word AND calling thread; a scope's end drops every row the calling
# thread made inside it and returns their positions (descending) so the door
# drops the same positions of its pin lists. Every call runs under the GIL
# (door code), which serialises the table with its pins.
#
# ONE REGISTRY PER PROCESS (core/shared_slot.mojo): every extension module of
# every wheel that imports this file reads the same table. Not tier-keyed:
# the held verdict ("finite float32 C-order") is the same in every mode.
# Numerics: none.
from std.ffi import external_call

from core.shared_slot import SharedSlot
from core.admitted_held_plan import HeldBufferDescriptor

#: Layout version of `AdmittedTable` (core/shared_slot.mojo rule). V4 (lane
#: resident-pin-tables): words only; the Python pins moved to the door.
comptime _ADMITTED_LAYOUT = 4
comptime _ADMITTED_NAME = "MojolearnAdmittedRegistryV4"


struct AdmittedTable(Defaultable, Movable):
    # id() of each held matrix (pinned by the door at the same position)
    var items: List[Int]
    var addresses: List[Int]
    var rows: List[Int]
    var cols: List[Int]
    # the thread that held each row
    var threads: List[Int]
    # 1: tested finite; 0: deferred (not tested on the host yet)
    var verdicts: List[Int]
    # id() of the held matrix each row was gathered out of, or 0
    var parents: List[Int]
    # the open-scope depth of each thread that has one open
    var depth_threads: List[Int]
    var depths: List[Int]

    def __init__(out self):
        self.items = List[Int]()
        self.addresses = List[Int]()
        self.rows = List[Int]()
        self.cols = List[Int]()
        self.threads = List[Int]()
        self.verdicts = List[Int]()
        self.parents = List[Int]()
        self.depth_threads = List[Int]()
        self.depths = List[Int]()


comptime ADMITTED_TABLE = SharedSlot[AdmittedTable, _ADMITTED_NAME, _ADMITTED_LAYOUT]


def _table() raises -> UnsafePointer[AdmittedTable, MutAnyOrigin]:
    var g = ADMITTED_TABLE.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()
    return g


def admitted_thread() -> Int:
    # pthread_t: an unsigned long on Linux, a pointer on macOS; either way a
    # nonzero word naming the live calling thread (core/device_plan.mojo).
    return Int(external_call["pthread_self", UInt]())


def _depth_index(g: UnsafePointer[AdmittedTable, MutAnyOrigin], me: Int) -> Int:
    for i in range(len(g[].depth_threads)):  # small-loop(depth_threads: threads with an open scope, a few)
        if g[].depth_threads[i] == me:
            return i
    return -1


def admitted_run_scope_begin() raises -> Int:
    """Open a hold scope on the calling thread; returns the mark (the
    thread's held count)."""
    var g = _table()
    var me = admitted_thread()
    var d = _depth_index(g, me)
    if d < 0:
        g[].depth_threads.append(me)
        g[].depths.append(1)
    else:
        g[].depths[d] += 1
    var n = 0
    for k in range(len(g[].threads)):  # small-loop(threads: the held matrices of the running calls, a few)
        if g[].threads[k] == me:
            n += 1
    return n


def admitted_run_scope_end(mark: Int) raises -> List[Int]:
    """Close the scope `mark` opened: drop every row the calling thread held
    inside it (other threads' rows stay). Returns the dropped positions in
    descending order: the door drops the same positions of its pin lists."""
    var g = _table()
    var me = admitted_thread()
    var drop = List[Int]()
    var seen = 0
    for k in range(len(g[].threads)):  # small-loop(threads: the held matrices of the running calls, a few)
        if g[].threads[k] == me:
            if seen >= mark:
                drop.append(k)
            seen += 1
    var dropped = List[Int](capacity=len(drop))
    var j = len(drop) - 1
    while j >= 0:  # small-loop(drop: this thread's rows of the closing scope, a few)
        var k = drop[j]
        _ = g[].items.pop(k)
        _ = g[].addresses.pop(k)
        _ = g[].rows.pop(k)
        _ = g[].cols.pop(k)
        _ = g[].threads.pop(k)
        _ = g[].verdicts.pop(k)
        _ = g[].parents.pop(k)
        dropped.append(k)
        j -= 1
    var d = _depth_index(g, me)
    if d >= 0:
        g[].depths[d] -= 1
        if g[].depths[d] <= 0:
            _ = g[].depths.pop(d)
            _ = g[].depth_threads.pop(d)
    return dropped^


def admitted_run_hold_row(item: Int, address: Int, rows: Int, cols: Int, verdict: Int, parent: Int) raises -> Bool:
    """Append a row for the calling thread; False (nothing held) outside a
    scope. The door appends the pins at the same position when True."""
    var g = _table()
    var me = admitted_thread()
    if _depth_index(g, me) < 0:
        return False
    g[].items.append(item)
    g[].addresses.append(address)
    g[].rows.append(rows)
    g[].cols.append(cols)
    g[].threads.append(me)
    g[].verdicts.append(verdict)
    g[].parents.append(parent)
    return True


def admitted_run_scope_open() raises -> Bool:
    """True inside a hold scope of the calling thread."""
    return _depth_index(_table(), admitted_thread()) >= 0


def admitted_run_held_index(item: Int) raises -> Int:
    """The row the calling thread holds the matrix with id() word `item` in,
    or -1. Outside every scope the table is empty: one load and a compare."""
    var g = _table()
    if len(g[].items) == 0:
        return -1
    var me = admitted_thread()
    for k in range(len(g[].items)):  # small-loop(items: the held matrices of the running calls, a few)
        if g[].threads[k] == me and g[].items[k] == item:
            return k
    return -1


def admitted_held_address(index: Int) raises -> Int:
    return _table()[].addresses[index]


def admitted_held_rows(index: Int) raises -> Int:
    return _table()[].rows[index]


def admitted_held_cols(index: Int) raises -> Int:
    return _table()[].cols[index]


def admitted_held_finite(index: Int) raises -> Bool:
    """True when the held row was tested finite (not deferred)."""
    return _table()[].verdicts[index] == 1


def admitted_held_set_finite(index: Int) raises:
    """Record that the held row's words were tested finite."""
    _table()[].verdicts[index] = 1


def admitted_run_snapshot(index: Int, mut buffers: List[HeldBufferDescriptor], mut positions: List[Int]) raises -> Int:
    """The calling thread's rows as descriptors (and their positions, so the
    door can pin the same rows' owners); returns the snapshot index of row
    `index`, or -1."""
    var g = _table()
    var me = admitted_thread()
    var selected = -1
    for k in range(len(g[].items)):  # small-loop(items: the calling thread's retained buffer descriptors, a few)
        if g[].threads[k] == me:
            if k == index:
                selected = len(buffers)
            buffers.append(HeldBufferDescriptor(g[].items[k], g[].parents[k], g[].addresses[k], g[].rows[k],
                                                g[].cols[k], g[].verdicts[k] == 1))
            positions.append(k)
    return selected


def admitted_run_apply(buffers: List[HeldBufferDescriptor], proved: List[Int]) raises:
    """Record finite verdicts proved by RUN on a snapshot; another thread may
    have compacted rows since, so each is matched again by identity word and
    words."""
    var g = _table()
    for i in range(len(proved)):  # small-loop(proved: native finite verdicts for retained owners, a few)
        var old = proved[i]
        if old < 0 or old >= len(buffers):
            raise Error("admitted input: invalid finite-verdict proof")
        var current = admitted_run_held_index(buffers[old].identity)
        if current >= 0:
            if (g[].addresses[current] == buffers[old].address
                    and g[].rows[current] == buffers[old].rows
                    and g[].cols[current] == buffers[old].cols):
                g[].verdicts[current] = 1
