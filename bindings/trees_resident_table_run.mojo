# SPDX-License-Identifier: Apache-2.0
# The tree family's device-resident fitted state: the TYPED table (RUN; no
# Python). Split from bindings/trees_resident_table.mojo (lane
# resident-pin-tables, 2026-11-12): the Python pins (the minting binding and
# the mirrored public fitted arrays) live in the door-side pin store of
# bindings/py_trees_resident.mojo, keyed by the same handle word; this table
# holds only words and the native release entry of each snapshot.
#
# Rules (plan docs/plans/NATIVE_ENTRY_POINTS_AFTER_0866.md section A; unchanged
# by the split):
# - Key: (model key, generation, kind), plus the validity words the snapshot
#   was built for (dims, mode, aggregation) and the IDENTITY words of the
#   minting binding and of the public fitted arrays it mirrors. The door pins
#   those objects for as long as the handle is listed here, so a word can
#   never be reused by another object while it is compared.
# - A handle whose generation or words differ is released on the next call of
#   its model; the model's death releases every entry it owns.
# - Memory: at most TREES_RESIDENT_SLOTS entries and TREES_RESIDENT_KEEP_BYTES
#   bytes per minting binding; past either, the least recently used unpinned
#   entry is released. A snapshot larger than the byte cap is not kept.
# - Pins: a handle a door is launching on is pinned for the call; a pinned
#   entry asked to go is parked and released at its last unpin.
# - Every handle that leaves the table for good (released) is appended to
#   `gone`; the door drains it (`trees_run_take_gone`) at the end of each of
#   its calls and drops that handle's Python pins. No callback into Python.
# - Numerics: none.
#
# Every mutation runs with the GIL held (door code), which serializes the
# table; a release runs after its entry has left the table.
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call

comptime TREES_RESIDENT_SLOTS = 64
comptime TREES_RESIDENT_KEEP_BYTES = 1 << 30

# Entry kinds (one per snapshot shape a door keeps).
comptime TREES_RESIDENT_FOREST = 1
comptime TREES_RESIDENT_DART = 2
comptime TREES_RESIDENT_FOREST_STRICT = 3


struct TreesResidentWords(Movable):
    var model_key: Int
    var generation: Int
    var kind: Int
    # id() of the minting binding (pinned by the door).
    var native: Int
    # The snapshot's native release entry, called with `release_op` and the
    # params [handle] (the descriptor shape of core/native_descriptor_abi.mojo).
    var release: NativeDescriptorCall
    var release_op: Int
    var handle: Int
    var words: List[Int]
    # id() of each mirrored public fitted array (pinned by the door).
    var mirrors: List[Int]
    var bytes: Int
    var used: Int
    var pins: Int

    def __init__(out self, model_key: Int, generation: Int, kind: Int, native: Int,
                 release: NativeDescriptorCall, release_op: Int, handle: Int, var words: List[Int],
                 var mirrors: List[Int], bytes: Int, used: Int):
        self.model_key = model_key
        self.generation = generation
        self.kind = kind
        self.native = native
        self.release = release
        self.release_op = release_op
        self.handle = handle
        self.words = words^
        self.mirrors = mirrors^
        self.bytes = bytes
        self.used = used
        self.pins = 0


struct TreesResidentTable(Defaultable, Movable):
    var entries: List[TreesResidentWords]
    # Entries asked to go while pinned: released at their last unpin.
    var parked: List[TreesResidentWords]
    # Handles released since the door last drained (their pins go).
    var gone: List[Int]
    var tick: Int
    var uploads: Int

    def __init__(out self):
        self.entries = List[TreesResidentWords]()
        self.parked = List[TreesResidentWords]()
        self.gone = List[Int]()
        self.tick = 0
        self.uploads = 0


# V4 (lane resident-pin-tables): words only; pins moved to the door.
comptime TREES_RESIDENT = SharedSlot[TreesResidentTable, "MojolearnTreesResidentTableV4", 4]


def _release_now(release: NativeDescriptorCall, release_op: Int, handle: Int):
    """The snapshot's compiled release entry; quiet, as before: a binding
    torn down at interpreter exit has nothing left to release."""
    if handle < 1:
        return
    try:
        var addrs = List[Int]()
        var ints: List[Int] = [handle]
        var floats: List[Float64] = [Float64(handle)]
        _ = native_descriptor_call(release, release_op, addrs, ints, floats)
    except:
        pass


def _drop_at(i: Int) raises:
    """Take entry i out of the table, then release it (or park it while a
    call still launches on it)."""
    var table = TREES_RESIDENT.get_or_create_ptr()
    var entry = table[].entries.pop(i)
    if entry.pins > 0:
        table[].parked.append(entry^)
        return
    var release = entry.release
    var op = entry.release_op
    var handle = entry.handle
    _ = entry^
    table = TREES_RESIDENT.get_or_create_ptr()
    table[].gone.append(handle)
    _release_now(release, op, handle)


def _same(held: List[Int], words: List[Int]) -> Bool:
    if len(held) != len(words):
        return False
    for i in range(len(words)):  # small-loop(words: the snapshot's validity or mirror words, about six)
        if held[i] != words[i]:
            return False
    return True


def trees_run_find(model_key: Int, generation: Int, kind: Int, native: Int, words: List[Int],
                   mirrors: List[Int]) raises -> Int:
    """The live handle of (`model_key`, `generation`, `kind`) minted by the
    binding `native` for exactly `words` over exactly the `mirrors` words,
    else 0. A stale entry of the same model and kind is released here."""
    var table = TREES_RESIDENT.get_or_create_ptr()
    table[].tick += 1
    for i in range(len(table[].entries)):  # driver-loop: over resident tree snapshots (at most TREES_RESIDENT_SLOTS), owner lookup
        if table[].entries[i].model_key != model_key or table[].entries[i].kind != kind:
            continue
        if (table[].entries[i].generation == generation and table[].entries[i].native == native
                and _same(table[].entries[i].words, words) and _same(table[].entries[i].mirrors, mirrors)):
            table[].entries[i].used = table[].tick
            return table[].entries[i].handle
        _drop_at(i)
        return 0
    return 0


def trees_run_put(model_key: Int, generation: Int, kind: Int, native: Int, release: NativeDescriptorCall,
                  release_op: Int, handle: Int, var words: List[Int], var mirrors: List[Int],
                  bytes: Int) raises -> Bool:
    """Keep `handle` for the model. False when it is larger than the byte cap
    or every kept snapshot of this binding is in a running call: then nothing
    is kept and the caller releases it after its call."""
    if bytes > TREES_RESIDENT_KEEP_BYTES:
        return False
    var table = TREES_RESIDENT.get_or_create_ptr()
    while True:  # driver-loop: over least-recently-used evictions (at most TREES_RESIDENT_SLOTS)
        var count = 0
        var held = 0
        var oldest = -1
        for i in range(len(table[].entries)):  # driver-loop: over resident tree snapshots of this binding
            if table[].entries[i].native != native:
                continue
            count += 1
            held += table[].entries[i].bytes
            if table[].entries[i].pins == 0 and (oldest < 0 or table[].entries[i].used < table[].entries[oldest].used):
                oldest = i
        if count < TREES_RESIDENT_SLOTS and held + bytes <= TREES_RESIDENT_KEEP_BYTES:
            break
        if oldest < 0:
            return False
        _drop_at(oldest)
        table = TREES_RESIDENT.get_or_create_ptr()
    table[].tick += 1
    table[].uploads += 1
    table[].entries.append(TreesResidentWords(model_key, generation, kind, native, release, release_op, handle,
                                              words^, mirrors^, bytes, table[].tick))
    return True


def trees_run_pin(handle: Int) raises:
    """Hold `handle` live for the running call."""
    var table = TREES_RESIDENT.get_or_create_ptr()
    for i in range(len(table[].entries)):  # driver-loop: over resident tree snapshots (at most TREES_RESIDENT_SLOTS)
        if table[].entries[i].handle == handle:
            table[].entries[i].pins += 1
            return


def trees_run_unpin(handle: Int) raises:
    """End the running call's hold; a parked snapshot is released at its
    last unpin."""
    var table = TREES_RESIDENT.get_or_create_ptr()
    for i in range(len(table[].entries)):  # driver-loop: over resident tree snapshots (at most TREES_RESIDENT_SLOTS)
        if table[].entries[i].handle == handle:
            if table[].entries[i].pins > 0:
                table[].entries[i].pins -= 1
            return
    for i in range(len(table[].parked)):  # small-loop(parked: snapshots asked to go mid-call, usually none)
        if table[].parked[i].handle == handle:
            table[].parked[i].pins -= 1
            if table[].parked[i].pins <= 0:
                var entry = table[].parked.pop(i)
                var release = entry.release
                var op = entry.release_op
                var gone = entry.handle
                _ = entry^
                table = TREES_RESIDENT.get_or_create_ptr()
                table[].gone.append(gone)
                _release_now(release, op, gone)
            return


def trees_run_forget(model_key: Int) raises:
    """Release every snapshot of `model_key` (its death, or its refit)."""
    var table = TREES_RESIDENT.get_or_create_ptr()
    var i = 0
    while i < len(table[].entries):  # driver-loop: over resident tree snapshots (at most TREES_RESIDENT_SLOTS)
        if table[].entries[i].model_key == model_key:
            _drop_at(i)
            table = TREES_RESIDENT.get_or_create_ptr()
        else:
            i += 1


def trees_run_take_gone() raises -> List[Int]:
    """The handles released since the last drain (the door drops their pins)."""
    var table = TREES_RESIDENT.get_or_create_ptr()
    var gone = List[Int]()
    if len(table[].gone) == 0:
        return gone^
    swap(gone, table[].gone)
    return gone^


def trees_run_stats() raises -> List[Int]:
    """[snapshots kept, bytes kept, uploads so far] (tools/ checks only)."""
    var table = TREES_RESIDENT.get_or_create_ptr()
    var held = 0
    for i in range(len(table[].entries)):  # driver-loop: over resident tree snapshots (at most TREES_RESIDENT_SLOTS)
        held += table[].entries[i].bytes
    var out_words: List[Int] = [len(table[].entries), held, table[].uploads]
    return out_words^
