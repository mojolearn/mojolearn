# SPDX-License-Identifier: Apache-2.0
# The tree family's device-resident fitted state (workstream A, lane
# ne1-a-trees, 2026-11-06; docs/plans/NATIVE_ENTRY_POINTS_AFTER_0866.md A).
#
# One side table in the trees door (the classical owner binary), keyed by the
# model's identity and its generation word (bindings/native_model_records.mojo,
# the ne1 CONTRACT module), holding the device snapshot handles a fitted tree
# model's predicts launch on: the forest snapshot of `forest_prepare_gpu`
# (RandomForest*, ExtraTrees*, DecisionTree*), and any other kind a door adds
# under its own `TREES_RESIDENT_*` code. Before, the forest kept its snapshot
# lifetime in `fields["_resident_holder"]`, a dict in the model's `__dict__`
# (lesson 9: records never live there).
#
# Rules (plan section A):
# - Key: (model_identity_key, model_generation, kind), plus the validity words
#   the snapshot was built for (dims, mode, aggregation) and the IDENTITIES of
#   the public fitted arrays it mirrors (held strongly here, so their address
#   cannot be reused while the entry lives). Replacing an attribute changes an
#   identity and the next call re-uploads. No content hash anywhere.
# - Read-only: the door freezes every mirrored exact Array before the first
#   upload (bindings/p3nc_fitted_freeze.mojo `fitted_freeze`), so an in-place
#   NumPy write raises NumPy's read-only error.
# - A handle whose generation or words differ is released on the next call of
#   its model; the model's death releases every entry it owns (the
#   prepared-owner weakref path, `trees_resident_watch`).
# - Memory: at most TREES_RESIDENT_SLOTS entries and TREES_RESIDENT_KEEP_BYTES
#   bytes per minting binding (one binding = one device context of one numeric
#   tier in this process); past either, the least recently used unpinned entry
#   is released and its model re-uploads on its next call. The caps copy the
#   sequence store's (sequence/perf3_seq_resident.mojo: 64 blocks / 1 GiB):
#   memory reasoning about a handful of fitted models per process, never a
#   model or dataset shape. A snapshot larger than the byte cap is not kept: the
#   caller releases it after its one call (the per-call upload of before).
#   Profile item 14 measures the cap once on a box.
# - Pins: a handle a door is launching on is pinned for the call, so another
#   thread's eviction or a death callback never frees memory a kernel reads
#   (the kernels release the GIL). A pinned entry asked to go is marked and
#   released at its last unpin.
# - Numerics: none. The snapshot is the same bytes the per-call upload took.
#
# Every mutation runs with the GIL held (door code), which serializes the
# table; the release calls run after the entry has left the table, so a
# release that reenters the door sees a consistent table.
from std.ffi import _Global
from std.python import Python, PythonObject
from bindings.py_balanced import py_call
from bindings.trees_side_records import trees_side_get, trees_side_put

comptime TREES_RESIDENT_SLOTS = 64
comptime TREES_RESIDENT_KEEP_BYTES = 1 << 30

# Entry kinds (one per snapshot shape a door keeps).
comptime TREES_RESIDENT_FOREST = 1
comptime TREES_RESIDENT_DART = 2

comptime _DEATH_SLOT = "trees_resident_death_v1"


struct TreesResidentEntry(Movable):
    var model_key: Int
    var generation: Int
    var kind: Int
    var native: PythonObject
    var release: PythonObject
    var handle: Int
    var words: List[Int]
    var mirrors: List[PythonObject]
    var bytes: Int
    var used: Int
    var pins: Int
    var dead: Bool

    def __init__(out self, model_key: Int, generation: Int, kind: Int, native: PythonObject,
                 release: PythonObject, handle: Int, var words: List[Int], var mirrors: List[PythonObject],
                 bytes: Int, used: Int):
        self.model_key = model_key
        self.generation = generation
        self.kind = kind
        self.native = native
        self.release = release
        self.handle = handle
        self.words = words^
        self.mirrors = mirrors^
        self.bytes = bytes
        self.used = used
        self.pins = 0
        self.dead = False


struct TreesResidentTable(Defaultable, Movable):
    var entries: List[TreesResidentEntry]
    # Entries asked to go while pinned: released at their last unpin.
    var parked: List[TreesResidentEntry]
    var tick: Int
    var uploads: Int

    def __init__(out self):
        self.entries = List[TreesResidentEntry]()
        self.parked = List[TreesResidentEntry]()
        self.tick = 0
        self.uploads = 0


comptime TREES_RESIDENT = _Global[
    StorageType=TreesResidentTable, name="MojolearnTreesResidentTableV1", init_fn=TreesResidentTable.__init__,
]


def _release_now(release: PythonObject, handle: Int):
    """The snapshot's compiled release entry; quiet, as the former
    `TreesResident` lifetime: a binding torn down at interpreter exit has
    nothing left to release."""
    if handle < 1:
        return
    try:
        _ = py_call(release, PythonObject(handle))
    except:
        pass


def _drop_at(i: Int) raises:
    """Take entry i out of the table, then release it (or park it while a
    call still launches on it)."""
    var table = TREES_RESIDENT.get_or_create_ptr()
    var entry = table[].entries.pop(i)
    if entry.pins > 0:
        entry.dead = True
        table[].parked.append(entry^)
        return
    var release = entry.release
    var handle = entry.handle
    _ = entry^
    _release_now(release, handle)


def _same_mirrors(held: List[PythonObject], mirrors: List[PythonObject]) -> Bool:
    if len(held) != len(mirrors):
        return False
    for i in range(len(mirrors)):  # small-loop(mirrors: the snapshot's public fitted arrays, five or six)
        if held[i]._obj_ptr != mirrors[i]._obj_ptr:
            return False
    return True


def _same_words(held: List[Int], words: List[Int]) -> Bool:
    if len(held) != len(words):
        return False
    for i in range(len(words)):  # small-loop(words: the snapshot's validity words, about six)
        if held[i] != words[i]:
            return False
    return True


def trees_resident_find(model_key: Int, generation: Int, kind: Int, native: PythonObject, words: List[Int],
                        mirrors: List[PythonObject]) raises -> Int:
    """The live handle of (`model_key`, `generation`, `kind`) minted by
    `native` for exactly `words` over exactly the `mirrors` objects, else 0.
    A stale entry of the same model and kind (a refit, a replaced array,
    another mode, another binding) is released here."""
    var table = TREES_RESIDENT.get_or_create_ptr()
    table[].tick += 1
    for i in range(len(table[].entries)):  # driver-loop: over resident tree snapshots (at most TREES_RESIDENT_SLOTS), owner lookup
        if table[].entries[i].model_key != model_key or table[].entries[i].kind != kind:
            continue
        if (table[].entries[i].generation == generation and table[].entries[i].native._obj_ptr == native._obj_ptr
                and _same_words(table[].entries[i].words, words) and _same_mirrors(table[].entries[i].mirrors, mirrors)):
            table[].entries[i].used = table[].tick
            return table[].entries[i].handle
        _drop_at(i)
        return 0
    return 0


def trees_resident_put(model_key: Int, generation: Int, kind: Int, native: PythonObject, release: PythonObject,
                       handle: Int, var words: List[Int], var mirrors: List[PythonObject], bytes: Int) raises -> Bool:
    """Keep `handle` for the model. False when it is larger than the byte cap:
    then nothing is kept and the caller releases it after its call."""
    if bytes > TREES_RESIDENT_KEEP_BYTES:
        return False
    var table = TREES_RESIDENT.get_or_create_ptr()
    while True:  # driver-loop: over least-recently-used evictions (at most TREES_RESIDENT_SLOTS)
        var count = 0
        var held = 0
        var oldest = -1
        for i in range(len(table[].entries)):  # driver-loop: over resident tree snapshots of this binding
            if table[].entries[i].native._obj_ptr != native._obj_ptr:
                continue
            count += 1
            held += table[].entries[i].bytes
            if table[].entries[i].pins == 0 and (oldest < 0 or table[].entries[i].used < table[].entries[oldest].used):
                oldest = i
        if count < TREES_RESIDENT_SLOTS and held + bytes <= TREES_RESIDENT_KEEP_BYTES:
            break
        if oldest < 0:
            # Every kept snapshot of this binding is in a running call: do not
            # keep this one (the caller releases it after its call).
            return False
        _drop_at(oldest)
    table[].tick += 1
    table[].uploads += 1
    table[].entries.append(TreesResidentEntry(model_key, generation, kind, native, release, handle, words^, mirrors^,
                                              bytes, table[].tick))
    return True


def trees_resident_pin(handle: Int) raises:
    """Hold `handle` live for the running call."""
    var table = TREES_RESIDENT.get_or_create_ptr()
    for i in range(len(table[].entries)):  # driver-loop: over resident tree snapshots (at most TREES_RESIDENT_SLOTS)
        if table[].entries[i].handle == handle:
            table[].entries[i].pins += 1
            return


def trees_resident_unpin(handle: Int):
    """End the running call's hold; a snapshot that was asked to go while
    held is released now. Non-raising (it runs on error paths; lesson 6)."""
    try:
        _unpin(handle)
    except:
        pass


def _unpin(handle: Int) raises:
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
                var gone = entry.handle
                _ = entry^
                _release_now(release, gone)
            return


def trees_resident_forget(model_key: Int) raises:
    """Release every snapshot of `model_key` (its death)."""
    var table = TREES_RESIDENT.get_or_create_ptr()
    var i = 0
    while i < len(table[].entries):  # driver-loop: over resident tree snapshots (at most TREES_RESIDENT_SLOTS)
        if table[].entries[i].model_key == model_key:
            _drop_at(i)
        else:
            i += 1


struct TreesResidentDeath(Movable, Writable):
    """The model's death record (held by the prepared-owner registry through
    a weak reference to the model): its end releases the model's snapshots."""
    var model_key: Int

    def __init__(out self, model_key: Int):
        self.model_key = model_key

    def write_to(self, mut writer: Some[Writer]):
        writer.write("_TreesResidentDeath(", self.model_key, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def __deinit__(deinit self):
        try:
            trees_resident_forget(self.model_key)
        except:
            pass


def trees_resident_watch(model: PythonObject, model_key: Int) raises -> Bool:
    """Register the model's death callback once. False when the model takes
    no weak reference: then the caller keeps nothing and releases its
    snapshot after the call."""
    var held = trees_side_get(_DEATH_SLOT, model)
    if held is not None:
        return True
    return trees_side_put(_DEATH_SLOT, model, PythonObject(alloc=TreesResidentDeath(model_key)))


def trees_resident_stats() raises -> PythonObject:
    """`(snapshots kept, bytes kept, uploads so far)`: a readout for the
    tools/ checks only."""
    var table = TREES_RESIDENT.get_or_create_ptr()
    var held = 0
    for i in range(len(table[].entries)):  # driver-loop: over resident tree snapshots (at most TREES_RESIDENT_SLOTS)
        held += table[].entries[i].bytes
    return Python.tuple(PythonObject(len(table[].entries)), PythonObject(held), PythonObject(table[].uploads))
