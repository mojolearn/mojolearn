# SPDX-License-Identifier: Apache-2.0
# Door-side pins of the tree family's device-resident fitted state (UNPACK
# and PACK only). The typed table, its keys, counters, eviction and native
# releases are RUN in bindings/trees_resident_table_run.mojo (split from
# bindings/trees_resident_table.mojo, lane resident-pin-tables, 2026-11-12).
#
# Each listed snapshot handle has one pin row here, keyed by the same handle
# word: the minting binding and the mirrored public fitted arrays, held
# strongly so their id() words, which the typed table compares, cannot be
# reused while the handle is listed. The typed table reports every handle it
# released (`trees_run_take_gone`); each door entry below drains that list
# before it returns and drops those rows. All of it runs with the GIL held.
from std.python import Python, PythonObject
from bindings.trees_side_records_unpack import trees_side_get, trees_side_put
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall
from bindings.trees_resident_table_run import (
    TREES_RESIDENT_SLOTS, TREES_RESIDENT_KEEP_BYTES, TREES_RESIDENT_FOREST, TREES_RESIDENT_DART,
    TREES_RESIDENT_FOREST_STRICT, trees_run_find, trees_run_put, trees_run_pin, trees_run_unpin,
    trees_run_forget, trees_run_take_gone, trees_run_stats,
)

comptime _DEATH_SLOT = "trees_resident_death_v1"


struct TreesResidentPins(Defaultable, Movable):
    # handle -> [binding, mirror, mirror, ...]
    var rows: Dict[Int, List[PythonObject]]

    def __init__(out self):
        self.rows = Dict[Int, List[PythonObject]]()


comptime _TREES_PINS = SharedSlot[TreesResidentPins, "MojolearnTreesResidentPinsV1", 1]


def _words(objects: List[PythonObject]) -> List[Int]:
    var words = List[Int](capacity=len(objects))
    for i in range(len(objects)):  # small-loop(mirrors: the snapshot's public fitted arrays, five or six)
        words.append(Int(objects[i]._obj_ptr))
    return words^


def _drain() raises:
    """Drop the pin rows of every handle the typed table released. The rows
    leave the map before any reference is released (a destructor may reenter
    the door)."""
    var gone = trees_run_take_gone()
    if len(gone) == 0:
        return
    var dropped = List[List[PythonObject]]()
    var pins = _TREES_PINS.get_or_create_ptr()
    for i in range(len(gone)):  # driver-loop: over snapshots released by this call (a few)
        if gone[i] in pins[].rows:
            dropped.append(pins[].rows.pop(gone[i]))
    _ = dropped^


def trees_resident_find(model_key: Int, generation: Int, kind: Int, native: PythonObject, words: List[Int],
                        mirrors: List[PythonObject]) raises -> Int:
    """The live handle of (`model_key`, `generation`, `kind`) minted by
    `native` for exactly `words` over exactly the `mirrors` objects, else 0.
    A stale entry of the same model and kind is released here."""
    var handle = trees_run_find(model_key, generation, kind, Int(native._obj_ptr), words, _words(mirrors))
    _drain()
    return handle


def trees_resident_put_native(model_key: Int, generation: Int, kind: Int, native: PythonObject,
                              release_entry: NativeDescriptorCall, release_op: Int, handle: Int, var words: List[Int],
                              var mirrors: List[PythonObject], bytes: Int) raises -> Bool:
    """Keep `handle` for the model (released by the native table entry
    `release_entry` called with `release_op` and [handle]). False when it is
    not kept: the caller releases it after its call."""
    var kept = trees_run_put(model_key, generation, kind, Int(native._obj_ptr), release_entry, release_op, handle,
                             words^, _words(mirrors), bytes)
    _drain()
    if kept:
        var row = List[PythonObject](capacity=len(mirrors) + 1)
        row.append(native)
        for i in range(len(mirrors)):  # small-loop(mirrors: the snapshot's public fitted arrays, five or six)
            row.append(mirrors[i])
        var pins = _TREES_PINS.get_or_create_ptr()
        pins[].rows[handle] = row^
    return kept


def trees_resident_pin(handle: Int) raises:
    """Hold `handle` live for the running call."""
    trees_run_pin(handle)


def trees_resident_unpin(handle: Int):
    """End the running call's hold; a snapshot that was asked to go while
    held is released now. Non-raising (it runs on error paths)."""
    try:
        trees_run_unpin(handle)
        _drain()
    except:
        pass


def trees_resident_forget(model_key: Int) raises:
    """Release every snapshot of `model_key` (its death or refit)."""
    trees_run_forget(model_key)
    _drain()


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
    var s = trees_run_stats()
    return Python.tuple(PythonObject(s[0]), PythonObject(s[1]), PythonObject(s[2]))
