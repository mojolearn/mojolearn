# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""UNPACK of the device-resident inputs of the `_mojolearn_metrics` kernel
entries (lane vdf-xm-kernels, 2026-11-24; addendum 3 of the vd-functions
briefs).

The ONE convention the estimators and the program runner already use
(python/mojolearn/_arena_io.py; bindings/arena_prog_native_door.mojo):

  * inside `resident()`, a host buffer of at least the scope's floor is put
    ONCE into the x_prep image's store, keyed by identity
    (`XprepResident.slot`), and every later metric call of the scope reads
    the same slot;
  * a `DeviceRows` fold view has its rows gathered ON THE DEVICE out of its
    base (`XP_DEV_TAKE_ROWS`), the base put once (through the scope when one
    is active, else for this call only).

A call with at least one slot runs the entry's device-input form
(`metric_slots`, metrics/slot_entry_device.mojo); a call with none runs the
old by-name entry, so `device=None` on host inputs outside `resident()` is
today's route plus one context read and one thread-local scope read per
call (`mc_slots_open`), the same reads the program runner makes per run.

Only the GPU image publishes `metrics_cluster_key`; the host image (a
CPU-only install, or `device="cpu"` beside a GPU set) has none, so it never
takes a slot: a DeviceRows input is read from its host base there, by the
doors' existing admission (the parent decision of addendum 3, item 4).
"""
from std.python import Python, PythonObject
from core.entry_table import EntryRef, entry_ref_unresolved
from core.prep_entry_table import PrepEntryKey, xp_transport_ref, xp_transport_call
from x_prep.transport_abi import XP_DEV_PUT, XP_DEV_FREE, XP_DEV_TAKE_ROWS
from bindings.py_balanced import py_item, py_type, py_call
from bindings.classical_arrays_door import classical_address
from bindings.metrics_keys_unpack import metrics_keys
from bindings.x_prep_native_owner_door import XprepResident, xprep_resident, xprep_resident_none
from bindings.x_prep_native_inputs_door import _rows_fields
from bindings.family_common_pack import family_call
from bindings.mcls_common_unpack import mc_entry


struct McSlots(Movable):
    """The device-input state of ONE metric call: the active `resident()`
    scope on the metrics image's store (or none), the store's transport
    entry, the slots this call made (gathers, puts outside a scope, codes),
    freed by `close`, and the Python owners read by address meanwhile."""
    var resident: XprepResident
    var store: EntryRef
    var live: Bool
    var rows_class: PythonObject
    var prog: PythonObject
    var owned: List[Int]
    var held: List[PythonObject]

    def __init__(out self, var resident: XprepResident, store: EntryRef, live: Bool, rows_class: PythonObject,
                 prog: PythonObject):
        self.resident = resident^
        self.store = store
        self.live = live
        self.rows_class = rows_class
        self.prog = prog
        self.owned = List[Int]()
        self.held = List[PythonObject]()

    def is_rows(self, value: PythonObject) raises -> Bool:
        """Whether `value` is an exact DeviceRows fold view."""
        return py_type(value) is self.rows_class

    def _store_ready(mut self, binding: PythonObject) raises -> Bool:
        # The metrics GPU image's (mode, vendor) names the x_prep store on
        # the same device: resolved once per call, at the first slot.
        if self.live:
            return True
        var key: PythonObject
        try:
            key = binding.metrics_cluster_key
        except:
            return False  # the host image: no device store
        try:
            self.store = xp_transport_ref(PrepEntryKey(Int(py=key[0]), Int(py=key[1])))
        except:
            return False  # no x_prep image connected: the host route
        self.resident.store = self.store
        self.live = True
        return True

    def slot(mut self, binding: PythonObject, value: PythonObject) raises -> Int:
        """The store slot a host float32 / int32 C-contiguous input of the
        call is read from in place: its resident copy inside `resident()`
        (put once per scope, a hit afterwards), else -1 (the host route)."""
        var size = Int(py=value.size)
        if not self.resident.keeps(False, size) or not self._store_ready(binding):
            return -1
        return self.resident.slot(value, Int(py=classical_address(value)), size)

    def rows(mut self, binding: PythonObject, value: PythonObject) raises -> Tuple[Int, Int, Int]:
        """(slot, rows, row words) of a DeviceRows input gathered on the
        device, or (-1, 0, 0) without a device store (the caller admits it
        on the host). The base goes up once: through the active scope, else
        into a slot this call owns."""
        if not self._store_ready(binding):
            return (-1, 0, 0)
        var fold = _rows_fields(value, self.prog)
        # A DeviceRows base is a float32 C-contiguous 2-D Array (_arena_io.py)
        if fold.dtype != "<f4" or fold.itemsize != 4:
            return (-1, 0, 0)
        var base = fold.base
        var n = fold.rows
        var d = fold.row_words
        if n < 1 or d < 1:
            return (-1, 0, 0)
        var address = Int(py=classical_address(base))
        var words = Int(py=base.size)
        var base_slot: Int
        if self.resident.keeps(True, words):
            base_slot = self.resident.slot(base, address, words)
        else:
            var src: List[Int] = [address]
            var count: List[Int] = [words]
            base_slot = xp_transport_call(self.store, XP_DEV_PUT, src, count)
            self.owned.append(base_slot)
        var indices = fold.indices
        var idx: List[Int] = [Int(py=classical_address(indices))]
        var shape: List[Int] = [base_slot, d, n]
        var slot = xp_transport_call(self.store, XP_DEV_TAKE_ROWS, idx, shape)
        self.owned.append(slot)
        self.held.append(indices)
        self.held.append(base)
        return (slot, n, d)

    def close(mut self) -> String:
        """Free the slots this call made, newest first, every one attempted;
        the first failures' text (empty: none). Resident-scope slots stay
        with the scope."""
        var failures = String()
        for i in range(len(self.owned)):  # driver-loop: over the slots one metric call made
            var id = self.owned[len(self.owned) - 1 - i]
            try:
                var ids: List[Int] = [id]
                _ = xp_transport_call(self.store, XP_DEV_FREE, List[Int](), ids)
            except error:
                failures += ("" if failures.byte_length() == 0 else "; ") + "slot " + String(id) + ": " + String(error)
        self.owned.clear()
        self.held.clear()
        return failures


def mc_slots_open(context: PythonObject) raises -> McSlots:
    """The call's device-input state: one context read, one thread-local
    scope read. The store entry is resolved only at the first input that
    can take a slot."""
    var prog = py_item(context, metrics_keys()[].k_family_metrics_prog_context)
    var rows_class = prog["maxabs_rows_class"]
    var resident = xprep_resident(prog, entry_ref_unresolved(), True)
    return McSlots(resident^, entry_ref_unresolved(), False, rows_class, prog)


def mc_slots_call(binding: PythonObject, op: Int, addrs: List[Int], slots: List[Int], ints: List[Int],
                  floats: List[Float64], outs: List[Int], mut state: McSlots) raises -> PythonObject:
    """The device-input entry (`metric_slots`) of the call's binding, then
    the call's own slots freed (a free failure after a good answer raises
    with the slots named)."""
    var a = Python.list()
    var s = Python.list()
    var i = Python.list()
    var f = Python.list()
    var o = Python.list()
    for j in range(len(addrs)):  # small-loop(addrs: the inputs of one metric entry, at most 3)
        _ = py_call(a.append, PythonObject(addrs[j]))
        _ = py_call(s.append, PythonObject(slots[j]))
    for j in range(len(ints)):  # small-loop(ints: the integer words of one metric entry, at most 6)
        _ = py_call(i.append, PythonObject(ints[j]))
    for j in range(len(floats)):  # small-loop(floats: at most one float word)
        _ = py_call(f.append, PythonObject(floats[j]))
    for j in range(len(outs)):  # small-loop(outs: the output addresses of one metric entry, at most 3)
        _ = py_call(o.append, PythonObject(outs[j]))
    var result: PythonObject
    try:
        result = family_call(mc_entry(binding, "metric_slots"), PythonObject(op), a, s, i, f, o)
    except error:
        _ = state.close()
        raise error^
    var failures = state.close()
    if failures.byte_length() > 0:
        raise Error("RuntimeError: metric device slots could not be freed: " + failures)
    return result
