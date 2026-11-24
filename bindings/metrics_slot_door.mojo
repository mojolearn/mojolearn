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

The host image (a CPU-only install, or `device="cpu"` beside a GPU set)
publishes `metrics_cluster_key` with vendor 0 and has no device store to
reach: a DeviceRows input is read from its host base there, by the doors'
existing admission (the parent decision of addendum 3, item 4). A GPU image
whose x_prep store cannot be reached is refused in words
(bindings/metric_store_refusal.mojo; lane vdf-xm-wire2), never read on the
host.
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
from bindings.mcls_common_unpack import mc_entry, mc_class_list
from bindings.metric_store_refusal import metric_store_refusal
from metrics.slot_ops import MS_ENCODE_PAIR, MS_GATHER
from core.cast_in import CAST_KIND_SIGNED


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
        var mode = Int(py=key[0])
        var vendor = Int(py=key[1])
        try:
            self.store = xp_transport_ref(PrepEntryKey(mode, vendor))
        except error:
            if vendor == 0:
                return False  # the host image (vendor 0): no device store to reach
            # A GPU image whose store cannot be reached is refused in words,
            # never sent to the host route (lane vdf-xm-wire2; the x_prep
            # image is connected with the metrics image at import,
            # python/mojolearn/_native_labels_bridge.py `_metrics`).
            raise Error(metric_store_refusal("metrics", mode, vendor, String(error)))
        self.resident.store = self.store
        self.live = True
        return True

    def slot(mut self, binding: PythonObject, value: PythonObject) raises -> Int:
        """The store slot a host float32 / int32 C-contiguous input of the
        call is read from in place: its resident copy inside `resident()`
        (put once per scope, a hit afterwards), else -1 (the host route)."""
        var size = Int(py=value.size)
        if not self.resident.keeps(False, size):
            return -1  # outside resident() or below its floor: no address read
        return self.slot_words(binding, value, Int(py=classical_address(value)), size)

    def slot_words(mut self, binding: PythonObject, owner: PythonObject, address: Int, words: Int) raises -> Int:
        """`slot` for `words` 4-byte words at `address` owned by `owner` (an
        int64 label buffer is 2n words; lane vdf-xm-wire)."""
        if not self.resident.keeps(False, words) or not self._store_ready(binding):
            return -1
        return self.resident.slot(owner, address, words)

    def codes(mut self, binding: PythonObject, n: Int) raises -> Int:
        """A fresh slot of `n` words this call owns (freed by `close`): a
        device stage's output, e.g. the codes of a resident label vector
        (`metric_slot_create`, the store's context). Needs the store, which
        a slot input of the call has already resolved."""
        if not self._store_ready(binding):
            raise Error("RuntimeError: metrics: a device code slot needs the GPU image's resident store")
        var id = Int(py=family_call(mc_entry(binding, "metric_slot_create"), PythonObject(n)))
        self.owned.append(id)
        return id

    def rows_shape(mut self, binding: PythonObject, value: PythonObject) raises -> Tuple[Int, Int]:
        """(rows, row words) of a DeviceRows input this call can gather on
        the device, or (-1, 0) when it cannot (no device store, a base that
        is not float32): the caller admits it on the host. Nothing moves."""
        if not self.is_rows(value) or not self._store_ready(binding):
            return (-1, 0)
        var fold = _rows_fields(value, self.prog)
        if fold.dtype != "<f4" or fold.itemsize != 4 or fold.rows < 1 or fold.row_words < 1:
            return (-1, 0)
        return (fold.rows, fold.row_words)

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
        var slot: Int
        try:
            slot = xp_transport_call(self.store, XP_DEV_TAKE_ROWS, idx, shape)
        except error:
            _ = self.close()  # the base put for this call only is freed with the failure
            raise error^
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


def mc_slots_step(binding: PythonObject, op: Int, addrs: List[Int], slots: List[Int], ints: List[Int],
                  floats: List[Float64], outs: List[Int], mut state: McSlots) raises -> PythonObject:
    """One device-input entry call (`metric_slots`) of a call with more
    device steps after it: the call's slots stay (freed by the last step,
    `mc_slots_call`); on an error they are freed and the error raised."""
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
    try:
        return family_call(mc_entry(binding, "metric_slots"), PythonObject(op), a, s, i, f, o)
    except error:
        _ = state.close()
        raise error^


def mc_slots_call(binding: PythonObject, op: Int, addrs: List[Int], slots: List[Int], ints: List[Int],
                  floats: List[Float64], outs: List[Int], mut state: McSlots) raises -> PythonObject:
    """The device-input entry (`metric_slots`) of the call's binding, then
    the call's own slots freed (a free failure after a good answer raises
    with the slots named)."""
    var result = mc_slots_step(binding, op, addrs, slots, ints, floats, outs, state)
    var failures = state.close()
    if failures.byte_length() > 0:
        raise Error("RuntimeError: metric device slots could not be freed: " + failures)
    return result


def mc_slot_inputs(binding: PythonObject, op: Int, values: List[PythonObject], ints: List[Int],
                   floats: List[Float64], outs: List[Int], context: PythonObject) raises -> Optional[PythonObject]:
    """The device-input form of one entry call whose inputs are the admitted
    host arrays `values` (C-contiguous float32 / int32): run when at least
    one is resident in the active `resident()` scope (read in place, never
    uploaded again), else None and the caller runs its by-name entry (the
    old route, unchanged)."""
    var state = mc_slots_open(context)
    var slots = List[Int](capacity=len(values))
    var addrs = List[Int](capacity=len(values))
    var any = False
    for i in range(len(values)):  # small-loop(values: the inputs of one metric entry, at most 3)
        var s = state.slot(binding, values[i])
        slots.append(s)
        addrs.append(Int(py=classical_address(values[i])))
        any = any or s >= 0
    if not any:
        return Optional[PythonObject]()
    var result = mc_slots_call(binding, op, addrs, slots, ints, floats, outs, state)
    _ = values  # read by address inside the call above
    return Optional[PythonObject](result)


# ------------------------------------------- resident label pairs (lane vdf-xm-wire)


struct McResidentPair(Movable):
    """A label pair encoded on the device: both vectors' int32 codes over the
    sorted union of their values (the host union encoder's codes) in two
    slots the call owns, the row count, the class count k, and (when asked)
    the k union classes as a Python int list (the class-list PACK)."""
    var codes_t: Int
    var codes_p: Int
    var n: Int
    var k: Int
    var classes: PythonObject

    def __init__(out self, codes_t: Int, codes_p: Int, n: Int, k: Int, classes: PythonObject):
        self.codes_t = codes_t
        self.codes_p = codes_p
        self.n = n
        self.k = k
        self.classes = classes

    def live(self) -> Bool:
        """Whether the pair was encoded on the device."""
        return self.codes_t >= 0


def mc_resident_none() -> McResidentPair:
    """No device pair: the caller's host route."""
    return McResidentPair(-1, -1, 0, 0, PythonObject(None))


def mc_resident_pair(binding: PythonObject, t_owner: PythonObject, t_address: Int, p_owner: PythonObject,
                     p_address: Int, n: Int, item: Int, want_classes: Bool,
                     mut state: McSlots) raises -> McResidentPair:
    """THE DEVICE UNION STEP of a label pair (MS_ENCODE_PAIR,
    metrics/slot_entry_device.mojo): when at least one of the two contiguous
    signed label vectors (`n` labels of `item` = 4 or 8 bytes at the
    addresses, owned by the owners) is resident in the active `resident()`
    scope (read in place; the other is uploaded once), both are concatenated
    and encoded ONCE on the device; the codes stay in two slots the call owns
    (`state.close` / the last `mc_slots_call` frees them). `.live()` False: no slot
    (outside a scope, below its floor, the host image, a vector too wide or
    too long) and the caller takes its host route, unchanged. The classes
    come back once, only with `want_classes`."""
    if n < 1 or n > 1073741823 or (item != 4 and item != 8):
        return mc_resident_none()
    var words = n * item // 4
    var st = state.slot_words(binding, t_owner, t_address, words)
    var sp = state.slot_words(binding, p_owner, p_address, words)
    if st < 0 and sp < 0:
        return mc_resident_none()
    var ct = state.codes(binding, n)
    var cp = state.codes(binding, n)
    # the class words (>= 2n items: the encoder may meet 2n distinct labels; only k are written)
    var classes = List[UInt8]()
    if want_classes:
        classes = List[UInt8](unsafe_uninit_length=2 * n * item)
    var addrs: List[Int] = [t_address, p_address]
    var slots: List[Int] = [st, sp]
    var ints: List[Int] = [n, CAST_KIND_SIGNED, item, ct, cp]
    var outs: List[Int] = [Int(classes.unsafe_ptr()) if want_classes else 0]
    var k = Int(py=mc_slots_step(binding, MS_ENCODE_PAIR, addrs, slots, ints, List[Float64](), outs, state))
    var listed = PythonObject(None)
    if want_classes:
        listed = mc_class_list(Int(classes.unsafe_ptr()), item, k)
    _ = classes^
    _ = t_owner  # read by address inside the step above
    _ = p_owner
    return McResidentPair(ct, cp, n, k, listed)


def mc_resident_order(binding: PythonObject, codes: Int, n: Int, table: PythonObject,
                      mut state: McSlots) raises -> Int:
    """A codes slot mapped through a door's class-order table (`table`: a
    C-contiguous int32 array of k words, the `mc_map_order` table) ON THE
    DEVICE into a fresh slot the call owns (MS_GATHER); its id. The table is
    O(k) host words, uploaded once."""
    var dst = state.codes(binding, n)
    var addrs: List[Int] = [0]
    var slots: List[Int] = [codes]
    var ints: List[Int] = [n, Int(py=table.size), dst]
    var outs: List[Int] = [Int(py=classical_address(table))]
    _ = mc_slots_step(binding, MS_GATHER, addrs, slots, ints, List[Float64](), outs, state)
    _ = table  # read by address inside the step above
    return dst
