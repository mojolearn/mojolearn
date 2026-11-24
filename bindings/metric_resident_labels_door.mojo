# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ONE resident label vector of a metric call encoded on the device (lane
vdf-xm-wire3, 2026-11-24; addendum 3 of the vd-functions briefs).

The single-vector twin of the label pair's device union step
(bindings/metrics_slot_door.mojo `mc_resident_pair`): a contiguous 1-D signed
integer label vector (4 or 8 bytes a label) that is resident in the active
`resident()` scope is read IN PLACE and encoded ONCE by the order rule on the
device (MS_ENCODE, metrics/slot_entry_device.mojo); its int32 codes stay in a
store slot the call owns and the k sorted classes come back once (the
class-list PACK, `mc_class_list`). The door's per-class tables (the index of
each class in a label order, or the 0/1 flag of a positive label: O(k) host
words) are applied to the codes on the device (MS_GATHER), and the x_metrics
program reads the mapped codes slot in place (`MetricDoorProg.pin_resident`).
The codes never come to the host.

Who uses it: the single-vector consumers of `mc_labels` / `mx_encode` (log_loss,
the probability and hinge losses, the curve and ranking doors). Each asks
`metric_resident_labels` first; `.live()` False (outside a scope, below its
floor, the host image, an explicit device="cpu", a list, strings, bool or
unsigned labels, a strided view) is the door's host route, unchanged.

The binding: the `_mojolearn_metrics` image of the call's mode, resolved
natively here, only inside an active scope (`estimator_binding_mode`, the same
selection the public classification shells' None binding slot gets in
providers/python/metric_binding_pyapi_pack.mojo). The context of an explicit
device="cpu" beside a GPU set (the CPU label scope, which names its x_metrics
entries) never reaches the device. Outside a scope a call pays one context read
and one thread-local scope read (`mc_slots_open`), the reads every wired metric
call makes.

No quiet host route for a resident input: when the label vector is resident on
the GPU image but the x_metrics program image does not read that image's store
(ranges off, another tier or vendor), the call is refused in words
(`metric_program_store_refusal`), never copied back to the host.
"""
from std.python import Python, PythonObject
from bindings.metrics_slot_door import McSlots, mc_slots_open, mc_slots_step, mc_resident_gather
from bindings.mcls_common_unpack import mc_b, mc_len, mc_index, mc_i32_array, mc_integer_view, mc_class_list
from bindings.mcls_common_run import MC_INT32_MAX
from bindings.classical_arrays_door import classical_address, classical_is_array
from bindings.family_common_pack import family_call
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_attr import py_attr
from bindings.py_balanced import py_get
from bindings.metrics_keys_unpack import metrics_keys
from bindings.metrics_program_door import metric_route_open
from core.native_descriptor_call import native_descriptor_call
from core.cast_in import CAST_KIND_SIGNED
from metrics.slot_ops import MS_ENCODE
from providers.python.estimator_binding_pyapi_door import estimator_binding_mode
from x_metrics.descriptor_schema import MP_MODE, MP_VENDOR


def metric_program_reads_slots(binding: PythonObject, mode: PythonObject, context: PythonObject) raises -> Bool:
    """Whether the x_metrics program image `mode` names reads the metrics
    image's store in place: it has the store (ranges on) and is of the same
    tier and vendor as the metrics image (`metrics_cluster_key`). Asked
    inside `resident()` only (lane vdf-xm-wire2; shared here by lane
    vdf-xm-wire3)."""
    var pin = metric_route_open(mode, context)
    if not pin.route.store:
        return False
    var key: PythonObject
    try:
        key = binding.metrics_cluster_key
    except:
        return False  # an image with no store key
    var tier = native_descriptor_call(pin.route.program, MP_MODE, [], [], [])
    var vendor = native_descriptor_call(pin.route.program, MP_VENDOR, [], [], [])
    _ = pin^
    return tier == Int(py=key[0]) and vendor == Int(py=key[1])


def metric_program_store_refusal(caller: String) -> String:
    """The words of a resident label input whose program image cannot read the
    GPU metrics image's store (lane vdf-xm-wire3): refused, never read on the
    host in its place."""
    return ("RuntimeError: mojolearn " + caller + ": a label input is resident on the GPU metrics image inside "
            + "resident(), but the x_metrics program image of this call does not read that image's store "
            + "(MOJOLEARN_ARENA_RANGES off, or binaries of another tier or vendor); install mojolearn binaries of "
            + "one build with ranges on, or call it outside resident()")


def metric_labels_binding(mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """The `_mojolearn_metrics` image a resident label vector is encoded on:
    the mode's image (`estimator_binding_mode`, exactly the public shells'
    selection), or None for the CPU label scope of an explicit device="cpu"
    beside a GPU set (it names its CPU x_metrics entries), which reads labels
    on the host. Resolved only inside an active `resident()` scope."""
    var nk = metrics_keys()
    if py_get(context, nk[].k_labels_metrics_entries) is not None:
        return PythonObject(None)
    return estimator_binding_mode(mode, "_mojolearn_metrics", context)


struct MetricResidentLabels(Movable):
    """One label vector encoded on the device: the call's slot state, the
    metrics image, the codes slot (-1: not live, the host route), the row
    count and the k sorted classes (a Python int list). Every slot the call
    made (the codes, the mapped codes) is freed by `close` after the program
    run, or by the destructor on an error path."""
    var state: McSlots
    var binding: PythonObject
    var codes: Int
    var n: Int
    var classes: PythonObject

    def __init__(out self, var state: McSlots, binding: PythonObject, codes: Int, n: Int, classes: PythonObject):
        self.state = state^
        self.binding = binding
        self.codes = codes
        self.n = n
        self.classes = classes

    def __del__(deinit self):
        # an error path between the encode and the run: the call's slots go with it
        _ = self.state.close()

    def live(self) -> Bool:
        """Whether the vector was encoded on the device."""
        return self.codes >= 0

    def k(self) raises -> Int:
        return mc_len(self.classes)

    def gather(mut self, table: PythonObject) raises -> Tuple[Int, Int]:
        """The codes mapped through a door's per-class int32 table (`k` words,
        a C-contiguous array) on the device into a fresh slot the call owns:
        (the slot, the first row mapped to -1, or -1)."""
        return mc_resident_gather(self.binding, self.codes, self.n, table, self.state)

    def order(mut self, order: PythonObject, context: PythonObject) raises -> Tuple[Int, Int]:
        """`mc_map_order` / `mx_codes` on the device: each row's index in
        `order`, -1 outside it; (the slot, the first row outside `order`, or
        -1). The table is O(k) host words."""
        return self.gather(metric_order_table(self.classes, order, context))

    def equal(mut self, value: PythonObject, context: PythonObject) raises -> Int:
        """`mc_map_equal` / `mx_map_eq` on the device: 1 for a row whose label
        equals `value`, else 0; the slot."""
        return self.gather(metric_equal_table(self.classes, value, context))[0]

    def row_class(mut self, order: PythonObject, first: Int, context: PythonObject) raises -> PythonObject:
        """The label of row `first`, which `order` (a label order the codes
        were mapped through) does not hold: the KeyError argument of the host
        gather's strict form. Found on the device by halving the missing
        classes (a gather per halving, each answering its first missing
        row): only on this error path, never a download of the codes."""
        var b = mc_b()
        var index = mc_index(order)
        var missing = Python.list()
        for c in range(self.k()):  # driver-loop: over the k classes (O(k) host words, error path only)
            var label = self.classes[c]
            if Int(py=family_call(index.get, label, PythonObject(-1))) < 0:
                _ = family_call(missing.append, label)
        var cands = missing
        while mc_len(cands) > 1:  # driver-loop: over halvings of the missing classes (log2 k gathers)
            var half = mc_len(cands) // 2
            var head = family_call(b.list, family_call(native_loaded_module("itertools").islice, cands, PythonObject(half)))
            var marks = family_call(py_attr(b.dict, "fromkeys"), head, PythonObject(-1))
            var table = family_call(b.list, family_call(b.map, py_attr(marks, "get"), self.classes,
                                                        family_call(native_loaded_module("itertools").repeat,
                                                                    PythonObject(0))))
            var got = self.gather(mc_i32_array(table, context))
            if got[1] == first:
                cands = head
            else:
                cands = family_call(b.list, family_call(native_loaded_module("itertools").islice, cands,
                                                        PythonObject(half), PythonObject(None)))
        if mc_len(cands) == 0:
            raise Error("RuntimeError: metrics: the device label map named a row outside the label order but no class")
        var found = cands[0]
        return found

    def close(mut self) raises:
        """Free the call's slots after the program run; a free failure after a
        good answer raises with the slots named."""
        var failures = self.state.close()
        if failures.byte_length() > 0:
            raise Error("RuntimeError: metric device label slots could not be freed: " + failures)


def metric_order_table(classes: PythonObject, order: PythonObject, context: PythonObject) raises -> PythonObject:
    """`mc_map_order`'s per-class table (the index of each class in `order`,
    -1 outside it) as a C-contiguous int32 array: O(k) host words."""
    var b = mc_b()
    var index = mc_index(order)
    var table = family_call(b.list, family_call(b.map, index.get, classes,
                                                family_call(native_loaded_module("itertools").repeat, PythonObject(-1))))
    return mc_i32_array(table, context)


def metric_equal_table(classes: PythonObject, value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`mc_map_equal`'s per-class table (`int(c == value)` for each class) as
    a C-contiguous int32 array: O(k) host words."""
    var b = mc_b()
    var eq = family_call(b.map, native_loaded_module("operator").eq, classes,
                         family_call(native_loaded_module("itertools").repeat, value))
    return mc_i32_array(family_call(b.list, family_call(b.map, b.int, family_call(b.map, b.bool, eq))), context)


def metric_resident_none(var state: McSlots) -> MetricResidentLabels:
    return MetricResidentLabels(state^, PythonObject(None), -1, 0, PythonObject(None))


def metric_resident_labels(values: PythonObject, mode: PythonObject, caller: String,
                           context: PythonObject) raises -> MetricResidentLabels:
    """THE DEVICE ENCODE of one label vector (MS_ENCODE), when it is a
    contiguous 1-D signed integer array (4 or 8 bytes a label, at most
    INT32_MAX rows) resident in the active `resident()` scope on the GPU
    metrics image of `mode`: read in place, codes kept in a slot, the classes
    back once. Anything else is not live (the caller's host route). A resident
    vector the program image cannot read in place is refused in words."""
    var state = mc_slots_open(context)
    if not state.resident.active():
        return metric_resident_none(state^)
    if not classical_is_array(values, context):
        return metric_resident_none(state^)
    var got = mc_integer_view(values, context)
    if not got:
        return metric_resident_none(state^)
    var item = got.value().item
    var n = got.value().count
    if not got.value().signed or got.value().stride != item or (item != 4 and item != 8) or n < 1 or n > MC_INT32_MAX:
        return metric_resident_none(state^)
    var binding = metric_labels_binding(mode, context)
    if binding is None:
        return metric_resident_none(state^)
    var address = Int(py=classical_address(got.value().view))
    var slot = state.slot_words(binding, values, address, n * item // 4)
    if slot < 0:
        return metric_resident_none(state^)  # below the floor, or the host image
    if not metric_program_reads_slots(binding, mode, context):
        _ = state.close()
        raise Error(metric_program_store_refusal(caller))
    var codes = state.codes(binding, n)
    # the class words: the encoder may meet n distinct labels; only k are written
    var words = List[UInt8](unsafe_uninit_length=n * item)
    var addrs: List[Int] = [0]
    var slots: List[Int] = [slot]
    var ints: List[Int] = [n, CAST_KIND_SIGNED, item, codes]
    var outs: List[Int] = [Int(words.unsafe_ptr())]
    var k = Int(py=mc_slots_step(binding, MS_ENCODE, addrs, slots, ints, List[Float64](), outs, state))
    if k < 1:
        _ = state.close()
        raise Error("RuntimeError: mojolearn " + caller + ": the device label encode returned no classes")
    var classes = mc_class_list(Int(words.unsafe_ptr()), item, k)
    _ = words^
    _ = got^  # the view's buffer is read by address inside the step above
    _ = values
    return MetricResidentLabels(state^, binding, codes, n, classes)
