# SPDX-License-Identifier: Apache-2.0
"""UNPACK: the one device route of a Samba stack.

A stack's device and tier are resolved ONCE (construction, restore, or the
first operation after its numeric mode object changed) into the shared
`DeviceRouteOwner` (providers/python/device_route_boundary_door.mojo) with
the rule and words of core/device_route.mojo. One route covers every part of
the stack together:

  training entries  `images`: the connected `_mojolearn_training` namespace
                    of the route's device (provider route, caps, AFN entries)
  primitives        `route.entry`: the Samba table entry of that image
                    (embedding, RMSNorm, head, loss, accumulate, the per-call
                    optimizer step and the generator's RNG, op 6)
  Mamba / attention `scope`: the block route word the stack's blocks are
  blocks            built with (`samba_build_blocks`); each block then holds
                    its own route on the same owner type (the block door)
  optimizer         the resident optimizer opens on the provider of `images`
  program / decode  `route.flags` bit 1: the image published the stack program

`route.flags` bit 0 is the CPU inference model (`SambaInference`, CPU by
class): its primitives are the `_mojolearn_neural_host` image's.

No operation reads or compares the device: a hit is the numeric-mode field
read, one record read and three pointer compares. A SambaStack on a GPU
installation has no CPU route: the block backward of a CPU-routed block has
no host training image loaded beside the GPU set (block door,
`neural_training_selection`), so an explicit `device="cpu"` there is refused
in words at construction and at restore; nothing of such a stack runs on a
device its caller did not name.

Metadata only: no independent units to split.
"""
from std.python import PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_get, py_item
from bindings.samba_keys_unpack import samba_keys
from bindings.samba_admission_door import samba_admit_model_fields
from bindings.p3_samba_caps_door import samba_caps, SAMBA_CAP_MODE, SAMBA_CAP_VENDOR
from bindings.causal_lm_construct_door import _model_bindings
from core.entry_table import EntryRef, entry_ref_unresolved
from core.samba_entry_table import (
    SambaEntryKey, samba_ref, samba_vendor_word, SAMBA_TABLE_TRAINING, SAMBA_TABLE_NEURAL_HOST,
)
from core.samba_entry_table_programs import samba_programs_available
from core.mode_answer import MODE_FAMILY_NEURAL, answer_mode_word
from core.device_route import DEVICE_CPU, DEVICE_INSTALL, device_route_plan, device_cpu_absent
from providers.python.device_route_boundary_door import (
    DeviceRouteOwner, device_admit, device_request, device_unpack, device_route_held, device_route_keep, device_route_clear,
)

comptime SAMBA_ROUTE_CPU_INFERENCE = 1
comptime SAMBA_ROUTE_PROGRAM = 2
comptime SAMBA_ROUTE_ENTRY = 4


@fieldwise_init
struct SambaRoute(Movable):
    """One operation's view of the stack's resolved route. `entries` is the
    connected training namespace (None for the CPU inference model), `mode`
    the connected mode word, `blocks` the block route word, `key` the table
    key of the image and `entry` its primitive entry."""
    var entries: PythonObject
    var mode: PythonObject
    var blocks: PythonObject
    var key: SambaEntryKey
    var entry: EntryRef
    var flags: Int

    def cpu_inference(self) -> Bool:
        return (self.flags & SAMBA_ROUTE_CPU_INFERENCE) != 0

    def program(self) -> Bool:
        """Whether the stack runs on the Samba stack program: its GPU
        training image published the program entry and it is not the CPU
        inference model (resolved once with the route)."""
        return (self.flags & SAMBA_ROUTE_PROGRAM) != 0

    def primitives(self) raises -> EntryRef:
        """The primitive entry of the route's image. A route whose table
        entry could not be read raises the table's own refusal here, at the
        operation that needs it."""
        if (self.flags & SAMBA_ROUTE_ENTRY) != 0:
            return self.entry
        if self.cpu_inference():
            return samba_ref[SAMBA_TABLE_NEURAL_HOST](self.key)
        return samba_ref[SAMBA_TABLE_TRAINING](self.key)


def samba_process_vendor(context: PythonObject) raises -> Int:
    """This process's vendor word for the device rule and the Samba table
    (0 on a CPU-only installation)."""
    var nk = samba_keys()
    var vendor = py_item(context, nk[].k_vendor)
    var word = samba_vendor_word(String("cpu") if vendor is None else String(vendor))
    if word < 0:
        raise Error("Samba: unsupported native vendor " + String(vendor))
    return word


def _samba_cpu_refuse(requested: Int, vendor: Int) raises:
    """A GPU installation asked for `device="cpu"`. The stack's training
    primitives have a host image, but its blocks' backward has none loaded
    beside the GPU set, so the whole class is refused (the one sentence)."""
    if requested == DEVICE_CPU and vendor != 0:
        device_cpu_absent(String("SambaStack"),
            String("the host training images of its blocks are not loaded beside the GPU set in this installation"))


def samba_device_admit(value: PythonObject, context: PythonObject) raises:
    """Check a `device` value where it is given (the constructor, a restore),
    before any field is published or any file is read: an unknown value, an
    explicit `'gpu'` on a CPU-only installation and an explicit `'cpu'` on a
    GPU installation are refused in the shared words. The default None is one
    identity compare."""
    if value is None:
        return
    var b = native_loaded_module("builtins")
    var vendor = samba_process_vendor(context)
    device_admit(value, vendor, b)
    _samba_cpu_refuse(device_unpack(value, b), vendor)


def _samba_route_build(requested: Int, mode: PythonObject, context: PythonObject) raises -> DeviceRouteOwner:
    """Resolve the training route of one stack (or of the process, for a
    state without a stack). The mode and vendor admission of the connected
    binding is the former `samba_entries` check, word for word."""
    var nk = samba_keys()
    var connected = py_item(context, nk[].k_mode)
    var entries = py_item(context, nk[].k_entries)
    var selected = connected if mode is None else mode
    # lane fx-fast-mask: the stack's word is answered (NEURAL) before it is
    # compared with the connected mode, so an answered `fast` stack runs the
    # connected identical binding.
    var text = String(selected)
    answer_mode_word(text, MODE_FAMILY_NEURAL)
    var want = 1 if text == "identical" else 0
    var vendor = samba_process_vendor(context)
    _samba_cpu_refuse(requested, vendor)
    var plan = device_route_plan(requested, vendor, want)
    if text != String(connected):
        raise Error("Samba: requested numeric mode differs from the connected native mode")
    # The binding's numeric mode and vendor words are fixed per compiled
    # module (bindings/p3_samba_caps_door.mojo).
    var caps = samba_caps(entries)
    if Int(py=caps[SAMBA_CAP_MODE]) != want or Bool(py=caps[SAMBA_CAP_VENDOR] != py_item(context, nk[].k_vendor)):
        raise Error("Samba: native training mode/vendor admission mismatch")
    var owner = DeviceRouteOwner(requested)
    var key = SambaEntryKey(want, vendor)
    owner.route.device = plan.device
    owner.route.mode = want
    owner.route.vendor = vendor
    owner.route.flags = SAMBA_ROUTE_PROGRAM if samba_programs_available(key) else 0
    try:
        owner.route.entry = samba_ref[SAMBA_TABLE_TRAINING](key)
        owner.route.flags = owner.route.flags | SAMBA_ROUTE_ENTRY
    except:
        # Kept unresolved: `SambaRoute.primitives` raises the table's words
        # at the operation that needs the entry.
        owner.route.entry = entry_ref_unresolved()
    owner.images = entries
    owner.scope = PythonObject("cpu" if plan.device == DEVICE_CPU else "gpu")
    owner.table = connected
    return owner^


def _samba_route_build_inference(mode: PythonObject, context: PythonObject) raises -> DeviceRouteOwner:
    """The CPU inference model's route (`SambaInference`): the identical
    `_mojolearn_neural_host` image, whose binding read loads and admits the
    image that publishes the key's entries."""
    var nk = samba_keys()
    var owner = DeviceRouteOwner(DEVICE_CPU)
    var key = SambaEntryKey(1, 0)
    owner.images = _model_bindings(PythonObject("cpu"), "mamba3", py_item(context, nk[].k_neural), PythonObject("identical"))["primitives"]
    owner.route.device = DEVICE_CPU
    owner.route.mode = 1
    owner.route.vendor = 0
    owner.route.flags = SAMBA_ROUTE_CPU_INFERENCE
    try:
        owner.route.entry = samba_ref[SAMBA_TABLE_NEURAL_HOST](key)
        owner.route.flags = owner.route.flags | SAMBA_ROUTE_ENTRY
    except:
        owner.route.entry = entry_ref_unresolved()
    owner.scope = PythonObject("cpu")
    owner.table = PythonObject("identical")
    return owner^


def _samba_route_view(owner: PythonObject) raises -> SambaRoute:
    var native = owner.downcast_value_ptr[DeviceRouteOwner]()
    var route = native[].route
    var entries = PythonObject(None)
    if (route.flags & SAMBA_ROUTE_CPU_INFERENCE) == 0:
        entries = native[].images
    var view = SambaRoute(entries, native[].table, native[].scope, SambaEntryKey(route.mode, route.vendor),
        route.entry, route.flags)
    _ = owner  # the owner pins the image through the reads above
    return view^


def _samba_route_resolve(model: PythonObject, fields: PythonObject, mode: PythonObject, connected: PythonObject,
    context: PythonObject, admit: Bool) raises -> DeviceRouteOwner:
    """A route miss: the one read of the stack's `device` and of the CPU
    inference class flag, both admitted here, at the read (`admit` False:
    the constructing door wrote the fields itself a moment ago)."""
    var nk = samba_keys()
    if admit:
        samba_admit_model_fields(model, [String("device"), String("_native_cpu_inference")], context)
    if Bool(py=py_get(fields, nk[].k__native_cpu_inference, PythonObject(False))):
        var inference = _samba_route_build_inference(mode, context)
        inference.mode = mode
        inference.default_mode = connected
        inference.context = context
        inference.resolved = True
        return inference^
    var built = _samba_route_build(device_request(fields, native_loaded_module("builtins")), mode, context)
    built.mode = mode
    built.default_mode = connected
    built.context = context
    built.resolved = True
    return built^


def samba_route(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> SambaRoute:
    """The stack's route for this operation. `fields` is the stack's
    instance dictionary. The device is read only on a miss (first operation
    after construction or restore, a changed numeric mode object, another
    connection context)."""
    var nk = samba_keys()
    var mode = py_get(fields, nk[].k_numeric_mode)
    var connected = py_item(context, nk[].k_mode)
    var owner = device_route_held(model, mode, connected, context)
    if owner is None:
        owner = device_route_keep(model, _samba_route_resolve(model, fields, mode, connected, context, True))
    return _samba_route_view(owner)


def samba_route_publish(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> SambaRoute:
    """Construction and restoration: resolve the stack's route from the
    fields the constructing door just wrote, replacing any route an earlier
    run of the constructor left (a refusal leaves no route)."""
    var nk = samba_keys()
    device_route_clear(model)
    var mode = py_get(fields, nk[].k_numeric_mode)
    var connected = py_item(context, nk[].k_mode)
    var owner = device_route_keep(model, _samba_route_resolve(model, fields, mode, connected, context, False))
    return _samba_route_view(owner)


def samba_route_process(mode: PythonObject, context: PythonObject) raises -> SambaRoute:
    """The process route: what a state operation without a stack runs on
    (validate, from_state before its stack exists, the distributed gradient
    and apply entries). No estimator, so no device to read: this
    installation's device, admitted for the state's mode word."""
    var built = _samba_route_build(DEVICE_INSTALL, mode, context)
    var view = SambaRoute(built.images, built.table, built.scope,
        SambaEntryKey(built.route.mode, built.route.vendor), built.route.entry, built.route.flags)
    _ = built^
    return view^
