# SPDX-License-Identifier: Apache-2.0
"""The per-estimator device route owner of ExponentialSmoothing, and the
lifetime owner of its resident device copy.

`TsaHwRouteOwner` is a registered native owner kept in the weak model-record
table (slot `tsa_hw_route_v1`): the resolved route (core/tsa_route.mojo), the
image pin, and the objects it was resolved for. `tsa_hw_route_call` is the
per-call read: one record read and three pointer compares (the numeric_mode
object, the default mode object when the mode is None, the connection
context). A miss resolves once: the selected binding's table, the CPU table
for `device="cpu"` on a GPU installation, or the forecast inference image's
table for the explicit host class.

`TsaHwResidentOwner` owns one resident handle: it holds the entry the handle
was minted through and releases it through that entry when the record dies
(the former Python release capsule).

Python objects appear here only as pins and identities; RUN is the call
through `route.ops` over typed request spans.
"""
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from bindings.mode_answer_door import answer_mode_object
from bindings.native_loaded_module_door import native_interpreter_finalizing
from bindings.native_owner_registry_pack import borrowed_native_owner
from bindings.native_owner_registry_unpack import register_native_owner_type
from bindings.py_balanced import py_call, py_type, py_get, py_item
from bindings.py_model_records import records_get, records_put, records_drop, RECORDS_PREPARED
from bindings.py_type_registry import py_type_absent, py_type_attach
from bindings.tsa_keys_unpack import tsa_keys
from core.entry_table import EntryRef, entry_ref_call, entry_ref_same
from core.linear_metadata import linear_vendor_code
from core.linear_route import (
    LINEAR_DEVICE_CPU, LINEAR_DEVICE_INSTALL, linear_device_code, linear_device_refusal,
    linear_device_installed,
)
from core.mode_answer import MODE_FAMILY_TSA, fast_is_answered_by_identical
from core.tsa_entry_table import TSA_OP_HW_RESIDENT_RELEASE, TSA_TABLE_FORECAST
from core.tsa_route import TsaHwRoute, tsa_hw_route_resolve

comptime _TSA_HW_ROUTE_SLOT = "tsa_hw_route_v1"


struct TsaHwRouteOwner(Movable, Writable):
    var route: TsaHwRoute
    var resolved: Bool
    var mode: PythonObject
    var default_mode: PythonObject
    var context: PythonObject
    var images: PythonObject

    def __init__(out self, device: Int):
        self.route = TsaHwRoute(device)
        self.resolved = False
        self.mode = PythonObject(None)
        self.default_mode = PythonObject(None)
        self.context = PythonObject(None)
        self.images = PythonObject(None)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("TsaHwRouteOwner(device=", self.route.device, ", vendor=", self.route.vendor, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def tsa_hw_route_owner_pins(value: Int, index: Int) abi("C") -> Int:
    if not value or index < -1 or index > 3:
        return -1
    try:
        var owner = borrowed_native_owner(value)
        var native = owner.downcast_value_ptr[TsaHwRouteOwner]()
        var result = 4
        if index == 0:
            result = Int(native[].mode._obj_ptr)
        elif index == 1:
            result = Int(native[].default_mode._obj_ptr)
        elif index == 2:
            result = Int(native[].context._obj_ptr)
        elif index == 3:
            result = Int(native[].images._obj_ptr)
        _ = owner
        return result
    except:
        return -1


struct TsaHwResidentOwner(Movable, Writable):
    """One resident handle and the entry that minted it. `images` pins the
    extension image the entry lives in until the handle is released."""
    var ops: EntryRef
    var handle: Int
    var images: PythonObject

    def __init__(out self, ops: EntryRef, handle: Int, images: PythonObject):
        self.ops = ops
        self.handle = handle
        self.images = images

    def write_to(self, mut writer: Some[Writer]):
        writer.write("TsaHwResidentOwner(handle=", self.handle, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def __deinit__(deinit self):
        if self.handle < 1:
            return
        try:
            # Quiet, as the former release capsule: at interpreter exit the
            # binding is being torn down and has nothing left to release.
            if not native_interpreter_finalizing():
                var words: List[Int] = [self.handle]
                _ = entry_ref_call(self.ops, TSA_OP_HW_RESIDENT_RELEASE, List[Int](), words, List[Float64]())
        except:
            pass


def tsa_hw_resident_owner_pins(value: Int, index: Int) abi("C") -> Int:
    if not value or index < -1 or index > 0:
        return -1
    try:
        var owner = borrowed_native_owner(value)
        var native = owner.downcast_value_ptr[TsaHwResidentOwner]()
        var result = 1 if index == -1 else Int(native[].images._obj_ptr)
        _ = owner
        return result
    except:
        return -1


def register_tsa_hw_route_owners(mut module: PythonModuleBuilder) raises:
    """Register both owner types with the module that allocates them; a
    second initialization in one process attaches the existing type objects
    (bindings/py_type_registry.mojo)."""
    var existing = PythonObject(None)
    var route_present = True
    try:
        existing = lookup_py_type_object[TsaHwRouteOwner]()
    except error:
        if not py_type_absent(error):
            raise error
        route_present = False
    if route_present:
        py_type_attach(module.module, "_TsaHwRouteOwner", existing)
    else:
        _ = module.add_type[TsaHwRouteOwner]("_TsaHwRouteOwner")
    var resident_present = True
    try:
        existing = lookup_py_type_object[TsaHwResidentOwner]()
    except error:
        if not py_type_absent(error):
            raise error
        resident_present = False
    if resident_present:
        py_type_attach(module.module, "_TsaHwResidentOwner", existing)
    else:
        _ = module.add_type[TsaHwResidentOwner]("_TsaHwResidentOwner")


def register_tsa_hw_route_owner_identities(published: PythonObject) raises:
    register_native_owner_type[TsaHwRouteOwner](published, tsa_hw_route_owner_pins)
    register_native_owner_type[TsaHwResidentOwner](published, tsa_hw_resident_owner_pins)


def tsa_hw_resident_new(route: TsaHwRoute, handle: Int, images: PythonObject) raises -> PythonObject:
    """The lifetime owner of `handle`, minted through `route`'s entry."""
    return PythonObject(alloc=TsaHwResidentOwner(route.ops, handle, images))


def tsa_hw_resident_matches(resident: PythonObject, route: TsaHwRoute) raises -> Bool:
    """The handle was minted through this route's entry (the same published
    pointer and layout word)."""
    return entry_ref_same(resident.downcast_value_ptr[TsaHwResidentOwner]()[].ops, route.ops)


def tsa_hw_resident_handle(resident: PythonObject) raises -> Int:
    return resident.downcast_value_ptr[TsaHwResidentOwner]()[].handle


def tsa_device_unpack(value: PythonObject, b: PythonObject) raises -> Int:
    """The request word of a `device` value: None is the installation's
    device, the exact words `gpu` and `cpu` their codes; anything else is
    refused in the linear family's words."""
    if value is None:
        return LINEAR_DEVICE_INSTALL
    if py_type(value) is b.str:
        var code = linear_device_code(String(value))
        if code >= 0:
            return code
    linear_device_refusal(String(py_call(b.repr, value)))
    return LINEAR_DEVICE_INSTALL


def tsa_hw_device_admit(value: PythonObject, context: PythonObject, b: PythonObject) raises:
    """The constructor's check of `device`: an unknown value and an explicit
    `gpu` on a CPU-only installation are refused before any field changes.
    The default None is one identity compare and reads nothing."""
    var device = tsa_device_unpack(value, b)
    if device == LINEAR_DEVICE_INSTALL or device == LINEAR_DEVICE_CPU:
        return
    var nk = tsa_keys()
    _ = linear_device_installed(device, linear_vendor_code(String(py_item(context, nk[].k_hw_install_vendor))))


def tsa_hw_route_clear(model: PythonObject) raises:
    """Forget the estimator's route (the constructor ran again)."""
    records_drop(RECORDS_PREPARED, model, _TSA_HW_ROUTE_SLOT)


def _tsa_hw_cells_route(device: Int, cells: PythonObject, b: PythonObject, mut owner: TsaHwRouteOwner) raises:
    # `cells` is one connected binding of python/mojolearn/_native_tsa_bridge.py:
    # the module (the image pin) and the table key it reported at import, or
    # the text of why it did not load.
    var nk = tsa_keys()
    if py_type(cells) is b.str:
        raise Error(String(cells))
    if py_type(cells) is not b.dict:
        raise Error("TypeError: the Holt-Winters binding connection must be an exact dictionary")
    var family = Int(py=py_item(cells, nk[].k_hw_cell_family))
    var mode = Int(py=py_item(cells, nk[].k_hw_cell_mode))
    var vendor = linear_vendor_code(String(py_item(cells, nk[].k_hw_cell_vendor)))
    owner.route = tsa_hw_route_resolve(device, family, mode, vendor)
    owner.images = py_item(cells, nk[].k_hw_cell_module)


def _tsa_hw_route_build(device: Int, host_class: Bool, mode: PythonObject,
    default_mode: PythonObject, context: PythonObject, b: PythonObject) raises -> TsaHwRouteOwner:
    var nk = tsa_keys()
    var owner = TsaHwRouteOwner(device)
    if host_class:
        # The explicit host inference class: the loaded forecast inference
        # image, identical only. lane fx-fast-mask: a model saved under
        # `fast` where the identical set answered it (TSA) holds identical
        # state; the host runs it.
        if mode is not None and (py_type(mode) is not b.str or (String(mode) != "identical" and not (
                String(mode) == "fast" and fast_is_answered_by_identical(MODE_FAMILY_TSA)))):
            raise Error("ValueError: mojolearn: " + String("ExponentialSmoothing")
                        + " runs IDENTICAL only on the host")
        var module = py_get(py_item(context, nk[].k_modules), py_item(context, nk[].k_hw_host_module))
        if module is None:
            raise Error("RuntimeError: mojolearn: the host " + String("ExponentialSmoothing")
                        + " has no loaded forecast binding; load the model with its own load()")
        if py_type(module) is not py_type(b):
            raise Error("TypeError: host forecast requires its native module")
        owner.route = tsa_hw_route_resolve(LINEAR_DEVICE_CPU, TSA_TABLE_FORECAST, 1, 0)
        owner.images = module
    else:
        var selected = mode
        if mode is None:
            selected = default_mode
        if py_type(selected) is not b.str:
            raise Error("ValueError: numeric_mode must be fast, identical or None")
        selected = py_call(py_call(selected.strip).lower)
        var mode_word = String(py=selected)
        if mode_word != "identical" and mode_word != "fast":
            raise Error("ValueError: numeric_mode must be fast, identical or None")
        answer_mode_object(selected, mode_word, MODE_FAMILY_TSA)
        var process_vendor = linear_vendor_code(String(py_item(context, nk[].k_hw_install_vendor)))
        var resolved_device = linear_device_installed(device, process_vendor)
        if resolved_device == LINEAR_DEVICE_CPU and process_vendor != 0:
            if String(py=selected) != "identical":
                raise Error("ValueError: device='cpu' runs in identical mode only; numeric_mode='fast' is for Apple GPUs")
            var cpu = py_get(context, nk[].k_hw_cpu_bindings)
            if cpu is None:
                raise Error("ImportError: device='cpu' is not connected in this installation")
            _tsa_hw_cells_route(resolved_device, py_item(cpu, selected), b, owner)
        else:
            _tsa_hw_cells_route(resolved_device, py_item(py_item(context, nk[].k_hw_bindings), selected), b, owner)
    owner.mode = mode
    owner.default_mode = default_mode
    owner.context = context
    owner.resolved = True
    return owner^


@fieldwise_init
struct TsaHwRouteCall(Movable):
    """One call's view of the route: the resolved words (a copy of a few
    machine words) and the owner object, which the caller keeps alive
    through its entry call (it pins the image)."""
    var route: TsaHwRoute
    var owner: PythonObject


def tsa_hw_route_call(model: PythonObject, fields: PythonObject, host_class: Bool,
    context: PythonObject, b: PythonObject) raises -> TsaHwRouteCall:
    """The estimator's route: the held one when it was resolved for this
    numeric_mode object, this default mode object and this connection
    context, else a fresh resolution kept for the next call."""
    var nk = tsa_keys()
    var mode = py_get(fields, nk[].k_numeric_mode)
    var default_mode = PythonObject(None)
    if mode is None:
        default_mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    var owner = records_get(RECORDS_PREPARED, model, _TSA_HW_ROUTE_SLOT)
    if owner is not None:
        var native = owner.downcast_value_ptr[TsaHwRouteOwner]()
        if (native[].resolved and native[].mode is mode and native[].default_mode is default_mode
                and native[].context is context):
            return TsaHwRouteCall(native[].route, owner)
    var device = tsa_device_unpack(py_get(fields, nk[].k_device), b)
    var fresh = PythonObject(alloc=_tsa_hw_route_build(device, host_class, mode, default_mode, context, b))
    _ = records_put(RECORDS_PREPARED, model, _TSA_HW_ROUTE_SLOT, fresh)
    return TsaHwRouteCall(fresh.downcast_value_ptr[TsaHwRouteOwner]()[].route, fresh)
