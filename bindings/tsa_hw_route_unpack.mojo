# SPDX-License-Identifier: Apache-2.0
"""ExponentialSmoothing's device route on the one route owner, and the
lifetime owner of its resident device copy.

The route is a `DeviceRouteOwner` (providers/python/device_route_boundary_door.mojo,
rule and words core/device_route.mojo): slot `entry` is the resolved
Holt-Winters operations entry (core/tsa_entry_table.mojo), `flags` its
TSA_HAS_* bits, `images` the module that published it. `tsa_hw_route` is the
per-call read: one record read and three pointer compares (the numeric_mode
object, the default mode object when the mode is None, the connection
context); no call reads or compares the device. A miss resolves once: the
selected binding's table, the CPU table for `device="cpu"` beside a GPU set,
or the forecast inference image's table for the explicit host class.

`TsaHwResidentOwner` is not a route: it owns one resident handle, holds the
entry the handle was minted through and releases it through that entry when
the record dies (the former Python release capsule).

Python objects appear here only as pins and identities; RUN is
core/tsa_hw_run.mojo.
"""
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from bindings.mode_answer_door import answer_mode_object
from bindings.native_loaded_module_door import native_interpreter_finalizing
from bindings.native_owner_registry_pack import borrowed_native_owner
from bindings.native_owner_registry_unpack import register_native_owner_type
from bindings.py_balanced import py_call, py_type, py_get, py_item, py_setitem
from bindings.py_type_registry import py_type_absent, py_type_attach
from bindings.tsa_keys_unpack import tsa_keys
from core.device_route import (
    DeviceRoute, DEVICE_CPU, DEVICE_INSTALL, device_mode_code, device_route_plan, device_vendor_code,
)
from core.entry_table import EntryRef, entry_ref_call, entry_ref_same
from core.mode_answer import MODE_FAMILY_TSA, fast_is_answered_by_identical
from core.tsa_entry_table import TSA_OP_HW_RESIDENT_RELEASE, TSA_TABLE_FORECAST
from core.tsa_hw_run import tsa_hw_route_resolve
from providers.python.device_route_boundary_door import (
    DeviceRouteOwner, device_admit, device_request, device_route_held, device_route_keep, device_route_value,
)


struct TsaHwResidentOwner(Movable, Writable):
    """One resident handle and the entry that minted it. `release` is the
    operation id of that entry's table that returns the handle (the
    Holt-Winters table's, or the ARIMA table's: bindings/tsa_arima_route_unpack.mojo).
    `images` pins the extension image the entry lives in until the handle is
    released."""
    var ops: EntryRef
    var release: Int
    var handle: Int
    var images: PythonObject

    def __init__(out self, ops: EntryRef, release: Int, handle: Int, images: PythonObject):
        self.ops = ops
        self.release = release
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
                _ = entry_ref_call(self.ops, self.release, List[Int](), words, List[Float64]())
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


def register_tsa_hw_resident_owner(mut module: PythonModuleBuilder) raises:
    """Register the lifetime owner type with the module that allocates it; a
    second initialization in one process attaches the existing type object
    (bindings/py_type_registry.mojo)."""
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[TsaHwResidentOwner]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[TsaHwResidentOwner]("_TsaHwResidentOwner")
        return
    py_type_attach(module.module, "_TsaHwResidentOwner", existing)


def register_tsa_hw_resident_owner_identity(published: PythonObject) raises:
    register_native_owner_type[TsaHwResidentOwner](published, tsa_hw_resident_owner_pins)


def tsa_hw_resident_new(route: DeviceRoute, handle: Int, images: PythonObject) raises -> PythonObject:
    """The lifetime owner of `handle`, minted through `route`'s entry."""
    return PythonObject(alloc=TsaHwResidentOwner(route.entry, TSA_OP_HW_RESIDENT_RELEASE, handle, images))


def tsa_resident_new(ops: EntryRef, release: Int, handle: Int, images: PythonObject) raises -> PythonObject:
    """The lifetime owner of a `handle` minted through `ops`, returned by
    that entry's `release` operation (`p [handle]`)."""
    return PythonObject(alloc=TsaHwResidentOwner(ops, release, handle, images))


def tsa_resident_same_entry(resident: PythonObject, ops: EntryRef) raises -> Bool:
    """The handle was minted through this very entry."""
    return entry_ref_same(resident.downcast_value_ptr[TsaHwResidentOwner]()[].ops, ops)


def tsa_hw_resident_matches(resident: PythonObject, route: DeviceRoute) raises -> Bool:
    """The handle was minted through this route's entry (the same published
    pointer and layout word)."""
    return entry_ref_same(resident.downcast_value_ptr[TsaHwResidentOwner]()[].ops, route.entry)


def tsa_hw_resident_handle(resident: PythonObject) raises -> Int:
    return resident.downcast_value_ptr[TsaHwResidentOwner]()[].handle


def tsa_hw_process_vendor(context: PythonObject) raises -> Int:
    """The vendor word of this installation's selected Holt-Winters binding
    (0 CPU only; -1 when it did not load: its own refusal is said by the
    first operation, not by the device check)."""
    var nk = tsa_keys()
    return device_vendor_code(String(py_item(context, nk[].k_hw_install_vendor)))


def tsa_hw_device_admit(value: PythonObject, context: PythonObject, b: PythonObject) raises:
    """The constructor's check of `device` (the shared `device_admit`): an
    unknown value and an explicit `gpu` on a CPU-only installation are
    refused before any field changes."""
    device_admit(value, tsa_hw_process_vendor(context), b)


def _tsa_hw_cells_route(device: Int, cpu_table: Bool, expected_vendor: Int, mode_code: Int, cells: PythonObject,
    b: PythonObject, mut owner: DeviceRouteOwner) raises:
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
    var vendor = device_vendor_code(String(py_item(cells, nk[].k_hw_cell_vendor)))
    if cpu_table and vendor != 0:
        raise Error("RuntimeError: device='cpu' requires the CPU identical binding")
    if not cpu_table and expected_vendor >= 0 and vendor != expected_vendor:
        raise Error("RuntimeError: Holt-Winters native vendor differs from selected execution")
    owner.route = tsa_hw_route_resolve(device, family, mode, vendor)
    owner.images = py_item(cells, nk[].k_hw_cell_module)


def _tsa_hw_route_build(device: Int, host_class: Bool, mode: PythonObject,
    default_mode: PythonObject, context: PythonObject, b: PythonObject) raises -> DeviceRouteOwner:
    var nk = tsa_keys()
    var owner = DeviceRouteOwner(device)
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
        owner.route = tsa_hw_route_resolve(DEVICE_CPU, TSA_TABLE_FORECAST, 1, 0)
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
        var process_vendor = tsa_hw_process_vendor(context)
        var plan = device_route_plan(device, process_vendor, device_mode_code(String(py=selected)))
        if plan.cpu_table:
            var cpu = py_get(context, nk[].k_hw_cpu_bindings)
            if cpu is None:
                raise Error("ImportError: device='cpu' is not connected in this installation")
            _tsa_hw_cells_route(plan.device, True, 0, 1, py_item(cpu, selected), b, owner)
        else:
            _tsa_hw_cells_route(plan.device, False, plan.expected_vendor, -1,
                py_item(py_item(context, nk[].k_hw_bindings), selected), b, owner)
    owner.mode = mode
    owner.default_mode = default_mode
    owner.context = context
    owner.resolved = True
    return owner^


@fieldwise_init
struct TsaHwCall(Movable):
    """One call's view of the route: the resolved words (a copy of a few
    machine words) and the owner object, which the caller keeps alive
    through its entry call (it pins the image)."""
    var route: DeviceRoute
    var owner: PythonObject


def tsa_hw_route(model: PythonObject, fields: PythonObject, host_class: Bool,
    context: PythonObject, b: PythonObject) raises -> TsaHwCall:
    """The estimator's route: the held one when it was resolved for this
    numeric_mode object, this default mode object and this connection
    context, else a fresh resolution (the device request is read once, here)
    kept for the next call."""
    var nk = tsa_keys()
    var mode = py_get(fields, nk[].k_numeric_mode)
    var default_mode = PythonObject(None)
    if mode is None:
        default_mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    var owner = device_route_held(model, mode, default_mode, context)
    if owner is None:
        owner = device_route_keep(model,
            _tsa_hw_route_build(device_request(fields, b), host_class, mode, default_mode, context, b))
    return TsaHwCall(device_route_value(owner), owner)


def tsa_function_route(context: PythonObject, want_cpu: Bool, b: PythonObject) raises -> TsaHwCall:
    """The route of the stationarity functions (`kpss_test`, `select_d`):
    no estimator holds it, so the resolved owner is kept per mode object in
    the connection context's `tsa_function_routes` table. A hit is the
    default-mode read and one table read (what the former by-name entry
    table cost, without its entry lookup). `want_cpu` (AutoARIMA with
    `device="cpu"`) takes the CPU table loaded beside a GPU set; in a
    CPU-only process the selected table already is the CPU one."""
    var nk = tsa_keys()
    var routes = py_item(context, nk[].k_tsa_function_routes)
    var key = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    var cpu = PythonObject(None)
    if want_cpu:
        cpu = py_get(context, nk[].k_hw_cpu_bindings)
        if cpu is not None:
            key = nk[].k_hw_cpu_bindings
    var owner = py_get(routes, key)
    if owner is None:
        var built = DeviceRouteOwner(DEVICE_INSTALL)
        if cpu is not None:
            _tsa_hw_cells_route(DEVICE_CPU, True, 0, 1, py_item(cpu, PythonObject("identical")), b, built)
        else:
            _tsa_hw_cells_route(DEVICE_INSTALL, False, -1, -1, py_item(py_item(context, nk[].k_hw_bindings), key), b, built)
        built.context = context
        built.resolved = True
        owner = PythonObject(alloc=built^)
        py_setitem(routes, key, owner)
    return TsaHwCall(device_route_value(owner), owner)
