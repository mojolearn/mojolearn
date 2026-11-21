# SPDX-License-Identifier: Apache-2.0
"""UNPACK of an estimator's device and numeric mode into its native route owner.

`device="gpu"` (default) or `device="cpu"` is a constructor parameter of Ridge
and LinearRegression. Its value is checked when it is given (constructor,
`set_params`) and read ONCE, at the estimator's first operation, into a
`LinearRouteOwner` kept in the weak model-record table. The owner holds the
resolved entries of the device table or of the CPU table (core/linear_route.mojo)
and the image pins of those tables. Every later fit, predict and score reads
the owner and calls through it: no per-call device read, table slot lookup,
schema String or vendor question.

What a call still compares, as before this owner existed: the estimator's
`numeric_mode` object and, when that is None, the process default mode
object (two pointer compares against the owner's pins). A changed mode
resolves again; `set_params` and a new `__init__` drop the owner.

The owner replaces `classical_entries` (bindings/classical_native_pack.mojo)
and `classical_host_entries` (bindings/classical_host_owner_unpack.mojo): the
per-call mode String, the bindings and verdict dictionary reads, the
connection-owner admission and the four `linear_table_entry` slot reads of
fit, predict and score.
"""
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from bindings.py_type_registry import py_type_absent, py_type_attach
from bindings.native_owner_registry_unpack import register_native_owner_type
from bindings.native_owner_registry_pack import borrowed_native_owner
from bindings.py_balanced import py_call, py_type, py_get, py_item
from bindings.py_model_records import records_get, records_put, records_drop, RECORDS_PREPARED
from bindings.classical_keys_unpack import classical_keys
from core.entry_table import EntryRef, entry_ref_word
from core.linear_entry_table import LINEAR_TABLE_X_LINEAR, LINEAR_TABLE_DECOMP, linear_table_entry
from core.linear_metadata import linear_mode_name, linear_vendor_code
from core.linear_route import (
    LinearRoute, LINEAR_DEVICE_GPU, LINEAR_DEVICE_CPU, linear_device_code, linear_device_refusal,
    linear_route_resolve, linear_route_weighted,
)
from providers.python.linear_entry_boundary_door import linear_entry_connection
from x_decomp.descriptor_schema import XD_ABI_NAME, XD_OPERATION_COUNT

comptime _LINEAR_ROUTE_SLOT = "linear_route_v1"


struct LinearRouteOwner(Movable, Writable):
    var route: LinearRoute
    var resolved: Bool
    # Why LinearRegression's TSQR entry is not held (said when a fit needs it).
    var decomp_refusal: String
    # The objects the route was resolved for. Held, so their addresses stay
    # theirs while this owner lives; a call compares identities only.
    var mode: PythonObject
    var default_mode: PythonObject
    var context: PythonObject
    # The entry dictionary and the connection owners of every resolved table:
    # the extension images stay loaded while a call runs with the GIL released.
    var images: PythonObject

    def __init__(out self, device: Int):
        self.route = LinearRoute(device)
        self.resolved = False
        self.decomp_refusal = String("")
        self.mode = PythonObject(None)
        self.default_mode = PythonObject(None)
        self.context = PythonObject(None)
        self.images = PythonObject(None)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("LinearRouteOwner(device=", self.route.device, ", vendor=", self.route.vendor, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def linear_route_owner_pins(value: Int, index: Int) abi("C") -> Int:
    if not value or index < -1 or index > 3:
        return -1
    try:
        var owner = borrowed_native_owner(value)
        var native = owner.downcast_value_ptr[LinearRouteOwner]()
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


def register_linear_route_owner(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[LinearRouteOwner]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[LinearRouteOwner]("_LinearRouteOwner")
        return
    py_type_attach(module.module, "_LinearRouteOwner", existing)


def register_linear_route_owner_identity(published: PythonObject) raises:
    # After PythonModuleBuilder.finalize() only: Mojo 1.2 creates the type
    # object in finalize() (bindings/native_owner_registry.mojo).
    register_native_owner_type[LinearRouteOwner](published, linear_route_owner_pins)


def linear_device_unpack(value: PythonObject, b: PythonObject) raises -> Int:
    """The device code of a constructor value: exact `'gpu'` or `'cpu'`.
    Anything else is refused in words, before any state changes. `b` is the
    builtins module the calling door already admitted."""
    if py_type(value) is b.str:
        var code = linear_device_code(String(value))
        if code >= 0:
            return code
    linear_device_refusal(String(py_call(b.repr, value)))
    return LINEAR_DEVICE_GPU


def linear_route_clear(model: PythonObject) raises:
    """Forget the estimator's route (a new `__init__`, `set_params`): its next
    operation reads the device and the mode again."""
    records_drop(RECORDS_PREPARED, model, _LINEAR_ROUTE_SLOT)


def _linear_route_build(device: Int, ridge: Bool, mode: PythonObject,
    default_mode: PythonObject, context: PythonObject, b: PythonObject) raises -> LinearRouteOwner:
    var nk = classical_keys()
    var owner = LinearRouteOwner(device)
    var selected = mode
    if mode is None:
        selected = default_mode
    var explicit = Bool(py=py_get(context, nk[].k_classical_explicit_cpu, False))
    var entries = PythonObject(None)
    var expected_vendor = 0
    var expected_mode = 1
    if explicit:
        # The explicit Host* classes (mojolearn.host_model): CPU identical by
        # class, whatever `device` their inherited constructor stored. Their
        # helper family is selected by the raw mode object, as before.
        if mode is not None and (py_type(mode) is not b.str or String(mode) != "identical"):
            raise Error("the host model runs identical only; this model was saved with a different numeric_mode")
        owner.route.device = LINEAR_DEVICE_CPU
        entries = py_item(py_item(context, nk[].k_classical_bindings), selected)
    else:
        if py_type(selected) is not b.str:
            raise Error("numeric_mode must be fast, identical or None")
        var mode_name = linear_mode_name(String(selected))
        expected_mode = 1 if mode_name == "identical" else 0
        var process_vendor = linear_vendor_code(String(py_item(context, nk[].k_vendor)))
        if device == LINEAR_DEVICE_CPU and process_vendor != 0:
            # A GPU installation: the CPU table of the host binaries, loaded
            # beside the device table at import
            # (python/mojolearn/_native_classical_bridge.py connect_cpu_device).
            if expected_mode != 1:
                raise Error("ValueError: device='cpu' runs in identical mode only; numeric_mode='fast' is for Apple GPUs")
            var cpu = py_get(context, "classical_cpu_bindings")
            if cpu is None:
                raise Error("ImportError: device='cpu' is not connected in this installation")
            entries = py_item(cpu, PythonObject(mode_name))
        else:
            # DECISION OWED (lane pilot-ridge-device): in a CPU-only process
            # only the CPU table exists and the default `device="gpu"` keeps
            # running on it, as every default-constructed estimator, the CPU
            # verification column and the CPU-only wheels do today. The plan
            # (docs/plans/PYTHON_AT_THE_BOUNDARY.md section 10.3) refuses
            # `device="gpu"` there in words; that refusal is one line here
            # (`device == LINEAR_DEVICE_GPU and process_vendor == 0`) once the
            # default for CPU-only installs is decided.
            entries = py_item(py_item(context, nk[].k_classical_bindings), PythonObject(mode_name))
            expected_vendor = process_vendor
    if py_type(entries) is b.str:
        raise Error("ImportError: " + String(entries))
    var connection = linear_entry_connection(entries)
    owner.route = linear_route_resolve(owner.route.device, connection.key, expected_vendor, expected_mode, explicit)
    var weighted_images = PythonObject(None)
    var decomp_images = PythonObject(None)
    if ridge:
        var weighted = linear_entry_connection(entries, LINEAR_TABLE_X_LINEAR, True)
        linear_route_weighted(owner.route, connection.key, weighted.key)
        weighted_images = weighted.images
    else:
        # LinearRegression's TSQR table. An unbuilt or foreign one is refused
        # only by a fit whose plan takes the TSQR route, in the same words.
        var decomp = py_get(entries, "decomp")
        if py_type(decomp) is not b.dict:
            owner.decomp_refusal = "ImportError: native LinearRegression TSQR entries unavailable: " + String(decomp)
        else:
            try:
                var decomp_connection = linear_entry_connection(decomp, LINEAR_TABLE_DECOMP)
                if (decomp_connection.key.mode != connection.key.mode
                        or decomp_connection.key.vendor != connection.key.vendor):
                    owner.decomp_refusal = "native LinearRegression TSQR execution key differs"
                else:
                    var entry = linear_table_entry[LINEAR_TABLE_DECOMP](decomp_connection.key, 0,
                        String(XD_ABI_NAME), XD_OPERATION_COUNT)
                    owner.route.decomp = EntryRef(entry, entry_ref_word(decomp_connection.key.mode,
                        decomp_connection.key.vendor, XD_OPERATION_COUNT))
                    owner.route.has_decomp = True
                    decomp_images = decomp_connection.images
            except error:
                owner.decomp_refusal = String(error)
    owner.mode = mode
    owner.default_mode = default_mode
    owner.context = context
    owner.images = Python.tuple(entries, connection.images, weighted_images, decomp_images)
    owner.resolved = True
    return owner^


@fieldwise_init
struct LinearRouteCall(Movable):
    """One operation's route: the resolved entries by value, and the owner
    whose pins keep their images loaded until the operation returns."""
    var route: LinearRoute
    var owner: PythonObject


def linear_route_call(model: PythonObject, fields: PythonObject, ridge: Bool,
    context: PythonObject, b: PythonObject) raises -> LinearRouteCall:
    """The estimator's route for this operation. A hit is one record read and
    three pointer compares; a miss (first operation, a changed numeric mode,
    an instance built without its constructor) resolves and stores it. `b`
    is the builtins module the calling door already admitted (read on a miss
    only)."""
    var nk = classical_keys()
    var mode = py_get(fields, nk[].k_numeric_mode)
    var default_mode = PythonObject(None)
    if mode is None:
        default_mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    var owner = records_get(RECORDS_PREPARED, model, _LINEAR_ROUTE_SLOT)
    if owner is not None:
        var native = owner.downcast_value_ptr[LinearRouteOwner]()
        if (native[].resolved and native[].mode is mode and native[].default_mode is default_mode
                and native[].context is context):
            return LinearRouteCall(native[].route, owner)
    # The one read of `device`. An instance without the field (built without
    # its constructor) runs the default.
    var device = LINEAR_DEVICE_GPU
    var given = py_get(fields, "device")
    if given is not None:
        device = linear_device_unpack(given, b)
    var fresh = PythonObject(alloc=_linear_route_build(device, ridge, mode, default_mode, context, b))
    # An estimator that takes no weak reference keeps no owner: it resolves
    # at every operation (records_put answers False and stores nothing).
    _ = records_put(RECORDS_PREPARED, model, _LINEAR_ROUTE_SLOT, fresh)
    return LinearRouteCall(fresh.downcast_value_ptr[LinearRouteOwner]()[].route, fresh)


def linear_route_decomp_refuse(owner: PythonObject) raises:
    """Raise why this estimator's route holds no TSQR entry (failure path of
    a LinearRegression fit whose plan takes the TSQR route)."""
    var text = owner.downcast_value_ptr[LinearRouteOwner]()[].decomp_refusal.copy()
    if text.byte_length() == 0:
        text = String("ImportError: native LinearRegression TSQR entries unavailable")
    raise Error(text)
