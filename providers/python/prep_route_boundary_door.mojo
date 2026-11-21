# SPDX-License-Identifier: Apache-2.0
"""UNPACK of a scaler's device into its native route owner (lane dw-prep).

`device=None` (default: this installation's device), `device="gpu"` or
`device="cpu"` is a constructor parameter of StandardScaler, MinMaxScaler,
MaxAbsScaler and RobustScaler. The rule and the words are the linear pilot's
(core/linear_route.mojo): the value is stored as given, checked where it is
given (constructor, `set_params`) and read ONCE, at the estimator's first
data operation, into a `PrepRouteOwner` kept in the weak model-record table.

What a route is for this family: the connected context the doors run in.
The scaler doors select every compiled entry (the scaler kernel table and
the x_prep transport table of core/prep_entry_table.mojo, and the image's
capabilities) from `context["scaler_bindings"]`, so the device is the choice
between two connected contexts: the process's own, and the CPU one that
import connects beside it on a GPU installation (`prep_cpu_context`: the
host images of the same bindings, as a CPU-only installation connects them).
The owner holds the context it resolved to. Every later fit, transform and
partial_fit reads the owner (one record read, one pointer compare) and runs
in that context: no public call reads `device`, tests the vendor or looks a
class up in the explicit host table.

The explicit Host* classes (mojolearn.host_model) resolve through the same
owner to their own host context, by class, whatever `device` their inherited
constructor stored; that replaces the two dictionary reads per call of
`scaler_host_contexts[type(model)]` in the data operations.

Resolution is a metadata operation with no independent numerical work to
split; it runs at an estimator's first data operation, never per row.
"""
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from bindings.py_type_registry import py_type_absent, py_type_attach
from bindings.native_owner_registry_unpack import register_native_owner_type
from bindings.native_owner_registry_pack import borrowed_native_owner
from bindings.py_balanced import py_call, py_type, py_get, py_fields
from bindings.py_model_records import records_get, records_put, records_drop, RECORDS_PREPARED
from bindings.prep_entry_cache_unpack import prep_entry_text
from core.linear_route import (
    LINEAR_DEVICE_GPU, LINEAR_DEVICE_CPU, LINEAR_DEVICE_INSTALL, linear_device_code, linear_device_refusal,
    linear_device_installed,
)
from x_prep.transport_abi import xp_vendor_word

comptime _PREP_ROUTE_SLOT = "prep_route_v1"


struct PrepRouteOwner(Movable, Writable):
    # The device the route resolved to (LINEAR_DEVICE_GPU or LINEAR_DEVICE_CPU).
    var device: Int
    var resolved: Bool
    # The connected context the route was resolved for (the door's own,
    # compared by identity per call) and the one the estimator runs in. Both
    # are import-time dictionaries; neither refers to an estimator.
    var root: PythonObject
    var context: PythonObject

    def __init__(out self):
        self.device = LINEAR_DEVICE_GPU
        self.resolved = False
        self.root = PythonObject(None)
        self.context = PythonObject(None)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("PrepRouteOwner(device=", self.device, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def prep_route_owner_pins(value: Int, index: Int) abi("C") -> Int:
    if not value or index < -1 or index > 1:
        return -1
    try:
        var owner = borrowed_native_owner(value)
        var native = owner.downcast_value_ptr[PrepRouteOwner]()
        var result = 2
        if index == 0:
            result = Int(native[].root._obj_ptr)
        elif index == 1:
            result = Int(native[].context._obj_ptr)
        _ = owner
        return result
    except:
        return -1


def register_prep_route_owner(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[PrepRouteOwner]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[PrepRouteOwner]("_PrepRouteOwner")
        return
    py_type_attach(module.module, "_PrepRouteOwner", existing)


def register_prep_route_owner_identity(published: PythonObject) raises:
    # After PythonModuleBuilder.finalize() only: Mojo 1.2 creates the type
    # object in finalize() (bindings/native_owner_registry.mojo).
    register_native_owner_type[PrepRouteOwner](published, prep_route_owner_pins)


def prep_device_unpack(value: PythonObject, context: PythonObject) raises -> Int:
    """The device code of a constructor value: None (this installation's
    device), or exact `'gpu'` or `'cpu'`. Anything else is refused in the
    pilot's words, before any state changes. The exact `str` type and the
    builtin `repr` are the two cells import connected into `context`
    (python/mojolearn/_native_prep_device.py); they are read only for a
    value that is not None, so the default reads nothing."""
    if value is None:
        return LINEAR_DEVICE_INSTALL
    if py_type(value) is context["prep_device_str"]:
        var code = linear_device_code(String(value))
        if code >= 0:
            return code
    linear_device_refusal(String(py_call(context["prep_device_repr"], value)))
    return LINEAR_DEVICE_INSTALL


def _prep_process_vendor(context: PythonObject) raises -> Int:
    """The selected vendor word of the process (0: a CPU-only process)."""
    return xp_vendor_word(String(context["vendor"]))


def _prep_explicit_host(cls: PythonObject, context: PythonObject) raises -> PythonObject:
    """The explicit host context of `cls` (a Host* class), or None."""
    var hosts = py_get(context, "scaler_host_contexts")
    if hosts is None:
        return PythonObject(None)
    var host = py_get(hosts, cls)
    return host


def prep_device_admit(value: PythonObject, cls: PythonObject, context: PythonObject) raises:
    """Check a device where it is given (constructor, `set_params`): an
    unknown value, and an explicit `'gpu'` in a CPU-only process, are refused
    in words before the estimator changes. The default None is one identity
    compare and reads nothing. The explicit Host* classes run on the CPU by
    class and take any admitted value."""
    if value is None:
        return
    var device = prep_device_unpack(value, context)
    if device != LINEAR_DEVICE_GPU:
        return
    if _prep_explicit_host(cls, context) is not None:
        return
    _ = linear_device_installed(device, _prep_process_vendor(context))


def prep_route_clear(model: PythonObject) raises:
    """Forget the estimator's route (a new `__init__`, `set_params`): its next
    data operation reads the device again."""
    records_drop(RECORDS_PREPARED, model, _PREP_ROUTE_SLOT)


def _prep_route_build(model: PythonObject, context: PythonObject) raises -> PrepRouteOwner:
    var owner = PrepRouteOwner()
    var host = _prep_explicit_host(py_type(model), context)
    if host is not None:
        # mojolearn.host_model's classes: their own CPU context, by class.
        owner.device = LINEAR_DEVICE_CPU
        owner.context = host
    else:
        # The one read of `device`. None, and an instance without the field
        # (built without its constructor, or pickled before the parameter
        # existed), run this installation's device.
        var device = prep_device_unpack(py_get(py_fields(model), "device"), context)
        var process_vendor = _prep_process_vendor(context)
        # An explicit 'gpu' in a CPU-only process is refused in words here too
        # (an instance that reached this point without its constructor's check).
        owner.device = linear_device_installed(device, process_vendor)
        if owner.device == LINEAR_DEVICE_CPU and process_vendor != 0:
            # A GPU installation: the CPU context connected beside the device
            # one at import (the bridges' `connect_cpu_device`).
            var cpu = py_get(context, "prep_cpu_context")
            if cpu is None:
                raise Error("ImportError: device='cpu' is not connected in this installation")
            if py_type(cpu) is context["prep_device_str"]:
                raise Error("ImportError: " + String(cpu))
            owner.context = cpu
        else:
            # The process's own context: the GPU one for a GPU request, the
            # CPU one in a CPU-only process.
            owner.context = context
    owner.root = context
    owner.resolved = True
    return owner^


def prep_route_context(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """The connected context this estimator's data operation runs in. A hit
    is one record read and one pointer compare; a miss (the estimator's first
    data operation, after `set_params` or a new `__init__`) resolves and
    stores it. `model` is an estimator instance, never a class."""
    var owner = records_get(RECORDS_PREPARED, model, _PREP_ROUTE_SLOT)
    if owner is not None:
        var native = owner.downcast_value_ptr[PrepRouteOwner]()
        if native[].resolved and native[].root is context:
            var held = native[].context
            _ = owner  # the owner pins the context through the copy above
            return held
    var fresh = PythonObject(alloc=_prep_route_build(model, context))
    # An estimator that takes no weak reference keeps no owner: it resolves
    # at every operation (records_put answers False and stores nothing).
    _ = records_put(RECORDS_PREPARED, model, _PREP_ROUTE_SLOT, fresh)
    var run = fresh.downcast_value_ptr[PrepRouteOwner]()[].context
    _ = fresh
    return run


def prep_route_vendor_used(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """`vendor_used()`: the vendor word the compiled x_prep image of this
    estimator's route reports about itself (`cpu`, `metal`, `cuda`, `hip`),
    or None for an image built before the read-back. A metadata operation,
    in no loop."""
    var run = prep_route_context(model, context)
    var str_type = context["prep_device_str"]
    var fields = py_fields(model)
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = run["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not str_type:
        raise Error("numeric_mode must be fast, identical or None")
    var word = py_call(py_call(mode.strip).lower)
    var entries = py_get(run["scaler_bindings"], word)
    if entries is None:
        raise Error("numeric_mode must be fast, identical or None")
    if py_type(entries) is str_type:
        raise Error("ImportError: " + String(entries))
    var text = prep_entry_text(entries, "x_prep_vendor")
    if text.byte_length() == 0:
        return PythonObject(None)
    return PythonObject(text)
