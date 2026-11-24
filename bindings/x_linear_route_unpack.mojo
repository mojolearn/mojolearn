# SPDX-License-Identifier: Apache-2.0
"""The route of a linear expansion estimator (UNPACK): which image its public
operations call, resolved ONCE per estimator and kept on the one route owner
(providers/python/device_route_boundary_door.mojo `DeviceRouteOwner`).

The 21 x_linear estimator classes resolve the x_linear operations entry
(core/linear_entry_table_ext.mojo) of the image their numeric mode and their
`device` select: the connected device table (`xl_bindings[mode]`), or, for
`device="cpu"` on a GPU installation, the CPU identical image connected at
import beside it (`xl_cpu_bindings`, `_mojolearn_x_linear_host`). Lasso and
ElasticNet resolve the solver operations entry (core/solver_entry_table.mojo)
the same way (`cd_solver` / `cd_solver_modes`, or `cd_solver_cpu`,
`_mojolearn_solver_host`).

A public call's hit is the mode read (and the default-mode read when it is
None), one model-record read and three pointer compares. It replaces, per
call: the mode String with strip and lower, `answer_mode_word`, the
`xl_bindings` read by a new str, the connection owner read, and the table slot
read with its schema String (`_xl_entries`); for the pair, the `cd_solver`
context read, two integer conversions and the solver table slot read
(`_xl_cd_entry`).

`scope` is the context whose metrics tables are the route's device: the
door's own context, or the CPU scope connected at import
(python/mojolearn/_native_x_linear_bridge.py `xl_cpu_scope`). The door hands
it to the score programs (accuracy, R^2), so a `device="cpu"` estimator's
score never runs on the process's GPU metrics image.

No public call reads or compares `device`: it is read on a route miss only
(first operation, a changed numeric mode, a constructor or `set_params` that
dropped the route). A refusal of the resolution (a binding that is not
connected, a numeric mode that is not a mode word) is kept as words on the
owner and raised by the operation step that needs the entry, where the
per-call resolution raised it, so the order of an operation's refusals is the
one it had.

Resolution is a metadata operation with no independent numerical work to
split; it runs at an estimator's first operation, never per row."""
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_item
from core.mode_answer import MODE_FAMILY_LINEAR, answer_mode_word
from core.entry_table import EntryRef, entry_ref_word, entry_ref_unresolved
from core.device_route import device_mode_code, device_vendor_code, device_vendor_word, device_route_plan, device_route_admit
from core.linear_entry_table import LINEAR_TABLE_X_LINEAR
from core.linear_entry_table_ext import xl_ops_entry, XL_OPERATION_COUNT
from core.native_descriptor_abi import NativeDescriptorCall
from core.solver_entry_table import SolverEntryKey, solver_ops_entry, SV_OPERATION_COUNT
from providers.python.linear_entry_boundary_door import linear_entry_connection
from providers.python.device_route_boundary_door import (
    DeviceRouteOwner, device_route_held, device_route_keep, device_request, device_admit, device_route_clear,
    device_cpu_connect_late,
)
from bindings.x_linear_keys_unpack import x_linear_keys
from bindings.native_loaded_module_door import NativeModules
from bindings.native_class_admission_door import class_record, class_admit


struct XLinearOps(Movable):
    """One public operation's route: the operations entry of the estimator's
    image (x_linear operations, or the solver operations for Lasso and
    ElasticNet), the owner whose pins keep that image loaded until the
    operation returns, and the context of the route's device. `ready` False:
    the entry is not held (the owner says why, `xl_ops_require`), or the
    class is an explicit host class that resolves none (`host_only`)."""
    var entry: NativeDescriptorCall
    var ready: Bool
    var host_only: Bool
    var owner: PythonObject
    var scope: PythonObject

    def __init__(out self, entry: NativeDescriptorCall, ready: Bool, host_only: Bool, var owner: PythonObject,
                 var scope: PythonObject):
        self.entry = entry
        self.ready = ready
        self.host_only = host_only
        self.owner = owner^
        self.scope = scope^


def _xl_mode_word(selected: PythonObject, b: PythonObject) raises -> String:
    """`_backend.binding`'s mode rule for a mode object that is not None:
    an exact str naming `fast` or `identical`, answered for the family."""
    if py_type(selected) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    # ASCII strip/lower natively; any other spelling takes Python's
    # Unicode-aware strip/lower (same accepted words, same refusal).
    var clean = String(String(String(py=selected).strip()).lower())
    if clean != "identical" and clean != "fast":
        clean = String(py=py_call(py_call(selected.strip).lower))
    if clean != "identical" and clean != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    answer_mode_word(clean, MODE_FAMILY_LINEAR)
    return clean


def _xl_process_vendor(context: PythonObject) raises -> Int:
    """The selected vendor word of this process (0: CPU only), from the
    vendor name the bridge recorded at import."""
    var code = device_vendor_code(String(py=py_item(context, PythonObject("vendor"))))
    if code < 0:
        raise Error("RuntimeError: mojolearn: the linear estimators' context names no known vendor")
    return code


def _xl_cpu_table(context: PythonObject, key: String) raises -> PythonObject:
    """One CPU key of a GPU installation (`xl_cpu_bindings`, `cd_solver_cpu`,
    `xl_cpu_scope`), or None when it is not connected. The CPU images load
    at the first device="cpu" admission (fx-host-import:
    python/mojolearn/_native_x_linear_bridge.py, no longer at import); an
    estimator restored without construction connects them here, once (the
    three keys are written by one load). Route resolution only."""
    var table = py_get(context, PythonObject(key))
    if table is None and device_cpu_connect_late(context):
        table = py_get(context, PythonObject(key))
    return table


def _xl_cpu_scope(context: PythonObject, b: PythonObject) raises -> PythonObject:
    """The CPU scope of a GPU installation, or the refusal recorded at
    its load."""
    var scope = _xl_cpu_table(context, String("xl_cpu_scope"))
    if scope is None:
        raise Error("ImportError: device='cpu' is not connected in this installation")
    if py_type(scope) is b.str:
        raise Error(String(py=scope))
    return scope


def _xl_resolve_ops(mut owner: DeviceRouteOwner, requested: Int, mode: PythonObject, default_mode: PythonObject,
                    context: PythonObject, b: PythonObject) raises:
    """The x_linear operations entry of the image the estimator's numeric
    mode and device select."""
    var nk = x_linear_keys()
    var selected = mode
    if mode is None:
        selected = default_mode
    var mode_name = _xl_mode_word(selected, b)
    var mode_code = device_mode_code(mode_name)
    var plan = device_route_plan(requested, _xl_process_vendor(context), mode_code)
    var entries = PythonObject(None)
    if plan.cpu_table:
        var cpu = _xl_cpu_table(context, String("xl_cpu_bindings"))
        if cpu is None:
            raise Error("ImportError: device='cpu' is not connected in this installation")
        entries = py_item(cpu, PythonObject(mode_name))
        owner.scope = _xl_cpu_scope(context, b)
    else:
        entries = py_item(py_item(context, nk[].k_xl_bindings), PythonObject(mode_name))
    if py_type(entries) is b.str:
        raise Error(String(py=entries))
    # The connection owner was admitted at import against the binding's
    # published table (python/mojolearn/_native_x_linear_bridge.py), so its
    # native key is the binding's compiled mode and vendor.
    var connection = linear_entry_connection(entries, LINEAR_TABLE_X_LINEAR)
    if connection.key.mode != mode_code:
        raise Error("RuntimeError: mojolearn: _mojolearn_x_linear was compiled for "
                    + ("identical" if connection.key.mode == 1 else "fast") + ", but this call requested "
                    + mode_name + "; rebuild the " + mode_name + " binding")
    if plan.cpu_table:
        # The CPU image beside a device set must be the CPU identical one.
        device_route_admit("x_linear", plan, connection.key.mode, connection.key.vendor, mode_code)
    var entry = xl_ops_entry(connection.key)
    owner.route.device = plan.device
    owner.route.mode = connection.key.mode
    owner.route.vendor = connection.key.vendor
    owner.route.entry = EntryRef(entry, entry_ref_word(connection.key.mode, connection.key.vendor, XL_OPERATION_COUNT))
    owner.images = Python.tuple(entries, connection.images)


def _xl_resolve_cd(mut owner: DeviceRouteOwner, requested: Int, mode: PythonObject, context: PythonObject,
                   b: PythonObject) raises:
    """The solver operations entry of Lasso / ElasticNet. `numeric_mode`
    None (or absent: a model saved before the parameter existed) is the
    image connected at import, as before the parameter; an explicit word is
    validated and answered as every x_linear estimator's is and selects that
    tier's image."""
    var nk = x_linear_keys()
    var process_vendor = _xl_process_vendor(context)
    var key = py_item(context, nk[].k_cd_solver)
    var mode_code = -1
    if mode is not None:
        var mode_name = _xl_mode_word(mode, b)
        mode_code = device_mode_code(mode_name)
        key = py_item(py_item(context, nk[].k_cd_solver_modes), PythonObject(mode_name))
    elif py_type(key) is not b.str:
        mode_code = Int(py=key[0])
    if mode_code < 0:
        # The image connected at import is a refusal: an explicit device is
        # still planned (identical), so its own refusal comes first.
        mode_code = 1
    var plan = device_route_plan(requested, process_vendor, mode_code)
    if plan.cpu_table:
        var cpu = _xl_cpu_table(context, String("cd_solver_cpu"))
        if cpu is None:
            raise Error("ImportError: device='cpu' is not connected in this installation")
        key = cpu
        owner.scope = _xl_cpu_scope(context, b)
    if py_type(key) is b.str:
        raise Error(String(py=key))
    var native = SolverEntryKey(Int(py=key[0]), Int(py=key[1]))
    if plan.cpu_table:
        device_route_admit("solver", plan, native.mode, native.vendor, mode_code)
    var entry = solver_ops_entry(native)
    owner.route.device = plan.device
    owner.route.mode = native.mode
    owner.route.vendor = native.vendor
    owner.route.entry = EntryRef(entry, entry_ref_word(native.mode, native.vendor, SV_OPERATION_COUNT))
    # The solver image is retained by the module that connected it
    # (python/mojolearn/_solver_impl.py; the CPU one by the bridge's context).
    owner.images = key


def _xl_route_build(requested: Int, mode: PythonObject, default_mode: PythonObject, context: PythonObject,
                    cd: Bool, b: PythonObject) raises -> DeviceRouteOwner:
    """Resolve an estimator's entry once. Runs at an estimator's first
    operation, after a changed numeric mode and after a constructor or
    `set_params` dropped the route; never per call. A refusal is kept as its
    words (the entry stays the refusing one, word 0)."""
    var owner = DeviceRouteOwner(requested)
    owner.scope = context
    try:
        if cd:
            _xl_resolve_cd(owner, requested, mode, context, b)
        else:
            _xl_resolve_ops(owner, requested, mode, default_mode, context, b)
    except error:
        owner.refusal = String(error)
        owner.scope = context
    owner.mode = mode
    owner.default_mode = default_mode
    owner.context = context
    owner.resolved = True
    return owner^


def xl_route(model: PythonObject, fields: PythonObject, context: PythonObject, cd: Bool) raises -> XLinearOps:
    """The estimator's route for this operation: a hit, or the one
    resolution. `cd`: the coordinate-descent pair (the solver entry)."""
    var nk = x_linear_keys()
    var mode = py_get(fields, nk[].k_numeric_mode)
    var default_mode = PythonObject(None)
    if mode is None:
        default_mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    var owner = device_route_held(model, mode, default_mode, context)
    if owner is None:
        var b = native_loaded_module("builtins")
        var device = device_request(fields, b)
        owner = device_route_keep(model, _xl_route_build(device, mode, default_mode, context, cd, b))
    var native = owner.downcast_value_ptr[DeviceRouteOwner]()
    var entry = native[].route.entry
    var scope = native[].scope
    return XLinearOps(entry.call, entry.word != 0, False, owner, scope)


def xl_route_vendor_used(model: PythonObject, fields: PythonObject, context: PythonObject, cd: Bool) raises -> PythonObject:
    """`vendor_used()`: the vendor word of the image the estimator's route
    holds (`cpu` for a `device="cpu"` estimator beside a GPU set), resolved
    as its next data operation would resolve it; a route that could not be
    resolved raises its own words. Metadata, in no loop."""
    var ops = xl_route(model, fields, context, cd)
    xl_ops_require(ops)
    var vendor = ops.owner.downcast_value_ptr[DeviceRouteOwner]()[].route.vendor
    _ = ops^  # the owner pins the route record read above
    return PythonObject(device_vendor_word(vendor))


def xl_route_host(context: PythonObject) raises -> XLinearOps:
    """The route of an explicit host inference class (it predicts a saved
    model on the CPU and resolves no entry): only its scope, which is the CPU
    scope on a GPU installation (connected here once when the first CPU
    admission has not run yet: the key is missing only then; a CPU-only
    process holds a text). A CPU scope that is not connected leaves the
    door's own context, as before the scope existed."""
    var scope = _xl_cpu_table(context, String("xl_cpu_scope"))
    if scope is None or py_type(scope) is py_type(PythonObject("")):
        scope = context
    return XLinearOps(entry_ref_unresolved().call, False, True, PythonObject(None), scope)


comptime _XL_HOST_ONLY_SLOT = "x_linear_host_inference_only_v1"


def xl_host_only(model: PythonObject) raises -> Bool:
    """An explicit host inference class (`_HOST_INFERENCE_ONLY` in its class
    dictionaries): it predicts a saved model on the CPU and never fits.
    The fit door (bindings/x_linear_native_door.mojo) and the Cd member
    admission (providers/python/member_cd_boundary_door.mojo) both read it.
    Depends only on the class rows: resolved once per class state
    (native_class_admission re-checks after any class modification)."""
    var nk = x_linear_keys()
    var mods = NativeModules()
    var cls = py_type(model)
    var record = class_record(mods, _XL_HOST_ONLY_SLOT, cls)
    if record is not None:
        return Bool(py=record)
    var b = native_loaded_module("builtins")
    var mro = py_call(py_item(b.type.__dict__, nk[].k___mro__).__get__, cls)
    var found = False
    for k in range(len(mro)):  # driver-loop: over the model class's MRO
        var class_fields = py_call(py_item(b.type.__dict__, nk[].k___dict__).__get__, mro[k])
        if py_get(class_fields, nk[].k__HOST_INFERENCE_ONLY) is not None:
            found = True
            break
    class_admit(mods, _XL_HOST_ONLY_SLOT, cls, PythonObject(found))
    return found


def xl_ops_require(ops: XLinearOps) raises:
    """Raise why the estimator's entry is not held, in the words its
    resolution raised (the step of an operation that calls the entry)."""
    if ops.ready:
        return
    if ops.owner is None:
        raise Error("ImportError: mojolearn: no entry is resolved for an explicit host class")
    var text = ops.owner.downcast_value_ptr[DeviceRouteOwner]()[].refusal.copy()
    if text.byte_length() == 0:
        text = String("ImportError: mojolearn: the linear estimators' binding is not connected")
    raise Error(text)


def xl_device_admit(model: PythonObject, value: PythonObject, context: PythonObject) raises:
    """Check a `device` where it is given (a constructor, `set_params`),
    before the estimator changes: an unknown value, and an explicit `'gpu'`
    in a CPU-only process, are refused in the shared words. Every class of
    this door has a CPU implementation, so `'cpu'` is always accepted."""
    device_admit(value, _xl_process_vendor(context), native_loaded_module("builtins"))


def xl_route_drop(model: PythonObject) raises:
    """Forget the estimator's route (a constructor run again, `set_params`
    with a device): its next operation reads the device and the mode again."""
    device_route_clear(model)
