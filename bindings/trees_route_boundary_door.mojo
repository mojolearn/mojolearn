# SPDX-License-Identifier: Apache-2.0
"""The table set a tree estimator's operations run on, held on its device route.

`device=None` (default: this installation's device), `device="gpu"` or
`device="cpu"` is a constructor parameter. The rule, the words and the owner
are the shared ones (core/device_route.mojo,
providers/python/device_route_boundary_door.mojo `DeviceRouteOwner`): the
device is checked where it is given (constructor, `set_params`) and read
ONCE, at the estimator's first operation, into the estimator's binding
route. This file adds no owner type and no rule.

The tree doors reach every compiled entry through ONE table of the
connection context (`context["trees_bindings"][mode][binding]`,
bindings/trees_public_inputs_door.mojo `trees_table`). So what a tree
estimator's route holds is the context its operations run in: the connection
context (this installation's table set), or, for `device="cpu"` on a GPU
installation, the CPU run context, whose `trees_bindings` is the CPU table
set (`context["trees_cpu_bindings"]`, the host images the bridge loads
beside the device images at the first CPU admission: exactly what a CPU-only installation
connects). The run context is kept under the binding name
`trees_run_context` of the estimator's binding route, so every shared drop
of that route (`device_route_clear`, the shared constructor's
`device_admit`) forgets it.

Per public operation (`trees_route_context`, never per row): one record
read, one pointer compare of the context and the held binding's compares.
No public operation reads or compares the device. The 39 `trees_table`
sites of the doors are unchanged and none of them tests the device.

Resolution is a metadata operation with no independent numerical work to
split; it runs at an estimator's first operation, never per row.
"""
from std.python import PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_item, py_setitem, py_fields
from bindings.trees_keys_unpack import trees_keys
from bindings.trees_public_inputs_door import trees_builtins
from core.device_route import DEVICE_CPU, device_vendor_code
from providers.python.device_route_boundary_door import (
    device_admit, device_binding_held, device_binding_device, device_binding_keep, device_cpu_connect_late,
)

comptime _TREES_RUN = "trees_run_context"


def trees_device_admit(value: PythonObject, context: PythonObject) raises:
    """Check a device where it is given (a tree constructor, `set_params`),
    before any field changes: the shared check and words. The default None
    is one identity compare and reads nothing."""
    if value is None:
        return
    var nk = trees_keys()
    device_admit(value, device_vendor_code(String(py_item(context, nk[].k_device_vendor))), trees_builtins())


def _process_cpu_only(context: PythonObject) raises -> Bool:
    var nk = trees_keys()
    var vendor = py_get(context, nk[].k_device_vendor)
    return vendor is not None and String(vendor) == "cpu"


def trees_cpu_context(context: PythonObject) raises -> PythonObject:
    """The CPU run context of a GPU installation: the connection context with
    `trees_bindings` naming the CPU table set. Built once per process; built
    again only when the connection context gained a key since (a tree class
    registered later adds its fixed tables). Two held-key reads and one size
    compare on a hit."""
    var nk = trees_keys()
    var run = py_get(context, nk[].k_device_cpu_context)
    if run is not None:
        var size = py_get(context, nk[].k_device_cpu_size)
        if size is not None and Int(py=size) == len(context):
            return run
    var cpu = py_get(context, nk[].k_device_cpu_bindings)
    # Loaded at the first device="cpu" admission (fx-host-import:
    # python/mojolearn/_native_trees_bridge.py, no longer at import); an
    # estimator restored without construction connects it here, once. Before
    # the size below is counted: the load adds its keys first.
    if cpu is None and device_cpu_connect_late(context):
        cpu = py_get(context, nk[].k_device_cpu_bindings)
    if cpu is None:
        raise Error("ImportError: device='cpu' is not connected in this installation")
    var b = trees_builtins()
    if py_type(cpu) is b.str:
        raise Error(String(cpu))
    var fresh = py_call(b.dict, context)
    py_setitem(fresh, nk[].k_trees_bindings, cpu)
    # A door that asks the run context for the CPU context again gets itself.
    py_setitem(fresh, nk[].k_device_cpu_context, fresh)
    py_setitem(context, nk[].k_device_cpu_context, fresh)
    # Both keys exist in the connection context before its size is recorded.
    py_setitem(context, nk[].k_device_cpu_size, PythonObject(0))
    var counted = len(context)
    py_setitem(context, nk[].k_device_cpu_size, PythonObject(counted))
    py_setitem(fresh, nk[].k_device_cpu_size, PythonObject(counted))
    return fresh


def trees_route_context(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """The context `model`'s operation runs in. A hit is the binding route's
    read; a miss (the estimator's first operation, or its first after a
    constructor or `set_params`) reads the device once and keeps the
    answer."""
    if model is None:
        # An operation of no estimator (a parameter table question).
        return context
    var held = device_binding_held(model, String(_TREES_RUN), False, PythonObject(None), PythonObject(None), context)
    if held is not None:
        return held
    var fields = PythonObject(None)
    try:
        fields = py_fields(model)
    except:
        # An object without an instance dictionary carries no device.
        return context
    var cpu_only = _process_cpu_only(context)
    var device = device_binding_device(model, fields, context, cpu_only, trees_builtins())
    var run = context
    if device == DEVICE_CPU and not cpu_only:
        run = trees_cpu_context(context)
    device_binding_keep(model, device, String(_TREES_RUN), False, PythonObject(None), PythonObject(None), context, run)
    return run


def trees_route_vendor(context: PythonObject) raises -> PythonObject:
    """`vendor_used()`: the vendor the compiled x_trees image of the
    operation's table set reports about itself (`'cpu'` for the CPU table
    set). Without that image's probe: `'cpu'` in the CPU run context, else
    the process's selected vendor."""
    var nk = trees_keys()
    try:
        var tables = py_get(py_item(context, nk[].k_trees_bindings), PythonObject("identical"))
        if tables is not None:
            var table = py_get(tables, PythonObject("_mojolearn_x_trees"))
            if table is not None and py_type(table) is trees_builtins().dict:
                var probe = py_get(table, PythonObject("x_trees_vendor"))
                if probe is not None:
                    return py_call(probe)
    except:
        pass
    if py_get(context, nk[].k_device_cpu_context) is context:
        return PythonObject("cpu")
    return py_item(context, nk[].k_device_vendor)
