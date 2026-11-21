# SPDX-License-Identifier: Apache-2.0
"""UNPACK of a tree estimator's `device` into the entry tables its operation runs on.

`device=None` (default: this installation's device), `device="gpu"` or
`device="cpu"` is a constructor parameter of every tree estimator (forests,
decision trees, boosting, bagging, isolation forest, the meta-estimators).
The rule and the words are the pilot's (core/linear_route.mojo): None is this
installation's device, an explicit `'gpu'` without a GPU set is refused in
words, `'cpu'` always runs, anything else is refused in words.

The tree doors reach every compiled entry through ONE table of the
connection context (`context["trees_bindings"][mode][binding]`,
bindings/trees_public_inputs_door.mojo `trees_table`). So a tree estimator's
route is a table set, not an entry: on a GPU installation the CPU table set
(`context["trees_cpu_bindings"]`, the host images the bridge loads beside the
device images at import, python/mojolearn/_native_trees_bridge.py
`connect_cpu_device`) is exactly what a CPU-only installation connects under
`trees_bindings`. A `device="cpu"` operation runs in the CPU run context: the
connection context with `trees_bindings` naming the CPU table set. That
context is built ONCE PER PROCESS (not per estimator, not per call) and kept
in the connection context; the 39 `trees_table` sites of the doors are
unchanged and none of them tests the device.

What one public operation reads here (`trees_route_enter`, once per call,
never per row): the estimator's `device` field and its fitted-device marker,
two held-key reads of the instance dictionary and three pointer compares. The
default (`device=None`, fitted on this installation's table set) returns the
connection context as it is: no owner, no record, no allocation, no String. Because the field is read where the
operation starts, `set_params`, a rebuilt constructor and a directly assigned
`model.device` are all seen by the next operation with no invalidation hook
in any of the family's constructors.

The tree families check their parameters at fit (scikit-learn's convention),
and `device` follows it: it is stored as given, returned as given by
`get_params`, and refused at fit (and `_validate_params`).

A fitted tree model holds handles of the image set that fitted it. Every fit
records which table set fitted (`_mojolearn_fit_cpu` in the instance
dictionary: False this installation's set, True the CPU set of a GPU
installation; written only when it changes, so once per estimator in the
usual case). Using fitted state through the other table set is refused in
words instead of handing one image's handle to another image.

Resolution is a metadata operation with no independent numerical work to
split; it runs once per public call, never per row.
"""
from std.python import PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_item, py_setitem, py_fields
from bindings.trees_keys_unpack import trees_keys
from bindings.trees_public_inputs_door import trees_builtins
from core.linear_route import (
    LINEAR_DEVICE_GPU, LINEAR_DEVICE_CPU, LINEAR_DEVICE_INSTALL, linear_device_code, linear_device_refusal,
    linear_device_gpu_absent,
)


struct TreesRouteCall(Movable):
    """One operation's route: the context it runs in, and what the operation's
    end records on the estimator (the table set of a fit)."""
    var context: PythonObject
    var fields: PythonObject
    # The operation builds fitted state (its end records the table set).
    var fit: Bool
    # The operation runs on the CPU table set of a GPU installation.
    var cpu: Bool
    # The marker is already what this fit would record.
    var recorded: Bool

    def __init__(out self, context: PythonObject):
        self.context = context
        self.fields = PythonObject(None)
        self.fit = False
        self.cpu = False
        self.recorded = True


def _route_fits(name: String) -> Bool:
    """Whether operation `name` builds fitted state in the table set it runs
    on (`*_fit*`, an archive `load`); `is_fitted` only asks."""
    if name.find("is_fitted") >= 0:
        return False
    return name.find("fit") >= 0 or name.endswith("_load")


def _route_lifecycle(name: String) -> Bool:
    """Operations that read or replace parameters and never use fitted
    state: they run whatever the stored device and marker say."""
    return (name.find("params") >= 0 or name.find("construct") >= 0 or name.find("_init") >= 0
            or name.find("refresh") >= 0 or name.find("is_fitted") >= 0 or name.find("setstate") >= 0
            or name.find("vendor_used") >= 0)


def trees_device_unpack(value: PythonObject, strict: Bool) raises -> Int:
    """The device code of a stored constructor value: None (this
    installation's device), or exact `'gpu'` or `'cpu'`. Anything else is
    refused in the pilot's words when `strict` (a fit, `_validate_params`),
    and is this installation's device otherwise (parameter operations run on
    any stored value, as `get_params` must)."""
    if value is None:
        return LINEAR_DEVICE_INSTALL
    var b = trees_builtins()
    if py_type(value) is b.str:
        var code = linear_device_code(String(value))
        if code >= 0:
            return code
    if strict:
        linear_device_refusal(String(py_call(b.repr, value)))
    return LINEAR_DEVICE_INSTALL


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


def _route_refit(model: PythonObject, fitted_cpu: Bool) raises:
    """The one refusal of fitted state reached through the other table set."""
    var name = String(py_type(model).__name__)
    var was = String("'cpu'") if fitted_cpu else String("the GPU")
    var now = String("the GPU") if fitted_cpu else String("'cpu'")
    raise Error("ValueError: this " + name + " was fitted with device=" + was + " and its device is now " + now
        + ". Fit it again, or set device back, before using the fitted model")


def trees_route_enter(model: PythonObject, name: String, context: PythonObject) raises -> TreesRouteCall:
    """The context operation `name` of `model` runs in. Default path
    (`device=None`, fitted on this installation's table set): two held-key
    reads and three pointer compares, the connection context returned as it
    is. An estimator no fit has recorded yet also has its operation name
    looked at (is this a fit), until its first fit."""
    var route = TreesRouteCall(context)
    if model is None:
        # An operation of no estimator (a parameter table question).
        return route^
    var nk = trees_keys()
    var fields = PythonObject(None)
    try:
        fields = py_fields(model)
    except:
        # An object without an instance dictionary carries no device.
        return route^
    var stored = py_get(fields, nk[].k_device)
    var marker = py_get(fields, nk[].k_device_fit_cpu)
    var install_set = PythonObject(False)
    if stored is None:
        if marker is install_set:
            return route^
        if marker is None:
            # Never recorded: a fit on this installation's set records it.
            if _route_fits(name):
                route.fields = fields
                route.fit = True
                route.recorded = False
            return route^
    # From here on: an explicit device, or a model fitted on the CPU table
    # set of a GPU installation. Never the default path.
    var fits = _route_fits(name)
    var device = trees_device_unpack(stored, fits or name.find("validate_params") >= 0)
    var fitted_cpu = marker is not None and marker is not install_set and Bool(py=marker)
    var fitted_install = marker is install_set
    if _process_cpu_only(context):
        # A CPU-only installation: its one table set is the CPU's. An
        # explicit 'gpu' is refused in words at fit; the marker of a model
        # unpickled here names a table set this process does not have.
        if device == LINEAR_DEVICE_GPU and fits:
            linear_device_gpu_absent()
        if fits and not fitted_install:
            route.fields = fields
            route.fit = True
            route.recorded = False
        return route^
    route.cpu = device == LINEAR_DEVICE_CPU
    if not fits and not _route_lifecycle(name):
        if (route.cpu and fitted_install) or (not route.cpu and fitted_cpu):
            _route_refit(model, fitted_cpu)
    if fits:
        route.fields = fields
        route.fit = True
        route.recorded = fitted_cpu if route.cpu else fitted_install
    if route.cpu:
        route.context = trees_cpu_context(context)
    return route^


def trees_route_leave(route: TreesRouteCall) raises:
    """After a fit that returned: record the table set that fitted. Nothing
    is written when the marker already says so (every fit after an
    estimator's first, on the default path)."""
    if not route.fit or route.recorded:
        return
    var nk = trees_keys()
    py_setitem(route.fields, nk[].k_device_fit_cpu, PythonObject(route.cpu))


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
