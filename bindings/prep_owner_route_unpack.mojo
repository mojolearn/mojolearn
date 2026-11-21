# SPDX-License-Identifier: Apache-2.0
"""The x_prep owner family's resolver on the ONE device route owner (lane
dw-prep-encode, 2026-11-21).

`DeviceRouteOwner` (providers/python/device_route_boundary_door.mojo) holds an
estimator's route; the rule and the words are core/device_route.mojo. This
file is only the resolver of the classes the family door serves through the
encode owner and the label owner (bindings/prep_owner_native_door.mojo:
SimpleImputer, KBinsDiscretizer, OrdinalEncoder, OneHotEncoder,
TargetEncoder, SplineTransformer, SelectKBest, LabelEncoder, LabelBinarizer,
MultiLabelBinarizer). The route's layout is the scalers'
(bindings/scaler_route_unpack.mojo: `context` the door's connected context,
`scope` the context the estimator runs in, `images` / `mode` the held entry
selection, `route.entry` the x_prep transport entry, `route.flags` the
image's capability bits), so the scalers' selection functions
(`scaler_selection_held`, `scaler_selection_keep`, `scaler_route_held`,
`scaler_held_caps`) serve this family unchanged.

What differs from the scalers is where the CPU context comes from. The
family's connected context is completed after its bridge returns (the
selector context, the encode entry, the native estimator entry of the
model-selection family), so the bridge cannot snapshot it. Import connects
only the host image's entry table (`prep_cpu_bindings`); the CPU context is
built HERE, once per process, by the first estimator whose route resolves to
the CPU beside a device set: the connected context copied, with
`vendor="cpu"`, the host entry table as its bindings and the host column of
the label steps (`labels_base_binding` False). It is stored in the connected
context (`prep_cpu_context`) and every later CPU route reads that one object.

The device is read ONCE, by `_prep_route_build`, at the estimator's first
data operation. No public call reads or compares it. A data operation reads
the owner (one record read, one pointer compare) for its context.

Resolution is a metadata operation with no independent numerical work to
split; it runs once per estimator, never per row.
"""
from std.python import PythonObject
from bindings.py_balanced import py_type, py_get, py_fields, py_call, py_setitem
from bindings.scaler_route_unpack import (
    SCALER_ROUTE_CPU_TABLE, scaler_route_held, scaler_held_caps,
)
from bindings.x_prep_native_owner_run import XprepCaps
from bindings.prep_entry_cache_unpack import prep_entry_text
from core.device_route import DEVICE_GPU, DEVICE_CPU, device_installed, device_cpu_absent
from providers.python.device_route_boundary_door import (
    DeviceRouteOwner, device_request, device_route_owner, device_route_keep,
)
from x_prep.transport_abi import xp_vendor_word


def _prep_cpu_scope(root: PythonObject, b: PythonObject) raises -> PythonObject:
    """The CPU context of the family beside a device set: read when it
    exists, built and stored by the first CPU route of the process."""
    var cpu = py_get(root, "prep_cpu_context")
    if cpu is not None:
        return cpu
    var images = py_get(root, "prep_cpu_bindings")
    if images is None:
        raise Error("ImportError: device='cpu' is not connected in this installation")
    if py_type(images) is b.str:
        raise Error("ImportError: " + String(images))
    var built = py_call(b.dict, root)
    py_setitem(built, PythonObject("vendor"), PythonObject("cpu"))
    py_setitem(built, PythonObject("scaler_bindings"), images)
    # The label steps of a CPU route run the host bodies in this process
    # (bindings/classical_labels_door.mojo reads this cell).
    py_setitem(built, PythonObject("labels_base_binding"), PythonObject(False))
    py_setitem(root, PythonObject("prep_cpu_context"), built)
    return built


def _prep_route_build(model: PythonObject, context: PythonObject) raises -> DeviceRouteOwner:
    var b = context["prep_builtins"]
    var owner = DeviceRouteOwner(DEVICE_GPU)
    # The one read of `device`. None, and an instance without the field, run
    # this installation's device; an explicit 'gpu' in a CPU-only process is
    # refused in the shared words (an instance that reached here without its
    # constructor's check).
    var process_vendor = xp_vendor_word(String(context["vendor"]))
    owner.route.device = device_installed(device_request(py_fields(model), b), process_vendor)
    if owner.route.device == DEVICE_CPU and process_vendor != 0:
        owner.route.flags = SCALER_ROUTE_CPU_TABLE
        owner.scope = _prep_cpu_scope(context, b)
    else:
        owner.scope = context
    owner.cls = b.str
    owner.context = context
    owner.resolved = True
    return owner^


def prep_route_context(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """The connected context this estimator's data operation runs in. A hit
    is one record read and one pointer compare; a miss (the estimator's
    first data operation, after `set_params` or a new `__init__`) resolves
    and stores the route. `model` is an estimator instance, never a class."""
    var owner = device_route_owner(model)
    if owner is not None:
        var native = owner.downcast_value_ptr[DeviceRouteOwner]()
        if native[].resolved and native[].context is context:
            var held = native[].scope
            _ = owner  # the owner pins the context through the copy above
            return held
    var fresh = device_route_keep(model, _prep_route_build(model, context))
    var run = fresh.downcast_value_ptr[DeviceRouteOwner]()[].scope
    _ = fresh
    return run


def prep_route_caps(model: PythonObject, entries: PythonObject) raises -> XprepCaps:
    """The x_prep run capabilities the estimator's route holds for `entries`
    (one record read, one pointer compare); `held` False in the result when
    no route holds this selection, and the run then reads them from
    `entries` as before. The caller keeps `entries` named through its run."""
    var held = scaler_route_held(model, entries)
    return scaler_held_caps(held)


def prep_route_refuse_cpu_table(model: PythonObject, shown: String, reason: String) raises:
    """Refuse, in the shared words, a step of this estimator that has no
    route of its own yet when the estimator runs on the CPU beside a device
    set (never run it on the device its caller did not name). One record
    read; nothing is said for an estimator on the installation's device."""
    var owner = device_route_owner(model)
    if owner is None:
        return
    var native = owner.downcast_value_ptr[DeviceRouteOwner]()
    if native[].resolved and (native[].route.flags & SCALER_ROUTE_CPU_TABLE) != 0:
        device_cpu_absent(shown, reason)


def prep_route_vendor_used(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """`vendor_used()`: the vendor word the compiled x_prep image of this
    estimator's route reports about itself (`cpu`, `metal`, `cuda`, `hip`),
    or None for an image built before the read-back. The image is the one
    of the numeric mode the estimator was fitted in, else of the process
    mode. A metadata operation, in no loop."""
    var run = prep_route_context(model, context)
    var str_type = context["prep_builtins"].str
    var mode = py_get(py_fields(model), "numeric_mode_")
    if mode is None:
        mode = run["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not str_type:
        raise Error("numeric_mode must be fast, identical or None")
    var entries = py_get(run["scaler_bindings"], mode)
    if entries is None:
        raise Error("numeric_mode must be fast, identical or None")
    if py_type(entries) is str_type:
        raise Error("ImportError: " + String(entries))
    var text = prep_entry_text(entries, "x_prep_vendor")
    if text.byte_length() == 0:
        return PythonObject(None)
    return PythonObject(text)
