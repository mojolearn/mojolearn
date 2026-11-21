# SPDX-License-Identifier: Apache-2.0
"""The stateless preprocessing owner's resolver on the ONE device route owner
(lane dw-prep, round 3).

`DeviceRouteOwner` (providers/python/device_route_boundary_door.mojo) holds an
estimator's route; the rule and the words are core/device_route.mojo. This
file is only what the route of a Binarizer, PolynomialFeatures,
VarianceThreshold, QuantileTransformer, PowerTransformer, Normalizer or a
selector's transform holds, and how the stateless door reads it.

The stateless door builds a fresh operation context per call from a frozen
template (bindings/prep_context_native_door.mojo), so a route cannot pin the
operation context. It pins what is stable across calls:
  * `context`: the connected root the public method hands in (identity);
  * `scope`: the `scaler_bindings` table of the template the estimator runs
    on (the installation's, or the CPU template's on a GPU installation):
    every operation copy of that template holds this same table object;
  * `images`, `mode`, `default_mode`, `route.entry`, `route.flags`: the
    selection of the scaler resolver (bindings/scaler_route_unpack.mojo),
    made by the door's own `_entries` checks on a miss and held after.

The device is read ONCE, by `_stateless_route_build`, at the estimator's
first data operation. No public call reads or compares it: a data operation
reads the owner (one record read, one pointer compare) for its template, and
each entry selection reads it again in place of the type test, the word
compares, the dictionary reads and the two mode probes `_entries` makes.

Resolution is a metadata operation with no independent numerical work to
split; it runs once per estimator, never per row.
"""
from std.python import PythonObject
from bindings.py_balanced import py_fields, py_item
from bindings.prep_context_native_door import prep_operation_context, prep_cpu_connected
from bindings.scaler_keys_unpack import scaler_keys
from bindings.scaler_lifecycle_native_door import _entries
from bindings.scaler_route_unpack import (
    scaler_selection_held, scaler_selection_keep, scaler_route_refuse_fast, scaler_route_held, scaler_held_caps,
    SCALER_ROUTE_CPU_TABLE,
)
from bindings.x_prep_native_owner_run import XprepCaps
from core.device_route import DEVICE_GPU, DEVICE_CPU, device_installed
from providers.python.device_route_boundary_door import (
    DeviceRouteOwner, device_request, device_route_owner, device_route_keep,
)
from x_prep.transport_abi import xp_vendor_word


def _stateless_route_build(model: PythonObject, root: PythonObject, b: PythonObject) raises -> DeviceRouteOwner:
    var owner = DeviceRouteOwner(DEVICE_GPU)
    # The one read of `device`. None, and an instance without the field, run
    # this installation's device; an explicit 'gpu' in a CPU-only process is
    # refused in the shared words (an instance that reached here without its
    # constructor's check).
    var process_vendor = xp_vendor_word(String(root["vendor"]))
    owner.route.device = device_installed(device_request(py_fields(model), b), process_vendor)
    if owner.route.device == DEVICE_CPU and process_vendor != 0 and prep_cpu_connected(root, b):
        owner.route.flags = SCALER_ROUTE_CPU_TABLE
    owner.cls = b.str
    owner.context = root
    owner.resolved = True
    return owner^


def stateless_route_context(model: PythonObject, root: PythonObject, b: PythonObject) raises -> PythonObject:
    """The operation context of this estimator's data operation: the copy of
    the template its route names. A hit is one record read and one pointer
    compare before the copy; a miss (the estimator's first data operation,
    after `set_params` or a new `__init__`) resolves and stores the route.
    `model` is an estimator instance, never a class."""
    var owner = device_route_owner(model)
    if owner is not None:
        var native = owner.downcast_value_ptr[DeviceRouteOwner]()
        if native[].resolved and native[].context is root:
            var cpu = (native[].route.flags & SCALER_ROUTE_CPU_TABLE) != 0
            _ = owner
            return prep_operation_context(root, cpu)
    var built = _stateless_route_build(model, root, b)
    var context = prep_operation_context(root, (built.route.flags & SCALER_ROUTE_CPU_TABLE) != 0)
    var nk = scaler_keys()
    built.scope = py_item(context, nk[].k_scaler_bindings)
    var fresh = device_route_keep(model, built^)
    _ = fresh
    return context


def stateless_entries(model: PythonObject, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_entries(mode, context)` of an estimator's operation: the selection
    its device route holds for `mode` on this template (one dictionary read,
    one record read and pointer compares; no type test, word compare, mode
    probe or String build), or the selection made here by `_entries`' own
    checks, in its order and words, and kept in the route. `device="cpu"`
    with the fast mode is refused in the shared words first. An estimator
    without a route for this template selects exactly as before."""
    var nk = scaler_keys()
    var tables = py_item(context, nk[].k_scaler_bindings)
    var held = scaler_selection_held(model, mode, tables)
    if held is not None:
        return held
    scaler_route_refuse_fast(model, mode, tables)
    var entries = _entries(mode, context)
    scaler_selection_keep(model, mode, entries, tables)
    return entries


def stateless_caps(model: PythonObject, entries: PythonObject) raises -> XprepCaps:
    """The x_prep run capabilities the estimator's route holds for `entries`
    (one record read, one pointer compare), in place of the five by-name
    reads and the table slot read of a run without a route. Not held: the
    run reads them from `entries` as before. The caller keeps `entries`
    named until its operation returns."""
    return scaler_held_caps(scaler_route_held(model, entries))
