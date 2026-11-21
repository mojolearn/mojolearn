# SPDX-License-Identifier: Apache-2.0
"""The Ridge and LinearRegression resolver of the shared device route owner.

The owner, its slot, its hit test, the device reader and every device word
are the shared ones (providers/python/device_route_boundary_door.mojo,
core/device_route.mojo). This file is only what the family resolves on a
miss: the device table (`context["classical_bindings"][mode]`) or the CPU
table (`context["classical_cpu_bindings"]`), the fit and predict entries,
Ridge's weighted solve and LinearRegression's TSQR table
(core/linear_route.mojo names the slots).

`device` is checked when it is given (constructor, `set_params`) and read
ONCE, at the estimator's first operation. Every later fit, predict and score
reads the owner (one record read and three pointer compares: the
`numeric_mode` object, the process default mode object when that is None,
the connection context) and calls through it.

# TOMBSTONE: `LinearRouteOwner` (its pins, registration, slot `linear_route_v1`, `linear_device_unpack`,
# `linear_route_clear`) deleted 2026-11-20 by lane dw-core: one owner for every family
# (`DeviceRouteOwner`). Code recoverable at 3a99eaa0dc.
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_type, py_get, py_item, py_contains
from bindings.classical_keys_unpack import classical_keys
from core.entry_table import EntryRef, entry_ref_word
from core.linear_entry_table import LINEAR_TABLE_X_LINEAR, LINEAR_TABLE_DECOMP, linear_table_entry
from core.linear_metadata import linear_mode_name, linear_vendor_code
from core.device_route import (
    DeviceRoute, DEVICE_CPU, DEVICE_INSTALL, device_installed, device_cpu_fast,
)
from core.linear_route import linear_route_resolve, linear_route_weighted
from providers.python.device_route_boundary_door import (
    DeviceRouteOwner, device_unpack, device_request, device_route_held, device_route_keep, device_route_value,
    device_route_refuse, device_cpu_connect_late,
)
from providers.python.linear_entry_boundary_door import linear_entry_connection
from providers.python.metrics_route_boundary_door import linear_route_metrics_cpu
from x_decomp.descriptor_schema import XD_ABI_NAME, XD_OPERATION_COUNT


def linear_device_admit(value: PythonObject, context: PythonObject, b: PythonObject) raises:
    """Check a device where it is given (constructor, `set_params`): an
    unknown value, and an explicit `'gpu'` in a CPU-only process, are refused
    in words before the estimator changes. The default None reads nothing
    here. The explicit Host* classes run on the CPU by class and take any
    admitted value."""
    var device = device_unpack(value, b)
    if device == DEVICE_INSTALL or device == DEVICE_CPU:
        return
    var nk = classical_keys()
    if Bool(py=py_get(context, nk[].k_classical_explicit_cpu, False)):
        return
    _ = device_installed(device, linear_vendor_code(String(py_item(context, nk[].k_vendor))))


def _linear_route_build(device: Int, ridge: Bool, mode: PythonObject,
    default_mode: PythonObject, context: PythonObject, b: PythonObject) raises -> DeviceRouteOwner:
    var nk = classical_keys()
    var owner = DeviceRouteOwner(device)
    var selected = mode
    if mode is None:
        selected = default_mode
    var explicit = Bool(py=py_get(context, nk[].k_classical_explicit_cpu, False))
    var entries = PythonObject(None)
    var expected_vendor = 0
    var expected_mode = 1
    var cpu_beside_gpu = False
    if explicit:
        # The explicit Host* classes (mojolearn.host_model): CPU identical by
        # class, whatever `device` their inherited constructor stored. Their
        # helper family is selected by the raw mode object, as before.
        if mode is not None and (py_type(mode) is not b.str or String(mode) != "identical"):
            raise Error("the host model runs identical only; this model was saved with a different numeric_mode")
        owner.route.device = DEVICE_CPU
        entries = py_item(py_item(context, nk[].k_classical_bindings), selected)
    else:
        if py_type(selected) is not b.str:
            raise Error("numeric_mode must be fast, identical or None")
        var mode_name = linear_mode_name(String(selected))
        expected_mode = 1 if mode_name == "identical" else 0
        var process_vendor = linear_vendor_code(String(py_item(context, nk[].k_vendor)))
        # None is this installation's device; an explicit 'gpu' in a CPU-only
        # process is refused in words (an instance that reached here without
        # its constructor's check: unpickled, or its field assigned).
        owner.route.device = device_installed(device, process_vendor)
        if owner.route.device == DEVICE_CPU and process_vendor != 0:
            # A GPU installation: the CPU table of the host binaries, loaded
            # at the first device="cpu" admission (fx-audit 9: no longer at
            # import; python/mojolearn/_native_classical_bridge.py
            # connect_cpu_device). An estimator restored without
            # construction connects it here, once, at its first route.
            if expected_mode != 1:
                device_cpu_fast()
            # The table is filled in place (one dictionary shared by the
            # contexts copied from this one), so a miss is "not loaded yet".
            var cpu = py_get(context, "classical_cpu_bindings")
            if (cpu is None or not py_contains(cpu, PythonObject(mode_name))) and device_cpu_connect_late(context):
                cpu = py_get(context, "classical_cpu_bindings")
            if cpu is None or not py_contains(cpu, PythonObject(mode_name)):
                raise Error("ImportError: device='cpu' is not connected in this installation")
            entries = py_item(cpu, PythonObject(mode_name))
            cpu_beside_gpu = True
        else:
            # The process's own table: the GPU table for a GPU request, the
            # CPU table in a CPU-only process (where every request that got
            # here resolved to the CPU).
            entries = py_item(py_item(context, nk[].k_classical_bindings), PythonObject(mode_name))
            expected_vendor = process_vendor
    if py_type(entries) is b.str:
        raise Error("ImportError: " + String(entries))
    var connection = linear_entry_connection(entries)
    owner.route = linear_route_resolve(owner.route.device, connection.key, expected_vendor, expected_mode, explicit)
    if cpu_beside_gpu:
        # `score` of this estimator runs its R^2 program on the CPU metrics
        # table too (it ran on the process's GPU one).
        linear_route_metrics_cpu(owner.route, entries, b)
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
            owner.refusal = "ImportError: native LinearRegression TSQR entries unavailable: " + String(decomp)
        else:
            try:
                var decomp_connection = linear_entry_connection(decomp, LINEAR_TABLE_DECOMP)
                if (decomp_connection.key.mode != connection.key.mode
                        or decomp_connection.key.vendor != connection.key.vendor):
                    owner.refusal = "native LinearRegression TSQR execution key differs"
                else:
                    var entry = linear_table_entry[LINEAR_TABLE_DECOMP](decomp_connection.key, 0,
                        String(XD_ABI_NAME), XD_OPERATION_COUNT)
                    owner.route.fourth = EntryRef(entry, entry_ref_word(decomp_connection.key.mode,
                        decomp_connection.key.vendor, XD_OPERATION_COUNT))
                    owner.route.has_fourth = True
                    decomp_images = decomp_connection.images
            except error:
                owner.refusal = String(error)
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
    var route: DeviceRoute
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
    var owner = device_route_held(model, mode, default_mode, context)
    if owner is None:
        # The one read of `device`. None, and an instance without the field
        # (built without its constructor), run this installation's device.
        var device = device_request(fields, b)
        owner = device_route_keep(model, _linear_route_build(device, ridge, mode, default_mode, context, b))
    return LinearRouteCall(device_route_value(owner), owner)


def linear_route_decomp_refuse(owner: PythonObject) raises:
    """Raise why this estimator's route holds no TSQR entry (failure path of
    a LinearRegression fit whose plan takes the TSQR route)."""
    device_route_refuse(owner, String("ImportError: native LinearRegression TSQR entries unavailable"))
