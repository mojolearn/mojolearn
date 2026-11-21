# SPDX-License-Identifier: Apache-2.0
"""ARIMA's and AutoARIMA's device route on the one route owner.

The route is a `DeviceRouteOwner` (providers/python/device_route_boundary_door.mojo,
rule and words core/device_route.mojo): slot `entry` is the resolved ARIMA
operations entry (core/arima_entry_table.mojo), `flags` its ARIMA_HAS_*
bits, `images` the module that published it. `arima_route` is the per-call
read: one record read and three pointer compares (the numeric_mode object,
the default mode object when the mode is None, the connection context); no
call reads or compares the device, and the class question (a host inference
class) is asked only on a miss. A miss resolves once: the selected binding's
table, the CPU table for `device="cpu"` beside a GPU set, or the forecast
inference image's table for the explicit host class.

AutoARIMA holds one route (its own `device`, the default numeric mode, as
its per-group models always ran) and hands it to every per-group fit and
query, so a per-group model never resolves one of its own.

The resident device copy's lifetime owner is bindings/tsa_hw_route_unpack.mojo's
`TsaHwResidentOwner`, minted here with the ARIMA table's release operation.

Python objects appear here only as pins and identities; RUN is
core/arima_run.mojo.
"""
from std.python import PythonObject
from bindings.mode_answer_door import answer_mode_object
from bindings.mode_metadata_native_door import _class_fields, _truth
from bindings.native_class_admission_door import class_record, class_admit
from bindings.native_loaded_module_door import NativeModules
from bindings.py_balanced import py_call, py_type, py_get, py_item, py_contains
from bindings.tsa_hw_route_unpack import TsaHwCall, tsa_resident_new
from bindings.tsa_keys_unpack import tsa_keys
from core.arima_entry_table import ARIMA_OP_RESIDENT_RELEASE, ARIMA_TABLE_FORECAST
from core.arima_run import arima_route_resolve
from core.device_route import (
    DeviceRoute, DEVICE_CPU, device_mode_code, device_route_plan, device_vendor_code,
)
from core.mode_answer import MODE_FAMILY_TSA, fast_is_answered_by_identical
from providers.python.device_route_boundary_door import (
    DeviceRouteOwner, device_admit, device_request, device_route_held, device_route_keep, device_route_value,
)

comptime _A = "mojolearn ARIMA: "
comptime _HOST_ONLY_SLOT = "tsa_arima_host_only_v1"


def arima_process_vendor(context: PythonObject) raises -> Int:
    """The vendor word of this installation's selected ARIMA binding (0 CPU
    only; -1 when it did not load: its own refusal is said by the first
    operation, not by the device check)."""
    var nk = tsa_keys()
    return device_vendor_code(String(py_item(context, nk[].k_arima_install_vendor)))


def arima_device_admit(value: PythonObject, context: PythonObject, b: PythonObject) raises:
    """The constructor's check of `device` (the shared `device_admit`): an
    unknown value and an explicit `gpu` on a CPU-only installation are
    refused before any field changes."""
    device_admit(value, arima_process_vendor(context), b)


def _arima_host_only(model: PythonObject, b: PythonObject) raises -> Bool:
    """The class is a host inference class (`_HOST_INFERENCE_ONLY`). The
    answer is kept per class under the type watcher (an edit of the class or
    an ancestor misses); asked on a route miss only."""
    var nk = tsa_keys()
    var cls = py_type(model)
    var mods = NativeModules()
    var kept = class_record(mods, _HOST_ONLY_SLOT, cls)
    if kept is not None:
        return Bool(py=kept)
    var host_only = False
    var ancestry = py_call(py_item(b.type.__dict__, nk[].k___mro__).__get__, cls)
    for i in range(len(ancestry)):  # driver-loop: native MRO metadata, never class lookup protocols
        var namespace = _class_fields(ancestry[i])
        if py_contains(namespace, nk[].k__HOST_INFERENCE_ONLY):
            host_only = _truth(py_item(namespace, nk[].k__HOST_INFERENCE_ONLY))
            break
    class_admit(mods, _HOST_ONLY_SLOT, cls, PythonObject(host_only))
    return host_only


def _arima_cells_route(device: Int, cpu_table: Bool, expected_vendor: Int, want: PythonObject, cells: PythonObject,
    b: PythonObject, mut owner: DeviceRouteOwner) raises:
    # `cells` is one connected binding of python/mojolearn/_native_tsa_bridge.py:
    # the module (the image pin) and the table key it reported at import, or
    # the text of why it did not load.
    var nk = tsa_keys()
    if py_type(cells) is b.str:
        raise Error(String(cells))
    if py_type(cells) is not b.dict:
        raise Error("TypeError: the ARIMA binding connection must be an exact dictionary")
    var family = Int(py=py_item(cells, nk[].k_hw_cell_family))
    var mode = Int(py=py_item(cells, nk[].k_hw_cell_mode))
    var vendor = device_vendor_code(String(py_item(cells, nk[].k_hw_cell_vendor)))
    if cpu_table and vendor != 0:
        raise Error("RuntimeError: device='cpu' requires the CPU identical binding")
    if not cpu_table and expected_vendor >= 0 and vendor != expected_vendor:
        raise Error("RuntimeError: ARIMA native vendor differs from selected execution")
    owner.route = arima_route_resolve(device, family, mode, vendor)
    # The binary's own tier, read back: a binary that disagrees with the mode
    # the package asked for is refused before it runs.
    if owner.route.mode != (1 if String(py=want) == "identical" else 0):
        raise Error("RuntimeError: " + _A + "numeric_mode=" + String(py=py_call(b.repr, want)) + " was requested but "
                    + String(py=py_item(cells, nk[].k_module_name)) + " reports compile-time mode code "
                    + String(owner.route.mode) + "; the binary and the directory it sits in disagree, rebuild it")
    owner.images = py_item(cells, nk[].k_hw_cell_module)


def _arima_route_build(device: Int, host_only: Bool, mode: PythonObject,
    default_mode: PythonObject, context: PythonObject, b: PythonObject) raises -> DeviceRouteOwner:
    var nk = tsa_keys()
    var owner = DeviceRouteOwner(device)
    if host_only:
        # lane fx-fast-mask: a model saved under `fast` where the identical set
        # answered it (TSA) holds identical state; the host runs it.
        if mode is not None and (py_type(mode) is not b.str or (String(py=mode) != "identical" and not (
                String(py=mode) == "fast" and fast_is_answered_by_identical(MODE_FAMILY_TSA)))):
            raise Error("ValueError: mojolearn: " + String("ARIMA")
                        + " runs IDENTICAL only on the host; this model was saved " + String(py=py_call(b.repr, mode)))
        var module = py_get(py_item(context, nk[].k_modules), py_item(context, nk[].k_hw_host_module))
        if module is None:
            raise Error("RuntimeError: mojolearn: the host " + String("ARIMA")
                        + " has no loaded forecast binding; load the model with its own load()")
        if py_type(module) is not py_type(b):
            raise Error("TypeError: host ARIMA requires its native module")
        owner.route = arima_route_resolve(DEVICE_CPU, ARIMA_TABLE_FORECAST, 1, 0)
        owner.images = module
    else:
        var selected = mode
        if mode is None:
            selected = default_mode
        if py_type(selected) is not b.str:
            raise Error("ValueError: numeric_mode must be fast, identical or None")
        var text = String(py=selected)
        if text != "identical" and text != "fast":  # exact modes skip two Python calls
            selected = py_call(py_call(selected.strip).lower)
        var mode_word = String(py=selected)
        if mode_word != "identical" and mode_word != "fast":
            raise Error("ValueError: numeric_mode must be fast, identical or None")
        answer_mode_object(selected, mode_word, MODE_FAMILY_TSA)
        var plan = device_route_plan(device, arima_process_vendor(context), device_mode_code(mode_word))
        if plan.cpu_table:
            var cpu = py_get(context, nk[].k_arima_cpu_bindings)
            if cpu is None:
                raise Error("ImportError: device='cpu' is not connected in this installation")
            _arima_cells_route(plan.device, True, 0, selected, py_item(cpu, selected), b, owner)
        else:
            _arima_cells_route(plan.device, False, plan.expected_vendor, selected,
                py_item(py_item(context, nk[].k_arima_bindings), selected), b, owner)
    owner.mode = mode
    owner.default_mode = default_mode
    owner.context = context
    owner.resolved = True
    return owner^


def arima_route(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject) raises -> TsaHwCall:
    """An ARIMA's route: the held one when it was resolved for this
    numeric_mode object, this default mode object and this connection
    context, else a fresh resolution (the device request and the class
    question are read once, here) kept for the next call."""
    var nk = tsa_keys()
    var mode = py_get(fields, nk[].k_numeric_mode)
    var default_mode = PythonObject(None)
    if mode is None:
        default_mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    var owner = device_route_held(model, mode, default_mode, context)
    if owner is None:
        owner = device_route_keep(model,
            _arima_route_build(device_request(fields, b), _arima_host_only(model, b), mode, default_mode, context, b))
    return TsaHwCall(device_route_value(owner), owner)


def autoarima_route(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject) raises -> TsaHwCall:
    """An AutoARIMA's one route, shared by its search, its per-group fits
    and their queries: its own `device`, the default numeric mode (its
    per-group models carry numeric_mode None)."""
    var nk = tsa_keys()
    var mode = PythonObject(None)
    var default_mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    var owner = device_route_held(model, mode, default_mode, context)
    if owner is None:
        owner = device_route_keep(model,
            _arima_route_build(device_request(fields, b), False, mode, default_mode, context, b))
    return TsaHwCall(device_route_value(owner), owner)


def arima_resident_new(route: DeviceRoute, handle: Int, images: PythonObject) raises -> PythonObject:
    """The lifetime owner of `handle`, minted through `route`'s entry and
    returned through its ARIMA_OP_RESIDENT_RELEASE."""
    return tsa_resident_new(route.entry, ARIMA_OP_RESIDENT_RELEASE, handle, images)
