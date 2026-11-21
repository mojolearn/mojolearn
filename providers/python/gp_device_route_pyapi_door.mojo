# SPDX-License-Identifier: Apache-2.0
"""UNPACK of a Gaussian process estimator's device into its route record.

`device=None` (default: this installation's device), `device="gpu"` or
`device="cpu"` is a constructor parameter of GaussianProcessRegressor and
GaussianProcessClassifier, with the rule and the words of
core/linear_route.mojo. The constructor stores it verbatim, as it stores every
other parameter of these classes; it is read ONCE, at the estimator's first
operation, into a route record kept in the weak model-record table (the
GP_SVM family of bindings/py_model_records.mojo, beside the estimator's
admitted-state records). The record holds the images the estimator runs on:

- the GP image: the selected device image `_mojolearn_gp`, or, for
  `device="cpu"` in a GPU installation, the CPU image `_mojolearn_gp_host`
  (column 1 of core/gp_entry_table.mojo), connected at import beside the
  device image (python/mojolearn/_native_gp_device_bridge.py);
- for a classifier, the base image its label helpers live on (`_mojolearn`,
  or the CPU `_mojolearn_core_host`);
- for a CPU route in a GPU installation, the R^2 program factory of the CPU
  metrics image (`_mojolearn_x_metrics_host`), which `score` runs.

Every later fit, predict, sample and score reads the record and runs on the
images it holds: no per-call class-plan read, route String, binding registry
walk or device question.

What a call compares (pointer compares against the record's pins): the
estimator's `numeric_mode` object, the process default mode object when that
is None, its `device` object and the connection context. A changed mode or
device (`set_params` rebuilds through the constructor; a field assigned
directly) resolves again. A copy, a clone and an unpickled estimator have no
record and resolve at their first operation.

The record replaces, per public call: the class-plan lookup and its route
String, the mode word normalization, the binding registry reads of
`estimator_binding` (connections, offerings, the mode set and its modules
dictionary), the compiled-mode check of the image and, for the explicit
host classes, the live `sys.modules` read and the host image's mode call.

Resolution is a metadata operation with no independent numerical work to
split; it runs at an estimator's first operation, never per row.
"""
from std.python import Python, PythonObject
from bindings.cpython_borrowed_pack import borrowed_item
from bindings.gp_keys_unpack import gp_keys
from bindings.py_balanced import py_call, py_get, py_item, py_type
from bindings.py_model_records import records_get, records_put, RECORDS_GP_SVM
from core.linear_route import (
    LINEAR_DEVICE_CPU, LINEAR_DEVICE_INSTALL, linear_device_code, linear_device_refusal, linear_device_installed,
)

comptime _GP_ROUTE_SLOT = "gp_device_route_v1"

# The record: an exact tuple of these slots.
comptime GP_ROUTE_MODE = 0
comptime GP_ROUTE_DEFAULT_MODE = 1
comptime GP_ROUTE_DEVICE = 2
comptime GP_ROUTE_CONTEXT = 3
comptime GP_ROUTE_BINDING = 4
comptime GP_ROUTE_BASE = 5
comptime GP_ROUTE_METRICS = 6
comptime GP_ROUTE_VENDOR = 7


def gp_device_unpack(value: PythonObject, b: PythonObject) raises -> Int:
    """The device code of a constructor value: None (this installation's
    device), or exact `'gpu'` or `'cpu'`. Anything else is refused in the
    words of core/linear_route.mojo. `b` is the builtins module the calling
    door already admitted."""
    if value is None:
        return LINEAR_DEVICE_INSTALL
    if py_type(value) is b.str:
        var code = linear_device_code(String(value))
        if code >= 0:
            return code
    linear_device_refusal(String(py_call(b.repr, value)))
    return LINEAR_DEVICE_INSTALL


def gp_route_hit(model: PythonObject, mode: PythonObject, default_mode: PythonObject, device: PythonObject,
                 context: PythonObject) raises -> PythonObject:
    """The estimator's route record when it was resolved for these exact
    objects, else None: one record read and four pointer compares."""
    var record = records_get(RECORDS_GP_SVM, model, String(_GP_ROUTE_SLOT))
    if record is None:
        return record
    if (Int(borrowed_item(record, GP_ROUTE_MODE, True)) != Int(mode._obj_ptr)
            or Int(borrowed_item(record, GP_ROUTE_DEFAULT_MODE, True)) != Int(default_mode._obj_ptr)
            or Int(borrowed_item(record, GP_ROUTE_DEVICE, True)) != Int(device._obj_ptr)
            or Int(borrowed_item(record, GP_ROUTE_CONTEXT, True)) != Int(context._obj_ptr)):
        return PythonObject(None)
    return record


def gp_route_keep(model: PythonObject, mode: PythonObject, default_mode: PythonObject, device: PythonObject,
                  context: PythonObject, binding: PythonObject, base: PythonObject, metrics: PythonObject,
                  cpu_beside_gpu: Bool) raises -> PythonObject:
    """Store the resolved route for `model` and return it. The record holds
    the objects it was resolved for (their addresses stay theirs while it
    lives) and the images; it never holds the estimator. An estimator that
    takes no weak reference keeps no record and resolves at every operation."""
    var vendor = PythonObject("cpu") if cpu_beside_gpu else PythonObject(None)
    var record = Python.tuple(mode, default_mode, device, context, binding, base, metrics, vendor)
    _ = records_put(RECORDS_GP_SVM, model, String(_GP_ROUTE_SLOT), record)
    return record


def gp_route_item(record: PythonObject, slot: Int) raises -> PythonObject:
    """One slot of a route record (a borrowed read of an exact tuple)."""
    return PythonObject(from_borrowed=borrowed_item(record, slot, True))


@fieldwise_init
struct GPCpuImages(Movable):
    """The CPU images of a `device="cpu"` route in a GPU installation."""
    var binding: PythonObject
    var base: PythonObject
    var metrics: PythonObject


def _gp_cpu_cell(cells: PythonObject, key: PythonObject, b: PythonObject) raises -> PythonObject:
    # A CPU binary that is not built is kept as its refusal text.
    var cell = py_item(cells, key)
    if py_type(cell) is b.str:
        raise Error("ImportError: " + String(cell))
    return cell


def gp_route_device(device: PythonObject, context: PythonObject, b: PythonObject) raises -> Bool:
    """The one read of `device`: True when the estimator runs the CPU images
    beside this installation's GPU set, False when it runs the process's own
    images (the GPU set for None or 'gpu'; the CPU set in a CPU-only process).
    An unknown value, and an explicit 'gpu' in a CPU-only process, are
    refused in words."""
    var nk = gp_keys()
    var cpu_only = py_item(py_item(context, nk[].k_backend_fields), nk[].k__CPU_ONLY) is not None
    var resolved = linear_device_installed(gp_device_unpack(device, b), 0 if cpu_only else 1)
    return resolved == LINEAR_DEVICE_CPU and not cpu_only


def gp_route_cpu_images(mode: PythonObject, default_mode: PythonObject, classifier: Bool,
                        context: PythonObject, b: PythonObject) raises -> GPCpuImages:
    """The CPU images of a `device="cpu"` estimator in a GPU installation
    (context `gp_cpu_bindings`, connected at import). The CPU runs identical
    only."""
    var nk = gp_keys()
    var selected = mode
    if mode is None:
        selected = default_mode
    if py_type(selected) is not b.str:
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var word = String(selected)
    if word != "identical":
        if word != "fast":
            # Only a non-canonical spelling pays the two str method calls.
            word = String(py_call(py_call(selected.strip).lower))
        if word == "fast":
            raise Error("ValueError: device='cpu' runs in identical mode only; numeric_mode='fast' is for Apple GPUs")
        if word != "identical":
            raise Error("ValueError: numeric_mode must be fast, identical or None")
    var cells = py_get(context, nk[].k_gp_cpu_bindings)
    if cells is None:
        raise Error("ImportError: device='cpu' is not connected in this installation")
    var binding = _gp_cpu_cell(cells, nk[].k__mojolearn_gp, b)
    var base = PythonObject(None)
    if classifier:
        base = _gp_cpu_cell(cells, nk[].k__mojolearn, b)
    # The metrics factory is refused by the `score` that needs it, not here:
    # a fit or a predict runs without the CPU metrics image.
    var metrics = py_item(cells, nk[].k_metrics)
    return GPCpuImages(binding, base, metrics)
