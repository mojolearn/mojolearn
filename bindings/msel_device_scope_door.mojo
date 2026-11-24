# SPDX-License-Identifier: Apache-2.0
"""The admission and the run context of model selection's `device=` (lane
vdf-msel, 2026-11-24): the leaf part of bindings/msel_function_device_door.mojo
(its module note is the description), split out so the splitter door
(bindings/msel_split_door.mojo) can select its context where it already reads
the splitter's state, with no import of the doors that import it. Imports no
model-selection door."""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_item, py_setitem, py_type, py_contains
from bindings.metrics_keys_unpack import metrics_keys
from core.function_device import FunctionDevice
from providers.python.estimator_binding_pyapi_door import estimator_binding_device
from providers.python.function_device_boundary_door import function_device_admit, function_device_scope

# Which implementations model selection's functions have (door constants, the
# module note): both, on every function and splitter of this family.
comptime MSEL_HAS_GPU = True
comptime MSEL_HAS_CPU = True

comptime _MSEL_CPU_FAST_WORDS = "ValueError: device='cpu' runs in identical mode only; numeric_mode='fast' is for Apple GPUs"


def msel_device_admit(value: PythonObject, name: String, context: PythonObject,
                      b: PythonObject) raises -> FunctionDevice:
    """UNPACK of a model-selection function's `device` argument (`value`):
    the shared admission with the family's constants. `name` is the public
    name (it leads the refusal words)."""
    return function_device_admit(value, name, MSEL_HAS_GPU, MSEL_HAS_CPU, context, b)


def msel_device_context(device: FunctionDevice, context: PythonObject, b: PythonObject) raises -> PythonObject:
    """The context the call runs in: `context` itself for every admitted value
    but an explicit "cpu" beside a GPU set, which runs in `msel_cpu_scope`."""
    if not device.host_beside:
        return context
    return msel_cpu_scope(device, context, b)


def msel_cpu_scope(device: FunctionDevice, context: PythonObject, b: PythonObject) raises -> PythonObject:
    """Model selection's CPU scope beside a GPU set: a copy of the shared CPU
    label scope (`function_device_scope`: the host label kernels, the CPU
    metrics tables and x_metrics runner entries) in which

    * `family_msel_entries` names the CPU column of the model-selection and
      base helper tables (`msel_native_key` / `base_native_key` = (mode, 0),
      published by `_mojolearn_core_host`, which `estimator_binding_device`
      connects first); the process's own key stays readable as
      `msel_process_key` (bindings/msel_member_device_unpack.mojo
      `msel_process_vendor`: this process still has a GPU set);
    * `family_metrics_prog_context` runs the CPU x_metrics image (identical
      only: its fast slot holds `device_cpu_fast`'s words);
    * `family_msel_resident_context` reads its input entries from this scope
      (the resident stores of train_test_split and the fold rows).

    Built once per CPU label scope and kept in it (a scope is copied again
    when the shared context grows; the copy builds its own once). A CPU image
    that is not connected is refused in words: the call never runs the GPU
    images in its place."""
    var scope = function_device_scope(device, context)
    var held = py_get(scope, "family_msel_cpu_scope")
    if held is not None:
        return held
    # The core host image publishes the CPU helper tables at its import; its
    # connection also refuses a fast default mode (`device_cpu_fast`).
    _ = estimator_binding_device(device, PythonObject(None), "_mojolearn", context)
    var programs = py_get(scope, "labels_metrics_entries")
    if programs is None:
        raise Error("ImportError: device='cpu' needs the CPU x_metrics image beside the GPU set, "
            + "which is not connected in this installation")
    if py_type(programs) is b.str:
        raise Error(String(py=programs))
    var made = py_call(b.dict, scope)
    var entries = py_call(b.dict, context["family_msel_entries"])
    var key = py_get(entries, "msel_native_key")
    if key is not None and py_type(key) is b.tuple:
        var host = Python.tuple(key[0], PythonObject(0))
        py_setitem(entries, "base_native_key", host)
        py_setitem(entries, "msel_native_key", host)
        py_setitem(entries, "msel_process_key", key)
    # (a recorded refusal string stays: the first helper call raises it)
    py_setitem(made, "family_msel_entries", entries)
    var nk = metrics_keys()
    var prog = py_call(b.dict, py_item(context, nk[].k_family_metrics_prog_context))
    var bindings = Python.dict()
    py_setitem(bindings, "identical", programs)
    py_setitem(bindings, "fast", PythonObject(_MSEL_CPU_FAST_WORDS))
    py_setitem(prog, "prog_bindings", bindings)
    if py_contains(prog, "vendor"):
        py_setitem(prog, "vendor", PythonObject("cpu"))
    py_setitem(made, nk[].k_family_metrics_prog_context, prog)
    var resident = py_call(b.dict, context["family_msel_resident_context"])
    py_setitem(resident, "arena_msel_input_context", made)
    py_setitem(made, "family_msel_resident_context", resident)
    py_setitem(scope, "family_msel_cpu_scope", made)
    return made
