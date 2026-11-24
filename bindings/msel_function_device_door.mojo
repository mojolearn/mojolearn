# SPDX-License-Identifier: Apache-2.0
"""`device=` of model selection's public functions and splitters (lane
vdf-msel, 2026-11-24): the family-local part of the ONE device mechanism
(core/function_device.mojo, providers/python/function_device_boundary_door.mojo).

Every model-selection function that computes runs both sides today: on a GPU
installation its fold, gather and label helpers are the base binding's
device column (core/msel_helpers_device.mojo), its random draws and scorers
are x_metrics programs on the device image, and its member fits run on each
member's route; the CPU column of each exists (core/msel_helpers_host.mojo,
published by `_mojolearn_core_host`; the `_mojolearn_x_metrics_host` image;
the members' CPU images). So every door here admits with HAS_GPU = HAS_CPU =
True, and:

* `device=None` is today's route: the door hands the call the context it ran
  in before (one identity compare in `function_device_admit`);
* `"gpu"` is today's route on a GPU installation (refused in the estimators'
  `device_gpu_absent` words on a CPU-only one);
* `"cpu"` on a CPU-only installation is today's route; beside a GPU set the
  door hands the call `msel_cpu_scope`: the shared CPU label scope
  (`function_device_scope`) with model selection's helper key, base helper
  key and x_metrics program runner pointed at the CPU column, and the
  resident store's input context pointed at that scope. Nothing in RUN
  changes: the typed routes it resolves from its context are the CPU ones.

THE MEMBER RULE (`msel_tool_member`): a cv tool's explicit `device` is
handed to the member it fits; a member constructed with a different explicit
`device` is refused before anything runs (`function_device_member`'s words).
Only one case changes the member: `"cpu"` beside a GPU set over a member that
names none. That member is cloned ONCE per tool call with `device='cpu'`
(through its own constructor or set_params, so its class's own device check
runs), and every fold or candidate clone the tool makes copies it. In every
other admitted case the member already runs where the tool says.

A splitter stores the `device` its constructor was given (only when given:
None stores nothing, so get_params and repr are as before) and its `split` /
`get_n_splits` run in the context that device selects
(`msel_splitter_device`).
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_item, py_setitem, py_type, py_contains
from bindings.metrics_keys_unpack import metrics_keys
from bindings.msel_estimator_native_door import msel_estimator_plan, msel_native_clone, msel_native_set_params
from bindings.msel_split_door import _splitter_fields
from core.device_route import DEVICE_CPU, DEVICE_INSTALL, device_cpu_absent
from core.function_device import FunctionDevice, function_device_member
from providers.python.device_route_boundary_door import device_request
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


def msel_tool_member(device: FunctionDevice, estimator: PythonObject, tool: String, context: PythonObject,
                     b: PythonObject) raises -> PythonObject:
    """THE MEMBER RULE of a cv tool (the module note): the estimator the tool
    fits for `device`. Called once per tool call, never per fold or
    candidate, and only for an explicit `device` (None never reaches here)."""
    var plan = msel_estimator_plan(estimator, context)
    var fields = py_call(plan["fields_get"], estimator, py_type(estimator))
    var own = device_request(fields, b)
    var name = String(py=py_type(estimator).__name__)
    _ = function_device_member(device.request, own, tool, name)
    if not device.host_beside or own == DEVICE_CPU:
        return estimator
    # `own` is DEVICE_INSTALL: the member would run on the GPU set.
    if not py_contains(plan["parameters"], "device"):
        device_cpu_absent(name, "its constructor takes no device, so " + tool + " cannot hand it device='cpu'")
    var member = msel_native_clone(estimator, context)
    var params = Python.dict()
    py_setitem(params, "device", PythonObject("cpu"))
    _ = msel_native_set_params(member, params, context)
    return member


def msel_tool_args(device: FunctionDevice, args: PythonObject, tool: String, context: PythonObject,
                   b: PythonObject) raises -> PythonObject:
    """A cv tool's argument tuple with its estimator (item 0) replaced by
    `msel_tool_member`'s answer: the same tuple when the member is kept."""
    var member = msel_tool_member(device, args[0], tool, context, b)
    if member is args[0]:
        return args
    var items = py_call(b.list, args)
    py_setitem(items, PythonObject(0), member)
    return py_call(b.tuple, items)


def msel_splitter_device(model: PythonObject, name: String, context: PythonObject,
                         b: PythonObject) raises -> FunctionDevice:
    """The admitted `device` a splitter's constructor stored (absent: None),
    for its `split` / `get_n_splits`: one dictionary read per call."""
    return msel_device_admit(py_get(_splitter_fields(model, context), "device"), name, context, b)
