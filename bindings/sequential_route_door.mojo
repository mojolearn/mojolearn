# SPDX-License-Identifier: Apache-2.0
"""Resolver of a Sequential model's route onto the one device route owner.

`device=None` (default: this installation's device), `device="gpu"` or
`device="cpu"` is a constructor parameter of Sequential (and of
`from_checkpoint` / `load`). The rule and the words are the shared ones
(core/device_route.mojo); the storage is the shared `DeviceRouteOwner`
(providers/python/device_route_boundary_door.mojo), whose `table` pin holds
the entry dictionary of the selected compiled binding: the process's own
table of the numeric mode (`context["bindings"][mode]`) or the CPU table
loaded beside it on a GPU installation
(`context["sequential_cpu_bindings"]["identical"]`,
python/mojolearn/_x_sequence_model.py).

The device is checked and refused at construction (and at a checkpoint
load) and read ONCE, at the model's first operation, into the owner. No
public call reads or compares the device: a hit is one record read and three
pointer compares (the `numeric_mode` object, the process default mode object
when the mode is None, the connection context). A new `__init__` drops the
owner. Assigning `model.device` directly is not a supported way to change it.

Resolution is a metadata operation with no independent numerical work to
split; it runs at a model's first operation, never per row.
"""
from core.mode_answer import MODE_FAMILY_NEURAL, answer_mode_word
from core.device_route import DEVICE_CPU, device_cpu_fast, device_installed
from std.python import PythonObject
from bindings.native_loaded_module_door import NativeModules
from bindings.py_balanced import py_call, py_fields, py_get, py_item, py_setitem, py_type
from autograd.model_descriptor import MODEL_ABI_NAME
from sequence.neural_model_dispatch import MODEL_OPERATION_COUNT
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_reuse
from providers.python.device_route_boundary_door import (
    DeviceRouteOwner, device_admit, device_unpack, device_route_clear, device_route_held, device_route_keep,
)
from bindings.sequential_keys_unpack import sequential_keys


def sequential_device_admit(value: PythonObject, context: PythonObject) raises:
    """Check a device where it is given (the constructor, a checkpoint
    load) through the shared admission: an unknown value, and an explicit
    `'gpu'` in a CPU-only process, are refused in the shared words before
    the model changes. The default None reads nothing here."""
    if value is None:
        return
    var nk = sequential_keys()
    var mods = NativeModules()
    device_admit(value, 0 if String(py_item(context, nk[].k_vendor)) == "cpu" else 1, mods.module("builtins"))


def sequential_route_clear(model: PythonObject) raises:
    """Forget the model's route (a new `__init__`): its next operation reads
    the device and the mode again."""
    device_route_clear(model)


def _sequential_route_build(model: PythonObject, mode: PythonObject, default_mode: PythonObject,
                            context: PythonObject) raises -> DeviceRouteOwner:
    var nk = sequential_keys()
    # Miss path only: the loaded-module table of this call.
    var mods = NativeModules()
    var b = mods.module("builtins")
    var selected = mode
    if mode is None:
        selected = default_mode
    var word = String()
    if py_type(selected) is b.str:
        # An exact canonical spelling needs no isinstance, strip or lower call.
        word = String(selected)
    if word != "identical" and word != "fast":
        if not Bool(py=py_call(b.isinstance, selected, b.str)):
            raise Error("numeric_mode must be fast, identical or None")
        selected = selected.strip().lower()
        word = String(selected)
    if word != "identical" and word != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    answer_mode_word(word, MODE_FAMILY_NEURAL)
    var vendor = py_item(context, nk[].k_vendor)
    var process_gpu = String(vendor) != "cpu"
    # The one read of `device`. None, and a model without the field (built
    # without its constructor, or pickled before the parameter existed), run
    # this installation's device; an explicit 'gpu' in a CPU-only process is
    # refused in words.
    var device = device_installed(device_unpack(py_get(py_fields(model), nk[].k_device), b), 1 if process_gpu else 0)
    var owner = DeviceRouteOwner(device)
    var entries = PythonObject(None)
    var expected = vendor
    if device == DEVICE_CPU and process_gpu:
        # A GPU installation: the CPU table of the host binary, loaded beside
        # the device table at import.
        if word != "identical":
            device_cpu_fast()
        var cpu = py_get(context, nk[].k_sequential_cpu_bindings)
        if cpu is None:
            raise Error("ImportError: device='cpu' is not connected in this installation")
        entries = py_item(cpu, PythonObject(word))
        expected = PythonObject("cpu")
    else:
        # The process's own table: the GPU table for a GPU request, the CPU
        # table in a CPU-only process.
        entries = py_item(py_item(context, nk[].k_bindings), PythonObject(word))
    if py_type(entries) is not b.dict and Bool(py=py_call(b.isinstance, entries, b.str)):
        raise Error(String(entries))
    # The compiled mode/vendor of a connected binding cannot change within a
    # process: read them at the first resolution against a binding, then
    # reuse the verdict stored next to the provider.
    var checked = py_get(entries, nk[].k__sequential_checked_vendor)
    if checked is None or (checked is not expected and String(checked) != String(expected)):
        var actual_mode = Int(py=py_item(entries, nk[].k_x_sequence_numeric_mode)())
        var actual_vendor = String(py_item(entries, nk[].k_x_sequence_vendor)())
        if actual_mode != (1 if word == "identical" else 0):
            raise Error("RuntimeError: Sequential native numerical mode differs from the requested mode")
        if actual_vendor != String(expected):
            raise Error("RuntimeError: Sequential native vendor differs from the selected execution route")
        py_setitem(entries, nk[].k__sequential_checked_vendor, expected)
    # Connection/metadata admission only. Exact compiled factory/module identity
    # owns the shared provider; no per-model pointer cache or Python callable is
    # used by the subsequent native fit/predict execution.
    var provider = neural_descriptor_reuse(py_get(entries, nk[].k_native_model_provider),
        py_item(entries, nk[].k_model_native_api_v1), String(MODEL_ABI_NAME), MODEL_OPERATION_COUNT)
    py_setitem(entries, nk[].k_native_model_provider, provider)
    owner.mode = mode
    owner.default_mode = default_mode
    owner.context = context
    owner.table = entries
    owner.resolved = True
    return owner^


def sequential_route_entries(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """The entry dictionary of the model's route. A hit is one record read
    and three pointer compares; a miss (first operation, a changed numeric
    mode, a model built without its constructor) resolves and stores it."""
    var nk = sequential_keys()
    var mode = py_item(py_fields(model), nk[].k_numeric_mode)
    var default_mode = PythonObject(None)
    if mode is None:
        default_mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    var owner = device_route_held(model, mode, default_mode, context)
    if owner is None:
        owner = device_route_keep(model, _sequential_route_build(model, mode, default_mode, context))
    var entries = owner.downcast_value_ptr[DeviceRouteOwner]()[].table
    _ = owner  # the owner stays pinned through the read
    return entries
