# SPDX-License-Identifier: Apache-2.0
"""UNPACK of a Sequential model's device and numeric mode into its route owner.

`device=None` (default: this installation's device), `device="gpu"` or
`device="cpu"` is a constructor parameter of Sequential (and of
`from_checkpoint` / `load`). The rule and the words are the pilot's
(core/linear_route.mojo): the value is stored as given, checked where it is
given, and resolved ONCE per model into a `SequentialRouteOwner` kept in the
weak model-record table. The owner holds the entry dictionary of the selected
compiled binding: the process's own table of the numeric mode
(`context["bindings"][mode]`) or the CPU table loaded beside it on a GPU
installation (`context["sequential_cpu_bindings"]["identical"]`,
python/mojolearn/_x_sequence_model.py).

What a public call still compares: the model's `numeric_mode` and `device`
objects, the process default mode object when the mode is None, and the
connection context (identity compares against the owner's pins; one record
read). A miss resolves again with every check the former per-call selection
made (mode word, answered mode, compiled mode and vendor of the binding, the
shared provider of the binding's exact factory).

The owner replaces the per-call body of `sequential_entries`
(bindings/sequential_execution_pack.mojo): the mode word String, the answered
mode, the bindings dictionary read with a new key str, the vendor verdict
read and the provider reuse compare of every fit, predict, forward, save and
weight load.

Resolution is a metadata operation with no independent numerical work to
split; it runs at a model's first operation, never per row.
"""
from core.mode_answer import MODE_FAMILY_NEURAL, answer_mode_word
from core.linear_route import (
    LINEAR_DEVICE_CPU, LINEAR_DEVICE_INSTALL, linear_device_code, linear_device_refusal, linear_device_installed,
)
from std.python import PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_fields, py_get, py_item, py_setitem, py_type
from bindings.py_model_records import records_get, records_put, records_drop, RECORDS_PREPARED
from autograd.model_descriptor import MODEL_ABI_NAME
from sequence.neural_model_dispatch import MODEL_OPERATION_COUNT
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_reuse
from bindings.sequential_keys_unpack import sequential_keys

comptime _SEQUENTIAL_ROUTE_SLOT = "sequential_route_v1"


struct SequentialRouteOwner(Movable, Writable):
    # The device the route resolved to (LINEAR_DEVICE_GPU or LINEAR_DEVICE_CPU).
    var device: Int
    # The objects the route was resolved for. Held, so their addresses stay
    # theirs while this owner lives; a call compares identities only.
    var mode: PythonObject
    var default_mode: PythonObject
    var requested: PythonObject
    var context: PythonObject
    # The entry dictionary of the selected compiled binding (it keeps the
    # binding's entries and its shared provider, so the image stays loaded).
    var entries: PythonObject

    def __init__(out self, device: Int):
        self.device = device
        self.mode = PythonObject(None)
        self.default_mode = PythonObject(None)
        self.requested = PythonObject(None)
        self.context = PythonObject(None)
        self.entries = PythonObject(None)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("SequentialRouteOwner(device=", self.device, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def sequential_device_unpack(value: PythonObject) raises -> Int:
    """The device code of a constructor value: None (this installation's
    device), or exact `'gpu'` or `'cpu'`. Anything else is refused in the
    pilot's words, before any state changes."""
    if value is None:
        return LINEAR_DEVICE_INSTALL
    var b = native_loaded_module("builtins")
    if py_type(value) is b.str:
        var code = linear_device_code(String(value))
        if code >= 0:
            return code
    linear_device_refusal(String(py_call(b.repr, value)))
    return LINEAR_DEVICE_INSTALL


def sequential_device_admit(value: PythonObject, context: PythonObject) raises:
    """Check a device where it is given (the constructor, a checkpoint
    load): an unknown value, and an explicit `'gpu'` in a CPU-only process,
    are refused in words before the model changes. The default None reads
    nothing here."""
    if value is None:
        return
    var device = sequential_device_unpack(value)
    if device == LINEAR_DEVICE_CPU:
        return
    var nk = sequential_keys()
    _ = linear_device_installed(device, 0 if String(py_item(context, nk[].k_vendor)) == "cpu" else 1)


def sequential_route_clear(model: PythonObject) raises:
    """Forget the model's route (a new `__init__`): its next operation reads
    the device and the mode again."""
    records_drop(RECORDS_PREPARED, model, _SEQUENTIAL_ROUTE_SLOT)


def _sequential_route_build(requested: PythonObject, mode: PythonObject, default_mode: PythonObject,
                            context: PythonObject) raises -> SequentialRouteOwner:
    var nk = sequential_keys()
    var b = native_loaded_module("builtins")
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
    # None is this installation's device; an explicit 'gpu' in a CPU-only
    # process is refused in words (a model that reached here without its
    # constructor's check: unpickled, or its field assigned).
    var device = linear_device_installed(sequential_device_unpack(requested), 1 if process_gpu else 0)
    var owner = SequentialRouteOwner(device)
    var entries = PythonObject(None)
    var expected = vendor
    if device == LINEAR_DEVICE_CPU and process_gpu:
        # A GPU installation: the CPU table of the host binary, loaded beside
        # the device table at import.
        if word != "identical":
            raise Error("ValueError: device='cpu' runs in identical mode only; numeric_mode='fast' is for Apple GPUs")
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
    owner.requested = requested
    owner.context = context
    owner.entries = entries
    return owner^


def sequential_route_entries(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """The entry dictionary of the model's route. A hit is one record read
    and four pointer compares; a miss (first operation, a changed numeric
    mode or device, a model built without its constructor) resolves and
    stores it."""
    var nk = sequential_keys()
    var fields = py_fields(model)
    var mode = py_item(fields, nk[].k_numeric_mode)
    var default_mode = PythonObject(None)
    if mode is None:
        default_mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    # A model without the field (built without its constructor, or restored
    # from a pickle of an older version) runs this installation's device.
    var requested = py_get(fields, nk[].k_device)
    var kept = records_get(RECORDS_PREPARED, model, _SEQUENTIAL_ROUTE_SLOT)
    if kept is not None:
        var native = kept.downcast_value_ptr[SequentialRouteOwner]()
        if (native[].mode is mode and native[].default_mode is default_mode
                and native[].requested is requested and native[].context is context):
            var held = native[].entries
            _ = kept  # the owner stays pinned through the read
            return held
    var fresh = PythonObject(alloc=_sequential_route_build(requested, mode, default_mode, context))
    # A model that takes no weak reference keeps no owner: it resolves at
    # every operation (records_put answers False and stores nothing).
    _ = records_put(RECORDS_PREPARED, model, _SEQUENTIAL_ROUTE_SLOT, fresh)
    var entries = fresh.downcast_value_ptr[SequentialRouteOwner]()[].entries
    _ = fresh
    return entries
