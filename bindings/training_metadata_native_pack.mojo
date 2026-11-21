# SPDX-License-Identifier: Apache-2.0
# Public training metadata: fixed compiled entries, no runtime loader callbacks.
from core.mode_answer import MODE_FAMILY_NEURAL, answer_mode_word
from bindings.mode_answer_door import answer_mode_object
from std.python import PythonObject
from bindings.py_balanced import py_call, py_type
from bindings.optimizer_descriptor_boundary_door import optimizer_native_mode
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_connect_factory_once, neural_descriptor_execute
from training.optimizer_native_api import OPTIMIZER_NATIVE_ABI, OPT_NATIVE_COUNT, OPT_NATIVE_VENDOR
from training.optimizer_native_extra import TRAINING_VENDOR_NONE
from core.device_route import device_vendor_word
from bindings.optimizer_keys_unpack import optimizer_keys
from bindings.optimizer_keys_unpack import OPTIMIZER_OPN_TRAINING_NUMERIC_MODE
from bindings.py_balanced import py_item


def training_metadata(operation: Int, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = optimizer_keys()
    var types = py_item(context, nk[].k_training_metadata_types)
    var selected = mode
    # The former public wrapper treated an empty string as the default too.
    # Check its exact native type before truth/strip/lower can reach a hook.
    if selected is None or (py_type(selected) is types[0] and len(selected) == 0):
        selected = py_item(py_item(context, nk[].k_backend), nk[].k__DEFAULT_MODE)
    if py_type(selected) is not types[0]:
        raise Error("ValueError: numeric_mode requires None or a native str ('fast' or 'identical'); Python conversion hooks cannot execute")
    selected = py_call(py_item(context, nk[].k_training_metadata_lower), py_call(py_item(context, nk[].k_training_metadata_strip), selected))
    var requested = String(selected)
    if requested != "identical" and requested != "fast":
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    answer_mode_object(selected, requested, MODE_FAMILY_NEURAL)
    answer_mode_word(requested, MODE_FAMILY_NEURAL)
    var entries = py_item(context, nk[].k_training_metadata_bindings)[selected]
    if py_type(entries) is types[0]:
        # Import-time binding failures retain their original category/message.
        raise Error(String(entries))
    # lane dw-autograd-seq (2026-11-21): the record is the image's compiled
    # native table factory (`optimizer_native_api_v1`), connected once per
    # factory. The tier is the table's own OPT_NATIVE_MODE answer, read when
    # the image was connected; the vendor is one OPT_NATIVE_VENDOR call.
    # Before: the image's `training_numeric_mode` / `training_vendor` Python
    # entries, called per read. Both still answer from the called binary's
    # constants, never from a directory or a requested label.
    if py_type(entries) is not types[3] or len(entries) != 1:
        raise Error("TypeError: training metadata requires its connected compiled entries")
    if py_type(entries[0]) is not types[2]:
        raise Error("TypeError: training metadata requires its connected compiled entries")
    var provider = neural_descriptor_connect_factory_once(entries[0], String(OPTIMIZER_NATIVE_ABI), OPT_NATIVE_COUNT)
    var numeric = optimizer_native_mode(provider)
    var compiled = String("identical") if numeric == 1 else String("fast") if numeric == 0 else String("unknown")
    if compiled != requested:
        if py_item(py_item(context, nk[].k_backend), nk[].k__CPU_ONLY) is not None:
            raise Error("ImportError: mojolearn: _mojolearn_training was compiled " + compiled + " but numeric_mode asked for " + requested + " -- a binary is in the wrong directory; rebuild the sets with\n    bash bindings/build_training.sh\n    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_training.sh")
        raise Error("RuntimeError: mojolearn: _mojolearn_training was compiled for " + compiled + ", but this call requested " + requested + "; rebuild the " + requested + " binding")
    if operation == OPTIMIZER_OPN_TRAINING_NUMERIC_MODE:
        return PythonObject(compiled)
    var vendor = neural_descriptor_execute(provider, OPT_NATIVE_VENDOR, List[Int](), List[Int](), List[Float64](), False)
    if vendor == TRAINING_VENDOR_NONE:
        return PythonObject("none")
    return PythonObject(device_vendor_word(vendor))
