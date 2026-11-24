# SPDX-License-Identifier: Apache-2.0
"""`device=` of the `mojolearn.training` functions (lane vdf-linalg,
2026-11-24): clip_grad_norm_, cross_entropy, accumulate_grads, the embedding,
RMS-normalization and linear layers forward and backward,
chunked_lm_head_loss, mlp_inference_sessions, residual_dropout and
residual_dropout_backward.

The Python shell passes `device` as the LAST item of the operation's argument
tuple; the router (bindings/optimizer_native_router_door.mojo
`optimizer_native`) reads it once, before any work, through the shared UNPACK
(providers/python/function_device_boundary_door.mojo `function_device_admit`;
the rule and its words are core/function_device.mojo). `device=None` costs
one integer range test and one identity compare: the router then resolves
today's entries (`optimizer_entries`) exactly as before.

Sides. The training binding has both: the device image `_mojolearn_training`
and the host image `_mojolearn_training_host` (host_surface's training family,
the whole route on a CPU-only install, which exports the same entries but the
`*_dev` device-array ones). mlp_inference_sessions runs the training
binding's `OPT_NATIVE_MLP_SESSIONS`, which the host image exports too
(training/host/optimizer_native_extra.mojo `nn_mlp_sessions_host`): lane
vdf-linalg2 (2026-11-24) routes its "cpu" there like every other function
here (was refused as `device_cpu_absent`, on the wrong belief that it ran the
x_sequence `small_mlp_program`).

An explicit "cpu" beside a GPU set runs the host image's entries, connected at
the first such call (`training_entries_cpu`, the bridge's `cpu_load_later`
load; python/mojolearn/_native_optimizer_bridge.py), identical mode only. An
input that already lives on the device (a `TrainDeviceArray`) disagrees with
it and is refused in words before any work; nothing is copied across.
"""
from std.python import PythonObject
from bindings.py_balanced import py_call, py_get, py_item, py_type
from bindings.native_loaded_module_door import native_loaded_module
from bindings.optimizer_keys_unpack import (
    optimizer_keys, OPTIMIZER_OPN_RESIDUAL_DROPOUT, OPTIMIZER_OPN_RESIDUAL_DROPOUT_BACKWARD,
    OPTIMIZER_OPN_LINEAR_FORWARD, OPTIMIZER_OPN_LINEAR_BACKWARD, OPTIMIZER_OPN_RMS_NORM_FORWARD,
    OPTIMIZER_OPN_RMS_NORM_BACKWARD, OPTIMIZER_OPN_EMBEDDING_FORWARD, OPTIMIZER_OPN_EMBEDDING_BACKWARD,
    OPTIMIZER_OPN_CROSS_ENTROPY, OPTIMIZER_OPN_CHUNKED_LM_HEAD_LOSS, OPTIMIZER_OPN_MLP_INFERENCE_SESSIONS,
    OPTIMIZER_OPN_CLIP, OPTIMIZER_OPN_ACCUMULATE,
)
from core.device_route import device_cpu_fast
from core.function_device import FunctionDevice
from providers.python.device_route_boundary_door import device_cpu_connect_late
from providers.python.function_device_boundary_door import function_device_admit

comptime TRAINING_HAS_GPU = True
comptime TRAINING_HAS_CPU = True
comptime _TRAINING_CPU_ENTRIES = "training_bindings_cpu"


def training_function_op(operation: Int) -> Bool:
    """True for the public functions that take `device` (integer compares)."""
    return (operation == OPTIMIZER_OPN_CLIP or operation == OPTIMIZER_OPN_ACCUMULATE
        or operation == OPTIMIZER_OPN_RESIDUAL_DROPOUT or operation == OPTIMIZER_OPN_RESIDUAL_DROPOUT_BACKWARD
        or (operation >= OPTIMIZER_OPN_LINEAR_FORWARD and operation <= OPTIMIZER_OPN_CHUNKED_LM_HEAD_LOSS)
        or operation == OPTIMIZER_OPN_MLP_INFERENCE_SESSIONS)


def _training_function_name(operation: Int) -> String:
    """The public name a `device=` refusal of `operation` leads with."""
    if operation == OPTIMIZER_OPN_CLIP:
        return "mojolearn.training.clip_grad_norm_"
    if operation == OPTIMIZER_OPN_ACCUMULATE:
        return "mojolearn.training.accumulate_grads"
    if operation == OPTIMIZER_OPN_RESIDUAL_DROPOUT:
        return "mojolearn.training.residual_dropout"
    if operation == OPTIMIZER_OPN_RESIDUAL_DROPOUT_BACKWARD:
        return "mojolearn.training.residual_dropout_backward"
    if operation == OPTIMIZER_OPN_LINEAR_FORWARD:
        return "mojolearn.training.linear_forward"
    if operation == OPTIMIZER_OPN_LINEAR_BACKWARD:
        return "mojolearn.training.linear_backward"
    if operation == OPTIMIZER_OPN_RMS_NORM_FORWARD:
        return "mojolearn.training.rms_norm_forward"
    if operation == OPTIMIZER_OPN_RMS_NORM_BACKWARD:
        return "mojolearn.training.rms_norm_backward"
    if operation == OPTIMIZER_OPN_EMBEDDING_FORWARD:
        return "mojolearn.training.embedding_forward"
    if operation == OPTIMIZER_OPN_EMBEDDING_BACKWARD:
        return "mojolearn.training.embedding_backward"
    if operation == OPTIMIZER_OPN_CROSS_ENTROPY:
        return "mojolearn.training.cross_entropy"
    if operation == OPTIMIZER_OPN_CHUNKED_LM_HEAD_LOSS:
        return "mojolearn.training.chunked_lm_head_loss"
    return "mojolearn.training.mlp_inference_sessions"


def training_function_device(operation: Int, item: PythonObject, values: PythonObject,
                             context: PythonObject) raises -> FunctionDevice:
    """UNPACK of a training function's explicit (non-None) `device` item:
    the shared admission, then, for an explicit "cpu" beside a GPU set, the
    refusal of an input that lives on the device. `values` is the argument
    tuple; its last item is `item`."""
    var name = _training_function_name(operation)
    var device = function_device_admit(item, name, TRAINING_HAS_GPU, TRAINING_HAS_CPU,
                                       context, native_loaded_module("builtins"))
    if device.host_beside:
        var nk = optimizer_keys()
        var device_class = py_item(context, nk[].k_device_class)
        for i in range(len(values) - 1):  # driver-loop: over the call's public arguments (at most nine)
            if py_type(values[i]) is device_class:
                raise Error("ValueError: " + name + " was given device='cpu' but argument " + String(i + 1)
                    + " is a TrainDeviceArray, which lives on the GPU; copy it to the host with to_host(), "
                    + "or leave device=None")
    return device


def training_entries_cpu(mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """The host training image's entries for an explicit "cpu" beside a GPU
    set: identical mode only (the estimators' `device_cpu_fast` words for any
    other), connected once at the first such call (the bridge registered the
    load with `cpu_load_later`; `device_cpu_connect_late` runs it), refused in
    the words recorded at that load where the image is missing. It never
    returns the device entries in its place."""
    var b = native_loaded_module("builtins")
    var nk = optimizer_keys()
    var word = mode
    if word is None:
        word = py_item(py_item(context, nk[].k_backend), nk[].k__DEFAULT_MODE)
    if py_type(word) is not b.str or String(py=py_call(word.strip).lower()) != "identical":
        device_cpu_fast()
    var cells = py_get(context, PythonObject(_TRAINING_CPU_ENTRIES))
    if cells is None:
        _ = device_cpu_connect_late(context)
        cells = py_get(context, PythonObject(_TRAINING_CPU_ENTRIES))
    if cells is None:
        raise Error("ImportError: device='cpu' needs the CPU identical training image beside the GPU set, "
            + "which is not connected in this installation")
    if py_type(cells) is b.str:
        raise Error(String(cells))
    return cells
