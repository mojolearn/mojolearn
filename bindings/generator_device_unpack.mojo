# SPDX-License-Identifier: Apache-2.0
"""`device=` of `mojolearn.training.Generator` (lane vdf-sched, 2026-11-24).

A Generator's state is its seed and its next-stream counter; every draw is a
pure function of (seed, stream, element index), so both sides give the same
bits. Its draws run on the training image's primitive operation 6
(bindings/generator_draw_native_door.mojo): on a GPU installation the device
image's `neural_rng_kernel` (core/philox_neural.mojo `neural_rng_host`, one
thread per element), on the host image `neural_rng_host_into`.

The word is admitted ONCE, at construction, through the shared UNPACK
(providers/python/function_device_boundary_door.mojo `function_device_admit`,
core/function_device.mojo's words). None, "gpu", and "cpu" on a CPU-only
installation keep today's route (the installation's training image). An
explicit "cpu" beside a GPU set runs the host training image
`_mojolearn_training_host`, identical mode only (the estimators'
`device_cpu_fast` words otherwise), loaded at the first such construction
(python/mojolearn/_native_samba_bridge.py registers the load with
`cpu_load_later`; `device_cpu_connect_late` runs it) and refused in the words
recorded at that load where the image is missing; it never runs the device
image in its place. The construction stores the public word (`device`) and,
for that one case, `_host_device = True`; the draw reads that one field to
pick the table key (vendor 0) and nothing else.
"""
from std.python import PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_type
from core.device_route import device_cpu_fast
from providers.python.device_route_boundary_door import device_cpu_connect_late
from providers.python.function_device_boundary_door import function_device_admit

comptime GENERATOR_HAS_GPU = True
comptime GENERATOR_HAS_CPU = True
#: the samba context's record of the host training image load (True, or the refusal text)
comptime _GENERATOR_HOST = "generator_host"


def generator_mode_identical(mode: PythonObject, context: PythonObject) raises -> Bool:
    """True when the Generator's numeric mode (its own, else the library
    default) is identical, by the spelling rule of the draw route."""
    var word = mode
    if not Bool(py=word):
        word = context["generator_backend"]._DEFAULT_MODE
    var b = native_loaded_module("builtins")
    if py_type(word) is not b.str:
        return False
    return String(py=py_call(word.strip).lower()) == "identical"


def generator_device_admit(device: PythonObject, mode: PythonObject, context: PythonObject) raises -> Bool:
    """UNPACK of a Generator's explicit (non-None) `device`: True for an
    explicit "cpu" beside a GPU set, after the host training image is
    connected (once per process). Refused in the shared words for an unknown
    word, "gpu" on a CPU-only installation, and "cpu" with a fast mode."""
    var fdev = function_device_admit(device, "mojolearn.training.Generator", GENERATOR_HAS_GPU,
                                     GENERATOR_HAS_CPU, context, native_loaded_module("builtins"))
    if not fdev.host_beside:
        return False
    if not generator_mode_identical(mode, context):
        device_cpu_fast()
    var loaded = py_get(context, _GENERATOR_HOST)
    if loaded is None:
        _ = device_cpu_connect_late(context)
        loaded = py_get(context, _GENERATOR_HOST)
    if loaded is None:
        raise Error("ImportError: device='cpu' needs the CPU identical training image beside the GPU set, "
            + "which is not connected in this installation")
    if py_type(loaded) is native_loaded_module("builtins").str:
        raise Error(String(loaded))
    return True
