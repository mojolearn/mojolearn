# SPDX-License-Identifier: Apache-2.0
"""`device=` of the learning-rate schedules (lane vdf-sched, 2026-11-24):
ConstantLR, WarmupLinearLR, WarmupCosineLR, StepLR, ExponentialLR and
OneCycleLR.

A schedule's rate at step t is a closed form the optimizer reads once per
step as one float32 scalar argument of its own step kernel; the schedule
keeps no buffer. Its `device` names the side of the optimizer it serves:
admitted once at construction (and at a state restore) through the shared
UNPACK (providers/python/function_device_boundary_door.mojo
`function_device_admit`, the words of core/function_device.mojo), stored as
the public word on the instance, and compared once when an optimizer takes
the schedule (`schedule_device_agrees`: at SGD / Adam / AdamW construction
and set_params, SambaStack construction). Nothing is read per step: the
step's rate evaluation (`schedule_run`) is unchanged and never reads it.

The Python shell passes `device` as the LAST item of the constructor's
argument tuple. A tuple without it (a state rebuild of a schedule saved
before `device` existed, `from_config`) leaves the field absent, which reads
as None.
"""
from std.python import PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_fields, py_get, py_setitem, py_type
from providers.python.function_device_boundary_door import function_device_admit

#: a schedule's rate is a host-evaluated closed form handed to either side's
#: optimizer kernel as a scalar: both sides serve it
comptime SCHEDULE_HAS_GPU = True
comptime SCHEDULE_HAS_CPU = True


def schedule_argument_count(kind: Int) -> Int:
    """The constructor arguments of schedule `kind` before `device`
    (bindings/optimizer_schedule_native_door.mojo `schedule_construct`):
    four for the warmup schedules (0-2), three for StepLR / ExponentialLR
    (3-4), seven for OneCycleLR (5)."""
    if kind < 3:
        return 4
    if kind < 5:
        return 3
    return 7


def schedule_device_item(values: PythonObject, kind: Int) raises -> PythonObject:
    """The `device` item of a schedule constructor tuple, or None when the
    tuple has none."""
    var count = schedule_argument_count(kind)
    if len(values) > count:
        return values[count]
    return PythonObject(None)


def schedule_device_admit(owner: PythonObject, device: PythonObject, label: String,
                          context: PythonObject) raises:
    """UNPACK and PACK of a schedule's `device`, once per construction or
    state restore, before the rate block is bound: the shared admission
    (an unknown word, and 'gpu' on a CPU-only installation, are refused in
    the shared words), then the public word on the instance. `label` is the
    schedule's class name. None: one identity compare and one store."""
    var fields = py_fields(owner)
    if device is not None:
        _ = function_device_admit(device, "mojolearn." + label, SCHEDULE_HAS_GPU, SCHEDULE_HAS_CPU,
                                  context, native_loaded_module("builtins"))
    py_setitem(fields, "device", device)


def schedule_device_agrees(schedule: PythonObject, device: PythonObject, owner_name: String) raises:
    """The refusal of an optimizer given a schedule built for the other
    side: both words explicit and different. Called once where an optimizer
    takes a schedule (construction, set_params), after the optimizer's own
    `device` was admitted; never per step. A schedule without a word (None,
    or saved before `device` existed) serves any optimizer, and an optimizer
    with device=None takes the schedule's side as given."""
    if schedule is None or device is None:
        return
    var word = py_get(py_fields(schedule), "device")
    if word is None:
        return
    var mine = String(device)
    var theirs = String(word)
    if mine != theirs:
        var cls = String(py_type(schedule).__name__)
        raise Error("ValueError: " + owner_name + " was given device='" + mine + "' but its lr_schedule "
            + cls + " was constructed with device='" + theirs
            + "'; give both the same device, or leave one of them None")
