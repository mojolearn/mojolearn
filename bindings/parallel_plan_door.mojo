# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Shared admission for the native multi-GPU driver doors
(lane/native-parallel-driver-20261024,
plan NATIVE_PARALLEL_DRIVER_20261024 (git history)).

A cooperative driver used to pickle its estimator to ONE worker process
whose visibility mask held the selected devices. Its door now validates the
devices exactly as `_parallel_pool.DevicePool` did, refuses the routes the
pool refused, binds the devices to the calling thread in the family's GPU
binding (`device_plan_enter`, bindings/device_plan_binding.mojo) and runs
the plain owner, releasing the plan whatever happens (`plan_exit` in the
door's `finally`). No Python frame runs: every call below is a compiled
entry or a builtin.
"""
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_fields
from bindings.family_common_pack import family_call
from core.device_plan import DEVICE_PLAN_WORKER, DEVICE_PLAN_CHOLESKY
from bindings.parallel_plan_run import (
    PLAN_DEVICES_REFUSAL, plan_indices_admit, plan_vendor_admit, plan_vendor_is_gpu, plan_mode_admitted,
)


def plan_indices(devices: PythonObject, b: PythonObject) raises -> List[Int]:
    """UNPACK: an exact tuple of exact ints as typed indices (a value that
    is not an exact int is refused in the devices words). `b` is the
    builtins module the caller already holds."""
    var indices = List[Int](capacity=len(devices))
    for i in range(len(devices)):  # driver-loop: over the selected device indices
        if py_type(devices[i]) is not b.int:
            raise Error(PLAN_DEVICES_REFUSAL)
        indices.append(Int(py=devices[i]))
    return indices^


def plan_devices(value: PythonObject, operation: String, cooperative: Bool,
                 context: PythonObject) raises -> PythonObject:
    """`devices` as an exact tuple of distinct nonnegative ints, admitted
    for the installed route with the old pool's refusals and wording
    (`plan_indices_admit`, `plan_vendor_admit`: typed, in
    bindings/parallel_plan_run.mojo). `cooperative`: the binding shards
    inside one call (refused on a CPU-only install); otherwise the door
    shards and a CPU-only install runs the host binding per shard."""
    var b = native_loaded_module("builtins")
    if py_type(value) is not b.tuple and py_type(value) is not b.list and py_type(value) is not b.range:
        raise Error("TypeError: devices requires a native tuple, list or range; Python iteration hooks are unsupported")
    var devices = py_call(b.tuple, value)
    if len(devices) == 0:
        raise Error(PLAN_DEVICES_REFUSAL)
    var indices = plan_indices(devices, b)
    plan_indices_admit(indices)
    plan_vendor_admit(indices, String(py_get(context, PythonObject("vendor"), PythonObject("cpu"))), operation, cooperative)
    return devices


def plan_enter(binding: PythonObject, devices: PythonObject, rebuild: String,
               families: Int = DEVICE_PLAN_WORKER) raises:
    """Bind `devices` to the calling thread in `binding`'s DSO for the owner
    families the old worker enabled (`families`; the default is its
    environment, every family but Cholesky). A binding built before the plan
    entries is refused by name (`rebuild`, the old worker's capability
    refusal)."""
    var b = native_loaded_module("builtins")
    var enter = py_call(b.getattr, binding, PythonObject("device_plan_enter"), PythonObject(None))
    var leave = py_call(b.getattr, binding, PythonObject("device_plan_exit"), PythonObject(None))
    if enter is None or leave is None:
        raise Error("ImportError: " + rebuild)
    _ = family_call(enter, devices, PythonObject(families))


def plan_exit(binding: PythonObject) raises:
    """Release the calling thread's plan in `binding`'s DSO."""
    var b = native_loaded_module("builtins")
    _ = family_call(py_call(b.getattr, binding, PythonObject("device_plan_exit")))


def plan_field(model: PythonObject, name: String) raises -> PythonObject:
    """`model.__dict__.get(name)`, read without class or instance hooks (no
    Python frame runs)."""
    return py_get(py_fields(model), PythonObject(name), PythonObject(None))


def plan_require_identical(model: PythonObject, message: String, family: Int) raises:
    """The drivers' `numeric_mode not in (None, 'identical')` refusal. The
    word `fast` passes where the identical set answers it for the driver's
    door `family` (core/mode_answer.mojo; lane fx-fast-mask): the driver then
    runs the identical kernels, as the estimator's own fit does. Once per
    driver call, no loop; an `identical` word pays nothing more."""
    var mode = plan_field(model, "numeric_mode")
    if mode is None:
        return
    var b = native_loaded_module("builtins")
    if py_type(mode) is not b.str:
        raise Error("ValueError: " + message)
    if not plan_mode_admitted(String(mode), family):
        raise Error("ValueError: " + message)


def plan_enter_all(bindings: PythonObject, devices: PythonObject, rebuild: String,
                   families: Int = DEVICE_PLAN_WORKER) raises:
    """`plan_enter` in every binding of `bindings` (a tuple of binding
    modules, each loaded once in this process). The old worker's environment
    reached every native module of its process; a plan is per DSO, so a
    driver binds it in each binding its plain owner reaches. A failure
    releases the plans already bound before it raises."""
    var bound = 0
    try:
        for i in range(len(bindings)):  # driver-loop: over the driver's bindings
            plan_enter(bindings[i], devices, rebuild, families)
            bound += 1
    except error:
        for i in range(bound):  # driver-loop: over the bindings already bound
            try:
                plan_exit(bindings[i])
            except:
                pass
        raise error


def plan_exit_all(bindings: PythonObject) raises:
    """Release the plan in every binding of `bindings`; the first failure is
    raised after every release was attempted."""
    var failure = String()
    for i in range(len(bindings)):  # driver-loop: over the driver's bindings
        try:
            plan_exit(bindings[i])
        except error:
            if failure == "":
                failure = String(error)
    if failure != "":
        raise Error(failure)

