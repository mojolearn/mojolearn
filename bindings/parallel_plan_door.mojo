# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Shared admission for the native multi-GPU driver doors
(lane/native-parallel-driver-20261024,
docs/plans/NATIVE_PARALLEL_DRIVER_20261024.md).

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
from bindings.py_balanced import py_call, py_type, py_get, py_fields
from bindings.family_common import family_call
from core.device_plan import DEVICE_PLAN_WORKER, DEVICE_PLAN_CHOLESKY


def plan_devices(value: PythonObject, operation: String, cooperative: Bool,
                 context: PythonObject) raises -> PythonObject:
    """`devices` as an exact tuple of distinct nonnegative ints, admitted
    for the installed route with the old pool's refusals and wording.
    `cooperative`: the binding shards inside one call (refused on a CPU-only
    install); otherwise the door shards and a CPU-only install runs the
    host binding per shard."""
    var b = Python.import_module("builtins")
    var devices = py_call(b.tuple, value)
    if len(devices) == 0:
        raise Error("ValueError: devices must be distinct nonnegative integer indices")
    for i in range(len(devices)):  # driver-loop: over the selected device indices
        if py_type(devices[i]) is not b.int or Int(py=devices[i]) < 0:
            raise Error("ValueError: devices must be distinct nonnegative integer indices")
        for j in range(i):  # driver-loop: over the earlier selected device indices
            if Int(py=devices[j]) == Int(py=devices[i]):
                raise Error("ValueError: devices must be distinct nonnegative integer indices")
    var vendor = String(py_get(context, PythonObject("vendor"), PythonObject("cpu")))
    if vendor == "cpu" or vendor == "None":
        if cooperative:
            raise Error("NotImplementedError: no CPU implementation of the cooperative multi-GPU driver "
                        + operation + " yet: its shards are device row tiles, chunks or ranges inside "
                        + "the GPU binding, which no host binding restates")
        return devices
    if vendor == "metal" and (len(devices) != 1 or Int(py=devices[0]) != 0):
        raise Error("ValueError: device selection is unavailable for this vendor/device group")
    if vendor != "cuda" and vendor != "hip" and vendor != "metal":
        raise Error("ValueError: device selection is unavailable for this vendor/device group")
    return devices


def plan_on_gpu(context: PythonObject) raises -> Bool:
    """True on a CUDA, HIP or Metal install (a plan is bound); False on a
    CPU-only install (no plan: the host binding runs the owner)."""
    var vendor = String(py_get(context, PythonObject("vendor"), PythonObject("cpu")))
    return vendor == "cuda" or vendor == "hip" or vendor == "metal"


def plan_enter(binding: PythonObject, devices: PythonObject, rebuild: String,
               families: Int = DEVICE_PLAN_WORKER) raises:
    """Bind `devices` to the calling thread in `binding`'s DSO for the owner
    families the old worker enabled (`families`; the default is its
    environment, every family but Cholesky). A binding built before the plan
    entries is refused by name (`rebuild`, the old worker's capability
    refusal)."""
    var b = Python.import_module("builtins")
    var enter = py_call(b.getattr, binding, PythonObject("device_plan_enter"), PythonObject(None))
    var leave = py_call(b.getattr, binding, PythonObject("device_plan_exit"), PythonObject(None))
    if enter is None or leave is None:
        raise Error("ImportError: " + rebuild)
    _ = family_call(enter, devices, PythonObject(families))


def plan_exit(binding: PythonObject) raises:
    """Release the calling thread's plan in `binding`'s DSO."""
    var b = Python.import_module("builtins")
    _ = family_call(py_call(b.getattr, binding, PythonObject("device_plan_exit")))


def plan_field(model: PythonObject, name: String) raises -> PythonObject:
    """`model.__dict__.get(name)`, read without class or instance hooks (no
    Python frame runs)."""
    return py_get(py_fields(model), PythonObject(name), PythonObject(None))


def plan_require_identical(model: PythonObject, message: String) raises:
    """The drivers' `numeric_mode not in (None, 'identical')` refusal."""
    var mode = plan_field(model, "numeric_mode")
    if mode is None:
        return
    var b = Python.import_module("builtins")
    if py_type(mode) is not b.str or String(mode) != "identical":
        raise Error("ValueError: " + message)
