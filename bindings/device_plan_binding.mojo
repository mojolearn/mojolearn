# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The two entries every GPU binding with a multi-GPU owner loop registers
as `device_plan_enter` / `device_plan_exit` (lane/native-parallel-driver-
20261024, docs/plans/NATIVE_PARALLEL_DRIVER_20261024.md).

A native driver door (bindings/parallel_plan_door.mojo) calls them around
the plain owner: the plan lives in the binding's own DSO, where its owner
loops read it (core/device_plan.mojo), bound to the calling thread."""
from std.python import PythonObject

from core.device_plan import device_plan_enter, device_plan_exit


def device_plan_enter_binding(devices: PythonObject, families: PythonObject) raises -> PythonObject:
    """Bind `devices` (distinct nonnegative device indices; rank r runs on
    `devices[r]`, the root on `devices[0]`) to the calling thread until
    `device_plan_exit`, for the owner families in `families`
    (core/device_plan.mojo DEVICE_PLAN_* bits). Returns the number of
    devices."""
    var ids = List[Int]()
    for i in range(len(devices)):  # driver-loop: over the selected device indices
        ids.append(Int(py=devices[i]))
    device_plan_enter(ids, Int(py=families))
    return PythonObject(len(ids))


def device_plan_exit_binding() raises -> PythonObject:
    """Release the calling thread's device plan."""
    device_plan_exit()
    return PythonObject(None)
