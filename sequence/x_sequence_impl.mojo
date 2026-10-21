# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed helpers behind the sequence-lane GPU binding (`bindings/_mojolearn_x_sequence.mojo`):
device executor construction for partitioned and distributed programs."""
from max.gpu.host import DeviceContext
from sequence.exec_device import DeviceExec
from sequence.launch_profile import launch_profile_level


def device_executors(ids: List[Int]) raises -> List[DeviceExec]:
    """Open one `DeviceExec` per physical device id, in order."""
    var executors = List[DeviceExec]()
    if len(ids) > 1 and launch_profile_level() != 0:
        raise Error("distributed programs: process-global launch profiling cannot track concurrent GPUs")
    for k in range(len(ids)):  # small-loop(ids: physical devices): independent context ownership
        var context = DeviceContext(device_id=ids[k])
        executors.append(DeviceExec(context))
    return executors^
