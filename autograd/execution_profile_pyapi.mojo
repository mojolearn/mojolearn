# SPDX-License-Identifier: Apache-2.0
# ABI serialization only: every counter originates in a native executor.
from std.python import Python, PythonObject
from sequence.execution_profile import ExecutionProfile


def execution_profile_object(p: ExecutionProfile) raises -> PythonObject:
    var out = Python.import_module("builtins").dict()
    out["format"] = "mojolearn-execution-profile/1"
    out["device"] = p.device
    out["complete"] = p.opaque_calls == 0
    out["opaque_external_calls"] = p.opaque_calls
    out["timing"] = "inclusive host call duration; kernel enqueue is not device compute time"
    out["memory_scope"] = "executor-held buffers, workspaces, retired buffers and pinned staging leases; excludes free pool cache and driver allocations"
    out["allocations"] = p.allocations
    out["allocated_bytes"] = p.allocated_bytes
    out["frees"] = p.frees
    out["freed_bytes"] = p.freed_bytes
    out["live_bytes"] = p.live_bytes
    out["retained_bytes"] = p.retained_bytes
    out["peak_retained_bytes"] = p.peak_retained_bytes
    out["retired_bytes"] = p.retired_bytes
    out["staging_bytes"] = p.staging_bytes
    out["peak_staging_bytes"] = p.peak_staging_bytes
    out["staging_acquires"] = p.staging_acquires
    out["staging_acquired_bytes"] = p.staging_acquired_bytes
    out["upload_calls"] = p.upload_calls
    out["upload_bytes"] = p.upload_bytes
    out["download_calls"] = p.download_calls
    out["download_bytes"] = p.download_bytes
    out["device_copy_calls"] = p.device_copy_calls
    out["device_copy_bytes"] = p.device_copy_bytes
    out["host_copy_calls"] = p.host_copy_calls
    out["host_copy_bytes"] = p.host_copy_bytes
    out["upload_host_ns"] = p.upload_host_ns
    out["download_host_ns"] = p.download_host_ns
    out["device_copy_host_ns"] = p.device_copy_host_ns
    out["sync_calls"] = p.sync_calls
    out["sync_ns"] = p.sync_ns
    out["kernel_enqueues"] = p.kernels
    out["fill_calls"] = p.fill_calls
    out["fill_bytes"] = p.fill_bytes
    var rows = Python.list()
    for i in range(len(p.ops)):  # driver-loop: over native operation profile rows
        var row = Python.import_module("builtins").dict()
        row["code"] = p.ops[i]
        row["calls"] = p.calls[i]
        row["elements"] = p.elements[i]
        row["host_call_ns"] = p.enqueue_ns[i]
        rows.append(row)
    out["operations"] = rows
    return out
