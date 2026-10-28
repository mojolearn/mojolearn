# SPDX-License-Identifier: Apache-2.0
# ABI serialization only: every counter originates in a native executor.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_setitem
from sequence.execution_profile import ExecutionProfile


def execution_profile_object(p: ExecutionProfile) raises -> PythonObject:
    var out = Python.import_module("builtins").dict()
    py_setitem(out, "format", "mojolearn-execution-profile/1")
    py_setitem(out, "device", p.device)
    py_setitem(out, "complete", p.opaque_calls == 0)
    py_setitem(out, "opaque_external_calls", p.opaque_calls)
    py_setitem(out, "timing", "inclusive host call duration; kernel enqueue is not device compute time")
    py_setitem(out, "memory_scope", "executor-held buffers, workspaces, retired buffers and pinned staging leases; excludes free pool cache and driver allocations")
    py_setitem(out, "allocations", p.allocations)
    py_setitem(out, "allocated_bytes", p.allocated_bytes)
    py_setitem(out, "frees", p.frees)
    py_setitem(out, "freed_bytes", p.freed_bytes)
    py_setitem(out, "live_bytes", p.live_bytes)
    py_setitem(out, "retained_bytes", p.retained_bytes)
    py_setitem(out, "peak_retained_bytes", p.peak_retained_bytes)
    py_setitem(out, "retired_bytes", p.retired_bytes)
    py_setitem(out, "staging_bytes", p.staging_bytes)
    py_setitem(out, "peak_staging_bytes", p.peak_staging_bytes)
    py_setitem(out, "staging_acquires", p.staging_acquires)
    py_setitem(out, "staging_acquired_bytes", p.staging_acquired_bytes)
    py_setitem(out, "upload_calls", p.upload_calls)
    py_setitem(out, "upload_bytes", p.upload_bytes)
    py_setitem(out, "download_calls", p.download_calls)
    py_setitem(out, "download_bytes", p.download_bytes)
    py_setitem(out, "device_copy_calls", p.device_copy_calls)
    py_setitem(out, "device_copy_bytes", p.device_copy_bytes)
    py_setitem(out, "host_copy_calls", p.host_copy_calls)
    py_setitem(out, "host_copy_bytes", p.host_copy_bytes)
    py_setitem(out, "upload_host_ns", p.upload_host_ns)
    py_setitem(out, "download_host_ns", p.download_host_ns)
    py_setitem(out, "device_copy_host_ns", p.device_copy_host_ns)
    py_setitem(out, "sync_calls", p.sync_calls)
    py_setitem(out, "sync_ns", p.sync_ns)
    py_setitem(out, "kernel_enqueues", p.kernels)
    py_setitem(out, "fill_calls", p.fill_calls)
    py_setitem(out, "fill_bytes", p.fill_bytes)
    var rows = Python.list()
    for i in range(len(p.ops)):  # driver-loop: over native operation profile rows
        var row = Python.import_module("builtins").dict()
        py_setitem(row, "code", p.ops[i])
        py_setitem(row, "calls", p.calls[i])
        py_setitem(row, "elements", p.elements[i])
        py_setitem(row, "host_call_ns", p.enqueue_ns[i])
        _ = py_call(rows.append, row)
    py_setitem(out, "operations", rows)
    return out
