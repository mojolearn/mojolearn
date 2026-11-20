# SPDX-License-Identifier: Apache-2.0
# ABI serialization only: every counter originates in a native executor.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_setitem
from bindings.native_loaded_module_door import native_loaded_module
from sequence.execution_profile import ExecutionProfile
from bindings.autograd_keys_unpack import autograd_keys


def execution_profile_object(p: ExecutionProfile) raises -> PythonObject:
    var out = Python.dict()
    var nk = autograd_keys()
    py_setitem(out, nk[].k_format, "mojolearn-execution-profile/1")
    py_setitem(out, nk[].k_device, p.device)
    py_setitem(out, nk[].k_complete, p.opaque_calls == 0)
    py_setitem(out, nk[].k_opaque_external_calls, p.opaque_calls)
    py_setitem(out, nk[].k_timing, "inclusive host call duration; kernel enqueue is not device compute time")
    py_setitem(out, nk[].k_memory_scope, "executor-held buffers, workspaces, retired buffers and pinned staging leases; excludes free pool cache and driver allocations")
    py_setitem(out, nk[].k_allocations, p.allocations)
    py_setitem(out, nk[].k_allocated_bytes, p.allocated_bytes)
    py_setitem(out, nk[].k_frees, p.frees)
    py_setitem(out, nk[].k_freed_bytes, p.freed_bytes)
    py_setitem(out, nk[].k_live_bytes, p.live_bytes)
    py_setitem(out, nk[].k_retained_bytes, p.retained_bytes)
    py_setitem(out, nk[].k_peak_retained_bytes, p.peak_retained_bytes)
    py_setitem(out, nk[].k_retired_bytes, p.retired_bytes)
    py_setitem(out, nk[].k_staging_bytes, p.staging_bytes)
    py_setitem(out, nk[].k_peak_staging_bytes, p.peak_staging_bytes)
    py_setitem(out, nk[].k_staging_acquires, p.staging_acquires)
    py_setitem(out, nk[].k_staging_acquired_bytes, p.staging_acquired_bytes)
    py_setitem(out, nk[].k_upload_calls, p.upload_calls)
    py_setitem(out, nk[].k_upload_bytes, p.upload_bytes)
    py_setitem(out, nk[].k_download_calls, p.download_calls)
    py_setitem(out, nk[].k_download_bytes, p.download_bytes)
    py_setitem(out, nk[].k_device_copy_calls, p.device_copy_calls)
    py_setitem(out, nk[].k_device_copy_bytes, p.device_copy_bytes)
    py_setitem(out, nk[].k_host_copy_calls, p.host_copy_calls)
    py_setitem(out, nk[].k_host_copy_bytes, p.host_copy_bytes)
    py_setitem(out, nk[].k_upload_host_ns, p.upload_host_ns)
    py_setitem(out, nk[].k_download_host_ns, p.download_host_ns)
    py_setitem(out, nk[].k_device_copy_host_ns, p.device_copy_host_ns)
    py_setitem(out, nk[].k_sync_calls, p.sync_calls)
    py_setitem(out, nk[].k_sync_ns, p.sync_ns)
    py_setitem(out, nk[].k_kernel_enqueues, p.kernels)
    py_setitem(out, nk[].k_fill_calls, p.fill_calls)
    py_setitem(out, nk[].k_fill_bytes, p.fill_bytes)
    var rows = Python.list()
    for i in range(len(p.ops)):  # driver-loop: over native operation profile rows
        var row = Python.dict()
        py_setitem(row, nk[].k_code, p.ops[i])
        py_setitem(row, nk[].k_calls, p.calls[i])
        py_setitem(row, nk[].k_elements, p.elements[i])
        py_setitem(row, nk[].k_host_call_ns, p.enqueue_ns[i])
        _ = py_call(rows.append, row)
    py_setitem(out, nk[].k_operations, rows)
    return out
