# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host-only half of the control plane, asserted against their source.

No `DeviceContext` is touched here; everything below is value types.
`checks/gpu_lib_worker_check.mojo` is the half that needs a device.

This file used to only PRINT. A probe that cannot fail is a probe that has
never checked anything, so every line now carries an expected value and the
CatBoost line that fixes it.
"""

from gbdt.gpu_lib.fwd import EPtrType, is_host_ptr
from gbdt.gpu_lib.gpu_base import TCudaStream, TCudaStreamsProvider, DEFAULT_STREAM
from gbdt.gpu_lib.gpu_profiler import EProfileMode, TCudaProfiler
from gbdt.gpu_lib.mapping import (
    DEVICE_COUNT,
    TMirrorMapping,
    TSingleMapping,
    TSingleMappingBuilder,
    TStripeMapping,
    TStripeMappingBuilder,
)
from gbdt.gpu_lib.slice import TSlice
from gbdt.gpu_lib.device_id import TDeviceId
from gbdt.gpu_lib.task import (
    ECommandType,
    ECpuFuncType,
    TCommand,
    is_blocking_host_task,
    stop_worker_command,
    wait_submit_command,
)
from gbdt.gpu_lib.tasks_queue.single_host_task_queue import TSingleHostTaskQueue


def expect(what: String, got: Int, want: Int) raises:
    if got != want:
        raise Error(
            String("FAIL ") + what + ": got " + String(got) + ", want "
            + String(want)
        )
    print("  ok  ", what, "=", got)


def expect_bool(what: String, got: Bool, want: Bool) raises:
    if got != want:
        raise Error(
            String("FAIL ") + what + ": got " + String(got) + ", want "
            + String(want)
        )
    print("  ok  ", what, "=", got)


def main() raises:
    # fwd.h / cuda_base.h:115 -- everything that is not device memory is host
    # memory, which is what makes TCudaHostBufferPtr dereferenceable.
    print("EPtrType (cuda_base.h:115)")
    expect_bool("CudaHost is host", is_host_ptr(EPtrType.CudaHost), True)
    expect_bool("Host is host", is_host_ptr(EPtrType.Host), True)
    expect_bool("CudaDevice is not", is_host_ptr(EPtrType.CudaDevice), False)

    # cuda_base.h:75-83 -- RequestStream recycles from the free list before it
    # creates, which is why a per-level request/drop loop does not leak
    # cudaStream_t.
    print("TCudaStreamsProvider (cuda_base.h:75-83, :50-54)")
    var p = TCudaStreamsProvider()
    var s1 = p.request_stream()
    var s2 = p.request_stream()
    expect("first stream", Int(s1.get_stream()), 1)
    expect("second stream", Int(s2.get_stream()), 2)
    p.free_stream(s1)
    expect("recycled, not fresh", Int(p.request_stream().get_stream()), 1)
    p.free_stream(TCudaStream(0))
    expect("the default stream is never freed", len(p.free_list), 0)
    expect_bool("stream 0 is the default", DEFAULT_STREAM.is_default(), True)

    # slice.h
    print("TSlice (slice.h:9-105)")
    expect("Size (slice.h:25-28)", TSlice(3, 9).size(), 6)
    expect("one-object ctor (slice.h:13-17)", TSlice(4).right, 5)
    expect("operator+= (slice.h:30-34)", (TSlice(2, 5) + 3).left, 5)
    expect(
        "operator*= shifts WHOLE slices (slice.h:36-41)",
        (TSlice(2, 5) * 2).left,
        8,
    )
    expect_bool("IsEmpty is >= (slice.h:51-53)", TSlice(7, 7).is_empty(), True)
    expect_bool(
        "two empty slices compare equal (slice.h:103-105)",
        TSlice(7, 7) == TSlice(0, 0),
        True,
    )
    expect(
        "a disjoint Intersection collapses to [0,0) (slice.h:59-67)",
        TSlice.intersection(TSlice(0, 3), TSlice(5, 9)).size(),
        0,
    )
    expect_bool(
        "an empty slice is Contained by anything (slice.h:69-71)",
        TSlice(0, 10).contains(TSlice(7, 7)),
        True,
    )
    var rest = TSlice.remove(TSlice(0, 10), TSlice(3, 6))
    expect("Remove leaves two pieces (slice.h:73-90)", len(rest), 2)
    expect("left piece ends where the cut starts", rest[0].right, 3)
    expect("right piece starts where the cut ends", rest[1].left, 6)

    # device_id.h. Local is HostId == 0, and the defaulted pair is -1, which
    # is UNSET (`device_id.h:10-11`, `single_device.h:284-286`).
    print("TDeviceId (device_id.h)")
    expect_bool("default is NOT local", TDeviceId().is_local(), False)
    expect_bool("TDeviceId(0, 0) is local", TDeviceId(0, 0).is_local(), True)
    expect(
        "hash is (HostId << 32) | DeviceId (device_id.h:60-64)",
        Int(TDeviceId(0, 3).hash_value()),
        3,
    )
    expect_bool(
        "ordering is lexicographic (device_id.h:33-36)",
        TDeviceId(0, 1) < TDeviceId(0, 2),
        True,
    )

    # task.h:133-140 -- DeviceBlocking is the one that costs.
    print("ECommandType / ECpuFuncType (task.h:12-23, :133-140)")
    expect(
        "SerializedCommand is their tenth value (task.h:22)",
        Int(ECommandType.SerializedCommand.value),
        9,
    )
    expect_bool(
        "DeviceBlocking blocks", is_blocking_host_task(ECpuFuncType.DeviceBlocking), True
    )
    expect_bool(
        "DeviceNonblocking does not",
        is_blocking_host_task(ECpuFuncType.DeviceNonblocking),
        False,
    )

    # single_host_task_queue.h:19-24 -- FIFO, and Dequeue on empty is a
    # CB_ENSURE.
    print("TSingleHostTaskQueue (single_host_task_queue.h:15-52)")
    var q = TSingleHostTaskQueue()
    q.add_task(TCommand(ECommandType.StreamKernel, DEFAULT_STREAM))
    q.add_task(wait_submit_command())
    q.add_task(stop_worker_command())
    expect("three queued", q.size(), 3)
    expect(
        "FIFO, first out is the StreamKernel",
        Int(q.dequeue().get_command_type().value),
        Int(ECommandType.StreamKernel.value),
    )
    _ = q.dequeue()
    _ = q.dequeue()
    expect_bool("drained", q.is_empty(), True)
    var dequeue_refused = False
    try:
        _ = q.dequeue()
    except:
        dequeue_refused = True
    expect_bool("Dequeue on empty raises", dequeue_refused, True)

    # mapping.h. One device, so every GetDeviceCount() reads 1 and the three
    # mappings collapse onto one stripe. The arithmetic is still theirs.
    print("mappings (mapping.h:13-392)")
    expect("DEVICE_COUNT (mapping.h:65)", DEVICE_COUNT, 1)
    var stripe = TStripeMapping.split_between_devices(10, 4)
    expect("CountAt (mapping.h:218-220)", stripe.count_at(0), 10)
    expect("MemorySize (mapping.h:35-37)", stripe.memory_size(), 40)
    expect(
        "MemoryOffset (mapping.h:39-41)", stripe.memory_offset(TSlice(2, 5)), 8
    )
    expect(
        "DeviceMemoryOffset (mapping.h:53-61)",
        stripe.device_memory_offset(0, TSlice(2, 5)),
        8,
    )
    expect(
        "ToLocalSlice renumbers from 0 (mapping.h:243-254)",
        stripe.to_local_slice(TSlice(2, 5)).count_at(0),
        3,
    )
    var builder = TStripeMappingBuilder()
    builder.set_size_at(0, 7)
    builder.update_max_size_at(0, 3)
    expect(
        "UpdateMaxSizeAt is a max, not a set (mapping.h:323-326)",
        builder.build(2).count_at(0),
        7,
    )
    expect(
        "TMirrorMapping::MemoryUsageAt (mapping.h:49-51)",
        TMirrorMapping(5, 3).memory_usage_at(0),
        15,
    )
    var single = TSingleMapping(0, 6, 2)
    expect("TSingleMapping::CountAt here", single.count_at(0), 6)
    expect(
        "TSingleMapping::CountAt elsewhere (mapping.h:118-123)",
        single.count_at(1),
        0,
    )
    expect_bool(
        "DeviceSlice elsewhere is empty (mapping.h:134-139)",
        single.device_slice(1).is_empty(),
        True,
    )
    var single_builder = TSingleMappingBuilder()
    single_builder.set_size_at(0, 4)
    expect(
        "TMappingBuilder<TSingleMapping>::Build (mapping.h:377-388)",
        single_builder.build().count_at(0),
        4,
    )

    # cuda_profiler.h. LabelAsync is their default (`cuda_manager.cpp:13`) and
    # drains nothing, so this stays host-only.
    print("TCudaProfiler (cuda_profiler.h:132-207)")
    var prof = TCudaProfiler(EProfileMode.LabelAsync, 0, False)
    expect_bool(
        "their default mode (cuda_manager.cpp:13)",
        prof.get_default_profile_mode() == EProfileMode.LabelAsync,
        True,
    )
    with prof.profile("Compute histograms"):
        with prof.profile("Leaves split"):
            pass
    prof.print_info()

    print("gpu_lib host check: all assertions hold")
