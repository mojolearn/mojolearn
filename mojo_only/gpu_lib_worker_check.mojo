"""Does our worker synchronize where CatBoost synchronizes, and nowhere else.

Every assertion below cites the line of CatBoost that decides it. This file
used to assert our OLD behavior, in which `WaitSubmit` drained the device and
every `WaitComplete` cost a drain whether or not a stream was active. Both
were wrong against `gpu_single_worker.cpp` and both are now asserted the other
way round.

The counted quantity is `sync_count`, which is OURS, not theirs: it is the
number of times `TGpuOneDeviceWorker._device_sync` was reached. Their
equivalent is the number of `cudaStreamSynchronize` calls their worker makes,
and the point of the check is that the two are the same number in the same
places.
"""

from max.gpu.host import DeviceContext

from ported.gpu_lib.device_id import TDeviceId
from ported.gpu_lib.gpu_base import DEFAULT_STREAM
from ported.gpu_lib.gpu_manager import TCudaManager
from ported.gpu_lib.task import (
    ECommandType,
    ECpuFuncType,
    TCommand,
    blocking_device_sync_command,
    host_task_command,
    request_stream_command,
    stop_worker_command,
    wait_submit_command,
)


def head(msg: String) -> String:
    """First line's worth of an error message, without assuming its length.

    `String(e)[byte=0:48]` crashed here with "slice end index 48 is out of
    bounds, valid range is 0 to 38". A fixed slice width is an assertion
    about the message, and the message is the thing under test, so it is the
    one string whose length this file must not assume.
    """
    var n = msg.byte_length()
    if n > 48:
        return msg[byte=0:48] + "..."
    return msg


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
    # ------------------------------------------------------------------
    # device_id.h: local is HostId == 0, and the defaulted TDeviceId is
    # HostId = -1, which is UNSET and therefore not local
    # (`device_id.h:10-11`, `single_device.h:284-286`).
    # ------------------------------------------------------------------
    print("device_id (device_id.h:10-11, single_device.h:284-286)")
    expect_bool("default TDeviceId is not local", TDeviceId().is_local(), False)
    expect_bool("default TDeviceId is remote", TDeviceId().is_remote(), True)
    expect_bool("TDeviceId(0, 0) is local", TDeviceId(0, 0).is_local(), True)
    var remote_refused = False
    try:
        _ = TDeviceId(1, 0)
    except:
        remote_refused = True
    expect_bool(
        "TDeviceId(1, 0) refused (device_id.h:20-22)", remote_refused, True
    )

    var ctx = DeviceContext()
    var mgr = TCudaManager(ctx^, sync_budget=-1)
    expect("one device (cuda_manager.h:214-216)", mgr.get_device_count(), 1)
    expect(
        "default stream is 0 (cuda_manager.h:354-356)",
        Int(mgr.default_stream().get_stream()),
        0,
    )

    var s = mgr.request_stream()
    expect(
        "requested stream indexes Streams[] (gpu_single_worker.cpp:88)",
        Int(s.get_stream()),
        1,
    )
    expect("no drain to request a stream", mgr.sync_count(), 0)

    # ------------------------------------------------------------------
    # gpu_single_worker.cpp:77-93. The StreamKernel case guards ONLY when
    # streamId == 0. A kernel on any other stream is submitted and the host
    # does not wait.
    # ------------------------------------------------------------------
    print("StreamKernel on a non-default stream (gpu_single_worker.cpp:77-93)")
    for _ in range(20):
        mgr.stream_kernel(s)
    expect("20 launches recorded", mgr.launch_count(), 20)
    expect("20 launches cost no drain", mgr.sync_count(), 0)

    # ------------------------------------------------------------------
    # gpu_single_worker.h:285-289 with :194-200. WaitSubmitAndSync drains
    # the ACTIVE streams. One stream is active, so one drain.
    # ------------------------------------------------------------------
    print("WaitComplete (single_device.h:349-351, gpu_single_worker.h:285-289)")
    mgr.wait_complete()
    expect("first WaitComplete drains the one active stream", mgr.sync_count(), 1)

    # ------------------------------------------------------------------
    # gpu_single_worker.h:122-126. Synchronize() clears IsActiveFlag, and
    # SyncActiveStreams (`:194-200`) only visits active streams. So the
    # second WaitComplete drains NOTHING. This is the assertion the old
    # version of this file had backwards.
    # ------------------------------------------------------------------
    mgr.wait_complete()
    expect(
        "second WaitComplete drains nothing (gpu_single_worker.h:122-126)",
        mgr.sync_count(),
        1,
    )

    # ------------------------------------------------------------------
    # gpu_single_worker.cpp:107-109. WaitSubmit is WaitAllTaskToSubmit and
    # nothing else. It is NOT a drain. Their StreamSynchronize pairs it with
    # a TSyncStreamKernel precisely because on its own it does not wait for
    # the device (`single_device.h:343-347`).
    # ------------------------------------------------------------------
    print("WaitSubmit (gpu_single_worker.cpp:107-109)")
    mgr.stream_kernel(s)
    mgr.worker.add_task(wait_submit_command())
    mgr.worker.drain_queue()
    expect("WaitSubmit is not a drain", mgr.sync_count(), 1)

    # ------------------------------------------------------------------
    # gpu_single_worker.cpp:111-118 with task.h:138-140. A host task drains
    # if and only if it is DeviceBlocking.
    # ------------------------------------------------------------------
    print("HostTask (gpu_single_worker.cpp:111-118, task.h:138-140)")
    mgr.worker.add_task(host_task_command(ECpuFuncType.DeviceNonblocking))
    mgr.worker.drain_queue()
    expect("DeviceNonblocking host task is not a drain", mgr.sync_count(), 1)

    mgr.worker.add_task(blocking_device_sync_command())
    mgr.worker.drain_queue()
    expect(
        "DeviceBlocking host task drains the active stream", mgr.sync_count(), 2
    )

    # ------------------------------------------------------------------
    # split_points.cpp:37-162. Their whole split is six calls on ONE stream
    # with nothing between them. On the default stream the guard at
    # gpu_single_worker.cpp:80-87 runs SyncActiveStreams(true), which SKIPS
    # stream 0, so back-to-back default-stream launches cost nothing.
    # ------------------------------------------------------------------
    print("six back-to-back launches on one stream (split_points.cpp:37-162)")
    var before = mgr.sync_count()
    for _ in range(6):
        mgr.stream_kernel()
    expect("no drain between them", mgr.sync_count() - before, 0)

    # ------------------------------------------------------------------
    # gpu_single_worker.cpp:80-87. A kernel on stream 0 IS an ordering point
    # across streams: it drains every OTHER active stream first, and not
    # stream 0.
    # ------------------------------------------------------------------
    print("stream 0 orders the others (gpu_single_worker.cpp:80-87)")
    mgr.stream_kernel(s)
    before = mgr.sync_count()
    mgr.stream_kernel()
    expect("default-stream launch drains the user stream", mgr.sync_count() - before, 1)

    # ------------------------------------------------------------------
    # gpu_single_worker.cpp:125-132. FreeStream waits for submission and
    # then calls SyncStream on each freed id unconditionally -- their
    # Synchronize() (`gpu_single_worker.h:122-126`) does not check
    # IsActiveFlag -- before the id goes back on the free list.
    # ------------------------------------------------------------------
    print("FreeStream (gpu_single_worker.cpp:125-132)")
    before = mgr.sync_count()
    mgr.free_stream(s)
    expect("freeing a stream drains it", mgr.sync_count() - before, 1)

    # ------------------------------------------------------------------
    # single_device.h:342-347. StreamSynchronize is a TSyncStreamKernel on
    # the stream plus a TWaitSubmitCommand, so it launches AND drains. This
    # is what split_properties_helper.cpp:961 calls once per split.
    # ------------------------------------------------------------------
    print("StreamSynchronize (single_device.h:342-347)")
    var launches_before = mgr.launch_count()
    before = mgr.sync_count()
    mgr.barrier()
    expect("Barrier launches TSyncStreamKernel", mgr.launch_count() - launches_before, 1)
    expect("Barrier drains", mgr.sync_count() - before, 1)

    # ------------------------------------------------------------------
    # gpu_single_worker.cpp:120-123. The RequestStream case writes the id
    # into the command. It is readable only through a direct run(), because
    # the queue hands the worker a copy and the promise half of their design
    # (`future/local_promise_future.h`) is not ported.
    # ------------------------------------------------------------------
    print("RequestStream writes its answer (gpu_single_worker.cpp:120-123)")
    var req = request_stream_command()
    _ = mgr.worker.run(req)
    expect_bool("stream id was written back", req.stream_id_result >= 0, True)

    # ------------------------------------------------------------------
    # OURS, not theirs: the sync budget. CatBoost counts nothing and refuses
    # nothing. The budget exists so a drain that should not be there is an
    # error at the call that made it. See the DEVIATION BLOCK in
    # gpu_single_worker.mojo.
    # ------------------------------------------------------------------
    print("sync budget (OURS, not CatBoost's)")
    mgr.reset_counters()
    expect("counters reset", mgr.sync_count(), 0)
    mgr.worker.sync_budget = 0
    mgr.stream_kernel()
    var budget_enforced = False
    try:
        mgr.wait_complete()
    except e:
        budget_enforced = True
        print("       ", head(String(e)))
    expect_bool("a drain past the budget raises", budget_enforced, True)
    mgr.worker.sync_budget = -1

    # ------------------------------------------------------------------
    # gpu_single_worker.cpp:165-179. Their Run() breaks on shouldStop and
    # THEN runs the leak checks. A StopWorker that arrives through the queue
    # must reach them, or the queue is the way to skip them. The stream
    # requested just above was never freed, so the check must fire.
    # ------------------------------------------------------------------
    print("StopWorker reaches the leak checks (gpu_single_worker.cpp:165-179)")
    mgr.worker.add_task(stop_worker_command())
    var leak_caught = False
    try:
        mgr.worker.drain_queue()
    except e:
        leak_caught = True
        print("       ", head(String(e)))
    expect_bool("an unfreed stream is refused at stop", leak_caught, True)
    expect_bool(
        "and the worker is NOT reported stopped (gpu_single_worker.cpp:179)",
        mgr.worker.is_running(),
        True,
    )

    # A command type with no single-device meaning raises rather than
    # silently doing nothing (gpu_single_worker.cpp:139-141).
    var serialized = TCommand(ECommandType.SerializedCommand, DEFAULT_STREAM)
    var refused = False
    try:
        _ = mgr.worker.run(serialized)
    except:
        refused = True
    expect_bool("SerializedCommand is unreachable", refused, True)

    print("gpu_lib worker check: all assertions hold")
