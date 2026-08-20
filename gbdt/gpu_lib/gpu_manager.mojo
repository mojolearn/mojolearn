"""The host-side facade every caller goes through.

PORT OF `catboost/cuda/cuda_lib/cuda_manager.h` and the parts of
`cuda_manager.cpp` that back it, at CatBoost `54a8143a`. Transliterated where
it transliterates. See the DEVIATION BLOCK.

`TCudaManager` is what `greedy_search_helper.cpp` actually talks to:
`NCudaLib::GetCudaManager().DefaultStream().Synchronize()`
(`split_properties_helper.cpp:961`), `GetDeviceCount()`, `WaitComplete()`,
`GetProfiler()`. Everything below it is worker and queue.

The shape worth keeping is that the manager owns the ONE drain. In their
tree, `WaitComplete()` (`cuda_manager.h:239`) and `Barrier()`
(`cuda_manager.h:235`) are the only ways to make the host wait, and they are
named so you notice writing one.

================================ DEVIATION BLOCK ======================
**Single device.** `get_device_count()` returns 1 and the device-list
arguments that thread through their signatures are dropped rather than
carried empty, because a parameter that is always the same value hides which
call sites would have to change if it were not.

**One stream numbering, not two.** Theirs keeps `TVector<TDistributedObject
<ui32>> Streams` (`cuda_manager.h:124`) mapping a manager-level stream id to a
per-device stream id, because device 0's stream 3 need not be device 1's
stream 3. With one device the two numberings are equal, so this port has
exactly one: the worker's index into its own `streams`
(`gpu_single_worker.cpp:88`). `request_stream` and `free_stream` therefore go
straight to the worker.

**Their `TComputationStream` RAII is not reproduced.** Theirs returns a
move-only handle whose destructor puts the id back on `FreeStreams`
(`cuda_manager.h:310-314`). Mojo destructors cannot reach an owner, so
`free_stream` is a call. `TGpuOneDeviceWorker.stop` refuses to stop while a
stream is still out, which is where a missed call is caught.

**No child managers.** `StartChild`/`StopChild`/`IsLocked`
(`cuda_manager.h:231-251`, `cuda_manager.cpp:45-82`) exist so several host
threads can each drive a subset of the devices. One device, one thread. The
profiler's `add`, which is the only piece of that machinery with a
single-device meaning, IS ported.
======================================================================
"""

from max.gpu.host import DeviceContext

from .device_id import TDeviceId
from .gpu_base import DEFAULT_STREAM, TCudaStream
from .gpu_profiler import EProfileMode, TCudaProfiler
from .gpu_single_worker import TGpuOneDeviceWorker


struct TCudaManager(Movable):
    """Their `TCudaManager` (`cuda_manager.h:107`)."""

    var worker: TGpuOneDeviceWorker
    var profiler: TCudaProfiler
    var has_profiler: Bool

    def __init__(out self, var ctx: DeviceContext, sync_budget: Int = -1):
        """Their `Start` (`cuda_manager.cpp:200-211`), which requests the
        devices and then calls `CreateProfiler`.

        `CreateProfiler` builds it as
        `TCudaProfiler(EProfileMode::LabelAsync, 0, false)`
        (`cuda_manager.cpp:12-14`): host timing only, every level, and no
        printout unless somebody asks.
        """
        self.profiler = TCudaProfiler(
            EProfileMode.LabelAsync, 0, False, Optional(ctx.copy())
        )
        self.has_profiler = True
        self.worker = TGpuOneDeviceWorker(ctx^, sync_budget)

    def get_device_count(self) -> Int:
        """Their `GetDeviceCount()` (`cuda_manager.h:214-216`). One, here."""
        return 1

    def has_devices(self) -> Bool:
        """Their `HasDevices()` (`cuda_manager.h:258-260`)."""
        return True

    def get_device_id(self, dev: Int) raises -> TDeviceId:
        """Their `GetDeviceId(dev)` (`cuda_manager.h:350-352`)."""
        if dev != 0:
            raise Error(
                String("Illegal device id #") + String(dev) + ": one device"
            )
        return TDeviceId(0, 0)

    def default_stream(self) -> TCudaStream:
        """Their `DefaultStream()` (`cuda_manager.h:354-356`), which is
        stream id 0."""
        return DEFAULT_STREAM

    def request_stream(mut self) -> TCudaStream:
        """Their `RequestStream()` (`cuda_manager.cpp:228-245`)."""
        return self.worker.request_stream()

    def free_stream(mut self, stream: TCudaStream) raises:
        """Their `~TComputationStream` plus `FreeStream`
        (`cuda_manager.h:310-314`, `cuda_manager.cpp:110-121`)."""
        self.worker.free_stream(stream)

    def stream_kernel(mut self, stream: TCudaStream = DEFAULT_STREAM) raises:
        """Their `LaunchKernel` (`cuda_manager.h:193-201`) as far as the
        control plane is concerned: the command reaches the worker's
        StreamKernel case. The caller launches on `self.worker.ctx`.
        """
        self.worker.stream_kernel(stream)

    def wait_complete(mut self) raises:
        """Their `WaitComplete()` (`cuda_manager.h:239-241`), which is
        `WaitComplete(GetActiveDevices())` (`cuda_manager.cpp:183-196`): one
        `TCudaSingleDevice::WaitComplete()` per active device, then wait on
        each future.

        It drains every ACTIVE stream and nothing else
        (`gpu_single_worker.h:194-200`), so calling it twice with no launch in
        between costs one drain, not two.
        """
        self.worker.wait_complete()

    def barrier(mut self) raises:
        """Their `Barrier()` (`cuda_manager.h:235-237`), which is
        `DefaultStream().Synchronize()` and reaches
        `TCudaSingleDevice::StreamSynchronize` per active device
        (`cuda_manager.h:305-310`).

        Their searcher calls this behind `if (!IsOnlyDefaultStream())`
        (`split_properties_helper.cpp:1143`, `:1257`), a condition this port
        can never satisfy; see the DEVIATION BLOCK in `gpu_base.mojo`. It is
        ported anyway because `MakeSplit` calls the same drain unguarded
        (`split_properties_helper.cpp:961`).
        """
        self.stream_synchronize(DEFAULT_STREAM)

    def stream_synchronize(mut self, stream: TCudaStream) raises:
        """Their `TComputationStream::Synchronize()`
        (`cuda_manager.h:305-310`), which is
        `Devices[dev]->StreamSynchronize(At(dev))` for each active device.
        This is what `NCudaLib::GetCudaManager().DefaultStream().Synchronize()`
        runs at `split_properties_helper.cpp:961`, once per split.
        """
        self.worker.stream_synchronize(stream)

    def get_profiler(self) -> TCudaProfiler:
        """Their `GetProfiler()` (`cuda_manager.h:253-256`).

        Returns the handle, not a copy of the statistics; see DEVIATION 2 in
        `gpu_profiler.mojo`. Use it as

            with mgr.get_profiler().profile("Compute histograms"):
                ...
        """
        return self.profiler.copy()

    def reset_profiler(mut self, print_info: Bool) raises:
        """Their `ResetProfiler` (`cuda_manager.cpp:22-30`)."""
        if self.has_profiler:
            if print_info:
                self.profiler.print_info()
            self.has_profiler = False

    def stop(mut self) raises:
        """Their `Stop()` (`cuda_manager.cpp:213-226`).

        Theirs is `FreeComputationStreams(); WaitComplete(); FreeDevices();
        ResetProfiler(true)` (`cuda_manager.cpp:211-224`).
        `FreeComputationStreams` (`cuda_manager.cpp:159-166`) enqueues a
        FreeStream command per stream and asserts every one was already handed
        back; ours ran those commands synchronously at each `free_stream`, so
        what is left of that step is the assertion, and
        `TGpuOneDeviceWorker._finish_run` is where it lives
        (`gpu_single_worker.cpp:171-179`). `FreeDevices` has no counterpart:
        the `DeviceContext` belongs to the caller.

        `worker.stop()` is their StopWorker case plus that post-loop, so the
        drain their `WaitComplete()` does is inside it and is not repeated
        here (`gpu_single_worker.cpp:134-138`).
        """
        self.worker.stop()
        self.reset_profiler(True)

    def sync_count(self) -> Int:
        """OURS, not theirs. See the DEVIATION BLOCK in
        `gpu_single_worker.mojo`."""
        return self.worker.sync_count

    def launch_count(self) -> Int:
        """OURS, not theirs."""
        return self.worker.launch_count

    def reset_counters(mut self):
        """OURS, not theirs."""
        self.worker.reset_counters()
