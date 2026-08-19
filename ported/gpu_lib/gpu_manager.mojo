"""The host-side facade every caller goes through.

PORT OF `catboost/cuda/cuda_lib/cuda_manager.h` at CatBoost `54a8143a`.
Transliterated where it transliterates. See the DEVIATION BLOCK.

`TCudaManager` is what `greedy_search_helper.cpp` actually talks to:
`NCudaLib::GetCudaManager().DefaultStream().Synchronize()`
(`split_properties_helper.cpp:961`), `GetDeviceCount()`, `WaitComplete()`.
Everything below it is worker and queue.

The shape worth keeping is that the manager owns the ONE drain. In their
tree, `WaitComplete()` and `DefaultStream().Synchronize()` are the only ways
to make the host wait, and they are named so you notice writing one.

================================ DEVIATION BLOCK ======================
Single device. `get_device_count()` returns 1 and the device-list arguments
that thread through their signatures are dropped rather than carried empty,
because a parameter that is always the same value hides which call sites
would have to change if it were not.

`TProfileMode` and the profiler live in their manager too; ours holds the
counters directly on the worker, so `sync_count` is readable without turning
profiling on. Their profiler is NOT ported. See NOT_PORTED.md.
======================================================================
"""

from max.gpu.host import DeviceContext

from .gpu_base import DEFAULT_STREAM, TCudaStream
from .gpu_single_worker import TGpuOneDeviceWorker
from .task import ECommandType, TCommand


struct TCudaManager(Movable):
    """Their `TCudaManager` (`cuda_manager.h:56`)."""

    var worker: TGpuOneDeviceWorker

    def __init__(out self, var ctx: DeviceContext, sync_budget: Int = -1):
        self.worker = TGpuOneDeviceWorker(ctx^, sync_budget)

    def get_device_count(self) -> Int:
        """Their `GetDeviceCount()`. One, here."""
        return 1

    def default_stream(self) -> TCudaStream:
        """Their `DefaultStream()` (`cuda_manager.h:338`)."""
        return DEFAULT_STREAM

    def request_stream(mut self) -> TCudaStream:
        """Their `RequestStream()`."""
        return self.worker.request_stream()

    def free_stream(mut self, stream: TCudaStream):
        self.worker.free_stream(stream)

    def stream_kernel(mut self, stream: TCudaStream = DEFAULT_STREAM):
        """Record an async launch. The caller launches on `self.worker.ctx`.
        """
        self.worker.stream_kernel(stream)

    def wait_complete(mut self) raises:
        """Their `WaitComplete()` (`cuda_manager.h:239`). The only drain."""
        self.worker.wait_complete()

    def sync_count(self) -> Int:
        return self.worker.sync_count

    def launch_count(self) -> Int:
        return self.worker.launch_count

    def reset_counters(mut self):
        self.worker.reset_counters()
