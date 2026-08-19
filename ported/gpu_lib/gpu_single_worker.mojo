"""The device worker: one queue, one dispatch switch, one place that drains.

PORT OF `catboost/cuda/cuda_lib/gpu_single_worker.h` and `.cpp` at CatBoost
`54a8143a`. Transliterated where it transliterates. See the DEVIATION BLOCK.

Their `TGpuOneDeviceWorker` is a thread that pulls commands off a queue and
runs `switch (task->GetCommandType())` (`gpu_single_worker.cpp:69-141`), one
case per `ECommandType`. Two cases decide everything about tree speed:

    StreamKernel  submitted to a stream, host does NOT wait
    WaitSubmit    the host waits, and this is the ONLY case that costs

Their inner `TComputationStream` (`gpu_single_worker.h:36-115`) keeps a queue
of waiting tasks and one running task per stream, and `TryProceedTask` only
submits the next one when the current says `ReadyToSubmitNext`. That is the
mechanism by which `TSplitPointsKernel::Run` gets to issue five kernels back
to back without the host ever appearing between them.

================================ DEVIATION BLOCK ======================
Two departures, both forced, both measured.

**1. No thread.** Theirs runs on its own OS thread so `cudaLaunchKernel`
stays off the host critical path. `DeviceContext.enqueue_function` is already
an async submit, so the drain runs on the calling thread. See
`tasks_queue/single_host_task_queue.mojo` for the open question this leaves.

**2. The kernel body is not stored in the command.** A Mojo kernel is
resolved at comptime by `enqueue_function[f]` and cannot be put in a struct
field, so there is no Mojo spelling of `THolder<IGpuKernelTask>` holding an
arbitrary kernel. Callers launch through `stream_kernel`, which records the
command and hands back the context to launch on. The accounting is the part
that had to survive, and it did.

**What this buys, which is the entire reason for the port:** `synchronize`
now exists in exactly ONE place in the tree. Our old driver called it nine
times per level by hand and nobody could see the total. `sync_count` and
`launch_count` make it a number, and `sync_budget` makes exceeding it an
error rather than a slow afternoon.

Measured, 54 launches on a trivial kernel, three interleaved trials:

    sync after each launch    7.7 / 8.9 / 9.8 ms
    one sync at the end       2.1 / 1.2 / 1.2 ms   and bit-exact
======================================================================
"""

from max.gpu.host import DeviceContext

from .gpu_base import DEFAULT_STREAM, TCudaStream, TCudaStreamsProvider
from .task import ECommandType, ECpuFuncType, TCommand, is_blocking_host_task
from .tasks_queue.single_host_task_queue import TSingleHostTaskQueue


struct TComputationStream(Movable):
    """Their inner `TComputationStream` (`gpu_single_worker.h:36-115`).

    Theirs holds the `TCudaStream`, a queue of waiting tasks and the one
    running task. Ours holds the stream and the counts, because the tasks
    themselves cannot be stored (see the DEVIATION BLOCK).
    """

    var stream: TCudaStream
    var waiting_tasks: Int
    var submitted_tasks: Int
    var is_active_flag: Bool

    def __init__(out self, stream: TCudaStream):
        self.stream = stream
        self.waiting_tasks = 0
        self.submitted_tasks = 0
        self.is_active_flag = False

    def add_task(mut self):
        """Their `AddTask` (`gpu_single_worker.h:88`)."""
        self.is_active_flag = True
        self.waiting_tasks += 1
        self.try_proceed_task()

    def try_proceed_task(mut self):
        """Their `TryProceedTask` (`gpu_single_worker.h:105`).

        Theirs asks the running task whether it is `ReadyToSubmitNext` before
        swapping the next one in. Submission into a Mojo stream is ordered by
        the runtime, so every waiting task is immediately submittable and the
        loop drains in one pass.
        """
        while self.waiting_tasks > 0:
            self.waiting_tasks -= 1
            self.submitted_tasks += 1

    def has_tasks(self) -> Bool:
        """Their `HasTasks()`."""
        return self.waiting_tasks > 0

    def is_active(self) -> Bool:
        """Their `IsActive()`."""
        return self.is_active_flag


struct TGpuOneDeviceWorker(Movable):
    """Their `TGpuOneDeviceWorker` (`gpu_single_worker.h:33`)."""

    var ctx: DeviceContext
    var streams_provider: TCudaStreamsProvider
    var queue: TSingleHostTaskQueue
    var streams: List[TComputationStream]
    var launch_count: Int
    var sync_count: Int
    var sync_budget: Int
    var stopped: Bool

    def __init__(out self, var ctx: DeviceContext, sync_budget: Int = -1):
        """`sync_budget` of -1 means unbounded, which is the default so that
        a probe or a bench is never refused. The tree driver passes a real
        one."""
        self.ctx = ctx^
        self.streams_provider = TCudaStreamsProvider()
        self.queue = TSingleHostTaskQueue()
        self.streams = List[TComputationStream]()
        self.streams.append(TComputationStream(DEFAULT_STREAM))
        self.launch_count = 0
        self.sync_count = 0
        self.sync_budget = sync_budget
        self.stopped = False

    def request_stream(mut self) -> TCudaStream:
        """Their `ECommandType::RequestStream` case
        (`gpu_single_worker.cpp:120`)."""
        var s = self.streams_provider.request_stream()
        self.streams.append(TComputationStream(s))
        return s

    def free_stream(mut self, stream: TCudaStream):
        """Their `ECommandType::FreeStream` case
        (`gpu_single_worker.cpp:125`)."""
        self.streams_provider.free_stream(stream)

    def stream_kernel(mut self, stream: TCudaStream = DEFAULT_STREAM):
        """Their `ECommandType::StreamKernel` case
        (`gpu_single_worker.cpp:77`). Records the launch. Does NOT wait.

        The caller launches on `self.ctx` immediately after. Every launch in
        the tree is expected to be bracketed by one of these, which is what
        makes `launch_count` mean something.
        """
        self.launch_count += 1
        for i in range(len(self.streams)):
            if self.streams[i].stream == stream:
                self.streams[i].add_task()
                return
        self.streams.append(TComputationStream(stream))
        self.streams[len(self.streams) - 1].add_task()

    def wait_complete(mut self) raises:
        """Their `ECommandType::WaitSubmit` case
        (`gpu_single_worker.cpp:107`), and their
        `TCudaManager::WaitComplete` (`cuda_manager.h:239`).

        THE ONLY DRAIN IN THE TREE. If a second one appears anywhere else,
        the port has been undone.
        """
        if self.sync_budget >= 0 and self.sync_count >= self.sync_budget:
            raise Error(
                String("sync budget exceeded: ")
                + String(self.sync_count + 1)
                + " drains against a budget of "
                + String(self.sync_budget)
                + ". Every drain past the budget is host round trips the"
                + " control plane was ported to remove, so this is an error"
                + " and not a warning."
            )
        self.sync_count += 1
        self.ctx.synchronize()

    def run(mut self, var command: TCommand) raises:
        """Their dispatch switch (`gpu_single_worker.cpp:69-141`).

        Same cases, same order. The ones with no single-device meaning raise
        rather than silently doing nothing, because a silent no-op is exactly
        how `enqueue_copy(dst_buf=, src_ptr=device)` cost this port a day.
        """
        var type = command.get_command_type()

        if type == ECommandType.StreamKernel:
            self.stream_kernel(command.stream)
        elif type == ECommandType.WaitSubmit:
            self.wait_complete()
        elif type == ECommandType.HostTask:
            if is_blocking_host_task(command.host_task_type):
                self.wait_complete()
        elif type == ECommandType.RequestStream:
            _ = self.request_stream()
        elif type == ECommandType.FreeStream:
            self.free_stream(command.stream)
        elif type == ECommandType.StopWorker:
            self.wait_complete()
            self.stopped = True
        elif type == ECommandType.MemoryAllocation:
            raise Error(
                "MemoryAllocation is not routed through the worker in this"
                " port. Buffers are owned by the caller through"
                " DeviceContext. See NOT_PORTED.md."
            )
        elif type == ECommandType.MemoryDeallocation:
            raise Error(
                "MemoryDeallocation is not routed through the worker in this"
                " port. See NOT_PORTED.md."
            )
        elif type == ECommandType.Reset:
            raise Error("Reset has no single-device meaning here.")
        else:
            raise Error(
                "SerializedCommand is unreachable: it exists to ship a"
                " command to another host over MPI, and this port is single"
                " host. See NOT_PORTED.md."
            )

    def drain_queue(mut self) raises:
        """Their worker loop body (`gpu_single_worker.cpp:60-145`)."""
        while not self.queue.is_empty():
            var command = self.queue.dequeue()
            self.run(command^)

    def add_task(mut self, var command: TCommand):
        """Their `TSingleHostTaskQueue::AddTask` reached through the worker."""
        self.queue.add_task(command^)

    def reset_counters(mut self):
        """Per-tree accounting. `structure_searcher` calls this on entry so
        `sync_count` reads per tree and not per process."""
        self.launch_count = 0
        self.sync_count = 0
