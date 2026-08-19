"""The device worker: one queue, one dispatch switch, one place that drains.

PORT OF `catboost/cuda/cuda_lib/gpu_single_worker.h` and `.cpp` at CatBoost
`54a8143a`. Transliterated where it transliterates. See the DEVIATION BLOCK.

Their `TGpuOneDeviceWorker` is a thread that pulls commands off a queue and
runs `switch (task->GetCommandType())` (`gpu_single_worker.cpp:69-141`), one
case per `ECommandType`. Read that switch before changing anything here; four
of its cases are guards, not work:

    StreamKernel  submitted to a stream, host does NOT wait -- EXCEPT on
                  stream 0, which first waits for every other stream
                  (`gpu_single_worker.cpp:80-87`)
    WaitSubmit    waits for tasks to be SUBMITTED. It does NOT drain the
                  device (`gpu_single_worker.cpp:107-109`)
    HostTask      drains only when the task is `DeviceBlocking`
                  (`gpu_single_worker.cpp:114-116`)
    FreeStream    waits for submission and drains each freed stream before
                  the id goes back on the free list
                  (`gpu_single_worker.cpp:127-131`)

The one command that always drains is a `DeviceBlocking` host task, which is
what `TCudaSingleDevice::WaitComplete()` sends (`single_device.h:349-351`).

Their inner `TComputationStream` (`gpu_single_worker.h:35-127`) keeps a queue
of waiting tasks and one running task per stream, and `TryProceedTask` only
submits the next one when the current says `ReadyToSubmitNext`. That is the
mechanism by which `TSplitPointsKernel::Run` gets to issue five kernels back
to back without the host ever appearing between them.

================================ DEVIATION BLOCK ======================
Four departures, all forced.

**1. No thread.** Theirs runs on its own OS thread so `cudaLaunchKernel`
stays off the host critical path. `DeviceContext.enqueue_function` is already
an async submit, so the drain runs on the calling thread. See
`tasks_queue/single_host_task_queue.mojo` for the open question this leaves.

**2. The kernel body is not stored in the command.** A Mojo kernel is
resolved at comptime by `enqueue_function[f]` and cannot be put in a struct
field, so there is no Mojo spelling of `THolder<IGpuKernelTask>` holding an
arbitrary kernel. Callers launch through `stream_kernel`, which runs their
StreamKernel case and hands back the context to launch on.

**3. `RunningTask` is always empty.** Theirs asks a running task
`ReadyToSubmitNext(stream, context)` (`gpu_single_worker.h:57`) and holds the
next task back until it says yes. We hold no task object and Mojo's submit
returns once the work is on the queue, so a task is submitted the moment it
is added and `try_proceed_task` drains the waiting count in one pass.
`has_tasks()` keeps their `running || waiting` spelling so that the day a
task object exists here, only `try_proceed_task` changes.

**4. `ObjectsToFree` is always empty.** Their lazy-delete list
(`gpu_single_worker.h:161`) is fed by the MemoryDeallocation case, which this
port raises on because buffers belong to `DeviceContext`. The list and the
branches that read it (`gpu_single_worker.cpp:83-86`) are transcribed anyway,
because deleting a branch is how a state machine silently inverts.

**Ours that theirs does not have: `sync_budget`.** CatBoost counts nothing
and refuses nothing. `sync_count`, `launch_count` and `sync_budget` are OURS.
They exist because our old driver called `synchronize` nine times per level
by hand and nobody could see the total; the budget turns a slow afternoon
into an error at the call that caused it. Every device drain in this file
goes through `_device_sync`, so the count cannot be evaded.

Measured, 54 launches on a trivial kernel, three interleaved trials:

    sync after each launch    7.7 / 8.9 / 9.8 ms
    one sync at the end       2.1 / 1.2 / 1.2 ms   and bit-exact
======================================================================
"""

from max.gpu.host import DeviceContext

from .gpu_base import DEFAULT_STREAM, TCudaStream, TCudaStreamsProvider
from .task import ECommandType, TCommand, is_blocking_host_task
from .tasks_queue.single_host_task_queue import TSingleHostTaskQueue


struct TComputationStream(Movable):
    """Their inner `TComputationStream` (`gpu_single_worker.h:35-127`).

    Theirs holds the `TCudaStream`, a queue of waiting tasks and the one
    running task. Ours holds the stream and the counts; see DEVIATION 3.
    """

    var stream: TCudaStream
    var waiting_tasks: Int
    var running_tasks: Int
    var submitted_tasks: Int
    var is_active_flag: Bool

    def __init__(out self, stream: TCudaStream):
        """Their constructor (`gpu_single_worker.h:78-82`), which takes its
        stream straight from the provider."""
        self.stream = stream
        self.waiting_tasks = 0
        self.running_tasks = 0
        self.submitted_tasks = 0
        self.is_active_flag = False

    def get_stream(self) -> TCudaStream:
        """Their `GetStream()` (`gpu_single_worker.h:100-102`)."""
        return self.stream

    def add_task(mut self):
        """Their `AddTask` (`gpu_single_worker.h:89-94`)."""
        self.is_active_flag = True
        self.waiting_tasks += 1
        self.try_proceed_task()

    def try_proceed_task(mut self):
        """Their `TryProceedTask` (`gpu_single_worker.h:108-120`).

        Theirs asks the running task whether it is `ReadyToSubmitNext` before
        swapping the next one in. See DEVIATION 3: submission into a Mojo
        stream completes in place, so every waiting task is immediately
        submittable and the loop drains in one pass.
        """
        while self.waiting_tasks > 0:
            self.waiting_tasks -= 1
            self.submitted_tasks += 1

    def has_tasks(self) -> Bool:
        """Their `HasTasks()` (`gpu_single_worker.h:96-98`), which is
        `!RunningTask.IsEmpty() || WaitingTasks.size()`."""
        return self.running_tasks > 0 or self.waiting_tasks > 0

    def is_active(self) -> Bool:
        """Their `IsActive()` (`gpu_single_worker.h:104-106`)."""
        return self.is_active_flag

    def on_synchronized(mut self) raises:
        """The bookkeeping half of their `Synchronize()`
        (`gpu_single_worker.h:122-126`).

        Theirs is `Stream.Synchronize(); IsActiveFlag = false;
        CB_ENSURE(RunningTask.IsEmpty())`. The drain itself is not here
        because only the worker holds the `DeviceContext`, and every drain has
        to be counted in one place; `TGpuOneDeviceWorker.sync_stream` drains
        and then calls this.
        """
        self.is_active_flag = False
        if self.running_tasks > 0:
            raise Error("Some tasks are not completed")

    def check_completed(self) raises:
        """Their destructor's two checks (`gpu_single_worker.h:84-87`)."""
        if self.running_tasks > 0:
            raise Error("Some tasks are not completed")
        if self.waiting_tasks > 0:
            raise Error("Some tasks are waiting for processing")


struct TGpuOneDeviceWorker(Movable):
    """Their `TGpuOneDeviceWorker` (`gpu_single_worker.h:33`)."""

    var ctx: DeviceContext
    var local_device_id: Int32
    var streams_provider: TCudaStreamsProvider
    var queue: TSingleHostTaskQueue
    var streams: List[TComputationStream]
    var free_streams: List[Int32]
    var objects_to_free: List[UInt64]
    var launch_count: Int
    var sync_count: Int
    var sync_budget: Int
    var stopped: Bool

    def __init__(
        out self,
        var ctx: DeviceContext,
        sync_budget: Int = -1,
        local_device_id: Int32 = 0,
    ):
        """`sync_budget` of -1 means unbounded, which is the default so that
        a probe or a bench is never refused. The tree driver passes a real
        one. See the DEVIATION BLOCK: the budget is ours, not theirs.

        `Stopped` starts `true` in their tree (`gpu_single_worker.h:181`) and
        is cleared when the worker thread enters `Run()` (`cpp:155`). We have
        no thread, so the worker is running from construction.
        """
        self.ctx = ctx^
        self.local_device_id = local_device_id
        self.streams_provider = TCudaStreamsProvider()
        self.queue = TSingleHostTaskQueue()
        self.streams = List[TComputationStream]()
        self.free_streams = List[Int32]()
        self.objects_to_free = List[UInt64]()
        self.launch_count = 0
        self.sync_count = 0
        self.sync_budget = sync_budget
        self.stopped = False
        # Their `Run()` creates the default stream before the loop and makes
        # it the thread's default (`gpu_single_worker.cpp:162-163`).
        self.create_new_computation_stream()

    # ---------------------------------------------------------------- drains

    def _device_sync(mut self) raises:
        """The single point where this port waits on the device.

        OURS, not theirs: the budget check. Their `SyncStream` is a bare
        `cudaStreamSynchronize`. Every drain in this file funnels through
        here so `sync_count` cannot be evaded and a caller that exceeds its
        budget hears about it at the call that did it.
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

    def sync_stream(mut self, id: Int32) raises:
        """Their `SyncStream` (`gpu_single_worker.h:227-229`).

        On Metal one queue backs every handle, so the drain is device-wide.
        Over-ordering never produces a wrong answer; see the DEVIATION BLOCK
        in `gpu_base.mojo`.
        """
        var idx = Int(id)
        if idx < 0 or idx >= len(self.streams):
            raise Error(
                String("SyncStream: no stream ") + String(id)
            )
        self._device_sync()
        self.streams[idx].on_synchronized()

    def check_running_tasks(mut self) -> Bool:
        """Their `CheckRunningTasks` (`gpu_single_worker.h:202-213`)."""
        var has_running = False
        for i in range(len(self.streams)):
            if self.streams[i].is_active():
                if self.streams[i].has_tasks():
                    has_running = True
                    self.streams[i].try_proceed_task()
        return has_running

    def wait_all_task_to_submit(mut self):
        """Their `WaitAllTaskToSubmit` (`gpu_single_worker.h:184-192`).

        Theirs spins, yielding every 1000 turns, until no stream still has a
        task the device has not been handed. It does NOT wait for the device.
        There is no spin here because `try_proceed_task` submits everything
        it is given (DEVIATION 3), so the loop settles on the first pass.
        """
        while self.check_running_tasks():
            pass

    def sync_active_streams(mut self, skip_default: Bool = False) raises:
        """Their `SyncActiveStreams` (`gpu_single_worker.h:194-200`).

        Only ACTIVE streams are drained. A stream goes inactive the moment it
        is synchronised (`gpu_single_worker.h:124`), so draining twice in a
        row costs one drain, not two.
        """
        var first = 1 if skip_default else 0
        for i in range(first, len(self.streams)):
            if self.streams[i].is_active():
                self.sync_stream(Int32(i))

    def delete_objects(mut self) raises:
        """Their `DeleteObjects` (`gpu_single_worker.h:215-220`).

        Their list holds `IFreeMemoryTask`s and this calls `Exec()` on each.
        Ours is always empty; see DEVIATION 4.
        """
        if len(self.objects_to_free) > 0:
            raise Error(
                "objects_to_free is non-empty, which cannot happen while"
                " MemoryDeallocation raises. See NOT_PORTED.md."
            )

    def wait_submit_and_sync(mut self, skip_default: Bool = False) raises:
        """Their `WaitSubmitAndSync` (`gpu_single_worker.h:285-289`).

        This is what a `DeviceBlocking` host task runs, and a
        `DeviceBlocking` host task is what `TCudaSingleDevice::WaitComplete()`
        sends (`single_device.h:349-351`).
        """
        self.wait_all_task_to_submit()
        self.sync_active_streams(skip_default)
        self.delete_objects()

    def wait_complete(mut self) raises:
        """Their `TCudaSingleDevice::WaitComplete()` (`single_device.h:349`)
        and their `TCudaManager::WaitComplete()` (`cuda_manager.h:239-241`).

        THE ONLY DRAIN IN THE TREE. If a second one appears anywhere else,
        the port has been undone.
        """
        self.wait_submit_and_sync()

    # --------------------------------------------------------------- streams

    def create_new_computation_stream(mut self):
        """Their `CreateNewComputationStream`
        (`gpu_single_worker.h:291-293`). The `TComputationStream` constructor
        is what takes a stream from the provider
        (`gpu_single_worker.h:79`)."""
        self.streams.append(
            TComputationStream(self.streams_provider.request_stream())
        )

    def request_stream_impl(mut self) -> Int32:
        """Their `RequestStreamImpl` (`gpu_single_worker.h:295-303`).

        The id is an INDEX into `streams`, which is what a command's stream id
        means everywhere in their worker (`gpu_single_worker.cpp:88`,
        `Streams[streamId]`). It is not the provider's handle.
        """
        if len(self.free_streams) == 0:
            self.free_streams.append(Int32(len(self.streams)))
            self.create_new_computation_stream()
        var id = self.free_streams[len(self.free_streams) - 1]
        _ = self.free_streams.pop()
        return id

    def request_stream(mut self) -> TCudaStream:
        """Their `ECommandType::RequestStream` case
        (`gpu_single_worker.cpp:120-123`)."""
        return TCudaStream(self.request_stream_impl())

    def free_stream(mut self, stream: TCudaStream) raises:
        """Their `ECommandType::FreeStream` case
        (`gpu_single_worker.cpp:125-132`).

        Theirs waits for submission, drains each stream, and only then puts
        the id back on the free list. The `TComputationStream` itself stays
        alive at that index; the provider handle it holds is returned when
        the worker stops (`gpu_single_worker.cpp:175`).
        """
        var ids = List[Int32]()
        ids.append(stream.get_stream())
        self.free_streams_impl(ids^)

    def free_streams_impl(mut self, var ids: List[Int32]) raises:
        """The body of their FreeStream case, which frees a LIST
        (`gpu_single_worker.cpp:126-131`)."""
        self.wait_all_task_to_submit()
        for i in range(len(ids)):
            var id = ids[i]
            if id == 0:
                raise Error("Error: can't free the default stream")
            self.sync_stream(id)
            self.free_streams.append(id)

    # ---------------------------------------------------------------- launch

    def stream_kernel(mut self, stream: TCudaStream = DEFAULT_STREAM) raises:
        """Their `ECommandType::StreamKernel` case
        (`gpu_single_worker.cpp:77-93`). Records the launch. Does NOT wait,
        UNLESS the stream is the default one.

        Their guard, transcribed: a kernel on stream 0 first waits for every
        other stream to submit, drains every OTHER active stream, and, if
        anything is pending deletion, deletes it and drains stream 0 too
        (`gpu_single_worker.cpp:80-87`). That is what makes the default stream
        an ordering point across streams.

        The caller launches on `self.ctx` immediately after. Every launch in
        the tree is expected to be bracketed by one of these, which is what
        makes `launch_count` mean something.
        """
        var stream_id = stream.get_stream()
        var idx = Int(stream_id)
        if idx < 0 or idx >= len(self.streams):
            raise Error(
                String("StreamKernel: no stream ")
                + String(stream_id)
                + ". Streams come from request_stream(); their worker indexes"
                + " Streams[streamId] directly (gpu_single_worker.cpp:88)."
            )

        if stream_id == 0:
            self.wait_all_task_to_submit()
            self.sync_active_streams(True)
            if len(self.objects_to_free) > 0:
                self.delete_objects()
                self.sync_stream(0)

        self.launch_count += 1
        self.streams[idx].add_task()

    # -------------------------------------------------------------- the switch

    def reset(mut self, command: TCommand) raises:
        """Their `Reset` (`gpu_single_worker.h:305-323`).

        Theirs drops both memory providers and rebuilds them at the requested
        sizes. Memory belongs to `DeviceContext` here, so the rebuild has no
        counterpart and a request for a non-zero size is refused rather than
        accepted and ignored. The shutdown spelling `TResetCommand(0.0, 0)`
        (`single_device.h:252`) is honoured: there is nothing to rebuild.
        """
        if command.gpu_memory_part != 0.0 or command.pinned_memory_size != 0:
            raise Error(
                "Reset with a non-zero memory size has no counterpart in this"
                " port: memory providers are owned by DeviceContext. See"
                " NOT_PORTED.md."
            )

    def run(mut self, var command: TCommand) raises:
        """Their dispatch switch (`gpu_single_worker.cpp:69-145`).

        Same cases, same order. The ones with no single-device meaning raise
        rather than silently doing nothing, because a silent no-op is exactly
        how `enqueue_copy(dst_buf=, src_ptr=device)` cost this port a day.
        """
        var type = command.get_command_type()

        if type == ECommandType.Reset:
            self.wait_submit_and_sync()
            self.reset(command)
        elif type == ECommandType.StreamKernel:
            self.stream_kernel(command.stream)
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
        elif type == ECommandType.WaitSubmit:
            # `gpu_single_worker.cpp:107-109`. Submission only. NOT a drain.
            self.wait_all_task_to_submit()
        elif type == ECommandType.HostTask:
            # `gpu_single_worker.cpp:111-118`. Theirs then runs the task body
            # through `Exec(*this)`; a host task body cannot be stored here
            # for the same reason a kernel body cannot (DEVIATION 2), so the
            # caller runs it around this call.
            if is_blocking_host_task(command.get_host_task_type()):
                self.wait_submit_and_sync()
        elif type == ECommandType.RequestStream:
            command.set_stream_id(self.request_stream_impl())
        elif type == ECommandType.FreeStream:
            self.free_streams_impl(command.streams.copy())
        elif type == ECommandType.StopWorker:
            self.wait_submit_and_sync()
            self.stopped = True
        else:
            # `gpu_single_worker.cpp:139-141` is `Y_UNREACHABLE()`, because a
            # SerializedCommand is deserialised into a real command before the
            # switch (`gpu_single_worker.cpp:65-67`). Deserialisation exists to
            # receive a command from another host over MPI; this port is
            # single host. See NOT_PORTED.md.
            raise Error(
                "SerializedCommand is unreachable: it exists to ship a"
                " command to another host over MPI, and this port is single"
                " host. See NOT_PORTED.md."
            )

    def run_iteration(mut self) raises -> Bool:
        """Their `RunIteration` (`gpu_single_worker.cpp:49-152`), minus the
        blocking `InputTaskQueue.Wait` on an empty queue: there is no producer
        thread to wait for.

        Returns their `shouldStop`.
        """
        _ = self.check_running_tasks()
        if self.queue.is_empty():
            return self.stopped
        var command = self.queue.dequeue()
        self.run(command^)
        return self.stopped

    def drain_queue(mut self) raises:
        """Their worker loop (`gpu_single_worker.cpp:165-170`)."""
        while not self.queue.is_empty():
            if self.run_iteration():
                break

    def add_task(mut self, var command: TCommand):
        """Their `TSingleHostTaskQueue::AddTask` reached through the worker."""
        self.queue.add_task(command^)

    def is_running(self) -> Bool:
        """Their `IsRunning()` (`gpu_single_worker.h:353-355`)."""
        return not self.stopped

    def stop(mut self) raises:
        """Their post-loop checks (`gpu_single_worker.cpp:171-179`).

        Every one of these is a CB_ENSURE in their tree, and each catches a
        different way of leaking: a command queued after the stop, a stream
        handed out and never freed, an object still pending deletion.
        """
        self.stopped = True
        if not self.queue.is_empty():
            raise Error("Error: found tasks after stop command")
        if (1 + len(self.free_streams)) != len(self.streams):
            raise Error(
                String("Error: ")
                + String(len(self.streams) - 1 - len(self.free_streams))
                + " user streams were never freed"
            )
        if len(self.objects_to_free) > 0:
            raise Error("Error: objects are still pending deletion")
        for i in range(len(self.streams)):
            self.streams[i].check_completed()
            self.streams_provider.free_stream(self.streams[i].get_stream())
        self.streams.clear()
        self.free_streams.clear()
        self.objects_to_free.clear()

    def reset_counters(mut self):
        """OURS, not theirs. Per-tree accounting: `structure_searcher` calls
        this on entry so `sync_count` reads per tree and not per process."""
        self.launch_count = 0
        self.sync_count = 0
