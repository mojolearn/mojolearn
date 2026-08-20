"""The queue the worker drains.

PORT OF `catboost/cuda/cuda_lib/tasks_queue/single_host_task_queue.h` at
CatBoost `54a8143a`. Transliterated where it transliterates. See the
DEVIATION BLOCK.

Their queue is `NThreading::TOneOneQueue` (`single_host_task_queue.h:13`), a
lock-free single-producer single-consumer ring, plus a `TManualEvent` so the
consumer can sleep instead of spinning forever. `Wait`
(`single_host_task_queue.h:26-46`) spins for a full second first, yielding
every 10000 iterations, and only then blocks on the event, because on a busy
tree the next command is almost always already there.

`Wait` and `EmplaceTask` (`single_host_task_queue.h:54-58`) are not ported.
`Wait` has no referent without a consumer thread: the only caller is their
worker's idle branch (`gpu_single_worker.cpp:61`), and ours returns instead of
idling. `EmplaceTask` is `AddTask(MakeHolder<TTask>(args...))`, and our
commands are values built by the factory functions in `task.mojo`.

================================ DEVIATION BLOCK ======================
Ours is a plain `List[TCommand]` with a head index, drained on the CALLING
thread. There is no producer thread and no consumer thread.

The reason is that `DeviceContext.enqueue_function` is ALREADY an async
submit: it returns without waiting for the device. Their worker thread exists
to keep `cudaLaunchKernel` off the host's critical path, and Mojo's runtime
has done that for us before our code is reached. A thread here would add a
hop and buy nothing we can name.

OPEN, and MEASURABLE, so it is written down rather than assumed: 54 launches
behind one drain still cost 1.2 to 2.1 ms, about 22 microseconds a launch,
and we have NOT established whether that is submission cost on the calling
thread or real device time. If it is submission cost, a worker thread starts
paying and this deviation should be revisited with a number attached.
======================================================================
"""

from ..task import TCommand


struct TSingleHostTaskQueue(Movable):
    """Their `TSingleHostTaskQueue` (`single_host_task_queue.h:11`)."""

    var input_task_queue: List[TCommand]
    var head: Int

    def __init__(out self):
        self.input_task_queue = List[TCommand]()
        self.head = 0

    def is_empty(self) -> Bool:
        """Their `IsEmpty()` (`single_host_task_queue.h:15-17`)."""
        return self.head >= len(self.input_task_queue)

    def add_task(mut self, var task: TCommand):
        """Their `AddTask` (`single_host_task_queue.h:48-52`).

        Theirs also signals `JobsEvent` to wake a sleeping consumer. We have
        no consumer to wake.
        """
        self.input_task_queue.append(task^)

    def dequeue(mut self) raises -> TCommand:
        """Their `Dequeue()` (`single_host_task_queue.h:19-24`). Same
        contract: `CB_ENSURE(done, "Error: dequeue failed")`, so draining an
        empty queue raises.

        The copy is because a `List` element cannot be moved out from under an
        index. It costs one `TCommand`, whose only heap field is the stream
        list that only FreeStream fills.
        """
        if self.is_empty():
            raise Error("Error: dequeue failed")
        var task = self.input_task_queue[self.head].copy()
        self.head += 1
        if self.head == len(self.input_task_queue):
            self.input_task_queue.clear()
            self.head = 0
        return task^

    def size(self) -> Int:
        """OURS, not theirs: their `TOneOneQueue` exposes no size. It is here
        for probes, and nothing in the control plane branches on it."""
        return len(self.input_task_queue) - self.head
