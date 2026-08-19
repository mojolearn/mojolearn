"""The queue the worker drains.

PORT OF `catboost/cuda/cuda_lib/tasks_queue/single_host_task_queue.h` at
CatBoost `54a8143a`. Transliterated where it transliterates. See the
DEVIATION BLOCK.

Their queue is `NThreading::TOneOneQueue`, a lock-free single-producer
single-consumer ring, plus a `TManualEvent` so the consumer can sleep instead
of spinning forever. `Wait` spins for a full second first, yielding every
10000 iterations, and only then blocks on the event, because on a busy tree
the next command is almost always already there.

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
        """Their `IsEmpty()`."""
        return self.head >= len(self.input_task_queue)

    def add_task(mut self, var task: TCommand):
        """Their `AddTask` (`single_host_task_queue.h:50`).

        Theirs also signals `JobsEvent` to wake a sleeping consumer. We have
        no consumer to wake.
        """
        self.input_task_queue.append(task^)

    def dequeue(mut self) raises -> TCommand:
        """Their `Dequeue()`. Same contract: it is an error to drain empty."""
        if self.is_empty():
            raise Error("Error: dequeue failed")
        var task = self.input_task_queue[self.head].copy()
        self.head += 1
        if self.head == len(self.input_task_queue):
            self.input_task_queue.clear()
            self.head = 0
        return task^

    def size(self) -> Int:
        return len(self.input_task_queue) - self.head
