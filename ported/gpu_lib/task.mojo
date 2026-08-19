"""What the worker can be asked to do, and the tag it dispatches on.

PORT OF `catboost/cuda/cuda_lib/task.h` at CatBoost `54a8143a`.
Transliterated where it transliterates. See the DEVIATION BLOCK.

`ECommandType` is not decoration. `gpu_single_worker.cpp:69-141` is a
`switch (task->GetCommandType())` with one case per value, and the two cases
that matter to a tree are

    StreamKernel   async, goes on a stream, host does NOT wait
    HostTask       sync, ensures every task in the stream completed first

Everything expensive about our current driver is that we made every command a
`HostTask` by hand.

================================ DEVIATION BLOCK ======================
Their `ICommand` is an abstract base with virtual `Save`/`Load`, and the
queue holds `THolder<ICommand>`. Mojo 1.0 has NO dynamic trait objects:
`List[ArcPointer[ICommand]]` fails with

    'ArcPointer' parameter 'T' has 'Deinitable & Movable' type,
    but value has type 'AnyTrait[ICommand]'

and `AnyTrait` is not a spellable name. So the queue holds a TAGGED UNION,
`TCommand`, dispatched on `command_type`.

This is closer to their design than it looks, because their worker already
dispatches on the tag rather than on the vtable. The virtual half of
`ICommand` is `Save` and `Load`, and those exist ONLY to ship a command to
another host over MPI (`TSerializedCommand`, `serialization/task_factory.h`).
We are single host. Nothing is lost that we could use.
======================================================================
"""

from .gpu_base import TCudaStream, DEFAULT_STREAM


struct ECommandType(Copyable, ImplicitlyCopyable, Movable):
    """Their `ECommandType` (`task.h:12-23`), all ten values, same order.

    `SerializedCommand` is kept so the numbering matches theirs; it is
    unreachable here and `TGpuOneDeviceWorker` raises on it.
    """

    var value: Int32

    comptime StreamKernel = Self(0)
    comptime HostTask = Self(1)
    comptime MemoryAllocation = Self(2)
    comptime MemoryDeallocation = Self(3)
    comptime RequestStream = Self(4)
    comptime FreeStream = Self(5)
    comptime WaitSubmit = Self(6)
    comptime Reset = Self(7)
    comptime StopWorker = Self(8)
    comptime SerializedCommand = Self(9)

    def __init__(out self, value: Int32):
        self.value = value

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    def to_string(self) -> String:
        if self == Self.StreamKernel:
            return "StreamKernel"
        if self == Self.HostTask:
            return "HostTask"
        if self == Self.MemoryAllocation:
            return "MemoryAllocation"
        if self == Self.MemoryDeallocation:
            return "MemoryDeallocation"
        if self == Self.RequestStream:
            return "RequestStream"
        if self == Self.FreeStream:
            return "FreeStream"
        if self == Self.WaitSubmit:
            return "WaitSubmit"
        if self == Self.Reset:
            return "Reset"
        if self == Self.StopWorker:
            return "StopWorker"
        return "SerializedCommand"


struct ECpuFuncType(Copyable, ImplicitlyCopyable, Movable):
    """Their `ECpuFuncType` (`task.h:127-130`).

    `DeviceBlocking` is the one that costs. It is their name for exactly the
    thing our driver did nine times per level.
    """

    var value: Int32

    comptime DeviceBlocking = Self(0)
    comptime DeviceNonblocking = Self(1)

    def __init__(out self, value: Int32):
        self.value = value

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value


def is_blocking_host_task(type: ECpuFuncType) -> Bool:
    """Their `IsBlockingHostTask` (`task.h:132`)."""
    return type == ECpuFuncType.DeviceBlocking


struct TCommand(Copyable, Movable):
    """The tagged union that replaces `THolder<ICommand>`.

    Their `ICommand` carries only the tag; each concrete command adds its own
    payload. Ours carries the tag plus the union of the payloads the oblivious
    path actually uses, which is small: a stream id, and the index of the
    kernel body to run.

    `payload_kernel` is an index into the caller's own launch table rather
    than a function pointer, because a Mojo kernel is resolved at comptime by
    `enqueue_function[f]` and cannot be stored. `TGpuKernelTask` in
    `tasks_impl/kernel_task.mojo` is what closes over the real launch.
    """

    var command_type: ECommandType
    var stream: TCudaStream
    var payload_kernel: Int32
    var payload_size: Int
    var host_task_type: ECpuFuncType

    def __init__(out self, command_type: ECommandType, stream: TCudaStream):
        self.command_type = command_type
        self.stream = stream
        self.payload_kernel = -1
        self.payload_size = 0
        self.host_task_type = ECpuFuncType.DeviceNonblocking

    def get_command_type(self) -> ECommandType:
        """Their `GetCommandType()` (`task.h:37`)."""
        return self.command_type

    def get_stream_id(self) -> Int32:
        """Their `IGpuKernelTask::GetStreamId()` (`kernel_task.h:42`)."""
        return self.stream.get_stream()


def stop_worker_command() -> TCommand:
    """Their `TStopWorkerCommand` (`task.h:152`)."""
    return TCommand(ECommandType.StopWorker, DEFAULT_STREAM)


def wait_submit_command() -> TCommand:
    """Their `WaitSubmit` case (`gpu_single_worker.cpp:107`).

    This is the ONLY command that is allowed to make the host wait, and the
    port exists so that a level issues it once instead of nine times.
    """
    return TCommand(ECommandType.WaitSubmit, DEFAULT_STREAM)
