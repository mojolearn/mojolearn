"""What the worker can be asked to do, and the tag it dispatches on.

PORT OF `catboost/cuda/cuda_lib/task.h` at CatBoost `54a8143a`, plus the two
command payloads that live in `tasks_impl/request_stream_task.h`.
Transliterated where it transliterates. See the DEVIATION BLOCK.

`ECommandType` is not decoration. `gpu_single_worker.cpp:69-141` is a
`switch (task->GetCommandType())` with one case per value, and the two cases
that matter to a tree are

    StreamKernel   async, goes on a stream, host does NOT wait
    HostTask       sync IF it is DeviceBlocking, and only then

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

What the union DOES cost that their class hierarchy does not: every command
carries every payload field, so a `StreamKernel` pays for the `List[Int32]`
that only `TFreeStreamCommand` (`request_stream_task.h:48-69`) uses. Their
`TFreeStreamCommand` is the only command with a heap field, and only it pays.
Measured cost of the empty list here is one pointer-sized field and no
allocation until something appends to it.

The one payload with no Mojo spelling at all is the KERNEL BODY. Their
`TGpuKernelTask<TKernel>` (`tasks_impl/kernel_task.h`) stores the kernel
object in the command; a Mojo kernel is resolved at comptime by
`enqueue_function[f]` and cannot be stored in a struct field. Callers
therefore launch through `TGpuOneDeviceWorker.stream_kernel`, which runs the
StreamKernel case of their switch and hands the caller back the context to
launch on. No field stands in for the kernel, because a field that cannot
hold it would be a lie.
======================================================================
"""

from .fwd import EPtrType
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
    """Their `ECpuFuncType` (`task.h:133-136`).

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
    """Their `IsBlockingHostTask` (`task.h:138-140`)."""
    return type == ECpuFuncType.DeviceBlocking


struct TCommand(Copyable, Movable):
    """The tagged union that replaces `THolder<ICommand>`.

    Field by field, and which of their commands owns each one:

        command_type        `ICommand::Type`                    task.h:27
        stream              `IGpuKernelTask::Stream`            kernel_task.h:67
                            read by `GetStreamId()`             kernel_task.h:43-45
        host_task_type      `IHostTask::GetHostTaskType()`      task.h:150
        handle/size/ptr_type
                            `IAllocateMemoryTask`               task.h:116-120
        gpu_memory_part, pinned_memory_size
                            `TResetCommand`                     task.h:103-104
        streams             `TFreeStreamCommand::Streams`       request_stream_task.h:50
                            read by `GetStreams()`              request_stream_task.h:64
        stream_id_result    `TRequestStreamCommand::StreamId`   request_stream_task.h:25
                            written by `SetStreamId`            request_stream_task.h:41-43

    Every field has a counterpart in one of their commands. No field here
    stands for something they do not have; a field we invented would be a
    deviation, and there is none in this struct.

    `stream_id_result` is where their `RequestStream` case writes its answer
    (`gpu_single_worker.cpp:122`). Theirs writes through a promise the caller
    already holds, so the caller sees it wherever it is; ours writes into the
    command, so the caller has to be holding THAT command. See
    `request_stream_command` below for what that costs.
    """

    var command_type: ECommandType
    var stream: TCudaStream
    var host_task_type: ECpuFuncType
    var handle: UInt64
    var size: UInt64
    var ptr_type: EPtrType
    var gpu_memory_part: Float64
    var pinned_memory_size: UInt64
    var streams: List[Int32]
    var stream_id_result: Int32

    def __init__(out self, command_type: ECommandType, stream: TCudaStream):
        self.command_type = command_type
        self.stream = stream
        self.host_task_type = ECpuFuncType.DeviceNonblocking
        self.handle = 0
        self.size = 0
        self.ptr_type = EPtrType.CudaDevice
        self.gpu_memory_part = 0.0
        self.pinned_memory_size = 0
        self.streams = List[Int32]()
        self.stream_id_result = -1

    def get_command_type(self) -> ECommandType:
        """Their `GetCommandType()` (`task.h:38-40`)."""
        return self.command_type

    def get_stream_id(self) -> Int32:
        """Their `IGpuKernelTask::GetStreamId()`, read at
        `gpu_single_worker.cpp:79`."""
        return self.stream.get_stream()

    def get_host_task_type(self) -> ECpuFuncType:
        """Their `IHostTask::GetHostTaskType()` (`task.h:150`)."""
        return self.host_task_type

    def get_handle(self) -> UInt64:
        """Their `IAllocateMemoryTask::GetHandle()` (`task.h:116`)."""
        return self.handle

    def get_size(self) -> UInt64:
        """Their `IAllocateMemoryTask::GetSize()` (`task.h:118`)."""
        return self.size

    def get_ptr_type(self) -> EPtrType:
        """Their `IAllocateMemoryTask::GetPtrType()` (`task.h:120`)."""
        return self.ptr_type

    def get_streams(self) -> List[Int32]:
        """Their `TFreeStreamCommand::GetStreams()`
        (`request_stream_task.h:64-66`)."""
        return self.streams.copy()

    def set_stream_id(mut self, id: Int32):
        """Their `IRequestStreamCommand::SetStreamId`
        (`request_stream_task.h:19`), called from the worker at
        `gpu_single_worker.cpp:122`."""
        self.stream_id_result = id


def stream_kernel_command(stream: TCudaStream = DEFAULT_STREAM) -> TCommand:
    """Their `TGpuKernelTask` as the worker sees it
    (`gpu_single_worker.cpp:77-93`). The kernel body is not in the command;
    see the DEVIATION BLOCK."""
    return TCommand(ECommandType.StreamKernel, stream)


def host_task_command(
    type: ECpuFuncType = ECpuFuncType.DeviceNonblocking,
) -> TCommand:
    """Their `IHostTask` (`task.h:142-153`)."""
    var cmd = TCommand(ECommandType.HostTask, DEFAULT_STREAM)
    cmd.host_task_type = type
    return cmd^


def blocking_device_sync_command() -> TCommand:
    """Their `TBlockingSyncDevice`, which is what `TCudaSingleDevice::
    WaitComplete()` launches (`single_device.h:349-351`).

    It is a `HostTask` of type `DeviceBlocking`, so the worker's HostTask case
    (`gpu_single_worker.cpp:111-118`) runs `WaitSubmitAndSync()`. This, and
    not `WaitSubmit`, is the command that drains the device.
    """
    return host_task_command(ECpuFuncType.DeviceBlocking)


def allocate_memory_command(
    handle: UInt64, size: UInt64, ptr_type: EPtrType
) -> TCommand:
    """Their `IAllocateMemoryTask` (`task.h:109-121`)."""
    var cmd = TCommand(ECommandType.MemoryAllocation, DEFAULT_STREAM)
    cmd.handle = handle
    cmd.size = size
    cmd.ptr_type = ptr_type
    return cmd^


def free_memory_command(handle: UInt64) -> TCommand:
    """Their `IFreeMemoryTask` (`task.h:123-131`)."""
    var cmd = TCommand(ECommandType.MemoryDeallocation, DEFAULT_STREAM)
    cmd.handle = handle
    return cmd^


def reset_command(
    gpu_memory_part: Float64 = 0.0, pinned_memory_size: UInt64 = 0
) -> TCommand:
    """Their `TResetCommand` (`task.h:88-107`).

    `TCudaSingleDevice::Stop` sends `TResetCommand(0.0, 0)`
    (`single_device.h:252`), which is the shutdown spelling: drop the memory
    providers and allocate nothing back.
    """
    var cmd = TCommand(ECommandType.Reset, DEFAULT_STREAM)
    cmd.gpu_memory_part = gpu_memory_part
    cmd.pinned_memory_size = pinned_memory_size
    return cmd^


def request_stream_command() -> TCommand:
    """Their `IRequestStreamCommand` (`request_stream_task.h:7-20`).

    Read the answer back out of `stream_id_result` after the worker runs it,
    and read it through a DIRECT `TGpuOneDeviceWorker.run(cmd)`, which takes
    the command by `mut`. The queue path cannot hand it back: theirs writes
    into a promise the caller already holds
    (`TRequestStreamCommand::SetStreamId` -> `StreamId.SetValue(id)`,
    `request_stream_task.h:41-43`), and the promise half of their design is
    `future/local_promise_future.h`, which is not ported. See NOT_PORTED.md.
    `TCudaManager.request_stream` therefore calls `request_stream_impl`
    straight, which is the same code the case runs.
    """
    return TCommand(ECommandType.RequestStream, DEFAULT_STREAM)


def free_stream_command(var streams: List[Int32]) -> TCommand:
    """Their `TFreeStreamCommand` (`request_stream_task.h:48-69`).

    It frees a LIST of streams, not one: `TCudaSingleDevice::Stop` hands over
    every user stream at once (`single_device.h:242`).
    """
    var cmd = TCommand(ECommandType.FreeStream, DEFAULT_STREAM)
    cmd.streams = streams^
    return cmd^


def stop_worker_command() -> TCommand:
    """Their `TStopWorkerCommand` (`task.h:155-163`)."""
    return TCommand(ECommandType.StopWorker, DEFAULT_STREAM)


def wait_submit_command() -> TCommand:
    """Their `TWaitSubmitCommand`, sent by `TCudaSingleDevice::
    StreamSynchronize` (`single_device.h:346`) and handled at
    `gpu_single_worker.cpp:107-109`.

    It waits for every task to be SUBMITTED. It does NOT drain the device.
    The drain in their tree is the `TSyncStreamKernel` that
    `StreamSynchronize` launches just before this
    (`single_device.h:345`), or a `DeviceBlocking` host task.
    """
    return TCommand(ECommandType.WaitSubmit, DEFAULT_STREAM)
