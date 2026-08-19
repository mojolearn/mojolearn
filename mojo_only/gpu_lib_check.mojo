from ported.gpu_lib.fwd import EPtrType, is_host_ptr
from ported.gpu_lib.gpu_base import TCudaStream, TCudaStreamsProvider, DEFAULT_STREAM
from ported.gpu_lib.slice import TSlice
from ported.gpu_lib.device_id import TDeviceId
from ported.gpu_lib.task import (
    ECommandType,
    ECpuFuncType,
    TCommand,
    is_blocking_host_task,
    stop_worker_command,
    wait_submit_command,
)
from ported.gpu_lib.tasks_queue.single_host_task_queue import TSingleHostTaskQueue


def main() raises:
    print("host ptr:", is_host_ptr(EPtrType.CudaHost), is_host_ptr(EPtrType.CudaDevice))
    var p = TCudaStreamsProvider()
    var s1 = p.request_stream()
    var s2 = p.request_stream()
    print("streams:", s1.get_stream(), s2.get_stream())
    p.free_stream(s1)
    var s3 = p.request_stream()
    print("recycled:", s3.get_stream())
    print("slice:", TSlice(3, 9).to_string(), "size", TSlice(3, 9).size())
    print("dev:", TDeviceId().to_string())
    print("cmd:", ECommandType.StreamKernel.to_string(), ECommandType.WaitSubmit.to_string())
    print("blocking:", is_blocking_host_task(ECpuFuncType.DeviceBlocking))
    var q = TSingleHostTaskQueue()
    q.add_task(TCommand(ECommandType.StreamKernel, DEFAULT_STREAM))
    q.add_task(wait_submit_command())
    q.add_task(stop_worker_command())
    print("queue size:", q.size())
    while not q.is_empty():
        var c = q.dequeue()
        print("  drained", c.get_command_type().to_string(), "stream", c.get_stream_id())
