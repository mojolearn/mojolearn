from max.gpu.host import DeviceContext
from ported.gpu_lib.gpu_base import DEFAULT_STREAM
from ported.gpu_lib.gpu_manager import TCudaManager
from ported.gpu_lib.task import ECommandType, TCommand


def main() raises:
    var ctx = DeviceContext()
    var mgr = TCudaManager(ctx^, sync_budget=2)
    var s = mgr.request_stream()
    print("devices", mgr.get_device_count(), "default", mgr.default_stream().get_stream(), "new", s.get_stream())
    for _ in range(20):
        mgr.stream_kernel(s)
    mgr.wait_complete()
    print("after 20 launches: launches", mgr.launch_count(), "syncs", mgr.sync_count())
    mgr.wait_complete()
    print("second drain ok, syncs", mgr.sync_count())
    try:
        mgr.wait_complete()
        print("BUDGET NOT ENFORCED")
    except e:
        print("budget enforced:", String(e)[byte=0:40])
    mgr.reset_counters()
    print("after reset: launches", mgr.launch_count(), "syncs", mgr.sync_count())
    # dispatch switch
    mgr.worker.add_task(TCommand(ECommandType.StreamKernel, DEFAULT_STREAM))
    mgr.worker.drain_queue()
    print("drained one StreamKernel, launches", mgr.launch_count(), "syncs", mgr.sync_count())
    try:
        mgr.worker.run(TCommand(ECommandType.SerializedCommand, DEFAULT_STREAM))
    except e:
        print("serialized refused:", String(e)[byte=0:32])
