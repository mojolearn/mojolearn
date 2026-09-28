# Diagnostic only (lane/cluster-apple-prof, never merged): MOJOLEARN_STAGE_PROF=1
# synchronizes at each mark and prints the stage's wall time.
from std.os import getenv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext


def prof_on() -> Bool:
    return getenv("MOJOLEARN_STAGE_PROF") != ""


def prof_mark(ctx: DeviceContext, name: String, mut t: Int) raises:
    if prof_on():
        ctx.synchronize()
        var now = Int(perf_counter_ns())
        if t > 0:
            print("STAGEPROF", name, Float64(now - t) / 1e6)
        t = now
