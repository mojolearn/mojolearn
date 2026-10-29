# Compile-only helper coverage; never execute this artifact.
from core.device_nonfinite_pair import DeviceNonfinitePair
from max.gpu.host import DeviceContext, DeviceBuffer

@export
def compile_nonfinite_pair(ctx: DeviceContext, mut first: DeviceBuffer[DType.float32],
                           mut second: DeviceBuffer[DType.float32], capacity: Int,
                           count: Int, scan_first: Bool) raises -> Tuple[Int, Int]:
    var scratch = DeviceNonfinitePair(ctx, capacity, scan_first)
    return scratch.first_nonfinite(ctx, first, second, count)
