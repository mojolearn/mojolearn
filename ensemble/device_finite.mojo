"""FAST on Apple (lane apple-fast-rfet-scan): the RandomForest and ExtraTrees
fits refuse a non-finite training cell through a device scan of the X they
already uploaded, so the Python layer skips its one-thread host scan of
every cell (`all_finite` in `randomforest.py` / `extratrees.py`) and the RF
binding skips its pooled host NaN scan. The refusal raises
`FOREST_NONFINITE_REFUSAL`, which the Python fit turns into the same
ValueError it raised before. The iforest pattern (bfd1d7cc6).

`-D MOJOLEARN_FOREST_DEVICE_FINITE_OFF` keeps the host scans (the A/B arm);
IDENTICAL compiles the old path. No bits of any fit move: the scan only
reads X."""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import bitcast
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

comptime FOREST_DEVICE_FINITE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_FOREST_DEVICE_FINITE_OFF"]()
)

comptime FOREST_NONFINITE_REFUSAL = (
    "X contains NaN or infinity; the forest has no missing-value arm"
)
"""The device scan's refusal: the Python layer's own message, which the
Python fit re-raises as ValueError."""

comptime FINITE_SCAN_TPB = 256
comptime FINITE_SCAN_MAX_BLOCKS = 4096


def forest_nonfinite_scan_kernel(
    data: MutPointer[Float32, MutAnyOrigin],
    n: Int64,
    flag: MutPointer[Int32, MutAnyOrigin],
):
    """Every thread walks `i = gid, gid + grid, ...` of `[0, n)` and stores 1
    to `flag[0]` on a NaN or an inf (exponent all ones). Every writer stores
    the same word, so the flag is the same whatever the order."""
    var gid = Int64(block_idx.x) * Int64(block_dim.x) + Int64(thread_idx.x)
    var stride = Int64(grid_dim.x) * Int64(block_dim.x)
    var bad = False
    var i = gid
    while i < n:
        var bits = bitcast[DType.uint32](data.unsafe_load(Int(i)))
        if (bits & UInt32(0x7F800000)) == UInt32(0x7F800000):
            bad = True
        i += stride
    if bad:
        flag.unsafe_store(0, Int32(1))


struct ForestFiniteScan(Movable):
    """The flag of one enqueued scan: `enqueue` after X's upload, read with
    `refuse_if_bad` after the caller's next synchronize (no added drain)."""

    var dflag: DeviceBuffer[DType.int32]
    var hflag: HostBuffer[DType.int32]

    def __init__(out self, ctx: DeviceContext) raises:
        self.dflag = ctx.enqueue_create_buffer[DType.int32](1)
        self.hflag = ctx.enqueue_create_host_buffer[DType.int32](1)

    def enqueue(
        mut self, ctx: DeviceContext, data: DeviceBuffer[DType.float32], n: Int
    ) raises:
        self.dflag.enqueue_fill(Int32(0))
        if n > 0:
            var blocks = min(
                (n + FINITE_SCAN_TPB * 16 - 1) // (FINITE_SCAN_TPB * 16),
                FINITE_SCAN_MAX_BLOCKS,
            )
            ctx.enqueue_function[forest_nonfinite_scan_kernel](
                data.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                Int64(n),
                self.dflag.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                grid_dim=(blocks, 1, 1),
                block_dim=(FINITE_SCAN_TPB, 1, 1),
            )
        ctx.enqueue_copy(dst_ptr=self.hflag.unsafe_ptr(), src_buf=self.dflag)

    def refuse_if_bad(self, who: String) raises:
        """Call after a synchronize that follows `enqueue`."""
        if self.hflag.unsafe_ptr().unsafe_load(0) != 0:
            raise Error(who + FOREST_NONFINITE_REFUSAL)
