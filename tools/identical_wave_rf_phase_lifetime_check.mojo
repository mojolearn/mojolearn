"""Delayed-copy sensitivity check for RF's two pinned phase upload sources.

The single-source arm deliberately reproduces the rejected lifetime pattern.
It must corrupt the histogram payload; two persistent sources must preserve
both distinct payloads. This checks the async lifetime mechanism, not forest
accuracy; full DART repeat and cross-column checks remain separate gates.
"""
from max.gpu.host import DeviceContext


def delay_kernel(p: MutPointer[UInt32, MutAnyOrigin], iterations: Int32):
    var x = p[unsafe_offset=0]
    for _ in range(Int(iterations)):
        x = (x ^ (x >> 13)) * UInt32(1664525) + UInt32(1013904223)
    p[unsafe_offset=0] = x


def check(ctx: DeviceContext, separate: Bool) raises -> Bool:
    var first = ctx.enqueue_create_host_buffer[DType.uint32](1024)
    var spare = ctx.enqueue_create_host_buffer[DType.uint32](1024)
    var histogram = ctx.enqueue_create_buffer[DType.uint32](1024)
    var partition = ctx.enqueue_create_buffer[DType.uint32](1024)
    var sink = ctx.enqueue_create_buffer[DType.uint32](1)
    var got_hist = ctx.enqueue_create_host_buffer[DType.uint32](1024)
    var got_part = ctx.enqueue_create_host_buffer[DType.uint32](1024)
    ctx.enqueue_memset(sink, UInt32(17))
    ctx.synchronize()
    for i in range(1024):
        first.unsafe_ptr()[unsafe_offset=i] = UInt32(i + 100)
    ctx.enqueue_function[delay_kernel](
        sink.unsafe_ptr(), Int32(10000000), grid_dim=1, block_dim=1
    )
    ctx.enqueue_copy(dst_buf=histogram, src_ptr=first.unsafe_ptr())
    if separate:
        swap(first, spare)
    for i in range(1024):
        first.unsafe_ptr()[unsafe_offset=i] = UInt32(i + 9000)
    ctx.enqueue_copy(dst_buf=partition, src_ptr=first.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=got_hist, src_buf=histogram)
    ctx.enqueue_copy(dst_buf=got_part, src_buf=partition)
    ctx.synchronize()
    var valid = True
    for i in range(1024):
        if got_hist.unsafe_ptr()[unsafe_offset=i] != UInt32(i + 100):
            valid = False
        if got_part.unsafe_ptr()[unsafe_offset=i] != UInt32(i + 9000):
            raise Error("partition upload corrupted")
    # Keep BOTH pinned spans alive through the queue completion.
    _ = first^
    _ = spare^
    return valid


def main() raises:
    var ctx = DeviceContext()
    if check(ctx, False):
        raise Error("single-source sabotage was not detected: sensitivity unverified")
    if not check(ctx, True):
        raise Error("double-buffered phase upload corrupted")
    print("RF_PHASE_LIFETIME PASS: rejected overwritten source; preserved both live sources")
