"""Does a ported kernel actually enqueue on Metal?

Not a port and not a test: a COMPILE PROBE. Every "it compiles" claim before
this one was `mojo build --emit=object`, which targets the HOST. A kernel
body that typechecks for the host proves nothing about whether it can be
launched, and the whole tree was written without one line of launch code.
This file is the smallest thing that answers the question.
"""

from max.gpu.host import DeviceContext

from .histogram_utils import substract_histograms_kernel


def probe() raises:
    var ctx = DeviceContext()
    var from_ids = ctx.enqueue_create_buffer[DType.uint32](4)
    var what_ids = ctx.enqueue_create_buffer[DType.uint32](4)
    var hist = ctx.enqueue_create_buffer[DType.float32](256)
    ctx.enqueue_function[substract_histograms_kernel](
        from_ids.unsafe_ptr(),
        what_ids.unsafe_ptr(),
        Int32(64),
        hist.unsafe_ptr(),
        grid_dim=(1, 4, 1),
        block_dim=(256, 1, 1),
    )
    ctx.synchronize()
    print("enqueued")
