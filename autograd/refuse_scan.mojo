# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Lane ag-pass6-host (2026-10-19, PASS6_IDEAS idea J): the autograd rules'
# refusal scan with its answer read at the caller's own wait. It reuses
# core/device_scan.mojo's kernels and geometry unchanged; it lives here, not
# in that shared module, so only the autograd bindings rebuild.
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from core.device_scan import (
    NONFINITE_NONE,
    SCAN_TPB,
    _scan_blocks,
    min_partials_kernel,
    nonfinite_partial_kernel,
)
from core.scratch_pool import take_dev_i32, take_host_i32
from core.step_phase import step_count_d2h, step_count_launch


struct DeferredNonfiniteScan(Movable):
    """`device_first_nonfinite` with its answer read at the caller's own
    wait (lane ag-pass6-host, PASS6_IDEAS idea J): the constructor queues
    the partial scan and the device fold of the partials into one word;
    `enqueue_readback` queues the one-word copy; after the caller's next
    synchronize of the same context, `result_after_sync` reads it. No wait
    of its own, so a layer call drains once, not twice. The same kernels,
    geometry and integer minimum as the free function: the same index.

    The caller keeps this value alive until after its wait (the buffers are
    fields; Mojo frees them at the last use, `result_after_sync`). Work
    queued between the constructor and the wait runs on the scanned input
    whatever the verdict: a caller that refuses on the result states that
    its outputs are written before the refusal."""

    var part: DeviceBuffer[DType.int32]
    var res: DeviceBuffer[DType.int32]
    var host: HostBuffer[DType.int32]
    var copied: Bool

    def __init__(out self, ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int) raises:
        var blocks = _scan_blocks(n) if n > 0 else 0
        self.part = take_dev_i32(ctx, blocks)
        self.res = take_dev_i32(ctx, 1)
        self.host = take_host_i32(ctx, 1)
        self.copied = False
        if blocks > 0:
            step_count_launch()
            ctx.enqueue_function[nonfinite_partial_kernel](
                self.part.unsafe_ptr(),
                buf.unsafe_ptr(),
                Int32(n),
                grid_dim=(blocks, 1, 1),
                block_dim=(SCAN_TPB, 1, 1),
            )
        # the partials fold on the device into one word (no block writes
        # NONFINITE_NONE, as does n = 0)
        step_count_launch()
        ctx.enqueue_function[min_partials_kernel](
            self.res.unsafe_ptr(),
            self.part.unsafe_ptr(),
            Int32(blocks),
            grid_dim=(1, 1, 1),
            block_dim=(SCAN_TPB, 1, 1),
        )

    def enqueue_readback(mut self, ctx: DeviceContext) raises:
        """Queue the verdict's copy to the host; it lands at the next wait."""
        var head = self.res.create_sub_buffer[DType.int32](0, 1)
        step_count_d2h()
        ctx.enqueue_copy(dst_ptr=self.host.unsafe_ptr(), src_buf=head)
        _ = head^
        self.copied = True

    def result_after_sync(mut self) raises -> Int:
        """The first flat index of a NaN or infinity, or -1. Only after the
        caller's synchronize that followed `enqueue_readback`."""
        if not self.copied:
            raise Error("deferred nonfinite scan: read before its copy was queued")
        var best = self.host.unsafe_ptr().unsafe_load(0)
        if best == NONFINITE_NONE:
            return -1
        return Int(best)
