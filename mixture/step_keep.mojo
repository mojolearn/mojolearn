# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Scratch the GaussianMixture E-step and M-step hold across an EM loop
(lane n3b-sgd-on-program-fit, 2026-11-09).

Some arms of `gmm_e_step` (the component-batched and the stacked
Mahalanobis products) and `gmm_m_step` (the nk levels that do not fit the
step's scratch, the transposed responsibilities, the paired centering)
allocate a buffer per call and drained the stream before dropping it (a
"lifetime synchronize"). Inside the EM device loop (mixture/estimator.mojo
`GmmEmStep`) that drain ran in the middle of every chunk of iterations, so
the loop read once per chunk but drained once per iteration anyway.

A `GmmStepKeep` owned by the loop's step holds those buffers instead: a
slot is allocated once at its first use and reused by every later
iteration (the shapes do not change inside a fit); a slot that must grow
retires its old buffer to the keep, so a kernel still in flight never
loses its memory. The keep is dropped after the fit's last drain. A call
without a keep (`keep_addr == 0`: the checks, the sharded E-step) allocates
and drains as before. No kernel, operand or order changes: no bit moves.
"""
from max.gpu.host import DeviceBuffer, DeviceContext

#: The keep's slots (one buffer each).
comptime GMM_KEEP_NK_LEVELS = 1
comptime GMM_KEEP_RESP_T = 2
comptime GMM_KEEP_PAIR_DIFF = 3
comptime GMM_KEEP_PAIR_SCALED = 4
comptime GMM_KEEP_BATCH_PREC = 11
comptime GMM_KEEP_BATCH_Y = 12
comptime GMM_KEEP_BATCH_MU = 13
comptime GMM_KEEP_BATCH_WS = 14
comptime GMM_KEEP_STACK_PREC = 21
comptime GMM_KEEP_STACK_Y = 22
comptime GMM_KEEP_STACK_MU = 23
comptime GMM_KEEP_STACK_WS = 24


struct GmmStepKeep(Movable):
    """The held scratch buffers, by slot."""
    var slots: List[Int]
    var bufs: List[DeviceBuffer[DType.float32]]
    var retired: List[DeviceBuffer[DType.float32]]

    def __init__(out self):
        self.slots = List[Int]()
        self.bufs = List[DeviceBuffer[DType.float32]]()
        self.retired = List[DeviceBuffer[DType.float32]]()

    def buf(mut self, ctx: DeviceContext, slot: Int, floats: Int) raises -> DeviceBuffer[DType.float32]:
        """Slot `slot` as a buffer of exactly `floats` words (a view of the
        held buffer), allocated or grown on the stream when needed."""
        var need = max(floats, 1)
        for i in range(len(self.slots)):  # driver-loop: over the keep's few slots
            if self.slots[i] == slot:
                if len(self.bufs[i]) < need:
                    self.retired.append(self.bufs[i].copy())
                    self.bufs[i] = ctx.enqueue_create_buffer[DType.float32](need)
                return self.bufs[i].create_sub_buffer[DType.float32](0, need)
        self.slots.append(slot)
        self.bufs.append(ctx.enqueue_create_buffer[DType.float32](need))
        return self.bufs[len(self.bufs) - 1].create_sub_buffer[DType.float32](0, need)

    def address(mut self) -> Int:
        """This keep's address, the `keep_addr` the step functions take."""
        return Int(UnsafePointer(to=self))


def gmm_keep_buffer(ctx: DeviceContext, keep_addr: Int, slot: Int, floats: Int) raises -> DeviceBuffer[DType.float32]:
    """Slot `slot` of the keep at `keep_addr`, or a fresh buffer the caller
    drains before dropping when there is no keep (0)."""
    if keep_addr == 0:
        return ctx.enqueue_create_buffer[DType.float32](max(floats, 1))
    var keep = MutPointer[GmmStepKeep, MutAnyOrigin](unsafe_from_address=keep_addr)
    return keep[].buf(ctx, slot, floats)
