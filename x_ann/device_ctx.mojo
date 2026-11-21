# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ONE process-lifetime DeviceContext for every x_ann device entry (the x_cnn
`_Global` pattern; CURRENT DIRECTIVES, "x_cluster and x_neighbors hang on the
SECOND GPU call"). A context per call can be torn down while a call's buffers
still hold its allocations, and on Metal it exhausts the per-process command
queues. The slot keeps a reference for the life of the process. One slot per
numeric tier, so a FAST and an IDENTICAL .so in one process never share it.
Moves no bit: the kernels, their launch shapes and their order are unchanged.
Regression test: python/mojolearn/tests/test_x_ann_repeat.py."""
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock


struct _AnnContext(Defaultable, Movable):
    var ctx: Optional[DeviceContext]
    #: Lane fx-gil (2026-11-21): the first creation was serialized by the
    #: GIL, which a door releases around its kernels and a member's worker
    #: never holds. Held for the test, the creation and the handle copy.
    var ctx_lock: List[Int64]

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()
        self.ctx_lock = List[Int64](length=1, fill=Int64(0))


comptime _CTX_NAME = "MojolearnXAnnContextIdenticalV4" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnXAnnContextFastV4"
comptime X_ANN_CONTEXT = SharedSlot[_AnnContext, _CTX_NAME, 2]


def x_ann_ctx() raises -> DeviceContext:
    """The shared context, created on first use."""
    var slot = X_ANN_CONTEXT.get_or_create_ptr()
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](slot[].ctx_lock.unsafe_ptr())
    shared_slot_lock(word)
    try:
        if not slot[].ctx:
            slot[].ctx = DeviceContext()
        return slot[].ctx.value().copy()
    finally:
        shared_slot_unlock(word)
