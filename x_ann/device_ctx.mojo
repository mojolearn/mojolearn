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
from core.shared_slot import SharedSlot


struct _AnnContext(Defaultable, Movable):
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()


comptime _CTX_NAME = "MojolearnXAnnContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnXAnnContextFastV3"
comptime X_ANN_CONTEXT = SharedSlot[_AnnContext, _CTX_NAME]


def x_ann_ctx() raises -> DeviceContext:
    """The shared context, created on first use."""
    var slot = X_ANN_CONTEXT.get_or_create_ptr()
    if not slot[].ctx:
        slot[].ctx = DeviceContext()
    return slot[].ctx.value().copy()
