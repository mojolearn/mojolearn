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
from std.ffi import _Global
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


struct _AnnContext(Defaultable, Movable):
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()


comptime _CTX_NAME = "MojoXAnnContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoXAnnContextFast"
comptime X_ANN_CONTEXT = _Global[StorageType=_AnnContext, name=_CTX_NAME, init_fn=_AnnContext.__init__]


def x_ann_ctx() raises -> DeviceContext:
    """The shared context, created on first use."""
    var slot = X_ANN_CONTEXT.get_or_create_ptr()
    if not slot[].ctx:
        slot[].ctx = DeviceContext()
    return slot[].ctx.value().copy()


# Lane ann-apple2 (2026-09-28): a small pool of further process-lifetime
# contexts (each its own command queue), so independent device work can run
# side by side from host tasks (the IVF-PQ subspace codebooks, opt-in by
# MOJOLEARN_ANN_PQ_CB_STREAMS). Same lifetime rule as the slot above.
comptime _ID = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime X_ANN_POOL_0 = _Global[StorageType=_AnnContext, name="MojoXAnnPool0Identical" if _ID else "MojoXAnnPool0Fast", init_fn=_AnnContext.__init__]
comptime X_ANN_POOL_1 = _Global[StorageType=_AnnContext, name="MojoXAnnPool1Identical" if _ID else "MojoXAnnPool1Fast", init_fn=_AnnContext.__init__]
comptime X_ANN_POOL_2 = _Global[StorageType=_AnnContext, name="MojoXAnnPool2Identical" if _ID else "MojoXAnnPool2Fast", init_fn=_AnnContext.__init__]
comptime X_ANN_POOL_3 = _Global[StorageType=_AnnContext, name="MojoXAnnPool3Identical" if _ID else "MojoXAnnPool3Fast", init_fn=_AnnContext.__init__]
comptime X_ANN_POOL_SIZE = 4


def x_ann_pool_ctx(i: Int) raises -> DeviceContext:
    """Pool context i in [0, X_ANN_POOL_SIZE), created on first use."""
    if i == 0:
        var slot = X_ANN_POOL_0.get_or_create_ptr()
        if not slot[].ctx:
            slot[].ctx = DeviceContext()
        return slot[].ctx.value().copy()
    elif i == 1:
        var slot = X_ANN_POOL_1.get_or_create_ptr()
        if not slot[].ctx:
            slot[].ctx = DeviceContext()
        return slot[].ctx.value().copy()
    elif i == 2:
        var slot = X_ANN_POOL_2.get_or_create_ptr()
        if not slot[].ctx:
            slot[].ctx = DeviceContext()
        return slot[].ctx.value().copy()
    var slot = X_ANN_POOL_3.get_or_create_ptr()
    if not slot[].ctx:
        slot[].ctx = DeviceContext()
    return slot[].ctx.value().copy()
