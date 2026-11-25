# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The x_prep image's process-lifetime DeviceContext and its resident input
store, the two process slots every `resident()` scope and `DeviceRows`
gather reads (python/mojolearn/_arena_io.py: "the store is the x_prep
binding's").

Moved out of `x_prep/device.mojo` unchanged (lane vdf-xm-kernels,
2026-11-24) so a binding that only READS a resident slot in place (the
`_mojolearn_metrics` kernel entries' device-input form,
`metrics/slot_inputs.mojo`) imports these two slots without the x_prep
program runner. Same struct, same slot names: the layout word is the one
the x_prep images publish. `x_prep/device.mojo` imports them from here.
"""
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.device_store import DeviceStore
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock


struct _PrepContext(Defaultable, Movable):
    """ONE process-lifetime DeviceContext for every `x_prep_run` (the x_cnn
    `_Global` pattern; CURRENT DIRECTIVES, 2026-09-27: a context per call hung
    the SECOND call of x_cluster / x_neighbors on an RTX 4090, and on Metal a
    context per call exhausts the per-process command queues). The slot keeps
    a reference for the life of the process, so every call's buffers die
    inside it. One slot per numeric tier, so a FAST and an IDENTICAL .so in
    one process never share it. Context lifetime moves no bit."""
    var ctx: Optional[DeviceContext]
    #: Lane fx-gil (2026-11-21): the first creation was serialized by the
    #: GIL, which a door releases around its kernels and a member's worker
    #: never holds. Held for the test, the creation and the handle copy.
    var ctx_lock: List[Int64]

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()
        self.ctx_lock = List[Int64](length=1, fill=Int64(0))


comptime _CTX_NAME = "MojolearnXPrepContextIdenticalV4" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnXPrepContextFastV4"
comptime X_PREP_CONTEXT = SharedSlot[_PrepContext, _CTX_NAME, 2]


def x_prep_ctx() raises -> DeviceContext:
    """The shared context, created on first use."""
    var slot = X_PREP_CONTEXT.get_or_create_ptr()
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](slot[].ctx_lock.unsafe_ptr())
    shared_slot_lock(word)
    try:
        if not slot[].ctx:
            slot[].ctx = DeviceContext()
        return slot[].ctx.value().copy()
    finally:
        shared_slot_unlock(word)


#: The binding's resident input store (core/device_store.mojo; lane
#: py-shared): `x_prep_dev_put` / `_free` / `_live`, one per tier.
# V2 (lane linalg-owner, 2026-11-22): DeviceStore gained its lock word
comptime _STORE_NAME = "MojolearnXPrepStoreIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnXPrepStoreFastV3"
comptime X_PREP_STORE = SharedSlot[DeviceStore, _STORE_NAME, 3]
