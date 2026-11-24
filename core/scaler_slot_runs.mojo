# SPDX-License-Identifier: Apache-2.0
"""A scaler member's fold rows into the x_prep store for the scaler cores' x_prep programs (lane vdf-scope-neural6,
2026-11-24).

The StandardScaler / MinMaxScaler members keep a model-selection fold in the preprocessing image's member store
(preprocessing/prep_member_store.mojo, on the preprocessing context) and run the direct fit and transform on that
slot in place. A NaN X, sample weights or an all-NaN column take the scaler cores' x_prep stage programs
(core/scaler_core.mojo), which the x_prep runner executes over an arena; its ranges form
(x_prep/transport_abi.mojo XP_RUN_RANGES, core/arena_io.mojo `upload_ranges`) fills an input range from a slot of
the x_prep store (x_prep/store_slot.mojo X_PREP_STORE) device to device. `scaler_slot_to_xprep` makes that slot from
the member store's slot: ONE device-to-device copy enqueued on the x_prep context (the member store's take
synchronized its own context before the member got the id; a store free synchronizes, so the copy completes before
either slot is released). The rows never visit the host; the same words in the same order as the host gather, so the
programs compute the same bits.

Both stores are process-wide named slots (core/shared_slot.mojo): this module names the member store by its slot
name, as core/fold_x_device.mojo names the x_prep store, so importing it pulls neither image's kernels. Only a GPU
route reaches these (the host image publishes no member store, so its members never hold a slot).

Serial by nature: one copy enqueue and one bookkeeping update per call.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.device_store import DeviceStore
from core.shared_slot import SharedSlot
from core.fold_x_device import FOLD_X_STORE
from x_prep.store_slot import x_prep_ctx

#: preprocessing/prep_member_store.mojo `PREP_MEMBER_STORE` (same name, same layout word); keep the two in step.
comptime _PREP_STORE_NAME = (
    "MojolearnPrepMemberStoreIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    else "MojolearnPrepMemberStoreFastV1"
)
comptime _PREP_STORE = SharedSlot[DeviceStore, _PREP_STORE_NAME, 2]


def scaler_slot_to_xprep(prep_id: Int, words: Int) raises -> Int:
    """A fresh x_prep store slot holding the first `words` words of member store slot `prep_id` (module note);
    the caller frees it with `scaler_xprep_free` after its program ran."""
    var ctx = x_prep_ctx()
    var store = FOLD_X_STORE.get_or_create_ptr()
    var id = store[].create(ctx, words)
    try:
        var dst = store[]._held(id, words)
        _PREP_STORE.get_or_create_ptr()[].copy_into(ctx, prep_id, dst, 0, words)
        _ = dst^  # the queued copy holds its own reference
    except error:
        store[].free(ctx, id)
        raise error^
    return id


def scaler_xprep_free(id: Int) raises:
    """Release a slot `scaler_slot_to_xprep` made (waits for the x_prep context: the program that read it)."""
    FOLD_X_STORE.get_or_create_ptr()[].free(x_prep_ctx(), id)
