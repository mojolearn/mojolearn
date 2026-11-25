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

Lane vdf-tail-a (2026-11-24): the way back. A NaN X's filled block (the MinMax fit's and the transform's NaN fill
programs) is written by the x_prep ranges runner into an x_prep slot (`scaler_xprep_create`, core/arena_io.mojo slot
outputs), and `scaler_xprep_to_prep` copies it into a fresh member store slot ONE device-to-device copy on the
preprocessing context, where the direct slot fit / transform (SCALER_OP_FIT_SLOT / TRANSFORM_SLOT) read it in place:
the filled block never visits the host. The preprocessing context is named by its slot name
(preprocessing/estimator.mojo `_DEVCTX_SLOT`; keep the two in step), as the store is.

Serial by nature: one copy enqueue and one bookkeeping update per call.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.device_store import DeviceStore
from core.neural_context import process_ctx
from core.shared_slot import SharedSlot
from core.fold_x_device import FOLD_X_STORE
from x_prep.store_slot import x_prep_ctx

#: preprocessing/prep_member_store.mojo `PREP_MEMBER_STORE` (same name, same layout word); keep the two in step.
comptime _PREP_STORE_NAME = (
    "MojolearnPrepMemberStoreIdenticalV2" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    else "MojolearnPrepMemberStoreFastV2"
)
comptime _PREP_STORE = SharedSlot[DeviceStore, _PREP_STORE_NAME, 3]
#: preprocessing/estimator.mojo `_DEVCTX_SLOT` (the context the member store and the direct kernels use).
comptime _PREP_CTX_NAME = (
    "MojolearnPreprocessingContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    else "MojolearnPreprocessingContextFastV3"
)


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


def scaler_xprep_create(words: Int) raises -> Int:
    """A fresh x_prep store slot of `words` words, written by a program's slot output before any read (lane
    vdf-tail-a); freed with `scaler_xprep_free`."""
    return FOLD_X_STORE.get_or_create_ptr()[].create(x_prep_ctx(), words)


def scaler_xprep_to_prep(xp_id: Int, words: Int) raises -> Int:
    """A fresh member store slot holding the first `words` words of x_prep slot `xp_id` (lane vdf-tail-a): ONE
    device-to-device copy enqueued on the preprocessing context, ahead of the direct slot kernels the caller
    launches there (stream order). The x_prep runner that wrote `xp_id` synchronized its context before it
    returned. The caller frees the result with `scaler_prep_free`, and `xp_id` only after that kernel ran."""
    var ctx = process_ctx[_PREP_CTX_NAME]()
    var store = _PREP_STORE.get_or_create_ptr()
    var id = store[].create(ctx, words)
    try:
        var dst = store[]._held(id, words)
        FOLD_X_STORE.get_or_create_ptr()[].copy_into(ctx, xp_id, dst, 0, words)
        _ = dst^  # the queued copy holds its own reference
    except error:
        store[].free(ctx, id)
        raise error^
    return id


def scaler_prep_free(id: Int) raises:
    """Release a slot `scaler_xprep_to_prep` made (waits for the preprocessing context)."""
    _PREP_STORE.get_or_create_ptr()[].free(process_ctx[_PREP_CTX_NAME](), id)
