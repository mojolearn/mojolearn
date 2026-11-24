# SPDX-License-Identifier: Apache-2.0
"""A model-selection fold's X as a kernel entry reads it from the device
(lane vdf-scope-linear, 2026-11-24): THE FOLD X WORD.

A member that answers `MEMBER_CAN_SCOPE` / `MEMBER_CAN_ROW_INDEX`
(core/member_abi.mojo) is handed the WHOLE host X, the fold's row numbers
and the caller's resident scope. It keeps X resident in the scope's cache
(the x_prep store, bindings/arena_cache_run.mojo: the one store the scope's
cache holds today), takes the fold's rows on the device into a fresh slot of
that store (`XP_DEV_TAKE_ROWS`, core/device_store.mojo `take_rows`, which
waits for its own context before it answers) and hands its kernel entry
the slot as a NEGATIVE X address word: `fold_x_word(slot) = -(slot + 1)`.
A host address is never negative, so the entry's existing request spans
carry it unchanged. Only an entry whose image says it reads the word (the
family's flags bit, e.g. core/linear_fit_types.mojo `LF_HAS_SLOT_X`) is
handed one; every other entry keeps host addresses.

At the entry's own upload of X, `fold_x_fill` either copies the host words
(the old call, unchanged) or enqueues one device-to-device copy of the slot
into the entry's own buffer on the entry's context: the rows never visit
the host. The slot belongs to the x_prep store's context; `take_rows`
synchronized that context before the member got the id, so the words are
complete; the entry synchronizes its own stream before it returns (every
fit and predict entry downloads its result), before the member frees the
slot. Same bytes as the host upload (a byte copy of the same rows in the
same order as the host gather, core/admitted_rows_gather.mojo): no bit
moves.

THE STORE NAME is x_prep/device.mojo `_STORE_NAME` (a process-wide named
slot, so every binding that names it reads the same store); keep the two in
step. Importing x_prep/device.mojo itself would pull the x_prep kernels into
every family binding.
"""
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.device_store import DeviceStore
from core.shared_slot import SharedSlot
# The numbering lives in core/fold_x_word.mojo (no device import; lane vdf-scope-cluster3).
from core.fold_x_word import fold_x_slot, fold_x_word

comptime _FOLD_X_STORE_NAME = (
    "MojolearnXPrepStoreIdenticalV2" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnXPrepStoreFastV2"
)
#: x_prep/device.mojo `X_PREP_STORE` (same name, same layout word).
comptime FOLD_X_STORE = SharedSlot[DeviceStore, _FOLD_X_STORE_NAME, 2]

def fold_x_fill(ctx: DeviceContext, d_x: DeviceBuffer[DType.float32], x_word: Int, words: Int) raises:
    """Fill the entry's device X: from the host words at `x_word` (the old
    upload: the whole of `d_x`), or, for a negative word, one device copy
    of the first `words` words of its store slot (no wait: stream order)."""
    if x_word < 0:
        if words <= 0:
            return
        var store = FOLD_X_STORE.get_or_create_ptr()
        store[].copy_into(ctx, fold_x_slot(x_word), d_x, 0, words)
        return
    ctx.enqueue_copy(dst_buf=d_x, src_ptr=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=x_word))


def fold_x_store_into(ctx: DeviceContext, x_word: Int, src: DeviceBuffer[DType.float32], words: Int) raises:
    """`src`'s first `words` words into the slot a negative word names (an
    addition, lane vdf-scope-linear4); waits."""
    var store = FOLD_X_STORE.get_or_create_ptr()
    store[].copy_from(ctx, fold_x_slot(x_word), src, words)
    ctx.synchronize()


def fold_x_new(ctx: DeviceContext, words: Int) raises -> Int:
    """A fresh slot of `words` unspecified words of the fold store as its
    fold X word, for a device stage that writes every word before any read
    (an addition, lane vdf-scope-linear4)."""
    var store = FOLD_X_STORE.get_or_create_ptr()
    return fold_x_word(store[].create(ctx, max(words, 1)))


def fold_x_release(ctx: DeviceContext, x_word: Int) raises:
    """Free the slot a negative word names (no-op for a host word)."""
    if x_word < 0:
        var store = FOLD_X_STORE.get_or_create_ptr()
        store[].free(ctx, fold_x_slot(x_word))
