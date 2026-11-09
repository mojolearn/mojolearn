# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A model-selection fold's X gathered on the device for a kernel entry
(lane a6-device-x-family-classical, 2026-11-09; plan
docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md section 3.2 step 4 and the
cross-validation row of section 4).

Cross-validation keeps X and every fold's row indices resident in model
selection's store (`MselStore`, the process-wide `_Global` below; put once
per fold set by `_Resident._entry`). A family or classical estimator that
opts in (`_mojolearn_device_rows`) is handed a fold as `_arena_io.DeviceRows`;
its door resolves the store ids (bindings/device_rows_door.mojo
`door_rows_store`) and a kernel entry that takes them calls

    msel_device_x_f32(ctx, base_id, base_rows, cols, index_id, rows, d_x, d_status, slot)

instead of its host upload: one grid-stride gather out of the resident base
straight into the entry's own device buffer `d_x` (rows x cols float32,
C order), with the finite test in the same pass (word `slot` of `d_status`
is set to 1 for a NaN or an infinity, word `slot + 1` for an index out of
range). Nothing is synchronised here: the entry reads `d_status` back with
the result it downloads anyway (core/admitted_finite_status.mojo contract),
and refuses through `door_rows_refuse` (bindings/device_rows_door.mojo).

THE CONTEXT. The store's buffers belong to the identical base binding's
process context (slot `MSEL_FOLD_CONTEXT_SLOT`, `process_ctx`): an entry that gathers a
fold runs its kernels on `msel_device_x_context()`, the same DeviceContext,
so no buffer crosses contexts. The arena runner's `_rows_gather`
(bindings/arena_prog_native.mojo) is the model: one gather, no download.

Numerics: none. The words are copied, never converted; the same rows in the
same order as model selection's own device gather and as the host gather
(core/admitted_rows_gather.mojo), so a fit or a score sees the same bytes
on every vendor and on the host column.
"""

from std.ffi import _Global
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.msel_device import MselStore, MSEL_STORE_LAYOUT, MSEL_TPB, MSEL_MAX_BLOCKS
from core.ctx_key import ctx_cache_key
from core.neural_context import process_ctx
from core.shared_slot import shared_slot_check, shared_slot_word

#: The base binding's process-lifetime context slot (bindings/_mojolearn.mojo
#: `_DEVCTX_SLOT`) and store name in this build's numeric mode: the base
#: binding's own msel entries (bindings/msel_device.mojo) use these.
comptime MSEL_CONTEXT_SLOT = (
    "MojolearnCoreContextIdenticalV2" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnCoreContextFastV2"
)
comptime MSEL_STORE_NAME = (
    "MojolearnMselStoreIdenticalV2" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnMselStoreFastV2"
)
#: The model-selection store (one definition; bindings/msel_device.mojo
#: imports it). A `_Global` is named process-wide, so every binding module
#: that names it reads the same store.
comptime MSEL_STORE = _Global[StorageType=MselStore, name=MSEL_STORE_NAME, init_fn=MselStore.__init__]
#: THE store cross-validation folds live in: model selection always connects
#: the IDENTICAL base binding's msel entries (python/mojolearn/
#: _native_arena_rows.py `_native('msel_put')`), whatever mode the estimator's
#: own binding was built in, so a fold reader names the identical store and
#: context explicitly (in an identical build these are MSEL_STORE and
#: MSEL_CONTEXT_SLOT themselves).
comptime MSEL_FOLD_CONTEXT_SLOT = "MojolearnCoreContextIdenticalV2"
comptime MSEL_FOLD_STORE_NAME = "MojolearnMselStoreIdenticalV2"
comptime MSEL_FOLD_STORE = _Global[StorageType=MselStore, name=MSEL_FOLD_STORE_NAME, init_fn=MselStore.__init__]


def msel_store() raises -> UnsafePointer[MselStore, MutAnyOrigin]:
    """This build's store (`MSEL_STORE`), refused when its creator had
    another layout. Its buffers belong to `msel_store_context()`."""
    var store = MSEL_STORE.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()
    shared_slot_check(store[].layout, shared_slot_word[MselStore](MSEL_STORE_LAYOUT), MSEL_STORE_NAME)
    return store


def msel_store_context() raises -> DeviceContext:
    """The context `msel_store()`'s buffers live in (slot MSEL_CONTEXT_SLOT):
    every put, read, gather and scatter of that store runs on it."""
    return process_ctx[MSEL_CONTEXT_SLOT]()


def msel_fold_store() raises -> UnsafePointer[MselStore, MutAnyOrigin]:
    """The identical store cross-validation folds live in, refused when its
    creator had another layout. Its buffers belong to
    `msel_device_x_context()`."""
    var store = MSEL_FOLD_STORE.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()
    shared_slot_check(store[].layout, shared_slot_word[MselStore](MSEL_STORE_LAYOUT), MSEL_FOLD_STORE_NAME)
    return store

comptime MDX_FINITE: Int32 = 0
comptime MDX_FLAGGED: Int32 = 1

comptime _MDX_U32 = MutPointer[UInt32, MutAnyOrigin]
comptime _MDX_I64 = MutPointer[Int64, MutAnyOrigin]
comptime _MDX_I32 = MutPointer[Int32, MutAnyOrigin]


def msel_device_x_context() raises -> DeviceContext:
    """The context the fold store's buffers live in: an entry gathering a
    fold runs on it."""
    return process_ctx[MSEL_FOLD_CONTEXT_SLOT]()


def _mdx_grid(total: Int) -> Int:
    var b = (total + MSEL_TPB - 1) // MSEL_TPB
    if b < 1:
        return 1
    if b > MSEL_MAX_BLOCKS:
        return MSEL_MAX_BLOCKS
    return b


def _mdx_take_finite_kernel(
    src: _MDX_U32, n_src: Int64, words: Int64, idx: _MDX_I64, n_idx: Int64, dst: _MDX_U32,
    status: _MDX_I32, slot: Int32,
):
    # Each thread stores the constant 1 into its flag word: no atomics, no
    # order (the core/admitted_finite_status.mojo rule; Apple has no 64-bit
    # atomic). The finite test is on bits, never a float compare.
    var w = Int(words)
    var total = Int(n_idx) * w
    var step = Int(grid_dim.x) * Int(block_dim.x)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var exp = UInt32(0x7F800000)
    while t < total:
        var r = t // w
        var c = t - r * w
        var s = Int(idx.unsafe_load(r))
        if s < 0 or s >= Int(n_src):
            status.unsafe_store(Int(slot) + 1, MDX_FLAGGED)
        else:
            var word = src.unsafe_load(s * w + c)
            dst.unsafe_store(t, word)
            if (word & exp) == exp:
                status.unsafe_store(Int(slot), MDX_FLAGGED)
        t += step


def msel_device_x_f32(
    ctx: DeviceContext, base_id: Int, base_rows: Int, cols: Int, index_id: Int, rows: Int,
    mut d_x: DeviceBuffer[DType.float32], d_status: _MDX_I32, slot: Int,
) raises:
    """Enqueue the gather of fold rows `index_id` (int64, `rows` words) of the
    resident float32 base `base_id` (`base_rows` x `cols`) into `d_x`
    (at least rows x cols words, C order), with the finite and range flags
    in words `slot` / `slot + 1` of `d_status` (the caller filled them with
    0). `ctx` must be `msel_device_x_context()`. No synchronisation."""
    if rows < 1 or cols < 1:
        return
    if len(d_x) < rows * cols:
        raise Error("msel_device_x_f32: destination holds " + String(len(d_x)) + " words, "
                    + String(rows * cols) + " needed")
    # the store's buffers belong to its own context slot: a reader on any
    # other context (another binding's slot name) would race the writer's
    # stream, so it is refused instead (design note hazard 2)
    if ctx_cache_key(ctx) != ctx_cache_key(msel_device_x_context()):
        raise Error("msel_device_x_f32: ctx is not the fold store's context (msel_device_x_context())")
    var store = msel_fold_store()
    store[].check(base_id, base_rows * cols * 4)
    store[].check(index_id, rows * 8)
    # the store hands out immutable views; the kernel declares mutable
    # pointers (core/msel_device.mojo device_take_rows, the same casts)
    var idx = store[].bufs[index_id].unsafe_ptr().bitcast[Int64]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var src = store[].bufs[base_id].unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]().bitcast[UInt32]()
    var dst = d_x.unsafe_ptr().bitcast[UInt32]()
    ctx.enqueue_function[_mdx_take_finite_kernel](
        src, Int64(base_rows), Int64(cols), idx, Int64(rows), dst, d_status, Int32(slot),
        grid_dim=_mdx_grid(rows * cols), block_dim=MSEL_TPB,
    )

