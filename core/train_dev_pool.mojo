# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The training device-array pool, light (lane vdf-linalg4, 2026-11-24).

The pool of `TrainDeviceArray` storage (`mojolearn._training_impl.TrainDeviceArray`), moved out of
training/dev_tensors.mojo so that every binding image can read a device array IN PLACE without importing the GEMM,
the estimators or the training layers: the linalg family (`_mojolearn_linalg`: matmul, the low-bit products and
conversions, qr / eigh / svd / svdvals) and the decomposition kit images read their operands from here and write
their results into new handles here. training/dev_tensors.mojo re-imports every name, so its callers are unchanged.

The slot is process-wide (core/shared_slot.mojo: one value per NAME in the process, every extension module finds the
same one). A pool buffer is a plain allocation on device 0; every entry that reads or writes one synchronizes its own
context before it returns, so a handle written through the training binding's context is complete before the
linalg binding's context reads it, and the reverse.

DTYPE (the one device-array convention, extended with a storage dtype rather than a second kind): the pool holds
float32 WORDS. A float32 array of n elements holds n words; an int8 (codes, int15 planes), int32 (row exponents) or
uint16 (bf16 bits) array of n elements
holds `train_dev_words(n, itemsize)` words, its bytes packed from word 0 (`train_dev_view_as`). The Python owner
carries the element dtype; the native side only ever sees words and the typed view.
"""
from std.sys import size_of
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.shared_slot import SharedSlot

comptime _FP = MutPointer[Float32, MutUntrackedOrigin]

#: Storage dtype codes of a device array (the owner's `dtype` word; 0 is every array made before the code existed).
comptime TRAIN_DEV_F32 = 0
comptime TRAIN_DEV_I8 = 1
comptime TRAIN_DEV_I32 = 2
comptime TRAIN_DEV_U16 = 3


def train_dev_itemsize(code: Int) raises -> Int:
    """Bytes per element of storage dtype `code`."""
    if code == TRAIN_DEV_F32:
        return 4
    if code == TRAIN_DEV_I8:
        return 1
    if code == TRAIN_DEV_I32:
        return 4
    if code == TRAIN_DEV_U16:
        return 2
    raise Error("training device array: unknown storage dtype code " + String(code))


def train_dev_words(n: Int, itemsize: Int) -> Int:
    """The float32 words `n` elements of `itemsize` bytes occupy (at least one)."""
    return max(1, (n * itemsize + 3) // 4)


struct _DevPool(Defaultable, Movable):
    var b: List[DeviceBuffer[DType.float32]]
    #: floats held by each slot; 0 marks a free slot
    var n: List[Int]

    def __init__(out self):
        self.b = List[DeviceBuffer[DType.float32]]()
        self.n = List[Int]()


# The name is unchanged from training/dev_tensors.mojo (same struct, same layout): images built before and after the
# move share one pool.
comptime _DEV_POOL_NAME = "MojolearnTrainingDevTensorsIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnTrainingDevTensorsFastV1"
comptime _DEV_POOL = SharedSlot[_DevPool, _DEV_POOL_NAME]


def train_dev_alloc(ctx: DeviceContext, n: Int) raises -> Int:
    """A device array of `n` floats (uninitialized); its handle."""
    if n < 1:
        raise Error("train_dev_alloc: n must be >= 1, got " + String(n))
    var pool = _DEV_POOL.get_or_create_ptr()
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    var h = -1
    for j in range(len(pool[].n)):  # small-loop(pool: live device-array handles): free handle slot search, no data
        if pool[].n[j] == 0 and h < 0:
            h = j
    if h < 0:
        pool[].b.append(buf^)
        pool[].n.append(n)
        h = len(pool[].n) - 1
    else:
        pool[].b[h] = buf^
        pool[].n[h] = n
    return h


def train_dev_size(h: Int) raises -> Int:
    """The floats handle `h` holds; refuses a handle that is not open."""
    var pool = _DEV_POOL.get_or_create_ptr()
    if h < 0 or h >= len(pool[].n) or pool[].n[h] == 0:
        raise Error("training device array: handle " + String(h) + " is not open")
    return pool[].n[h]


def train_dev_free(ctx: DeviceContext, h: Int) raises:
    """Free handle `h` (after the queue drains)."""
    _ = train_dev_size(h)
    ctx.synchronize()
    var pool = _DEV_POOL.get_or_create_ptr()
    pool[].b[h] = ctx.enqueue_create_buffer[DType.float32](1)
    pool[].n[h] = 0


def train_dev_view(ctx: DeviceContext, h: Int, n: Int, name: String) raises -> DeviceBuffer[DType.float32]:
    """A non-owning buffer over the first `n` floats of handle `h` (the
    pool keeps the storage alive; the handle must hold at least `n`)."""
    var have = train_dev_size(h)
    if n < 1 or have < n:
        raise Error(
            "training device array: " + name + " needs " + String(n) + " floats, handle "
            + String(h) + " holds " + String(have)
        )
    var pool = _DEV_POOL.get_or_create_ptr()
    var p = pool[].b[h].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    return DeviceBuffer[DType.float32](ctx, p, n, owning=False)


def train_dev_view_as[D: DType](ctx: DeviceContext, h: Int, n: Int, name: String) raises -> DeviceBuffer[D]:
    """A non-owning buffer of `n` elements of dtype `D` over handle `h`'s words (the typed storage of the one
    device-array convention: an int8 / int32 / uint16 array packs its bytes from word 0)."""
    if n < 1:
        raise Error("training device array: " + name + " needs at least one element")
    var words = train_dev_words(n, size_of[D]())
    var have = train_dev_size(h)
    if have < words:
        raise Error(
            "training device array: " + name + " needs " + String(words) + " words, handle "
            + String(h) + " holds " + String(have)
        )
    var pool = _DEV_POOL.get_or_create_ptr()
    var p = pool[].b[h].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]().bitcast[Scalar[D]]()
    return DeviceBuffer[D](ctx, p, n, owning=False)


def train_dev_put(ctx: DeviceContext, h: Int, src: _FP, n: Int) raises:
    """Host `n` floats into handle `h`."""
    var v = train_dev_view(ctx, h, n, "the upload")
    ctx.enqueue_copy(dst_buf=v, src_ptr=src)
    ctx.synchronize()
    _ = v^


def train_dev_get(ctx: DeviceContext, h: Int, dst: _FP, n: Int) raises:
    """Handle `h`'s first `n` floats into host memory."""
    var v = train_dev_view(ctx, h, n, "the download")
    ctx.enqueue_copy(dst_ptr=dst, src_buf=v)
    ctx.synchronize()
    _ = v^


def train_dev_put_bytes(ctx: DeviceContext, h: Int, src_address: Int, nbytes: Int) raises:
    """Host `nbytes` bytes (a typed array's elements) into handle `h` from word 0."""
    var v = train_dev_view_as[DType.uint8](ctx, h, nbytes, "the upload")
    ctx.enqueue_copy(dst_buf=v, src_ptr=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=src_address))
    ctx.synchronize()
    _ = v^


def train_dev_get_bytes(ctx: DeviceContext, h: Int, dst_address: Int, nbytes: Int) raises:
    """Handle `h`'s first `nbytes` bytes into host memory (exactly `nbytes`: never the word padding)."""
    var v = train_dev_view_as[DType.uint8](ctx, h, nbytes, "the download")
    ctx.enqueue_copy(dst_ptr=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=dst_address), src_buf=v)
    ctx.synchronize()
    _ = v^
