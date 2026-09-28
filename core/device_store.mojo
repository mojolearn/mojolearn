# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Device-resident word buffers named by an integer id (lane py-shared,
2026-09-28): the shared store behind `mojolearn._arena_io.DeviceCache`.

A Python loop that hands a binding the same array call after call (a
cross-validation scorer, an imputer round, an optimizer's state, a model's
weights) uploads it once into a `DeviceStore` slot and passes the id; the
binding copies device to device, or launches on the slot's pointer. A slot
holds 4-byte WORDS (float32, or int32 bits: every copy is a byte copy, so
what a slot holds is exactly the bytes the host gave it).

EACH BINDING OWNS ITS OWN STORE, next to its own DeviceContext, because a
device buffer belongs to the context that made it:

    comptime MY_STORE = _Global[StorageType=DeviceStore, name="MojoMyStore<Tier>",
                                init_fn=DeviceStore.__init__]
    var s = MY_STORE.get_or_create_ptr()
    var id = s[].put(my_ctx(), host_addr, n_words)

and exports `<prefix>_dev_put(addr, n_words) -> id`, `<prefix>_dev_free(id)`
and `<prefix>_dev_live() -> count` (x_metrics/resident.mojo is the pattern,
three lines each), which `DeviceCache` calls by that prefix.

FREED DETERMINISTICALLY: `free` waits for the context (so no queued launch
still reads the slot), then drops the buffer at once (the slot keeps a
shared one-word placeholder), and the id is reused. Nothing is pooled, so
live device memory is exactly the live slots. Where bytes live moves no bit.
"""
from max.gpu.host import DeviceBuffer, DeviceContext

comptime StoreWP = MutPointer[Float32, MutAnyOrigin]


struct DeviceStore(Defaultable, Movable):
    var bufs: List[DeviceBuffer[DType.float32]]
    #: words held by each id; -1 for a freed id
    var words: List[Int]
    var released: List[Int]
    var live: Int

    def __init__(out self):
        self.bufs = List[DeviceBuffer[DType.float32]]()
        self.words = List[Int]()
        self.released = List[Int]()
        self.live = 0

    def check(self, id: Int, need: Int) raises:
        """Raises unless `id` is live and holds at least `need` words."""
        if id < 0 or id >= len(self.words) or self.words[id] < 0:
            raise Error(String("device store: id ", id, " is not live"))
        if self.words[id] < need:
            raise Error(String("device store: id ", id, " holds ", self.words[id], " words, ", need, " read"))

    def put(mut self, ctx: DeviceContext, src_addr: Int, n_words: Int) raises -> Int:
        """A new slot holding the `n_words` host words at `src_addr`; waits
        for the copy (the host memory may change after this returns)."""
        if n_words < 1 or src_addr == 0:
            raise Error("device store: put needs at least one word at a non-null address")
        var buf = ctx.enqueue_create_buffer[DType.float32](n_words)
        ctx.enqueue_copy(dst_buf=buf, src_ptr=StoreWP(unsafe_from_address=src_addr))
        ctx.synchronize()
        var id: Int
        if len(self.released) > 0:
            id = self.released.pop()
            self.bufs[id] = buf^
            self.words[id] = n_words
        else:
            self.bufs.append(buf^)
            self.words.append(n_words)
            id = len(self.words) - 1
        self.live += 1
        return id

    def write(mut self, ctx: DeviceContext, id: Int, src_addr: Int, n_words: Int) raises:
        """Overwrite the slot's first `n_words` words from the host; waits."""
        self.check(id, n_words)
        if n_words > 0:
            ctx.enqueue_copy(
                dst_buf=self.bufs[id].create_sub_buffer[DType.float32](0, n_words),
                src_ptr=StoreWP(unsafe_from_address=src_addr),
            )
        ctx.synchronize()

    def read(self, ctx: DeviceContext, id: Int, dst_addr: Int, n_words: Int) raises:
        """The slot's first `n_words` words into host memory, after every
        launch queued before it; waits."""
        self.check(id, n_words)
        if n_words > 0:
            ctx.enqueue_copy(
                dst_ptr=StoreWP(unsafe_from_address=dst_addr),
                src_buf=self.bufs[id].create_sub_buffer[DType.float32](0, n_words),
            )
        ctx.synchronize()

    def copy_into(self, ctx: DeviceContext, id: Int, dst: DeviceBuffer[DType.float32], at: Int, n_words: Int) raises:
        """Enqueue a device-to-device copy of the slot's first `n_words`
        words into `dst[at, at + n_words)` (no wait: stream order)."""
        self.check(id, n_words)
        if n_words > 0:
            ctx.enqueue_copy(
                dst_buf=dst.create_sub_buffer[DType.float32](at, n_words),
                src_buf=self.bufs[id].create_sub_buffer[DType.float32](0, n_words),
            )

    def ptr(self, id: Int, need: Int) raises -> StoreWP:
        """The slot's device pointer, for a binding's own launches."""
        self.check(id, need)
        return StoreWP(unsafe_from_address=Int(self.bufs[id].unsafe_ptr()))

    def free(mut self, ctx: DeviceContext, id: Int) raises:
        """Release the slot now: wait for the context, drop the buffer."""
        self.check(id, 0)
        ctx.synchronize()
        self.bufs[id] = ctx.enqueue_create_buffer[DType.float32](1)
        self.words[id] = -1
        self.released.append(id)
        self.live -= 1
