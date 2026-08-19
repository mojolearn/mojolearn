"""Streams, and the handle a stream is addressed by.

PORT OF `catboost/cuda/cuda_lib/cuda_base.h` at CatBoost `54a8143a`.
Transliterated where it transliterates. See the DEVIATION BLOCK.

Their `TCudaStreamsProvider` (`cuda_base.h:27-83`) is a free list. A stream
is handed out by `RequestStream()` and returned to the list by
`~TCudaStream()`, so a tree that requests and drops streams every level
recycles the same few `cudaStream_t` rather than creating them.

The stream is the whole point of the control plane. `TSplitPointsKernel::Run`
(`split_points.cpp:232`) issues five kernels back to back on ONE stream and
blocks nowhere, because ordering within a stream is guaranteed by the device,
not by the host.

================================ DEVIATION BLOCK ======================
MEASURED 2026-08-19 on Apple M4: `DeviceContext.stream()` EXISTS in the Mojo
API and raises

    Metal stream not implemented
    (max/mojo/max/gpu/host/device_context.mojo:2172)

So on Metal there is exactly ONE queue. `request_stream` therefore hands out
distinct HANDLES that all resolve to that one queue.

That is correct but stricter than CatBoost. Two of their streams may overlap;
our two handles serialise. Over-ordering never produces a wrong answer, it
only leaves parallelism on the table, so the port is safe today and gets
faster for free on CUDA and HIP the day `stream()` is implemented, with no
change to any caller.

What we DO get on Metal, and what the whole port is for, is that ordering
WITHIN a stream is free. Measured, 54 launches on a trivial kernel:

    sync after each launch    7.7 / 8.9 / 9.8 ms
    one sync at the end       2.1 / 1.2 / 1.2 ms   and bit-exact

======================================================================
"""

from max.gpu.host import DeviceContext


struct TCudaStream(Copyable, ImplicitlyCopyable, Movable):
    """Their `TCudaStreamsProvider::TCudaStream` (`cuda_base.h:35-66`).

    Theirs holds a `cudaStream_t` and a back pointer to the provider it
    returns itself to. Ours holds the handle only; see the DEVIATION BLOCK
    for why there is nothing underneath it to hold on Metal.
    """

    var id: Int32

    def __init__(out self, id: Int32):
        self.id = id

    def __eq__(self, other: Self) -> Bool:
        return self.id == other.id

    def __ne__(self, other: Self) -> Bool:
        return self.id != other.id

    def get_stream(self) -> Int32:
        """Their `GetStream()`."""
        return self.id

    def is_default(self) -> Bool:
        return self.id == 0


comptime DEFAULT_STREAM = TCudaStream(0)


struct TCudaStreamsProvider(Movable):
    """Their free list (`cuda_base.h:27-83`), same request-and-return shape.

    `synchronize` is on the PROVIDER rather than on the stream because on
    Metal a drain is device-wide. Keeping it here instead of on `TCudaStream`
    stops a caller writing `stream.Synchronize()` and believing it drained
    only that stream, which on Metal it does not.
    """

    var free_list: List[Int32]
    var next_id: Int32

    def __init__(out self):
        self.free_list = List[Int32]()
        self.next_id = 1

    def request_stream(mut self) -> TCudaStream:
        """Their `RequestStream()`. Recycle before allocating."""
        if len(self.free_list) > 0:
            var id = self.free_list[len(self.free_list) - 1]
            _ = self.free_list.pop()
            return TCudaStream(id)
        var id = self.next_id
        self.next_id += 1
        return TCudaStream(id)

    def free_stream(mut self, stream: TCudaStream):
        """Their `~TCudaStream()`, made explicit.

        Mojo destructors cannot reach back into an owner the way theirs does,
        so the return to the free list is a call. `TComputationStream` in
        `gpu_single_worker.mojo` is the only thing expected to make it.
        """
        if stream.id != 0:
            self.free_list.append(stream.id)
