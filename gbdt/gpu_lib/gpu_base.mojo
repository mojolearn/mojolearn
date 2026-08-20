"""Streams, and the handle a stream is addressed by.

PORT OF `catboost/cuda/cuda_lib/cuda_base.h` at CatBoost `54a8143a`.
Transliterated where it transliterates. See the DEVIATION BLOCK.

Their `TCudaStreamsProvider` (`cuda_base.h:27-84`) is a free list. A stream
is handed out by `RequestStream()` (`cuda_base.h:75-83`) and returned to the
list by `~TCudaStream()` (`cuda_base.h:50-54`), so a tree that requests and
drops streams every level recycles the same few `cudaStream_t` rather than
creating them.

NOT ported from `cuda_base.h`, and why:

    TDefaultStreamRef, GetDefaultStream, SetDefaultStream  (cuda_base.h:92-113)
        a thread-local pointer to whichever stream the worker made first.
        We have one worker on the calling thread and a `DEFAULT_STREAM`
        constant, so the indirection has no referent.
    TCudaMemoryAllocation, TMemoryCopier, TMemoryCopyKind  (cuda_base.h:133-238)
        allocation and copies belong to `DeviceContext` here. See NOT_PORTED.md
        and the MEMORY PROVIDER note below, which is the evidence for that
        sentence rather than the assertion of it.
    TCudaDeviceProperties, NCudaHelpers                    (cuda_base.h:262-320)
        their `TResetCommand` handler is the only reader
        (`gpu_single_worker.h:315`), and the memory-provider half of Reset is
        not ported. See NOT_PORTED.md.
    CheckLastError                                         (cuda_base.h:244-246)
        called after every worker iteration (`gpu_single_worker.cpp:147`).
        Mojo surfaces device errors by raising from the submitting call, so
        there is no error flag to poll.
    GetStreamsProvider                                     (cuda_base.h:86-88)
        theirs is a `FastTlsSingleton`, one provider per HOST THREAD, shared
        by every worker on it. Ours is a member of `TGpuOneDeviceWorker`. With
        one worker on one thread the two are the same object; with two they
        would not be, which is why the difference is written down here rather
        than left to be noticed.
    ~TCudaStreamsProvider                                  (cuda_base.h:69-73)
        `cudaStreamDestroy` on every stream it still holds. A handle here has
        nothing underneath it to destroy, so there is nothing to run.

=============================== MEMORY PROVIDER ========================
CHECKED 2026-08-19 against the published MAX docs, because "belongs to
`DeviceContext`" was an assertion in this file and in `NOT_PORTED.md` and
this repository has been wrong three times about what Mojo ships.

WHAT THEY HAVE. `memory_provider_trait.h:14` makes
`memory_pool/stack_like_memory_pool.h`'s `TStackLikeMemoryPool` the memory
provider for both device and pinned-host memory, unless `USE_CUDA_MALLOC` is
defined, in which case `memory_pool/cuda_malloc_wrapper.h` calls `cudaMalloc`
straight through. Their own comment states the design
(`stack_like_memory_pool.h:7-13`):

    For most GPU task we don't need sophisticated allocators. We just need to
    allocate big bunch of memory, do some job and deallocate it. ... So for
    buffers we use simple stack-based scheme ... If we don't have enough
    memory we simply defragment it

One slab, carved into `MEMORY_ALIGMENT_BYTES = 256` aligned slices
(`:19`, `:212-216`); a request is served from the first free block if it fits
and from the last block otherwise (`Create`, `:287-313`); a freed block flips
`IsFree` and merges with its free neighbours with no driver call
(`~TMemoryBlock`, `:234-240`, and `MergeFreeBlocks`, `:96-124`); and when the
last block cannot cover the request plus `MEMORY_REQUEST_ADJUSTMENT`
(`= 256 * 32`, `:22`) or would be left with under
`MINIMUM_FREE_MEMORY_TO_DEFRAGMENTATION` (16 MB, `:21`), it compacts through
the last block and retries (`TryDefragment`, `:315-332`, and
`MemoryDefragmentation`, `:144-210`). Exhausting it after that throws
`TOutOfMemoryError` (`:302-305`).

THE SLAB SIZE IS THEIR FREE MEMORY, NOT A CONSTANT. `gpu_single_worker.h:311-321`
calls `cudaMemGetInfo`, warns when under 75% of the device is free, and takes
`gpuMemorySize = free * GpuMemoryPart` with `GpuMemoryPartByWorker = 0.95`
(`devices_provider.h:47`). The pinned-host pool is a flat
`PinnedMemorySize = 1024 * MB` (`devices_provider.h:46`).

WHAT MOJO SHIPS, AND WHY THIS IS NOT PORTED. `DeviceContext` already is a
pooling allocator, stated four separate ways in the published docs:

    select_stream           the returned view "shares this context's full
                            stream set, driver context, and device memory
                            pool"
    __deinit__              releasing a context releases "any cached memory
                            buffers and compiled device functions", so a
                            freed buffer is CACHED rather than returned to
                            the driver
    MODULAR_DEBUG_DEVICE_ALLOCATOR
                            its `poison-all` mode fills "every memory-manager
                            allocation", so every allocation goes through a
                            memory manager
    MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM
                            toggles "the VMM defragmenting allocator, which
                            avoids external-fragmentation OOMs", default on
                            NVIDIA, opt in on AMD MI300

The last one is their `MemoryDefragmentation` under another name, and the
first two are the slab and the free-and-reuse. Porting a stack pool on top of
it would be a second allocator over a first, and it would still ask
`enqueue_create_buffer` for its slab. PORTING_RULES `0b-i` keeps exactly one
substitution alive -- a CLOSED library on their dispatch path -- and the MAX
memory manager is one: it is the runtime, there is no source here to port.

TWO THINGS THE DOCS DO NOT SAY, written down rather than assumed. They do not
describe the Metal backend's memory manager, so the defragmenting behaviour is
claimed only where the docs claim it, on NVIDIA by default and AMD MI300 by
opt-in. And they do not state a size class or reuse policy, so "a repeated
allocation of the same size is cheap" is not established here, only that a
pool and a cache exist.

A SEPARATE WALL, and it is the one that would decide this even if MAX pooled
nothing. A stack pool's product is a derived pointer into one slab, and rule 4
of PORTING_RULES already records that `enqueue_function` refuses derived
pointers as aliasing. Handing kernels slab slices is the shape Mojo rejects.
======================================================================

The stream is the whole point of the control plane. `TSplitPointsKernel::Run`
(`split_points.cpp:37-162`) issues its entire split as six calls on the ONE
`stream` it is handed and blocks nowhere:

    SplitAndMakeSequenceInLeaves        split_points.cpp:41
    SortByFlagsInLeaves                 split_points.cpp:56
    GatherInplaceLeqSize                split_points.cpp:115  (small-leaf path)
    CopyHistograms                      split_points.cpp:135
    UpdatePartitionsAfterSplit          split_points.cpp:143
    UpdatePartitionsPropsForSplit       split_points.cpp:151

Ordering within a stream is guaranteed by the device, not by the host, so the
host never appears between them. The single-leaf twin
`TSplitPointsSingleLeafKernel::Run` (`split_points.cpp:232-352`) is the same
sequence over one leaf. An earlier version of this file put
`TSplitPointsKernel::Run` at `split_points.cpp:232` and called it five
kernels; `:232` is the single-leaf twin and the count is six.

================================ DEVIATION BLOCK ======================
MEASURED 2026-08-19 on Apple M4: `DeviceContext.stream()` EXISTS in the Mojo
API and raises

    Metal stream not implemented
    (max/mojo/max/gpu/host/device_context.mojo:2172)

So on Metal there is exactly ONE queue. `request_stream` therefore hands out
distinct HANDLES that all resolve to that one queue.

That is correct but stricter than CatBoost. Two of their streams may overlap;
our two handles serialize. Over-ordering never produces a wrong answer, it
only leaves parallelism on the table, so the port is safe today and gets
faster for free on CUDA and HIP the day `stream()` is implemented, with no
change to any caller.

WHAT WE LOSE, exactly, read out of their tree rather than guessed. The only
part of the searcher that asks for more than one stream is the by-blocks
histogram, and it asks conditionally:

    config.StreamCount = statCount <= 2 ? 1 : 3;
                                        compute_by_blocks_helper.cpp:386

    if (MaxStreamCount > 1) { ... RequestStream() ... }
    else                    { Streams.push_back(DefaultStream()); }
                                        split_properties_helper.h:88-95

So at `statCount <= 2`, which is every single-output regression and binary
classification run, CatBoost itself is single stream and we lose NOTHING.
Above it they overlap three blocks of the histogram, and that three-way
overlap is what the Metal collapse costs us.

The collapse also DELETES work rather than only serializing it. Their two
cross-stream barriers in that loop are guarded:

    if (!IsOnlyDefaultStream()) { NCudaLib::GetCudaManager().Barrier(); }
                                        split_properties_helper.cpp:1143, :1257

and `IsOnlyDefaultStream()` (`split_properties_helper.h:128-130`) is true
whenever the only stream is stream 0, which here is always. Both barriers are
unreachable in this port, by their own condition, not by our choice.

What we DO get on Metal, and what the whole port is for, is that ordering
WITHIN a stream is free. Measured, 54 launches on a trivial kernel:

    sync after each launch    7.7 / 8.9 / 9.8 ms
    one sync at the end       2.1 / 1.2 / 1.2 ms   and bit-exact

======================================================================
"""


struct TCudaStream(Copyable, ImplicitlyCopyable, Movable):
    """Their `TCudaStreamsProvider::TCudaStream` (`cuda_base.h:35-67`).

    Theirs holds a `cudaStream_t` and a back pointer to the provider it
    returns itself to. Ours holds the handle only; see the DEVIATION BLOCK
    for why there is nothing underneath it to hold on Metal.

    Their `Synchronize()` (`cuda_base.h:56-58`) has no counterpart here and no
    counterpart on the provider either, because neither holds a
    `DeviceContext` to drain. The drain lives on the worker, as their
    `SyncStream` does (`gpu_single_worker.h:227-229`), and that is the only
    place in this port that can wait.
    """

    var id: Int32

    def __init__(out self, id: Int32):
        self.id = id

    def __eq__(self, other: Self) -> Bool:
        return self.id == other.id

    def __ne__(self, other: Self) -> Bool:
        return self.id != other.id

    def get_stream(self) -> Int32:
        """Their `GetStream()` (`cuda_base.h:64-66`)."""
        return self.id

    def is_default(self) -> Bool:
        """Stream 0 is the default stream. Their worker special-cases exactly
        this test (`gpu_single_worker.cpp:80`) and their manager hands out
        `TComputationStream(0, this)` as the default
        (`cuda_manager.h:354-356`)."""
        return self.id == 0


comptime DEFAULT_STREAM = TCudaStream(0)


struct TCudaStreamsProvider(Movable):
    """Their free list (`cuda_base.h:27-84`), same request-and-return shape.

    There is no `synchronize` here. On Metal a drain is device-wide, so a
    method on this type would let a caller write `stream.Synchronize()` and
    believe it drained one stream when it drained everything. Draining is the
    worker's `sync_stream`, their `SyncStream` (`gpu_single_worker.h:227`).
    """

    var free_list: List[Int32]
    var next_id: Int32

    def __init__(out self):
        self.free_list = List[Int32]()
        self.next_id = 1

    def request_stream(mut self) -> TCudaStream:
        """Their `RequestStream()` (`cuda_base.h:75-83`). Recycle before
        allocating."""
        if len(self.free_list) > 0:
            var id = self.free_list[len(self.free_list) - 1]
            _ = self.free_list.pop()
            return TCudaStream(id)
        var id = self.next_id
        self.next_id += 1
        return TCudaStream(id)

    def free_stream(mut self, stream: TCudaStream):
        """Their `~TCudaStream()` (`cuda_base.h:50-54`), made explicit.

        Mojo destructors cannot reach back into an owner the way theirs does,
        so the return to the free list is a call. `TGpuOneDeviceWorker` in
        `gpu_single_worker.mojo` is the only thing expected to make it.

        Their `if (Stream && Owner)` guard drops a moved-from stream. Ours
        drops handle 0, which is the default stream: their `RequestStream`
        never hands out `cudaStream_t` 0 either, since `NewStream` always
        creates a fresh non-blocking stream (`cuda_base.cpp:14-18`).
        """
        if stream.id != 0:
            self.free_list.append(stream.id)
