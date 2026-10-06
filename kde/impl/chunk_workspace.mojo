# SPDX-License-Identifier: Apache-2.0
"""I20 caller-owned bounded scratch for the existing chunked KDE fold.

Only storage is retained. Every logical partial, log weight and score seam
is recomputed by the existing kernels on every call. Immutable fit handles
are source keys; explicit rebinding drains and invalidates storage before
reuse. No cache of numerical values, hidden upload, or device fallback.
The caller owns one context and may not begin overlapping uses of a pool.
"""
from max.gpu.host import DeviceContext,DeviceBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE,NUMERIC_IDENTICAL
from std.sys.compile import is_defined
from std.atomic import Atomic,Ordering

# I20 PENDING: compile evidence alone does not qualify device correctness, quality or speed.
# Pool requires IDENTICAL + MOJOLEARN_IDN_KDE_PARTIAL_POOL; original allocation path retained.
# Default OFF; enabling it remains a pending experiment, not a promoted route.
comptime KDE_CHUNK_POOL_ON=GLOBAL_NUMERIC_MODE==NUMERIC_IDENTICAL and is_defined["MOJOLEARN_IDN_KDE_PARTIAL_POOL"]()
comptime KDE_CHUNK_POOL_BYTES=64*1024*1024


struct KdeChunkScratch(Movable):
    var part: DeviceBuffer[DType.float32]
    var psum: DeviceBuffer[DType.float32]
    var lse: DeviceBuffer[DType.float32]
    var logw: DeviceBuffer[DType.float32]

    def __init__(out self,ctx: DeviceContext,cells: Int,queries: Int,weights: Int) raises:
        self.part=ctx.enqueue_create_buffer[DType.float32](cells)
        self.psum=ctx.enqueue_create_buffer[DType.float32](cells)
        self.lse=ctx.enqueue_create_buffer[DType.float32](queries)
        self.logw=ctx.enqueue_create_buffer[DType.float32](weights)


struct KdeChunkWorkspace(Movable):
    var scratch: Optional[KdeChunkScratch]
    var budget_bytes: Int
    var source_key: UInt64
    var n_train: Int
    var n_features: Int
    var busy: Bool
    var closed: Bool
    var allocations: Int
    var reuse_calls: Int
    var oversized_calls: Int
    var invalidations: Int
    var lease: Int32

    def __init__(out self,budget_bytes: Int=KDE_CHUNK_POOL_BYTES) raises:
        if budget_bytes<16:raise Error("KDE retained scratch budget must hold four words")
        self.scratch=Optional[KdeChunkScratch]()
        self.budget_bytes=budget_bytes;self.source_key=UInt64(0)
        self.n_train=0;self.n_features=0
        self.busy=False;self.closed=False
        self.allocations=0;self.reuse_calls=0;self.oversized_calls=0;self.invalidations=0
        self.lease=Int32(0)

    def try_lease(mut self) -> Bool:
        var expected=Int32(0)
        return Atomic.compare_exchange[success_ordering=Ordering.ACQUIRE,failure_ordering=Ordering.RELAXED](MutPointer(to=self.lease),expected,Int32(1))

    def release_lease(mut self):
        Atomic.store[ordering=Ordering.RELEASE](MutPointer(to=self.lease),Int32(0))

    def retained_bytes(self) -> Int:
        if self.scratch:
            ref sc=self.scratch.value()
            return 4*(len(sc.part)+len(sc.psum)+len(sc.lse)+len(sc.logw))
        return 0

    def bind(mut self,ctx: DeviceContext,key: UInt64,n_train: Int,n_features: Int) raises:
        if self.closed or self.busy:raise Error("cannot bind closed or in-flight KDE scratch")
        if key==UInt64(0) or n_train<1 or n_features<1:raise Error("invalid immutable KDE source descriptor")
        if key==self.source_key:
            if n_train!=self.n_train or n_features!=self.n_features:raise Error("KDE source key reused with a different fit descriptor")
            return
        if self.scratch:
            self.scratch=Optional[KdeChunkScratch]()
            ctx.synchronize()  # drain allocator frees on the owning context
            self.invalidations+=1
        self.source_key=key;self.n_train=n_train;self.n_features=n_features

    def prepare(mut self,ctx: DeviceContext,key: UInt64,queries: Int,chunks: Int,has_weights: Bool) raises -> Bool:
        if self.closed or self.busy:raise Error("cannot reuse closed or in-flight KDE scratch")
        if key==UInt64(0) or key!=self.source_key:raise Error("KDE scratch does not own this immutable fit source")
        if queries<1 or queries>2147483647 or chunks<1 or chunks>32768 or self.n_train>2147483647 or self.n_features>2147483647:
            raise Error("KDE scratch kernel index bound not proven")
        var cells=queries*chunks
        var weights=self.n_train if has_weights else 1
        var words=2*cells+queries+weights
        if words>self.budget_bytes//4:
            self.oversized_calls+=1
            return False  # caller charges the unchanged temporary-buffer path
        if self.scratch:
            ref sc=self.scratch.value()
            if len(sc.part)>=cells and len(sc.psum)>=cells and len(sc.lse)>=queries and len(sc.logw)>=weights:
                self.reuse_calls+=1
                return True
            # Release before growth to keep retained allocation within budget.
            self.scratch=Optional[KdeChunkScratch]()
            ctx.synchronize()
        self.scratch=KdeChunkScratch(ctx,cells,queries,weights)
        self.allocations+=4
        return True

    def begin(mut self) raises:
        if self.closed or self.busy or not self.scratch:raise Error("KDE scratch cannot begin this consumer")
        self.busy=True

    def complete(mut self,ctx: DeviceContext) raises:
        ctx.synchronize()
        self.busy=False

    def close(mut self,ctx: DeviceContext) raises:
        # Safe after a failed enqueue as well as after a completed score.
        ctx.synchronize()
        self.scratch=Optional[KdeChunkScratch]()
        ctx.synchronize()
        self.busy=False;self.closed=True
