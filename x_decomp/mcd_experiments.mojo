# SPDX-License-Identifier: Apache-2.0
"""F06 bounded phase-owned scheduling/cache state. All numerical work is GPU.

Compaction preserves ascending original candidate IDs. It adds a count
read/completion before submitting the reduced grid, a cost that full-fit
timing must include. Covariance reuse admits only identical support masks
inside one immutable phase; initializer, rank path and reweighting stay live.
"""
from std.atomic import Atomic, Ordering
from std.ffi import _Global
from std.gpu import block_idx, thread_idx, block_dim
from std.memory import stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from neighbors.impl.ball_cover.scan import (rbc_exclusive_scan_kernel, rbc_pscan_local_kernel, rbc_pscan_chunks_kernel, rbc_pscan_add_kernel, RBC_PSCAN_CHUNK, RBC_SCAN_TPB)
from x_decomp.cells import F32Ptr, I32Ptr

# F06 PENDING: default off. Source builds do not qualify fit quality/speed.
# NEVER RUN — PENDING VALIDATION: active-candidate compaction.
comptime MCD_FAST_ACTIVE_COMPACT = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and is_defined["MOJOLEARN_MCD_FAST_ACTIVE_COMPACT"]()
# NEVER RUN — PENDING VALIDATION: phase-local covariance reuse.
comptime MCD_FAST_COV_REUSE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and is_defined["MOJOLEARN_MCD_FAST_COV_REUSE"]()
# Three Int32 candidate planes occupy at most768KiB; larger phases retain
# the incumbent route. This resource limit is independent of dataset shapes.
comptime MCD_EXPERIMENT_CANDIDATES = 65536

struct _McdExperimentAudit(Defaultable, Movable):
    var compact: Int64
    var preparations: Int64
    var reused: Int64
    def __init__(out self):
        self.compact = Int64(0)
        self.preparations = Int64(0)
        self.reused = Int64(0)

comptime _MCD_EXPERIMENT_AUDIT = _Global[StorageType=_McdExperimentAudit, name="McdExperimentAuditV1", init_fn=_McdExperimentAudit.__init__]

def mcd_experiment_count(stage: Int) raises -> Int:
    # 0 compact submissions, 1 reuse preparations, 2 actual reused candidates.
    if stage < 0 or stage >= 3: raise Error("invalid MCD experiment stage")
    comptime if MCD_FAST_ACTIVE_COMPACT or MCD_FAST_COV_REUSE:
        ref audit = _MCD_EXPERIMENT_AUDIT.get_or_create_ptr()[]
        if stage == 0: return Int(Atomic.load[ordering=Ordering.RELAXED](MutPointer(to=audit.compact)))
        if stage == 1: return Int(Atomic.load[ordering=Ordering.RELAXED](MutPointer(to=audit.preparations)))
        return Int(Atomic.load[ordering=Ordering.RELAXED](MutPointer(to=audit.reused)))
    return 0

def mcd_experiment_hit(stage: Int, count: Int = 1) raises:
    comptime if MCD_FAST_ACTIVE_COMPACT or MCD_FAST_COV_REUSE:
        ref audit = _MCD_EXPERIMENT_AUDIT.get_or_create_ptr()[]
        if stage == 0: _ = Atomic.fetch_add[ordering=Ordering.RELAXED](MutPointer(to=audit.compact), Int64(count))
        elif stage == 1: _ = Atomic.fetch_add[ordering=Ordering.RELAXED](MutPointer(to=audit.preparations), Int64(count))
        else: _ = Atomic.fetch_add[ordering=Ordering.RELAXED](MutPointer(to=audit.reused), Int64(count))

def compact_flags_kernel(gate: I32Ptr, flags: I32Ptr, nc: Int32):
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c < Int(nc): flags.unsafe_store(c, Int32(1) if gate.unsafe_load(c) != 0 else Int32(0))

def compact_ids_kernel(flags: I32Ptr, prefix: I32Ptr, ids: I32Ptr, total: I32Ptr, nc: Int32):
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c < Int(nc) and flags.unsafe_load(c) != 0:
        ids.unsafe_store(Int(prefix.unsafe_load(c)), Int32(c))
    if c == 0: total.unsafe_store(0, prefix.unsafe_load(Int(nc)))

struct McdCompactWorkspace(Movable):
    var flags: DeviceBuffer[DType.int32]
    var prefix: DeviceBuffer[DType.int32]
    var ids: DeviceBuffer[DType.int32]
    var total: DeviceBuffer[DType.int32]
    var chunks: DeviceBuffer[DType.int32]
    var capacity: Int
    def __init__(out self, ctx: DeviceContext, nc: Int) raises:
        if nc <= 0 or nc > MCD_EXPERIMENT_CANDIDATES: raise Error("MCD compact capacity refused")
        self.capacity = nc
        self.flags = ctx.enqueue_create_buffer[DType.int32](nc)
        self.prefix = ctx.enqueue_create_buffer[DType.int32](nc + 1)
        self.ids = ctx.enqueue_create_buffer[DType.int32](nc)
        self.total = ctx.enqueue_create_buffer[DType.int32](1)
        self.chunks = ctx.enqueue_create_buffer[DType.int32]((nc + RBC_PSCAN_CHUNK - 1) // RBC_PSCAN_CHUNK + 1)

def _device_i32(buf: DeviceBuffer[DType.int32]) -> I32Ptr:
    # Match mcd_fast._i: kernels mutate phase-owned device allocations,
    # while the host borrow leaves the buffer owner/metadata unchanged.
    # Preserve the owner through its phase completion; no allocation/copy.
    return I32Ptr(unsafe_from_address=Int(buf.unsafe_ptr()))

def compact_candidate_count(ctx: DeviceContext, ws: McdCompactWorkspace, gate: I32Ptr, nc: Int) raises -> Int:
    if nc <= 0 or nc > ws.capacity: raise Error("MCD compact source capacity refused")
    ctx.enqueue_function[compact_flags_kernel](gate, _device_i32(ws.flags), Int32(nc), grid_dim=(nc+255)//256, block_dim=256)
    # Stock integer scan kernels; chunk sums are phase-owned, eliminating
    # a temporary-allocation drain before the mandatory count read.
    if nc <= RBC_PSCAN_CHUNK:
        ctx.enqueue_function[rbc_exclusive_scan_kernel](_device_i32(ws.prefix), _device_i32(ws.flags), Int32(nc), grid_dim=1, block_dim=RBC_SCAN_TPB)
    else:
        var chunks = (nc + RBC_PSCAN_CHUNK - 1) // RBC_PSCAN_CHUNK
        ctx.enqueue_function[rbc_pscan_local_kernel](_device_i32(ws.prefix), _device_i32(ws.chunks), _device_i32(ws.flags), Int32(nc), grid_dim=chunks, block_dim=RBC_SCAN_TPB)
        ctx.enqueue_function[rbc_pscan_chunks_kernel](_device_i32(ws.chunks), Int32(chunks), grid_dim=1, block_dim=RBC_SCAN_TPB)
        ctx.enqueue_function[rbc_pscan_add_kernel](_device_i32(ws.prefix), _device_i32(ws.chunks), Int32(nc), Int32(chunks), grid_dim=(nc + 1 + RBC_SCAN_TPB - 1)//RBC_SCAN_TPB, block_dim=RBC_SCAN_TPB)
    ctx.enqueue_function[compact_ids_kernel](_device_i32(ws.flags), _device_i32(ws.prefix), _device_i32(ws.ids), _device_i32(ws.total), Int32(nc), grid_dim=(nc+255)//256, block_dim=256)
    var count = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_ptr=count.unsafe_ptr(), src_buf=ws.total)
    ctx.synchronize()
    var live = Int(count[0])
    if live < 0 or live > nc: raise Error("MCD compact count invalid")
    return live

struct McdCovReuseWorkspace(Movable):
    var gate: DeviceBuffer[DType.int32]
    var reused: DeviceBuffer[DType.int32]
    def __init__(out self, ctx: DeviceContext, nc: Int) raises:
        self.gate = ctx.enqueue_create_buffer[DType.int32](nc)
        self.reused = ctx.enqueue_create_buffer[DType.int32](1)

def reuse_covariance_kernel(mask: I32Ptr, previous: I32Ptr, active: I32Ptr, gate: I32Ptr,
    loc: F32Ptr, prev_loc: F32Ptr, cov: F32Ptr, prev_cov: F32Ptr, reused: I32Ptr,
    rows: Int32, d: Int32, step: Int32):
    var c = Int(block_idx.x); var tid = Int(thread_idx.x)
    var changed = stack_allocation[1, Int32, address_space=AddressSpace.SHARED]()
    if tid == 0: changed[0] = Int32(0)
    barrier()
    if step > 0 and active.unsafe_load(c) != 0:
        for i in range(tid, Int(rows), 256):
            if mask.unsafe_load(c * Int(rows) + i) != previous.unsafe_load(c * Int(rows) + i):
                _ = Atomic.fetch_add[ordering=Ordering.RELAXED](changed, Int32(1))
    barrier()
    var reuse = step > 0 and active.unsafe_load(c) != 0 and changed[0] == 0
    if tid == 0:
        gate.unsafe_store(c, Int32(1) if active.unsafe_load(c) != 0 and not reuse else Int32(0))
        if reuse: _ = Atomic.fetch_add[ordering=Ordering.RELAXED](reused, Int32(1))
    if reuse:
        # Prior moments are valid only for this exact immutable phase/support.
        # No cached determinant/precision: incumbent rank/control still run.
        for j in range(tid, Int(d), 256): loc.unsafe_store(c * Int(d) + j, prev_loc.unsafe_load(c * Int(d) + j))
        for j in range(tid, Int(d) * Int(d), 256): cov.unsafe_store(c * Int(d) * Int(d) + j, prev_cov.unsafe_load(c * Int(d) * Int(d) + j))
