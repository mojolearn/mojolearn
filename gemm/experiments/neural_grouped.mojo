# SPDX-License-Identifier: Apache-2.0
"""NN05/NN07: independent projections and immutable operand staging.

These are explicit neural component entry points, not public-model routing.
All numerical work is Mojo; no classical path imports this module. No build
or verification has been run. Reused I03 machinery is identified separately
from the new per-thread shared-left operand arm.
"""
from std.sys.compile import is_defined
from std.gpu import block_idx,block_dim,thread_idx
from max.gpu.host import DeviceBuffer,DeviceContext
from checks.numerics import ftz
from checks.rtf_seam import rtf_mul_add
from gemm.contract import CONTRACT_K_LEAF_MIN
from gemm.experiments.neural_profile import (
    NEURAL_EXPERIMENTS_ALLOWED,neural_partition,neural_strides,neural_validate,
    neural_profile_kernel,neural_profile_host,
)
from gemm.experiments.grouped_jobs import grouped_gemm

# OFF until full-caller combined NVIDIA/AMD evidence and all-column identity.
# I03 component wins/losses are inherited context, not these arms' outcomes.
comptime NN05 = NEURAL_EXPERIMENTS_ALLOWED and is_defined["MOJOLEARN_IDN_NEURAL_NN05"]()
comptime NN07 = NEURAL_EXPERIMENTS_ALLOWED and is_defined["MOJOLEARN_IDN_NEURAL_NN07"]()


def _neural_shared_left_kernel[JOBS: Int](
    c: MutPointer[Float32,MutAnyOrigin],a: MutPointer[Float32,MutAnyOrigin],
    b: MutPointer[Float32,MutAnyOrigin],m: Int32,n: Int32,k: Int32,
    leaf: Int32,leaves: Int32,asi: Int32,asp: Int32,bsp: Int32,bsj: Int32,
):
    comptime assert JOBS >= 1 and JOBS <= 4, "one to four independent jobs"
    var cell = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell >= Int(m)*Int(n):
        return
    var row = cell//Int(n)
    var col = cell%Int(n)
    var stack = SIMD[DType.float32,64](0.0)
    var occupied = 0
    for t in range(Int(leaves)):
        var value = SIMD[DType.float32,4](0.0)
        var begin = t*Int(leaf)
        var end = min((t+1)*Int(leaf),Int(k))
        for p in range(begin,end):
            # One global A load feeds JOBS independent contractions. No jobs'
            # k dimensions, FMA chains or fold nodes are concatenated.
            var av = ftz(a.unsafe_load(row*Int(asi)+p*Int(asp)))
            comptime for job in range(JOBS):
                var bv = ftz(b.unsafe_load(job*Int(n)*Int(k)+p*Int(bsp)+col*Int(bsj)))
                value[job] = rtf_mul_add(av,bv,value[job])
        var placed = False
        comptime for level in range(16):
            if not placed:
                if (occupied & (1 << level)) != 0:
                    comptime for job in range(JOBS):
                        value[job] = ftz(ftz(stack[job*16+level])+ftz(value[job]))
                    occupied -= 1 << level
                else:
                    comptime for job in range(JOBS):
                        stack[job*16+level] = ftz(value[job])
                    occupied += 1 << level
                    placed = True
    var result = SIMD[DType.float32,4](0.0)
    var have = False
    comptime for level in range(16):
        if (occupied & (1 << level)) != 0:
            comptime for job in range(JOBS):
                if have:
                    result[job] = ftz(ftz(stack[job*16+level])+ftz(result[job]))
                else:
                    result[job] = stack[job*16+level]
            have = True
    comptime for job in range(JOBS):
        c.unsafe_store(job*Int(m)*Int(n)+cell,ftz(result[job]))


def neural_grouped_ab[JOBS: Int = 3,CANDIDATE: Bool = False,SHARE_LEFT: Bool = False](
    ctx: DeviceContext,mut c: DeviceBuffer[DType.float32],mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],m: Int,n: Int,k: Int,op: Int,
) raises:
    """B independent launches; A1 grouped grid (I03); A2 shared-left registers.

    JOBS is a semantic projection group, not a matrix-dimension route rule.
    An actual model must identify truly independent products and retain
    separate backward contractions. Cold timing includes descriptor packing.
    """
    comptime assert JOBS >= 1 and JOBS <= 4, "one to four independent jobs"
    neural_validate(m,n,k,op)
    if len(c)<JOBS*m*n or len(a)<m*k or len(b)<JOBS*n*k:
        raise Error("neural grouped storage too short")
    if m == 0 or n == 0:
        return
    var part = neural_partition[CONTRACT_K_LEAF_MIN](k)
    var st = neural_strides(op,m,n,k)
    comptime if CANDIDATE and NN05:
        comptime if SHARE_LEFT:
            ctx.enqueue_function[_neural_shared_left_kernel[JOBS]](c,a,b,
                Int32(m),Int32(n),Int32(k),Int32(part[0]),Int32(part[1]),
                Int32(st[0]),Int32(st[1]),Int32(st[2]),Int32(st[3]),
                grid_dim=((m*n+127)//128,1,1),block_dim=(128,1,1))
            return
        else:
            # Existing I03 adapter rejects k=0. Retain defined zero products
            # through the same explicit independent-launch control below.
            if k > 0:
                grouped_gemm(c,a,b,ctx,m,n,k,op,JOBS)
                return
    comptime for job in range(JOBS):
        ctx.enqueue_function[neural_profile_kernel[1,16]](
            c.unsafe_ptr()+job*m*n,a.unsafe_ptr(),b.unsafe_ptr()+job*n*k,
            Int32(m),Int32(n),Int32(k),Int32(part[0]),Int32(part[1]),
            Int32(st[0]),Int32(st[1]),Int32(st[2]),Int32(st[3]),
            grid_dim=((m*n+127)//128,1,1),block_dim=(128,1,1))


def neural_grouped_host[JOBS: Int = 3](
    c: MutPointer[Float32,MutAnyOrigin],a: MutPointer[Float32,MutAnyOrigin],
    b: MutPointer[Float32,MutAnyOrigin],m: Int,n: Int,k: Int,op: Int,
) raises:
    comptime for job in range(JOBS):
        neural_profile_host[CONTRACT_K_LEAF_MIN,1](c+job*m*n,a,b+job*n*k,m,n,k,op)


def _neural_operand_pack_kernel(
    packed: MutPointer[Float32,MutAnyOrigin],source: MutPointer[Float32,MutAnyOrigin],
    rows: Int32,cols: Int32,row_stride: Int32,col_stride: Int32,
):
    var i = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i < Int(rows)*Int(cols):
        # Copy raw Float32 words; FTZ remains at the consuming contraction's
        # read seam, so caching never canonicalizes NaNs or signed zeros.
        packed.unsafe_store(i,source.unsafe_load((i//Int(cols))*Int(row_stride)+(i%Int(cols))*Int(col_stride)))


struct NeuralOperandStage(Movable):
    """Bounded, model-owned staging for one immutable owner/generation.

    The caller supplies an owner token unique to the operand allocation and
    increments generation after ANY mutation. One object belongs to one
    in-order DeviceContext for its entire lifetime. Cross-stream sharing is
    forbidden. A cache hit is valid only while the source owner stays alive.
    This component does not silently guess a weight's mutation generation.
    """
    var buffer: DeviceBuffer[DType.float32]
    var max_floats: Int
    var owner: Int
    var generation: Int
    var rows: Int
    var cols: Int
    var row_stride: Int
    var col_stride: Int
    var valid: Bool

    def __init__(out self,ctx: DeviceContext,max_floats: Int) raises:
        if max_floats < 1:
            raise Error("positive operand staging budget required")
        self.buffer = ctx.enqueue_create_buffer[DType.float32](1)
        self.max_floats = max_floats
        self.owner = 0
        self.generation = 0
        self.rows = 0
        self.cols = 0
        self.row_stride = 0
        self.col_stride = 0
        self.valid = False

    def prepare[CANDIDATE: Bool = False](mut self,ctx: DeviceContext,
        mut source: DeviceBuffer[DType.float32],rows: Int,cols: Int,
        row_stride: Int,col_stride: Int,owner: Int,generation: Int) raises -> Bool:
        """Return True only for reused staging; B repacks on every call.

        Producer preparation belongs inside cold/repeated measurement
        boundaries. Budget excess is explicit, never a smaller workload.
        """
        if rows<1 or cols<1 or row_stride<0 or col_stride<0:
            raise Error("invalid neural operand view")
        var last = (rows-1)*row_stride+(cols-1)*col_stride
        if rows*cols>self.max_floats or rows*cols>2147483647 or last>2147483647 or last>=len(source):
            raise Error("neural operand view exceeds storage or retained budget")
        comptime if CANDIDATE and NN07:
            if (self.valid and self.owner==owner and self.generation==generation
                and self.rows==rows and self.cols==cols
                and self.row_stride==row_stride and self.col_stride==col_stride):
                return True
        self.valid = False
        if len(self.buffer)<rows*cols:
            ctx.synchronize()
            self.buffer = ctx.enqueue_create_buffer[DType.float32](rows*cols)
        ctx.enqueue_function[_neural_operand_pack_kernel](self.buffer,source,
            Int32(rows),Int32(cols),Int32(row_stride),Int32(col_stride),
            grid_dim=((rows*cols+127)//128,1,1),block_dim=(128,1,1))
        self.owner = owner
        self.generation = generation
        self.rows = rows
        self.cols = cols
        self.row_stride = row_stride
        self.col_stride = col_stride
        self.valid = True
        return False

    def invalidate(mut self):
        self.valid = False

    def close(mut self,ctx: DeviceContext) raises:
        ctx.synchronize()
        self.valid = False
        self.buffer = ctx.enqueue_create_buffer[DType.float32](1)
