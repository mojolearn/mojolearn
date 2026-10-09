# SPDX-License-Identifier: Apache-2.0
"""NN07: immutable operand staging (NN05's independent-projection arms deleted 2026-10-08).

These are explicit neural component entry points, not public-model routing.
All numerical work is Mojo; no classical path imports this module. No build
or verification has been run.
"""
from gemm.experiments.neural_profile_device import neural_profile_kernel
from std.sys.compile import is_defined
from std.gpu import block_idx,block_dim,thread_idx
from max.gpu.host import DeviceBuffer,DeviceContext
from checks.numerics import ftz
from checks.rtf_seam import rtf_mul_add
from gemm.contract import CONTRACT_K_LEAF_MIN
from gemm.experiments.neural_profile import (
    NEURAL_EXPERIMENTS_ALLOWED,neural_partition,neural_strides,neural_validate,
    neural_profile_host,
)

# OFF until full-caller combined NVIDIA/AMD evidence and all-column identity.
# I03 component wins/losses are inherited context, not these arms' outcomes.
# TOMBSTONE NN05 (MOJOLEARN_IDN_NEURAL_NN05, neural_gemm_pair=nn05, deleted
# 2026-10-08 by lane grid-act-4): the gate/up projection pair as one kernel
# sharing each A load (_neural_pair_kernel, gemm/neural_dispatch.mojo) plus the
# shared-left / grouped A arms of neural_grouped_ab. IDENTICAL grid ge123e6f9
# nn05/off NV/AMD: lm-forward 2.19x/2.80x, lm-train-step 3.72x/2.44x,
# transformer-forward 3.85x/3.47x SLOWER; bits same. Recoverable at main 4e3da4282.
comptime NN07 = NEURAL_EXPERIMENTS_ALLOWED and is_defined["MOJOLEARN_IDN_NEURAL_NN07"]()


def neural_grouped_ab[JOBS: Int = 3](
    ctx: DeviceContext,mut c: DeviceBuffer[DType.float32],mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],m: Int,n: Int,k: Int,op: Int,
) raises:
    """B independent launches (the A1 grouped-grid and A2 shared-left arms
    were NN05's, deleted 2026-10-08; see the tombstone above).

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
