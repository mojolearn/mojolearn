# SPDX-License-Identifier: Apache-2.0
"""Serial diagnostic resident GEMM contract; no production imports.

prepare uploads and completes before timing. run_read launches once, downloads
and completes, but retains allocations/context. release occurs after first read.
No GEMM warmup, no pipeline precompile, no repeated launch per preparation.
"""
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.ffi import _Global
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_APPLE
from core.neural_context import process_ctx
from core.gemm import gemm_nt
from core.device_zero import enqueue_fill
from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from experiments.apple_fast.gemm.mma import apple_gemm_experiment

comptime ENABLED = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and TARGET_COLUMN == COLUMN_APPLE and not is_defined["MOJOLEARN_COLUMN_CPU"]() and is_defined["MOJOLEARN_APPLE_GEMM_RESIDENT_PROBE"]()
comptime FPtr = UnsafePointer[Float32, MutAnyOrigin]

struct ResidentState(Defaultable, Movable):
    var bufs: List[DeviceBuffer[DType.float32]]
    var counts: InlineArray[Int, 11]
    var m: Int
    var n: Int
    var k: Int
    var nt: Bool
    var used: Bool
    def __init__(out self):
        self.bufs = List[DeviceBuffer[DType.float32]]()
        self.counts = InlineArray[Int, 11](fill=0)
        self.m = 0
        self.n = 0
        self.k = 0
        self.nt = False
        self.used = False

comptime STATE = _Global[StorageType=ResidentState, name="CatalogResidentGemmState", init_fn=ResidentState.__init__]

def abi_py() raises -> PythonObject:
    return PythonObject(1)

def enabled_py() raises -> PythonObject:
    return PythonObject(Int(ENABLED))

def count_py(arm_py: PythonObject) raises -> PythonObject:
    var arm = Int(py=arm_py)
    if arm < 0 or arm > 10:
        raise Error("resident GEMM unknown arm")
    return PythonObject(STATE.get_or_create_ptr()[].counts[arm])

def prepare_py(aa: PythonObject, bb: PythonObject, dims: PythonObject) raises -> PythonObject:
    comptime if not ENABLED:
        raise Error("resident GEMM requires opt-in FAST Apple build")
    else:
        if len(dims) != 5:
            raise Error("resident GEMM expects [m,n,k,nt,alias]")
        var m = Int(py=dims[0])
        var n = Int(py=dims[1])
        var k = Int(py=dims[2])
        var nt = Int(py=dims[3]) != 0
        var aliased_inputs = Int(py=dims[4]) != 0
        if m <= 0 or n < 2 or k < 0 or max(m, max(n, k)) > 2147483647:
            raise Error("resident GEMM matrix-only contract requires m>0,n>=2,k>=0")
        if max(m * n, max(m * k, n * k)) > 2147483647:
            raise Error("resident GEMM oversized shape")
        var aaddr = Int(py=aa)
        var baddr = Int(py=bb)
        if aaddr == 0 or baddr == 0:
            raise Error("resident GEMM null input")
        if aliased_inputs and (not nt or m != n or aaddr != baddr):
            raise Error("resident GEMM alias requires one square Gram input")
        var slot = STATE.get_or_create_ptr()
        if len(slot[].bufs) != 0:
            raise Error("release prior resident buffers before preparing again")
        var ctx = process_ctx["MojoCatalogResidentGemmFast"]()
        var da = ctx.enqueue_create_buffer[DType.float32](max(m * k, 1))
        var db = da if aliased_inputs else ctx.enqueue_create_buffer[DType.float32](max(n * k, 1))
        var dc = ctx.enqueue_create_buffer[DType.float32](m * n)
        if k > 0:
            ctx.enqueue_copy(dst_buf=da, src_ptr=FPtr(unsafe_from_address=aaddr))
            if not aliased_inputs:
                ctx.enqueue_copy(dst_buf=db, src_ptr=FPtr(unsafe_from_address=baddr))
        # Output initialization is preparation, not a GEMM warmup.
        enqueue_fill(ctx, dc, Float32(-1234567.0))
        ctx.synchronize()
        slot[].bufs.append(da^)
        slot[].bufs.append(db^)
        slot[].bufs.append(dc^)
        slot[].m, slot[].n, slot[].k = m, n, k
        slot[].nt = nt
        slot[].used = False
        return PythonObject(1)

def run_read_py(arm_py: PythonObject, cc: PythonObject) raises -> PythonObject:
    comptime if not ENABLED:
        raise Error("resident GEMM requires opt-in FAST Apple build")
    else:
        var arm = Int(py=arm_py)
        var caddr = Int(py=cc)
        if arm < 0 or arm > 10 or caddr == 0:
            raise Error("resident GEMM invalid arm/output")
        var slot = STATE.get_or_create_ptr()
        if len(slot[].bufs) != 3 or slot[].used:
            raise Error("resident GEMM requires unused prepared buffers")
        # Consume token BEFORE launch: even failure cannot replay this call.
        slot[].used = True
        var m, n, k = slot[].m, slot[].n, slot[].k
        var nt = slot[].nt
        var da = slot[].bufs[0]
        var db = slot[].bufs[1]
        var dc = slot[].bufs[2]
        var ctx = process_ctx["MojoCatalogResidentGemmFast"]()
        var cp = FPtr(unsafe_from_address=caddr)
        if arm == 0:
            if k == 0:
                # Empty product contract, not an SDK zero-extent matmul.
                enqueue_fill(ctx, dc, Float32(0.0))
            elif nt:
                gemm_nt(ctx, dc, da, db, m, n, k)
            else:
                var tz = TileTensor(dc, row_major(m, n))
                var tx = TileTensor(da, row_major(m, k))
                var ty = TileTensor(db, row_major(k, n))
                matmul[target="gpu"](tz, tx, ty, ctx)
        elif arm == 1:
            if nt:
                apple_gemm_experiment[64, 64, 16, False, 0, True](ctx, dc, da, db, m, n, k)
            else:
                apple_gemm_experiment[64, 64, 16, False, 0, False](ctx, dc, da, db, m, n, k)
        elif arm == 2:
            if nt:
                apple_gemm_experiment[32, 32, 16, False, 0, True](ctx, dc, da, db, m, n, k)
            else:
                apple_gemm_experiment[32, 32, 16, False, 0, False](ctx, dc, da, db, m, n, k)
        elif arm == 3:
            if nt:
                apple_gemm_experiment[64, 128, 16, False, 0, True](ctx, dc, da, db, m, n, k)
            else:
                apple_gemm_experiment[64, 128, 16, False, 0, False](ctx, dc, da, db, m, n, k)
        elif arm == 4:
            if nt:
                apple_gemm_experiment[128, 64, 16, False, 0, True](ctx, dc, da, db, m, n, k)
            else:
                apple_gemm_experiment[128, 64, 16, False, 0, False](ctx, dc, da, db, m, n, k)
        elif arm == 5:
            if nt:
                apple_gemm_experiment[64, 64, 16, True, 0, True](ctx, dc, da, db, m, n, k)
            else:
                apple_gemm_experiment[64, 64, 16, True, 0, False](ctx, dc, da, db, m, n, k)
        elif arm == 6:
            if nt:
                apple_gemm_experiment[64, 64, 32, True, 0, True](ctx, dc, da, db, m, n, k)
            else:
                apple_gemm_experiment[64, 64, 32, True, 0, False](ctx, dc, da, db, m, n, k)
        elif arm == 7:
            if nt:
                apple_gemm_experiment[64, 64, 16, True, 4, True](ctx, dc, da, db, m, n, k)
            else:
                apple_gemm_experiment[64, 64, 16, True, 4, False](ctx, dc, da, db, m, n, k)
        elif arm == 8:
            if nt:
                apple_gemm_experiment[64, 128, 32, True, 0, True](ctx, dc, da, db, m, n, k)
            else:
                apple_gemm_experiment[64, 128, 32, True, 0, False](ctx, dc, da, db, m, n, k)
        elif arm == 9:
            if nt:
                apple_gemm_experiment[128, 64, 32, True, 0, True](ctx, dc, da, db, m, n, k)
            else:
                apple_gemm_experiment[128, 64, 32, True, 0, False](ctx, dc, da, db, m, n, k)
        elif arm == 10:
            if nt:
                apple_gemm_experiment[64, 64, 32, True, 4, True](ctx, dc, da, db, m, n, k)
            else:
                apple_gemm_experiment[64, 64, 32, True, 4, False](ctx, dc, da, db, m, n, k)
        ctx.enqueue_copy(dst_ptr=cp, src_buf=dc)
        ctx.synchronize()
        slot[].counts[arm] += 1
        return PythonObject(arm)

def release_py() raises -> PythonObject:
    var slot = STATE.get_or_create_ptr()
    while len(slot[].bufs) > 0:
        _ = slot[].bufs.pop()
    return PythonObject(1)

@export
def PyInit__mojolearn_resident_gemm_probe() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_resident_gemm_probe")
        m.def_function[abi_py]("abi_version")
        m.def_function[enabled_py]("enabled")
        m.def_function[count_py]("count")
        m.def_function[prepare_py]("prepare")
        m.def_function[run_read_py]("run_read")
        m.def_function[release_py]("release")
        return m.finalize()
    except e:
        abort(String("failed to create resident GEMM probe: ", e))
