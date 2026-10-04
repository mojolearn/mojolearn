# SPDX-License-Identifier: Apache-2.0
"""Explicit catalog G1-G10 probe; no production dispatch imports this module."""
from std.os import abort
from std.collections import InlineArray
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.python._cpython import GILReleased
from std.ffi import _Global
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from core.gemm import gemm_nt
from core.device_zero import enqueue_fill
from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from experiments.apple_fast.gemm.mma import apple_gemm_experiment

comptime ENABLED = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and is_defined["MOJOLEARN_APPLE_GEMM_PROBE"]()
comptime FPtr = UnsafePointer[Float32, MutAnyOrigin]

struct Counters(Defaultable, Movable):
    var counts: InlineArray[Int, 11]
    def __init__(out self):
        self.counts = InlineArray[Int, 11](fill=0)

comptime COUNTERS = _Global[StorageType=Counters, name="CatalogGemmProbeCounts", init_fn=Counters.__init__]

def abi_py() raises -> PythonObject:
    return PythonObject(2)

def enabled_py() raises -> PythonObject:
    return PythonObject(Int(ENABLED))

def count_py(arm_py: PythonObject) raises -> PythonObject:
    var arm = Int(py=arm_py)
    var p = COUNTERS.get_or_create_ptr()
    if arm < 0 or arm > 10:
        raise Error("unknown catalog GEMM arm")
    return PythonObject(p[].counts[arm])

def gemm_py(aa: PythonObject, bb: PythonObject, cc: PythonObject, dims: PythonObject) raises -> PythonObject:
    comptime if not ENABLED:
        raise Error("catalog GEMM probe requires opt-in FAST Apple build")
    else:
        var m = Int(py=dims[0])
        var n = Int(py=dims[1])
        var k = Int(py=dims[2])
        var nt = Int(py=dims[3]) != 0
        var arm = Int(py=dims[4])
        var aliased_inputs = Int(py=dims[5]) != 0
        if m <= 0 or n <= 0 or k < 0 or max(m, max(n, k)) > 2147483647:
            raise Error("catalog probe invalid extent")
        if max(m * n, max(m * k, n * k)) > 2147483647:
            raise Error("catalog probe invalid or oversized shape")
        if arm < 0 or arm > 10:
            raise Error("unknown catalog GEMM arm")
        if aliased_inputs and (not nt or m != n):
            raise Error("aliased Gram requires square NT output")
        var aaddr = Int(py=aa)
        var baddr = Int(py=bb)
        var caddr = Int(py=cc)
        if aaddr == 0 or baddr == 0 or caddr == 0:
            raise Error("catalog probe null pointer")
        if (caddr < aaddr + 4 * m * k and aaddr < caddr + 4 * m * n) or (caddr < baddr + 4 * n * k and baddr < caddr + 4 * m * n):
            raise Error("catalog probe output overlaps an input")
        if aliased_inputs and aaddr != baddr:
            raise Error("aliased Gram requires one input pointer")
        var ap = FPtr(unsafe_from_address=aaddr)
        var bp = FPtr(unsafe_from_address=baddr)
        var cp = FPtr(unsafe_from_address=caddr)
        with GILReleased(Python()):
            var ctx = DeviceContext()
            var da = ctx.enqueue_create_buffer[DType.float32](max(m * k, 1))
            if k > 0:
                ctx.enqueue_copy(dst_buf=da, src_ptr=ap)
            var db = da if aliased_inputs else ctx.enqueue_create_buffer[DType.float32](max(n * k, 1))
            if not aliased_inputs and k > 0:
                ctx.enqueue_copy(dst_buf=db, src_ptr=bp)
            var dc = ctx.enqueue_create_buffer[DType.float32](m * n)
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
            _ = dc^
            _ = db^
            _ = da^
            _ = ctx^
        # GIL held: counter proves selected callable completed, not a flag alone.
        var counts = COUNTERS.get_or_create_ptr()
        counts[].counts[arm] += 1
        return PythonObject(arm)

@export
def PyInit__mojolearn_gemm_probe() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_gemm_probe")
        m.def_function[abi_py]("abi_version")
        m.def_function[enabled_py]("enabled")
        m.def_function[count_py]("count")
        m.def_function[gemm_py]("gemm")
        return m.finalize()
    except e:
        abort(String("failed to create GEMM probe: ", e))
