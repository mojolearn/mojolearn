# SPDX-License-Identifier: Apache-2.0
"""Explicit catalog G1/G5 probe; no production dispatch imports this module."""
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.python._cpython import GILReleased
from std.ffi import _Global
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from core.gemm import gemm_nt
from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from experiments.apple_fast.gemm.mma import apple_gemm_experiment

comptime ENABLED = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and is_defined["MOJOLEARN_APPLE_GEMM_PROBE"]()
comptime FPtr = UnsafePointer[Float32, MutAnyOrigin]

struct Counters(Defaultable, Movable):
    var incumbent: Int
    var direct: Int
    var shared: Int
    def __init__(out self):
        self.incumbent = 0
        self.direct = 0
        self.shared = 0

comptime COUNTERS = _Global[StorageType=Counters, name="CatalogGemmProbeCounts", init_fn=Counters.__init__]

def enabled_py() raises -> PythonObject:
    return PythonObject(Int(ENABLED))

def count_py(arm_py: PythonObject) raises -> PythonObject:
    var arm = Int(py=arm_py)
    var p = COUNTERS.get_or_create_ptr()
    if arm == 0:
        return PythonObject(p[].incumbent)
    if arm == 1:
        return PythonObject(p[].direct)
    if arm == 5:
        return PythonObject(p[].shared)
    raise Error("unknown catalog GEMM arm")

def gemm_py(aa: PythonObject, bb: PythonObject, cc: PythonObject, dims: PythonObject) raises -> PythonObject:
    comptime if not ENABLED:
        raise Error("catalog GEMM probe requires opt-in FAST Apple build")
    else:
        var m = Int(py=dims[0])
        var n = Int(py=dims[1])
        var k = Int(py=dims[2])
        var nt = Int(py=dims[3]) != 0
        var arm = Int(py=dims[4])
        var alias = Int(py=dims[5]) != 0
        if m <= 0 or n <= 0 or k < 0 or max(m * n, max(m * k, n * k)) > 2147483647:
            raise Error("catalog probe invalid or oversized shape")
        if arm != 0 and arm != 1 and arm != 5:
            raise Error("unknown catalog GEMM arm")
        if alias and (not nt or m != n):
            raise Error("aliased Gram requires square NT output")
        var aaddr = Int(py=aa)
        var baddr = Int(py=bb)
        var caddr = Int(py=cc)
        if aaddr == 0 or baddr == 0 or caddr == 0:
            raise Error("catalog probe null pointer")
        if (caddr < aaddr + 4 * m * k and aaddr < caddr + 4 * m * n) or (caddr < baddr + 4 * n * k and baddr < caddr + 4 * m * n):
            raise Error("catalog probe output overlaps an input")
        if alias and aaddr != baddr:
            raise Error("aliased Gram requires one input pointer")
        var ap = FPtr(unsafe_from_address=aaddr)
        var bp = FPtr(unsafe_from_address=baddr)
        var cp = FPtr(unsafe_from_address=caddr)
        with GILReleased(Python()):
            var ctx = DeviceContext()
            var da = ctx.enqueue_create_buffer[DType.float32](max(m * k, 1))
            ctx.enqueue_copy(dst_buf=da, src_ptr=ap)
            var db = da if alias else ctx.enqueue_create_buffer[DType.float32](max(n * k, 1))
            if not alias:
                ctx.enqueue_copy(dst_buf=db, src_ptr=bp)
            var dc = ctx.enqueue_create_buffer[DType.float32](m * n)
            if arm == 0:
                if nt and k > 0:
                    gemm_nt(ctx, dc, da, db, m, n, k)
                else:
                    var tz = TileTensor(dc, row_major(m, n))
                    var tx = TileTensor(da, row_major(m, k))
                    if nt:
                        var ty = TileTensor(db, row_major(n, k))
                        matmul[transpose_b=True, target="gpu"](tz, tx, ty, ctx)
                    else:
                        var ty = TileTensor(db, row_major(k, n))
                        matmul[target="gpu"](tz, tx, ty, ctx)
            elif arm == 1:
                if nt:
                    apple_gemm_experiment[64, 64, 16, False, 0, True](ctx, dc, da, db, m, n, k)
                else:
                    apple_gemm_experiment[64, 64, 16, False, 0, False](ctx, dc, da, db, m, n, k)
            else:
                if nt:
                    apple_gemm_experiment[64, 64, 16, True, 0, True](ctx, dc, da, db, m, n, k)
                else:
                    apple_gemm_experiment[64, 64, 16, True, 0, False](ctx, dc, da, db, m, n, k)
            ctx.enqueue_copy(dst_ptr=cp, src_buf=dc)
            ctx.synchronize()
            _ = dc^
            _ = db^
            _ = da^
            _ = ctx^
        # GIL held: counter proves selected callable completed, not a flag alone.
        var counts = COUNTERS.get_or_create_ptr()
        if arm == 0:
            counts[].incumbent += 1
        elif arm == 1:
            counts[].direct += 1
        else:
            counts[].shared += 1
        return PythonObject(arm)

@export
def PyInit__mojolearn_gemm_probe() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_gemm_probe")
        m.def_function[enabled_py]("enabled")
        m.def_function[count_py]("count")
        m.def_function[gemm_py]("gemm")
        return m.finalize()
    except e:
        abort(String("failed to create GEMM probe: ", e))
