# SPDX-License-Identifier: Apache-2.0
"""Serial quality-only probe of actual shared production dispatch entrances."""
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.python._cpython import GILReleased
from std.ffi import _Global
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from core.gemm import gemm_nt, gemm_nt_gram
from core.device_zero import enqueue_fill
from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from experiments.apple_fast.gemm.shared_dispatch import select_audit_arm, audit_count, try_shared_gemm
from gemm.checks.gemm_identical import _fast_vendor_gemm
from gemm.contract import OP_NN, OP_NT, OP_TN

comptime ENABLED = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and is_defined["MOJOLEARN_APPLE_FAST_SHARED_GEMM_AUDIT"]()
comptime FPtr = UnsafePointer[Float32, MutAnyOrigin]

struct Counters(Defaultable, Movable):
    var incumbent: Int
    var direct: Int
    var shared: Int
    def __init__(out self):
        self.incumbent = 0
        self.direct = 0
        self.shared = 0

comptime COUNTERS = _Global[StorageType=Counters, name="SharedGemmProbeCounts", init_fn=Counters.__init__]

def abi_py() raises -> PythonObject:
    return PythonObject(1)

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
        var route = Int(py=dims[6])
        if route < 0 or route > 3:
            raise Error("invalid shared GEMM route")
        if (route != 2) != nt:
            raise Error("route/transpose mismatch")
        var aliased_inputs = Int(py=dims[5]) != 0
        if m <= 0 or n <= 0 or k < 0 or max(m, max(n, k)) > 2147483647:
            raise Error("catalog probe invalid extent")
        if max(m * n, max(m * k, n * k)) > 2147483647:
            raise Error("catalog probe invalid or oversized shape")
        if arm != 0 and arm != 1 and arm != 5:
            raise Error("unknown catalog GEMM arm")
        if aliased_inputs and (not nt or m != n):
            raise Error("aliased Gram requires square NT output")
        if route == 1 and (not aliased_inputs or n == 1):
            raise Error("core Gram requires aliased input and n > 1")
        select_audit_arm(arm)
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
            if k == 0:
                # Verify shared adapter refuses empty products, then use the
                # explicit zero contract: do not feed zero extents to SDK.
                var served = False
                if route == 0:
                    served = try_shared_gemm[True, 0](ctx, dc, da, db, m, n, k)
                elif route == 1:
                    served = try_shared_gemm[True, 1](ctx, dc, da, db, m, n, k)
                elif route == 2:
                    served = try_shared_gemm[False, 2](ctx, dc, da, db, m, n, k)
                else:
                    served = try_shared_gemm[True, 3](ctx, dc, da, db, m, n, k)
                if served:
                    raise Error("shared GEMM must refuse K=0")
                enqueue_fill(ctx, dc, Float32(0.0))
            elif route == 0:
                gemm_nt(ctx, dc, da, db, m, n, k)
            elif route == 1:
                gemm_nt_gram(ctx, dc, da, m, n, k)
            else:
                if not _fast_vendor_gemm(ctx, dc, da, db, m, n, k, OP_NN if route == 2 else OP_NT):
                    raise Error("vendor entrance refused NN/NT")
            # Unsupported TN must refuse without writing or launching.
            if _fast_vendor_gemm(ctx, dc, da, db, m, n, k, OP_TN):
                raise Error("vendor entrance accepted unsupported TN")
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

def route_count_py(route_py: PythonObject, column_py: PythonObject) raises -> PythonObject:
    return PythonObject(audit_count(Int(py=route_py), Int(py=column_py)))

@export
def PyInit__mojolearn_shared_gemm_probe() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_shared_gemm_probe")
        m.def_function[abi_py]("abi_version")
        m.def_function[enabled_py]("enabled")
        m.def_function[count_py]("count")
        m.def_function[route_count_py]("route_count")
        m.def_function[gemm_py]("gemm")
        return m.finalize()
    except e:
        abort(String("failed to create GEMM probe: ", e))
