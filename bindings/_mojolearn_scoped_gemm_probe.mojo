# SPDX-License-Identifier: Apache-2.0
"""Quality-only bindings to actual decomp GEMM and PCA covariance entrances."""
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE
from experiments.apple_fast.gemm.scoped_dispatch import (
    AUDIT, ENABLED, TALL, DENSE, GRAM, NARROW, SPLITS, PCA,
    scoped_count, scoped_last,
)
from x_decomp.device import launch_gemm, DECOMP_FAST_GEMM_MMA
from decomposition.impl.linalg.detail.pca import compute_covariance

comptime FPtr = UnsafePointer[Float32, MutAnyOrigin]

def abi_py() raises -> PythonObject:
    return PythonObject(1)

def flags_py() raises -> PythonObject:
    return PythonObject(Int(TALL) + 2 * Int(DENSE) + 4 * Int(GRAM) + 8 * Int(NARROW) + 16 * Int(SPLITS) + 32 * Int(PCA))

def enabled_py() raises -> PythonObject:
    return PythonObject(Int(ENABLED and AUDIT and DECOMP_FAST_GEMM_MMA))

def mode_py() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))

def route_count_py(route: PythonObject, arm: PythonObject) raises -> PythonObject:
    return PythonObject(scoped_count(Int(py=route), Int(py=arm)))

def metadata_py(index: PythonObject) raises -> PythonObject:
    return PythonObject(scoped_last(Int(py=index)))

def gemm_py(aa: PythonObject, bb: PythonObject, cc: PythonObject, dims: PythonObject) raises -> PythonObject:
    comptime if not (ENABLED and AUDIT and DECOMP_FAST_GEMM_MMA):
        raise Error("scoped probe requires FAST Apple and scoped AUDIT")
    else:
        if len(dims) != 6:
            raise Error("expected m,n,k,ta,tb,alias")
        var m = Int(py=dims[0])
        var n = Int(py=dims[1])
        var k = Int(py=dims[2])
        var ta = Int(py=dims[3]) != 0
        var tb = Int(py=dims[4]) != 0
        var aliased_inputs = Int(py=dims[5]) != 0
        if m < 1 or n < 1 or k < 1 or max(m * n, max(m * k, n * k)) > 2147483647:
            raise Error("positive bounded extents required")
        if aliased_inputs and (m != n or ta == tb or Int(py=aa) != Int(py=bb)):
            raise Error("alias requires one square Gram input")
        var ap = FPtr(unsafe_from_address=Int(py=aa))
        var bp = FPtr(unsafe_from_address=Int(py=bb))
        var cp = FPtr(unsafe_from_address=Int(py=cc))
        var ctx = DeviceContext()
        var da = ctx.enqueue_create_buffer[DType.float32](m * k)
        ctx.enqueue_copy(dst_buf=da, src_ptr=ap)
        var db = da if aliased_inputs else ctx.enqueue_create_buffer[DType.float32](n * k)
        if not aliased_inputs:
            ctx.enqueue_copy(dst_buf=db, src_ptr=bp)
        var dc = ctx.enqueue_create_buffer[DType.float32](m * n)
        # This probe requires the existing AFN default, which does not use scratch.
        var scratch = ctx.enqueue_create_buffer[DType.float32](1)
        launch_gemm(ctx,
            da.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            db.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            dc.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            scratch.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), m, k, n, ta, tb,
        )
        ctx.enqueue_copy(dst_ptr=cp, src_buf=dc)
        ctx.synchronize()
        return PythonObject(1)

def covariance_py(xx: PythonObject, cc: PythonObject, mm: PythonObject, after: PythonObject, dims: PythonObject) raises -> PythonObject:
    comptime if not (ENABLED and AUDIT and DECOMP_FAST_GEMM_MMA):
        raise Error("scoped probe requires FAST Apple and scoped AUDIT")
    else:
        if len(dims) != 3:
            raise Error("expected rows,cols,restore")
        var nr = Int(py=dims[0])
        var nc = Int(py=dims[1])
        var restore = Int(py=dims[2]) != 0
        if nr < 2 or nc < 1 or max(nr * nc, nc * nc) > 2147483647:
            raise Error("invalid covariance shape")
        var ctx = DeviceContext()
        var dx = ctx.enqueue_create_buffer[DType.float32](nr * nc)
        ctx.enqueue_copy(dst_buf=dx, src_ptr=FPtr(unsafe_from_address=Int(py=xx)))
        # These are independent transpose/partial-sum scratch buffers.
        var xa = ctx.enqueue_create_buffer[DType.float32](nr * nc)
        var xb = ctx.enqueue_create_buffer[DType.float32](nr * nc)
        var mu = ctx.enqueue_create_buffer[DType.float32](nc)
        var cov = ctx.enqueue_create_buffer[DType.float32](nc * nc)
        compute_covariance(ctx, dx, xa, xb, mu, cov, nr, nc, restore)
        ctx.enqueue_copy(dst_ptr=FPtr(unsafe_from_address=Int(py=cc)), src_buf=cov)
        ctx.enqueue_copy(dst_ptr=FPtr(unsafe_from_address=Int(py=mm)), src_buf=mu)
        ctx.enqueue_copy(dst_ptr=FPtr(unsafe_from_address=Int(py=after)), src_buf=dx)
        ctx.synchronize()
        return PythonObject(1)

@export
def PyInit__mojolearn_scoped_gemm_probe() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_scoped_gemm_probe")
        m.def_function[abi_py]("abi_version")
        m.def_function[flags_py]("flags")
        m.def_function[enabled_py]("enabled")
        m.def_function[mode_py]("numeric_mode")
        m.def_function[route_count_py]("route_count")
        m.def_function[metadata_py]("metadata")
        m.def_function[gemm_py]("gemm")
        m.def_function[covariance_py]("covariance")
        return m.finalize()
    except e:
        abort(String("failed scoped GEMM probe: ", e))
