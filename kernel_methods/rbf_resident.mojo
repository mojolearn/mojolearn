# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane apple-fast-w2-kfeat (2026-10-04): RBFSampler.fit_transform in ONE
binding call, DEFAULT in FAST + Apple (rollback
`-D MOJOLEARN_KM_FAST_RBF_RESIDENT_OFF`).

Main's fit_transform is two binding calls. `fit` makes two fresh device
buffers, waits, draws W and b, waits, and downloads each through a fresh
pinned buffer, its own wait and a serial element-by-element copy into a
List; Python then holds the words. `transform` copies W back out of Python
(`read_f32`), makes a fresh device buffer for X (88 MB at the board's
istella shape), scans it (a wait), uploads W and b through two more fresh
pinned buffers with a wait each, makes a fresh device buffer for the
100 MB projection and a fresh GEMM workspace that the FAST Apple GEMM never
reads, waits, runs the GEMM, waits, runs the epilogue, waits, and copies the
result down (a wait). About ten waits and five fresh device allocations of
up to a few hundred MB per call.

Here: W and b are drawn on the device and STAY there for the GEMM (their
words still come down once, for `random_weights_` / `random_offset_`); X
goes up from the caller's buffer into an exact-size pooled buffer
(`core/device_pool`, so the board's repeated rounds allocate nothing);
its finiteness partials are taken on the device; the GEMM takes main's
dispatch (`afn_gemm_fp32_into` when compiled, else `_fast_vendor_gemm`,
the route `identical_gemm_into` takes on FAST, so no workspace is made);
main's epilogue kernel runs in place; W, b, the partials and the result are
enqueued down in the same queue, and there is ONE wait. The same kernels in
the same order on the same words: every output bit is main's. A NaN or
infinity in X raises main's transform message after that wait (the outputs
are then never handed out).
"""
from std.ffi import _Global
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from core.device_pool import pool_give, pool_take
from core.device_scan import (
    NONFINITE_NONE,
    SCAN_BLOCKS,
    SCAN_TPB,
    _scan_blocks,
    device_classify_nonfinite,
    nonfinite_partial_kernel,
    min_partials_kernel,
)
from gemm.afn_apple_fast import AFN_GEMM_FP32_MMA, afn_gemm_fp32_into
from gemm.checks.gemm_identical import _fast_vendor_gemm, identical_gemm_into, identical_gemm_workspace_max_floats
from gemm.contract import OP_NN
from kernel_methods.checks.km_sabotage import KMSAB_NONE
from kernel_methods.checks.random_features import (
    KM_RF_TPB,
    km_feature_map_epilogue,
    km_feature_scale,
    km_random_offsets,
    km_random_weights,
    km_weight_sigma,
)
from kernel_methods.estimator import _family_ctx
from core.staged_download import download_f32_into


# KM_FAST_RBF_RESIDENT, DEFAULT in FAST + Apple (lane/apple-fast-w2-kfeat):
# M3, one run per arm: rbf-sampler istella 93.4 -> 76.3 ms; w2-kfeat-rbf-q-r1
# PASS (byte-identical outputs). MOJOLEARN_KM_FAST_RBF_RESIDENT_OFF restores
# main's two-call fit + transform.
comptime KM_FAST_RBF_RESIDENT = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_KM_FAST_RBF_RESIDENT_OFF"]()
    and not is_defined["MOJOLEARN_FAMILY_CTX_PER_CALL"]()  # the pools belong to the one context
)


# KM_FAST_RBF_STAGED, FAST+Apple DEFAULT (rollback -D MOJOLEARN_KM_FAST_RBF_STAGED_OFF;
# needs KM_FAST_RBF_RESIDENT). M3, source ea6b2035e, one run per arm:
# rbf-sampler istella 77.6 -> 57.1 ms; w2-w3kf-rbf-q PASS (byte-identical).
# Lane apple-fast-w3-kfeat, 2026-10-04: the
# projection (m x q; 100,000 x 256 = 102 MB at the board's istella shape)
# comes down through core/staged_download.mojo (`download_f32_into`: 8 MiB
# chunks DMA'd into two pooled pinned stages while one thread copies the
# other stage out into the caller's array) instead of ONE raw host-pointer
# copy, which on Apple runs at ~3 GB/s (about 21 ms per 64 MB measured on the
# M4, so ~33 ms of rbf-sampler istella's 76.3 ms). The same transport took
# x_prep's GB outputs from 563 to 345 ms (label-binarizer taxi, M3,
# X_PREP_FAST_STAGED_OUT), i.e. ~7.5 ms per 64 MB end to end. Below
# DOWNLOAD_STAGE_MIN (1M floats) the helper keeps the raw copy. A transport
# choice: the same bytes land in the same places. W, b and the scan partials
# are small and keep their raw copies, queued before the staged pipeline's
# first wait (which therefore also covers the GEMM and the epilogue).
# The output stays the caller's ordinary (mapped) memory: handing out the
# pinned stage itself would leave the caller reading write-combined memory.
comptime KM_FAST_RBF_STAGED = KM_FAST_RBF_RESIDENT and not is_defined["MOJOLEARN_KM_FAST_RBF_STAGED_OFF"]()
comptime _RBF_STAGE_POOL = "MojoKmRbfDownloadStagesFast"


struct _RbfStage(Defaultable, Movable):
    """The scan partials (device) and their pinned mirror, kept for the process."""
    var part: Optional[DeviceBuffer[DType.int32]]
    var host: Optional[HostBuffer[DType.int32]]

    def __init__(out self):
        self.part = Optional[DeviceBuffer[DType.int32]]()
        self.host = Optional[HostBuffer[DType.int32]]()


comptime _RBF_STAGE = _Global[StorageType=_RbfStage, name="MojoKmRbfResidentStageFast", init_fn=_RbfStage.__init__]


def _rbf_gemm(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """`identical_gemm_into(..., OP_NN)`'s FAST dispatch without the caller
    workspace it never reads there: the Apple FAST simdgroup route when
    compiled, then the vendor route (which serves every OP_NN shape). The
    pinned-plan fallback (unreachable on FAST OP_NN) keeps main's form,
    with its own workspace and a wait."""
    comptime if AFN_GEMM_FP32_MMA:
        if afn_gemm_fp32_into(ctx, c, a, b, m, n, k, OP_NN):
            return
    if _fast_vendor_gemm(ctx, c, a, b, m, n, k, OP_NN):
        return
    var ws = ctx.enqueue_create_buffer[DType.float32](identical_gemm_workspace_max_floats(m, n, k))
    identical_gemm_into(ctx, c, a, b, ws, m, n, k, OP_NN)
    ctx.synchronize()
    _ = ws^


def rbf_sampler_fit_transform_resident(
    d: Int,
    q: Int,
    gamma: Float32,
    seed: UInt64,
    xaddr: Int,
    m: Int,
    w_addr: Int,
    b_addr: Int,
    z_addr: Int,
) raises -> SIMD[DType.float32, 2]:
    """`RBFSampler(gamma, q, seed).fit(X).transform(X)` for X (m x d) at
    xaddr: W (d x q) to w_addr, b (q) to b_addr, the features (m x q) to
    z_addr; returns (sigma, scale). Main's refusals, texts and order: the
    fit's (n_features, n_components, gamma), then the transform's shape,
    then its NaN / infinity."""
    if d <= 0:
        raise Error("rbf_sampler_fit_host: n_features must be positive, got " + String(d))
    if q <= 0:
        raise Error(
            "rbf_sampler_fit_host: n_components must be positive, got "
            + String(q)
            + ". scikit-learn's constraint is Interval(Integral, 1, None,"
            " closed='left'). DEVIATION 1686"
        )
    if gamma != gamma or not (gamma > Float32(0.0)):
        raise Error(
            "rbf_sampler_fit_host: gamma must be POSITIVE; got a value that"
            " is not greater than zero (spelled `not (gamma > 0)` so a NaN"
            " is refused by the same test). At gamma = 0 every weight is"
            " zero and the feature map is a constant; at gamma < 0 the"
            " square root in sqrt(2 gamma) is NaN. DEVIATION 1686"
        )
    if m <= 0:
        raise Error("rbf_sampler transform X: need positive dimensions, got " + String(m) + " x " + String(d))
    if xaddr == 0 or w_addr == 0 or b_addr == 0 or z_addr == 0:
        raise Error("rbf_sampler_fit_transform: null buffer address")
    var nx = m * d
    var nz = m * q
    if nx > 2147483647 or nz > 2147483647:
        raise Error("rbf_sampler_fit_transform: more than 2^31 - 1 cells")
    var sigma = km_weight_sigma(gamma)
    var scale = km_feature_scale(q)
    var ctx = _family_ctx()
    var st = _RBF_STAGE.get_or_create_ptr()
    if not st[].part:
        # SCAN_BLOCKS partials plus one slot for their device-folded minimum
        st[].part = ctx.enqueue_create_buffer[DType.int32](SCAN_BLOCKS + 1)
        st[].host = ctx.enqueue_create_host_buffer[DType.int32](1)
    var dw = pool_take["MojoKmRbfResW"](ctx, d * q)
    var db = pool_take["MojoKmRbfResB"](ctx, q)
    var dx = pool_take["MojoKmRbfResX"](ctx, nx)
    var dz = pool_take["MojoKmRbfResZ"](ctx, nz)
    km_random_weights(ctx, dw, seed, d, q, sigma, KM_RF_TPB, KMSAB_NONE)
    km_random_offsets(ctx, db, seed, q, KM_RF_TPB)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=MutPointer[Float32, MutAnyOrigin](unsafe_from_address=xaddr))
    var blocks = _scan_blocks(nx)
    ctx.enqueue_function[nonfinite_partial_kernel](
        st[].part.value().unsafe_ptr(), dx.unsafe_ptr(), Int32(nx),
        grid_dim=(blocks, 1, 1), block_dim=(SCAN_TPB, 1, 1),
    )
    # the partials fold on the device into slot SCAN_BLOCKS: one word home
    ctx.enqueue_function[min_partials_kernel](
        st[].part.value().unsafe_ptr() + SCAN_BLOCKS, st[].part.value().unsafe_ptr(),
        Int32(blocks), grid_dim=(1, 1, 1), block_dim=(SCAN_TPB, 1, 1),
    )
    _rbf_gemm(ctx, dz, dx, dw, m, q, d)
    km_feature_map_epilogue(ctx, dz, db, m, q, scale, KM_RF_TPB, KMSAB_NONE)
    var psub = st[].part.value().create_sub_buffer[DType.int32](SCAN_BLOCKS, 1)
    ctx.enqueue_copy(dst_ptr=st[].host.value().unsafe_ptr(), src_buf=psub)
    ctx.enqueue_copy(dst_ptr=MutPointer[Float32, MutAnyOrigin](unsafe_from_address=w_addr), src_buf=dw)
    ctx.enqueue_copy(dst_ptr=MutPointer[Float32, MutAnyOrigin](unsafe_from_address=b_addr), src_buf=db)
    comptime if KM_FAST_RBF_STAGED:
        # waits inside (its first wait covers everything queued above)
        download_f32_into[_RBF_STAGE_POOL](ctx, dz, nz, MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=z_addr))
    else:
        ctx.enqueue_copy(dst_ptr=MutPointer[Float32, MutAnyOrigin](unsafe_from_address=z_addr), src_buf=dz)
        ctx.synchronize()
    _ = psub^
    var best = st[].host.value().unsafe_ptr()[0]
    var bad = -1 if best == NONFINITE_NONE else Int(best)
    var is_nan = False
    if bad >= 0:
        is_nan = device_classify_nonfinite(ctx, dx, bad)
    pool_give["MojoKmRbfResW"](dw^)
    pool_give["MojoKmRbfResB"](db^)
    pool_give["MojoKmRbfResX"](dx^)
    pool_give["MojoKmRbfResZ"](dz^)
    if bad >= 0:
        if is_nan:
            raise Error("rbf_sampler transform X: NaN at flat index " + String(bad) + "; refused by name (DEVIATION 1686)")
        raise Error("rbf_sampler transform X: infinity at flat index " + String(bad) + "; refused by name (DEVIATION 1686)")
    return SIMD[DType.float32, 2](sigma, scale)


def rbf_sampler_fit_transform_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """`RBFSampler.fit_transform(X)` under KM_FAST_RBF_RESIDENT. Returns 0.

    `addrs`: 0 weights_out (d*q f32), 1 offset_out (q f32), 2 scalars_out
    (2 f64: sigma, scale), 3 x (m*d f32, read), 4 out (m*q f32).
    `params`: 0 d, 1 q, 2 gamma, 3 seed, 4 m."""
    if len(addrs) != 5:
        raise Error("rbf_sampler_fit_transform: addrs must contain 5 addresses, got " + String(len(addrs)))
    if len(params) != 5:
        raise Error("rbf_sampler_fit_transform: params must contain 5 values, got " + String(len(params)))
    var wa = Int(py=addrs[0])
    var ba = Int(py=addrs[1])
    var sa = Int(py=addrs[2])
    var xa = Int(py=addrs[3])
    var za = Int(py=addrs[4])
    if sa == 0:
        raise Error("rbf_sampler_fit_transform: null scalars address")
    var d = Int(py=params[0])
    var q = Int(py=params[1])
    var gamma = Float32(Float64(py=params[2]))
    var seed = UInt64(Int(py=params[3]))
    var m = Int(py=params[4])
    var r = SIMD[DType.float32, 2](0, 0)
    with GILReleased(Python()):
        r = rbf_sampler_fit_transform_resident(d, q, gamma, seed, xa, m, wa, ba, za)
    var sp = MutPointer[Float64, MutAnyOrigin](unsafe_from_address=sa)
    sp.unsafe_store(0, Float64(r[0]))
    sp.unsafe_store(1, Float64(r[1]))
    return PythonObject(0)


def rbf_staged_binding() raises -> PythonObject:
    """1 when KM_FAST_RBF_STAGED is compiled in (the quality pair's reach)."""
    comptime if KM_FAST_RBF_STAGED:
        return PythonObject(1)
    return PythonObject(0)
