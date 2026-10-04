# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane apple-fast-w2-kfeat (2026-10-04): the chi2 samplers' fit entries on
the GPU binding, FAST + Apple. SCHI2_MOJO_MT is DEFAULT; ACHI2_DEVSCAN stays
opt-in (HOLD, see below).

`-D MOJOLEARN_XN_FAST_ACHI2_DEVSCAN` (bit 1 of `x_neighbors_kfeat_flags`):
AdditiveChi2Sampler.fit's negative check (main: `X.min() < 0`, one
sequential host pass over the 22M floats of the board's istella X, about
9 of its 12.3 ms) as ONE binding call: X copied from the caller's buffer
into a pooled device buffer (`pool_take`, exact size, so the board's rounds
allocate nothing), `core/device_scan.negative_partial_kernel` (one Int32 per
block, by bits), the at most 512 partials into a pooled pinned host buffer,
ONE synchronize, the integer minimum over the partials. The dropped
ACHI2_FAST_DEVCHECK (lane/apple-fast-gap-kapprox2, +1% istella, 0.9 -> 2.8
ms taxi) made a fresh device buffer for X and a fresh flag per call and
ran one thread per element against one flag word; this keeps nothing fresh.
HOLD (opt-in): M3, one run per arm, istella 11.6 -> 2.9 ms but taxi 0.9 ->
1.9 ms (w2-kfeat-achi2-*): the device round trip loses on small X.
lane apple-fast-w3-kfeat adds that size gate (`XN_ACHI2_DEVSCAN_MIN` =
2^22 entries, derived from those two points below): smaller X keeps main's
host minimum, so taxi keeps main's route; A/B owed before any default.
Semantics: refuses iff some entry is negative by bits (`x < 0`, and a NaN
with its sign bit set), wherever it sits; main's sequential minimum can
hide a negative behind an earlier NaN (min(list) keeps the NaN). Every
NaN-free input answers as main.

SCHI2_MOJO_MT, DEFAULT in FAST + Apple (bit 2; rollback
`-D MOJOLEARN_XN_FAST_SCHI2_MOJO_MT_OFF`). M3, one run per arm: skewed-chi2
istella 9.0 -> 2.2 ms, taxi 2.5 -> 1.5 ms; w2-kfeat-xn-q-r1 PASS
(byte-identical outputs and refusals). SkewedChi2Sampler.fit's draws
(main: `_LegacyRandomState`, numpy's legacy MT19937 stream, drawn by a
Python loop of d * n_components `random()` calls, then a nested list
comprehension for pi/2 * u and `Array.from_list`, about 6 of istella's
7.3 ms) as ONE binding call: the SAME stream (init_genrand(seed & 2^32-1),
genrand_res53) in Mojo into a pooled pinned buffer, pi/2 * u formed in IEEE
double and rounded once to float32 (main's words), uploaded once, main's
`skew_weights_kernel` on it, the weights down into the caller's array and
the offsets (0 + 2 pi * u, main's double expression, rounded once) written
directly, ONE synchronize. Bit-identical to main by construction. The
stream itself is sequential by its definition (sklearn's numbers); it is
the parameter draw, not a pass over X, and main draws it on the host too.

SCHI2_LAZYW, opt-in (bit 4, `-D MOJOLEARN_XN_FAST_SCHI2_LAZYW`, lane
apple-fast-w3-kfeat): fit without device work, the weights made inside
the first transform's single call (section at the end of this file).
"""
from std.ffi import _Global
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from core.device_pool import pool_give, pool_take
from core.device_scan import NONFINITE_NONE, SCAN_BLOCKS, SCAN_TPB, _scan_blocks, negative_partial_kernel
from x_neighbors.device_ops import BLOCK, _grid, skew_transform_kernel, skew_weights_kernel, unary_kernel, xn_ctx
from x_neighbors.items import FP, U_LOG


comptime _KFEAT_FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
comptime XN_FAST_ACHI2_DEVSCAN = _KFEAT_FAST_APPLE and is_defined["MOJOLEARN_XN_FAST_ACHI2_DEVSCAN"]()
comptime XN_FAST_SCHI2_MOJO_MT = _KFEAT_FAST_APPLE and not is_defined["MOJOLEARN_XN_FAST_SCHI2_MOJO_MT_OFF"]()
# lane apple-fast-w3-kfeat, opt-in (needs SCHI2_MOJO_MT, i.e. not its _OFF):
# see `kfeat_schi2_draw_binding`.
comptime XN_FAST_SCHI2_LAZYW = XN_FAST_SCHI2_MOJO_MT and is_defined["MOJOLEARN_XN_FAST_SCHI2_LAZYW"]()
comptime XN_KFEAT_ANY = XN_FAST_ACHI2_DEVSCAN or XN_FAST_SCHI2_MOJO_MT


struct _KfeatStage(Defaultable, Movable):
    """Pinned host buffers kept for the process: the scan partials and the
    skewed sampler's float32 uniforms (grown, never shrunk)."""
    var part: Optional[DeviceBuffer[DType.int32]]
    var host: Optional[HostBuffer[DType.int32]]
    var z: Optional[HostBuffer[DType.float32]]
    var z_cap: Int

    def __init__(out self):
        self.part = Optional[DeviceBuffer[DType.int32]]()
        self.host = Optional[HostBuffer[DType.int32]]()
        self.z = Optional[HostBuffer[DType.float32]]()
        self.z_cap = 0


comptime KFEAT_STAGE = _Global[StorageType=_KfeatStage, name="MojoXNeighborsKfeatStageFast", init_fn=_KfeatStage.__init__]


def kfeat_flags_binding() raises -> PythonObject:
    """bit 1 ACHI2_DEVSCAN, bit 2 SCHI2_MOJO_MT, bit 4 SCHI2_LAZYW (0 on
    every other build)."""
    var f = 0
    comptime if XN_FAST_ACHI2_DEVSCAN:
        f |= 1
    comptime if XN_FAST_SCHI2_MOJO_MT:
        f |= 2
    comptime if XN_FAST_SCHI2_LAZYW:
        f |= 4
    return PythonObject(f)


# lane apple-fast-w3-kfeat: ACHI2_DEVSCAN's size gate. The device scan only
# from XN_ACHI2_DEVSCAN_MIN entries; below it the binding answers -2 and
# AdditiveChi2Sampler.fit keeps main's host `X.min()` (same refusals).
# From the two M3 points (one run per arm, w2-kfeat-achi2-*), with the
# board's n = 100,000 rows: taxi d = 11 (1.1M entries) host 0.9 / device
# 1.9 ms, istella d = 220 (22.0M) host 11.6 / device 2.9 ms. Host: about
# 0.34 ms + 0.51 ms per million entries (the sequential min); device: about
# 1.85 ms + 0.048 ms per million (the fixed upload-launch-wait round trip
# dominates, the 88 MB upload is ~1 ms). They cross at ~3.3M entries. The
# gate sits at 2^22 = 4,194,304 entries (16 MiB of float32), above the
# crossing, where the model gives device 2.05 vs host 2.49 ms: a margin
# larger than the 0.2 ms the same row moved between two main runs (taxi
# 0.7 board vs 0.9 A arm), so one noisy fixed cost cannot put a small X on
# the slower route. Taxi keeps main's route and time; istella keeps the
# device scan.
comptime XN_ACHI2_DEVSCAN_MIN = 1 << 22


def kfeat_first_negative_binding(xaddr: PythonObject, n: PythonObject) raises -> PythonObject:
    """The first flat index i < n with X[i] < 0 by bits (sign set, magnitude
    nonzero), or -1: one upload into a pooled buffer, one scan, one wait.
    -2 below XN_ACHI2_DEVSCAN_MIN entries: not scanned, the caller takes
    main's host minimum."""
    var cnt = Int(py=n)
    if cnt <= 0:
        return PythonObject(-1)
    if cnt < XN_ACHI2_DEVSCAN_MIN:
        return PythonObject(-2)
    if cnt > 2147483647:
        raise Error("x_neighbors_kfeat_first_negative: more than 2^31 - 1 entries")
    var src = FP(unsafe_from_address=Int(py=xaddr))
    var best = NONFINITE_NONE
    with GILReleased(Python()):
        var ctx = xn_ctx()
        var st = KFEAT_STAGE.get_or_create_ptr()
        if not st[].part:
            st[].part = ctx.enqueue_create_buffer[DType.int32](SCAN_BLOCKS)
            st[].host = ctx.enqueue_create_host_buffer[DType.int32](SCAN_BLOCKS)
        var blocks = _scan_blocks(cnt)
        var buf = pool_take["MojoXNeighborsKfeatScan"](ctx, cnt)
        ctx.enqueue_copy(dst_buf=buf, src_ptr=src)
        ctx.enqueue_function[negative_partial_kernel](
            st[].part.value().unsafe_ptr(), buf.unsafe_ptr(), Int32(cnt),
            grid_dim=(blocks, 1, 1), block_dim=(SCAN_TPB, 1, 1),
        )
        var psub = st[].part.value().create_sub_buffer[DType.int32](0, blocks)
        ctx.enqueue_copy(dst_ptr=st[].host.value().unsafe_ptr(), src_buf=psub)
        ctx.synchronize()
        _ = psub^
        var hp = st[].host.value().unsafe_ptr()
        for i in range(blocks):
            var v = hp[i]
            if v < best:
                best = v
        pool_give["MojoXNeighborsKfeatScan"](buf^)
    return PythonObject(-1 if best == NONFINITE_NONE else Int(best))


struct _MT19937(Movable):
    """numpy's legacy RandomState(seed) stream as `_LegacyRandomState` sets
    it: init_genrand(seed), position 624 (the first draw twists), the
    standard tempering, `random()` = genrand_res53."""
    var mt: List[UInt32]
    var idx: Int

    def __init__(out self, seed: UInt32):
        self.idx = 624
        var mt = List[UInt32](capacity=624)
        mt.append(seed)
        var prev = UInt64(seed)
        for i in range(1, 624):
            var v = (UInt64(1812433253) * (prev ^ (prev >> 30)) + UInt64(i)) & UInt64(0xFFFFFFFF)
            mt.append(UInt32(v))
            prev = v
        self.mt = mt^

    def _twist(mut self):
        for k in range(624):
            var y = (self.mt[k] & UInt32(0x80000000)) | (self.mt[(k + 1) % 624] & UInt32(0x7FFFFFFF))
            var v = self.mt[(k + 397) % 624] ^ (y >> 1)
            if (y & UInt32(1)) != UInt32(0):
                v = v ^ UInt32(0x9908B0DF)
            self.mt[k] = v
        self.idx = 0

    def next_u32(mut self) -> UInt32:
        if self.idx >= 624:
            self._twist()
        var y = self.mt[self.idx]
        self.idx += 1
        y = y ^ (y >> 11)
        y = y ^ ((y << 7) & UInt32(0x9D2C5680))
        y = y ^ ((y << 15) & UInt32(0xEFC60000))
        y = y ^ (y >> 18)
        return y

    def random(mut self) -> Float64:
        """genrand_res53: (a * 2^26 + b) * 2^-53, a = u32 >> 5, b = u32 >> 6
        (every step exact in double)."""
        var a = Int(self.next_u32() >> 5)
        var b = Int(self.next_u32() >> 6)
        return (Float64(a) * 67108864.0 + Float64(b)) * (1.0 / 9007199254740992.0)


def _schi2_draw(seed: UInt32, count: Int, nc: Int, hz: FP, op: FP):
    """`_LegacyRandomState(seed)`: z[t] = pi/2 * random() for t < count
    (rs.random_sample(d * nc); flat order = draw order), rounded once to
    float32 (Array.from_list "<f4"), then the offsets 0 + 2 pi * random()
    (rs.uniform(0.0, 2 pi, nc)), rounded once. Main's words."""
    # math.pi / 2.0 and 2.0 * math.pi, bit for bit (both exact scalings of pi)
    var half_pi = bitcast[DType.float64](UInt64(0x3FF921FB54442D18))
    var two_pi = bitcast[DType.float64](UInt64(0x401921FB54442D18))
    var mt = _MT19937(seed)
    for t in range(count):
        hz[t] = Float32(half_pi * mt.random())
    for c in range(nc):
        op[c] = Float32(0.0 + two_pi * mt.random())


def kfeat_schi2_fit_binding(p: PythonObject, w_out: PythonObject, off_out: PythonObject) raises -> PythonObject:
    """p = [seed & 2^32-1, d, n_components]: `random_weights_` (d x nc, into
    the float32 array at w_out) and `random_offset_` (nc, at off_out), the
    words main's SkewedChi2Sampler.fit makes from `_LegacyRandomState(seed)`
    (module docstring). Returns 0."""
    var seed = UInt32(Int(py=p[0]) & 0xFFFFFFFF)
    var d = Int(py=p[1])
    var nc = Int(py=p[2])
    if d <= 0 or nc <= 0:
        raise Error("x_neighbors_kfeat_schi2_fit: n_features and n_components must be positive")
    var count = d * nc
    var wp = FP(unsafe_from_address=Int(py=w_out))
    var op = FP(unsafe_from_address=Int(py=off_out))
    with GILReleased(Python()):
        var ctx = xn_ctx()
        var st = KFEAT_STAGE.get_or_create_ptr()
        if st[].z_cap < count:
            st[].z = ctx.enqueue_create_host_buffer[DType.float32](count)
            st[].z_cap = count
        var hz = st[].z.value().unsafe_ptr()
        _schi2_draw(seed, count, nc, FP(unsafe_from_address=Int(hz)), op)
        var dz = pool_take["MojoXNeighborsKfeatSkewZ"](ctx, count)
        var dw = pool_take["MojoXNeighborsKfeatSkewW"](ctx, count)
        ctx.enqueue_copy(dst_buf=dz, src_ptr=hz)
        ctx.enqueue_function[skew_weights_kernel](
            dz.unsafe_ptr(), dw.unsafe_ptr(), Int64(count),
            grid_dim=_grid(count), block_dim=(BLOCK if count > 1 else 1),
        )
        ctx.enqueue_copy(dst_ptr=wp, src_buf=dw)
        ctx.synchronize()
        pool_give["MojoXNeighborsKfeatSkewZ"](dz^)
        pool_give["MojoXNeighborsKfeatSkewW"](dw^)
    return PythonObject(0)


# ---------------------------------------------------------------- SCHI2_LAZYW
# lane apple-fast-w3-kfeat (2026-10-04), opt-in `-D MOJOLEARN_XN_FAST_SCHI2_LAZYW`
# (bit 4; needs SCHI2_MOJO_MT). After SCHI2_MOJO_MT the board's skewed-chi2
# taxi fit (d = 11, 256 components: 2,816 weights) is 1.5 ms against
# sklearn's 0.6; what is left is not the draw (microseconds) but the one
# GPU round trip that turns 11 KB of uniforms into the weights (upload,
# skew_weights_kernel, download, wait), and `transform` then pays two more
# (main's `_unary` log of X down to the host and back up, then
# `skew_transform`). Here:
#   fit       draws z = pi/2 * u (float32, main's words) and the offsets on
#             the host into the estimator's own arrays: NO device work;
#   transform ONE call, ONE wait: X up into a pooled buffer, main's
#             `unary_kernel` (log(x + skewedness)) into a pooled buffer that
#             stays on the device, the weights from z by main's
#             `skew_weights_kernel` on the first transform (downloaded
#             once into `random_weights_`; uploaded as words afterwards),
#             main's `skew_transform_kernel`, the result down;
#   random_weights_ read before any transform: one call (z up, kernel,
#             down, wait), the same words.
# Same kernels on the same words in the same order: every bit is main's.
# Fit + transform go from three waits to one.


def kfeat_schi2_draw_binding(p: PythonObject, z_out: PythonObject, off_out: PythonObject) raises -> PythonObject:
    """p = [seed & 2^32-1, d, n_components]: z (d * nc float32 at z_out, the
    words main hands `skew_weights_kernel`) and `random_offset_` (nc at
    off_out). Host only: the draw is sklearn's sequential stream."""
    var seed = UInt32(Int(py=p[0]) & 0xFFFFFFFF)
    var d = Int(py=p[1])
    var nc = Int(py=p[2])
    if d <= 0 or nc <= 0:
        raise Error("x_neighbors_kfeat_schi2_draw: n_features and n_components must be positive")
    var zp = FP(unsafe_from_address=Int(py=z_out))
    var op = FP(unsafe_from_address=Int(py=off_out))
    with GILReleased(Python()):
        _schi2_draw(seed, d * nc, nc, zp, op)
    return PythonObject(0)


def kfeat_schi2_weights_binding(z_addr: PythonObject, w_addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """`random_weights_` (n floats at w_addr) from z (n floats at z_addr):
    main's `skew_weights_kernel`, one wait. Returns 0."""
    var count = Int(py=n)
    if count <= 0:
        raise Error("x_neighbors_kfeat_schi2_weights: empty weights")
    var zp = FP(unsafe_from_address=Int(py=z_addr))
    var wp = FP(unsafe_from_address=Int(py=w_addr))
    with GILReleased(Python()):
        var ctx = xn_ctx()
        var dz = pool_take["MojoXNeighborsKfeatSkewZ"](ctx, count)
        var dw = pool_take["MojoXNeighborsKfeatSkewW"](ctx, count)
        ctx.enqueue_copy(dst_buf=dz, src_ptr=zp)
        ctx.enqueue_function[skew_weights_kernel](
            dz.unsafe_ptr(), dw.unsafe_ptr(), Int64(count),
            grid_dim=_grid(count), block_dim=(BLOCK if count > 1 else 1),
        )
        ctx.enqueue_copy(dst_ptr=wp, src_buf=dw)
        ctx.synchronize()
        pool_give["MojoXNeighborsKfeatSkewZ"](dz^)
        pool_give["MojoXNeighborsKfeatSkewW"](dw^)
    return PythonObject(0)


def kfeat_schi2_transform_binding(p: PythonObject, a: PythonObject, f: PythonObject) raises -> PythonObject:
    """SkewedChi2Sampler.transform under SCHI2_LAZYW. p = [n, d, nc, pending];
    a = [x (n*d), wz, w_out, off (nc), out (n*nc)]; f = [skewedness as
    float32]. pending = 1: wz is z, the weights are made here and written
    to w_out; 0: wz is `random_weights_` (w_out unused). The caller has
    made main's `X.min() <= -skewedness` refusal. Returns 0."""
    var n = Int(py=p[0])
    var d = Int(py=p[1])
    var nc = Int(py=p[2])
    var pending = Int(py=p[3]) != 0
    if n <= 0 or d <= 0 or nc <= 0:
        raise Error("x_neighbors_kfeat_schi2_transform: need positive n, d, n_components")
    var nx = n * d
    var nw = d * nc
    var nz = n * nc
    if nx > 2147483647 or nz > 2147483647:
        raise Error("x_neighbors_kfeat_schi2_transform: more than 2^31 - 1 cells")
    var xp = FP(unsafe_from_address=Int(py=a[0]))
    var wzp = FP(unsafe_from_address=Int(py=a[1]))
    var wop = FP(unsafe_from_address=Int(py=a[2]))
    var offp = FP(unsafe_from_address=Int(py=a[3]))
    var outp = FP(unsafe_from_address=Int(py=a[4]))
    var skew = Float32(Float64(py=f[0]))
    with GILReleased(Python()):
        var ctx = xn_ctx()
        var dx = pool_take["MojoXNeighborsKfeatSkewTX"](ctx, nx)
        var dl = pool_take["MojoXNeighborsKfeatSkewTL"](ctx, nx)
        var dw = pool_take["MojoXNeighborsKfeatSkewW"](ctx, nw)
        var doff = pool_take["MojoXNeighborsKfeatSkewTO"](ctx, nc)
        var dres = pool_take["MojoXNeighborsKfeatSkewTR"](ctx, nz)
        ctx.enqueue_copy(dst_buf=dx, src_ptr=xp)
        # main's `_unary(X, _U_LOG, 1.0, skewedness)`: log(1 * x + skewedness)
        ctx.enqueue_function[unary_kernel](
            dx.unsafe_ptr(), dl.unsafe_ptr(), Int64(nx), Int64(U_LOG), Float32(1.0), skew,
            grid_dim=_grid(nx), block_dim=(BLOCK if nx > 1 else 1),
        )
        if pending:
            var dz = pool_take["MojoXNeighborsKfeatSkewZ"](ctx, nw)
            ctx.enqueue_copy(dst_buf=dz, src_ptr=wzp)
            ctx.enqueue_function[skew_weights_kernel](
                dz.unsafe_ptr(), dw.unsafe_ptr(), Int64(nw),
                grid_dim=_grid(nw), block_dim=(BLOCK if nw > 1 else 1),
            )
            ctx.enqueue_copy(dst_ptr=wop, src_buf=dw)
            ctx.enqueue_copy(dst_buf=doff, src_ptr=offp)
            ctx.enqueue_function[skew_transform_kernel](
                dl.unsafe_ptr(), dw.unsafe_ptr(), doff.unsafe_ptr(), dres.unsafe_ptr(), Int64(n), Int64(d), Int64(nc),
                grid_dim=_grid(nz), block_dim=(BLOCK if nz > 1 else 1),
            )
            ctx.enqueue_copy(dst_ptr=outp, src_buf=dres)
            ctx.synchronize()
            pool_give["MojoXNeighborsKfeatSkewZ"](dz^)
        else:
            ctx.enqueue_copy(dst_buf=dw, src_ptr=wzp)
            ctx.enqueue_copy(dst_buf=doff, src_ptr=offp)
            ctx.enqueue_function[skew_transform_kernel](
                dl.unsafe_ptr(), dw.unsafe_ptr(), doff.unsafe_ptr(), dres.unsafe_ptr(), Int64(n), Int64(d), Int64(nc),
                grid_dim=_grid(nz), block_dim=(BLOCK if nz > 1 else 1),
            )
            ctx.enqueue_copy(dst_ptr=outp, src_buf=dres)
            ctx.synchronize()
        pool_give["MojoXNeighborsKfeatSkewTX"](dx^)
        pool_give["MojoXNeighborsKfeatSkewTL"](dl^)
        pool_give["MojoXNeighborsKfeatSkewW"](dw^)
        pool_give["MojoXNeighborsKfeatSkewTO"](doff^)
        pool_give["MojoXNeighborsKfeatSkewTR"](dres^)
    return PythonObject(0)
