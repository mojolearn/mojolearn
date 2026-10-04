# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane apple-fast-w2-kfeat (2026-10-04): the chi2 samplers' fit entries on
the GPU binding, FAST + Apple, OPT-IN until their quality pair and M3 A/Bs.

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
Semantics: refuses iff some entry is negative by bits (`x < 0`, and a NaN
with its sign bit set), wherever it sits; main's sequential minimum can
hide a negative behind an earlier NaN (min(list) keeps the NaN). Every
NaN-free input answers as main.

`-D MOJOLEARN_XN_FAST_SCHI2_MOJO_MT` (bit 2): SkewedChi2Sampler.fit's draws
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
from x_neighbors.device_ops import BLOCK, _grid, skew_weights_kernel, xn_ctx
from x_neighbors.items import FP


comptime _KFEAT_FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
comptime XN_FAST_ACHI2_DEVSCAN = _KFEAT_FAST_APPLE and is_defined["MOJOLEARN_XN_FAST_ACHI2_DEVSCAN"]()
comptime XN_FAST_SCHI2_MOJO_MT = _KFEAT_FAST_APPLE and is_defined["MOJOLEARN_XN_FAST_SCHI2_MOJO_MT"]()
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
    """bit 1 ACHI2_DEVSCAN, bit 2 SCHI2_MOJO_MT (0 on every other build)."""
    var f = 0
    comptime if XN_FAST_ACHI2_DEVSCAN:
        f |= 1
    comptime if XN_FAST_SCHI2_MOJO_MT:
        f |= 2
    return PythonObject(f)


def kfeat_first_negative_binding(xaddr: PythonObject, n: PythonObject) raises -> PythonObject:
    """The first flat index i < n with X[i] < 0 by bits (sign set, magnitude
    nonzero), or -1: one upload into a pooled buffer, one scan, one wait."""
    var cnt = Int(py=n)
    if cnt <= 0:
        return PythonObject(-1)
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
    # math.pi / 2.0 and 2.0 * math.pi, bit for bit (both exact scalings of pi)
    var half_pi = bitcast[DType.float64](UInt64(0x3FF921FB54442D18))
    var two_pi = bitcast[DType.float64](UInt64(0x401921FB54442D18))
    with GILReleased(Python()):
        var ctx = xn_ctx()
        var st = KFEAT_STAGE.get_or_create_ptr()
        if st[].z_cap < count:
            st[].z = ctx.enqueue_create_host_buffer[DType.float32](count)
            st[].z_cap = count
        var hz = st[].z.value().unsafe_ptr()
        var mt = _MT19937(seed)
        # rs.random_sample(d * nc), then z[f][c] = pi/2 * u[f * nc + c]
        # rounded once to float32 (Array.from_list "<f4"): flat order = draw order
        for t in range(count):
            hz[t] = Float32(half_pi * mt.random())
        # rs.uniform(0.0, 2 pi, nc) = 0.0 + (2 pi - 0.0) * random(), rounded once
        for c in range(nc):
            op[c] = Float32(0.0 + two_pi * mt.random())
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
