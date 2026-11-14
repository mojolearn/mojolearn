# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and float64 references of the CNN layer checks (lane
tmig-sequence-1, 2026-11-14; ported from `test_x_cnn_conv.py`,
`test_x_cnn_pool.py`, `test_x_cnn_batchnorm.py`, `test_x_cnn_blocks.py`,
`test_x_cnn_trainer.py` and `test_cnn_padding_gradient_ownership.py`).

The references restate PyTorch's nn.Conv2d / MaxPool2d / AvgPool2d /
BatchNorm / adaptive pool semantics in float64, as the Python tests' NumPy
references did (those references were themselves held to torch by the
ORACLE tests that stay in Python). The product side is the host image's x_cnn
entries (`x_cnn/host/entry_runs_host.mojo`) reached through the RUN halves
(`x_cnn/conv_run.mojo`, `pool_run.mojo`, `bn_run.mojo`) with the same words,
pads and per-group composition the doors use (bindings/xcnn_*_door.mojo).
The inputs are seeded draws of this module's own generator (the tests drew
them from NumPy; the reference is computed from the same draws, so the
relation and the tolerances are unchanged). No `main()`: importable.
"""
from std.math import sqrt, cos, log
from core.entry_table import EntryRef
from core.xcnn_entry_table import XcnnEntryKey, xcnn_entry_ref
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_cnn.bn_run import BnSettings, bn_run_prep, bn_run_aux_head, bn_run_aux_count, bn_run_forward, bn_run_backward
from x_cnn.conv_run import (
    ConvGeom, conv_run_explicit, conv_run_conv_pad, conv_run_words, conv_run_padded, conv_run_out,
    conv_run_pad_params, conv_run_pad_code, conv_run_take_channels, conv_run_put_channels, conv_run_copy_words,
    conv_run_pad_forward, conv_run_pad_backward, conv_run_forward, conv_run_backward,
)

comptime PAD_ZEROS = 0
comptime PAD_REFLECT = 1
comptime PAD_REPLICATE = 2
comptime PAD_CIRCULAR = 3


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def cnn_entry() raises -> EntryRef:
    """The host image's x_cnn entry (published by the suite's run_checks)."""
    return xcnn_entry_ref(XcnnEntryKey(Int(GLOBAL_NUMERIC_MODE), 0))


def addr(mut values: List[Float32]) -> Int:
    return Int(values.unsafe_ptr())


def addr_i32(mut values: List[Int32]) -> Int:
    return Int(values.unsafe_ptr())


def addr_at(mut values: List[Float32], offset: Int) -> Int:
    return Int(values.unsafe_ptr() + offset)


def zeros32(n: Int) -> List[Float32]:
    return List[Float32](length=n, fill=0.0)


def prod(shape: List[Int]) -> Int:
    var n = 1
    for i in range(len(shape)):  # small-loop(tensor rank)
        n *= shape[i]
    return n


# ---------------------------------------------------------------- seeded draws

struct CheckRng(Movable):
    """splitmix64 with Box-Muller normals: the fixtures' own seeded stream."""
    var state: UInt64

    def __init__(out self, seed: Int):
        self.state = UInt64(seed) * UInt64(0x9E3779B97F4A7C15) + UInt64(0x632BE59BD9B4E019)

    def next_u64(mut self) -> UInt64:
        self.state += UInt64(0x9E3779B97F4A7C15)
        var z = self.state
        z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
        return z ^ (z >> 31)

    def uniform(mut self) -> Float64:
        """In [0, 1)."""
        return Float64(self.next_u64() >> 11) * (1.0 / 9007199254740992.0)

    def normal(mut self) -> Float64:
        var u1 = 1.0 - self.uniform()   # (0, 1]
        var u2 = self.uniform()
        return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2)

    def integer(mut self, lo: Int, hi: Int) -> Int:
        """In [lo, hi)."""
        return lo + Int(self.next_u64() % UInt64(hi - lo))


def normals(n: Int, seed: Int, scale: Float64 = 1.0, shift: Float64 = 0.0) -> List[Float32]:
    """n float32 draws of scale * N(0, 1) + shift."""
    var rng = CheckRng(seed)
    var out = List[Float32](capacity=n)
    for _ in range(n):  # small-loop(fixture draws)
        out.append(Float32(rng.normal() * scale + shift))
    return out^


def integers(n: Int, seed: Int, lo: Int, hi: Int) -> List[Float32]:
    """n float32 integers in [lo, hi) (ties inside a pool or a max)."""
    var rng = CheckRng(seed)
    var out = List[Float32](capacity=n)
    for _ in range(n):  # small-loop(fixture draws)
        out.append(Float32(rng.integer(lo, hi)))
    return out^


def widen(values: List[Float32]) -> List[Float64]:
    var out = List[Float64](capacity=len(values))
    for i in range(len(values)):  # small-loop(fixture values)
        out.append(Float64(values[i]))
    return out^


# ---------------------------------------------------------------- comparisons

def check_close(got: List[Float32], want: List[Float64], rtol: Float64, atol: Float64, what: String) raises:
    """`np.testing.assert_allclose(got, want, rtol, atol)`: |got - want| <=
    atol + rtol |want| for every word (a NaN fails)."""
    require(len(got) == len(want), what + ": length " + String(len(got)) + ", expected " + String(len(want)))
    for i in range(len(got)):  # small-loop(fixture outputs)
        var g = Float64(got[i])
        var w = want[i]
        if not (abs(g - w) <= atol + rtol * abs(w)):
            raise Error(what + ": word " + String(i) + " is " + String(g) + ", expected " + String(w))


def check_equal(got: List[Float32], want: List[Float32], what: String) raises:
    """`np.testing.assert_array_equal`: the same values word for word."""
    require(len(got) == len(want), what + ": length " + String(len(got)) + ", expected " + String(len(want)))
    for i in range(len(got)):  # small-loop(fixture outputs)
        if not (got[i] == want[i]):
            raise Error(what + ": word " + String(i) + " is " + String(got[i]) + ", expected " + String(want[i]))


# ---------------------------------------------------------------- convolution reference

def pad_source(i: Int, size: Int, before: Int, mode: Int) -> Int:
    """The source index of padded position `i` (np.pad's constant, reflect,
    edge and wrap modes); -1 for a zero pad position."""
    var j = i - before
    if mode == PAD_ZEROS:
        return j if j >= 0 and j < size else -1
    if mode == PAD_REFLECT:
        if j < 0:
            j = -j
        if j >= size:
            j = 2 * (size - 1) - j
        return j
    if mode == PAD_REPLICATE:
        return 0 if j < 0 else (size - 1 if j >= size else j)
    return ((j % size) + size) % size


@fieldwise_init
struct ConvCase(Copyable, Movable):
    """One convolution: (N, C, H, W) input, OC outputs, kernel, stride,
    explicit pads (top, bottom, left, right), dilation, groups, pad mode and
    the bias switch."""
    var n: Int
    var c: Int
    var h: Int
    var w: Int
    var oc: Int
    var kh: Int
    var kw: Int
    var sh: Int
    var sw: Int
    var pads: List[Int]
    var dh: Int
    var dw: Int
    var groups: Int
    var mode: Int
    var bias: Bool

    def out_hw(self) -> Tuple[Int, Int]:
        var hp = self.h + self.pads[0] + self.pads[1]
        var wp = self.w + self.pads[2] + self.pads[3]
        return ((hp - self.dh * (self.kh - 1) - 1) // self.sh + 1, (wp - self.dw * (self.kw - 1) - 1) // self.sw + 1)

    def x_size(self) -> Int:
        return self.n * self.c * self.h * self.w

    def w_size(self) -> Int:
        return self.oc * (self.c // self.groups) * self.kh * self.kw

    def y_size(self) -> Int:
        var o = self.out_hw()
        return self.n * self.oc * o[0] * o[1]

    def mode_name(self) -> String:
        if self.mode == PAD_REFLECT:
            return "reflect"
        if self.mode == PAD_REPLICATE:
            return "replicate"
        if self.mode == PAD_CIRCULAR:
            return "circular"
        return "zeros"


struct ConvResult(Movable):
    var y: List[Float32]
    var dx: List[Float32]
    var dw: List[Float32]
    var db: List[Float32]

    def __init__(out self, var y: List[Float32], var dx: List[Float32], var dw: List[Float32], var db: List[Float32]):
        self.y = y^
        self.dx = dx^
        self.dw = dw^
        self.db = db^


struct ConvReference(Movable):
    var y: List[Float64]
    var dx: List[Float64]
    var dw: List[Float64]
    var db: List[Float64]

    def __init__(out self, var y: List[Float64], var dx: List[Float64], var dw: List[Float64], var db: List[Float64]):
        self.y = y^
        self.dx = dx^
        self.dw = dw^
        self.db = db^


def ref_conv(cs: ConvCase, x: List[Float32], w: List[Float32], b: List[Float32], g: List[Float32]) -> ConvReference:
    """float64 out, dx (the pad's adjoint sums every padded position back onto
    its source pixel), dW and db of an explicitly padded, grouped, strided,
    dilated convolution (`ref_conv_options` of the Python test)."""
    var o = cs.out_hw()
    var oh = o[0]
    var ow = o[1]
    var cg = cs.c // cs.groups
    var og = cs.oc // cs.groups
    var y = List[Float64](length=cs.y_size(), fill=0.0)
    var dx = List[Float64](length=cs.x_size(), fill=0.0)
    var dw = List[Float64](length=cs.w_size(), fill=0.0)
    var db = List[Float64](length=cs.oc, fill=0.0)
    for n in range(cs.n):  # small-loop(reference images)
        for oc in range(cs.oc):  # small-loop(reference output channels)
            var grp = oc // og
            for i in range(oh):  # small-loop(reference output rows)
                for j in range(ow):  # small-loop(reference output columns)
                    var yi = ((n * cs.oc + oc) * oh + i) * ow + j
                    var acc = Float64(b[oc]) if cs.bias else 0.0
                    var gv = Float64(g[yi]) if len(g) > 0 else 0.0
                    db[oc] += gv
                    for ci in range(cg):  # small-loop(reference group channels)
                        var c = grp * cg + ci
                        for a in range(cs.kh):  # small-loop(kernel rows)
                            var si = pad_source(i * cs.sh + a * cs.dh, cs.h, cs.pads[0], cs.mode)
                            for bb in range(cs.kw):  # small-loop(kernel columns)
                                var sj = pad_source(j * cs.sw + bb * cs.dw, cs.w, cs.pads[2], cs.mode)
                                var wi = ((oc * cg + ci) * cs.kh + a) * cs.kw + bb
                                if si < 0 or sj < 0:
                                    continue
                                var xi = ((n * cs.c + c) * cs.h + si) * cs.w + sj
                                acc += Float64(x[xi]) * Float64(w[wi])
                                dw[wi] += gv * Float64(x[xi])
                                dx[xi] += gv * Float64(w[wi])
                    y[yi] = acc
    return ConvReference(y^, dx^, dw^, db^)


def run_conv(cs: ConvCase, mut x: List[Float32], mut w: List[Float32], mut b: List[Float32],
             mut g: List[Float32]) raises -> ConvResult:
    """The product: the host Conv2d forward and backward as the door composes
    them (bindings/xcnn_conv_native_door.mojo `xcnn_conv2d_forward_fields`,
    `_conv_backward_host`): an explicit pad entry when the mode or the pads
    ask for it, one conv entry per group over its channels and weight rows,
    then the pad's backward entry."""
    var entry = cnn_entry()
    var xs: List[Int] = [cs.n, cs.c, cs.h, cs.w]
    var pads = cs.pads.copy()
    var explicit = conv_run_explicit(cs.mode_name(), pads)
    var cp = conv_run_conv_pad(explicit, pads)
    var geom = ConvGeom(cs.groups, cs.kh, cs.kw, cs.sh, cs.sw, cp[0], cp[1], cs.dh, cs.dw, cs.oc, cs.bias)
    var ps = conv_run_padded(explicit, pads, xs)
    var pad_words = conv_run_pad_params(xs, pads, conv_run_pad_code(cs.mode_name()), -1)
    var xp = zeros32(prod(ps))
    if explicit:
        conv_run_pad_forward(entry, addr(x), addr(xp), pad_words)
    else:
        conv_run_copy_words(addr(xp), 0, addr(x), len(x))
    var prm = conv_run_words(geom, ps)
    var oshape = conv_run_out(geom, ps, xs)
    var cg = cs.c // cs.groups
    var og = cs.oc // cs.groups
    var per = cs.w_size() // cs.groups
    var gshape: List[Int] = [ps[0], cg, ps[2], ps[3]]
    var yshape: List[Int] = [oshape[0], og, oshape[2], oshape[3]]
    var y = zeros32(prod(oshape))
    var dxp = zeros32(prod(ps))
    var dw = zeros32(cs.w_size())
    var db = zeros32(cs.oc)
    for grp in range(cs.groups):  # driver-loop: over convolution groups
        var xg = zeros32(prod(gshape))
        conv_run_take_channels(addr(xg), addr(xp), ps, grp * cg, cg)
        var yg = zeros32(prod(yshape))
        conv_run_forward(entry, addr(xg), addr_at(w, grp * per), addr_at(b, grp * og), addr(yg), prm)
        conv_run_put_channels(addr(y), addr(yg), oshape, grp * og, og)
        var gg = zeros32(prod(yshape))
        conv_run_take_channels(addr(gg), addr(g), oshape, grp * og, og)
        var dxg = zeros32(prod(gshape))
        var dwg = zeros32(per)
        var dbg = zeros32(og)
        conv_run_backward(entry, addr(xg), addr_at(w, grp * per), addr(gg), addr(dxg), addr(dwg), addr(dbg), prm)
        conv_run_put_channels(addr(dxp), addr(dxg), ps, grp * cg, cg)
        conv_run_copy_words(addr(dw), grp * per, addr(dwg), per)
        conv_run_copy_words(addr(db), grp * og, addr(dbg), og)
        _ = xg^
        _ = yg^
        _ = gg^
        _ = dxg^
        _ = dwg^
        _ = dbg^
    var dx = zeros32(cs.x_size())
    if explicit:
        conv_run_pad_backward(entry, addr(dxp), addr(dx), pad_words)
    else:
        conv_run_copy_words(addr(dx), 0, addr(dxp), len(dx))
    _ = xp^
    _ = dxp^
    return ConvResult(y^, dx^, dw^, db^)


def pad_host(cs: ConvCase, x: List[Float32]) -> List[Float32]:
    """np.pad(x, the case's pads, mode) in float32 (the padding-adjoint
    test's independent pad)."""
    var hp = cs.h + cs.pads[0] + cs.pads[1]
    var wp = cs.w + cs.pads[2] + cs.pads[3]
    var out = zeros32(cs.n * cs.c * hp * wp)
    for nc in range(cs.n * cs.c):  # small-loop(fixture planes)
        for i in range(hp):  # small-loop(padded rows)
            var si = pad_source(i, cs.h, cs.pads[0], cs.mode)
            for j in range(wp):  # small-loop(padded columns)
                var sj = pad_source(j, cs.w, cs.pads[2], cs.mode)
                if si >= 0 and sj >= 0:
                    out[(nc * hp + i) * wp + j] = x[(nc * cs.h + si) * cs.w + sj]
    return out^


def pad_adjoint_host(cs: ConvCase, dxp: List[Float32]) -> List[Float32]:
    """np.add.at of the padded gradient onto its source pixels, in the padded
    array's row-major order, in float32."""
    var hp = cs.h + cs.pads[0] + cs.pads[1]
    var wp = cs.w + cs.pads[2] + cs.pads[3]
    var out = zeros32(cs.x_size())
    for nc in range(cs.n * cs.c):  # small-loop(fixture planes)
        for i in range(hp):  # small-loop(padded rows)
            var si = pad_source(i, cs.h, cs.pads[0], cs.mode)
            for j in range(wp):  # small-loop(padded columns)
                var sj = pad_source(j, cs.w, cs.pads[2], cs.mode)
                if si >= 0 and sj >= 0:
                    out[(nc * cs.h + si) * cs.w + sj] += dxp[(nc * hp + i) * wp + j]
    return out^


# ---------------------------------------------------------------- batch normalization

struct BnReference(Movable):
    """y, dx, dgamma, dbeta and the updated running mean and variance."""
    var y: List[Float64]
    var dx: List[Float64]
    var dgamma: List[Float64]
    var dbeta: List[Float64]
    var rm: List[Float64]
    var rv: List[Float64]

    def __init__(out self, var y: List[Float64], var dx: List[Float64], var dgamma: List[Float64],
                 var dbeta: List[Float64], var rm: List[Float64], var rv: List[Float64]):
        self.y = y^
        self.dx = dx^
        self.dgamma = dgamma^
        self.dbeta = dbeta^
        self.rm = rm^
        self.rv = rv^


def ref_bn(n: Int, c: Int, hw: Int, x: List[Float64], gamma: List[Float64], beta: List[Float64],
           rm: List[Float64], rv: List[Float64], training: Bool, momentum: Float64, eps: Float64,
           g: List[Float64]) -> BnReference:
    """`ref_bn` of the Python test: PyTorch's batch_norm over (N, C, HW) in
    float64 (training: batch statistics and the running update with the
    unbiased variance; eval: the running statistics). x and g are float64
    (the BasicBlock reference feeds it float64 activations)."""
    var count = n * hw
    var mean = List[Float64](length=c, fill=0.0)
    var var_ = List[Float64](length=c, fill=0.0)
    var new_rm = rm.copy()
    var new_rv = rv.copy()
    for ch in range(c):  # small-loop(reference channels)
        if training:
            var s = 0.0
            for i in range(n):  # small-loop(reference images)
                for k in range(hw):  # small-loop(reference positions)
                    s += x[(i * c + ch) * hw + k]
            var m = s / Float64(count)
            var q = 0.0
            for i in range(n):  # small-loop(reference images)
                for k in range(hw):  # small-loop(reference positions)
                    var d = x[(i * c + ch) * hw + k] - m
                    q += d * d
            mean[ch] = m
            var_[ch] = q / Float64(count)
            new_rm[ch] = (1.0 - momentum) * rm[ch] + momentum * m
            new_rv[ch] = (1.0 - momentum) * rv[ch] + momentum * var_[ch] * Float64(count) / Float64(count - 1)
        else:
            mean[ch] = rm[ch]
            var_[ch] = rv[ch]
    var total = n * c * hw
    var y = List[Float64](length=total, fill=0.0)
    var dx = List[Float64](length=total, fill=0.0)
    var dgamma = List[Float64](length=c, fill=0.0)
    var dbeta = List[Float64](length=c, fill=0.0)
    for ch in range(c):  # small-loop(reference channels)
        var invstd = 1.0 / sqrt(var_[ch] + eps)
        for i in range(n):  # small-loop(reference images)
            for k in range(hw):  # small-loop(reference positions)
                var at = (i * c + ch) * hw + k
                var xhat = (x[at] - mean[ch]) * invstd
                y[at] = xhat * gamma[ch] + beta[ch]
                dbeta[ch] += g[at]
                dgamma[ch] += g[at] * xhat
        for i in range(n):  # small-loop(reference images)
            for k in range(hw):  # small-loop(reference positions)
                var at = (i * c + ch) * hw + k
                if training:
                    var xhat = (x[at] - mean[ch]) * invstd
                    dx[at] = (g[at] - dbeta[ch] / Float64(count) - xhat * dgamma[ch] / Float64(count)) * (
                        invstd * gamma[ch])
                else:
                    dx[at] = g[at] * invstd * gamma[ch]
    return BnReference(y^, dx^, dgamma^, dbeta^, new_rm^, new_rv^)


struct BnLayer(Movable):
    """The product layer's state the door keeps (bindings/xcnn_bn_native_door.mojo):
    gamma, beta, the running block [mean | var], the batch counter and the
    last forward's aux block (eps, factor, then seven words per channel)."""
    var c: Int
    var gamma: List[Float32]
    var beta: List[Float32]
    var running: List[Float32]
    var aux: List[Float32]
    var tracked: Int
    var stats: Bool

    def __init__(out self, c: Int):
        self.c = c
        self.gamma = List[Float32](length=c, fill=1.0)
        self.beta = zeros32(c)
        self.running = zeros32(2 * c)
        for ch in range(c):  # small-loop(unit running variances)
            self.running[c + ch] = 1.0
        self.aux = zeros32(bn_run_aux_count(c))
        self.tracked = 0
        self.stats = True

    def forward(mut self, mut x: List[Float32], n: Int, hw: Int, training: Bool, track: Bool,
                has_momentum: Bool, momentum: Float64) raises -> List[Float32]:
        """`_prep` then the host forward entry; an untracked layer normalizes
        with batch statistics into a scratch running block."""
        var prep = bn_run_prep(BnSettings(self.c, 1e-5, training, track, self.tracked, has_momentum, momentum))
        self.tracked = prep.tracked
        self.stats = prep.batch_stats
        self.aux = zeros32(bn_run_aux_count(self.c))
        bn_run_aux_head(addr(self.aux), 1e-5, prep.factor)
        for ch in range(self.c):  # small-loop(gamma and beta words)
            self.aux[2 + 5 * self.c + ch] = self.gamma[ch]
            self.aux[2 + 6 * self.c + ch] = self.beta[ch]
        var y = zeros32(len(x))
        if track:
            bn_run_forward(cnn_entry(), addr(x), addr(y), addr(self.running), addr(self.aux), n, self.c, hw,
                           prep.batch_stats)
        else:
            var scratch = zeros32(2 * self.c)
            for ch in range(self.c):  # small-loop(unit running variances)
                scratch[self.c + ch] = 1.0
            bn_run_forward(cnn_entry(), addr(x), addr(y), addr(scratch), addr(self.aux), n, self.c, hw,
                           prep.batch_stats)
            _ = scratch^
        return y^

    def backward(mut self, mut x: List[Float32], mut g: List[Float32], n: Int, hw: Int) raises -> List[Float32]:
        var dx = zeros32(len(x))
        var shape: List[Int] = [n, self.c, hw]
        bn_run_backward(cnn_entry(), addr(x), addr(g), addr(dx), addr(self.aux), 0, False, shape, self.c,
                        1 if self.stats else 0)
        return dx^

    def grad_bias(self) -> List[Float32]:
        var out = zeros32(self.c)
        for ch in range(self.c):  # small-loop(channel words)
            out[ch] = self.aux[2 + 3 * self.c + ch]
        return out^

    def grad_weight(self) -> List[Float32]:
        var out = zeros32(self.c)
        for ch in range(self.c):  # small-loop(channel words)
            out[ch] = self.aux[2 + 4 * self.c + ch]
        return out^

    def running_mean(self) -> List[Float32]:
        var out = zeros32(self.c)
        for ch in range(self.c):  # small-loop(channel words)
            out[ch] = self.running[ch]
        return out^

    def running_var(self) -> List[Float32]:
        var out = zeros32(self.c)
        for ch in range(self.c):  # small-loop(channel words)
            out[ch] = self.running[self.c + ch]
        return out^
