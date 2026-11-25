# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""Session.interpolate, pixel_shuffle, pixel_unshuffle and pad(mode='reflect' | 'replicate') (lane
session-ops-resize, 2026-11-25; session codes 150..153, autograd/session_ops_resize.mojo), through the call the
Python door makes (`OpsSession.op(code, [x], ints, floats)`).

test_session_ops_resize_cases, per case (shapes as small as the edge they exercise):
  - SEQUENTIAL'S BITS: where Sequential has the same layer (every case but the per-axis scale and the recomputed
    scale), the Session op's forward words and x's gradient words equal those of Sequential's own layer
    (autograd/nn_conv_blocks.mojo nnc_forward, layer codes 63..66, differentiated with ModelRules) on the same
    input and seed;
  - the forward against a float64 oracle written here from PyTorch's definitions, |a - e| <= 2e-6 (1 + |e|);
  - x's gradient of sum(seed * y) (the recording session's scalar backward through LayerRules) against central
    finite differences of the oracle's same loss, every element probed (h = 1/64; the inputs are distinct
    multiples of 1/16, every op is linear in x), |g - fd| <= 2e-5 (1 + |fd|);
  - the seeded backward (Session.vjp with the seed: backward_seed through LayerRules) gives the scalar
    backward's words;
  - the inference session's forward equals the recording session's words;
  - one `SESSION_OPS_RESIZE <case> digest` line over the forward and gradient words.
  Cases: interpolate nearest x2 (N, C, H, W); bilinear x1.5 (unequal height weights); bilinear align_corners to
  size (5, 7); linear x2 over (N, C, L) (Sequential's (N, C, 1, L) view); nearest down to size (3, 4); bilinear
  per-axis scale (2, 1.5) (Session only); bilinear x1.5 with recompute_scale_factor (Session only, the scale from
  the sizes); pixel_shuffle r2; pixel_unshuffle r2; pad reflect 2-D (left 2 = W - 2, top 1, bottom 2 = H - 1);
  pad reflect 1-D (left 4 = L - 1, the largest count); pad replicate 2-D (left 3 > W); pad replicate 1-D.
test_session_ops_resize_refusals: each refused with its words (linear on 4-D, bilinear on 3-D, a 5-D input,
  align_corners with nearest, size and scale both, a per-axis size of the wrong length, a shuffle factor that
  does not divide the channels, an unshuffle factor that does not divide H and W, a reflection count equal to
  the axis, a pad on the channel axis, a negative replicate count).
Columns: the CPU column runs every case on HostExec. The device column runs every case on DeviceExec AND
HostExec and requires the same words (forward and gradient).
Fault injection (tools/identity_lanes/coverage_sabotage.tsv, lane session-ops-resize's arms): the bilinear
height weights swapped makes the bilinear cases FAIL against the oracle on the column the arm is built for; the
reflection's outermost left column at +0.0 makes the reflect cases FAIL the same way.
Serial by nature: each case opens sessions through the process-wide session table (autograd/ops_api.mojo
`_OPS_SESSIONS`, unlocked lists), as core_ops_session_blocks says; the cases are a few hundred words each.
"""
from std.math import isfinite
from std.memory import bitcast
from std.testing import TestSuite
from autograd.ops_api import OpsSession, ops_session, ops_session_open, ops_session_close, OPS_HOST, OPS_DEVICE
from autograd.session_ops_resize import OPS_INTERPOLATE, OPS_PIXEL_SHUFFLE, OPS_PIXEL_UNSHUFFLE, OPS_PAD_MODE
from autograd.tape import Tape, Tensor
from autograd.rankn_api import rn_shape
from autograd.model_blocks import ModelRules
from autograd.nn_conv_blocks import nnc_forward
from sequence.ops import dummy_ptr
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.ops_higher_common import tensor, total, values, gradient, same_words, op2
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD, check_cases_serial, require

comptime _MUL = 4
comptime _H = Float64(1.0) / Float64(64)

comptime K_INTERP = 0
comptime K_SHUFFLE = 1
comptime K_UNSHUFFLE = 2
comptime K_PAD = 3

# Sequential layer codes (autograd/nn_conv_blocks.mojo)
comptime L_REFLECTION_PAD = 63
comptime L_REPLICATION_PAD = 64
comptime L_UPSAMPLE = 65
comptime L_PIXEL_SHUFFLE = 66


struct _Case(Copyable, Movable):
    var name: String
    var kind: Int
    var x: List[Int]          # (N, C, L) or (N, C, H, W)
    var ints: List[Int]       # the Session call's words
    var fs: List[Float32]     # the Session call's floats
    var seq: Bool             # Sequential has the same layer: compare words
    var seq_code: Int
    var xargs: List[Int]      # Sequential's six argument words
    var f: Float32            # Sequential's float argument

    def __init__(out self, name: String, kind: Int, x: List[Int], ints: List[Int], fs: List[Float32]):
        self.name = name
        self.kind = kind
        self.x = x.copy()
        self.ints = ints.copy()
        self.fs = fs.copy()
        self.seq = False
        self.seq_code = 0
        self.xargs = List[Int]()
        self.f = Float32(0)


def _interp(name: String, x: List[Int], mode: Int, align: Int, recompute: Int, size_kind: Int, s0: Int, s1: Int,
            scale_kind: Int, f0: Float32, f1: Float32, seq: Bool) -> _Case:
    var ints: List[Int] = [mode, align, recompute, size_kind, s0, s1, scale_kind]
    var fs: List[Float32] = [f0, f1]
    var c = _Case(name, K_INTERP, x, ints, fs)
    if seq:
        c.seq = True
        c.seq_code = L_UPSAMPLE
        var two = len(x) == 4
        var oh = s0 if (size_kind != 0 and two) else 0
        var ow = (s1 if two else s0) if size_kind != 0 else 0
        c.xargs = [oh, ow, mode, align, 0, 0]
        c.f = f0 if scale_kind != 0 else Float32(0)
    return c^


def _shuffle(name: String, x: List[Int], r: Int, unshuffle: Bool) -> _Case:
    var ints: List[Int] = [r]
    var c = _Case(name, K_UNSHUFFLE if unshuffle else K_SHUFFLE, x, ints, List[Float32]())
    c.seq = True
    c.seq_code = L_PIXEL_SHUFFLE
    c.xargs = [r, 1 if unshuffle else 0, 0, 0, 0, 0]
    return c^


def _pad(name: String, x: List[Int], mode: Int, left: Int, right: Int, top: Int, bottom: Int) -> _Case:
    # Session.pad's ONNX order: every axis's count before, then every axis's count after
    var nd = len(x)
    var ints: List[Int] = [mode]
    for k in range(nd):  # small-loop(rank)
        ints.append(left if k == nd - 1 else (top if (k == 2 and nd == 4) else 0))
    for k in range(nd):  # small-loop(rank)
        ints.append(right if k == nd - 1 else (bottom if (k == 2 and nd == 4) else 0))
    var c = _Case(name, K_PAD, x, ints, List[Float32]())
    c.seq = True
    c.seq_code = L_REFLECTION_PAD if mode == 1 else L_REPLICATION_PAD
    c.xargs = [left, right, top, bottom, nd - 2, 0]
    return c^


def _cases() -> List[_Case]:
    var out = List[_Case]()
    out.append(_interp("interpolate nearest x2", [1, 2, 2, 3], 0, 0, 0, 0, 0, 0, -1, 2.0, 2.0, True))
    out.append(_interp("interpolate bilinear x1.5", [1, 1, 3, 4], 2, 0, 0, 0, 0, 0, -1, 1.5, 1.5, True))
    out.append(_interp("interpolate bilinear align size 5x7", [1, 1, 3, 4], 2, 1, 0, 2, 5, 7, 0, 0.0, 0.0, True))
    out.append(_interp("interpolate linear 1d x2", [1, 2, 5], 1, 0, 0, 0, 0, 0, -1, 2.0, 2.0, True))
    out.append(_interp("interpolate nearest down to 3x4", [1, 1, 4, 6], 0, 0, 0, 2, 3, 4, 0, 0.0, 0.0, True))
    out.append(_interp("interpolate bilinear per-axis 2 x 1.5", [1, 1, 2, 3], 2, 0, 0, 0, 0, 0, 2, 2.0, 1.5, False))
    out.append(_interp("interpolate bilinear x1.5 recompute", [1, 1, 3, 4], 2, 0, 1, 0, 0, 0, -1, 1.5, 1.5, False))
    out.append(_shuffle("pixel_shuffle r2", [1, 8, 2, 2], 2, False))
    out.append(_shuffle("pixel_unshuffle r2", [1, 2, 4, 2], 2, True))
    out.append(_pad("pad reflect 2d", [1, 2, 3, 4], 1, 2, 1, 1, 2))
    out.append(_pad("pad reflect 1d", [1, 1, 5], 1, 4, 1, 0, 0))
    out.append(_pad("pad replicate 2d", [1, 1, 2, 3], 2, 3, 0, 0, 2))
    out.append(_pad("pad replicate 1d", [1, 2, 3], 2, 1, 2, 0, 0))
    return out^


def _count(dims: List[Int]) -> Int:
    var n = 1
    for d in dims:  # small-loop(rank)
        n *= d
    return n


def _code(c: _Case) -> Int:
    if c.kind == K_INTERP:
        return OPS_INTERPOLATE
    if c.kind == K_SHUFFLE:
        return OPS_PIXEL_SHUFFLE
    if c.kind == K_UNSHUFFLE:
        return OPS_PIXEL_UNSHUFFLE
    return OPS_PAD_MODE


def _distinct(n: Int, salt: Int) -> List[Float32]:
    """Distinct multiples of 1/16 (a permutation of 0..n-1 for a step coprime with n)."""
    var step = 37 + 2 * salt
    while _gcd(step, n) != 1:
        step += 2
    var v = List[Float32](length=n, fill=0)
    for i in range(n):  # small-loop(check data)
        v[i] = Float32((i * step) % n - n // 2) / Float32(16)
    return v^


def _gcd(a: Int, b: Int) -> Int:
    var x = a
    var y = b
    while y != 0:
        var t = x % y
        x = y
        y = t
    return x


def _dyadic(n: Int) -> List[Float32]:
    var v = List[Float32](length=n, fill=0)
    for i in range(n):  # small-loop(check data)
        v[i] = Float32((i * 3) % 11 - 5) / Float32(16)
    return v^


# ---------------------------------------------------------------- the float64 oracle (PyTorch's definitions)
def _four(x: List[Int]) -> List[Int]:
    if len(x) == 4:
        return x.copy()
    var d: List[Int] = [x[0], x[1], 1, x[2]]
    return d^


def _axis(c: _Case, axis: Int, isize: Int) -> Tuple[Int, Float32]:
    """(output size, float32 source scale) of spatial axis `axis` (0 height, 1 width), PyTorch's
    interpolate: size or floor(isize * scale_factor); align_corners (isize - 1) / (osize - 1); a given scale
    factor 1 / scale_factor unless recomputed; else isize / osize."""
    var two = len(c.x) == 4
    if not two and axis == 0:
        return (1, Float32(1) if c.ints[1] == 0 or c.ints[0] == 0 else Float32(0))
    var k = axis if two else 0
    var osize: Int
    var given = Float64(0)
    if c.ints[3] != 0:
        osize = c.ints[4 + k] if c.ints[3] != -1 else c.ints[4]
    else:
        var f = Float64(c.fs[k] if c.ints[6] != -1 else c.fs[0])
        osize = Int(Float64(isize) * f)
        if c.ints[2] == 0:
            given = f
    var align = c.ints[1] == 1 and c.ints[0] != 0
    if align:
        return (osize, Float32(Float64(isize - 1) / Float64(osize - 1)) if osize > 1 else Float32(0))
    if given > 0:
        return (osize, Float32(1.0 / given))
    return (osize, Float32(Float64(isize) / Float64(osize)))


def _lin(o: Int, isize: Int, scale: Float32, align: Bool) -> Tuple[Int, Int, Float64, Float64]:
    var s = Float64(scale)
    var src = s * Float64(o) if align else s * (Float64(o) + 0.5) - 0.5
    if src < 0:
        src = 0
    var i0 = Int(src)
    var i1 = i0 + 1 if i0 < isize - 1 else i0
    var l1 = src - Float64(i0)
    return (i0, i1, 1.0 - l1, l1)


def _oracle(c: _Case, x: List[Float64]) -> List[Float64]:
    var f = _four(c.x)
    var N = f[0]; var C = f[1]; var H = f[2]; var W = f[3]
    var out = List[Float64]()
    if c.kind == K_INTERP:
        var ah = _axis(c, 0, H)
        var aw = _axis(c, 1, W)
        var align = c.ints[1] == 1
        for r in range(N * C):  # small-loop(check oracle)
            for oh in range(ah[0]):
                for ow in range(aw[0]):
                    var b = r * H * W
                    if c.ints[0] == 0:
                        var h = min(Int(Float64(oh) * Float64(ah[1])), H - 1)
                        var w = min(Int(Float64(ow) * Float64(aw[1])), W - 1)
                        out.append(x[b + h * W + w])
                        continue
                    var th = _lin(oh, H, ah[1], align)
                    var tw = _lin(ow, W, aw[1], align)
                    out.append(th[2] * (tw[2] * x[b + th[0] * W + tw[0]] + tw[3] * x[b + th[0] * W + tw[1]])
                               + th[3] * (tw[2] * x[b + th[1] * W + tw[0]] + tw[3] * x[b + th[1] * W + tw[1]]))
        return out^
    if c.kind == K_PAD:
        var L = c.xargs[0]; var R = c.xargs[1]; var T = c.xargs[2]; var Bm = c.xargs[3]
        var reflect = c.ints[0] == 1
        for r in range(N * C):  # small-loop(check oracle)
            for hp in range(H + T + Bm):
                for wp in range(W + L + R):
                    var h = hp - T
                    var w = wp - L
                    if reflect:
                        h = -h if h < 0 else (2 * (H - 1) - h if h >= H else h)
                        w = -w if w < 0 else (2 * (W - 1) - w if w >= W else w)
                    else:
                        h = 0 if h < 0 else (H - 1 if h >= H else h)
                        w = 0 if w < 0 else (W - 1 if w >= W else w)
                    out.append(x[(r * H + h) * W + w])
        return out^
    var rr = c.ints[0]
    if c.kind == K_SHUFFLE:
        var Co = C // (rr * rr)
        for n in range(N):  # small-loop(check oracle)
            for co in range(Co):
                for oh in range(H * rr):
                    for ow in range(W * rr):
                        var ci = co * rr * rr + (oh % rr) * rr + (ow % rr)
                        out.append(x[((n * C + ci) * H + oh // rr) * W + ow // rr])
        return out^
    for n in range(N):  # small-loop(check oracle)
        for co in range(C * rr * rr):
            var ci = co // (rr * rr)
            var i = (co // rr) % rr
            var j = co % rr
            for oh in range(H // rr):
                for ow in range(W // rr):
                    out.append(x[((n * C + ci) * H + oh * rr + i) * W + ow * rr + j])
    return out^


def _out_dims(c: _Case) -> List[Int]:
    var f = _four(c.x)
    var out = List[Int]()
    out.append(f[0])
    if c.kind == K_INTERP:
        out.append(f[1])
        var ah = _axis(c, 0, f[2])
        var aw = _axis(c, 1, f[3])
        if len(c.x) == 4:
            out.append(ah[0])
        out.append(aw[0])
    elif c.kind == K_PAD:
        out.append(f[1])
        if len(c.x) == 4:
            out.append(f[2] + c.xargs[2] + c.xargs[3])
        out.append(f[3] + c.xargs[0] + c.xargs[1])
    elif c.kind == K_SHUFFLE:
        var r = c.ints[0]
        out.append(f[1] // (r * r))
        out.append(f[2] * r)
        out.append(f[3] * r)
    else:
        var r = c.ints[0]
        out.append(f[1] * r * r)
        out.append(f[2] // r)
        out.append(f[3] // r)
    return out^


def _loss(c: _Case, x: List[Float64], seed: List[Float32]) -> Float64:
    var y = _oracle(c, x)
    var s = Float64(0)
    for i in range(len(y)):  # small-loop(check loss)
        s += Float64(seed[i]) * y[i]
    return s


def _close(name: String, got: List[Float32], want: List[Float64], tol: Float64) raises:
    require(len(got) == len(want), name + ": " + String(len(got)) + " values, " + String(len(want)) + " expected")
    for i in range(len(got)):  # small-loop(check comparison)
        var a = Float64(got[i])
        require(isfinite(a), name + ": non-finite at " + String(i))
        require(abs(a - want[i]) <= tol * (1.0 + abs(want[i])),
                name + ": index " + String(i) + " got " + String(got[i]) + " want " + String(want[i]))


# ---------------------------------------------------------------- the two sides
struct _Run(Movable):
    var y: List[Float32]
    var g: List[Float32]

    def __init__(out self):
        self.y = List[Float32]()
        self.g = List[Float32]()


def _call[E: Exec & Movable & Deinitable](mut s: OpsSession[E], c: _Case, hx: Int) raises -> Int:
    var hs: List[Int] = [hx]
    return s.op(_code(c), hs, c.ints, c.fs)


def _session[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, rid: Int, c: _Case, xv: List[Float32], seed: List[Float32],
) raises -> _Run:
    var r = _Run()
    var od = _out_dims(c)
    var s = ops_session[E, KIND](rid)
    var hx = tensor(s[], xv, c.x, True)
    var y = _call(s[], c, hx)
    r.y = values(s[], y, od)
    var hseed = tensor(s[], seed, od)
    var weighted = op2(s[], _MUL, y, hseed)
    var loss = total(s[], weighted)
    s[].backward(loss, False)
    r.g = gradient(s[], hx, c.x)
    s[].reset()
    # the seeded backward (vjp with the seed itself) gives the scalar backward's words
    var y2 = _call(s[], c, hx)
    var wrt: List[Int] = [hx]
    var g = s[].vjp(y2, wrt, hseed, False)
    same_words(values(s[], g[0], c.x), r.g, c.name + ": vjp differs from backward")
    s[].reset()
    var p = ops_session[E, KIND](sid)
    var px = tensor(p[], xv, c.x)
    var py = _call(p[], c, px)
    same_words(values(p[], py, od), r.y, c.name + ": the recorded forward differs from the unrecorded one")
    p[].reset()
    return r^


def _sequential[E: Exec](var ex: E, c: _Case, xv: List[Float32], seed: List[Float32]) raises -> _Run:
    """Sequential's own layer (nnc_forward, as autograd/model.mojo records it) on a (batch, features) input."""
    var r = _Run()
    var tape = Tape()
    var B = c.x[0]
    var h = tape.tensor(ex, xv, B, len(xv) // B, True)
    var sample = List[Int]()
    for k in range(1, len(c.x)):  # small-loop(rank)
        sample.append(c.x[k])
    var y = nnc_forward(tape, ex, c.seq_code, c.xargs, 0, c.f, rn_shape(sample, "check"), dummy_ptr(), dummy_ptr(),
                        0, h, True)
    r.y = tape.values(ex, y)
    var st = tape.tensor(ex, seed, tape.rows(y), tape.cols(y))
    tape.backward_with_rules(ex, y, st, ModelRules())
    r.g = tape.gradients(ex, h)
    return r^


def _open_and_run[E: Exec & Movable & Deinitable, KIND: Int](
    var plain: E, var recording: E, c: _Case, xv: List[Float32], seed: List[Float32],
) raises -> _Run:
    """The case on a fresh inference session over `plain` and a fresh recording one over `recording`."""
    var sid = ops_session_open[E, KIND](plain^, False)
    var rid = ops_session_open[E, KIND](recording^, True)
    try:
        var r = _session[E, KIND](sid, rid, c, xv, seed)
        ops_session_close[E, KIND](rid)
        ops_session_close[E, KIND](sid)
        return r^
    except error:
        ops_session_close[E, KIND](rid)
        ops_session_close[E, KIND](sid)
        raise error


def _check[E: Exec & Movable & Deinitable, KIND: Int](var plain: E, var recording: E, var seq_ex: E, which: Int) raises -> _Run:
    var cases = _cases()
    var c = cases[which].copy()
    var xv = _distinct(_count(c.x), which)
    var seed = _dyadic(_count(_out_dims(c)))
    var r = _open_and_run[E, KIND](plain^, recording^, c, xv, seed)
    var x64 = List[Float64](length=len(xv), fill=0)
    for i in range(len(xv)):  # small-loop(check data)
        x64[i] = Float64(xv[i])
    _close(c.name + " forward", r.y, _oracle(c, x64), 2e-6)
    var fd = List[Float64](length=len(xv), fill=0)
    for i in range(len(xv)):  # small-loop(every element probed)
        var up = x64.copy()
        up[i] = x64[i] + _H
        var down = x64.copy()
        down[i] = x64[i] - _H
        fd[i] = (_loss(c, up, seed) - _loss(c, down, seed)) / (2.0 * _H)
    _close(c.name + " gradient", r.g, fd, 2e-5)
    if c.seq:
        var q = _sequential(seq_ex^, c, xv, seed)
        same_words(r.y, q.y, c.name + ": the Session forward differs from Sequential's layer")
        same_words(r.g, q.g, c.name + ": the Session gradient differs from Sequential's layer")
    var digest = UInt64(1469598103934665603)
    for i in range(len(r.y)):  # small-loop(check digest)
        digest = (digest ^ UInt64(bitcast[DType.uint32](r.y[i]))) * UInt64(1099511628211)
    for i in range(len(r.g)):  # small-loop(check digest)
        digest = (digest ^ UInt64(bitcast[DType.uint32](r.g[i]))) * UInt64(1099511628211)
    print("SESSION_OPS_RESIZE", c.name, "digest", digest)
    return r^


def test_session_ops_resize_cases() raises:
    var n = len(_cases())
    comptime if CPU_COLUMN_BUILD:
        def host_case(which: Int) raises:
            _ = _check[HostExec, OPS_HOST](HostExec(), HostExec(), HostExec(), which)
        check_cases_serial("session_ops_resize", n, host_case)
    else:
        def device_case(which: Int) raises:
            var dev = _check[DeviceExec, OPS_DEVICE](DeviceExec(), DeviceExec(), DeviceExec(), which)
            var host = _check[HostExec, OPS_HOST](HostExec(), HostExec(), HostExec(), which)
            same_words(dev.y, host.y, "session_ops_resize case " + String(which) + ": device and host forward")
            same_words(dev.g, host.g, "session_ops_resize case " + String(which) + ": device and host gradient")
        check_cases_serial("session_ops_resize", n, device_case)


# ---------------------------------------------------------------- refusals
def _refused[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], code: Int, hx: Int, ints: List[Int], fs: List[Float32], words: String,
) raises:
    var hs: List[Int] = [hx]
    var raised = False
    try:
        _ = s.op(code, hs, ints, fs)
    except problem:
        raised = True
        require(String(problem).find(words) >= 0, "refused with '" + String(problem) + "', not '" + words + "'")
    require(raised, "not refused: " + words)


def _refusals[E: Exec & Movable & Deinitable, KIND: Int](sid: Int) raises:
    var s = ops_session[E, KIND](sid)
    var d4: List[Int] = [1, 2, 2, 3]
    var d3: List[Int] = [1, 2, 3]
    var d5: List[Int] = [1, 1, 2, 2, 2]
    var x4 = tensor(s[], _distinct(12, 1), d4)
    var x3 = tensor(s[], _distinct(6, 2), d3)
    var x5 = tensor(s[], _distinct(8, 3), d5)
    var two: List[Float32] = [Float32(2), Float32(2)]
    var none: List[Float32] = [Float32(0), Float32(0)]
    var linear: List[Int] = [1, 0, 0, 0, 0, 0, -1]
    _refused(s[], OPS_INTERPOLATE, x4, linear, two, "mode 'linear' needs a 3-D input")
    var bilinear: List[Int] = [2, 0, 0, 0, 0, 0, -1]
    _refused(s[], OPS_INTERPOLATE, x3, bilinear, two, "mode 'bilinear' needs a 4-D input")
    var nearest: List[Int] = [0, 0, 0, 0, 0, 0, -1]
    _refused(s[], OPS_INTERPOLATE, x5, nearest, two, "x must be (N, C, L) or (N, C, H, W)")
    var aligned: List[Int] = [0, 1, 0, 0, 0, 0, -1]
    _refused(s[], OPS_INTERPOLATE, x4, aligned, two, "align_corners can only be set with mode")
    var both: List[Int] = [0, 0, 0, -1, 4, 4, -1]
    _refused(s[], OPS_INTERPOLATE, x4, both, two, "give exactly one of size or scale_factor")
    var short: List[Int] = [0, 0, 0, 1, 4, 4, 0]
    _refused(s[], OPS_INTERPOLATE, x4, short, none, "size must have 2 values for a 4-D input, got 1")
    var r2: List[Int] = [2]
    _refused(s[], OPS_PIXEL_SHUFFLE, x4, r2, List[Float32](), "must be divisible by upscale_factor squared")
    var r3: List[Int] = [3]
    _refused(s[], OPS_PIXEL_UNSHUFFLE, x4, r3, List[Float32](), "must be divisible by downscale_factor")
    # (N, C, L) pads in ONNX order: [mode, N before, C before, L before, N after, C after, L after]
    var at_axis: List[Int] = [1, 0, 0, 3, 0, 0, 0]
    _refused(s[], OPS_PAD_MODE, x3, at_axis, List[Float32](), "must be smaller than the axis it pads")
    var channel: List[Int] = [2, 0, 1, 0, 0, 0, 0]
    _refused(s[], OPS_PAD_MODE, x3, channel, List[Float32](), "the batch and channel counts must be 0")
    var negative: List[Int] = [2, 0, 0, -1, 0, 0, 0]
    _refused(s[], OPS_PAD_MODE, x3, negative, List[Float32](), "takes non-negative counts")
    s[].reset()


def test_session_ops_resize_refusals() raises:
    comptime if CPU_COLUMN_BUILD:
        var sid = ops_session_open[HostExec, OPS_HOST](HostExec(), False)
        try:
            _refusals[HostExec, OPS_HOST](sid)
        except error:
            ops_session_close[HostExec, OPS_HOST](sid)
            raise error
        ops_session_close[HostExec, OPS_HOST](sid)
    else:
        var sid = ops_session_open[DeviceExec, OPS_DEVICE](DeviceExec(), False)
        try:
            _refusals[DeviceExec, OPS_DEVICE](sid)
        except error:
            ops_session_close[DeviceExec, OPS_DEVICE](sid)
            raise error
        ops_session_close[DeviceExec, OPS_DEVICE](sid)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
