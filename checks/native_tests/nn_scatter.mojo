# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""scatter_add and the segment reductions of lane nn-blocks-scatter (2026-11-25): autograd/nn_scatter.mojo,
autograd/nn_scatter_ops.mojo, and their VJPs through ImportRules (the rules every session and program backward uses).

One test, test_nn_scatter, runs seven independent units across host cores (each on its own HostExec):
  - units 0-2, segment_sum / segment_mean / segment_max of a (13, 3) data block by 13 ids over 6 segments (ids
    repeated up to four times, segments 2 and 5 EMPTY, rows 2 and 4 equal so segment 3's maxima TIE), the values
    mixing magnitudes 1e3 and 1e-2 so the order of the additions shows in the last bits:
      the values against (a) the stated words, a float32 fold of each segment's rows in ascending row order (the
      mean then one correctly rounded division by the count, an empty segment 0, the max the first largest) and
      (b) a float64 oracle;
      the data gradient of sum(w * y) against the stated words (w at the row's segment; divided by the count; at
      the first argmax only) and that closed form against central finite differences of the float64 objective
      (max: every cell but the tied ones, where the derivative does not exist);
      segment_max of a segment holding a NaN is NaN.
  - unit 3, scatter_add along dim 1 of x (2, 5, 3) with src and index (2, 7, 3) (repeated indices, one output cell
    no index names keeps x's word), unit 4 along dim -1 of x (3, 4) with (3, 6): the values against the words of
    x then the sources added in ascending source position, and a float64 oracle; the gradients (x: w; src: w at
    its target) against the stated words and central differences.
  - unit 5, the refusals in words: an index or a segment id out of range (by position and value), index not of
    src's shape, src not matching x off dim, dim out of range, a segment id count unlike data's rows, num_segments 0.
  - unit 6, the same reductions on 4097 rows over 3 segments, past one first-bad search block, against the words.
  - one `NN_SCATTER <case> digest` line over the value and gradient words of each case.

Columns: the CPU column runs on HostExec. The device column runs each unit on DeviceExec AND HostExec and requires
the same words.
"""
from std.math import isnan
from std.memory import bitcast
from std.testing import TestSuite
from std.time import perf_counter_ns
from core.host_parallel import host_parallelize
from autograd.tape import Tape, Tensor
from autograd.ops import AXIS_ALL
from autograd.rankn import ShapeN
from autograd.rankn_api import rn_as_shape, rn_shape
from autograd.import_api import ImportRules
from autograd.nn_scatter import nn_scatter_add, nn_segment, SEGMENT_SUM, SEGMENT_MEAN, SEGMENT_MAX, segment_op_name
from sequence.ops import FP
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD


def _fail(message: String) raises:
    print("FAIL NN_SCATTER", message)
    raise Error(message)


def _require(ok: Bool, message: String) raises:
    if not ok:
        _fail(message)


def _digest(words: List[Float32]) -> UInt64:
    var d = UInt64(1469598103934665603)
    for i in range(len(words)):  # small-loop(check digest): fixture-sized, no product work
        d = (d ^ UInt64(bitcast[DType.uint32](words[i]))) * UInt64(1099511628211)
    return d


def _same_words(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        _fail(name + ": word counts differ " + String(len(a)) + " vs " + String(len(b)))
    for i in range(len(a)):  # small-loop(check comparison): fixture-sized
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            _fail(name + ": words differ at " + String(i) + " (" + String(a[i]) + " vs " + String(b[i]) + ")")


def _close(name: String, actual: List[Float32], expected: List[Float64], scale: List[Float64]) raises:
    """|actual - expected| within float32 rounding of the terms' magnitude (scale: the sum of |terms| per cell)."""
    for i in range(len(actual)):  # small-loop(check comparison): fixture-sized
        var tol = 1.0e-6 * scale[i] + 1.0e-6 * abs(expected[i]) + 1.0e-30
        if not (abs(Float64(actual[i]) - expected[i]) <= tol):
            _fail(name + ": index " + String(i) + " is " + String(actual[i]) + ", the float64 oracle "
                  + String(expected[i]))


def _lcg(mut state: UInt64) -> Float64:
    """A uniform value in [-1, 1) of the fixture generator."""
    state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    return Float64(Int((state >> 33) % UInt64(2000)) - 1000) * 0.001


def _ids_words(ids: List[Int]) -> List[Float32]:
    """Each id as the bit pattern of an int32 word (the executor's ids)."""
    var w = List[Float32]()
    for i in range(len(ids)):  # small-loop(fixture)
        w.append(bitcast[DType.float32](Int32(ids[i])))
    return w^


def _ids_tensor[E: Exec](mut tape: Tape, mut ex: E, ids: List[Int]) raises -> FP:
    var t = tape.tensor(ex, _ids_words(ids), len(ids), 1, False)
    return tape.value_of(ex, t)


def _mixed(n: Int, seed: UInt64) -> List[Float32]:
    """n values, every third of magnitude up to 1e3, the rest up to 1e-2."""
    var state = seed
    var out = List[Float32]()
    for k in range(n):  # small-loop(fixture)
        var u = _lcg(state)
        out.append(Float32(u * (1000.0 if k % 3 == 0 else 0.01)))
    return out^


def _weights(n: Int, seed: UInt64) -> List[Float32]:
    var state = seed
    var out = List[Float32]()
    for _ in range(n):  # small-loop(fixture)
        out.append(Float32(_lcg(state) + 0.0625))
    return out^


def _loss_backward[E: Exec](mut tape: Tape, mut ex: E, y: Tensor, w: List[Float32]) raises:
    """sum(w * y) over y's matrix view, then the backward through ImportRules."""
    var r = tape.rows(y)
    var c = tape.cols(y)
    var ym = rn_as_shape(tape, ex, y, ShapeN.matrix(r, c))
    var wt = tape.tensor(ex, w, r, c, False)
    var loss = tape.sum(ex, tape.multiply(ex, ym, wt), AXIS_ALL)
    tape.backward_scalar_with_rules(ex, loss, ImportRules())


# ---------------------------------------------------------------- the segment reductions
comptime _SN = 13
comptime _SD = 3
comptime _SS = 6


def _seg_ids() -> List[Int]:
    var ids: List[Int] = [3, 0, 3, 1, 3, 0, 4, 4, 1, 3, 0, 1, 4]
    return ids^


def _seg_data() -> List[Float32]:
    var x = _mixed(_SN * _SD, UInt64(20261125))
    for j in range(_SD):  # small-loop(fixture): rows 2 and 4 (both segment 3) equal, a tied maximum
        x[4 * _SD + j] = x[2 * _SD + j]
    return x^


def _seg_obj64(x: List[Float64], w: List[Float32], ids: List[Int], kind: Int) -> Float64:
    """sum(w * reduce(x)) in float64 (the oracle the finite differences differentiate)."""
    var acc = List[Float64](length=_SS * _SD, fill=0.0)
    var cnt = List[Int](length=_SS, fill=0)
    var seen = List[Bool](length=_SS * _SD, fill=False)
    for r in range(_SN):  # small-loop(fixture)
        var s = ids[r]
        cnt[s] += 1
        for j in range(_SD):  # small-loop(fixture)
            var v = x[r * _SD + j]
            if kind == SEGMENT_MAX:
                if not seen[s * _SD + j] or v > acc[s * _SD + j]:
                    acc[s * _SD + j] = v
                seen[s * _SD + j] = True
            else:
                acc[s * _SD + j] += v
    var total = 0.0
    for c in range(_SS * _SD):  # small-loop(fixture)
        var v = acc[c]
        if kind == SEGMENT_MEAN and cnt[c // _SD] > 0:
            v /= Float64(cnt[c // _SD])
        total += Float64(w[c]) * v
    return total


def run_segment_case[E: Exec](mut ex: E, kind: Int) raises -> List[Float32]:
    var name = segment_op_name(kind)
    var ids = _seg_ids()
    var x = _seg_data()
    var w = _weights(_SS * _SD, UInt64(77 + kind))
    var tape = Tape()
    var dt = tape.tensor(ex, x, _SN, _SD, True)
    var ip = _ids_tensor(tape, ex, ids)
    var y = nn_segment(tape, ex, dt, ip, _SN, _SS, kind)
    _require(tape.rows(y) == _SS and tape.cols(y) == _SD, name + ": the result is not (6, 3)")
    var got = tape.values(ex, y)
    _loss_backward(tape, ex, y, w)
    var g = tape.gradients(ex, dt)
    tape.release(ex)
    # the stated words: each segment's rows folded in ascending row order (float32), then the kind's step
    var want = List[Float32](length=_SS * _SD, fill=Float32(0.0))
    var arg = List[Int](length=_SS * _SD, fill=-1)
    var cnt = List[Int](length=_SS, fill=0)
    var o64 = List[Float64](length=_SS * _SD, fill=0.0)
    var mag = List[Float64](length=_SS * _SD, fill=0.0)
    for r in range(_SN):  # small-loop(fixture)
        var s = ids[r]
        cnt[s] += 1
        for j in range(_SD):  # small-loop(fixture)
            var c = s * _SD + j
            var v = x[r * _SD + j]
            if kind == SEGMENT_MAX:
                if arg[c] < 0 or v > want[c]:
                    want[c] = v
                    arg[c] = r
                    o64[c] = Float64(v)
            else:
                want[c] = want[c] + v
                o64[c] += Float64(v)
                mag[c] += abs(Float64(v))
    if kind == SEGMENT_MEAN:
        for c in range(_SS * _SD):  # small-loop(fixture)
            var n = cnt[c // _SD]
            if n > 0:
                want[c] = want[c] / Float32(n)
                o64[c] /= Float64(n)
                mag[c] /= Float64(n)
    _same_words(name + " values (the stated fold order)", got, want)
    _close(name + " values (float64 oracle)", got, o64, mag)
    _require(got[2 * _SD] == 0 and got[5 * _SD + 1] == 0, name + ": an empty segment is not 0")
    # the gradient: the stated words, and the closed form against central differences of the float64 objective
    var gwant = List[Float32](length=_SN * _SD, fill=Float32(0.0))
    for r in range(_SN):  # small-loop(fixture)
        var s = ids[r]
        for j in range(_SD):  # small-loop(fixture)
            var c = s * _SD + j
            if kind == SEGMENT_SUM:
                gwant[r * _SD + j] = w[c]
            elif kind == SEGMENT_MEAN:
                gwant[r * _SD + j] = w[c] / Float32(cnt[s])
            elif arg[c] == r:
                gwant[r * _SD + j] = w[c]
    _same_words(name + " gradient (the stated words)", g, gwant)
    var x64 = List[Float64]()
    for i in range(len(x)):  # small-loop(fixture)
        x64.append(Float64(x[i]))
    for i in range(len(x)):  # small-loop(check): fixture-sized
        var r = i // _SD
        var j = i - r * _SD
        if kind == SEGMENT_MAX and ids[r] == 3 and (r == 2 or r == 4):
            continue  # the tie: no derivative there; the rule gives it to row 2 (checked by the words above)
        var h = 1.0e-7 * max(1.0, abs(x64[i]))
        var xp = x64.copy()
        var xm = x64.copy()
        xp[i] += h
        xm[i] -= h
        var fd = (_seg_obj64(xp, w, ids, kind) - _seg_obj64(xm, w, ids, kind)) / (2.0 * h)
        var an = Float64(gwant[i])
        _require(abs(fd - an) <= 1.0e-5 + 1.0e-4 * abs(an), name + ": gradient " + String(an)
                 + " vs central difference " + String(fd) + " at row " + String(r) + " column " + String(j))
    var words = got.copy()
    for i in range(len(g)):  # small-loop(fixture)
        words.append(g[i])
    if kind == SEGMENT_MAX:
        # a NaN in a segment makes that value NaN (the first NaN is kept)
        var nt = Tape()
        var xn: List[Float32] = [Float32(1.0), bitcast[DType.float32](UInt32(0x7FC00000)), Float32(3.0), Float32(2.0)]
        var nids: List[Int] = [0, 0, 0, 1]
        var nd = nt.tensor(ex, xn, 4, 1, False)
        var ny = nn_segment(nt, ex, nd, _ids_tensor(nt, ex, nids), 4, 2, SEGMENT_MAX)
        var nv = nt.values(ex, ny)
        nt.release(ex)
        _require(isnan(nv[0]) and nv[1] == 2.0, "segment_max: a NaN in a segment did not give NaN")
    print("NN_SCATTER " + name, "digest", _digest(words))
    return words^


# ---------------------------------------------------------------- scatter_add
def _scatter_ref(
    x: List[Float32], src: List[Float32], index: List[Int], O: Int, K: Int, V: Int, I: Int,
) -> Tuple[List[Float32], List[Float64], List[Float64], List[Int]]:
    """The stated words (x, then each source added in ascending source position), the float64 oracle, the
    magnitude of each cell's terms, and each source's flat target."""
    var out = x.copy()
    var o64 = List[Float64]()
    var mag = List[Float64]()
    for i in range(len(x)):  # small-loop(fixture)
        o64.append(Float64(x[i]))
        mag.append(abs(Float64(x[i])))
    var key = List[Int]()
    for p in range(O * K * I):  # small-loop(fixture)
        var o = p // (K * I)
        var i = p % I
        var c = (o * V + index[p]) * I + i
        key.append(c)
        out[c] = out[c] + src[p]
        o64[c] += Float64(src[p])
        mag[c] += abs(Float64(src[p]))
    return (out^, o64^, mag^, key^)


def _scatter_obj64(x: List[Float64], src: List[Float64], key: List[Int], w: List[Float32]) -> Float64:
    var out = x.copy()
    for p in range(len(src)):  # small-loop(fixture)
        out[key[p]] += src[p]
    var t = 0.0
    for c in range(len(out)):  # small-loop(fixture)
        t += Float64(w[c]) * out[c]
    return t


def run_scatter_case[E: Exec](mut ex: E, rank3: Bool) raises -> List[Float32]:
    var name = String("scatter_add dim 1 (2, 5, 3)") if rank3 else String("scatter_add dim -1 (3, 4)")
    var O = 2 if rank3 else 3
    var K = 7 if rank3 else 6
    var V = 5 if rank3 else 4
    var I = 3 if rank3 else 1
    var index = List[Int]()
    for p in range(O * K * I):  # small-loop(fixture): repeated targets; target 2 of row 0 column 0 never named
        var v = (p * 7 + p // 5) % V
        if rank3 and p % I == 0 and p // (K * I) == 0 and v == 2:
            v = 3
        index.append(v)
    var x = _mixed(O * V * I, UInt64(91))
    var src = _mixed(O * K * I, UInt64(93))
    var w = _weights(O * V * I, UInt64(95))
    var xs: ShapeN
    var ss: ShapeN
    if rank3:
        var xd: List[Int] = [O, V, I]
        var sd: List[Int] = [O, K, I]
        xs = rn_shape(xd, "fixture")
        ss = rn_shape(sd, "fixture")
    else:
        xs = ShapeN.matrix(O, V)
        ss = ShapeN.matrix(O, K)
    var tape = Tape()
    var xt = tape.tensor_nd(ex, x, xs, True)
    var st = tape.tensor_nd(ex, src, ss, True)
    var ip = _ids_tensor(tape, ex, index)
    var y = nn_scatter_add(tape, ex, xt, ip, ss, st, 1 if rank3 else -1)
    _require(tape.shape_of(y).same(xs), name + ": the result does not have x's shape")
    var got = tape.values(ex, y)
    _loss_backward(tape, ex, y, w)
    var gx = tape.gradients(ex, xt)
    var gs = tape.gradients(ex, st)
    tape.release(ex)
    var expect = _scatter_ref(x, src, index, O, K, V, I)
    _same_words(name + " values (the stated order)", got, expect[0])
    _close(name + " values (float64 oracle)", got, expect[1], expect[2])
    if rank3:
        _require(bitcast[DType.uint32](got[2 * I]) == bitcast[DType.uint32](x[2 * I]),
                 name + ": a cell no index names did not keep x's word")
    _same_words(name + " gradient of x", gx, w)
    var gsw = List[Float32]()
    for p in range(len(src)):  # small-loop(fixture)
        gsw.append(w[expect[3][p]])
    _same_words(name + " gradient of src", gs, gsw)
    var x64 = List[Float64]()
    var s64 = List[Float64]()
    for i in range(len(x)):  # small-loop(fixture)
        x64.append(Float64(x[i]))
    for i in range(len(src)):  # small-loop(fixture)
        s64.append(Float64(src[i]))
    for i in range(len(src)):  # small-loop(check): fixture-sized
        var h = 1.0e-6 * max(1.0, abs(s64[i]))
        var sp = s64.copy()
        var sm = s64.copy()
        sp[i] += h
        sm[i] -= h
        var fd = (_scatter_obj64(x64, sp, expect[3], w) - _scatter_obj64(x64, sm, expect[3], w)) / (2.0 * h)
        _require(abs(fd - Float64(gsw[i])) <= 1.0e-5 + 1.0e-4 * abs(fd), name + ": src gradient vs central difference at "
                 + String(i))
    for i in range(len(x)):  # small-loop(check): fixture-sized
        var h = 1.0e-6 * max(1.0, abs(x64[i]))
        var xp = x64.copy()
        var xm = x64.copy()
        xp[i] += h
        xm[i] -= h
        var fd = (_scatter_obj64(xp, s64, expect[3], w) - _scatter_obj64(xm, s64, expect[3], w)) / (2.0 * h)
        _require(abs(fd - Float64(w[i])) <= 1.0e-5 + 1.0e-4 * abs(fd), name + ": x gradient vs central difference at "
                 + String(i))
    var words = got.copy()
    for i in range(len(gx)):  # small-loop(fixture)
        words.append(gx[i])
    for i in range(len(gs)):  # small-loop(fixture)
        words.append(gs[i])
    print("NN_SCATTER " + name, "digest", _digest(words))
    return words^


# ---------------------------------------------------------------- refusals
def _refused(what: String, err: String, want: String) raises:
    if want not in err:
        _fail(what + ": refused with '" + err + "', expected '" + want + "'")


def run_refusals[E: Exec](mut ex: E) raises -> List[Float32]:
    var x: List[Float32] = [Float32(1), Float32(2), Float32(3), Float32(4)]
    var bad: List[Int] = [0, 1, 9, 2]
    var cases = 0
    var t1 = Tape()
    try:
        var d = t1.tensor(ex, x, 4, 1, False)
        _ = nn_segment(t1, ex, d, _ids_tensor(t1, ex, bad), 4, 4, SEGMENT_SUM)
        _fail("segment_sum accepted a segment id out of range")
    except e:
        _refused("segment id", String(e), "ValueError: segment_sum: segment id 9 at position 2 is outside [0, 4)")
        cases += 1
    t1.release(ex)
    var t2 = Tape()
    try:
        var xt = t2.tensor(ex, x, 2, 2, False)
        var st = t2.tensor(ex, x, 2, 2, False)
        _ = nn_scatter_add(t2, ex, xt, _ids_tensor(t2, ex, bad), ShapeN.matrix(2, 2), st, 0)
        _fail("scatter_add accepted an index out of range")
    except e:
        _refused("index", String(e), "ValueError: scatter_add: index 9 at position 2 is outside [0, 2)")
        cases += 1
    t2.release(ex)
    var ok: List[Int] = [0, 1, 1, 0]
    var t3 = Tape()
    try:
        var xt = t3.tensor(ex, x, 2, 2, False)
        var st = t3.tensor(ex, x, 2, 2, False)
        _ = nn_scatter_add(t3, ex, xt, _ids_tensor(t3, ex, ok), ShapeN.matrix(4, 1), st, 0)
        _fail("scatter_add accepted an index of another shape")
    except e:
        _refused("index shape", String(e), "ValueError: scatter_add: index must have the shape of src")
        cases += 1
    t3.release(ex)
    var t4 = Tape()
    try:
        var xt = t4.tensor(ex, x, 2, 2, False)
        var st = t4.tensor(ex, x, 1, 4, False)
        _ = nn_scatter_add(t4, ex, xt, _ids_tensor(t4, ex, ok), ShapeN.matrix(1, 4), st, 0)
        _fail("scatter_add accepted src of another size off dim")
    except e:
        _refused("src size", String(e), "ValueError: scatter_add: src must have x's size on every dimension except dim")
        cases += 1
    t4.release(ex)
    var t5 = Tape()
    try:
        var xt = t5.tensor(ex, x, 2, 2, False)
        var st = t5.tensor(ex, x, 2, 2, False)
        _ = nn_scatter_add(t5, ex, xt, _ids_tensor(t5, ex, ok), ShapeN.matrix(2, 2), st, 2)
        _fail("scatter_add accepted dim 2 of a 2-D tensor")
    except e:
        _refused("dim", String(e), "ValueError: scatter_add: dim 2 is out of range for a tensor of 2 dimensions")
        cases += 1
    t5.release(ex)
    var t6 = Tape()
    try:
        var d = t6.tensor(ex, x, 2, 2, False)
        _ = nn_segment(t6, ex, d, _ids_tensor(t6, ex, ok), 4, 2, SEGMENT_MEAN)
        _fail("segment_mean accepted four ids for two rows")
    except e:
        _refused("id count", String(e), "ValueError: segment_mean: segment_ids must hold one id per row of data (2 rows, 4 ids)")
        cases += 1
    t6.release(ex)
    var t7 = Tape()
    try:
        var d = t7.tensor(ex, x, 4, 1, False)
        _ = nn_segment(t7, ex, d, _ids_tensor(t7, ex, ok), 4, 0, SEGMENT_MAX)
        _fail("segment_max accepted num_segments 0")
    except e:
        _refused("num_segments", String(e), "ValueError: segment_max: num_segments must be a positive integer, got 0")
        cases += 1
    t7.release(ex)
    _require(cases == 7, "refusals: " + String(cases) + " of 7 refused")
    var words: List[Float32] = [Float32(cases)]
    return words^


# ---------------------------------------------------------------- past one id-check block
def run_long_case[E: Exec](mut ex: E) raises -> List[Float32]:
    var N = 4097
    var ids = List[Int]()
    for r in range(N):  # small-loop(fixture)
        ids.append((r * r + 3 * r) % 3)
    var x = _mixed(N, UInt64(4097))
    var words = List[Float32]()
    for kind in range(3):  # small-loop(three kinds)
        var tape = Tape()
        var d = tape.tensor(ex, x, N, 1, False)
        var y = nn_segment(tape, ex, d, _ids_tensor(tape, ex, ids), N, 3, kind)
        var got = tape.values(ex, y)
        tape.release(ex)
        var want = List[Float32](length=3, fill=Float32(0.0))
        var cnt = List[Int](length=3, fill=0)
        for r in range(N):  # small-loop(fixture)
            var s = ids[r]
            if kind == SEGMENT_MAX:
                if cnt[s] == 0 or x[r] > want[s]:
                    want[s] = x[r]
            else:
                want[s] = want[s] + x[r]
            cnt[s] += 1
        if kind == SEGMENT_MEAN:
            for s in range(3):  # small-loop(fixture)
                want[s] = want[s] / Float32(cnt[s])
        _same_words("4097 rows " + segment_op_name(kind), got, want)
        for s in range(3):  # small-loop(fixture)
            words.append(got[s])
    print("NN_SCATTER long digest", _digest(words))
    return words^


# ---------------------------------------------------------------- tests
comptime _UNITS = 7


def _run_unit[E: Exec](mut ex: E, unit: Int) raises -> List[Float32]:
    """Units: 0-2 the segment reductions, 3-4 scatter_add, 5 the refusals, 6 the long case."""
    var t0 = perf_counter_ns()
    var words: List[Float32]
    if unit < 3:
        words = run_segment_case(ex, unit)
    elif unit < 5:
        words = run_scatter_case(ex, unit == 3)
    elif unit == 5:
        words = run_refusals(ex)
    else:
        words = run_long_case(ex)
    print("NN_SCATTER unit", unit, "seconds", Float64(perf_counter_ns() - t0) * 1.0e-9)
    return words^


def test_nn_scatter() raises:
    # Independent units: on the host column each runs on its own HostExec across host cores; the device column
    # then runs them in order on the one device and requires the host words.
    var failed = List[Int](length=_UNITS, fill=0)
    var host_words = List[List[Float32]](length=_UNITS, fill=List[Float32]())
    var fp = failed.unsafe_ptr()
    var wp = host_words.unsafe_ptr()
    def task(unit: Int) {imm fp, imm wp}:
        try:
            var host = HostExec()
            wp[unit] = _run_unit(host, unit)
        except error:
            print("FAIL NN_SCATTER unit", unit, error)
            fp[unit] = 1
    host_parallelize(task, _UNITS)
    for u in range(_UNITS):  # small-loop(_UNITS: failure flags)
        if failed[u] != 0:
            _fail("unit " + String(u) + " failed on the host column")
    comptime if not CPU_COLUMN_BUILD:
        for unit in range(_UNITS):  # driver-loop: over the units on the one device
            var dev = DeviceExec()
            var dw = _run_unit(dev, unit)
            _same_words("unit " + String(unit) + " host vs device", host_words[unit], dw)
    _ = failed^
    _ = host_words^
    print("PASS NN_SCATTER units", _UNITS)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
