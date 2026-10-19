# SPDX-License-Identifier: Apache-2.0
# The tape's blocked fold over large and lopsided shapes (lane ag-pass3-fold,
# autograd/ops.mojo "THE BLOCKED FOLD"). For each shape and axis (all, rows,
# cols) two paths are checked against a float64 host sum of the same terms:
#   direct: autograd/fold.mojo blocked_fold (OP_AG_REDUCE_PART, the tree
#     levels OP_AGF_LEVEL, OP_AG_REDUCE_FOLD; or the one-launch fold);
#   vjp:    the tape's broadcast-operand gradient, d(x * w)/dw with a seed g
#     (OP_AGK_BIN_VJP_PART then blocked_fold_finish), terms mul(g, x).
# Two data sets: "exact" (small integers: every partial sum is an integer far
# below 2^24, so any order gives the float64 sum exactly; a wrong, missing or
# repeated term fails) and "round" (fractions on a 2^-23 grid: within a
# worst-case rounding bound, digests printed). Every FOLD line carries a
# digest of the output bits: HostExec, Metal, NVIDIA and AMD print the same
# lines (diff the cpu and gpu outputs), and fold v2 against
# -D MOJOLEARN_AUTOGRAD_FOLD_V2_OFF moves only the "round" digests of folds
# over more than 4096 terms.
from std.memory import bitcast
from autograd import Tape, Tensor
from sequence.exec_trait import Exec
from sequence.ops import FP, mul
from autograd.ops import AXIS_ALL, AXIS_ROWS, AXIS_COLS, AUTOGRAD_FOLD_V2
from autograd.fold import blocked_fold, fold_partials, fold_extent, fold_outputs


def _mix(i: Int, salt: Int) -> UInt64:
    var h = UInt64(i) * UInt64(0x5851F42D4C957F2D) + UInt64(salt) * UInt64(0x14057B7EF767814F) + UInt64(1)
    h ^= h >> UInt64(31)
    h *= UInt64(0x2545F4914F6CDD1D)
    h ^= h >> UInt64(29)
    return h


def _data(n: Int, salt: Int, exact: Bool) -> List[Float32]:
    """exact: integers -3 .. 3 with mean 0 (so sums stay far below 2^24);
    otherwise [-1, 1) on a 2^-23 grid."""
    var v = List[Float32](length=n, fill=0)
    for i in range(n):
        var h = _mix(i, salt)
        if exact:
            var k = Int(h >> UInt64(61))
            v[i] = Float32(k - 4) if k != 0 else Float32(0.0)
        else:
            v[i] = Float32(Int(h >> UInt64(40))) / Float32(8388608.0) - Float32(1.0)
    return v^


def _reference(
    terms: List[Float32], rows: Int, cols: Int, axis: Int, mut sums: List[Float64], mut mags: List[Float64],
):
    """Each output's float64 sum and sum of magnitudes of its terms."""
    var outs = fold_outputs(rows, cols, axis)
    sums = List[Float64](length=outs, fill=0.0)
    mags = List[Float64](length=outs, fill=0.0)
    for r in range(rows):
        for c in range(cols):
            var o = 0 if axis == AXIS_ALL else (c if axis == AXIS_ROWS else r)
            var t = Float64(terms[r * cols + c])
            sums[o] += t
            mags[o] += abs(t)


def _axis_name(axis: Int) -> String:
    if axis == AXIS_ALL:
        return "all"
    if axis == AXIS_ROWS:
        return "rows"
    return "cols"


def _compare(
    name: String, got: List[Float32], sums: List[Float64], mags: List[Float64], exact: Bool, extent: Int,
) raises -> UInt64:
    if len(got) != len(sums):
        raise Error(name + ": wrong output length")
    var digest = UInt64(1469598103934665603)
    var worst = Float64(0.0)
    for o in range(len(got)):
        var err = abs(Float64(got[o]) - sums[o])
        if exact:
            if err != 0.0:
                print("FAIL", name, "output", o, "actual", got[o], "expected", sums[o])
                raise Error(name + ": an exact fold differs from the float64 sum")
        else:
            # worst-case float32 rounding of any fixed-order fold whose
            # longest chain path is at most 8192 terms (v1's 4096 + 4096,
            # v2's <= 4 x 64): 8192 * 2^-24 < 5e-4 of the magnitude sum
            var tolerance = Float64(5e-4) * mags[o] + Float64(1e-30)
            if not (err <= tolerance):
                print("FAIL", name, "output", o, "actual", got[o], "expected", sums[o])
                raise Error(name + ": fold outside the rounding bound of the float64 sum")
            if mags[o] > 0.0:
                worst = max(worst, err / mags[o])
        digest = (digest ^ UInt64(bitcast[DType.uint32](got[o]))) * UInt64(1099511628211)
    print("FOLD", name, "extent", extent, "outputs", len(got), "worst_rel", worst, "digest", digest)
    return digest


def _direct[E: Exec](mut ex: E, data: List[Float32], rows: Int, cols: Int, axis: Int) raises -> List[Float32]:
    var n = rows * cols
    var src = ex.alloc(n)
    ex.upload(src, FP(unsafe_from_address=Int(data.unsafe_ptr())), n)
    var outs = fold_outputs(rows, cols, axis)
    var dst = ex.alloc(outs)
    var partials = ex.alloc(max(fold_partials(rows, cols, axis), 1))
    blocked_fold(ex, dst, src, rows, cols, axis, Float32(1), partials)
    var got = List[Float32](length=outs, fill=0)
    ex.download(FP(unsafe_from_address=Int(got.unsafe_ptr())), dst, outs)
    ex.free(partials)
    ex.free(dst)
    ex.free(src)
    return got^


def _vjp[E: Exec](
    mut ex: E, mut tape: Tape, x_data: List[Float32], g_data: List[Float32], rows: Int, cols: Int, axis: Int,
) raises -> List[Float32]:
    """dw of z = x * w, w broadcast over `axis` ((1, cols) for rows, (rows, 1)
    for cols, (1, 1) for all), seeded with g: the fold of mul(g, x)."""
    tape.reset()
    var wr = rows if axis == AXIS_COLS else 1
    var wc = cols if axis == AXIS_ROWS else 1
    var ones = List[Float32](length=wr * wc, fill=1)
    var x = tape.tensor(ex, x_data, rows, cols)
    var w = tape.tensor(ex, ones, wr, wc, True)
    var z = tape.multiply(ex, x, w)
    var seed = tape.tensor(ex, g_data, rows, cols)
    tape.backward_from(ex, z, seed)
    return tape.gradients(ex, w)


def run_fold_cases[E: Exec](mut ex: E) raises:
    print("FOLD geometry", String("v2") if AUTOGRAD_FOLD_V2 else String("v1"))
    var shapes = List[List[Int]]()
    shapes.append([512, 1])
    shapes.append([512, 256])
    shapes.append([4096, 4096])
    shapes.append([1, 100000])
    shapes.append([100000, 1])
    var axes = [AXIS_ALL, AXIS_ROWS, AXIS_COLS]
    var tape = Tape()
    var summary = UInt64(1469598103934665603)
    var cases = 0
    for si in range(len(shapes)):
        var rows = shapes[si][0]
        var cols = shapes[si][1]
        for e in range(2):
            var exact = e == 0
            var x = _data(rows * cols, 2 * si + 1, exact)
            var g = _data(rows * cols, 2 * si + 2, exact)
            var terms = List[Float32](length=rows * cols, fill=0)
            for i in range(rows * cols):
                terms[i] = mul(g[i], x[i])
            for ai in range(len(axes)):
                var axis = axes[ai]
                var base = (String(rows) + "x" + String(cols) + "/" + _axis_name(axis) + "/"
                            + (String("exact") if exact else String("round")))
                var extent = fold_extent(rows, cols, axis)
                var sums = List[Float64]()
                var mags = List[Float64]()
                _reference(x, rows, cols, axis, sums, mags)
                var d1 = _compare(base + "/direct", _direct(ex, x, rows, cols, axis), sums, mags, exact, extent)
                _reference(terms, rows, cols, axis, sums, mags)
                var d2 = _compare(base + "/vjp", _vjp(ex, tape, x, g, rows, cols, axis), sums, mags, exact, extent)
                summary = (summary ^ d1) * UInt64(1099511628211)
                summary = (summary ^ d2) * UInt64(1099511628211)
                cases += 2
    print("PASS fold cases", cases, "summary", summary)
    tape.release(ex)
