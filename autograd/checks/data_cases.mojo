# SPDX-License-Identifier: Apache-2.0
# Data-side checks: epoch permutation, batch gather and tape dropout, against
# host oracles written here independently of the executor launches. Every
# check prints PASS with a digest of the executor's result; the digests must
# be equal on the host, Metal, NVIDIA and AMD.
from std.memory import bitcast
from autograd import Tape
from autograd.data import permutation, epoch_order, gather_rows
from autograd.checks.support import check_values
from core.philox_neural import neural_unit_at
from checks.numerics import identical_div
from sequence.exec_trait import Exec
from sequence.ops import FP, OP_MLP_PERM
from sequence.mlp import mlp_epoch_key, mlp_perm_args


def _ptr(mut values: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(values.unsafe_ptr()))


def _digest(values: List[Float32]) -> UInt64:
    var digest = UInt64(1469598103934665603)
    for i in range(len(values)):
        digest = (digest ^ UInt64(bitcast[DType.uint32](values[i]))) * UInt64(1099511628211)
    return digest


def _download[E: Exec](mut ex: E, src: FP, n: Int) raises -> List[Float32]:
    var out = List[Float32](length=n, fill=0)
    ex.download(_ptr(out), src, n)
    return out^


def _order[E: Exec](mut ex: E, seed: UInt64, epoch: Int, n: Int) raises -> List[Float32]:
    var d = ex.alloc(n)
    permutation(ex, seed, epoch, n, d)
    var out = _download(ex, d, n)
    ex.free(d)
    return out^


def _is_permutation(name: String, order: List[Float32]) raises:
    var n = len(order)
    var seen = List[Bool](length=n, fill=False)
    for i in range(n):
        var v = order[i]
        var r = Int(v)
        if Float32(r) != v or r < 0 or r >= n or seen[r]:
            print("FAIL", name, "position", i, "row", v)
            raise Error(name + ": not a permutation of 0..n-1")
        seen[r] = True


def _differences(a: List[Float32], b: List[Float32]) -> Int:
    var d = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            d += 1
    return d


def _check_permutations[E: Exec](mut ex: E) raises:
    # Sizes: one row, a power of two, and sizes that need the cycle walk.
    var sizes: List[Int] = [1, 2, 37, 256, 1000, 4099]
    for s in range(len(sizes)):
        var n = sizes[s]
        var name = "permutation/n" + String(n)
        var e0 = _order(ex, UInt64(731), 0, n)
        _is_permutation(name + "/epoch0", e0)
        var again = _order(ex, UInt64(731), 0, n)
        if _differences(e0, again) != 0:
            raise Error(name + ": same (seed, epoch) gave a different order")
        var e1 = _order(ex, UInt64(731), 1, n)
        _is_permutation(name + "/epoch1", e1)
        var other_seed = _order(ex, UInt64(732), 0, n)
        _is_permutation(name + "/seed732", other_seed)
        if n >= 37:
            # A collision of two keyed Feistel orders over 37+ rows is not a
            # plausible outcome; equal orders mean the key is ignored.
            if _differences(e0, e1) == 0:
                raise Error(name + ": epochs 0 and 1 gave the same order")
            if _differences(e0, other_seed) == 0:
                raise Error(name + ": seeds 731 and 732 gave the same order")
        # The module launches the trainers' operation with their key, so its
        # order is OP_MLP_PERM's for the same (seed, epoch), bit for bit.
        var d = ex.alloc(n)
        ex.launch[OP_MLP_PERM](mlp_perm_args(n, mlp_epoch_key(UInt64(731), 1), d), n)
        var direct = _download(ex, d, n)
        ex.free(d)
        if _differences(e1, direct) != 0:
            raise Error(name + ": differs from OP_MLP_PERM with the trainers' key")
        print("PASS", name, "digest", _digest(e0), _digest(e1))
    # Without shuffle the order is 0..n-1.
    var n = 53
    var d = ex.alloc(n)
    epoch_order(ex, False, UInt64(731), 4, n, d)
    var iota = _download(ex, d, n)
    ex.free(d)
    for i in range(n):
        if iota[i] != Float32(i):
            raise Error("epoch_order/no_shuffle: row " + String(i) + " out of place")
    print("PASS epoch_order/no_shuffle digest", _digest(iota))


def _check_gather[E: Exec](mut ex: E) raises:
    var n = 41
    var D = 3
    var x = List[Float32](length=n * D, fill=0)
    for i in range(n * D):
        x[i] = Float32((i * 7) % 23 - 11) / Float32(8)
    var dx = ex.alloc(n * D)
    ex.upload(dx, _ptr(x), n * D)
    var didx = ex.alloc(n)
    permutation(ex, UInt64(9), 2, n, didx)
    var order = _download(ex, didx, n)
    var start = 16
    var count = 16
    var dst = ex.alloc(count * D)
    gather_rows(ex, dx, didx, start, count, D, dst)
    var batch = _download(ex, dst, count * D)
    var expected = List[Float32](length=count * D, fill=0)
    for r in range(count):
        var row = Int(order[start + r])
        for c in range(D):
            expected[r * D + c] = x[row * D + c]
    if _differences(batch, expected) != 0:
        raise Error("gather_rows: batch differs from the host gather of the order")
    print("PASS gather_rows elements", count * D, "digest", _digest(batch))
    ex.free(dst)
    ex.free(didx)
    ex.free(dx)


def _host_keep(seed: UInt64, counter: Int, i: Int, p: Float32) -> Bool:
    """The coin of element i, straight from the neural stream (no launch)."""
    var lo = UInt32(Int(seed & UInt64(0xFFFFFFFF)))
    var hi = UInt32(Int(seed >> UInt64(32)))
    return neural_unit_at(lo, hi, UInt32(counter), i) >= p


def _check_dropout[E: Exec](mut ex: E) raises:
    var rows = 64
    var cols = 64
    var n = rows * cols
    var p = Float32(0.25)
    var seed = UInt64(0x0123456789ABCDEF)
    var counter = 7
    var xs = List[Float32](length=n, fill=0)
    var gs = List[Float32](length=n, fill=0)
    for i in range(n):
        # Nonzero inputs and upstream values, so a zero output marks a drop.
        xs[i] = Float32((i * 5) % 17 + 1) / Float32(8)
        gs[i] = Float32((i * 3) % 11 + 1) / Float32(4)
    var scale = identical_div(Float32(1), Float32(1) - p)
    var tape = Tape()
    var x = tape.tensor(ex, xs, rows, cols, True)
    var y = tape.dropout(ex, x, p, seed, counter)
    var g = tape.tensor(ex, gs, rows, cols)
    tape.backward_from(ex, y, g)
    var values = tape.values(ex, y)
    var grads = tape.gradients(ex, x)
    var want_y = List[Float32](length=n, fill=0)
    var want_g = List[Float32](length=n, fill=0)
    var kept = 0
    for i in range(n):
        var keep = _host_keep(seed, counter, i, p)
        if keep:
            kept += 1
            want_y[i] = xs[i] * scale
            want_g[i] = gs[i] * scale
        if (values[i] != Float32(0)) != keep or (grads[i] != Float32(0)) != keep:
            print("FAIL dropout/mask element", i, "keep", keep, "y", values[i], "dx", grads[i])
            raise Error("dropout: executor mask differs from the host oracle")
    # Kept count ~ Binomial(n, 1 - p): mean 3072, sd ~27.7; six sd allowed.
    var mean = Float32(n) * (Float32(1) - p)
    if abs(Float32(kept) - mean) > Float32(166):
        print("FAIL dropout/keep_rate kept", kept, "of", n)
        raise Error("dropout: kept fraction is far from 1 - p")
    print("PASS dropout/keep_rate kept", kept, "of", n)
    check_values("dropout/value", values, want_y)
    check_values("dropout/vjp", grads, want_g)
    # The same (seed, counter) gives the same mask on a new graph; another
    # counter gives another mask.
    tape.reset()
    var x2 = tape.tensor(ex, xs, rows, cols, True)
    var same = tape.values(ex, tape.dropout(ex, x2, p, seed, counter))
    if _differences(same, values) != 0:
        raise Error("dropout: same seed and counter gave a different result")
    var next = tape.values(ex, tape.dropout(ex, x2, p, seed, counter + 1))
    if _differences(next, values) == 0:
        raise Error("dropout: counters 7 and 8 gave the same mask")
    print("PASS dropout/replay digest", _digest(same), "next", _digest(next))
    # p = 0 keeps every element unchanged.
    var none = tape.values(ex, tape.dropout(ex, x2, Float32(0), seed, counter))
    if _differences(none, xs) != 0:
        raise Error("dropout: p = 0 changed a value")
    print("PASS dropout/p0 digest", _digest(none))
    var refusals = 0
    try:
        var bad = tape.dropout(ex, x2, Float32(1), seed, counter)
    except:
        refusals += 1
    try:
        var bad = tape.dropout(ex, x2, p, seed, -1)
    except:
        refusals += 1
    if refusals != 2:
        raise Error("dropout: an invalid p or counter was accepted")
    print("PASS dropout/refusals", refusals)
    tape.release(ex)


def run_data_checks[E: Exec](mut ex: E) raises:
    _check_permutations(ex)
    _check_gather(ex)
    _check_dropout(ex)
    print("PASS data_checks")
