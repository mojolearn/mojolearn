# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Hashed fixtures for the metric checks. NOT A PORT.

Every value is a splitmix64 hash of its indices, so no two cells repeat by
construction and a permutation of cells moves every per-cell comparison
(`uniform-test-data-hides-permutation`). The label fixtures are SKEWED
(class mass falls geometrically), so a code that confused row sums with
column sums, or `a[i]*b[j]` with `a[j]*b[i]`, is caught by the numbers and
not only by the shape. The float fixtures span magnitudes so a fold order
is visible in the last bits, and plant a few terms whose SQUARE is
subnormal so FTZ is reached.
"""

from std.memory import bitcast


def splitmix(row: Int, k: Int, salt: Int) -> UInt64:
    var z = (
        UInt64(row + 1) * 0x9E3779B97F4A7C15
        + UInt64(k + 1) * 0xBF58476D1CE4E5B9
        + UInt64(salt + 1) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return z


def u01(row: Int, k: Int, salt: Int) -> Float64:
    return Float64(splitmix(row, k, salt) >> 11) * (1.0 / 9007199254740992.0)


def skewed_label(row: Int, n_classes: Int, salt: Int) -> Int32:
    """Class c gets mass ~ 0.55^c (geometric, renormalized): the first class
    holds about half the rows, the last almost none. Never uniform."""
    var u = u01(row, 7, salt)
    var total = 0.0
    var w = 1.0
    for _ in range(n_classes):
        total += w
        w *= 0.55
    var acc = 0.0
    w = 1.0
    for c in range(n_classes):
        acc += w / total
        if u < acc:
            return Int32(c)
        w *= 0.55
    return Int32(n_classes - 1)


def labels_true_pred(
    n: Int, n_true: Int, n_pred: Int, agree: Float64, salt: Int
) -> Tuple[List[Int32], List[Int32]]:
    """`y_true` skewed over `n_true` classes; `y_pred` follows `y_true`
    (mapped modulo `n_pred`) with probability `agree`, else a hashed
    other class. Labels start at 0 -- the contingency matrix offset is
    exercised by `labels_offset` below."""
    var yt = List[Int32]()
    var yp = List[Int32]()
    for i in range(n):
        var t = skewed_label(i, n_true, salt)
        yt.append(t)
        if u01(i, 11, salt) < agree:
            yp.append(Int32(Int(t) % n_pred))
        else:
            yp.append(skewed_label(i, n_pred, salt + 17))
    return (yt^, yp^)


def labels_offset(labels: List[Int32], offset: Int32) -> List[Int32]:
    """The same labels shifted: `minLabel` is no longer 0, so the `- outIdxOffset`
    in the contingency index is reached."""
    var out = List[Int32]()
    for i in range(len(labels)):
        out.append(labels[i] + offset)
    return out^


def plant_singleton(mut labels: List[Int32], at: Int, value: Int32):
    """One cluster with exactly one member: `labels[at] = value`, where
    `value` appears nowhere else (the caller picks one past the range)."""
    labels[at] = value


def constant_labels(n: Int, value: Int32) -> List[Int32]:
    var out = List[Int32]()
    for _ in range(n):
        out.append(value)
    return out^


def hashed_floats(n: Int, salt: Int, lo_exp: Int, hi_exp: Int) -> List[Float32]:
    """`n` Float32 values `sign * 2^e * m`, `e` hashed in `[lo_exp, hi_exp]`,
    `m` hashed in `[1, 2)`: magnitudes span the range, signs mix, no two
    equal. Wide exponents separate fold orders in the last bits."""
    var out = List[Float32]()
    var span = hi_exp - lo_exp + 1
    for i in range(n):
        var e = lo_exp + Int(splitmix(i, 1, salt) % UInt64(span))
        var m = 1.0 + u01(i, 2, salt)
        var s = -1.0 if (splitmix(i, 3, salt) & 1) == 1 else 1.0
        var v = s * m
        if e >= 0:
            for _ in range(e):
                v *= 2.0
        else:
            for _ in range(-e):
                v *= 0.5
        out.append(Float32(v))
    return out^


def plant_subnormal_squares(
    mut y: List[Float32], mut y_hat: List[Float32], count: Int, salt: Int
):
    """Make `count` hashed positions have `y - y_hat` of order 2^-69, so
    `(y - y_hat)^2 ~ 2^-138` is a Float32 SUBNORMAL (below 2^-126): the
    term FTZ flushes on Apple and a denormal-honoring vendor keeps. `y =
    2^-69 * (1 + u)`, `y_hat = 2^-69`: the difference `2^-69 * u` is exact
    in Float32 and its square is subnormal. The oracle applies `ftz` at
    the same seam; the sabotage that drops it must fail here."""
    var n = len(y)
    var scale = Float32(1.0)
    for _ in range(69):
        scale *= Float32(0.5)
    for c in range(count):
        var i = Int(splitmix(c, 5, salt) % UInt64(n))
        y[i] = scale * (Float32(1.0) + Float32(u01(c, 8, salt)))
        y_hat[i] = scale


def hashed_pdf(n: Int, salt: Int, zeros_every: Int) -> List[Float32]:
    """A positive vector normalized to sum 1 in Float64 then cast, with a
    zero planted every `zeros_every` positions (KL's `modelPDF == 0`
    branch) -- hashed masses spanning three decades so no two equal."""
    var raw = List[Float64]()
    var total = 0.0
    for i in range(n):
        var v = 0.0
        if zeros_every <= 0 or (i % zeros_every) != zeros_every - 1:
            v = 0.001 + u01(i, 4, salt) * u01(i, 9, salt) * 0.999
        raw.append(v)
        total += v
    var out = List[Float32]()
    for i in range(n):
        out.append(Float32(raw[i] / total))
    return out^


def hashed_points(n: Int, d: Int, n_clusters: Int, salt: Int) -> Tuple[List[Float32], List[Int32]]:
    """`n` points in `d` dims around `n_clusters` hashed centers (spread
    so clusters overlap a little), labels SKEWED over the clusters. Row-
    major."""
    var x = List[Float32]()
    var labels = List[Int32]()
    for i in range(n):
        var c = skewed_label(i, n_clusters, salt)
        labels.append(c)
        for f in range(d):
            var center = (u01(Int(c), f, salt + 101) - 0.5) * 6.0
            var jitter = (u01(i, f, salt + 202) - 0.5) * 2.0
            x.append(Float32(center + jitter))
    return (x^, labels^)


def bits32(v: Float32) -> String:
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def bits64(v: Float64) -> String:
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint64](v)
    var out = String("0x")
    for i in range(16):
        var nib = Int((u >> UInt64(60 - 4 * i)) & UInt64(0xF))
        out += String(DIGITS[byte=nib])
    return out
