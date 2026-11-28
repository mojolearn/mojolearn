# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Harmony (lane sc-stats-1, 2026-11-28; x_cluster/harmony.mojo over x_cluster/harmony_bodies.mojo) on the host
column's ops (HostOps, the same bodies the device kernels run one thread per cell).

    the ridge correction by hand: one cluster (R = 1), one covariate of two batches, 1-d embedding; the solve of
        (Phi Phi^T + diag(0, lamb, lamb)) W = Phi Z by the check's own float64 elimination, then
        Z_corr = Z - W_{b+1} per cell: the primitives hm_ridge, hm_solve, hm_correct give it to float32 rounding
    the update order is a permutation: (a s + c) mod n with a coprime to n visits every cell once
    the blocks are harmonypy's: ceil(1 / 0.05) = 20
    a full fit on two planted batches: R's rows sum to one, the centroids have unit norm, the objective after the
        last round is not above the initial one, the batch shift of the corrected embedding is below half the
        planted one, and a second fit with the same seed gives the same words
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import require
from x_cluster.host.host_ops import HostOps
from x_cluster.harmony import HarmonyParams, harmony_fit, harmony_blocks, _block_order, _gcd
from x_cluster.bodies import SplitMix64


def test_ridge_correction_by_hand() raises:
    # six cells, batches [0, 0, 0, 1, 1, 1], z = [1, 2, 3, 5, 6, 7]; one cluster, lamb = 1
    var n = 6
    var ops = HostOps()
    var z: List[Float32] = [1.0, 2.0, 3.0, 5.0, 6.0, 7.0]
    var r: List[Float32] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    var codes: List[Int32] = [0, 0, 0, 1, 1, 1]
    var lamb: List[Float32] = [1.0, 1.0]
    var zs = ops.put(z)
    var rs = ops.put(r)
    var cs = ops.put_i(codes)
    var ls = ops.put(lamb)
    var b1 = 3
    var am = ops.zeros(b1 * b1)
    var rhs = ops.zeros(b1)
    var w = ops.zeros(b1)
    var dst = ops.zeros(n)
    ops.hm_ridge(rs, zs, cs, ls, n, 1, 1, 1, b1, am, rhs)
    var a_words = ops.get(am, 9)
    var expected_a: List[Float64] = [6.0, 3.0, 3.0, 3.0, 4.0, 0.0, 3.0, 0.0, 4.0]
    for i in range(9):  # small-loop: the 3 x 3 system
        require(Float64(a_words[i]) == expected_a[i], "A[" + String(i) + "] = " + String(a_words[i]))
    ops.hm_solve(am, rhs, 1, b1, 1, w)
    ops.hm_correct(zs, rs, w, cs, n, 1, 1, 1, b1, dst)
    # float64 oracle: A x = [24, 6, 18]
    var m: List[Float64] = [6.0, 3.0, 3.0, 24.0, 3.0, 4.0, 0.0, 6.0, 3.0, 0.0, 4.0, 18.0]
    for col in range(3):  # small-loop: Gaussian elimination of the 3 x 4 augmented matrix
        for row in range(col + 1, 3):
            var f = m[row * 4 + col] / m[col * 4 + col]
            for k in range(col, 4):
                m[row * 4 + k] -= f * m[col * 4 + k]
    var x = List[Float64](length=3, fill=0.0)
    var row = 2
    while row >= 0:
        var s = m[row * 4 + 3]
        for k in range(row + 1, 3):  # small-loop: back substitution
            s -= m[row * 4 + k] * x[k]
        x[row] = s / m[row * 4 + row]
        row -= 1
    var got = ops.get(dst, n)
    for i in range(n):  # small-loop: Z_corr = z - W[b + 1]
        var want = Float64(z[i]) - x[1 + Int(codes[i])]
        require(abs(Float64(got[i]) - want) < 1e-5, "Z_corr[" + String(i) + "] = " + String(got[i]) + ", expected "
                + String(want))
    _ = ops^


def test_update_order_and_blocks() raises:
    require(harmony_blocks(1000, 0.05) == 20, "ceil(1 / 0.05) blocks")
    require(harmony_blocks(7, 0.05) == 7, "at most n blocks")
    var rng = SplitMix64(state=UInt64(3))
    for n in [10, 12, 97, 360]:  # small-loop: a few sizes
        var order = _block_order(rng, n)
        require(_gcd(order[0], n) == 1, "a coprime to n")
        var seen = List[Bool](length=n, fill=False)
        for s in range(n):  # small-loop: the n positions
            var i = (order[0] * s + order[1]) % n
            require(not seen[i], "cell visited twice")
            seen[i] = True


def _planted(n: Int, d: Int) -> Tuple[List[Float32], List[Int32]]:
    """Three cell types (rows i % 3) at separated centers, two batches ((i / 3) % 2) with batch 1 shifted by 1.5
    in every dimension, a small deterministic jitter."""
    var z = List[Float32](length=n * d, fill=Float32(0))
    var codes = List[Int32](length=n, fill=Int32(0))
    var rng = SplitMix64(state=UInt64(11))
    for i in range(n):  # small-loop: the planted embedding
        var cell = i % 3
        var batch = (i // 3) % 2
        codes[i] = Int32(batch)
        for f in range(d):
            var center = Float64(4 * ((cell + f) % 3)) - 4.0
            z[i * d + f] = Float32(center + 1.5 * Float64(batch) + 0.3 * (rng.unit() - 0.5))
    return (z^, codes^)


def _fit(z: List[Float32], codes: List[Int32], n: Int, d: Int, seed: Int) raises -> Tuple[List[Float32], List[Float32], List[Float32], List[Float64]]:
    var ops = HostOps()
    var offsets: List[Int32] = [0]
    var theta: List[Float64] = [2.0, 2.0]
    var lamb: List[Float64] = [1.0, 1.0]
    var p = HarmonyParams(n, d, 10, 1, 2, 10, 20, UInt64(seed), 0.1, 0.0, 0.05, 1.0e-5, 1.0e-4, theta^, lamb^)
    var res = harmony_fit(ops, z, codes, offsets, p)
    _ = ops^
    return (res.z_corr.copy(), res.r.copy(), res.y.copy(), res.objective.copy())


def test_full_fit_invariants() raises:
    var n = 300
    var d = 4
    var data = _planted(n, d)
    var fitted = _fit(data[0], data[1], n, d, 0)
    var zc = fitted[0].copy()
    var r = fitted[1].copy()
    var y = fitted[2].copy()
    var obj = fitted[3].copy()
    for i in range(n):  # small-loop: R's rows
        var s = Float64(0)
        for j in range(10):
            s += Float64(r[i * 10 + j])
        require(abs(s - 1.0) < 1e-4, "R row " + String(i) + " sums to " + String(s))
    for j in range(10):  # small-loop: the centroids
        var s = Float64(0)
        for f in range(d):
            s += Float64(y[j * d + f]) * Float64(y[j * d + f])
        require(abs(s - 1.0) < 1e-4, "centroid " + String(j) + " has squared norm " + String(s))
    require(obj[len(obj) - 1] <= obj[0], "objective rose: " + String(obj[0]) + " -> " + String(obj[len(obj) - 1]))
    # the batch shift: mean over cell types of |mean(batch 1) - mean(batch 0)|
    var before = Float64(0)
    var after = Float64(0)
    for cell in range(3):  # small-loop: the cell types
        for f in range(d):
            var s0 = Float64(0)
            var s1 = Float64(0)
            var t0 = Float64(0)
            var t1 = Float64(0)
            var c0 = 0
            var c1 = 0
            for i in range(n):
                if i % 3 != cell:
                    continue
                if data[1][i] == 0:
                    s0 += Float64(data[0][i * d + f])
                    t0 += Float64(zc[i * d + f])
                    c0 += 1
                else:
                    s1 += Float64(data[0][i * d + f])
                    t1 += Float64(zc[i * d + f])
                    c1 += 1
            before += abs(s1 / Float64(c1) - s0 / Float64(c0))
            after += abs(t1 / Float64(c1) - t0 / Float64(c0))
    require(after < 0.5 * before, "batch shift " + String(after) + " not below half of " + String(before))
    var again = _fit(data[0], data[1], n, d, 0)
    for i in range(n * d):  # small-loop: the same words
        require(again[0][i] == zc[i], "a second fit with the same seed differs at " + String(i))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
