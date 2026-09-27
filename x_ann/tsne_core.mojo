# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""t-SNE: THE PER-CELL ARITHMETIC, ONE SOURCE FOR THE GPU AND THE CPU
(lane/algos-ann, pass 1, 2026-09-27).

References: scikit-learn `sklearn/manifold/_t_sne.py` (`_joint_probabilities_nn`
:78-125, `_gradient_descent` :300-440, `TSNE._fit` / `_tsne` :850-1100) and
`_utils.pyx::_binary_search_perplexity`; cuML `cpp/src/tsne/` (the same
sparse-P + gradient-descent structure, Barnes-Hut and FFT repulsion).

THE FIXED-ORDER DESIGN
  * affinities: exact k-NN (k = min(n - 1, floor(3 * perplexity) + 1), the
    reference's count) by a per-row sorted insertion under (distance, index),
    then the per-row perplexity bisection in float32 (100 steps, tolerance
    1e-5, `identical_exp`), both as one cell per row.
  * symmetrization P = P_cond + P_cond^T and the global normalization are
    integer graph work plus one add per edge and one ascending sum; they run
    as the SAME host code in both drivers (`tsne_symmetrize`).
  * the gradient: attractive term over each row's CSR edges ascending (the
    UMAP CSR fold's shape); REPULSIVE TERM EXACT, one per-row fold over all
    points ascending (O(n^2) per iteration). cuML's Barnes-Hut tree insert
    and summarization use atomics, and its FFT arm reduces in cuFFT's order;
    neither is used (NOT_IMPLEMENTED.tsv). Z = sum of the per-row partial
    sums in ascending row order, one cell.
  * the optimizer: sklearn's gains / momentum update, every product through
    `identical_mul` (no contraction), a fixed iteration count (sklearn's
    `n_iter_without_progress` / `min_grad_norm` stops need the error each
    step; refused by name, the loop runs exactly `max_iter` steps).
"""

from checks.numerics import ftz, identical_div, identical_exp, identical_log, identical_mul, identical_mul_add

comptime F32P = MutPointer[Float32, MutAnyOrigin]
comptime I32P = MutPointer[Int32, MutAnyOrigin]


@always_inline
def ts_sqdist(x: F32P, i: Int, j: Int, d: Int) -> Float32:
    var acc = Float32(0.0)
    for c in range(d):
        var diff = ftz(ftz(x.unsafe_load(i * d + c)) - ftz(x.unsafe_load(j * d + c)))
        acc = ftz(identical_mul_add(diff, diff, acc))
    return acc


@always_inline
def ts_knn_cell(i: Int, x: F32P, n: Int, d: Int, nn: Int, nn_d: F32P, nn_i: I32P):
    """Row i's nn nearest other rows under (squared distance, index),
    ascending (DEVIATION 5810)."""
    var base = i * nn
    var filled = 0
    for j in range(n):
        if j == i:
            continue
        var dist = ts_sqdist(x, i, j, d)
        if filled == nn:
            var ld = nn_d.unsafe_load(base + nn - 1)
            var li = Int(nn_i.unsafe_load(base + nn - 1))
            if not (dist < ld or (dist == ld and j < li)):
                continue
        else:
            filled += 1
        var s = filled - 1
        while s > 0:
            var pd = nn_d.unsafe_load(base + s - 1)
            var pi = Int(nn_i.unsafe_load(base + s - 1))
            if dist < pd or (dist == pd and j < pi):
                nn_d.unsafe_store(base + s, pd)
                nn_i.unsafe_store(base + s, Int32(pi))
                s -= 1
            else:
                break
        nn_d.unsafe_store(base + s, dist)
        nn_i.unsafe_store(base + s, Int32(j))


@always_inline
def ts_perplexity_cell(i: Int, nn_d: F32P, nn: Int, log_perp: Float32, p: F32P):
    """sklearn `_binary_search_perplexity` for row i, float32, fixed steps,
    sums ascending (DEVIATION 5811)."""
    var base = i * nn
    var beta = Float32(1.0)
    var has_min = False
    var has_max = False
    var beta_min = Float32(0.0)
    var beta_max = Float32(0.0)
    for _ in range(100):
        var sum_p = Float32(0.0)
        for j in range(nn):
            var v = ftz(identical_exp(ftz(-identical_mul(nn_d.unsafe_load(base + j), beta))))
            p.unsafe_store(base + j, v)
            sum_p = ftz(sum_p + v)
        if sum_p == Float32(0.0):
            sum_p = Float32(1e-8)
        var sum_dp = Float32(0.0)
        for j in range(nn):
            var v = ftz(identical_div(p.unsafe_load(base + j), sum_p))
            p.unsafe_store(base + j, v)
            sum_dp = ftz(sum_dp + ftz(identical_mul(nn_d.unsafe_load(base + j), v)))
        var entropy = ftz(identical_log(sum_p) + ftz(identical_mul(beta, sum_dp)))
        var diff = ftz(entropy - log_perp)
        if abs(diff) <= Float32(1e-5):
            break
        if diff > Float32(0.0):
            beta_min = beta
            has_min = True
            if not has_max:
                beta = ftz(identical_mul(beta, Float32(2.0)))
            else:
                beta = ftz(identical_mul(ftz(beta + beta_max), Float32(0.5)))
        else:
            beta_max = beta
            has_max = True
            if not has_min:
                beta = ftz(identical_mul(beta, Float32(0.5)))
            else:
                beta = ftz(identical_mul(ftz(beta + beta_min), Float32(0.5)))


def _ts_sort_i32(mut a: List[Int32]):
    for t in range(1, len(a)):
        var v = a[t]
        var s = t - 1
        while s >= 0 and a[s] > v:
            a[s + 1] = a[s]
            s -= 1
        a[s + 1] = v


def tsne_symmetrize(
    n: Int, nn: Int, nn_i: List[Int32], p_cond: List[Float32],
    mut indptr: List[Int32], mut indices: List[Int32], mut values: List[Float32],
) raises:
    """P = (P_cond + P_cond^T) / sum, as CSR with ascending columns. Integer
    graph work plus one add per edge (the row's own term first) and one
    ascending sum (DEVIATION 5812); the same host code in both drivers."""
    var rows = List[List[Int32]](capacity=n)
    for _ in range(n):
        rows.append(List[Int32]())
    for i in range(n):
        for s in range(nn):
            var j = Int(nn_i[i * nn + s])
            rows[i].append(Int32(j))
            rows[j].append(Int32(i))
    indptr = List[Int32](capacity=n + 1)
    indices = List[Int32]()
    values = List[Float32]()
    indptr.append(Int32(0))
    for i in range(n):
        _ts_sort_i32(rows[i])
        var prev = -1
        for t in range(len(rows[i])):
            var j = Int(rows[i][t])
            if j == prev:
                continue
            prev = j
            var a = Float32(0.0)
            var b = Float32(0.0)
            for s in range(nn):
                if Int(nn_i[i * nn + s]) == j:
                    a = p_cond[i * nn + s]
                if Int(nn_i[j * nn + s]) == i:
                    b = p_cond[j * nn + s]
            indices.append(Int32(j))
            values.append(ftz(a + b))
        indptr.append(Int32(len(indices)))
    var total = Float32(0.0)
    for e in range(len(values)):
        total = ftz(total + values[e])
    if total < Float32(1.1920929e-07):
        total = Float32(1.1920929e-07)
    for e in range(len(values)):
        values[e] = ftz(identical_div(values[e], total))


@always_inline
def ts_q(y: F32P, i: Int, j: Int) -> Float32:
    """The Student-t kernel 1 / (1 + ||y_i - y_j||^2), two components."""
    var d0 = ftz(ftz(y.unsafe_load(2 * i)) - ftz(y.unsafe_load(2 * j)))
    var d1 = ftz(ftz(y.unsafe_load(2 * i + 1)) - ftz(y.unsafe_load(2 * j + 1)))
    var acc = ftz(identical_mul_add(d0, d0, Float32(0.0)))
    acc = ftz(identical_mul_add(d1, d1, acc))
    return ftz(identical_div(Float32(1.0), ftz(Float32(1.0) + acc)))


@always_inline
def ts_repulse_cell(i: Int, y: F32P, n: Int, row_z: F32P, rep: F32P):
    """row_z[i] = sum_{j != i} q_ij; rep[i] = sum_j q_ij^2 (y_i - y_j), j
    ascending (DEVIATION 5813; Z over rows ascending in `ts_sum_cell`)."""
    var z = Float32(0.0)
    var r0 = Float32(0.0)
    var r1 = Float32(0.0)
    var y0 = ftz(y.unsafe_load(2 * i))
    var y1 = ftz(y.unsafe_load(2 * i + 1))
    for j in range(n):
        if j == i:
            continue
        var q = ts_q(y, i, j)
        z = ftz(z + q)
        var qq = ftz(identical_mul(q, q))
        r0 = ftz(r0 + ftz(identical_mul(qq, ftz(y0 - ftz(y.unsafe_load(2 * j))))))
        r1 = ftz(r1 + ftz(identical_mul(qq, ftz(y1 - ftz(y.unsafe_load(2 * j + 1))))))
    row_z.unsafe_store(i, z)
    rep.unsafe_store(2 * i, r0)
    rep.unsafe_store(2 * i + 1, r1)


@always_inline
def ts_sum_cell(row_z: F32P, n: Int, z: F32P):
    var acc = Float32(0.0)
    for i in range(n):
        acc = ftz(acc + row_z.unsafe_load(i))
    z.unsafe_store(0, acc)


@always_inline
def ts_step_cell(
    e: Int, y: F32P, y_new: F32P, indptr: I32P, indices: I32P, values: F32P,
    rep: F32P, z: F32P, update: F32P, gains: F32P, exaggeration: Float32,
    momentum: Float32, learning_rate: Float32,
):
    """One coordinate e = 2 i + c: gradient (the CSR attraction ascending,
    DEVIATION 5814), gains (strict `update * grad < 0`, DEVIATION 5815),
    momentum update."""
    var i = e // 2
    var c = e % 2
    var yi = ftz(y.unsafe_load(e))
    var attr = Float32(0.0)
    for s in range(Int(indptr.unsafe_load(i)), Int(indptr.unsafe_load(i + 1))):
        var j = Int(indices.unsafe_load(s))
        var pq = ftz(identical_mul(ftz(values.unsafe_load(s)), ts_q(y, i, j)))
        attr = ftz(attr + ftz(identical_mul(pq, ftz(yi - ftz(y.unsafe_load(2 * j + c))))))
    var neg = ftz(identical_div(rep.unsafe_load(e), z.unsafe_load(0)))
    var grad = ftz(identical_mul(Float32(4.0), ftz(ftz(identical_mul(exaggeration, attr)) - neg)))
    var upd = update.unsafe_load(e)
    var gain = gains.unsafe_load(e)
    if ftz(identical_mul(upd, grad)) < Float32(0.0):
        gain = ftz(gain + Float32(0.2))
    else:
        gain = ftz(identical_mul(gain, Float32(0.8)))
    if gain < Float32(0.01):
        gain = Float32(0.01)
    grad = ftz(identical_mul(grad, gain))
    upd = ftz(ftz(identical_mul(momentum, upd)) - ftz(identical_mul(learning_rate, grad)))
    gains.unsafe_store(e, gain)
    update.unsafe_store(e, upd)
    y_new.unsafe_store(e, ftz(yi + upd))


@always_inline
def ts_kl_cell(i: Int, y: F32P, indptr: I32P, indices: I32P, values: F32P, z: F32P, kl: F32P):
    """Row i's share of KL(P || Q) over P's support (sklearn's BH error)."""
    var acc = Float32(0.0)
    var zz = z.unsafe_load(0)
    for s in range(Int(indptr.unsafe_load(i)), Int(indptr.unsafe_load(i + 1))):
        var j = Int(indices.unsafe_load(s))
        var p = ftz(values.unsafe_load(s))
        var q = ftz(identical_div(ts_q(y, i, j), zz))
        var pp = p if p > Float32(1.1920929e-07) else Float32(1.1920929e-07)
        var qq = q if q > Float32(1.1920929e-07) else Float32(1.1920929e-07)
        acc = ftz(acc + ftz(identical_mul(p, ftz(identical_log(ftz(identical_div(pp, qq)))))))
    kl.unsafe_store(i, acc)


def tsne_validate(n: Int, d: Int, perplexity: Float32, max_iter: Int, exploration: Int) raises:
    if n < 2 or d <= 0:
        raise Error("TSNE: at least two rows and one feature required")
    if not (perplexity > Float32(0.0)) or perplexity >= Float32(n):
        raise Error("TSNE: perplexity must be in (0, n_samples)")
    if max_iter < 1 or exploration < 0 or exploration > max_iter:
        raise Error("TSNE: need max_iter >= 1 and 0 <= exploration steps <= max_iter")


def tsne_nn(n: Int, perplexity: Float32) -> Int:
    var k = Int(identical_mul(Float32(3.0), perplexity)) + 1
    return k if k < n - 1 else n - 1
