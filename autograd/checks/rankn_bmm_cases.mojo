# SPDX-License-Identifier: Apache-2.0
# Lane ag-rankn-bmm (docs/plans/AUTOGRAD_RANKN.md section 2.3): checks of
# rankn_matmul on any executor (HostExec, Metal, NVIDIA, AMD). Each case
# prints one `RANKN bmm <case> digest <d>` line (the digest of the compared
# words, so the CPU, Metal, NVIDIA and AMD logs can be diffed) and raises on
# any difference; the run ends with PASS.
#   slices: [2, 3, 23, 37] @ [2, 3, 37, 19], the value and both gradients
#     against Tape.matmul on each of the six 2-D slices, word for word (every
#     K here is <= 64, outside gemm_split_shape, so the 2-D GEMM keeps its
#     chain).
#   permute/right, permute/left: a transposed operand through permute read
#     in place vs the same permute materialized first.
#   broadcast/*: broadcast operands (a [1, 3, K, N] weight, a rank-2 and a
#     rank-1 left operand, both sides broadcasting) against a host oracle
#     written elementwise in Mojo (never a NumPy product): the forward k
#     chain and each gradient's one chain over (broadcast batch coordinates
#     row-major, then the contraction), fma3 a term from +0.0.
#   attention: softmax(q @ k^T * s) @ v on [2, 4, 11, 8] against the
#     composition of per-head 2-D ops (matmul with a transpose view, scale,
#     softmax, matmul): the output and dq, dk, dv.
# Shapes are small, odd and off any board row.
from std.memory import bitcast
from autograd.tape import Tape, Tensor
from autograd.rankn import ShapeN, RN_MAX_RANK
from autograd.rankn_bmm import rankn_matmul, RankNBmmRules
from sequence.exec_trait import Exec
from sequence.ops import fma3


def _data(n: Int, salt: Int) -> List[Float32]:
    var v = List[Float32](length=n, fill=Float32(0))
    for i in range(n):  # small-loop(n: check data, host side)
        v[i] = Float32(((i * 7919 + salt * 104729) % 1000) - 500) * Float32(0.00123)
    return v^


def _digest(v: List[Float32], seed: UInt64 = UInt64(1469598103934665603)) -> UInt64:
    var digest = seed
    for i in range(len(v)):  # small-loop(len(v): check output digest)
        digest = (digest ^ UInt64(bitcast[DType.uint32](v[i]))) * UInt64(1099511628211)
    return digest


def _same(name: String, got: List[Float32], want: List[Float32]) raises:
    if len(got) != len(want):
        raise Error(name + ": lengths differ")
    for i in range(len(got)):  # small-loop(len(got): check comparison)
        if bitcast[DType.uint32](got[i]) != bitcast[DType.uint32](want[i]):
            print("FAIL", name, "index", i, "got", got[i], "want", want[i])
            raise Error(name + ": bits differ")


def _sh(dims: List[Int]) -> ShapeN:
    var d = InlineArray[Int, RN_MAX_RANK](fill=1)
    for k in range(len(dims)):  # small-loop(len(dims): tensor rank, at most 8)
        d[k] = dims[k]
    return ShapeN(len(dims), d)


def _count(dims: List[Int]) -> Int:
    var n = 1
    for v in dims:  # small-loop(len(dims): tensor rank, at most 8)
        n *= v
    return n


def _slice(v: List[Float32], s: Int, n: Int) -> List[Float32]:
    var r = List[Float32](length=n, fill=Float32(0))
    for i in range(n):  # small-loop(n: check data, host side)
        r[i] = v[s * n + i]
    return r^


def _shape_is(tape: Tape, x: Tensor, dims: List[Int], name: String) raises:
    var s = tape.shape_of(x)
    if not s.same(_sh(dims)):
        raise Error(name + ": wrong output shape")


def _report(name: String, parts: List[List[Float32]]) -> UInt64:
    var d = UInt64(1469598103934665603)
    for p in parts:  # small-loop(len(parts): compared buffers)
        d = _digest(p, d)
    print("RANKN bmm", name, "digest", d)
    return d


# ------------------------------------------------------------------ slices
def _case_slices[E: Exec](mut ex: E, mut tape: Tape) raises -> UInt64:
    comptime S = 6
    comptime M = 23
    comptime K = 37
    comptime N = 19
    var xd = _data(S * M * K, 1)
    var yd = _data(S * K * N, 2)
    var gd = _data(S * M * N, 3)
    tape.reset()
    var x = tape.tensor_nd(ex, xd, _sh([2, 3, M, K]), True)
    var y = tape.tensor_nd(ex, yd, _sh([2, 3, K, N]), True)
    var o = rankn_matmul(tape, ex, x, y)
    _shape_is(tape, o, [2, 3, M, N], "slices")
    var seed = tape.tensor_nd(ex, gd, _sh([2, 3, M, N]))
    tape.backward_with_rules(ex, o, seed, RankNBmmRules())
    var bv = tape.values(ex, o)
    var bx = tape.gradients(ex, x)
    var by = tape.gradients(ex, y)
    var wv = List[Float32]()
    var wx = List[Float32]()
    var wy = List[Float32]()
    for s in range(S):  # small-loop(S: check slices, one 2-D graph each)
        tape.reset()
        var xs = tape.tensor(ex, _slice(xd, s, M * K), M, K, True)
        var ys = tape.tensor(ex, _slice(yd, s, K * N), K, N, True)
        var os = tape.matmul(ex, xs, ys)
        var ss = tape.tensor(ex, _slice(gd, s, M * N), M, N)
        tape.backward_from(ex, os, ss)
        wv.extend(tape.values(ex, os))
        wx.extend(tape.gradients(ex, xs))
        wy.extend(tape.gradients(ex, ys))
    _same("slices/value", bv, wv)
    _same("slices/dx", bx, wx)
    _same("slices/dy", by, wy)
    var parts = List[List[Float32]]()
    parts.append(bv^)
    parts.append(bx^)
    parts.append(by^)
    return _report("slices", parts)


# ----------------------------------------------------------------- permute
def _permute_arm[E: Exec](mut ex: E, mut tape: Tape, right: Bool, dense: Bool) raises -> List[List[Float32]]:
    """[value, d dense operand, d permuted operand's source] of x @ permute(ys)
    (right) or permute(xs) @ y, the permute materialized first when dense."""
    comptime M = 13
    comptime K = 29
    comptime N = 11
    tape.reset()
    var p: Tensor
    var q: Tensor
    var o: Tensor
    if right:
        p = tape.tensor_nd(ex, _data(6 * M * K, 4), _sh([2, 3, M, K]), True)
        q = tape.tensor_nd(ex, _data(6 * N * K, 5), _sh([2, 3, N, K]), True)
        var t = tape.permute(ex, q, [0, 1, 3, 2])
        if dense:
            tape.materialize(ex, t)
        o = rankn_matmul(tape, ex, p, t)
    else:
        p = tape.tensor_nd(ex, _data(6 * K * M, 6), _sh([2, 3, K, M]), True)
        q = tape.tensor_nd(ex, _data(6 * K * N, 7), _sh([2, 3, K, N]), True)
        var t = tape.permute(ex, p, [0, 1, 3, 2])
        if dense:
            tape.materialize(ex, t)
        o = rankn_matmul(tape, ex, t, q)
    _shape_is(tape, o, [2, 3, M, N], "permute")
    var seed = tape.tensor_nd(ex, _data(6 * M * N, 8), _sh([2, 3, M, N]))
    tape.backward_with_rules(ex, o, seed, RankNBmmRules())
    var r = List[List[Float32]]()
    r.append(tape.values(ex, o))
    r.append(tape.gradients(ex, p))
    r.append(tape.gradients(ex, q))
    return r^


def _case_permute[E: Exec](mut ex: E, mut tape: Tape, right: Bool) raises -> UInt64:
    var name = String("permute/right") if right else String("permute/left")
    var view = _permute_arm(ex, tape, right, False)
    var dense = _permute_arm(ex, tape, right, True)
    _same(name + "/value", view[0], dense[0])
    _same(name + "/dp", view[1], dense[1])
    _same(name + "/dq", view[2], dense[2])
    return _report(name, view)


# --------------------------------------------------------------- broadcast
def _mdims(dims: List[Int]) -> Tuple[Int, Int, Int]:
    """(rows, cols, batch axes) of an operand's matrix view."""
    if len(dims) == 1:
        return (1, dims[0], 0)
    return (dims[len(dims) - 2], dims[len(dims) - 1], len(dims) - 2)


def _boff(coords: List[Int], dims: List[Int], nbx: Int, nb: Int) -> Int:
    """The operand's batch index (row-major over its own batch axes) at the
    output batch coordinates `coords`: 0 on an axis it broadcasts."""
    var off = 0
    for k in range(nbx):  # small-loop(nbx: tensor rank, at most 8)
        var c = coords[k + nb - nbx] if dims[k] > 1 else 0
        off = off * dims[k] + c
    return off


def _oracle(
    xd: List[Float32], xdims: List[Int], yd: List[Float32], ydims: List[Int], gd: List[Float32],
) raises -> List[List[Float32]]:
    """[x @ y, dX, dY] for seed gd, elementwise in Mojo: the forward chain k
    ascending; each gradient word one chain over the output batch
    coordinates that reach it (row-major), then the contraction; fma3 a
    term from +0.0."""
    var xm = _mdims(xdims)
    var ym = _mdims(ydims)
    var M = xm[0]
    var K = xm[1]
    var N = ym[1]
    var nbx = xm[2]
    var nby = ym[2]
    var nb = max(nbx, nby)
    var ob = List[Int]()
    for j in range(nb):  # small-loop(nb: tensor rank, at most 8)
        var ex_ = xdims[j - (nb - nbx)] if j >= nb - nbx else 1
        var ey = ydims[j - (nb - nby)] if j >= nb - nby else 1
        ob.append(max(ex_, ey))
    var OB = _count(ob)
    var xo = List[Int]()
    var yo = List[Int]()
    for bi in range(OB):  # small-loop(OB: check batch, host side)
        var coords = List[Int](length=nb, fill=0)
        var rem = bi
        var j = nb - 1
        while j >= 0:  # small-loop(nb: tensor rank, at most 8)
            coords[j] = rem % ob[j]
            rem = rem // ob[j]
            j -= 1
        xo.append(_boff(coords, xdims, nbx, nb))
        yo.append(_boff(coords, ydims, nby, nb))
    var vals = List[Float32](length=OB * M * N, fill=Float32(0))
    for bi in range(OB):  # small-loop(OB: check oracle)
        for m in range(M):  # small-loop(M: check oracle)
            for n in range(N):  # small-loop(N: check oracle)
                var acc = Float32(0.0)
                for k in range(K):  # small-loop(K: the oracle's chain)
                    acc = fma3(xd[(xo[bi] * M + m) * K + k], yd[(yo[bi] * K + k) * N + n], acc)
                vals[(bi * M + m) * N + n] = acc
    var XB = len(xd) // (M * K)
    var dx = List[Float32](length=len(xd), fill=Float32(0))
    for xb in range(XB):  # small-loop(XB: check oracle)
        for m in range(M):  # small-loop(M: check oracle)
            for k in range(K):  # small-loop(K: check oracle)
                var acc = Float32(0.0)
                for bi in range(OB):  # small-loop(OB: the oracle's chain, outer)
                    if xo[bi] != xb:
                        continue
                    for n in range(N):  # small-loop(N: the oracle's chain)
                        acc = fma3(gd[(bi * M + m) * N + n], yd[(yo[bi] * K + k) * N + n], acc)
                dx[(xb * M + m) * K + k] = acc
    var YB = len(yd) // (K * N)
    var dy = List[Float32](length=len(yd), fill=Float32(0))
    for yb in range(YB):  # small-loop(YB: check oracle)
        for k in range(K):  # small-loop(K: check oracle)
            for n in range(N):  # small-loop(N: check oracle)
                var acc = Float32(0.0)
                for bi in range(OB):  # small-loop(OB: the oracle's chain, outer)
                    if yo[bi] != yb:
                        continue
                    for m in range(M):  # small-loop(M: the oracle's chain)
                        acc = fma3(xd[(xo[bi] * M + m) * K + k], gd[(bi * M + m) * N + n], acc)
                dy[(yb * K + k) * N + n] = acc
    var res = List[List[Float32]]()
    res.append(vals^)
    res.append(dx^)
    res.append(dy^)
    return res^


def _case_broadcast[E: Exec](
    mut ex: E, mut tape: Tape, name: String, xdims: List[Int], ydims: List[Int], odims: List[Int], salt: Int,
) raises -> UInt64:
    var xd = _data(_count(xdims), salt)
    var yd = _data(_count(ydims), salt + 1)
    var gd = _data(_count(odims), salt + 2)
    tape.reset()
    var x = tape.tensor_nd(ex, xd, _sh(xdims), True)
    var y = tape.tensor_nd(ex, yd, _sh(ydims), True)
    var o = rankn_matmul(tape, ex, x, y)
    _shape_is(tape, o, odims, name)
    var seed = tape.tensor_nd(ex, gd, _sh(odims))
    tape.backward_with_rules(ex, o, seed, RankNBmmRules())
    var got = List[List[Float32]]()
    got.append(tape.values(ex, o))
    got.append(tape.gradients(ex, x))
    got.append(tape.gradients(ex, y))
    var want = _oracle(xd, xdims, yd, ydims, gd)
    _same(name + "/value", got[0], want[0])
    _same(name + "/dx", got[1], want[1])
    _same(name + "/dy", got[2], want[2])
    return _report(name, got)


# --------------------------------------------------------------- attention
def _case_attention[E: Exec](mut ex: E, mut tape: Tape) raises -> UInt64:
    comptime S = 8      # batch 2 x heads 4
    comptime T = 11
    comptime D = 8
    var factor = Float32(0.35355339)   # 1 / sqrt(D), the same constant in both arms
    var qd = _data(S * T * D, 21)
    var kd = _data(S * T * D, 22)
    var vd = _data(S * T * D, 23)
    var gd = _data(S * T * D, 24)
    tape.reset()
    var q = tape.tensor_nd(ex, qd, _sh([2, 4, T, D]), True)
    var k = tape.tensor_nd(ex, kd, _sh([2, 4, T, D]), True)
    var v = tape.tensor_nd(ex, vd, _sh([2, 4, T, D]), True)
    var kt = tape.permute(ex, k, [0, 1, 3, 2])
    var scores = rankn_matmul(tape, ex, q, kt)
    _shape_is(tape, scores, [2, 4, T, T], "attention/scores")
    var scaled = tape.scale(ex, scores, factor)
    var p = tape.softmax(ex, scaled)
    var o = rankn_matmul(tape, ex, p, v)
    _shape_is(tape, o, [2, 4, T, D], "attention")
    var seed = tape.tensor_nd(ex, gd, _sh([2, 4, T, D]))
    tape.backward_with_rules(ex, o, seed, RankNBmmRules())
    var bo = tape.values(ex, o)
    var bq = tape.gradients(ex, q)
    var bk = tape.gradients(ex, k)
    var bv = tape.gradients(ex, v)
    var wo = List[Float32]()
    var wq = List[Float32]()
    var wk = List[Float32]()
    var wv = List[Float32]()
    for s in range(S):  # small-loop(S: check heads, one 2-D graph each)
        tape.reset()
        var qs = tape.tensor(ex, _slice(qd, s, T * D), T, D, True)
        var ks = tape.tensor(ex, _slice(kd, s, T * D), T, D, True)
        var vs = tape.tensor(ex, _slice(vd, s, T * D), T, D, True)
        var kts = tape.transpose(ex, ks)
        var sc = tape.matmul(ex, qs, kts)
        var scs = tape.scale(ex, sc, factor)
        var ps = tape.softmax(ex, scs)
        var os = tape.matmul(ex, ps, vs)
        var ss = tape.tensor(ex, _slice(gd, s, T * D), T, D)
        tape.backward_from(ex, os, ss)
        wo.extend(tape.values(ex, os))
        wq.extend(tape.gradients(ex, qs))
        wk.extend(tape.gradients(ex, ks))
        wv.extend(tape.gradients(ex, vs))
    _same("attention/value", bo, wo)
    _same("attention/dq", bq, wq)
    _same("attention/dk", bk, wk)
    _same("attention/dv", bv, wv)
    var parts = List[List[Float32]]()
    parts.append(bo^)
    parts.append(bq^)
    parts.append(bk^)
    parts.append(bv^)
    return _report("attention", parts)


def run_rankn_bmm_checks[E: Exec](mut ex: E) raises:
    var tape = Tape()
    _ = _case_slices(ex, tape)
    _ = _case_permute(ex, tape, True)
    _ = _case_permute(ex, tape, False)
    _ = _case_broadcast(ex, tape, "broadcast/weight", [2, 3, 13, 9], [1, 3, 9, 7], [2, 3, 13, 7], 31)
    _ = _case_broadcast(ex, tape, "broadcast/rank2-left", [5, 6], [3, 6, 4], [3, 5, 4], 41)
    _ = _case_broadcast(ex, tape, "broadcast/both", [2, 1, 5, 6], [3, 6, 4], [2, 3, 5, 4], 51)
    _ = _case_broadcast(ex, tape, "broadcast/rank1-left", [6], [2, 6, 3], [2, 1, 3], 61)
    _ = _case_attention(ex, tape)
    print("RANKN bmm PASS")
