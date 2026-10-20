# SPDX-License-Identifier: Apache-2.0
# Lane ag-rankn-layout (docs/plans/AUTOGRAD_RANKN.md sections 2.5, 2.6 and
# lane 4): checks of concat, narrow, split and take on any executor
# (HostExec, Metal, NVIDIA, AMD). Each case prints one
#   RANKN layout <case> digest <d>
# line (the digest of the compared words) so the CPU, Metal, NVIDIA and AMD
# logs can be diffed, then PASS; any bit difference raises.
#   concat/axis<a>: three [3, 4, 5]-like pieces (2, 4 and 3 along axis a)
#     concatenated: the value against the host concatenation, split back
#     into the pieces bit for bit, the pieces' gradients equal the seed's
#     slabs; and concat(split(concat)) seeded: the same slabs through three
#     narrows of one tensor (one written, two accumulated).
#   concat/permuted: an operand that is a pending permute read in place
#     through its strides equals the same concat after materialize (value
#     and the permuted tensor's gradient).
#   narrow/second_consumer: two overlapping narrows of one tensor added:
#     each narrow's value is its slab, and the tensor's gradient is the
#     seed placed at both slabs and added (the tape's accumulate order).
#   take/axis1: take along axis 1 of a [3, 7, 5] with repeated ids: the
#     value against the host gather, the gradient against the embedding
#     profile's backward oracle on the permuted seed.
#   take/axis0_rank2: take along axis 0 of a [7, 5] equals Tape.embedding.
#   take/refused: an id equal to the axis length is refused on the
#     executor before any slice is read.
# Shapes are small, odd and off any board row; the host oracles are written
# elementwise in Mojo (no NumPy products).
from std.memory import bitcast
from autograd.tape import Tape, Tensor, IdsHandle
from autograd.rankn import ShapeN
from autograd.rankn_layout import (
    RankNLayoutRules, rankn_concat, rankn_narrow, rankn_split, rankn_take, rn_shape_of_dims,
)
from embedding.checks.embedding_oracle import EMB_NO_PADDING_IDX, EmbConfig, emb_backward_oracle
from sequence.exec_trait import Exec
from sequence.ops import FP


# ---------------------------------------------------------------- host helpers
def _data(n: Int, salt: Int) -> List[Float32]:
    var v = List[Float32](length=n, fill=Float32(0))
    for i in range(n):  # small-loop(n: check data, host side)
        v[i] = Float32(((i * 7919 + salt * 104729) % 1000) - 500) * Float32(0.00123)
    return v^


def _digest(v: List[Float32]) -> UInt64:
    var digest = UInt64(1469598103934665603)
    for i in range(len(v)):  # small-loop(len(v): check output digest)
        digest = (digest ^ UInt64(bitcast[DType.uint32](v[i]))) * UInt64(1099511628211)
    return digest


def _same(name: String, got: List[Float32], want: List[Float32]) raises -> UInt64:
    if len(got) != len(want):
        raise Error(name + ": lengths differ")
    for i in range(len(got)):  # small-loop(len(got): check comparison)
        if bitcast[DType.uint32](got[i]) != bitcast[DType.uint32](want[i]):
            print("FAIL", name, "index", i, "got", got[i], "want", want[i])
            raise Error(name + ": bits differ")
    return _digest(got)


def _mix(a: UInt64, b: UInt64) -> UInt64:
    return (a ^ b) * UInt64(1099511628211)


def _count(dims: List[Int]) -> Int:
    var n = 1
    for k in range(len(dims)):  # small-loop(ndim: check shape)
        n *= dims[k]
    return n


def _coords(index: Int, dims: List[Int]) -> List[Int]:
    var c = List[Int](length=len(dims), fill=0)
    var rem = index
    for k in range(len(dims)):  # small-loop(ndim: check shape)
        var ax = len(dims) - 1 - k
        c[ax] = rem % dims[ax]
        rem = rem // dims[ax]
    return c^


def _index(c: List[Int], dims: List[Int]) -> Int:
    var i = 0
    for k in range(len(dims)):  # small-loop(ndim: check shape)
        i = i * dims[k] + c[k]
    return i


def _slab_host(full: List[Float32], dims: List[Int], axis: Int, start: Int, length: Int) -> List[Float32]:
    """The [start, start + length) slab of `full` along `axis`, row-major."""
    var od = dims.copy()
    od[axis] = length
    var out = List[Float32](length=_count(od), fill=Float32(0))
    for i in range(len(out)):  # small-loop(len(out): check oracle)
        var c = _coords(i, od)
        c[axis] += start
        out[i] = full[_index(c, dims)]
    return out^


def _place_host(slab: List[Float32], dims: List[Int], axis: Int, start: Int, length: Int) -> List[Float32]:
    """A zero tensor of `dims` holding `slab` at [start, start + length) along `axis`."""
    var sd = dims.copy()
    sd[axis] = length
    var out = List[Float32](length=_count(dims), fill=Float32(0))
    for i in range(len(slab)):  # small-loop(len(slab): check oracle)
        var c = _coords(i, sd)
        c[axis] += start
        out[_index(c, dims)] = slab[i]
    return out^


def _piece_dims(axis: Int, k: Int) -> List[Int]:
    var d: List[Int] = [3, 4, 5]
    d[axis] = k
    return d^


def _put_ids[E: Exec](mut ex: E, ids: List[Int32]) raises -> FP:
    """The ids' bit patterns in executor words (Tape.ids' layout)."""
    var p = ex.alloc(len(ids))
    ex.upload(p, FP(unsafe_from_address=Int(ids.unsafe_ptr())), len(ids))
    return p


# ---------------------------------------------------------------- concat / split
def _concat_case[E: Exec](mut ex: E, mut tape: Tape, axis: Int) raises -> UInt64:
    var sizes: List[Int] = [2, 4, 3]
    var full_dims = _piece_dims(axis, 9)
    var host = List[List[Float32]]()
    var xs = List[Tensor]()
    for j in range(3):  # small-loop(3: concat pieces)
        host.append(_data(_count(_piece_dims(axis, sizes[j])), 20 + j))
    var seed_host = _data(_count(full_dims), 30)
    # the host concatenation
    var want = List[Float32](length=_count(full_dims), fill=Float32(0))
    var start = 0
    for j in range(3):  # small-loop(3: concat pieces)
        var pd = _piece_dims(axis, sizes[j])
        for i in range(len(host[j])):  # small-loop(piece: check oracle)
            var c = _coords(i, pd)
            c[axis] += start
            want[_index(c, full_dims)] = host[j][i]
        start += sizes[j]
    var name = String("concat/axis") + String(axis)

    # 1. concat, its value, split back, the slabs of the seed
    tape.reset()
    for j in range(3):  # small-loop(3: concat pieces)
        xs.append(tape.tensor_nd(ex, host[j], rn_shape_of_dims(_piece_dims(axis, sizes[j])), True))
    var cat = rankn_concat(tape, ex, xs, axis)
    var d = _same(name + "/value", tape.values(ex, cat), want)
    var pieces = rankn_split(tape, ex, cat, sizes, axis)
    for j in range(3):  # small-loop(3: concat pieces)
        d = _mix(d, _same(name + "/split" + String(j), tape.values(ex, pieces[j]), host[j]))
    var seed = tape.tensor_nd(ex, seed_host, rn_shape_of_dims(full_dims))
    tape.backward_with_rules(ex, cat, seed, RankNLayoutRules())
    start = 0
    for j in range(3):  # small-loop(3: concat pieces)
        d = _mix(d, _same(name + "/grad" + String(j), tape.gradients(ex, xs[j]),
                          _slab_host(seed_host, full_dims, axis, start, sizes[j])))
        start += sizes[j]

    # 2. concat(split(concat)): the same slabs through three narrows of one tensor
    tape.reset()
    xs = List[Tensor]()
    for j in range(3):  # small-loop(3: concat pieces)
        xs.append(tape.tensor_nd(ex, host[j], rn_shape_of_dims(_piece_dims(axis, sizes[j])), True))
    var cat1 = rankn_concat(tape, ex, xs, axis)
    var cat2 = rankn_concat(tape, ex, rankn_split(tape, ex, cat1, sizes, axis), axis)
    d = _mix(d, _same(name + "/roundtrip", tape.values(ex, cat2), want))
    var seed2 = tape.tensor_nd(ex, seed_host, rn_shape_of_dims(full_dims))
    tape.backward_with_rules(ex, cat2, seed2, RankNLayoutRules())
    start = 0
    for j in range(3):  # small-loop(3: concat pieces)
        d = _mix(d, _same(name + "/roundtrip_grad" + String(j), tape.gradients(ex, xs[j]),
                          _slab_host(seed_host, full_dims, axis, start, sizes[j])))
        start += sizes[j]
    print("RANKN layout", name, "digest", d)
    return d


def _concat_permuted_run[E: Exec](mut ex: E, mut tape: Tape, dense: Bool) raises -> List[List[Float32]]:
    """[value, d y] of concat([a, permute(y, (0, 2, 1)), b], axis=1), the
    permute materialized first when `dense`."""
    tape.reset()
    var a = tape.tensor_nd(ex, _data(3 * 2 * 5, 40), rn_shape_of_dims([3, 2, 5]), True)
    var y = tape.tensor_nd(ex, _data(3 * 5 * 4, 41), rn_shape_of_dims([3, 5, 4]), True)
    var b = tape.tensor_nd(ex, _data(3 * 3 * 5, 42), rn_shape_of_dims([3, 3, 5]), True)
    var q = tape.permute(ex, y, [0, 2, 1])
    if dense:
        tape.materialize(ex, q)
    var cat = rankn_concat(tape, ex, [a, q, b], 1)
    var seed = tape.tensor_nd(ex, _data(3 * 9 * 5, 43), rn_shape_of_dims([3, 9, 5]))
    tape.backward_with_rules(ex, cat, seed, RankNLayoutRules())
    var r = List[List[Float32]]()
    r.append(tape.values(ex, cat))
    r.append(tape.gradients(ex, y))
    return r^


def _concat_permuted[E: Exec](mut ex: E, mut tape: Tape) raises -> UInt64:
    var v = _concat_permuted_run(ex, tape, False)
    var w = _concat_permuted_run(ex, tape, True)
    var d = _same("concat/permuted/value", v[0], w[0])
    d = _mix(d, _same("concat/permuted/grad", v[1], w[1]))
    print("RANKN layout concat/permuted digest", d)
    return d


# ---------------------------------------------------------------- narrow
def _second_consumer[E: Exec](mut ex: E, mut tape: Tape) raises -> UInt64:
    tape.reset()
    var dims: List[Int] = [3, 9, 5]
    var x_host = _data(_count(dims), 50)
    var seed_host = _data(3 * 4 * 5, 51)
    var x = tape.tensor_nd(ex, x_host, rn_shape_of_dims(dims), True)
    var n1 = rankn_narrow(tape, ex, x, 1, 1, 4)
    var n2 = rankn_narrow(tape, ex, x, -2, 3, 4)
    var d = _same("narrow/second_consumer/n1", tape.values(ex, n1), _slab_host(x_host, dims, 1, 1, 4))
    d = _mix(d, _same("narrow/second_consumer/n2", tape.values(ex, n2), _slab_host(x_host, dims, 1, 3, 4)))
    var out = tape.add(ex, n1, n2)
    var seed = tape.tensor_nd(ex, seed_host, rn_shape_of_dims([3, 4, 5]))
    tape.backward_with_rules(ex, out, seed, RankNLayoutRules())
    # n2 (the later consumer) writes x's gradient, n1's slab is added to it
    var p1 = _place_host(seed_host, dims, 1, 1, 4)
    var p2 = _place_host(seed_host, dims, 1, 3, 4)
    var want = List[Float32](length=len(p1), fill=Float32(0))
    for i in range(len(want)):  # small-loop(len(want): check oracle)
        want[i] = p2[i] + p1[i]
    d = _mix(d, _same("narrow/second_consumer/grad", tape.gradients(ex, x), want))
    var refused = False
    try:
        _ = rankn_narrow(tape, ex, x, 1, 6, 4)
    except:
        refused = True
    if not refused:
        raise Error("narrow: a slab past the end of the axis was not refused")
    print("RANKN layout narrow/second_consumer digest", d)
    return d


# ---------------------------------------------------------------- take
def _take_axis1[E: Exec](mut ex: E, mut tape: Tape) raises -> UInt64:
    tape.reset()
    var dims: List[Int] = [3, 7, 5]
    comptime N = 6
    var ids_host: List[Int32] = [4, 0, 6, 4, 2, 4]
    var x_host = _data(_count(dims), 60)
    var seed_host = _data(3 * N * 5, 61)
    var x = tape.tensor_nd(ex, x_host, rn_shape_of_dims(dims), True)
    var ids = tape.ids(ex, _put_ids(ex, ids_host), N)
    var y = rankn_take(tape, ex, x, ids, 1)
    var out_dims: List[Int] = [3, N, 5]
    var want = List[Float32](length=_count(out_dims), fill=Float32(0))
    for i in range(len(want)):  # small-loop(len(want): check oracle)
        var c = _coords(i, out_dims)
        c[1] = Int(ids_host[c[1]])
        want[i] = x_host[_index(c, dims)]
    var d = _same("take/axis1/value", tape.values(ex, y), want)
    var seed = tape.tensor_nd(ex, seed_host, rn_shape_of_dims(out_dims))
    tape.backward_with_rules(ex, y, seed, RankNLayoutRules())
    # the embedding oracle on the permuted seed: dy[t, i * 5 + k] = seed[i, t, k]
    var dy = List[Float32](length=N * 15, fill=Float32(0))
    for i in range(len(seed_host)):  # small-loop(len(seed): check oracle)
        var c = _coords(i, out_dims)
        dy[c[1] * 15 + c[0] * 5 + c[2]] = seed_host[i]
    var dw = emb_backward_oracle(dy, ids_host, EmbConfig(7, 15, EMB_NO_PADDING_IDX, False), List[Float32]())
    var dx = List[Float32](length=_count(dims), fill=Float32(0))
    for i in range(len(dx)):  # small-loop(len(dx): check oracle)
        var c = _coords(i, dims)
        dx[i] = dw[c[1] * 15 + c[0] * 5 + c[2]]
    d = _mix(d, _same("take/axis1/grad", tape.gradients(ex, x), dx))
    print("RANKN layout take/axis1 digest", d)
    return d


def _take_axis0_rank2[E: Exec](mut ex: E, mut tape: Tape) raises -> UInt64:
    var ids_host: List[Int32] = [3, 3, 0, 6, 1]
    var r = List[List[Float32]]()
    for path in range(2):  # small-loop(2: take, then Tape.embedding)
        tape.reset()
        var w = tape.tensor_nd(ex, _data(7 * 5, 70), ShapeN.matrix(7, 5), True)
        var ids = tape.ids(ex, _put_ids(ex, ids_host), 5)
        var y: Tensor
        if path == 0:
            y = rankn_take(tape, ex, w, ids, 0)
        else:
            y = tape.embedding(ex, w, ids)
        var seed = tape.tensor_nd(ex, _data(5 * 5, 71), ShapeN.matrix(5, 5))
        tape.backward_with_rules(ex, y, seed, RankNLayoutRules())
        r.append(tape.values(ex, y))
        r.append(tape.gradients(ex, w))
    var d = _same("take/axis0_rank2/value", r[0], r[2])
    d = _mix(d, _same("take/axis0_rank2/grad", r[1], r[3]))
    print("RANKN layout take/axis0_rank2 digest", d)
    return d


def _take_refused[E: Exec](mut ex: E, mut tape: Tape) raises:
    tape.reset()
    var ids_host: List[Int32] = [1, 2, 7, 0]
    var x = tape.tensor_nd(ex, _data(3 * 7 * 5, 80), rn_shape_of_dims([3, 7, 5]), True)
    var ids = tape.ids(ex, _put_ids(ex, ids_host), 4)
    var raised = False
    try:
        _ = rankn_take(tape, ex, x, ids, 1)
    except e:
        var msg = String(e)
        if "id 7 at position 2 is outside [0, 7)" not in msg:
            raise Error("take/refused: unexpected message: " + msg)
        raised = True
    if not raised:
        raise Error("take/refused: id 7 along an axis of 7 was not refused")
    print("RANKN layout take/refused REFUSED")


# ---------------------------------------------------------------- run
def run_rankn_layout_checks[E: Exec](mut ex: E) raises:
    var tape = Tape()
    var summary = UInt64(1469598103934665603)
    for axis in range(3):  # small-loop(3: concat axes)
        summary = _mix(summary, _concat_case(ex, tape, axis))
    summary = _mix(summary, _concat_permuted(ex, tape))
    summary = _mix(summary, _second_consumer(ex, tape))
    summary = _mix(summary, _take_axis1(ex, tape))
    summary = _mix(summary, _take_axis0_rank2(ex, tape))
    _take_refused(ex, tape)
    tape.release(ex)
    print("RANKN layout summary digest", summary)
    print("PASS")
