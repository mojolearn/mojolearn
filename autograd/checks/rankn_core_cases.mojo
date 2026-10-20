# SPDX-License-Identifier: Apache-2.0
# Lane ag-rankn-core (2026-10-20, docs/plans/AUTOGRAD_RANKN.md lane 1): checks
# of the rank-N shape metadata, view_nd and permute on any executor
# (HostExec, Metal, NVIDIA, AMD). Each case prints one
#   RANKN core <case> digest <d>
# line (the digest of the compared words, so the CPU, Metal, NVIDIA and AMD
# logs can be diffed line for line) and raises on any bit difference;
# the run ends with a PASS line.
#   view_nd: a (3, 5, 7) tensor through (15, 7), (7, 15) and (3, 1, 5, 7)
#     and back: the same words, and the gradient is the seed's words.
#   permute dense: the values against a host reference; permute moving the
#     last axis first (or the first axis last) against the composition
#     reshape, 2-D transpose, reshape, word for word, values and gradients.
#   permute VJP: the gradient equals the inverse permute of the seed.
#   permute of a permute: the composed view against the identity.
#   permute into matmul: a view whose matrix view has two strides read in
#     place by the GEMM, and one that matmul copies first, against the
#     product of the materialized view; a rank-2 permute against the
#     transpose view. Value and both gradients.
#   shape inheritance: relu, softmax, linear, the trailing-dims broadcast
#     add, a last-axis sum.
# Shapes are small, odd and off any board row (size reasoning: they cover
# rank 3 and 4, a non-square GEMM and a one-leaf fold).
from std.memory import bitcast
from autograd.tape import Tape, Tensor
from autograd.rankn import ShapeN
from sequence.exec_trait import Exec


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


def _shape_is(name: String, got: ShapeN, want: List[Int]) raises:
    var w = ShapeN.of(want)
    if not got.same(w):
        print("FAIL", name, "shape", got, "want", w)
        raise Error(name + ": shape differs")


struct _Run(Movable):
    var summary: UInt64
    var cases: Int

    def __init__(out self):
        self.summary = UInt64(1469598103934665603)
        self.cases = 0

    def add(mut self, name: String, d: UInt64):
        print("RANKN core", name, "digest", d)
        self.summary = (self.summary ^ d) * UInt64(1099511628211)
        self.cases += 1


def _permuted(v: List[Float32], dims: List[Int], axes: List[Int]) -> List[Float32]:
    """Host reference: v (row-major, dims) with its axes reordered."""
    var nd = len(dims)
    var od = List[Int]()
    for j in range(nd):  # small-loop(ndim: check rank)
        od.append(dims[axes[j]])
    var src_stride = List[Int](length=nd, fill=1)
    for j in range(nd - 1):  # small-loop(ndim: check rank)
        var k = nd - 2 - j
        src_stride[k] = src_stride[k + 1] * dims[k + 1]
    var out = List[Float32](length=len(v), fill=Float32(0))
    for t in range(len(v)):  # small-loop(len(v): check reference, host side)
        var rem = t
        var src = 0
        for j in range(nd):  # small-loop(ndim: check rank)
            var k = nd - 1 - j
            var c = rem % od[k]
            rem = rem // od[k]
            src += c * src_stride[axes[k]]
        out[t] = v[src]
    return out^


def _view_nd_case[E: Exec](mut ex: E, mut tape: Tape, mut run: _Run) raises:
    tape.reset()
    var xs = _data(3 * 5 * 7, 1)
    var x = tape.tensor_nd(ex, xs, ShapeN.of([3, 5, 7]), True)
    _shape_is("view_nd/x", tape.shape_of(x), [3, 5, 7])
    if tape.rows(x) != 15 or tape.cols(x) != 7 or tape.ndim(x) != 3 or tape.dim(x, -2) != 5:
        raise Error("view_nd: the matrix view of (3, 5, 7) is not (15, 7)")
    var a = tape.view_nd(ex, x, ShapeN.of([15, 7]))
    var b = tape.view_nd(ex, a, ShapeN.of([7, 15]))
    var c = tape.view_nd(ex, b, ShapeN.of([3, 1, 5, 7]))
    var d = tape.view_nd(ex, c, ShapeN.of([3, 5, 7]))
    _shape_is("view_nd/c", tape.shape_of(c), [3, 1, 5, 7])
    run.add("view_nd/values", _same("view_nd/values", tape.values(ex, d), xs))
    var seed_words = _data(3 * 5 * 7, 2)
    var seed = tape.tensor_nd(ex, seed_words, ShapeN.of([3, 5, 7]))
    tape.backward_from(ex, d, seed)
    run.add("view_nd/grad", _same("view_nd/grad", tape.gradients(ex, x), seed_words))
    var refused = False
    try:
        _ = tape.view_nd(ex, x, ShapeN.of([4, 5, 7]))
    except:
        refused = True
    if not refused:
        raise Error("view_nd: a count-changing shape was not refused")


def _permute_values[E: Exec](mut ex: E, mut tape: Tape, mut run: _Run, axes: List[Int], salt: Int) raises:
    """permute's words against the host reference, read through values()
    (pending) and after materialize."""
    tape.reset()
    var dims: List[Int] = [3, 5, 7]
    var xs = _data(3 * 5 * 7, salt)
    var x = tape.tensor_nd(ex, xs, ShapeN.of(dims), True)
    var p = tape.permute(ex, x, axes)
    var want = _permuted(xs, dims, axes)
    var od = List[Int]()
    for j in range(3):  # small-loop(3: check rank)
        od.append(dims[axes[j]])
    _shape_is("permute/shape", tape.shape_of(p), od)
    var name = String("permute/values/") + String(axes[0]) + String(axes[1]) + String(axes[2])
    var pending = tape.values(ex, p)
    _ = _same(name + "/pending", pending, want)
    tape.materialize(ex, p)
    run.add(name, _same(name, tape.values(ex, p), want))
    # VJP: the gradient is the inverse permute of the seed
    var seed_words = _data(3 * 5 * 7, salt + 1)
    var seed = tape.tensor_nd(ex, seed_words, ShapeN.of(od))
    tape.backward_from(ex, p, seed)
    var inv = List[Int](length=3, fill=0)
    for j in range(3):  # small-loop(3: check rank)
        inv[axes[j]] = j
    var gname = String("permute/vjp/") + String(axes[0]) + String(axes[1]) + String(axes[2])
    run.add(gname, _same(gname, tape.gradients(ex, x), _permuted(seed_words, od, inv)))


def _permute_vs_transpose[E: Exec](mut ex: E, mut tape: Tape, mut run: _Run, last_first: Bool) raises:
    """permute moving the last axis first ([2, 0, 1]) or the first axis last
    ([1, 2, 0]) of (3, 5, 7) is a 2-D transpose of a matrix view: compare
    the values and the gradient with reshape, transpose, reshape."""
    var axes: List[Int] = [2, 0, 1]
    var lead = 15
    var tail = 7
    var out_dims: List[Int] = [7, 3, 5]
    var tag = String("last_first")
    if not last_first:
        axes = [1, 2, 0]
        lead = 3
        tail = 35
        out_dims = [5, 7, 3]
        tag = String("first_last")
    var xs = _data(3 * 5 * 7, 9)
    var seed_words = _data(3 * 5 * 7, 10)
    var r = List[List[Float32]]()
    for arm in range(2):  # small-loop(2: permute, composition)
        tape.reset()
        var x = tape.tensor_nd(ex, xs, ShapeN.of([3, 5, 7]), True)
        var y: Tensor
        if arm == 0:
            y = tape.permute(ex, x, axes)
        else:
            var m = tape.reshape(ex, x, lead, tail)
            var t = tape.transpose(ex, m)
            y = tape.view_nd(ex, t, ShapeN.of(out_dims))
        var seed = tape.tensor_nd(ex, seed_words, ShapeN.of(out_dims))
        tape.backward_from(ex, y, seed)
        r.append(tape.values(ex, y))
        r.append(tape.gradients(ex, x))
    run.add("permute/transpose/" + tag + "/value", _same("permute/transpose/" + tag + "/value", r[0], r[2]))
    run.add("permute/transpose/" + tag + "/grad", _same("permute/transpose/" + tag + "/grad", r[1], r[3]))


def _permute_twice[E: Exec](mut ex: E, mut tape: Tape, mut run: _Run) raises:
    """A permute of a pending permute composes strides: [2, 0, 1] then
    [1, 2, 0] is the identity; values and the gradient pass through."""
    tape.reset()
    var xs = _data(2 * 3 * 4 * 5, 11)
    var x = tape.tensor_nd(ex, xs, ShapeN.of([2, 3, 4, 5]), True)
    var a = tape.permute(ex, x, [3, 0, 2, 1])   # (5, 2, 4, 3)
    var b = tape.permute(ex, a, [1, 3, 2, 0])   # (2, 3, 4, 5)
    _shape_is("permute/twice/shape", tape.shape_of(b), [2, 3, 4, 5])
    run.add("permute/twice/value", _same("permute/twice/value", tape.values(ex, b), xs))
    var seed_words = _data(2 * 3 * 4 * 5, 12)
    var seed = tape.tensor_nd(ex, seed_words, ShapeN.of([2, 3, 4, 5]))
    tape.backward_from(ex, b, seed)
    run.add("permute/twice/grad", _same("permute/twice/grad", tape.gradients(ex, x), seed_words))


def _matmul_arm[E: Exec](
    mut ex: E, mut tape: Tape, dims: List[Int], axes: List[Int], n: Int, dense: Bool,
) raises -> List[List[Float32]]:
    """[value, d x, d w] of matmul(permute(x, axes), w), the view
    materialized first when `dense`."""
    tape.reset()
    var count = 1
    for k in range(len(dims)):  # small-loop(ndim: check rank)
        count *= dims[k]
    var x = tape.tensor_nd(ex, _data(count, 13), ShapeN.of(dims), True)
    var p = tape.permute(ex, x, axes)
    if dense:
        tape.materialize(ex, p)
    var k_in = tape.cols(p)
    var w = tape.tensor(ex, _data(k_in * n, 14), k_in, n, True)
    var out = tape.matmul(ex, p, w)
    var seed = tape.tensor(ex, _data(tape.rows(out) * n, 15), tape.rows(out), n)
    tape.backward_from(ex, out, seed)
    var r = List[List[Float32]]()
    r.append(tape.values(ex, out))
    r.append(tape.gradients(ex, x))
    r.append(tape.gradients(ex, w))
    return r^


def _matmul_case[E: Exec](mut ex: E, mut tape: Tape, mut run: _Run) raises:
    # (7, 3, 5) -> [1, 2, 0] -> (3, 5, 7): leading axes collapse (row
    # stride 1, column stride 15), the GEMM reads the view in place
    # (3, 5, 7) -> [0, 2, 1] -> (3, 7, 5): three strides, matmul copies first
    var names: List[String] = ["strided", "copied"]
    var all_dims: List[List[Int]] = [[7, 3, 5], [3, 5, 7]]
    var all_axes: List[List[Int]] = [[1, 2, 0], [0, 2, 1]]
    for c in range(2):  # small-loop(2: cases)
        var v = _matmul_arm(ex, tape, all_dims[c], all_axes[c], 9, False)
        var w = _matmul_arm(ex, tape, all_dims[c], all_axes[c], 9, True)
        for k in range(3):  # small-loop(3: value and two gradients)
            var name = String("permute/matmul/") + names[c] + "/" + String(k)
            run.add(name, _same(name, v[k], w[k]))
    # a rank-2 permute on the right against the transpose view
    var r = List[List[Float32]]()
    for arm in range(2):  # small-loop(2: permute, transpose)
        tape.reset()
        var a = tape.tensor(ex, _data(11 * 13, 16), 11, 13, True)
        var z = tape.tensor(ex, _data(6 * 13, 17), 6, 13, True)
        var zt: Tensor
        if arm == 0:
            zt = tape.permute(ex, z, [1, 0])
        else:
            zt = tape.transpose(ex, z)
        var out = tape.matmul(ex, a, zt)
        var seed = tape.tensor(ex, _data(11 * 6, 18), 11, 6)
        tape.backward_from(ex, out, seed)
        r.append(tape.values(ex, out))
        r.append(tape.gradients(ex, a))
        r.append(tape.gradients(ex, z))
    for k in range(3):  # small-loop(3: value and two gradients)
        var name = String("permute/matmul/rank2/") + String(k)
        run.add(name, _same(name, r[k], r[3 + k]))


def _inherit_case[E: Exec](mut ex: E, mut tape: Tape, mut run: _Run) raises:
    tape.reset()
    var x = tape.tensor_nd(ex, _data(3 * 5 * 7, 19), ShapeN.of([3, 5, 7]), True)
    var w = tape.tensor(ex, _data(7 * 4, 20), 7, 4, True)
    var b = tape.tensor(ex, _data(4, 21), 1, 4, True)
    var row = tape.tensor(ex, _data(7, 22), 1, 7, True)
    var vec = tape.tensor_nd(ex, _data(7, 23), ShapeN.of([7]), True)
    var h = tape.relu(ex, x)
    _shape_is("inherit/relu", tape.shape_of(h), [3, 5, 7])
    var s = tape.softmax(ex, h)
    _shape_is("inherit/softmax", tape.shape_of(s), [3, 5, 7])
    var l = tape.linear(ex, s, w, b)
    _shape_is("inherit/linear", tape.shape_of(l), [3, 5, 4])
    var m = tape.matmul(ex, x, w)
    _shape_is("inherit/matmul", tape.shape_of(m), [3, 5, 4])
    var a1 = tape.add(ex, x, row)
    _shape_is("inherit/add_row", tape.shape_of(a1), [3, 5, 7])
    var a2 = tape.add(ex, vec, a1)
    _shape_is("inherit/add_vec", tape.shape_of(a2), [3, 5, 7])
    var r = tape.sum(ex, a2, 1)
    _shape_is("inherit/sum_last", tape.shape_of(r), [3, 5, 1])
    var t = tape.add(ex, tape.sum(ex, l), tape.sum(ex, r))
    _shape_is("inherit/scalar", tape.shape_of(t), [1, 1])
    tape.backward(ex, t)
    var d = _digest(tape.values(ex, t))
    d = (d ^ _digest(tape.gradients(ex, x))) * UInt64(1099511628211)
    d = (d ^ _digest(tape.gradients(ex, vec))) * UInt64(1099511628211)
    run.add("inherit/values_and_grads", d)


def run_rankn_core_checks[E: Exec](mut ex: E) raises:
    var tape = Tape()
    var run = _Run()
    _view_nd_case(ex, tape, run)
    var perms: List[List[Int]] = [[0, 2, 1], [2, 0, 1], [1, 2, 0], [2, 1, 0], [1, 0, 2]]
    for i in range(len(perms)):  # small-loop(5: permutations)
        _permute_values(ex, tape, run, perms[i], 3 + 2 * i)
    _permute_vs_transpose(ex, tape, run, True)
    _permute_vs_transpose(ex, tape, run, False)
    _permute_twice(ex, tape, run)
    _matmul_case(ex, tape, run)
    _inherit_case(ex, tape, run)
    print("PASS rankn core cases", run.cases, "summary", run.summary)
    tape.release(ex)
