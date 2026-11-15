# SPDX-License-Identifier: Apache-2.0
"""The lowering of imported ONNX models against the eager session, moved from
test_import_lower.py (2026-11-15), through the typed native import the public
`importers.import_model` door calls (autograd/import_reader.mojo `onnx_read`,
autograd/import_lower.mojo `onnx_import` on a session of the process-wide
table, then `OpsSession.run_program` on the captured program).

Every model file and initializer is stored under
checks/native_tests/fixtures/import_lower/ by
tools/import_lower_native_fixtures.py (the bytes the Python test wrote with
tests/_onnx_fixture.py). The eager reference is a second inference session
making the same operation calls with the words the Python `ops.Session`
door hands over (providers/python/autograd_ops_frontend_pyapi_door.mojo), so
the words must be equal.

  0 mixed: Slice, Gather, Clip, Pow, Add, ReduceMean, Softmax, ArgMax on
    x (3, 7); the output metadata (m (3,), p (3, 3), a (3,) int32 words) and
    every output word for word against split / concat / maximum / minimum /
    pow_scalar / add / mean / softmax / argmax(int32).
  1 grouped conv: Conv group=2 auto_pad SAME_UPPER strides 2, then a
    depthwise Conv pads 1, on x (3, 4, 8, 9): word for word against pad +
    conv2d(groups=2) + conv2d(groups=6), and to a float64 reference of the
    ONNX definition (rtol 1e-5, atol 1e-5).
  2 padded pools: MaxPool pads 1, AveragePool SAME_LOWER count_include_pad 0
    and pads 1 count_include_pad 1 on x (3, 3, 7, 8): a second run gives
    the same words; word for word against the eager compositions; MaxPool
    equal to the float64 reference, the averages to it at rtol 1e-5,
    atol 1e-6.
  3 embedding: Gather by an int64 ids input (3, 4), MatMul, Add: the input
    is an ids input; a second run gives the same words; the program equals
    the eager embedding on an ids tensor, which equals the embedding on the
    ids array; new ids equal the eager composition on them.
  4 BatchNorm prologue: the two (1, 4, 1, 1) parameters are the eager fold
    (scale / sqrt(var + 1e-5), bias - mean * that), word for word.
  5 Gemm transB: the one (6, 5) parameter is the transposed weight.

The inputs are core_ops_common's seeded stand-ins (`normal`, ids from
`unit_value`) of the original shapes and seeds: every assertion compares the
program with the eager session on the same inputs, or with a float64
reference computed from them. The saved program, the inputs by name, int32
ids, the float-ids TypeError, the refusals and the supported-ops table stay
in the Python test.
"""
from std.memory import bitcast, unsafe_memcpy
from autograd.ops_api import OpsSession, ops_session, ops_session_open, ops_session_close, OPS_HOST, OPS_DEVICE
from autograd.import_lower import onnx_import, OnnxImport, IMPORT_OK
from autograd.import_reader import onnx_read, onnx_graph, onnx_buffers, onnx_close, ONNX_OK
from autograd.program import PROGRAM_IN_IDS
from autograd.rankn import ShapeN
from core.cast_in import CastIn, CAST_KIND_SIGNED
from sequence.exec_trait import Exec
from sequence.ops import FP
from checks.native_tests.ops_higher import tensor, op1, op2, reshape, values
from checks.native_tests.core_ops_common import normal, unit_value, allclose, exact, dims2, dims3, dims4, opi
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require, fixture_path, read_fixture_bytes


comptime CORE_IMPORT_LOWER_FIXTURES = 6
comptime _FAMILY = "import_lower"
comptime _BATCH = 3

comptime _MATMUL = 1
comptime _ADD = 2
comptime _SUB = 3
comptime _MUL = 4
comptime _DIV = 5
comptime _SQRT = 11
comptime _MEAN_N = 36
comptime _SOFTMAX = 17
comptime _FULL = 30
comptime _EMBEDDING = 32
comptime _CONCAT = 39
comptime _NARROW = 40
comptime _CONV2D = 44
comptime _MAXPOOL2D = 45
comptime _AVGPOOL2D = 46
comptime _POW = 65
comptime _MINIMUM = 66
comptime _MAXIMUM = 67
comptime _ARGMAX = 72
comptime _PAD = 73
comptime _NPY_INT64 = 9  # the door's dtype number of a signed 8-byte array


def _model(fixture_id: Int) -> String:
    var names: List[String] = ["mixed", "grouped_conv", "padded_pools", "embedding_linear",
                               "batchnorm_relu", "linear_ln_linear"]
    return names[fixture_id].copy()


def _model_path(fixture_id: Int) -> String:
    return fixture_path(_FAMILY, _model(fixture_id) + ".onnx")


def _init(fixture_id: Int, name: String) raises -> List[Float32]:
    """The test's `_init(f, name)`: the initializer's float32 words (a few
    dozen; read serially inside the fixture's own task)."""
    var path = fixture_path(_FAMILY, _model(fixture_id) + "-" + name + ".f32")
    var bytes = read_fixture_bytes(path)
    require(len(bytes) % 4 == 0, path + ": not a float32 vector")
    var count = len(bytes) // 4
    var result = List[Float32](length=count, fill=0)
    unsafe_memcpy(dest=result.unsafe_ptr().bitcast[UInt8](), src=bytes.unsafe_ptr(), count=count * 4)
    _ = bytes^
    return result^


def _count(dims: List[Int]) -> Int:
    var n = 1
    for d in dims:
        n *= d
    return n


def _same(got: List[Float32], want: List[Float32], what: String) raises:
    """The test's `_same`: equal float32 words."""
    require(len(got) == len(want), what + ": extent " + String(len(got)) + " vs " + String(len(want)))
    var differ = 0
    var first = -1
    for i in range(len(got)):
        if bitcast[DType.uint32](got[i]) != bitcast[DType.uint32](want[i]):
            if first < 0:
                first = i
            differ += 1
    require(differ == 0, what + ": " + String(differ) + " words differ, first at " + String(first))


def _same_dims(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for k in range(len(a)):
        if a[k] != b[k]:
            return False
    return True


def _ids(count: Int, seed: Int) -> List[Int64]:
    """`rng.integers(0, vocab, size)` stand-in: fixed ids in [0, 11)."""
    var out = List[Int64](length=count, fill=0)
    for i in range(count):
        out[i] = Int64(Int(unit_value(seed, i) * 11.0))
    return out^


# ------------------------------------------------------------------ the import and the program

def _import[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, h: Int, fixture_id: Int, input_name: String, dims: List[Int]
) raises -> OnnxImport:
    """`importers.import_model(path, shapes={input_name: dims})`."""
    var gp = onnx_graph(h)
    require(gp[].status == ONNX_OK, _model(fixture_id) + ": the reader refused the model (status "
            + String(gp[].status) + "): " + gp[].message)
    var names: List[String] = [input_name]
    var flat: List[Int] = [len(dims)]
    for d in dims:
        flat.append(d)
    var res = onnx_import[E, KIND](h, sid, names, flat, onnx_buffers(h), _model_path(fixture_id))
    if res.status != IMPORT_OK:
        var why = res.message.copy()
        if len(res.problems) > 0:
            why = res.problems[0].name + ": " + res.problems[0].why
        raise Error(_model(fixture_id) + ": import status " + String(res.status) + ": " + why)
    return res^


def _results(out_counts: List[Int]) -> List[List[Float32]]:
    var results = List[List[Float32]]()
    for n in out_counts:
        results.append(List[Float32](length=n, fill=Float32(-7.5)))
    return results^


def _dsts(mut results: List[List[Float32]]) -> List[FP]:
    var dsts = List[FP]()
    for k in range(len(results)):
        dsts.append(FP(unsafe_from_address=Int(results[k].unsafe_ptr())))
    return dsts^


def _run_x[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, pid: Int, x: List[Float32], dims: List[Int], out_counts: List[Int]
) raises -> List[List[Float32]]:
    """`prog.run(x)` for a float32 C-order input of `dims` (its matrix view)."""
    var cols = dims[len(dims) - 1]
    var srcs = List[CastIn]()
    srcs.append(CastIn.f32_c(Int(x.unsafe_ptr()), len(x) // cols, cols))
    var results = _results(out_counts)
    var dsts = _dsts(results)
    var s = ops_session[E, KIND](sid)
    s[].run_program(pid, srcs, dsts)
    _ = x
    _ = dsts^
    return results^


def _ids_src(ids: List[Int64], rows: Int, cols: Int) -> CastIn:
    """An int64 C-order (rows, cols) ids array."""
    return CastIn(Int(ids.unsafe_ptr()), CAST_KIND_SIGNED, 8, rows, cols, cols * 8, 8)


def _run_ids[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, pid: Int, ids: List[Int64], rows: Int, cols: Int, out_counts: List[Int]
) raises -> List[List[Float32]]:
    """`prog.run(ids)` for an int64 (rows, cols) ids array."""
    var srcs = List[CastIn]()
    srcs.append(_ids_src(ids, rows, cols))
    var results = _results(out_counts)
    var dsts = _dsts(results)
    var s = ops_session[E, KIND](sid)
    s[].run_program(pid, srcs, dsts)
    _ = ids
    _ = dsts^
    return results^


# ------------------------------------------------------------------ eager spellings (the door's words)

def _split[E: Exec & Movable & Deinitable](mut r: OpsSession[E], x: Int, sizes: List[Int], axis: Int) raises -> List[Int]:
    """`Session.split(x, sizes, axis)`: operation 40 with [axis, piece, 0, sizes...] per piece."""
    var hs: List[Int] = [x]
    var out = List[Int]()
    for k in range(len(sizes)):
        var spec: List[Int] = [axis, k, 0]
        for j in range(len(sizes)):
            spec.append(sizes[j])
        out.append(opi(r, _NARROW, hs, spec, List[Float32]()))
    return out^


def _full[E: Exec & Movable & Deinitable](mut r: OpsSession[E], dims: List[Int], value: Float32) raises -> Int:
    var fs: List[Float32] = [value]
    return opi(r, _FULL, List[Int](), dims, fs)


def _pad[E: Exec & Movable & Deinitable](mut r: OpsSession[E], x: Int, pads: List[Int], value: Float32) raises -> Int:
    var hs: List[Int] = [x]
    var fs: List[Float32] = [value]
    return opi(r, _PAD, hs, pads, fs)


def _conv[E: Exec & Movable & Deinitable](
    mut r: OpsSession[E], x: Int, w: Int, b: Int, stride: Int, padding: Int, groups: Int
) raises -> Int:
    """`Session.conv2d(x, w, b or None, stride, padding, groups)` (b == 0: no bias)."""
    var hs: List[Int] = [x, w]
    if b != 0:
        hs.append(b)
    var ints: List[Int] = [stride, padding, groups]
    return opi(r, _CONV2D, hs, ints, List[Float32]())


def _pool[E: Exec & Movable & Deinitable](mut r: OpsSession[E], code: Int, x: Int, kernel: Int, stride: Int) raises -> Int:
    var hs: List[Int] = [x]
    var ints: List[Int] = [kernel, stride]
    return opi(r, code, hs, ints, List[Float32]())


# ------------------------------------------------------------------ float64 references (test oracles)

def _conv64(
    x: List[Float64], n: Int, c: Int, h: Int, wd: Int, w: List[Float32], oc: Int, cg: Int, kh: Int, kw: Int,
    b: List[Float32], stride: Int, pt: Int, pl: Int, pb: Int, pr: Int, groups: Int,
) -> List[Float64]:
    """The test's `_np_conv`: ONNX Conv (2-D), zero pads (top, left, bottom, right); `b` empty: no bias."""
    var hp = h + pt + pb
    var wp = wd + pl + pr
    var oh = (hp - kh) // stride + 1
    var ow = (wp - kw) // stride + 1
    var out = List[Float64](length=n * oc * oh * ow, fill=0)
    var ocg = oc // groups
    for i_n in range(n):
        for o in range(oc):
            var g = o // ocg
            for i in range(oh):
                for j in range(ow):
                    var acc = Float64(0)
                    for ci in range(cg):
                        var ch = g * cg + ci
                        for a in range(kh):
                            var yy = i * stride + a - pt
                            if yy < 0 or yy >= h:
                                continue
                            for e in range(kw):
                                var xx = j * stride + e - pl
                                if xx < 0 or xx >= wd:
                                    continue
                                acc += x[((i_n * c + ch) * h + yy) * wd + xx] * Float64(w[((o * cg + ci) * kh + a) * kw + e])
                    if len(b) > 0:
                        acc += Float64(b[o])
                    out[((i_n * oc + o) * oh + i) * ow + j] = acc
    return out^


def _pool64(
    x: List[Float32], n: Int, c: Int, h: Int, w: Int, k: Int, stride: Int,
    pt: Int, pl: Int, pb: Int, pr: Int, is_max: Bool, include_pad: Bool,
) -> List[Float64]:
    """The test's `_np_pool`: ONNX MaxPool / AveragePool (2-D, square kernel)."""
    var oh = (h + pt + pb - k) // stride + 1
    var ow = (w + pl + pr - k) // stride + 1
    var out = List[Float64](length=n * c * oh * ow, fill=0)
    for p in range(n * c):
        for i in range(oh):
            for j in range(ow):
                var h0 = i * stride - pt
                var w0 = j * stride - pl
                var hs = h0 if h0 > 0 else 0
                var ws = w0 if w0 > 0 else 0
                var he = h0 + k if h0 + k < h else h
                var we = w0 + k if w0 + k < w else w
                var acc = Float64(0)
                var top = Float64(0)
                var first = True
                for yy in range(hs, he):
                    for xx in range(ws, we):
                        var v = Float64(x[(p * h + yy) * w + xx])
                        acc += v
                        if first or v > top:
                            top = v
                            first = False
                var at = (p * oh + i) * ow + j
                if is_max:
                    out[at] = top
                else:
                    var div = k * k if include_pad else (he - hs) * (we - ws)
                    out[at] = acc / Float64(div)
    return out^


def _f32(v: List[Float64]) -> List[Float32]:
    var out = List[Float32](length=len(v), fill=0)
    for i in range(len(v)):
        out[i] = Float32(v[i])
    return out^


def _wide(v: List[Float32]) -> List[Float64]:
    var out = List[Float64](length=len(v), fill=0)
    for i in range(len(v)):
        out[i] = Float64(v[i])
    return out^


# ------------------------------------------------------------------ fixtures

def _mixed[E: Exec & Movable & Deinitable, KIND: Int](sid: Int, h: Int, mut r: OpsSession[E]) raises:
    var res = _import[E, KIND](sid, h, 0, "x", dims2(_BATCH, 7))
    require(len(res.out_names) == 3 and res.out_names[0] == "m" and res.out_names[1] == "p"
            and res.out_names[2] == "a", "mixed: the output names")
    var one: List[Int] = [_BATCH]
    require(_same_dims(res.out_dims[0], one) and _same_dims(res.out_dims[1], dims2(_BATCH, 3))
            and _same_dims(res.out_dims[2], one), "mixed: the output shapes")
    require(res.out_int32[0] == 0 and res.out_int32[1] == 0 and res.out_int32[2] == 1,
            "mixed: ArgMax comes back as int32 words, the others as float32")
    var x = normal(_BATCH * 7, 11)
    var counts: List[Int] = [_BATCH, _BATCH * 3, _BATCH]
    var got = _run_x[E, KIND](sid, res.pid, x, dims2(_BATCH, 7), counts)
    var hx = tensor(r, x, dims2(_BATCH, 7))
    var s1: List[Int] = [1, 5, 1]
    var pieces = _split(r, hx, s1, 1)
    var xs = pieces[1]
    var s2: List[Int] = [1, 4]
    var s3: List[Int] = [2, 2, 1]
    var left = _split(r, xs, s2, 1)
    var right = _split(r, xs, s3, 1)
    var parts: List[Int] = [left[0], right[1]]
    var axis1: List[Int] = [1]
    var xg = opi(r, _CONCAT, parts, axis1, List[Float32]())
    var lo: List[Float32] = [-0.5]
    var hi: List[Float32] = [0.5]
    var xc = op2(r, _MINIMUM, op2(r, _MAXIMUM, xg, tensor(r, lo, dims2(1, 1))), tensor(r, hi, dims2(1, 1)))
    var hc: List[Int] = [xc]
    var two: List[Float32] = [2.0]
    var xb = op2(r, _ADD, opi(r, _POW, hc, List[Int](), two), tensor(r, _init(0, "bias"), dims2(1, 3)))
    var hb: List[Int] = [xb]
    var mean_words: List[Int] = [1, 1]  # keepdims True, axis 1
    var want_m = values(r, reshape(r, opi(r, _MEAN_N, hb, mean_words, List[Float32]()), dims2(1, _BATCH)), dims2(1, _BATCH))
    var want_p = values(r, op1(r, _SOFTMAX, xc), dims2(_BATCH, 3))
    var argmax_words: List[Int] = [1, 1, 0]  # axis 1, keepdims True, int32 words
    var want_a = values(r, reshape(r, opi(r, _ARGMAX, hc, argmax_words, List[Float32]()), dims2(1, _BATCH)), dims2(1, _BATCH))
    _same(got[0], want_m, "ReduceMean")
    _same(got[1], want_p, "Softmax")
    _same(got[2], want_a, "ArgMax (int32 words)")


def _grouped_conv[E: Exec & Movable & Deinitable, KIND: Int](sid: Int, h: Int, mut r: OpsSession[E]) raises:
    var shape = dims4(_BATCH, 4, 8, 9)
    var res = _import[E, KIND](sid, h, 1, "x", shape)
    var out_dims = dims4(_BATCH, 6, 4, 5)
    require(_same_dims(res.out_dims[0], out_dims), "grouped conv: the output shape")
    var x = normal(_count(shape), 13)
    var counts: List[Int] = [_count(out_dims)]
    var y = _run_x[E, KIND](sid, res.pid, x, shape, counts)
    var gw = _init(1, "gw")
    var gb = _init(1, "gb")
    var dw = _init(1, "dw")
    var pads: List[Int] = [0, 0, 0, 1, 0, 0, 1, 1]
    var hp = _pad(r, tensor(r, x, shape), pads, 0.0)
    var hg = _conv(r, hp, tensor(r, gw, dims4(6, 2, 3, 3)), tensor(r, gb, dims2(1, 6)), 2, 0, 2)
    var want = values(r, _conv(r, hg, tensor(r, dw, dims4(6, 1, 3, 3)), 0, 1, 1, 6), out_dims)
    _same(y[0], want, "grouped conv: the program against the eager composition")
    var first = _conv64(_wide(x), _BATCH, 4, 8, 9, gw, 6, 2, 3, 3, gb, 2, 0, 1, 1, 1, 2)
    var want64 = _conv64(first, _BATCH, 6, 4, 5, dw, 6, 1, 3, 3, List[Float32](), 1, 1, 1, 1, 1, 6)
    allclose(y[0], want64, 1e-5, 1e-5, "grouped conv: the float64 reference")


def _padded_pools[E: Exec & Movable & Deinitable, KIND: Int](sid: Int, h: Int, mut r: OpsSession[E]) raises:
    var shape = dims4(_BATCH, 3, 7, 8)
    var res = _import[E, KIND](sid, h, 2, "x", shape)
    var out_dims = dims4(_BATCH, 3, 4, 4)
    for k in range(3):
        require(_same_dims(res.out_dims[k], out_dims), "padded pools: the output shape " + String(k))
    var x = normal(_count(shape), 17)
    var n = _count(out_dims)
    var counts: List[Int] = [n, n, n]
    var got = _run_x[E, KIND](sid, res.pid, x, shape, counts)
    var again = _run_x[E, KIND](sid, res.pid, x, shape, counts)
    var what: List[String] = ["max", "avg same", "avg include pad"]
    for k in range(3):
        _same(got[k], again[k], what[k] + ": a second run")
    var hx = tensor(r, x, shape)
    var p1: List[Int] = [0, 0, 1, 1, 0, 0, 1, 1]
    var c: List[Int] = [0, 0, 1, 1, 0, 0, 1, 0]
    var want_m = values(r, _pool(r, _MAXPOOL2D, _pad(r, hx, p1, Float32(-1.0) / Float32(0.0)), 3, 2), out_dims)
    var total = _conv(r, _pad(r, hx, c, 0.0), _full(r, dims4(3, 1, 3, 3), 1.0), 0, 2, 0, 3)
    var count = _conv(r, _pad(r, _full(r, dims4(1, 1, 7, 8), 1.0), c, 0.0), _full(r, dims4(1, 1, 3, 3), 1.0), 0, 2, 0, 1)
    var want_a0 = values(r, op2(r, _DIV, total, count), out_dims)
    var want_a1 = values(r, _pool(r, _AVGPOOL2D, _pad(r, hx, p1, 0.0), 3, 2), out_dims)
    _same(got[0], want_m, "MaxPool pads")
    _same(got[1], want_a0, "AveragePool SAME_LOWER count_include_pad=0")
    _same(got[2], want_a1, "AveragePool count_include_pad=1")
    exact(got[0], _f32(_pool64(x, _BATCH, 3, 7, 8, 3, 2, 1, 1, 1, 1, True, False)), "MaxPool: the float64 reference")
    allclose(got[1], _pool64(x, _BATCH, 3, 7, 8, 3, 2, 1, 1, 1, 0, False, False), 1e-5, 1e-6,
             "AveragePool SAME_LOWER: the float64 reference")
    allclose(got[2], _pool64(x, _BATCH, 3, 7, 8, 3, 2, 1, 1, 1, 1, False, True), 1e-5, 1e-6,
             "AveragePool include pad: the float64 reference")


comptime _EMB_VOCAB = 11
comptime _EMB_T = 4
comptime _EMB_D = 6


def _embedding_tail[E: Exec & Movable & Deinitable](mut r: OpsSession[E], e: Int) raises -> List[Float32]:
    """The rest of the test's `_ref_embedding_linear`: reshape (batch, t, d), @ proj, + proj_b."""
    var m = op2(r, _MATMUL, reshape(r, e, dims3(_BATCH, _EMB_T, _EMB_D)), tensor(r, _init(3, "proj"), dims2(_EMB_D, 3)))
    return values(r, op2(r, _ADD, m, tensor(r, _init(3, "proj_b"), dims2(1, 3))), dims3(_BATCH, _EMB_T, 3))


def _embedding_ids[E: Exec & Movable & Deinitable](mut r: OpsSession[E], ids: List[Int64]) raises -> List[Float32]:
    """`_ref_embedding_linear(s, f, s.ids(ids))`."""
    var table = tensor(r, _init(3, "table"), dims2(_EMB_VOCAB, _EMB_D))
    var hid = r.upload_ids(_ids_src(ids, _BATCH, _EMB_T), ShapeN.of(dims2(_BATCH, _EMB_T)))
    _ = ids
    var hs: List[Int] = [table, hid]
    var option: List[Int] = [-1]
    return _embedding_tail(r, opi(r, _EMBEDDING, hs, option, List[Float32]()))


def _embedding_array[E: Exec & Movable & Deinitable](mut r: OpsSession[E], ids: List[Int64]) raises -> List[Float32]:
    """`_ref_embedding_linear(s, f, ids)`: the ids array itself (the door's 8-word descriptor)."""
    var table = tensor(r, _init(3, "table"), dims2(_EMB_VOCAB, _EMB_D))
    var src = _ids_src(ids, _BATCH, _EMB_T)
    var hs: List[Int] = [table]
    var words: List[Int] = [src.addr, _NPY_INT64, src.itemsize, src.rows, src.cols, src.s0, src.s1, -1]
    var e = opi(r, _EMBEDDING, hs, words, List[Float32]())
    _ = ids
    return _embedding_tail(r, e)


def _embedding[E: Exec & Movable & Deinitable, KIND: Int](sid: Int, h: Int, mut r: OpsSession[E]) raises:
    var res = _import[E, KIND](sid, h, 3, "ids", dims2(_BATCH, _EMB_T))
    # the native import names dtypes "i64" (autograd/import_reader.mojo dtype_name); the
    # Python boundary renames it "int64" (importers/_native_import_bridge.py)
    require(len(res.in_names) == 1 and res.in_names[0] == "ids" and _same_dims(res.in_dims[0], dims2(_BATCH, _EMB_T))
            and res.in_dtypes[0] == "i64", "embedding_linear: the program input (ids, (3, 4), i64)")
    require(len(res.prog.in_kinds) == 1 and res.prog.in_kinds[0] == PROGRAM_IN_IDS,
            "embedding_linear: the input is an ids input")
    var out_dims = dims3(_BATCH, _EMB_T, 3)
    require(len(res.out_names) == 1 and res.out_names[0] == "y" and _same_dims(res.out_dims[0], out_dims),
            "embedding_linear: the output (y, (3, 4, 3))")
    var ids = _ids(_BATCH * _EMB_T, 13)
    var other = _ids(_BATCH * _EMB_T, 14)
    var counts: List[Int] = [_count(out_dims)]
    var y = _run_ids[E, KIND](sid, res.pid, ids, _BATCH, _EMB_T, counts)
    var again = _run_ids[E, KIND](sid, res.pid, ids, _BATCH, _EMB_T, counts)
    _same(again[0], y[0], "embedding_linear: a second run")
    var moved = _run_ids[E, KIND](sid, res.pid, other, _BATCH, _EMB_T, counts)
    var want = _embedding_ids(r, ids)
    var want_arr = _embedding_array(r, ids)
    var want_other = _embedding_ids(r, other)
    _same(y[0], want, "embedding_linear: the program against the eager composition")
    _same(want_arr, want, "embedding_linear: the ids tensor against the ids array")
    _same(moved[0], want_other, "embedding_linear: new ids against the eager composition")


def _params_of[E: Exec & Movable & Deinitable, KIND: Int](sid: Int, res: OnnxImport, dims: List[Int]) raises -> List[List[Float32]]:
    """`[s.numpy(h) for h in prog.params if s.shape(h) == dims]`."""
    var s = ops_session[E, KIND](sid)
    var want = ShapeN.of(dims)
    var out = List[List[Float32]]()
    for j in range(len(res.prog.params)):
        var hp = res.prog.params[j]
        if s[].shape_nd(hp).same(want):
            out.append(values(s[], hp, dims))
    return out^


def _batchnorm[E: Exec & Movable & Deinitable, KIND: Int](sid: Int, h: Int, mut r: OpsSession[E]) raises:
    var res = _import[E, KIND](sid, h, 4, "x", dims4(_BATCH, 4, 5, 5))
    var pd = dims4(1, 4, 1, 1)
    var params = _params_of[E, KIND](sid, res, pd)
    require(len(params) == 2, "the folded scale and shift are the program's parameters (found "
            + String(len(params)) + ")")
    var sc = tensor(r, _init(4, "bn_scale"), dims2(1, 4))
    var b = tensor(r, _init(4, "bn_bias"), dims2(1, 4))
    var mean = tensor(r, _init(4, "bn_mean"), dims2(1, 4))
    var var_h = tensor(r, _init(4, "bn_var"), dims2(1, 4))
    var fs = op2(r, _DIV, sc, op1(r, _SQRT, op2(r, _ADD, var_h, _full(r, dims2(1, 1), 1e-5))))
    var ft = op2(r, _SUB, b, op2(r, _MUL, mean, fs))
    _same(params[0], values(r, reshape(r, fs, pd), pd), "the folded scale")
    _same(params[1], values(r, reshape(r, ft, pd), pd), "the folded shift")


def _gemm_transb[E: Exec & Movable & Deinitable, KIND: Int](sid: Int, h: Int, mut r: OpsSession[E]) raises:
    var res = _import[E, KIND](sid, h, 5, "x", dims2(_BATCH, 6))
    var w1 = _params_of[E, KIND](sid, res, dims2(6, 5))
    require(len(w1) == 1, "one (6, 5) parameter (found " + String(len(w1)) + ")")
    var w = _init(5, "w1")  # (5, 6)
    var t = List[Float32](length=30, fill=0)
    for i in range(5):
        for j in range(6):
            t[j * 5 + i] = w[i * 6 + j]
    _same(w1[0], t, "the transposed Gemm weight")


def run_core_import_lower_fixture[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, h: Int, mut r: OpsSession[E], fixture_id: Int
) raises:
    """Fixture `fixture_id` on import session `sid` (model graph `h`), the
    eager reference on inference session `r`."""
    require(fixture_id >= 0 and fixture_id < CORE_IMPORT_LOWER_FIXTURES, "unknown import-lower fixture")
    if fixture_id == 0:
        _mixed[E, KIND](sid, h, r)
    elif fixture_id == 1:
        _grouped_conv[E, KIND](sid, h, r)
    elif fixture_id == 2:
        _padded_pools[E, KIND](sid, h, r)
    elif fixture_id == 3:
        _embedding[E, KIND](sid, h, r)
    elif fixture_id == 4:
        _batchnorm[E, KIND](sid, h, r)
    else:
        _gemm_transb[E, KIND](sid, h, r)


# ---- The host column ----
# Each fixture imports into its own host session and builds its eager
# reference on a second, private host session; the fixtures run across host
# cores with a private failure slot each.
#
# onnx_import reaches its session and its model graph through the
# process-wide tables (autograd/ops_api.mojo _OPS_SESSIONS,
# autograd/import_reader.mojo _ONNX_GRAPHS), whose open/close append to and
# pop from unlocked lists (core_program_loop says the same). So the sessions
# are opened and the models read on this thread first, the tasks only look
# entries up (no table writes while they run), and everything is closed on
# this thread after the parallel section.
def _run_cpu(mut report: Report) raises:
    var cases = report.cases(CORE_IMPORT_LOWER_FIXTURES)
    var sids = List[Int]()
    var graphs = List[Int]()
    for fixture_id in range(CORE_IMPORT_LOWER_FIXTURES):
        sids.append(ops_session_open[HostExec, OPS_HOST](HostExec(), False))
        graphs.append(onnx_read(_model_path(fixture_id)))
    def task(fixture_id: Int) {imm cases, imm sids, imm graphs}:
        try:
            var r = OpsSession[HostExec](HostExec(), False)
            try:
                run_core_import_lower_fixture[HostExec, OPS_HOST](sids[fixture_id], graphs[fixture_id], r, fixture_id)
            except error:
                r.close()
                raise error
            r.close()
        except error:
            cases.fail(fixture_id, error)
    report.parallel(task, CORE_IMPORT_LOWER_FIXTURES)
    for fixture_id in range(CORE_IMPORT_LOWER_FIXTURES):
        onnx_close(graphs[fixture_id])
        ops_session_close[HostExec, OPS_HOST](sids[fixture_id])


# ---- The device column ----
# The same fixtures on device sessions, in order.
#
# DeviceExec shares sequence_ctx() and its pool, so independent concurrent
# sessions would race that context's ownership (as in ops_higher_gpu); the
# fixtures run one after another and each kernel keeps its own parallelism.
def _run_gpu(mut report: Report) raises:
    var cases = report.cases(CORE_IMPORT_LOWER_FIXTURES)
    for fixture_id in range(CORE_IMPORT_LOWER_FIXTURES):
        try:
            var sid = ops_session_open[DeviceExec, OPS_DEVICE](DeviceExec(), False)
            var h = onnx_read(_model_path(fixture_id))
            var r = OpsSession[DeviceExec](DeviceExec(), False)
            try:
                run_core_import_lower_fixture[DeviceExec, OPS_DEVICE](sid, h, r, fixture_id)
            except error:
                r.close()
                onnx_close(h)
                ops_session_close[DeviceExec, OPS_DEVICE](sid)
                raise error
            r.close()
            onnx_close(h)
            ops_session_close[DeviceExec, OPS_DEVICE](sid)
        except error:
            cases.fail(fixture_id, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("core_import_lower", cpu=True, gpu=True, serial="", run=run_cases)
