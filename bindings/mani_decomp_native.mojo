# SPDX-License-Identifier: Apache-2.0
# The decomp lane's manifold estimators, one native call per public method
# (python/mojolearn/_expansion_decomp.py): Isomap (`fit`, `fit_transform`,
# `transform`, `reconstruction_error`), ClassicalMDS (`fit`, `fit_transform`),
# MDS (`fit`, `fit_transform`) and LocallyLinearEmbedding (`fit`,
# `fit_transform`, `transform`). The native provider selects the decomp binding
# for the requested or fitted mode and the process-default helper kit; this owner
# makes the Python methods' refusals (same wording, same order) and runs their
# steps statement for statement on the kit (bindings/mani_kit.mojo): the same
# x_decomp entries, operands, float32 scalars and order, every matrix where the
# Python kit kept it (a GPU binding's results stay resident between steps), the
# Lanczos batches, the shift-invert LLE solve and SMACOF in the same entries
# with the same float64 stopping tests. The fitted `_M` state is the Python
# kit's own objects, so transform reads it where fit left it. Bits unchanged.
from std.math import sqrt
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_fields, py_setitem, py_contains
from bindings.py_attr import py_attr
from bindings.classical_attributes import classical_connect_numpy
from providers.python.estimator_binding_pyapi import estimator_binding_mode
from bindings.family_common import (
    family_repr, family_str, family_is_str, family_is_int, family_type_name, family_call, family_call_kw,
    family_field,
)
from bindings.mani_kit import (
    MKit, mk_from_input, mk_r, mk_c, mk_n, mk_rows, mk_t, mk_take_cols, mk_col_range, mk_desc_range, mk_out,
    mk_out2, mk_store, mk_store_ptr, mk_at, mk_of, mk_fill, mk_zeros, mk_reshape, mk_dev_view, mk_new,
    mk_on_dev, mk_take_rows_list, mk_ints, mk_addr, mk_store_addr, mk_store_len,
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_SQRT, OP_SQ, OP_SCALE, OP_FMA, OP_ADDS, OP_GTS, OP_MIN, OP_LE,
    SEL_MAXABS, SEL_MAX, SEL_MIN,
)

comptime _LANCZOS_MIN_N = 200
comptime _LANCZOS_MAX_NC = 10
comptime _LANCZOS_TOL = 1e-7
comptime _LANCZOS_MAX_M = 600
comptime _LANCZOS_DEV_MAX_FLOATS = 1 << 26
comptime _LLE_ITER_MIN_N = 200
comptime _LLE_ITER_MAX_K = 10
comptime _LLE_SUBSPACE_TOL = 1e-5
comptime _LLE_STALL_TOL = 1e-2
comptime _LLE_NULL_GUARD = 1e-3
comptime _LLE_NULL_FLOOR = 8.0
comptime _F32_EPS = 1.1920928955078125e-07


def _mani_finite(v: Float64) -> Bool:
    """`math.isfinite(v)`."""
    return v - v == 0.0


def _mani_eq(value: PythonObject, text: String) raises -> Bool:
    """`value == text`, as a truth value."""
    var b = Python.import_module("builtins")
    return Bool(py=py_call(b.bool, py_call(Python.import_module("operator").eq, value, PythonObject(text))))


def _mani_in(value: PythonObject, choices: PythonObject) raises -> Bool:
    """`value in choices`."""
    return Bool(py=py_call(Python.import_module("operator").contains, choices, value))


def _mani_truth(value: PythonObject) raises -> Bool:
    return Bool(py=py_call(Python.import_module("builtins").bool, value))


def _mani_int(value: PythonObject) raises -> Int:
    """`int(value)`."""
    return Int(py=family_call(Python.import_module("builtins").int, value))


def _mani_float(value: PythonObject) raises -> Float64:
    """`float(value)`."""
    return Float64(py=family_call(Python.import_module("builtins").float, value))


def _mani_f32(v: Float64) -> Float64:
    """`_f32(v)`: rounded once to float32."""
    return Float64(Float32(v))


def _mani_mode(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_mode(self.numeric_mode)`."""
    var mode = py_get(fields, PythonObject("numeric_mode"), PythonObject(None))
    if mode is None:
        return context["backend_fields"]["_DEFAULT_MODE"]
    return mode


def _mani_kit(mode: PythonObject, binding: PythonObject, context: PythonObject) raises -> MKit:
    """`_Kit(mode)`: `binding` is the decomp binding the shell resolved for
    `mode`."""
    return MKit(String(py=py_call(Python.import_module("builtins").str, mode)), binding, context)


def _mani_host_kit(binding: PythonObject, context: PythonObject) raises -> MKit:
    """`_host_kit()`: the default-mode kit (`binding` resolved for it)."""
    return MKit(String(py=context["backend_fields"]["_DEFAULT_MODE"]), binding, context)


def _mani_check(model: PythonObject, attr: String) raises:
    """`_check(attr)`."""
    if not py_contains(py_fields(model), PythonObject(attr)):
        raise Error("RuntimeError: " + family_type_name(model) + " is not fitted; call fit first")


def _mani_seed(random_state: PythonObject) raises -> PythonObject:
    """`_seed_of(random_state)`."""
    if random_state is None:
        return PythonObject(0)
    if not family_is_int(random_state):
        raise Error("TypeError: random_state must be None or an int (the counter-based RNG takes an integer seed)")
    return random_state


def _mani_eigen_solver(name: PythonObject, who: String) raises:
    """`_exact_eigen_solver(name, who)`."""
    if _mani_eq(name, "arpack"):
        raise Error("NotImplementedError: " + who + ": eigen_solver='arpack' is not implemented; the exact solve "
                    + "runs for 'auto' and 'dense', and running it under ARPACK's name "
                    + "would be a silent substitution (x_decomp/NOT_IMPLEMENTED.tsv)")
    if not _mani_in(name, Python.tuple(PythonObject("auto"), PythonObject("dense"))):
        raise Error(who + ": eigen_solver must be 'auto', 'dense' or 'arpack', got " + family_repr(name))


def _mani_metric_spec(metric: PythonObject, p_in: PythonObject, metric_params: PythonObject,
                      who: String, context: PythonObject) raises -> Tuple[Int, Float64]:
    """`_metric_spec(metric, p, metric_params, who)`: (kind, p)."""
    var b = Python.import_module("builtins")
    var p = p_in
    if _mani_truth(metric_params):
        var extra = py_call(Python.import_module("operator").sub, py_call(b.set, metric_params),
                            py_call(b.set, Python.list(PythonObject("p"))))
        if _mani_truth(extra):
            raise Error("NotImplementedError: " + who + ": metric_params " + family_repr(py_call(b.sorted, extra))
                        + " are not carried")
        p = py_call(metric_params.get, PythonObject("p"), p)
    if _mani_in(metric, Python.tuple(PythonObject("euclidean"), PythonObject("l2"))):
        return (Int(0), Float64(2.0))
    if _mani_eq(metric, "minkowski"):
        var pw = _mani_float(p)
        if not pw > 0:
            raise Error(who + ": minkowski p must be > 0")
        if pw == 2.0:
            return (Int(0), Float64(2.0))
        if pw == 1.0:
            return (Int(1), Float64(1.0))
        if pw > 0 and not _mani_finite(pw):
            return (Int(2), Float64(0.0))
        return (Int(3), pw)
    var kinds = context["family_mani_pd_kind"]
    if family_is_str(metric) and py_contains(kinds, metric):
        return (Int(py=kinds[metric]), Float64(0.0))
    raise Error("NotImplementedError: " + who + ": metric=" + family_repr(metric) + " is not carried (euclidean, "
                + "minkowski p, manhattan, chebyshev, cosine)")


def _mani_dist(mut k: MKit, a: PythonObject, b: PythonObject, kind: Int, pw: Float64,
               same: Bool) raises -> PythonObject:
    """`_dist(k, A, B, kind, pw, same)`."""
    if kind == 0:
        var sq = k.sqdist(a, b)
        return k.ew1(OP_SQRT, sq, 0.0)
    var d = k.pdist(a, b, kind, pw)
    if same and kind == 4:
        k.fill0(d, 0, mk_r(b) + 1, min(mk_r(a), mk_r(b)))
    return d


def _mani_knn(mut k: MKit, q: PythonObject, x: PythonObject, nn: Int, exclude_self: Bool, kind: Int,
              pw: Float64) raises -> Tuple[PythonObject, PythonObject]:
    """`_knn_mats(k, Q, X, nn, exclude_self, kind, pw)`."""
    var d: PythonObject
    if kind == 0:
        d = k.sqdist(q, x)
    else:
        d = _mani_dist(k, q, x, kind, pw, exclude_self)
    return k.graph_knn(d, nn, exclude_self)


def _mani_center(mut k: MKit, kk: PythonObject) raises -> Tuple[PythonObject, PythonObject, PythonObject]:
    """`_center_kernel(k, K)`: (Kc, column means 1 x n, total mean 1 x 1)."""
    var n = mk_r(kk)
    var col = k.ew1(OP_SCALE, k.colsum(kk), 1.0 / Float64(n))
    var allm = k.ew1(OP_SCALE, k.total(col), 1.0 / Float64(n))
    var row = k.ew1(OP_SCALE, k.rowsum(kk), 1.0 / Float64(mk_c(kk)))
    var t1 = k.ew2(OP_SUB, kk, col)
    var t2 = k.ew2(OP_SUB, t1, row)
    var kc = k.ew2(OP_ADD, t2, allm)
    return (kc, col, allm)


def _mani_kdot(mut k: MKit, a: PythonObject, b: PythonObject) raises -> Float64:
    """`_kdot(k, a, b)`: float(a^T b)."""
    var p = k.mm(a, b, True, False)
    return mk_at(p, 0, k.ctx)


def _mani_lanczos_top(mut k: MKit, a: PythonObject, nc: Int) raises -> Tuple[Bool, PythonObject, PythonObject]:
    """`_lanczos_top(k, A, nc)`: (found, values, vectors); not found when the
    basis cap is reached unconverged (the caller then runs the exact solve)."""
    var ctx = k.ctx
    var n = mk_r(a)
    var cap = min(n, _LANCZOS_MAX_M)
    if (cap + 1) * n > _LANCZOS_DEV_MAX_FLOATS:
        return (False, PythonObject(None), PythonObject(None))
    var q = k.ew1(OP_ADDS, k.rand(n, 1, PythonObject(0x1A2C05), 91, 0), -0.5)
    q = k.ew1(OP_SCALE, q, 1.0 / sqrt(_mani_kdot(k, q, q)))
    var ab = py_call(ctx["array_type"], PythonObject("f"), py_call(ctx["native_bytes_type"], PythonObject(2 * cap * 4)))
    var resident = k.res()
    var qd: PythonObject
    var abd = PythonObject(None)
    var aid = PythonObject(None)
    if resident:
        qd = k.dout(cap + 1, n)
        k.place_rows(qd, mk_reshape(q, 1, n, ctx), 0)
        abd = k.dout(1, 2 * cap)
        aid = k.did(a)
    else:
        qd = mk_zeros(cap + 1, n, ctx)
        k.place_rows(qd, mk_reshape(q, 1, n, ctx), 0)
    var alphas = List[Float64]()
    var betas = List[Float64]()
    var m = min(n, max(2 * nc + 1, 20))
    var j = 0
    var stop = False
    var th = PythonObject(None)
    var yv = PythonObject(None)
    var top = List[Int]()
    while True:  # driver-loop: over Lanczos basis growth rounds (at most log2(_LANCZOS_MAX_M / 20) + 1)
        if j < m and not stop:
            # `_lanczos_batch`: steps j .. m - 1, then their alphas and betas
            # up to the first breakdown
            if resident:
                _ = family_call(k.fn("x_decomp_dev_lanczos"), aid, qd._d.id, abd._d.id, mk_ints([n, j, m, cap]))
                _ = family_call(k.fn("x_decomp_dev_download"), abd._d.id, mk_store_addr(ab), PythonObject(2 * cap))
                _ = ab  # keep the Python object alive through the native call above
            else:
                var pa = mk_addr(a, ctx)
                _ = family_call(k.fn("x_decomp_lanczos"), pa, mk_addr(qd, ctx), mk_store_addr(ab),
                                mk_ints([n, j, m, cap]))
                _ = ab  # keep the Python object alive through the native call above
            var abp = mk_store_ptr(ab)
            _ = ab  # keep the Python object alive through the native call above
            var reached = m
            var broke = False
            for jj in range(j, m):  # driver-loop: over this batch's Lanczos steps (at most _LANCZOS_MAX_M)
                var av = Float64(abp[jj])
                _ = ab  # keep the Python object alive through the native call above
                var bv = Float64(abp[cap + jj])
                _ = ab  # keep the Python object alive through the native call above
                alphas.append(av)
                betas.append(bv)
                if bv <= 1e-30 * max(1.0, abs(av)) or jj + 1 == n:
                    reached = jj + 1
                    broke = True
                    break
            j = reached
            stop = broke
        # the tridiagonal T from the batch's alphas and betas
        var tvals = List[Float64](length=j * j, fill=0.0)
        for i in range(j):  # driver-loop: over the Lanczos steps' scalars into T (at most _LANCZOS_MAX_M)
            tvals[i * j + i] = alphas[i]
            if i + 1 < j:
                tvals[i * j + i + 1] = betas[i]
                tvals[(i + 1) * j + i] = betas[i]
        var eig = k.eigh(mk_of(tvals, j, j, ctx))
        th = eig[0]
        yv = eig[1]
        top = List[Int]()
        var stop_at = max(j - 1 - nc, -1)
        var t = j - 1
        while t > stop_at:  # small-loop(components: the wanted Ritz positions): bounded by n_components
            top.append(t)
            t -= 1
        if len(top) < nc:
            return (False, PythonObject(None), PythonObject(None))
        var big = k.word(k.reduce(mk_take_cols(th, top, ctx), SEL_MAXABS), 0)
        if big == 0.0:
            big = 1.0
        var last = mk_take_cols(mk_rows(yv, j - 1, j, ctx), top, ctx)
        var res = k.word(k.reduce(k.ew1(OP_SCALE, last, betas[j - 1]), SEL_MAXABS), 0)
        if stop or res <= _LANCZOS_TOL * big:
            break
        if m >= cap:
            return (False, PythonObject(None), PythonObject(None))
        m = min(min(n, 2 * m), _LANCZOS_MAX_M)
    var yt = mk_take_cols(yv, top, ctx)
    var qm: PythonObject
    if mk_on_dev(qd):
        qm = mk_dev_view(qd._d, j, n, ctx)
    else:
        var store = mk_store(qd, ctx)
        qm = mk_new(py_call(py_attr(store, "__getitem__"), py_call(Python.import_module("builtins").slice, PythonObject(0),
                                                       PythonObject(j * n))), j, n, ctx)
    var v = k.mm(qm, yt, True, False)
    _ = ab^
    _ = abd^
    return (True, mk_take_cols(th, top, ctx), v)


def _mani_top_eig(mut k: MKit, a: PythonObject, nc: Int, topk: Bool) raises -> Tuple[PythonObject, PythonObject]:
    """`_top_eig(k, A, nc, topk)`: the nc largest eigenpairs (descending),
    vectors in columns, each column signed by svd_flip(u_based_decision=True)."""
    var n = mk_r(a)
    var found = False
    var w = PythonObject(None)
    var v = PythonObject(None)
    if topk and n > _LANCZOS_MIN_N and nc < _LANCZOS_MAX_NC:
        var got = _mani_lanczos_top(k, a, nc)
        found = got[0]
        w = got[1]
        v = got[2]
    if not found:
        var eig = k.eigh(a)
        var order = mk_desc_range(n - 1, nc)
        w = mk_take_cols(eig[0], order, k.ctx)
        v = mk_take_cols(eig[1], order, k.ctx)
    var signs = k.absmax_signs(v, True)
    return (w, k.ew2(OP_MUL, v, signs))


# ---------------------------------------------------------------- ClassicalMDS
def _cmds_core(mut k: MKit, source: PythonObject, metric: PythonObject, metric_params: PythonObject,
               nc_param: PythonObject, context: PythonObject) raises -> Tuple[PythonObject, PythonObject, PythonObject, PythonObject]:
    """`ClassicalMDS.fit`'s steps: (dissimilarity_matrix_, eigenvalues_,
    embedding_m_, M)."""
    var m = mk_from_input(source, "X", context)
    var d2: PythonObject
    var dissimilarity: PythonObject
    if _mani_eq(metric, "precomputed"):
        d2 = k.ew1(OP_SQ, m, 0.0)
        dissimilarity = mk_out2(m, context)
    elif _mani_eq(metric, "euclidean"):
        d2 = k.sqdist(m, m)
        dissimilarity = mk_out2(k.ew1(OP_SQRT, d2, 0.0), context)
    else:
        var spec = _mani_metric_spec(metric, PythonObject(2), metric_params, "ClassicalMDS", context)
        var dm = _mani_dist(k, m, m, spec[0], spec[1], True)
        d2 = k.ew1(OP_SQ, dm, 0.0)
        dissimilarity = mk_out2(dm, context)
    var centered = _mani_center(k, k.ew1(OP_SCALE, d2, -0.5))
    var eig = _mani_top_eig(k, centered[0], _mani_int(nc_param), True)
    var w = eig[0]
    var v = eig[1]
    var values = mk_out(w, Python.tuple(mk_c(w)), context)
    var embedding = k.ew2(OP_MUL, v, k.ew1(OP_SQRT, w, 0.0))
    return (dissimilarity, values, embedding, m)


def _cmds_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`ClassicalMDS.fit(X)`; returns embedding_."""
    classical_connect_numpy(context)
    var fields = py_fields(model)
    var mode = _mani_mode(fields, context)
    py_setitem(fields, "numeric_mode_", mode)
    var k = _mani_kit(mode, args[0], context)
    var got = _cmds_core(k, args[2], family_field(fields, "metric"), family_field(fields, "metric_params"),
                         family_field(fields, "n_components"), context)
    py_setitem(fields, "dissimilarity_matrix_", got[0])
    py_setitem(fields, "eigenvalues_", got[1])
    py_setitem(fields, "embedding_m_", got[2])
    var embedding = mk_out2(got[2], context)
    py_setitem(fields, "embedding_", embedding)
    py_setitem(fields, "n_features_in_", PythonObject(mk_c(got[3])))
    return embedding


# ---------------------------------------------------------------- MDS
def _mds_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`MDS.fit_transform(X, init=init)`; returns embedding_. args = (decomp
    binding, default-mode binding, X, init)."""
    classical_connect_numpy(context)
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    var mode = _mani_mode(fields, context)
    py_setitem(fields, "numeric_mode_", mode)
    var k = _mani_kit(mode, args[0], context)
    var m = mk_from_input(args[2], "X", context)
    var metric = family_field(fields, "metric")
    var dis: PythonObject
    if _mani_eq(metric, "precomputed"):
        dis = m
    elif _mani_eq(metric, "euclidean"):
        dis = k.ew1(OP_SQRT, k.sqdist(m, m), 0.0)
    else:
        var spec = _mani_metric_spec(metric, PythonObject(2), family_field(fields, "metric_params"), "MDS", context)
        dis = _mani_dist(k, m, m, spec[0], spec[1], True)
    py_setitem(fields, "dissimilarity_matrix_", mk_out2(dis, context))
    var metric_mds = _mani_truth(family_field(fields, "metric_mds"))
    var normalized = family_field(fields, "normalized_stress")
    var norm = (normalized is PythonObject(True)) or (_mani_eq(normalized, "auto") and not metric_mds)
    py_setitem(fields, "_norm", PythonObject(norm))
    var n = mk_r(dis)
    var nc = _mani_int(family_field(fields, "n_components"))
    var seed = _mani_seed(family_field(fields, "random_state"))
    var init_arg = args[3]
    var init = family_field(fields, "init")
    var starts = List[PythonObject]()
    if not (init_arg is None):
        starts.append(mk_from_input(init_arg, "init", context))
    elif _mani_eq(init, "classical_mds"):
        # ClassicalMDS(nc, metric="precomputed", numeric_mode=self.numeric_mode_).fit(Dis.out()).embedding_m_
        var inner = _mani_kit(mode, args[0], context)
        var got = _cmds_core(inner, mk_out2(dis, context), PythonObject("precomputed"), PythonObject(None),
                             PythonObject(nc), context)
        _ = mk_out2(got[2], context)  # the inner fit's embedding_ (it reads the start back to the host)
        starts.append(got[2])
    elif _mani_eq(init, "random"):
        var restarts = _mani_int(family_field(fields, "n_init"))
        for r in range(restarts):  # driver-loop: over random restarts (n_init)
            starts.append(k.rand(n, nc, seed, 70 + r, 0))
    else:
        raise Error("init must be 'random', 'classical_mds' or an array")
    if not metric_mds and n * n > 2147483647:
        raise Error("MDS(metric_mds=False): " + String(n) + " rows exceed the 32-bit pair index (n * n <= 2**31 - 1)")
    var s = k.vstack(starts) if len(starts) > 1 else starts[0]
    var y = mk_zeros(n, nc, context)
    var pd = mk_addr(dis, context)
    var ps = mk_addr(s, context)
    _ = s  # keep the Python object alive through the native call above
    var params = mk_ints([n, nc, len(starts), _mani_int(family_field(fields, "max_iter")), 1 if metric_mds else 0,
                          1 if norm else 0])
    var tols = Python.list(PythonObject(_mani_float(family_field(fields, "eps"))))
    var result = family_call(k.fn("x_decomp_mds_fit"), pd, ps, mk_addr(y, context), params, tols)
    _ = y  # keep the Python object alive through the native call above
    _ = s  # keep the Python object alive through the native call above
    py_setitem(fields, "stress_", py_call(b.float, result[0]))
    _ = y  # keep the Python object alive through the native call above
    py_setitem(fields, "n_iter_", py_call(b.int, result[1]))
    _ = y  # keep the Python object alive through the native call above
    var embedding = mk_out2(y, context)
    py_setitem(fields, "embedding_", embedding)
    py_setitem(fields, "n_features_in_", PythonObject(mk_c(m)))
    _ = s^
    _ = dis^
    return embedding


# ---------------------------------------------------------------- Isomap
def _iso_graph(model: PythonObject, fields: PythonObject, mut k: MKit, m: PythonObject, n: Int, kind: Int,
               pw: Float64) raises -> Tuple[PythonObject, PythonObject]:
    """`Isomap._graph`: (the dense neighbor graph, n_neighbors or None)."""
    var radius = family_field(fields, "radius")
    var wg: PythonObject
    var nn_out = PythonObject(None)
    if not (radius is None):
        var d = _mani_dist(k, m, m, kind, pw, True)
        wg = k.graph_radius(d, _mani_f32(_mani_float(radius)))
    else:
        var nn = _mani_int(family_field(fields, "n_neighbors"))
        if nn >= n:
            # scikit-learn's kneighbors refusal (the query point itself is
            # one of the n_neighbors + 1 it asks for)
            raise Error("Expected n_neighbors <= n_samples_fit, but n_neighbors = " + String(nn + 1)
                        + ", n_samples_fit = " + String(n) + ", n_samples = " + String(n))
        var got = _mani_knn(k, m, m, nn, True, kind, pw)
        var idx = got[0]
        var sq = got[1]
        if kind == 0:
            sq = k.ew1(OP_SQRT, sq, 0.0)
        wg = k.graph_knn_dense(idx, sq, n)
        nn_out = PythonObject(nn)
    var comp = k.graph_components(wg)
    var count = comp[1]
    py_setitem(fields, "n_connected_components_", PythonObject(count))
    if count > 1:
        var keywords = Python.dict()
        py_setitem(keywords, "stacklevel", PythonObject(2))
        _ = family_call_kw(Python.import_module("warnings").warn, keywords,
                           PythonObject("The number of connected components of the neighbors graph is "
                                        + String(count) + " > 1. Completing the graph to fit Isomap might be slow."))
        var d = _mani_dist(k, m, m, kind, pw, True)
        k.graph_join(wg, d, comp[0], count)
    return (wg, nn_out)


def _iso_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Isomap.fit(X)`; returns embedding_."""
    classical_connect_numpy(context)
    var fields = py_fields(model)
    var mode = _mani_mode(fields, context)
    py_setitem(fields, "numeric_mode_", mode)
    var neighbors = family_field(fields, "n_neighbors")
    var radius = family_field(fields, "radius")
    if not (neighbors is None) and not (radius is None):
        raise Error("Both n_neighbors and radius are provided. Use Isomap(radius=" + family_str(radius)
                    + ", n_neighbors=None) if intended to use radius-based neighbors")
    if neighbors is None and radius is None:
        raise Error("Isomap: one of n_neighbors and radius must be set")
    if not (radius is None) and not (_mani_float(radius) >= 0):
        raise Error("Isomap: radius must be >= 0")
    if not _mani_in(family_field(fields, "path_method"), Python.tuple(PythonObject("auto"), PythonObject("FW"),
                                                                       PythonObject("D"))):
        raise Error("path_method must be 'auto', 'FW' or 'D'")
    var solver = family_field(fields, "eigen_solver")
    _mani_eigen_solver(solver, "Isomap")
    var spec = _mani_metric_spec(family_field(fields, "metric"), family_field(fields, "p"),
                                 family_field(fields, "metric_params"), "Isomap", context)
    var kind = spec[0]
    var pw = spec[1]
    py_setitem(fields, "_kind", PythonObject(kind))
    py_setitem(fields, "_pw", PythonObject(pw))
    var k = _mani_kit(mode, args[0], context)
    var m = mk_from_input(args[2], "X", context)
    var n = mk_r(m)
    var graph = _iso_graph(model, fields, k, m, n, kind, pw)
    var d = k.dijkstra(graph[0])
    py_setitem(fields, "dist_matrix_m_", d)
    py_setitem(fields, "dist_matrix_", mk_out2(d, context))
    var g = k.ew1(OP_SCALE, k.ew1(OP_SQ, d, 0.0), -0.5)
    var centered = _mani_center(k, g)
    var kc = centered[0]
    py_setitem(fields, "_k_col", centered[1])
    py_setitem(fields, "_k_all", centered[2])
    var topk = _mani_in(solver, Python.tuple(PythonObject("auto"), PythonObject("arpack")))
    var eig = _mani_top_eig(k, kc, _mani_int(family_field(fields, "n_components")), topk)
    py_setitem(fields, "eigenvalues_m_", eig[0])
    py_setitem(fields, "eigenvectors_m_", eig[1])
    var em = k.ew2(OP_MUL, eig[1], k.ew1(OP_SQRT, eig[0], 0.0))
    py_setitem(fields, "embedding_m_", em)
    var embedding = mk_out2(em, context)
    py_setitem(fields, "embedding_", embedding)
    py_setitem(fields, "_fit_X", m)
    py_setitem(fields, "_knn", graph[1])
    py_setitem(fields, "n_features_in_", PythonObject(mk_c(m)))
    py_setitem(fields, "_Kc", kc)
    return embedding


def _iso_transform(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Isomap.transform(X)`."""
    _mani_check(model, "embedding_m_")
    classical_connect_numpy(context)
    var fields = py_fields(model)
    var k = _mani_kit(family_field(fields, "numeric_mode_"), args[0], context)
    var q = mk_from_input(args[2], "X", context)
    var fit_x = family_field(fields, "_fit_X")
    var n = mk_r(fit_x)
    var d = family_field(fields, "dist_matrix_m_")
    var knn = family_field(fields, "_knn")
    var kind = _mani_int(family_field(fields, "_kind"))
    var pw = _mani_float(family_field(fields, "_pw"))
    var g = PythonObject(None)
    if knn is None:
        # `_radius_geodesic`
        var radius = family_field(fields, "radius")
        var r = _mani_f32(_mani_float(radius))
        var dq = _mani_dist(k, q, fit_x, kind, pw, False)
        var within = k.ew2(OP_LE, dq, mk_of([r], 1, 1, context))
        var hits = k.ew1(OP_GTS, k.rowsum(within), 0.5)
        var qr = mk_r(q)
        if qr > 0 and mk_at(k.total(hits), 0, context) != Float64(qr):
            var store = mk_store(hits, context)
            var hp = mk_store_ptr(store)
            _ = store  # keep the Python object alive through the native call above
            var first = 0
            for i in range(mk_store_len(store)):  # cpu-route: the refusal names the first query row without a training row in reach
                if Float64(hp[i]) == 0.0:
                    first = i
                    break
                _ = store  # keep the Python object alive through the native calls above
            raise Error("Isomap.transform: query row " + String(first) + " has no training row within radius "
                        + family_str(radius))
        g = k.graph_radius_geo(dq, d, r)
    else:
        var nn = _mani_int(knn)
        # the neighbour matrices as they come (indices as exact floats),
        # each neighbour's rows of D gathered on the device
        var got = _mani_knn(k, q, fit_x, nn, False, kind, pw)
        var im = got[0]
        var sq = got[1]
        if kind == 0:
            sq = k.ew1(OP_SQRT, sq, 0.0)
        for a in range(nn):  # driver-loop: over neighbor ranks (n_neighbors), one gather-and-min step each
            var rows = k.take_rows_at(d, im, mk_r(q), nn, a, 0)
            var cand = k.ew2(OP_ADD, rows, mk_take_cols(sq, mk_col_range(a, a + 1), context))
            if g is None:
                g = cand
            else:
                g = k.ew2(OP_MIN, g, cand)
    g = k.ew1(OP_SCALE, k.ew1(OP_SQ, g, 0.0), -0.5)
    var row = k.ew1(OP_SCALE, k.rowsum(g), 1.0 / Float64(n))
    var t1 = k.ew2(OP_SUB, g, family_field(fields, "_k_col"))
    var t2 = k.ew2(OP_SUB, t1, row)
    var kc = k.ew2(OP_ADD, t2, family_field(fields, "_k_all"))
    var v = k.ew2(OP_DIV, family_field(fields, "eigenvectors_m_"),
                  k.ew1(OP_SQRT, family_field(fields, "eigenvalues_m_"), 0.0))
    return mk_out2(k.mm(kc, v, False, False), context)


def _iso_reconstruction_error(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Isomap.reconstruction_error()`: a difference of two nearly equal sums,
    accumulated in binary64 (`_dsum_sq`)."""
    _mani_check(model, "embedding_m_")
    var fields = py_fields(model)
    var k = _mani_kit(family_field(fields, "numeric_mode_"), args[0], context)
    var kc = family_field(fields, "_Kc")
    var t = k.dsum_sq(kc) - k.dsum_sq(family_field(fields, "eigenvalues_m_"))
    if t > 0:
        return PythonObject(sqrt(t) / Float64(mk_r(kc)))
    return PythonObject(0.0)


# ---------------------------------------------------------------- LocallyLinearEmbedding
def _lle_smallest(mut k: MKit, f: PythonObject, nc: Int, max_iter: Int,
                  seed: PythonObject) raises -> Tuple[Bool, PythonObject, PythonObject]:
    """`_lle_smallest(k, F, nc, max_iter, seed)`: (found, embedding, sv) by
    the shift-invert subspace iteration on the deflated F0; not found sends
    the caller to the full SVD."""
    var ctx = k.ctx
    var n = mk_c(f)
    if mk_r(f) != n:
        return (False, PythonObject(None), PythonObject(None))
    var n1 = n - 1
    var p = min(n1, max(2 * nc + 1, 20))
    if p <= nc:
        return (False, PythonObject(None), PythonObject(None))
    var rn = 1.0 / sqrt(Float64(n))
    var hv = List[Float64](length=n, fill=rn)
    hv[n - 1] = rn - 1.0
    var coef = 2.0 / (Float64(n - 1) * (rn * rn) + (rn - 1.0) * (rn - 1.0))
    var h = mk_of(hv, n, 1, ctx)
    var un = mk_fill(rn, n, 1, ctx)
    var hrow = mk_fill(rn, 1, n1, ctx)
    var fh = k.mm(f, h, False, False)
    var hfull_v = List[Float64](length=n, fill=rn)
    hfull_v[n - 1] = 0.0
    var mask_v = List[Float64](length=n, fill=1.0)
    mask_v[n - 1] = 0.0
    var last_v = List[Float64](length=n, fill=0.0)
    last_v[n - 1] = rn
    var hfull = mk_of(hfull_v, 1, n, ctx)
    var mask = mk_of(mask_v, 1, n, ctx)
    var last = mk_of(last_v, 1, n, ctx)
    var dd = k.ew2(OP_SUB, f, k.ew1(OP_SCALE, k.mm(fh, hfull, False, False), coef))
    var f0 = k.ew3(OP_FMA, dd, mask, last, 0.0)
    var fu = k.mm(f, un, False, False)
    var g = sqrt(k.dsum_sq(fu))
    var tot = mk_at(k.total(k.ew1(OP_SQ, f, 0.0)), 0, ctx)
    # max(tot, 0.0): Python's order (tot unless 0.0 is larger)
    var kept = Float64(0.0) if 0.0 > tot else tot
    var rms = sqrt(kept / Float64(n))
    if not (g <= _LLE_NULL_GUARD * rms):
        return (False, PythonObject(None), PythonObject(None))
    var floor = _LLE_NULL_FLOOR * _F32_EPS * rms
    var lu: PythonObject
    var pm: PythonObject
    var im: PythonObject
    var big: Float64
    if (k.w4_flags() & 1) != 0:
        var got = k.lu_dev_aux(f0, True)
        lu = got[0]
        pm = got[1]
        im = got[2]
        big = got[3][0]
    else:
        var fac = k.lu(f0)
        lu = fac[0]
        var aux = k.lu_aux(lu, fac[1], True)
        big = aux[0][0]
        pm = aux[2]
        im = aux[3]
    if not (big > 0.0 and _mani_finite(big)):
        return (False, PythonObject(None), PythonObject(None))
    var en = mk_zeros(n, 1, ctx)
    mk_store_ptr(mk_store(en, ctx))[n - 1] = 1.0
    var z = k.trisolve(lu, im, en, 1)
    var zn = sqrt(k.dsum_sq(z))
    if not zn > 0.0 or not _mani_finite(zn):
        return (False, PythonObject(None), PythonObject(None))
    z = k.ew1(OP_SCALE, z, 1.0 / zn)
    # `_lle_iterate`
    var xm = mk_zeros(n1, p, ctx)
    var ym = mk_zeros(n1, nc, ctx)
    var sm = mk_zeros(1, p, ctx)
    var b = Python.import_module("builtins")
    var pf = mk_addr(f0, ctx)
    _ = f0  # keep the Python object alive through the native call above
    var pl = mk_addr(lu, ctx)
    var pp = mk_addr(pm, ctx)
    var pi = mk_addr(im, ctx)
    var pz = mk_addr(z, ctx)
    _ = z  # keep the Python object alive through the native call above
    var outs = Python.list(mk_addr(xm, ctx), mk_addr(ym, ctx), mk_addr(sm, ctx))
    _ = sm  # keep the Python object alive through the native call above
    _ = ym  # keep the Python object alive through the native call above
    _ = xm  # keep the Python object alive through the native call above
    var params = mk_ints([n, n1, nc, p, max(1, max_iter)])
    _ = py_call(params.append, py_call(Python.import_module("operator").and_, py_call(b.int, seed),
                                       PythonObject(0xFFFFFFFF)))
    var tols = Python.list(PythonObject(floor), PythonObject(_LLE_SUBSPACE_TOL), PythonObject(_LLE_STALL_TOL))
    var status = family_call(k.fn("x_decomp_lle_iterate"), pf, pl, pp, pi, pz, outs, params, tols)
    # The iteration receives integer addresses, not their matrix owners.
    # Retain the LU factor and both permutation buffers until its host reads
    # / device uploads finish; mk_addr was otherwise their last use.
    _ = lu^
    _ = pm^
    _ = im^
    _ = sm  # keep the Python object alive through the native call above
    _ = ym  # keep the Python object alive through the native call above
    _ = xm  # keep the Python object alive through the native call above
    _ = z  # keep the Python object alive through the native call above
    _ = f0  # keep the Python object alive through the native call above
    if Int(py=py_call(b.int, status[0])) == 2:
        raise Error("RuntimeError: LocallyLinearEmbedding: the shift-invert subspace iteration did not settle in "
                    + String(max_iter) + " iterations (last subspace change "
                    + String(py=py_call(b.format, py_call(b.float, status[1]), PythonObject(".3g")))
                    + "); pass eigen_solver='dense' for the full SVD. "
                    + "An unconverged embedding is not returned as if it were one.")
    var sv = mk_take_cols(sm, mk_desc_range(p - 1, nc), ctx)
    # back to R^n: H [Y; 0] = [Y; 0] - coef h (h^T [Y; 0])
    var t = k.mm(hrow, ym, False, False)
    var full = k.pad_zero_row(ym)
    var v = k.ew2(OP_SUB, full, k.ew1(OP_SCALE, k.mm(h, t, False, False), coef))
    var sq = k.colsum(k.ew1(OP_SQ, v, 0.0))
    var lo = k.word(k.reduce(sq, SEL_MIN), 0)
    var hi = k.word(k.reduce(sq, SEL_MAX), 0)
    if not (0.9 <= lo and hi <= 1.1) or not _mani_finite(k.word(k.reduce(sv, SEL_MAXABS), 0)):
        var norms = Python.list()
        var store = mk_store(sq, ctx)
        var sp = mk_store_ptr(store)
        _ = store  # keep the Python object alive through the native call above
        for i in range(mk_store_len(store)):  # small-loop(components: the refusal lists each column's squared norm): bounded by n_components
            _ = py_call(norms.append, PythonObject(Float64(sp[i])))
            _ = store  # keep the Python object alive through the native call above
        raise Error("RuntimeError: LocallyLinearEmbedding: the shift-invert subspace iteration returned columns of "
                    + "squared norm " + family_repr(norms) + " (not unit); pass eigen_solver='dense' for the full SVD.")
    var signs = k.absmax_signs(v, True)
    return (True, k.ew2(OP_MUL, v, signs), sv)


def _lle_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`LocallyLinearEmbedding.fit(X)`; returns embedding_."""
    classical_connect_numpy(context)
    var fields = py_fields(model)
    var mode = _mani_mode(fields, context)
    py_setitem(fields, "numeric_mode_", mode)
    var method = family_field(fields, "method")
    if not _mani_in(method, Python.tuple(PythonObject("standard"), PythonObject("hessian"), PythonObject("modified"),
                                         PythonObject("ltsa"))):
        raise Error("LocallyLinearEmbedding: unrecognized method " + family_repr(method))
    var solver = family_field(fields, "eigen_solver")
    _mani_eigen_solver(solver, "LocallyLinearEmbedding")
    var k = _mani_kit(mode, args[0], context)
    var hk = _mani_host_kit(args[1], context)
    var m = mk_from_input(args[2], "X", context)
    var n = mk_r(m)
    var nn = _mani_int(family_field(fields, "n_neighbors"))
    var nc = _mani_int(family_field(fields, "n_components"))
    if nn >= n:
        raise Error("Expected n_neighbors < n_samples, but n_samples = " + String(n) + ", n_neighbors = " + String(nn))
    var standard = _mani_eq(method, "standard")
    if _mani_eq(method, "hessian") and nn <= nc + nc * (nc + 1) // 2:
        raise Error("for method='hessian', n_neighbors must be greater than "
                    + "[n_components * (n_components + 3) / 2]")
    if _mani_eq(method, "modified") and nn < nc:
        raise Error("modified LLE requires n_neighbors >= n_components")
    var knn = _mani_knn(k, m, m, nn, True, 0, 2.0)
    var idm = knn[0]
    var iw: PythonObject
    if standard:
        var wb = k.barycenter(m, m, idm, family_field(fields, "reg"))
        iw = k.graph_lle_iw(idm, wb, n)
    elif _mani_eq(method, "ltsa"):
        iw = k.lle_local(m, idm, 0, nn, nc, 0.0)
    elif _mani_eq(method, "hessian"):
        iw = k.lle_local(m, idm, 1, nn, nc, _mani_float(family_field(fields, "hessian_tol")))
    else:
        iw = k.lle_local(m, idm, 2, nn, nc, _mani_float(family_field(fields, "modified_tol")))
    var found = False
    var em = PythonObject(None)
    var sv = PythonObject(None)
    if standard and _mani_eq(solver, "auto") and n > _LLE_ITER_MIN_N and nc + 1 < _LLE_ITER_MAX_K:
        k.enable_lle_precision()
        var got = _lle_smallest(k, iw, nc, _mani_int(family_field(fields, "max_iter")),
                                _mani_seed(family_field(fields, "random_state")))
        found = got[0]
        em = got[1]
        sv = got[2]
    if not found:
        var dec = k.svd(iw, hk)
        var rows = mk_desc_range(n - 2, nc)
        em = mk_t(mk_take_rows_list(dec[1], rows, hk, context), hk, context)
        sv = mk_take_cols(dec[0], rows, context)
    py_setitem(fields, "embedding_m_", em)
    var embedding = mk_out2(em, context)
    py_setitem(fields, "embedding_", embedding)
    py_setitem(fields, "reconstruction_error_", PythonObject(k.dsum_sq(sv)))
    py_setitem(fields, "_fit_X", m)
    py_setitem(fields, "_knn", PythonObject(nn))
    py_setitem(fields, "n_features_in_", PythonObject(mk_c(m)))
    return embedding


def _lle_transform(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`LocallyLinearEmbedding.transform(X)`."""
    _mani_check(model, "embedding_m_")
    classical_connect_numpy(context)
    var fields = py_fields(model)
    var k = _mani_kit(family_field(fields, "numeric_mode_"), args[0], context)
    var q = mk_from_input(args[2], "X", context)
    var fit_x = family_field(fields, "_fit_X")
    var knn = _mani_knn(k, q, fit_x, _mani_int(family_field(fields, "_knn")), False, 0, 2.0)
    var idm = knn[0]
    var wb = k.barycenter(q, fit_x, idm, family_field(fields, "reg"))
    return mk_out2(k.lle_apply(wb, idm, family_field(fields, "embedding_m_")), context)


def _mani_arguments(model: PythonObject, operation: String, args: PythonObject,
                    context: PythonObject) raises -> PythonObject:
    # Public shells hand over settings and buffers only. Resolve both legacy
    # binding slots after crossing the native boundary. Explicit modules remain
    # supported for internal/legacy callers and are admitted by MKit.
    if len(args) < 2:
        raise Error("TypeError: manifold operation requires two binding slots")
    var fields = py_fields(model)
    var requested = py_get(fields, PythonObject("numeric_mode"), PythonObject(None))
    if operation.endswith("_transform") or operation == "family_mani_iso_reconstruction_error":
        # A fit_transform is a fit, while transform uses the mode saved by fit.
        if not operation.endswith("_fit_transform"):
            requested = py_get(fields, PythonObject("numeric_mode_"), PythonObject(None))
    var binding = args[0]
    if binding is None:
        binding = estimator_binding_mode(requested, "_mojolearn_x_decomp", context)
    var default_binding = args[1]
    if default_binding is None:
        default_binding = estimator_binding_mode(PythonObject(None), "_mojolearn_x_decomp", context)
    var resolved = Python.list(binding, default_binding)
    for i in range(2, len(args)):  # small-loop: fixed public operation argument envelope
        _ = py_call(resolved.append, args[i])
    return py_call(Python.import_module("builtins").tuple, resolved)


def mani_decomp_operation(model: PythonObject, operation: String, var args: PythonObject,
                          context: PythonObject) raises -> PythonObject:
    """args = (optional decomp binding, optional default binding, X[, init])."""
    args = _mani_arguments(model, operation, args, context)
    if operation == "family_mani_cmds_fit":
        _ = _cmds_fit(model, args, context)
        return model
    if operation == "family_mani_cmds_fit_transform":
        return _cmds_fit(model, args, context)
    if operation == "family_mani_mds_fit":
        _ = _mds_fit(model, args, context)
        return model
    if operation == "family_mani_mds_fit_transform":
        return _mds_fit(model, args, context)
    if operation == "family_mani_iso_fit":
        _ = _iso_fit(model, args, context)
        return model
    if operation == "family_mani_iso_fit_transform":
        return _iso_fit(model, args, context)
    if operation == "family_mani_iso_transform":
        return _iso_transform(model, args, context)
    if operation == "family_mani_iso_reconstruction_error":
        return _iso_reconstruction_error(model, args, context)
    if operation == "family_mani_lle_fit":
        _ = _lle_fit(model, args, context)
        return model
    if operation == "family_mani_lle_fit_transform":
        return _lle_fit(model, args, context)
    if operation == "family_mani_lle_transform":
        return _lle_transform(model, args, context)
    raise Error("RuntimeError: unknown manifold operation " + operation)
