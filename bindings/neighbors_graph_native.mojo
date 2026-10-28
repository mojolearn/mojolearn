# SPDX-License-Identifier: Apache-2.0
# One native call per public method (lane native-neighbors-20261024): PageRank,
# Louvain, connected_components and SVGP public methods
# (python/mojolearn/_expansion_neighbors.py).
# Each `family_xn_graph_*` operation validates, prepares, runs the family's
# compiled kernels and returns, with the former Python method's refusals,
# operations and order, so the bits do not move.
#
# The shell hands over the x_neighbors binding `_bind` resolved (args[0]; for
# connected_components the one `_backend.binding` resolves for its
# numeric_mode). Every kernel call is `binding.xn_<op>(addresses, ints,
# floats)` exactly as `_XNeighbors._op` made it, with the same buffers, the
# same integers and the same float32-rounded scalars, in the same order. The
# graph inputs are read as before: an `(indptr, indices, n)` tuple or a
# scipy.sparse matrix through its own `tocsr()` (the caller's container), a
# dense matrix as float32. Outputs are new mojolearn Arrays (the Python
# shells' `empty` / `_empty_out`, zero-filled: every element is written by
# the kernel). MOJOLEARN_XN_LAZY_OUT (an opt-in memory-map A/B arm for
# outputs of 64 MB or more, same zeros and the same bits) is not consulted.
from providers.python.x_neighbors_descriptor_pyapi import xn_admit_arguments, xn_execute, xn_module, xn_is_module, xn_capabilities
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.family_common import (
    FAMILY_F32P, FAMILY_I32P, family_call, family_isinstance, family_str, family_array, family_as_f32,
    family_as_f32_dense,
)
from bindings.cfb_spectral import _as_i32
from bindings.neighbors_ann_native import ann_keep_array
from bindings.mcls_cluster import mcls_cluster_operation
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, PENDING_PYTHON_ERROR
from core.neighbors_graph_impl import xn_graph_even_rows
from providers.python.estimator_binding_pyapi import estimator_binding, estimator_binding_mode

#: `_expansion_neighbors._KERNELS["rbf"]` and `_U_IDENTITY`.
comptime _XN_KERNEL_RBF = 2
comptime _XN_U_IDENTITY = 6


# ---------------------------------------------------------------- glue


def _b() raises -> PythonObject:
    return native_loaded_module("builtins")


def _truth(value: PythonObject) raises -> Bool:
    """`bool(value)`."""
    return Bool(py=family_call(_b().bool, value))


def _eq(a: PythonObject, b: PythonObject) raises -> Bool:
    """`bool(a == b)`."""
    return _truth(family_call(native_loaded_module("operator").eq, a, b))


def _ne(a: PythonObject, b: PythonObject) raises -> Bool:
    """`bool(a != b)`."""
    return _truth(family_call(native_loaded_module("operator").ne, a, b))


def _int(value: PythonObject) raises -> PythonObject:
    """`int(value)`."""
    return family_call(_b().int, value)


def _float(value: PythonObject) raises -> Float64:
    """`float(value)`."""
    return Float64(py=family_call(_b().float, value))


def _f32_scalar(value: PythonObject) raises -> PythonObject:
    """`_expansion_neighbors._f32_scalar`: a Python double rounded once to
    float32 (overflow to infinity, as the `array('f')` store rounds)."""
    return PythonObject(Float64(Float32(_float(value))))


def _f32_word(value: Float64) -> PythonObject:
    """`_f32_scalar` of an already computed double."""
    return PythonObject(Float64(Float32(value)))


def _attr(model: PythonObject, name: String) raises -> PythonObject:
    """`model.<name>` of a stored attribute, refused as Python refused it."""
    var fields = py_fields(model)
    if not py_contains(fields, PythonObject(name)):
        raise Error("AttributeError: '" + String(py_type(model).__name__) + "' object has no attribute '" + name + "'")
    return fields[PythonObject(name)]


def _addr(array: PythonObject) raises -> PythonObject:
    return classical_address(array)


def _i32_at(array: PythonObject, index: Int) raises -> Int:
    """One int32 word of a C-order int32 Array (`int(a.tolist()[index])`)."""
    var words = FAMILY_I32P(unsafe_from_address=Int(py=classical_address(array)))
    var value = Int(words[index])
    _ = array  # keep the Python object alive through the read above
    return value


def _f32_at(array: PythonObject, index: Int) raises -> Float64:
    """One float32 word of a C-order float32 Array as a Python float would
    hold it (`a.tolist()[index]`)."""
    var words = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(array)))
    var value = Float64(words[index])
    _ = array  # keep the Python object alive through the read above
    return value


def _op(binding: PythonObject, name: String, addrs: PythonObject, ints: PythonObject,
        floats: PythonObject) raises:
    """`_XNeighbors._op(name, bufs, ints, floats)`: `binding.xn_<name>(addrs,
    ints, floats)`, the entry looked up as `getattr` looked it up."""
    xn_execute(binding, name, addrs, ints, floats)


def _empty_list() raises -> PythonObject:
    return Python.list()


def _pending_value() raises -> PythonObject:
    """A new reference to the value of the Python exception `family_call`
    stashed, or None. The stash is kept."""
    var pending = PENDING_PYTHON_ERROR.get_or_create_ptr()
    var address = pending[].words[1]
    if address == 0:
        return PythonObject(None)
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("Py_NewRef")(address)
    return PythonObject(from_owned=pointer)


def _drop_pending() raises:
    """Release the stashed exception: the caller raises its own instead."""
    ref cpy = Python().cpython()
    var release = cpy.lib.get_function[NoneType]("Py_DecRef")
    var pending = PENDING_PYTHON_ERROR.get_or_create_ptr()
    for t in range(3):  # small-loop(the stashed exception's three references)
        var address = pending[].words[t]
        if address:
            release(address)
            pending[].words[t] = 0


def _p2m_relabel(binding: PythonObject, lab: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_expansion_neighbors._p2m_relabel(est, lab)`: `(count, labels)`
    renumbered by first occurrence (`xn_p2m_relabel`), or None when a label
    falls outside [0, n)."""
    var n = Int(py=lab.shape[0])
    var out = family_array(Python.tuple(max(n, 1)), "<i4", context)
    var info = family_array(Python.tuple(2), "<i4", context)
    xn_execute(binding, "p2m_relabel", Python.list(_addr(lab), _addr(out), _addr(info)), Python.list(PythonObject(n)),
                    _empty_list())
    _ = lab  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = info  # keep the Python object alive through the native call above
    var cnt = _i32_at(info, 0)
    var bad = _i32_at(info, 1)
    if bad != 0:
        return PythonObject(None)
    return Python.tuple(PythonObject(cnt), out)


def _adjacency(source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_expansion_neighbors._adjacency`: float32 (`_f32(A, "adjacency")`),
    then the square refusal."""
    var a = family_as_f32_dense(source, 2, "adjacency", context)
    if Int(py=a.shape[0]) != Int(py=a.shape[1]):
        raise Error("the adjacency matrix must be square")
    return a


struct _Csr(Movable):
    """`_csr_of` / `_csr_graph_of`'s answer: the int32 lists, the float32
    weights (None for an (indptr, indices, n) tuple or for `_csr_of`) and
    n; `present` False for a dense input."""
    var present: Bool
    var indptr: PythonObject
    var indices: PythonObject
    var vals: PythonObject
    var n: Int

    def __init__(out self, present: Bool, indptr: PythonObject, indices: PythonObject, vals: PythonObject, n: Int):
        self.present = present
        self.indptr = indptr
        self.indices = indices
        self.vals = vals
        self.n = n


def _csr_lists(source: PythonObject, graph: Bool, context: PythonObject) raises -> _Csr:
    """`_csr_of(A)` (graph False) or `_csr_graph_of(A)` (graph True): a
    tuple `(indptr, indices, n)` or a scipy.sparse matrix converted with its
    own `tocsr()`, the lists as int32 (`as_i32_c`), the weights as float32
    (`as_f32_c(M.data, ndim=1, name="data")`), with the same refusals in the
    same order."""
    var b = _b()
    var indptr: PythonObject
    var indices: PythonObject
    var vals = PythonObject(None)
    var n: Int
    if family_isinstance(source, b.tuple) and Int(py=family_call(b.len, source)) == 3:
        indptr = source[0]
        indices = source[1]
        n = Int(py=_int(source[2]))
    elif _truth(family_call(b.hasattr, source, PythonObject("tocsr"))) and _truth(
            family_call(b.hasattr, source, PythonObject("shape"))):
        var shape = family_call(b.getattr, source, PythonObject("shape"))
        if Int(py=family_call(b.len, shape)) != 2 or _ne(shape[0], shape[1]):
            raise Error("the adjacency matrix must be square")
        var m = family_call(family_call(b.getattr, source, PythonObject("tocsr")))
        indptr = family_call(b.getattr, m, PythonObject("indptr"))
        indices = family_call(b.getattr, m, PythonObject("indices"))
        n = Int(py=_int(family_call(b.getattr, m, PythonObject("shape"))[0]))
        if graph:
            vals = family_as_f32(family_call(b.getattr, m, PythonObject("data")), 1, "data", context)
    else:
        return _Csr(False, PythonObject(None), PythonObject(None), PythonObject(None), 0)
    var ip = _as_i32(indptr, "indptr", context)
    var ix = _as_i32(indices, "indices", context)
    if Int(py=ip.shape[0]) != n + 1:
        if graph:
            raise Error("the CSR adjacency's indptr must hold n + 1 entries")
        raise Error("connected_components: indptr must hold n + 1 entries")
    if graph:
        var nnz = Int(py=ix.shape[0])
        if _i32_at(ip, n) != nnz:  # argument check: one scalar, the entry count
            raise Error("the CSR adjacency's indices must hold indptr[n] entries")
        if vals is not None and Int(py=vals.shape[0]) != nnz:
            raise Error("the CSR adjacency's data must hold one weight per entry")
    return _Csr(True, ip, ix, vals, n)


# ---------------------------------------------------------------- PageRank


def _unit(binding: PythonObject, v: PythonObject, n: Int, what: String,
          context: PythonObject) raises -> PythonObject:
    """`PageRank._unit(v, n, what)`: a caller's length-n non-negative vector
    divided by its sum on the device (`x_neighbors_unit_ff`)."""
    var b = _b()
    var unit_fn = PythonObject(None)
    var available = False
    if xn_is_module(binding):
        unit_fn = py_call(b.getattr, xn_module(binding), PythonObject("x_neighbors_unit_ff"), PythonObject(None))
        available = unit_fn is not None
    else:
        available = (xn_capabilities(binding) & 4) != 0
    if not available:
        raise Error("NotImplementedError: mojolearn PageRank " + what + ": this binary lacks x_neighbors_unit_ff "
                    + "(rebuild without -D MOJOLEARN_IDN_XN_UNIT_DEV_OFF)")
    var wv = family_as_f32(v, 1, what, context)
    if Int(py=wv.shape[0]) != n:
        raise Error(what + " must be n non-negative values, not all zero")
    var out = family_array(Python.tuple(n), "<f4", context)
    var uinfo = family_array(Python.tuple(2), "<i4", context)
    if xn_is_module(binding):
        _ = family_call(unit_fn, Python.list(_addr(wv), _addr(out), _addr(uinfo)), Python.list(PythonObject(n)),
                        _empty_list())
    else:
        xn_execute(binding, "unit_ff", Python.list(_addr(wv), _addr(out), _addr(uinfo)),
                   Python.list(PythonObject(n)), _empty_list())
    _ = wv  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = uinfo  # keep the Python object alive through the native call above
    if _i32_at(uinfo, 0) != 0 or _i32_at(uinfo, 1) != 0:
        raise Error(what + " must be n non-negative values, not all zero")
    return out


def _uniform(binding: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """PageRank.fit's `uniform()`: [1/n] * n, 1/n in IEEE double handed to
    `xn_p2m_fill` (which rounds it once to float32)."""
    var u = family_array(Python.tuple(n), "<f4", context)
    if n != 0:
        _op(binding, "p2m_fill", Python.list(_addr(u)), Python.list(PythonObject(n)),
            Python.list(PythonObject(1.0 / Float64(n))))
        _ = u  # keep the Python object alive through the native call above
    return u


def _pagerank_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`PageRank.fit(A)`."""
    classical_connect_numpy(context)
    var b = _b()
    var binding = args[0]
    var fields = py_fields(model)
    var weight = _attr(model, "weight")
    var binary = weight is None or (py_type(weight) is b.bool and not Bool(py=weight))
    var csr = _csr_lists(args[1], True, context)
    var a = PythonObject(None)
    var n: Int
    if csr.present:
        n = csr.n
    else:
        a = _adjacency(args[1], context)
        n = Int(py=a.shape[0])
    var info = family_array(Python.tuple(2), "<i4", context)
    # `struct.unpack("<Q", struct.pack("<d", n * float(tol)))[0]`, split
    # into its high and low 32-bit words
    var thr = bitcast[DType.uint64](Float64(n) * _float(_attr(model, "tol")))
    var thr_hi = PythonObject(Int(thr >> 32))
    var thr_lo = PythonObject(Int(thr & 0xFFFFFFFF))
    var personalization = _attr(model, "personalization")
    var dangling = _attr(model, "dangling")
    var nstart = _attr(model, "nstart")
    var x: PythonObject
    if csr.present:
        var p_uniform = personalization is None
        var p = family_array(Python.tuple(n), "<f4", context) if p_uniform else _unit(
            binding, personalization, n, "personalization", context)
        var dw_uniform = dangling is None and p_uniform
        var dw = p if dangling is None else _unit(binding, dangling, n, "dangling", context)
        var x_uniform = nstart is None
        x = family_array(Python.tuple(n), "<f4", context) if x_uniform else _unit(
            binding, nstart, n, "nstart", context)
        var has_vals = csr.vals is not None
        var vals = csr.vals if has_vals else family_array(Python.tuple(1), "<f4", context)
        var ints = Python.list(PythonObject(n), PythonObject(Int(py=csr.indices.shape[0])),
                               PythonObject(1 if has_vals else 0), _int(_attr(model, "max_iter")), thr_hi, thr_lo,
                               PythonObject(1 if binary else 0), PythonObject(1 if x_uniform else 0),
                               PythonObject(1 if p_uniform else 0), PythonObject(1 if dw_uniform else 0))
        var floats = Python.list(_f32_scalar(_attr(model, "alpha")))
        _op(binding, "pr_csr", Python.list(_addr(csr.indptr), _addr(csr.indices), _addr(vals), _addr(x), _addr(p),
                                           _addr(dw), _addr(info)), ints, floats)
        _ = vals  # keep the Python object alive through the native call above
        _ = p  # keep the Python object alive through the native call above
        _ = dw  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
        _ = info  # keep the Python object alive through the native call above
    else:
        var p = _uniform(binding, n, context) if personalization is None else _unit(
            binding, personalization, n, "personalization", context)
        var dw = p if dangling is None else _unit(binding, dangling, n, "dangling", context)
        # `x.copy()` of a vector this call made: a fresh Array either way.
        x = _uniform(binding, n, context) if nstart is None else _unit(binding, nstart, n, "nstart", context)
        var ints = Python.list(PythonObject(n), _int(_attr(model, "max_iter")), thr_hi, thr_lo,
                               PythonObject(1 if binary else 0))
        var floats = Python.list(_f32_scalar(_attr(model, "alpha")))
        _op(binding, "pr_iterate_sparse", Python.list(_addr(a), _addr(x), _addr(p), _addr(dw), _addr(info)),
            ints, floats)
        _ = a  # keep the Python object alive through the native call above
        _ = p  # keep the Python object alive through the native call above
        _ = dw  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
        _ = info  # keep the Python object alive through the native call above
    _ = csr^
    var it = _i32_at(info, 0)
    var ok = _i32_at(info, 1)
    if ok != 0:
        py_setitem(fields, "pagerank_", x)
        py_setitem(fields, "n_iter_", PythonObject(it))
        return
    raise Error("RuntimeError: PageRank: power iteration failed to converge within "
                + family_str(_attr(model, "max_iter")) + " iterations")


# ---------------------------------------------------------------- connected components


def _cc_relabel(binding: PythonObject, lab: PythonObject, return_labels: Bool,
                context: PythonObject) raises -> PythonObject:
    """`_cc_relabel(lab, return_labels, est)`."""
    if Int(py=lab.shape[0]) == 0:
        return Python.tuple(PythonObject(0), lab) if return_labels else PythonObject(0)
    var got = _p2m_relabel(binding, lab, context)
    if got is None:
        raise Error("RuntimeError: connected_components: a component label fell outside [0, n)")
    return got if return_labels else got[0]


def _p2m_iota(binding: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_p2m_iota(est, n)`: [0, 1, ..., n - 1] as int32 (`xn_p2m_iota`)."""
    var lab = family_array(Python.tuple(max(n, 0)), "<i4", context)
    if n > 0:
        _op(binding, "p2m_iota", Python.list(_addr(lab)), Python.list(PythonObject(n)), _empty_list())
        _ = lab  # keep the Python object alive through the native call above
    return lab


def _connected_components(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`connected_components(A, directed, connection, return_labels,
    numeric_mode)`: args = (binding, A, directed, connection,
    return_labels)."""
    classical_connect_numpy(context)
    var binding = args[0]
    var connection = args[3]
    if _truth(args[2]) and _eq(connection, PythonObject("strong")):
        raise Error("NotImplementedError: connected_components: connection='strong' is not implemented")
    if not py_contains(Python.tuple(PythonObject("weak"), PythonObject("strong")), connection):
        raise Error("connection must be 'weak' or 'strong'")
    var csr = _csr_lists(args[1], False, context)
    if csr.present:
        var n = csr.n
        var lab = _p2m_iota(binding, n, context)
        var info = family_array(Python.tuple(2), "<i4", context)
        _op(binding, "cc_iterate_csr", Python.list(_addr(csr.indptr), _addr(csr.indices), _addr(lab), _addr(info)),
            Python.list(PythonObject(n), PythonObject(Int(py=csr.indices.shape[0]))), _empty_list())
        _ = lab  # keep the Python object alive through the native call above
        _ = info  # keep the Python object alive through the native call above
        _ = csr^
        var k1 = _i32_at(info, 1)
        if k1 > 0:
            if _truth(args[4]):
                return Python.tuple(PythonObject(k1 - 1), lab)
            return PythonObject(k1 - 1)
        return _cc_relabel(binding, lab, _truth(args[4]), context)
    _ = csr^
    var a = _adjacency(args[1], context)
    var n = Int(py=a.shape[0])
    var lab = _p2m_iota(binding, n, context)
    var info = family_array(Python.tuple(1), "<i4", context)
    _op(binding, "cc_iterate", Python.list(_addr(a), _addr(lab), _addr(info)), Python.list(PythonObject(n)),
        _empty_list())
    _ = a  # keep the Python object alive through the native call above
    _ = lab  # keep the Python object alive through the native call above
    _ = info  # keep the Python object alive through the native call above
    return _cc_relabel(binding, lab, _truth(args[4]), context)


# ---------------------------------------------------------------- Louvain


def _louvain_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`Louvain.fit(A)`."""
    classical_connect_numpy(context)
    var binding = args[0]
    var fields = py_fields(model)
    var max_level = _attr(model, "max_level")
    var ml = PythonObject(0) if max_level is None else _int(max_level)
    if max_level is not None and _truth(family_call(native_loaded_module("operator").lt, ml, PythonObject(1))):
        raise Error("max_level must be a positive integer or None")
    var csr = _csr_lists(args[1], True, context)
    var n: Int
    var labels: PythonObject
    var info: PythonObject
    if csr.present:
        n = csr.n
        var has_vals = csr.vals is not None
        var vals = csr.vals if has_vals else family_array(Python.tuple(1), "<f4", context)
        labels = family_array(Python.tuple(n), "<i4", context)
        info = family_array(Python.tuple(2), "<f4", context)
        try:
            var floats = Python.list(_f32_scalar(_attr(model, "resolution")), _f32_scalar(_attr(model, "threshold")))
            _op(binding, "louvain_csr", Python.list(_addr(csr.indptr), _addr(csr.indices), _addr(vals), _addr(labels),
                                                    _addr(info)),
                Python.list(PythonObject(n), PythonObject(Int(py=csr.indices.shape[0])),
                            PythonObject(1 if has_vals else 0), ml), floats)
            _ = vals  # keep the Python object alive through the native call above
            _ = labels  # keep the Python object alive through the native call above
            _ = info  # keep the Python object alive through the native call above
        except error:
            # `except Exception as e`: the device checks' messages as
            # ValueErrors, from "Louvain:" on; anything else unchanged.
            if String(error) != PYTHON_ERROR_PENDING:
                raise error^
            var value = _pending_value()
            var message = String("") if value is None else family_str(value)
            var at = message.find("Louvain:")
            if at < 0:
                raise error^
            _drop_pending()
            raise Error("ValueError: " + String(message[byte=at:message.byte_length()]))
    else:
        var a = _adjacency(args[1], context)
        n = Int(py=a.shape[0])
        # The symmetry and no-edge checks run natively (`xn_graph_symmetry`).
        var flags = family_array(Python.tuple(2), "<i4", context)
        _op(binding, "graph_symmetry", Python.list(_addr(a), _addr(flags)), Python.list(PythonObject(n)),
            _empty_list())
        _ = a  # keep the Python object alive through the native call above
        _ = flags  # keep the Python object alive through the native call above
        if _i32_at(flags, 0) != 0:
            raise Error("Louvain: the adjacency matrix must be symmetric (an undirected graph)")
        if _i32_at(flags, 1) == 0:
            raise Error("Louvain: the graph has no edges")
        labels = family_array(Python.tuple(n), "<i4", context)
        info = family_array(Python.tuple(2), "<f4", context)
        var floats = Python.list(_f32_scalar(_attr(model, "resolution")), _f32_scalar(_attr(model, "threshold")))
        _op(binding, "louvain", Python.list(_addr(a), _addr(labels), _addr(info)),
            Python.list(PythonObject(n), ml), floats)
        _ = a  # keep the Python object alive through the native call above
        _ = labels  # keep the Python object alive through the native call above
        _ = info  # keep the Python object alive through the native call above
    _ = csr^
    var got = _p2m_relabel(binding, labels, context)
    if got is None:
        raise Error("RuntimeError: Louvain: a community label fell outside [0, n)")
    py_setitem(fields, "n_communities_", got[0])
    py_setitem(fields, "labels_", got[1])
    py_setitem(fields, "modularity_", PythonObject(_f32_at(info, 0)))
    py_setitem(fields, "n_levels_", _int(PythonObject(_f32_at(info, 1))))


# ---------------------------------------------------------------- SVGP


def _svgp_gamma(model: PythonObject) raises -> Float64:
    """`SVGP._gamma_value()`: 1 / (2 * ls * ls) in IEEE double, ls ** 2 as
    one product."""
    var ls = _float(_attr(model, "lengthscale"))
    var den = 2.0 * (ls * ls)
    if den == 0.0:
        raise Error("ZeroDivisionError: float division by zero")
    return 1.0 / den


def _svgp_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`SVGP.fit(X, y)`."""
    classical_connect_numpy(context)
    var b = _b()
    var binding = args[0]
    var fields = py_fields(model)
    var x = family_as_f32_dense(args[1], 2, "X", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var yv = family_as_f32(args[2], 1, "y", context)
    if Int(py=yv.shape[0]) != n:
        raise Error("X and y have different numbers of rows")
    var inducing = _attr(model, "inducing_points")
    var z: PythonObject
    if inducing is not None:
        z = ann_keep_array(inducing, family_as_f32_dense(inducing, 2, "inducing_points", context), context)
    else:
        var m_obj = family_call(b.min, _int(_attr(model, "n_inducing")), PythonObject(n))
        if _truth(family_call(native_loaded_module("operator").le, m_obj, PythonObject(0))):
            # `as_i32_c` of the empty row list.
            raise Error("mojolearn: rows is empty, shape (0,): Found array with 0 sample(s) (shape=(0,)) "
                        + "while a minimum of 1 is required.")
        var m = Int(py=m_obj)
        var rows = family_array(Python.tuple(m), "<i4", context)
        xn_graph_even_rows(FAMILY_I32P(unsafe_from_address=Int(py=classical_address(rows))), n, m)
        _ = rows  # keep the Python object alive through the native call above
        z = family_array(Python.tuple(m, d), "<f4", context)
        _op(binding, "take_rows", Python.list(_addr(x), _addr(rows), _addr(z)),
            Python.list(PythonObject(m), PythonObject(d), PythonObject(n)), _empty_list())
        _ = x  # keep the Python object alive through the native call above
        _ = rows  # keep the Python object alive through the native call above
        _ = z  # keep the Python object alive through the native call above
    var mz = Int(py=z.shape[0])
    var alpha = family_array(Python.tuple(mz), "<f4", context)
    var c = family_array(Python.tuple(mz, mz), "<f4", context)
    var qmu = family_array(Python.tuple(mz), "<f4", context)
    var qsqrt = family_array(Python.tuple(mz, mz), "<f4", context)
    var info = family_array(Python.tuple(2), "<f4", context)
    var floats = Python.list(_f32_word(_svgp_gamma(model)), _f32_scalar(_attr(model, "kernel_variance")),
                             _f32_scalar(_attr(model, "noise_variance")), _f32_scalar(_attr(model, "jitter")),
                             _f32_scalar(_attr(model, "kernel_variance")))
    _op(binding, "svgp_fit_ff", Python.list(_addr(x), _addr(z), _addr(yv), _addr(alpha), _addr(c), _addr(qmu),
                                            _addr(qsqrt), _addr(info)),
        Python.list(PythonObject(n), PythonObject(mz), PythonObject(d)), floats)
    _ = x  # keep the Python object alive through the native call above
    _ = z  # keep the Python object alive through the native call above
    _ = yv  # keep the Python object alive through the native call above
    _ = alpha  # keep the Python object alive through the native call above
    _ = c  # keep the Python object alive through the native call above
    _ = qmu  # keep the Python object alive through the native call above
    _ = qsqrt  # keep the Python object alive through the native call above
    _ = info  # keep the Python object alive through the native call above
    var elbo = _f32_at(info, 0)
    var ok = _f32_at(info, 1)
    py_setitem(fields, "precision_", PythonObject("float-float"))
    if ok == 0.0:
        raise Error("SVGP: the inducing system is not positive definite; raise jitter or noise_variance")
    py_setitem(fields, "Z_", z)
    py_setitem(fields, "_alpha", alpha)
    py_setitem(fields, "_C", c)
    py_setitem(fields, "q_mu_", qmu)
    py_setitem(fields, "q_sqrt_", qsqrt)
    py_setitem(fields, "elbo_", PythonObject(elbo))
    py_setitem(fields, "n_features_in_", PythonObject(d))


def _svgp_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`SVGP.predict(X)`: the mean of `predict_f(X)` (its variance is computed
    as `predict_f` computed it)."""
    return _svgp_predict_f(model, args, context)[0]


def _svgp_predict_y(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`SVGP.predict_y(X)`: `predict_f`'s mean, and its variance plus
    `noise_variance` (`_unary(var, _U_IDENTITY, 1.0, f32(noise_variance))`)."""
    var latent = _svgp_predict_f(model, args, context)
    var var_ = latent[1]
    var size = Int(py=var_.size)
    var out = family_array(Python.tuple(size), "<f4", context)
    _op(args[0], "unary", Python.list(_addr(var_), _addr(out)),
        Python.list(PythonObject(size), PythonObject(_XN_U_IDENTITY)),
        Python.list(PythonObject(1.0), _f32_scalar(_attr(model, "noise_variance"))))
    _ = var_  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    return Python.tuple(latent[0], out)


def _svgp_predict_f(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`SVGP.predict_f(X)`: (mean, var) of the latent function."""
    classical_connect_numpy(context)
    var binding = args[0]
    var q = family_as_f32_dense(args[1], 2, "X", context)
    var z = _attr(model, "Z_")
    var m = Int(py=z.shape[0])
    var nq = Int(py=q.shape[0])
    var d = Int(py=q.shape[1])
    var old_items = _truth(context["family_svc_xn_old_items"])
    var alpha = _attr(model, "_alpha")
    var c = _attr(model, "_C")
    if not old_items and nq != 0:
        # the fused chain: Ksu, the mean and the variance per row tile
        # (`xn_svgp_predict`)
        var mean = family_array(Python.tuple(nq), "<f4", context)
        var var_ = family_array(Python.tuple(nq), "<f4", context)
        var floats = Python.list(_f32_word(_svgp_gamma(model)), _f32_scalar(_attr(model, "kernel_variance")),
                                 _f32_scalar(_attr(model, "kernel_variance")))
        _op(binding, "svgp_predict", Python.list(_addr(q), _addr(z), _addr(alpha), _addr(c), _addr(mean),
                                                 _addr(var_)),
            Python.list(PythonObject(nq), PythonObject(m), PythonObject(d)), floats)
        _ = q  # keep the Python object alive through the native call above
        _ = z  # keep the Python object alive through the native call above
        _ = alpha  # keep the Python object alive through the native call above
        _ = c  # keep the Python object alive through the native call above
        _ = var_  # keep the Python object alive through the native call above
        _ = mean  # keep the Python object alive through the native call above
        return Python.tuple(mean, var_)
    # `_k(Q, Z_)`: the rbf kernel, then the kernel variance scale.
    var g = _svgp_gamma(model)
    var k = family_array(Python.tuple(nq, m), "<f4", context)
    _op(binding, "kernel", Python.list(_addr(q), _addr(z), _addr(k)),
        Python.list(PythonObject(nq), PythonObject(m), PythonObject(d), PythonObject(_XN_KERNEL_RBF), PythonObject(0)),
        Python.list(_f32_word(g), _f32_word(0.0)))
    _ = q  # keep the Python object alive through the native call above
    _ = z  # keep the Python object alive through the native call above
    _ = k  # keep the Python object alive through the native call above
    var ksu = family_array(Python.tuple(nq, m), "<f4", context)
    _op(binding, "unary", Python.list(_addr(k), _addr(ksu)), Python.list(PythonObject(nq * m), PythonObject(_XN_U_IDENTITY)),
        Python.list(_f32_scalar(_attr(model, "kernel_variance")), PythonObject(0.0)))
    _ = k  # keep the Python object alive through the native call above
    _ = ksu  # keep the Python object alive through the native call above
    # `matmul(Ksu, alpha.reshape((M, 1))).reshape((nq,))`: an (nq, 1) product
    # written into the (nq,) vector it is reshaped to.
    var mean = family_array(Python.tuple(nq), "<f4", context)
    _op(binding, "matmul", Python.list(_addr(ksu), _addr(alpha), _addr(mean)),
        Python.list(PythonObject(nq), PythonObject(m), PythonObject(1)), _empty_list())
    _ = ksu  # keep the Python object alive through the native call above
    _ = alpha  # keep the Python object alive through the native call above
    _ = mean  # keep the Python object alive through the native call above
    var var_ = family_array(Python.tuple(nq), "<f4", context)
    _op(binding, "svgp_var", Python.list(_addr(ksu), _addr(c), _addr(var_)),
        Python.list(PythonObject(nq), PythonObject(m)), Python.list(_f32_scalar(_attr(model, "kernel_variance"))))
    _ = ksu  # keep the Python object alive through the native call above
    _ = c  # keep the Python object alive through the native call above
    _ = var_  # keep the Python object alive through the native call above
    return Python.tuple(mean, var_)


def _svgp_score(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`RegressorScoreMixin.score(X, y, sample_weight)` as SVGP inherited
    it: y as float32 (`as_f32_c(y, ndim=None, name="y")`), the prediction,
    then `metrics.r2_score(y, pred, sample_weight=..., numeric_mode=...)`
    through the metrics owner; args = (binding, X, y, sample_weight[, the
    metrics binding]). Without a metrics binding the model's numeric_mode
    selects the import-connected one here, as `_get_binding` selected it."""
    classical_connect_numpy(context)
    var truth = family_as_f32(args[2], -1, "y", context)
    var prediction = _svgp_predict(model, args, context)
    var mode = py_get(py_fields(model), PythonObject("numeric_mode"), PythonObject(None))
    var metrics = args[4] if len(args) > 4 and args[4] is not None else estimator_binding(
        model, "_mojolearn_metrics", context)
    return mcls_cluster_operation("family_mcls_r2", Python.tuple(metrics, truth, prediction, args[3],
                                  PythonObject("uniform_average"), PythonObject(True), mode), context)


def neighbors_graph_operation(model: PythonObject, operation: String, var args: PythonObject,
                              context: PythonObject) raises -> PythonObject:
    if (operation == "family_xn_graph_connected_components" and len(args) > 0
            and py_type(args[0]) is _b().str):
        # connected_components(..., numeric_mode=<str>): the shell hands over
        # the requested mode in the binding slot (None was resolved to the
        # default-mode binding by the family door); select its binding here.
        var selected = Python.list(estimator_binding_mode(args[0], "_mojolearn_x_neighbors", context))
        for i in range(1, len(args)):  # driver-loop: over fixed public function arguments
            _ = py_call(selected.append, args[i])
        args = selected
    args = xn_admit_arguments(args, context)
    if operation == "family_xn_graph_pagerank_fit":
        _pagerank_fit(model, args, context)
        return model
    if operation == "family_xn_graph_louvain_fit":
        _louvain_fit(model, args, context)
        return model
    if operation == "family_xn_graph_louvain_fit_predict":
        _louvain_fit(model, args, context)
        return _attr(model, "labels_")
    if operation == "family_xn_graph_connected_components":
        return _connected_components(args, context)
    if operation == "family_xn_graph_svgp_fit":
        _svgp_fit(model, args, context)
        return model
    if operation == "family_xn_graph_svgp_predict":
        return _svgp_predict(model, args, context)
    if operation == "family_xn_graph_svgp_predict_f":
        return _svgp_predict_f(model, args, context)
    if operation == "family_xn_graph_svgp_predict_y":
        return _svgp_predict_y(model, args, context)
    if operation == "family_xn_graph_svgp_score":
        return _svgp_score(model, args, context)
    raise Error("RuntimeError: unknown neighbors operation " + operation)
