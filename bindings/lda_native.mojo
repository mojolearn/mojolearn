# SPDX-License-Identifier: Apache-2.0
# LatentDirichletAllocation's native operations (dispatched by
# bindings/decomp_native.mojo): the parameter checks the shell made, the
# priors `_init` set, X converted once, then one compiled driver
# (x_decomp/lda_owner.mojo: x_decomp_lda_owner, the kit steps the Python shell
# sequenced, the same entries, operands, draws and order, so the same bits),
# and the fitted attributes and the draw counters stored as the shell stored them.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_type, py_fields, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_address, classical_host_f32
from bindings.decomp_inputs import decomp_input
from bindings.decomp_owner_common import (
    decomp_attr, decomp_check, decomp_entries, decomp_eq, decomp_f32, decomp_float, decomp_int, decomp_seed,
    decomp_set_mode,
)

comptime _FIT = 0
comptime _PARTIAL = 1
comptime _TRANSFORM = 2
comptime _SCORE = 3
comptime _PERPLEXITY = 4


def _batch_size(fields: PythonObject) raises -> Int:
    """The online batch size: a positive int (not a bool)."""
    var b = Python.import_module("builtins")
    var bs = fields["batch_size"]
    if py_type(bs) is b.bool or not Bool(py=py_call(b.isinstance, bs, b.int)) or Int(py=bs) < 1:
        raise Error("batch_size must be a positive integer")
    return Int(py=bs)


def _init_fields(fields: PythonObject) raises:
    """`_init`'s attributes: the priors (1 / n_components when None), the
    seed and the draw counters."""
    var nc = fields["n_components"]
    var doc = fields["doc_topic_prior"]
    var topic = fields["topic_word_prior"]
    var inverse = PythonObject(1.0 / decomp_float(nc))
    py_setitem(fields, "doc_topic_prior_", inverse if doc is None else doc)
    py_setitem(fields, "topic_word_prior_", inverse if topic is None else topic)
    # `_seed_of`: refused when not None or an int; kept as given (None: 0)
    var seed = fields["random_state"]
    _ = decomp_seed(seed)
    py_setitem(fields, "_seed", PythonObject(0) if seed is None else seed)
    py_setitem(fields, "_draw", PythonObject(0))
    py_setitem(fields, "n_batch_iter_", PythonObject(1))
    py_setitem(fields, "n_iter_", PythonObject(0))


def _run(model: PythonObject, fields: PythonObject, op: Int, x: PythonObject, fresh: Bool, flag: Bool,
         total: Float64, message: String, context: PythonObject) raises -> PythonObject:
    """One x_decomp_lda_owner call: (n_iter_, draw, n_batch_iter, value,
    components, exp_dir, out)."""
    var entries = decomp_entries(model, fields, Python.tuple("x_decomp_lda_owner"), context)
    var n = Int(py=x.shape[0])
    var v = Int(py=x.shape[1])
    var nc = decomp_int(fields["n_components"])
    var state_c = PythonObject(None)
    var state_e = PythonObject(None)
    if not (op == _FIT or (op == _PARTIAL and fresh)):
        state_c = classical_host_f32(decomp_attr(model, fields, "components_"), 2, "components_", context)
        state_e = classical_host_f32(decomp_attr(model, fields, "exp_dirichlet_component_"), 2,
                                     "exp_dirichlet_component_", context)
        nc = Int(py=state_c.shape[0])
    var comps = decomp_f32(Python.tuple(nc, v), context)
    var exp_dir = decomp_f32(Python.tuple(nc, v), context)
    var out = decomp_f32(Python.tuple(n, nc), context) if op == _TRANSFORM else PythonObject(None)
    var spare = classical_address(comps)
    var online = decomp_eq(fields["learning_method"], PythonObject("online"))
    var bs = 1
    if op == _PARTIAL or (op == _FIT and online):
        bs = _batch_size(fields)
    var params = Python.list(
        op, classical_address(x), n, v, nc, 1 if fresh else 0, 1 if flag else 0,
        classical_address(state_c) if state_c is not None else spare,
        classical_address(state_e) if state_e is not None else spare, classical_address(comps),
        classical_address(exp_dir), classical_address(out) if out is not None else spare,
        decomp_int(fields["max_iter"]), 1 if (op == _FIT and online) else 0, bs,
        decomp_int(fields["max_doc_update_iter"]),
        decomp_seed(decomp_attr(model, fields, "_seed")))
    _ = py_call(params.append, py_get(fields, "_draw", PythonObject(0)))
    _ = py_call(params.append, py_get(fields, "n_batch_iter_", PythonObject(1)))
    _ = py_call(params.append, decomp_int(fields["evaluate_every"]))
    var floats = Python.list(
        decomp_float(decomp_attr(model, fields, "doc_topic_prior_")),
        decomp_float(decomp_attr(model, fields, "topic_word_prior_")), decomp_float(fields["learning_offset"]),
        decomp_float(fields["learning_decay"]), decomp_float(fields["mean_change_tol"]), total,
        decomp_float(fields["perp_tol"]))
    var got = py_call_typed(entries["x_decomp_lda_owner"], params, floats, PythonObject(message))
    _ = state_c^
    _ = state_e^
    return Python.tuple(got[0], got[1], got[2], got[3], comps, exp_dir, out)


def lda_fit(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`LatentDirichletAllocation.fit`; returns the converted X (kept for
    fit_transform)."""
    var fields = py_fields(model)
    decomp_set_mode(fields, context)
    var method = fields["learning_method"]
    if not Bool(py=py_call(Python.import_module("operator").contains, Python.tuple("batch", "online"), method)):
        raise Error("learning_method must be 'batch' or 'online'")
    var x = decomp_input(source, "LatentDirichletAllocation", context)
    _init_fields(fields)
    var n = Int(py=x.shape[0])
    var got = _run(model, fields, _FIT, x, True, False, Float64(n),
                   "Negative values in data passed to LatentDirichletAllocation.fit", context)
    py_setitem(fields, "_draw", got[1])
    py_setitem(fields, "n_batch_iter_", got[2])
    py_setitem(fields, "n_iter_", PythonObject(Int(py=got[0])))
    py_setitem(fields, "bound_", PythonObject(Float64(py=got[3])))
    py_setitem(fields, "components_", got[4])
    py_setitem(fields, "exp_dirichlet_component_", got[5])
    py_setitem(fields, "n_features_in_", PythonObject(Int(py=x.shape[1])))
    return x


def lda_partial_fit(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`LatentDirichletAllocation.partial_fit`: one online pass."""
    var fields = py_fields(model)
    var fresh = not py_contains(fields, "components_")
    if fresh:
        decomp_set_mode(fields, context)
    var x = decomp_input(source, "LatentDirichletAllocation", context)
    if fresh:
        _init_fields(fields)
    var got = _run(model, fields, _PARTIAL, x, fresh, False, decomp_float(fields["total_samples"]),
                   "Negative values in data passed to LatentDirichletAllocation.partial_fit", context)
    py_setitem(fields, "_draw", got[1])
    py_setitem(fields, "n_batch_iter_", got[2])
    py_setitem(fields, "components_", got[4])
    py_setitem(fields, "exp_dirichlet_component_", got[5])
    py_setitem(fields, "n_features_in_", PythonObject(Int(py=x.shape[1])))
    _ = x^
    return model


def lda_query(model: PythonObject, source: PythonObject, op: Int, flag: Bool, context: PythonObject) raises -> PythonObject:
    """transform (flag: normalize), score, perplexity (flag: sub_sampling)."""
    var fields = py_fields(model)
    decomp_check(model, fields, "components_")
    var x = decomp_input(source, "LatentDirichletAllocation", context)
    return lda_project(model, fields, x, op, flag, context)


def lda_project(model: PythonObject, fields: PythonObject, x: PythonObject, op: Int, flag: Bool,
                context: PythonObject) raises -> PythonObject:
    var got = _run(model, fields, op, x, False, flag, decomp_float(fields["total_samples"]),
                   "Negative values in data passed to LatentDirichletAllocation.transform", context)
    if op == _TRANSFORM:
        return got[6]
    return PythonObject(Float64(py=got[3]))
