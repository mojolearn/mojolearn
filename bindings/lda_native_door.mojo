# SPDX-License-Identifier: Apache-2.0
# LatentDirichletAllocation's native operations (dispatched by
# bindings/decomp_native.mojo): the parameter checks the shell made, the
# priors `_init` set, X converted once, then one compiled driver
# (x_decomp/lda_owner.mojo: lda_run, the kit steps the Python shell
# sequenced, the same operands, draws and order, so the same bits),
# and the fitted attributes and the draw counters stored as the shell stored them.
# UNPACK reads the estimator's fields and arrays into one typed request
# (core/decomp_factor_api.mojo); RUN is one call through the operations table
# of the estimator's image (`decomp_entries_routed`: the device image, or the
# CPU image for `device="cpu"`), the GIL released; PACK stores the fitted
# fields. No Python parameter list and no Python-callable driver entry.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_type, py_fields, py_setitem, py_contains
from std.python._cpython import GILReleased
from bindings.classical_arrays_door import classical_address
from bindings.admitted_input_unpack import admit_x
from bindings.decomp_inputs_unpack import decomp_input
from bindings.decomp_owner_common_unpack import (
    decomp_attr, decomp_check, decomp_eq, decomp_f32, decomp_float, decomp_int, decomp_seed,
    decomp_set_mode,
)
from bindings.decomp_kit_native_door import decomp_entries_routed, decomp_entries_ref
from core.decomp_factor_api import (
    LDARequest, lda_call, LDA_REFUSAL_FIT, LDA_REFUSAL_PARTIAL_FIT, LDA_REFUSAL_TRANSFORM,
)
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add
from bindings.py_balanced import py_item


# ---- from bindings/lda_keys_unpack.mojo (inlined: its one importer is this file) ----
# SPDX-License-Identifier: Apache-2.0
# The lda family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).



struct LdaKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k__draw: PythonObject
    var k__seed: PythonObject
    var k_batch_size: PythonObject
    var k_bound_: PythonObject
    var k_components_: PythonObject
    var k_doc_topic_prior: PythonObject
    var k_doc_topic_prior_: PythonObject
    var k_evaluate_every: PythonObject
    var k_exp_dirichlet_component_: PythonObject
    var k_learning_decay: PythonObject
    var k_learning_method: PythonObject
    var k_learning_offset: PythonObject
    var k_max_doc_update_iter: PythonObject
    var k_max_iter: PythonObject
    var k_mean_change_tol: PythonObject
    var k_n_batch_iter_: PythonObject
    var k_n_components: PythonObject
    var k_n_features_in_: PythonObject
    var k_n_iter_: PythonObject
    var k_perp_tol: PythonObject
    var k_random_state: PythonObject
    var k_topic_word_prior: PythonObject
    var k_topic_word_prior_: PythonObject
    var k_total_samples: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k__draw = PythonObject(None)
        self.k__seed = PythonObject(None)
        self.k_batch_size = PythonObject(None)
        self.k_bound_ = PythonObject(None)
        self.k_components_ = PythonObject(None)
        self.k_doc_topic_prior = PythonObject(None)
        self.k_doc_topic_prior_ = PythonObject(None)
        self.k_evaluate_every = PythonObject(None)
        self.k_exp_dirichlet_component_ = PythonObject(None)
        self.k_learning_decay = PythonObject(None)
        self.k_learning_method = PythonObject(None)
        self.k_learning_offset = PythonObject(None)
        self.k_max_doc_update_iter = PythonObject(None)
        self.k_max_iter = PythonObject(None)
        self.k_mean_change_tol = PythonObject(None)
        self.k_n_batch_iter_ = PythonObject(None)
        self.k_n_components = PythonObject(None)
        self.k_n_features_in_ = PythonObject(None)
        self.k_n_iter_ = PythonObject(None)
        self.k_perp_tol = PythonObject(None)
        self.k_random_state = PythonObject(None)
        self.k_topic_word_prior = PythonObject(None)
        self.k_topic_word_prior_ = PythonObject(None)
        self.k_total_samples = PythonObject(None)


# V2: the driver entry name key went (the driver is a table operation, lane dw-decomp).
comptime _LDA_KEYS = SharedSlot[LdaKeys, "MojolearnLdaKeysV2", 2]


def _lda_keys_fill(nk: UnsafePointer[LdaKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k__draw = py_key("_draw")
    nk[].k__seed = py_key("_seed")
    nk[].k_batch_size = py_key("batch_size")
    nk[].k_bound_ = py_key("bound_")
    nk[].k_components_ = py_key("components_")
    nk[].k_doc_topic_prior = py_key("doc_topic_prior")
    nk[].k_doc_topic_prior_ = py_key("doc_topic_prior_")
    nk[].k_evaluate_every = py_key("evaluate_every")
    nk[].k_exp_dirichlet_component_ = py_key("exp_dirichlet_component_")
    nk[].k_learning_decay = py_key("learning_decay")
    nk[].k_learning_method = py_key("learning_method")
    nk[].k_learning_offset = py_key("learning_offset")
    nk[].k_max_doc_update_iter = py_key("max_doc_update_iter")
    nk[].k_max_iter = py_key("max_iter")
    nk[].k_mean_change_tol = py_key("mean_change_tol")
    nk[].k_n_batch_iter_ = py_key("n_batch_iter_")
    nk[].k_n_components = py_key("n_components")
    nk[].k_n_features_in_ = py_key("n_features_in_")
    nk[].k_n_iter_ = py_key("n_iter_")
    nk[].k_perp_tol = py_key("perp_tol")
    nk[].k_random_state = py_key("random_state")
    nk[].k_topic_word_prior = py_key("topic_word_prior")
    nk[].k_topic_word_prior_ = py_key("topic_word_prior_")
    nk[].k_total_samples = py_key("total_samples")
    var ops = Python.dict()
    nk[].ops = ops
    nk[].ready = True


@always_inline
def lda_keys() raises -> UnsafePointer[LdaKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _LDA_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _lda_keys_fill(nk)
    return nk

comptime _FIT = 0
comptime _PARTIAL = 1
comptime _TRANSFORM = 2
comptime _SCORE = 3
comptime _PERPLEXITY = 4


def _address(array: PythonObject) raises -> Int:
    return Int(py=classical_address(array))


def _batch_size(fields: PythonObject) raises -> Int:
    """The online batch size: a positive int (not a bool)."""
    var nk = lda_keys()
    var b = native_loaded_module("builtins")
    var bs = py_item(fields, nk[].k_batch_size)
    if py_type(bs) is b.bool or not Bool(py=py_call(b.isinstance, bs, b.int)) or Int(py=bs) < 1:
        raise Error("batch_size must be a positive integer")
    return Int(py=bs)


def _init_fields(fields: PythonObject) raises:
    """`_init`'s attributes: the priors (1 / n_components when None), the
    seed and the draw counters."""
    var nk = lda_keys()
    var nc = py_item(fields, nk[].k_n_components)
    var doc = py_item(fields, nk[].k_doc_topic_prior)
    var topic = py_item(fields, nk[].k_topic_word_prior)
    var inverse = PythonObject(1.0 / decomp_float(nc))
    py_setitem(fields, nk[].k_doc_topic_prior_, inverse if doc is None else doc)
    py_setitem(fields, nk[].k_topic_word_prior_, inverse if topic is None else topic)
    # `_seed_of`: refused when not None or an int; kept as given (None: 0)
    var seed = py_item(fields, nk[].k_random_state)
    _ = decomp_seed(seed)
    py_setitem(fields, nk[].k__seed, PythonObject(0) if seed is None else seed)
    py_setitem(fields, nk[].k__draw, PythonObject(0))
    py_setitem(fields, nk[].k_n_batch_iter_, PythonObject(1))
    py_setitem(fields, nk[].k_n_iter_, PythonObject(0))


def _run(model: PythonObject, fields: PythonObject, op: Int, x: PythonObject, fresh: Bool, flag: Bool,
         total: Float64, refusal: Int, context: PythonObject) raises -> PythonObject:
    """One LDA driver call: (n_iter_, draw, n_batch_iter, value, components,
    exp_dir, out). `refusal` names the public method in the negativity
    refusal (LDA_REFUSAL_*)."""
    var nk = lda_keys()
    var entry = decomp_entries_ref(
        decomp_entries_routed(model, fields, decomp_attr(model, fields, "numeric_mode_"), context))
    var n = Int(py=x.shape[0])
    var v = Int(py=x.shape[1])
    var nc = decomp_int(py_item(fields, nk[].k_n_components))
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var state_c = Optional[PythonObject]()
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var state_e = Optional[PythonObject]()
    if not (op == _FIT or (op == _PARTIAL and fresh)):
        state_c = admit_x(decomp_attr(model, fields, "components_"), 2, "components_", context)
        state_e = admit_x(decomp_attr(model, fields, "exp_dirichlet_component_"), 2,
                                     "exp_dirichlet_component_", context)
        nc = Int(py=state_c.value().shape[0])
    # Only a fit writes comps_out / exp_out (x_decomp/lda_owner.mojo: the
    # LDA_FIT / LDA_PARTIAL arms copy_out into them; the query arms write
    # out_buf or nothing). A query (transform / score / perplexity) passes
    # the state's own words as those never-written slots, as the member
    # does (core/xdecomp_member.mojo `_query_lda`), instead of two fresh
    # (nc x v) arrays per call (lane ma-pub-transform round 9). Same RUN,
    # same words read: bits unchanged.
    var writes = op == _FIT or op == _PARTIAL
    var comps = decomp_f32(Python.tuple(nc, v), context) if writes else PythonObject(None)
    var exp_dir = decomp_f32(Python.tuple(nc, v), context) if writes else PythonObject(None)
    var out = decomp_f32(Python.tuple(n, nc), context) if op == _TRANSFORM else PythonObject(None)
    var spare = _address(comps) if writes else _address(state_c.value())
    var exp_slot = _address(exp_dir) if writes else _address(state_e.value())
    var online = decomp_eq(py_item(fields, nk[].k_learning_method), PythonObject("online"))
    var bs = 1
    if op == _PARTIAL or (op == _FIT and online):
        bs = _batch_size(fields)
    var request = LDARequest(
        op, _address(x),
        _address(state_c.value()) if Bool(state_c) else spare,
        _address(state_e.value()) if Bool(state_e) else spare,
        spare, exp_slot, _address(out) if out is not None else spare,
        n, v, nc, fresh, flag,
        decomp_int(py_item(fields, nk[].k_max_iter)), op == _FIT and online, bs,
        decomp_int(py_item(fields, nk[].k_max_doc_update_iter)),
        decomp_seed(decomp_attr(model, fields, "_seed")),
        decomp_int(py_item(fields, nk[].k_evaluate_every)), refusal,
        Int(py=py_get(fields, nk[].k__draw, PythonObject(0))),
        Int(py=py_get(fields, nk[].k_n_batch_iter_, PythonObject(1))),
        decomp_float(decomp_attr(model, fields, "doc_topic_prior_")),
        decomp_float(decomp_attr(model, fields, "topic_word_prior_")),
        decomp_float(py_item(fields, nk[].k_learning_offset)),
        decomp_float(py_item(fields, nk[].k_learning_decay)),
        decomp_float(py_item(fields, nk[].k_mean_change_tol)), total,
        decomp_float(py_item(fields, nk[].k_perp_tol)))
    var draw = 0
    var n_batch_iter = 0
    var iterations = 0
    var value = Float64(0.0)
    with GILReleased(Python()):
        var answer = lda_call(entry, request)
        iterations = answer.iterations
        draw = answer.draw
        n_batch_iter = answer.n_batch_iter
        value = answer.value
    _ = x  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = exp_dir  # keep the Python object alive through the native call above
    _ = comps  # keep the Python object alive through the native call above
    _ = state_e  # keep the Python object alive through the native call above
    _ = state_c  # keep the Python object alive through the native call above
    _ = state_c^
    _ = state_e^
    return Python.tuple(PythonObject(iterations), PythonObject(draw), PythonObject(n_batch_iter),
                        PythonObject(value), comps, exp_dir, out)


def lda_fit(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`LatentDirichletAllocation.fit`; returns the converted X (kept for
    fit_transform)."""
    var nk = lda_keys()
    var fields = py_fields(model)
    decomp_set_mode(fields, context)
    var method = py_item(fields, nk[].k_learning_method)
    if not Bool(py=py_call(native_loaded_module("operator").contains, Python.tuple("batch", "online"), method)):
        raise Error("learning_method must be 'batch' or 'online'")
    var x = decomp_input(source, "LatentDirichletAllocation", context)
    _init_fields(fields)
    var n = Int(py=x.shape[0])
    var got = _run(model, fields, _FIT, x, True, False, Float64(n),
                   LDA_REFUSAL_FIT, context)
    py_setitem(fields, nk[].k__draw, got[1])
    py_setitem(fields, nk[].k_n_batch_iter_, got[2])
    py_setitem(fields, nk[].k_n_iter_, PythonObject(Int(py=got[0])))
    py_setitem(fields, nk[].k_bound_, PythonObject(Float64(py=got[3])))
    py_setitem(fields, nk[].k_components_, got[4])
    py_setitem(fields, nk[].k_exp_dirichlet_component_, got[5])
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(Int(py=x.shape[1])))
    return x


def lda_partial_fit(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`LatentDirichletAllocation.partial_fit`: one online pass."""
    var nk = lda_keys()
    var fields = py_fields(model)
    var fresh = not py_contains(fields, nk[].k_components_)
    if fresh:
        decomp_set_mode(fields, context)
    var x = decomp_input(source, "LatentDirichletAllocation", context)
    if fresh:
        _init_fields(fields)
    var got = _run(model, fields, _PARTIAL, x, fresh, False, decomp_float(py_item(fields, nk[].k_total_samples)),
                   LDA_REFUSAL_PARTIAL_FIT, context)
    py_setitem(fields, nk[].k__draw, got[1])
    py_setitem(fields, nk[].k_n_batch_iter_, got[2])
    py_setitem(fields, nk[].k_components_, got[4])
    py_setitem(fields, nk[].k_exp_dirichlet_component_, got[5])
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(Int(py=x.shape[1])))
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
    var nk = lda_keys()
    var got = _run(model, fields, op, x, False, flag, decomp_float(py_item(fields, nk[].k_total_samples)),
                   LDA_REFUSAL_TRANSFORM, context)
    if op == _TRANSFORM:
        return got[6]
    return PythonObject(Float64(py=got[3]))
