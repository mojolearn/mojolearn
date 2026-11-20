# SPDX-License-Identifier: Apache-2.0
# The lda family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add



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
    var k_x_decomp_lda_owner: PythonObject

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
        self.k_x_decomp_lda_owner = PythonObject(None)


comptime _LDA_KEYS = SharedSlot[LdaKeys, "MojolearnLdaKeysV1"]


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
    nk[].k_x_decomp_lda_owner = py_key("x_decomp_lda_owner")
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
