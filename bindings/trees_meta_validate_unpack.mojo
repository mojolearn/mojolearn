# SPDX-License-Identifier: Apache-2.0
"""The meta-estimators' constructor refusals: ONE body for the fit doors and the member admission
(lane mf-refuse5, THE FALLBACK RULE; the pattern of core/gbdt_cat_words.mojo).

Each function here is the check a meta-estimator's fit door runs on its constructor values before it
reads X, in the door's order and words: VotingClassifier / VotingRegressor (`meta_voting_validate`,
`meta_voting_weights`), StackingClassifier / StackingRegressor (`meta_stacking_validate`),
OneVsRestClassifier (`meta_jobs`), CalibratedClassifierCV (`meta_calibrated_validate`),
BaggingClassifier / BaggingRegressor (`meta_bagging_validate`), AdaBoostClassifier / AdaBoostRegressor
(`meta_ada_validate`, `meta_seed`). The doors (bindings/trees_public_meta_door.mojo,
bindings/trees_public_boost_door.mojo) call them; so does the META kind's admission
(providers/python/member_ensemble_boundary_door.mojo), which catches the words and hands them to the
plan entry (core/ensemble_member.mojo `meta_plan_refuse`), whose FIT raises them where the door raises
them. Before this lane the admission answered False for such a value and its caller fell back to the
door only to raise these words.

UNPACK only (Python objects, the GIL held): every function reads constructor values once per fit or
admission, never per row or per member call. SERIAL BY NATURE: one estimator's settings.
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_item, py_fields
from bindings.trees_boost_admission_unpack import trees_member_kind
from bindings.trees_public_inputs_door import trees_builtins
from bindings.trees_public_forest_door import _int, _float, _repr, _str, _truthy, _is_bool, _is_integral
from bindings.trees_keys_unpack import trees_keys
from bindings.native_unit_threads_unpack import native_unit_jobs
from bindings.prep_native_estimator_door import prep_native_context, prep_estimator_admit
from bindings.msel_inputs_unpack import msel_admit
from bindings.py_typed_call import py_call_typed
from xtrees.trees_forest_run import _refuse
from xtrees.trees_boost_run import (
    boost_bagging_validate, boost_check_n_estimators, boost_check_learning_rate, boost_count_fraction_ok,
)


def meta_attr(owner: PythonObject, key: PythonObject, default: PythonObject) raises -> PythonObject:
    """`getattr(owner, name, default)` with the held key of the name (the meta door's `_attr`)."""
    return py_call(trees_builtins().getattr, owner, key, default)


def meta_owner(model: PythonObject) raises -> String:
    """The estimator's class name (the meta door's `_owner`)."""
    return String(py_type(model).__name__)


def meta_jobs(model: PythonObject, verbose: Bool) raises -> Int:
    """The worker bound of a meta-estimator's member fits from its `n_jobs`
    (`native_unit_jobs`: None and -1 every core, k > 0 at most k), after the
    `verbose` refusal where the estimator has that parameter. Members of
    published families fit in parallel (bindings/member_set_door.mojo);
    members on their own doors fit one after another whatever the bound.
    The results do not depend on it."""
    # TOMBSTONE: _refuse_jobs (n_jobs refused: "the members / columns / classes fit one after another") deleted
    # 2026-11-21 by lane ma-callers round 2: published members fit in parallel, so n_jobs is the worker bound.
    # Restore: git revert of that commit (no experiment define; a refusal, not an arm).
    var nk = trees_keys()
    if verbose and _truthy(meta_attr(model, nk[].k_verbose, PythonObject(0))):
        _refuse("verbose", "nothing logs per member.")
    return native_unit_jobs(meta_attr(model, nk[].k_n_jobs, PythonObject(None)), meta_owner(model))


def meta_cv_value(cv: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_trees_cv(cv)`: an int number of folds (None: 5), or `cv` itself when
    it is a splitter or an iterable of (train, test) index pairs."""
    var b = trees_builtins()
    if cv is None:
        return PythonObject(5)
    var native = prep_native_context(context)
    var nk = trees_keys()
    if py_type(py_type(cv)) is b.type and py_contains(py_item(native, nk[].k_family_msel_splitter_layouts), py_type(cv)):
        return cv
    if py_type(cv) is b.list or py_type(cv) is b.tuple:
        msel_admit(cv, native)
        return cv
    if py_type(cv) is not b.int:
        raise Error("TypeError: cv requires an integer, native splitter or list of index pairs; Python callbacks cannot execute")
    if _is_bool(cv) or not _is_integral(cv, context) or Bool(py=cv < PythonObject(2)):
        raise Error("cv=" + _repr(cv) + ": an int >= 2, a splitter with split(X, y), or an iterable of"
                    + " (train, test) index pairs")
    return py_call(b.int, cv)


def meta_voting_validate(model: PythonObject, classifier: Bool) raises -> Int:
    """`VotingClassifier` / `VotingRegressor.fit`'s refusals before X: the
    voting value (a classifier's), then `meta_jobs`. Answers the worker bound."""
    if classifier:
        var nk = trees_keys()
        var voting = meta_attr(model, nk[].k_voting, PythonObject(None))
        if not py_contains(Python.tuple("hard", "soft"), voting):
            raise Error("voting must be 'hard' or 'soft'")
    return meta_jobs(model, True)


struct MetaActive(Copyable, Movable):
    """`_trees_estimators(estimators)`: the active names and estimators."""
    var names: PythonObject
    var ests: PythonObject

    def __init__(out self, names: PythonObject, ests: PythonObject):
        self.names = names
        self.ests = ests


def meta_active(estimators: PythonObject, owner: String, context: PythonObject) raises -> MetaActive:
    """The Voting / Stacking doors' `_active`: the (name, estimator) entries
    (`meta_active_entries`), each active estimator admitted
    (`prep_estimator_admit`, a class with no native plan refused in its
    words). The META admission reads its members through it too."""
    var got = meta_active_entries(estimators)
    for i in range(len(got.ests)):  # driver-loop: over members
        prep_estimator_admit(got.ests[i], owner, context)
    return got^


def meta_active_entries(estimators: PythonObject) raises -> MetaActive:
    """The (name, estimator) entries of `estimators` with 'drop' left out,
    in the doors' words for a value they refuse. The META kind's PACK and
    its work-list read the same entries (no admission: its admission did)."""
    var b = trees_builtins()
    if not _truthy(estimators):
        raise Error("estimators must be a non-empty list of (name, estimator)")
    var items = py_call(b.list, estimators)
    var all_names = Python.list()
    var names = Python.list()
    var ests = Python.list()
    for i in range(len(items)):  # driver-loop: over the (name, estimator) entries
        var pair = py_call(b.tuple, items[i])
        if len(pair) != 2:
            raise Error("estimators must be a non-empty list of (name, estimator)")
        _ = py_call(all_names.append, pair[0])
    if len(py_call(b.set, all_names)) != len(all_names):
        raise Error("estimator names must be unique")
    for i in range(len(items)):  # driver-loop: over the (name, estimator) entries
        var pair = py_call(b.tuple, items[i])
        var est = pair[1]
        if py_type(est) is b.str and String(est) == "drop":
            continue
        _ = py_call(names.append, pair[0])
        _ = py_call(ests.append, est)
    if len(ests) == 0:
        raise Error("all estimators are 'drop'")
    return MetaActive(names, ests)


def meta_voting_weights(given: PythonObject, m: Int) raises -> PythonObject:
    """The vote's float weights over `m` members (`_voting_weights`): one per
    member, 1.0 each when `given` is None; every given item through `float`
    first, then the count refusal, as the door reads them after its fits."""
    var b = trees_builtins()
    var w = Python.list()
    if given is None:
        for _ in range(m):  # driver-loop: over members
            _ = py_call(w.append, PythonObject(1.0))
    else:
        var items = py_call(b.list, given)
        for i in range(len(items)):  # driver-loop: over the user weights (one per member)
            _ = py_call(w.append, py_call_typed(b.float, items[i]))
    if len(w) != m:
        raise Error("weights has " + String(len(w)) + " entries, there are " + String(m) + " estimators")
    return w


def meta_stacking_validate(model: PythonObject, context: PythonObject) raises:
    """`StackingClassifier` / `StackingRegressor.fit`'s refusals before X:
    `meta_jobs`, the stack_method value, the cv value (`meta_cv_value`)."""
    _ = meta_jobs(model, True)
    var nk = trees_keys()
    var method = meta_attr(model, nk[].k_stack_method, PythonObject("auto"))
    if not py_contains(Python.tuple("auto", "predict_proba", "decision_function", "predict"), method):
        raise Error("stack_method must be auto, predict_proba, decision_function or predict")
    _ = meta_cv_value(meta_attr(model, nk[].k_cv, PythonObject(None)), context)


def meta_calibrated_validate(model: PythonObject, context: PythonObject) raises:
    """`CalibratedClassifierCV.fit`'s refusals before X: method, n_jobs,
    ensemble, the cv value (`meta_cv_value`)."""
    var nk = trees_keys()
    var method = meta_attr(model, nk[].k_method, PythonObject("sigmoid"))
    if not py_contains(Python.tuple("sigmoid", "isotonic"), method):
        raise Error("method must be 'sigmoid' or 'isotonic'")
    _ = native_unit_jobs(meta_attr(model, nk[].k_n_jobs, PythonObject(None)), "CalibratedClassifierCV")
    var ensemble = meta_attr(model, nk[].k_ensemble, PythonObject(True))
    if not py_contains(Python.tuple(True, False), ensemble):
        _refuse("ensemble=" + _repr(ensemble), "True or False.")
    _ = meta_cv_value(meta_attr(model, nk[].k_cv, PythonObject(None)), context)


def meta_seed(random_state: PythonObject, context: PythonObject) raises -> Int:
    """`_trees_seed`."""
    if random_state is None:
        return 0
    if _is_bool(random_state) or not _is_integral(random_state, context) or Bool(py=random_state < PythonObject(0)):
        raise Error("random_state must be None or a nonnegative int (a RandomState object is refused:"
                    + " draws come from the lane's own counter RNG, xtrees/ops.mojo)")
    return Int(py=_int(random_state))


def meta_bagging_validate(fields: PythonObject, context: PythonObject) raises:
    """`_BaggingBase._validate_params`."""
    var oob = _truthy(fields["oob_score"])
    boost_bagging_validate(oob, oob and _truthy(fields["bootstrap"]), _truthy(fields["warm_start"]),
                           _truthy(fields["verbose"]))
    # n_jobs is the worker bound of the member fits (None and -1: every core):
    # published members fit in parallel waves (`_bagging_fit_bags`).
    _ = native_unit_jobs(fields["n_jobs"], "Bagging")
    boost_check_n_estimators(Int(py=_int(fields["n_estimators"])))
    _ = meta_seed(fields["random_state"], context)


def meta_ada_validate(fields: PythonObject, classifier: Bool, context: PythonObject) raises:
    """`AdaBoost*._validate_params`."""
    if classifier:
        var algorithm = fields["algorithm"]
        if not Bool(py=algorithm == PythonObject("SAMME")) and not Bool(py=algorithm == PythonObject("deprecated")):
            _refuse("algorithm=" + _repr(algorithm), "SAMME.R was removed from the reference (sklearn 1.6);"
                    + " SAMME is the algorithm.")
    else:
        var losses = context["trees_ada_losses"]
        if not py_contains(losses, fields["loss"]):
            raise Error("loss must be one of " + _repr(py_call(trees_builtins().tuple, losses)) + ", got "
                        + _repr(fields["loss"]))
    boost_check_n_estimators(Int(py=_int(fields["n_estimators"])))
    boost_check_learning_rate(Float64(py=_float(fields["learning_rate"])))
    _ = meta_seed(fields["random_state"], context)


struct MetaCount(Copyable, Movable):
    """`_BaggingBase._count`'s reading of max_samples / max_features before
    the total is known: an integral count, or a fraction in (0, 1]; `every`:
    max_samples=None, every row (scikit-learn 1.9.1's default,
    sklearn/ensemble/_bootstrap.py:44-45: n whatever the sample weights)."""
    var integral: Bool
    var count: Int
    var fraction: Float64
    var every: Bool

    def __init__(out self, integral: Bool, count: Int, fraction: Float64, every: Bool = False):
        self.integral = integral
        self.count = count
        self.fraction = fraction
        self.every = every


def meta_count_reading(v: PythonObject, name: String, context: PythonObject) raises -> MetaCount:
    """`_BaggingBase._count`'s reading and the refusals that need no total
    (the door's `_count` then resolves the count and refuses it)."""
    var b = trees_builtins()
    if v is None and name == "max_samples":
        return MetaCount(True, 0, 0.0, True)
    if _is_integral(v, context) and not _is_bool(v):
        return MetaCount(True, Int(py=_int(v)), 0.0)
    if Bool(py=py_call(b.isinstance, v, context["trees_real_type"])):
        var f = Float64(py=_float(v))
        if not boost_count_fraction_ok(f):
            raise Error(name + "=" + _str(v) + " is outside (0, 1]")
        return MetaCount(False, 0, f)
    raise Error(name + " must be an int or a float")


def meta_connected(cls: PythonObject, context: PythonObject) raises -> Bool:
    """A class whose ancestry holds a connected tree member kind."""
    var b = trees_builtins()
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var classes = context["trees_classes"]
    for i in range(len(mro)):  # driver-loop: over the class ancestry metadata
        var kind = py_get(classes, mro[i])
        if kind is not None:
            return trees_member_kind(kind)
    return False


def meta_require_tree(base: PythonObject, owner: String, context: PythonObject) raises:
    """A member must be a mojolearn tree estimator (its fit and predict run
    in this owner); any other estimator is a Python object."""
    if not meta_connected(py_type(base), context):
        raise Error("NotImplementedError: " + owner + ": estimator=" + _repr(base) + " is not a mojolearn "
                    + "DecisionTree*, RandomForest* or ExtraTrees* estimator; the native ensemble fits only "
                    + "those members. Refused by name rather than accepted and ignored.")


def boost_member_device(fields: PythonObject, base: PythonObject, owner: String) raises -> PythonObject:
    """The `device` every member of an ensemble stores: the ensemble's own
    stored value, so a member read back (`estimators_`) runs where the
    ensemble ran it. The members' operations run in the ensemble's run
    context (its device route, bindings/trees_route_boundary_door.mojo);
    nothing here selects a table. A member estimator given with its own
    explicit `device` that is not the ensemble's value is refused in words,
    never run on a device its caller did not name. Once per ensemble fit:
    two field reads and one pointer compare when both are the default."""
    var device = py_get(fields, "device")
    var own = py_get(py_fields(base), "device")
    if own is not None and own is not device:
        var same = False
        if device is not None:
            same = String(own) == String(device)
        if not same:
            raise Error(owner + ": estimator has device=" + _repr(own) + " but the ensemble has device="
                        + _repr(device) + "; the members run on the ensemble's device. Set device on the "
                        + "ensemble and leave the member's device None.")
    return device
