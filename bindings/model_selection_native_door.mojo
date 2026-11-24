# SPDX-License-Identifier: Apache-2.0
# model_selection's public scorer functions and the search estimators'
# delegations, one native call each (python/mojolearn/model_selection.py:
# `make_scorer`, `get_scorer`, `get_scorer_names`, `GridSearchCV` /
# `RandomizedSearchCV` `predict`, `predict_proba`, `decision_function`,
# `transform` and `score`). The scorer table is fixed at import; the
# refusals keep the Python functions' wording and order. A delegated method
# is the caller's fitted estimator's own (a mojolearn estimator's is itself
# one native call); its exception reaches the caller unchanged.
from bindings.native_loaded_module_door import native_loaded_module
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setattr, py_setitem
from bindings.py_attr import py_attr
from bindings.native_buffer_owner_pack import native_storage_allocate
from bindings.family_common_unpack import family_repr, family_is_str, family_isinstance
from bindings.family_common_pack import family_call
from bindings.msel_common_door import msel_get_scorer, msel_score, msel_admit, msel_flatten, msel_numeric_vector
from bindings.msel_scorer_native_door import msel_metric_plan, msel_scorer_call, msel_scorer_create
from bindings.msel_estimator_native_door import (
    msel_estimator_call, msel_estimator_plan, msel_native_clone, MSEL_MEMBER_ONLY_OWN_WORDS,
)
from bindings.msel_member_units_door import msel_member_fit_in_place
from bindings.msel_member_device_unpack import msel_own_device_admit, msel_process_vendor, member_says_cpu
from core.device_route import DEVICE_CPU, device_vendor_word
from providers.python.device_route_boundary_door import device_route_resolved
from bindings.parallel_cv_native_door import msel_parallel_operation
from bindings.msel_cv_door import msel_cross_validate, msel_cross_val_score, msel_cross_val_predict, msel_split_descriptor
from bindings.msel_search_door import (
    msel_grid_dicts, msel_grid_list, msel_grid_len, msel_sampler_check, msel_sampler_list, msel_sampler_len, msel_search_fit, msel_validation_curve,
    msel_learning_curve, msel_permutation_test_score, _search_index,
)
from bindings.msel_parameter_grid_door import PreparedParameterGrid
from bindings.msel_public_metadata_pack import msel_splitter_repr, msel_scorer_repr, msel_estimator_classes
from bindings.msel_search_state_door import msel_search_fields, msel_search_plan, msel_search_initialize, msel_search_update
from bindings.msel_split_door import (
    msel_split, msel_n_splits, msel_loo_at, msel_lpo_at, msel_logo_at, msel_lpgo_at, msel_check_cv,
    msel_train_test_split, msel_truth, msel_flag_truth, msel_splitter_initialize, msel_default_folds,
)


def _msel_attr(model: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    return _msel_field(msel_search_fields(model, context), name)


def _msel_field(fields: PythonObject, name: String) raises -> PythonObject:
    # One field of an already admitted search fields dict (perf2-msel: an
    # operation that reads several fields admits the dict once, not per field).
    if not py_contains(fields, PythonObject(name)):
        raise Error("AttributeError: native search field is unavailable: " + name)
    return fields[PythonObject(name)]


def _msel_best(model: PythonObject, context: PythonObject) raises -> PythonObject:
    return _msel_best_of(model, msel_search_fields(model, context))


def _msel_best_of(model: PythonObject, fields: PythonObject) raises -> PythonObject:
    if not py_contains(fields, "best_estimator_"):
        raise Error("AttributeError: This " + String(py_type(model).__name__)
                    + " instance was initialized with refit=False or is not fitted.")
    return fields["best_estimator_"]


def _msel_member_only_call(model: PythonObject, method: String, args: PythonObject, keywords: PythonObject,
                           context: PythonObject, b: PythonObject) raises -> PythonObject:
    """Lane w3f3-b: what a MEMBER-ONLY plan (no method recipes) serves through
    an owner's estimator call (RFE, IterativeImputer, the cv tools):
      * a search's `score`: its own score, the refit scorer over its best
        estimator (scikit-learn 1.9 model_selection/_search.py
        `BaseSearchCV.score`: `self.scorer_[self.refit]` or `self.scorer_`
        over `self.best_estimator_`; with scoring=None that scorer is the
        best estimator's own `score`), `_msel_search_score`;
      * a search's `fit`: its own fit door (`msel_search_fit`);
      * any other member-only class's `fit`: its member, fitted on every row
        and packed into it in place (bindings/msel_member_units_door.mojo
        `msel_member_fit_in_place`), which raises its family's words for
        what the member does not take. Fit keywords are not handed to the
        member route: a call with any is refused in words naming them.
    Reached only on that plan's miss path (`MSEL_MEMBER_ONLY_OWN_WORDS`)."""
    if method == "score":
        var y = args[1] if len(args) > 1 else PythonObject(None)
        return _msel_search_score(model, args[0], y, context, b)
    var y = args[1] if len(args) > 1 else PythonObject(None)
    if keywords is not None and Int(py=py_call(b.len, keywords)) > 0:
        raise Error("NotImplementedError: " + String(py_type(model).__name__) + ".fit keywords "
                    + String(py=py_call(b.str, py_call(b.list, keywords.keys())))
                    + " do not reach its native member when another estimator fits it; fit it on its own")
    var layouts = py_get(context, PythonObject("family_msel_search_layouts"))
    if layouts is not None and py_get(layouts, py_type(model)) is not None:
        _ = msel_search_fit(model, Python.tuple(args[0], y, PythonObject(None)), context)
        return model
    msel_member_fit_in_place(model, y, args[0], context, b)
    return model


def _msel_search_score(model: PythonObject, x: PythonObject, y: PythonObject, context: PythonObject,
                       b: PythonObject) raises -> PythonObject:
    """`BaseSearchCV.score` (the `family_msel_search_score` operation's body)."""
    # The scorer is resolved first, as `score` resolved it before
    # reaching the best estimator.
    # The fields dict is admitted once here and every field read from it
    # (perf2-msel). The reads that ran in between (msel_admit,
    # msel_flag_truth, msel_get_scorer) run no Python code that could
    # replace a search field, so the values read are the same.
    var fields = msel_search_fields(model, context)
    var scorer: PythonObject
    var multimetric = _msel_field(fields, "multimetric_")
    msel_admit(multimetric, context)
    if not msel_flag_truth(multimetric, context):
        scorer = msel_get_scorer(_msel_field(fields, "scoring"), context)
    else:
        var scorers = _msel_field(fields, "scorer_")
        var refit = _msel_field(fields, "refit")
        if py_type(scorers) is not b.dict or py_type(refit) is not b.str:
            raise Error("TypeError: fitted native search scorer state requires a dict and exact string key")
        scorer = scorers[refit]
    var best = _msel_best_of(model, fields)
    return msel_score(best, x, y, scorer, context)


def msel_operation(model: PythonObject, operation: String, args: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if operation.startswith("family_msel_parallel_"):
        return msel_parallel_operation(operation, args, context)
    if operation == "family_msel_estimator_call":
        # An owner's inner estimator (RFE, IterativeImputer;
        # bindings/prep_native_estimator.mojo): `model.<args[0]>(*args[1])`
        # as the compiled operation of its connected plan, never its Python method.
        if py_type(args[0]) is not b.str or py_type(args[1]) is not b.tuple:
            raise Error("TypeError: native estimator call requires a method name and an argument tuple")
        # args[3] True (prep_estimator_predict_internal): a package-internal
        # caller reading float32 words; `predict` takes the plan's internal
        # operation (`scorer_predict_operation`), as the native scorers do.
        var internal = len(args) > 3 and args[3] is PythonObject(True)
        var keywords = args[2] if len(args) > 2 else PythonObject(None)
        try:
            return msel_estimator_call(model, String(args[0]), args[1], context, internal, keywords)
        except e:
            # lane w3f3-b: a member-only plan's fit, a search's score (the
            # words are raised only on a member-only plan's miss path)
            if String(e) != MSEL_MEMBER_ONLY_OWN_WORDS:
                raise e^
            return _msel_member_only_call(model, String(args[0]), args[1], keywords, context, b)
    if operation == "family_msel_estimator_clone":
        # `type(model)(**model.get_params(deep=False))` from the constructor schema.
        return msel_native_clone(model, context)
    if operation == "family_msel_search_init":
        # The search's own `device` is the last constructor argument of the
        # three search classes (python/mojolearn/_native_msel_search.py). An
        # unknown value, and 'gpu' on a CPU-only install, are refused in
        # words before any field is stored; the default None is one compare.
        if len(args) > 0:
            msel_own_device_admit(args[len(args) - 1], b, context)
        return msel_search_initialize(model, args, context)
    if operation == "family_msel_grid_init":
        var grid = msel_grid_dicts(args[0], context)
        return msel_search_initialize(model, Python.tuple(grid), context)
    if operation == "family_msel_sampler_init":
        _ = msel_sampler_check(args[0], context)
        return msel_search_initialize(model, args, context)
    if operation == "family_msel_search_classes":
        return msel_estimator_classes(_msel_best(model, context), context)
    if operation == "family_msel_splitter_repr":
        return msel_splitter_repr(model, context)
    if operation == "family_msel_scorer_repr":
        return msel_scorer_repr(model, context)
    if operation == "family_msel_search_get_params":
        var plan = msel_search_plan(model, context)
        var fields = msel_search_fields(model, context)
        var result = Python.dict()
        var names = plan["parameters"]
        for k in range(len(names)):  # driver-loop: fixed search constructor schema
            py_setitem(result, names[k], fields[names[k]])
        return result
    if operation == "family_msel_search_set_params":
        var fields = msel_search_fields(model, context)
        if py_type(args[0]) is not b.dict:
            raise Error("TypeError: search parameter keywords must be a native dict")
        var names = py_call(b.list, args[0].keys())
        for k in range(len(names)):  # driver-loop: existing search setter accepts plain stored parameter names
            if py_type(names[k]) is not b.str:
                raise Error("TypeError: search parameter names must be exact strings")
            if py_contains(msel_search_plan(model, context)["protected"], names[k]):
                raise Error("TypeError: search parameters cannot override native method or storage hooks")
        # A new `device` is checked before any field changes (the same rule
        # and words as the constructor); a refusal leaves the search as it was.
        var device = py_get(args[0], PythonObject("device"))
        if device is not None:
            msel_own_device_admit(device, b, context)
        msel_search_update(fields, args[0], context)
        return model
    if operation == "family_msel_search_estimator_type":
        var plan = msel_estimator_plan(_msel_attr(model, "estimator", context), context)
        return plan["estimator_type"]
    if operation == "family_msel_get_scorer":
        return msel_get_scorer(args[0], context)
    if operation == "family_msel_scorer_names":
        return py_call(b.sorted, context["family_msel_scorers"])
    if operation == "family_msel_scorer_call":
        return msel_scorer_call(model, args[0], args[1], args[2], args[3], args[4], context)
    if operation == "family_msel_make_scorer":
        var sign = 1 if msel_flag_truth(args[2], context) else -1
        return msel_scorer_create(args[0], args[1], sign, args[3], context)
    if operation == "family_msel_delegate":
        var best = _msel_best(model, context)
        if py_type(args[0]) is not b.str:
            raise Error("TypeError: native delegated method must be an exact string")
        return msel_estimator_call(best, String(args[0]), Python.tuple(args[1]), context)
    if operation == "family_msel_search_vendor_used":
        # A fitted search answers with its best estimator, so the vendor it
        # used is the one that estimator answers on: "cpu" when the best
        # estimator's device route resolved to the CPU beside a GPU set (or,
        # before its first operation, when its admitted `device` says so),
        # else this process's vendor. A metadata answer: two record reads,
        # no member method is called.
        var best = _msel_best(model, context)
        var vendor = msel_process_vendor(context)
        if vendor != 0:
            var resolved = device_route_resolved(best)
            if resolved < 0:
                var member = msel_estimator_plan(best, context)
                if member_says_cpu(py_call(member["fields_get"], best, py_type(best)), b):
                    resolved = DEVICE_CPU
            if resolved == DEVICE_CPU:
                return PythonObject("cpu")
        return PythonObject(device_vendor_word(vendor))
    if operation == "family_msel_search_score":
        return _msel_search_score(model, args[0], args[1], context, b)
    if operation == "family_msel_splitter_init":
        return msel_splitter_initialize(model, args, context)
    if operation == "family_msel_split":
        # Preserve the public iterator contract without a Python generator
        # wrapper resuming around every native fold. Lazy native splitters
        # already return an iterator; iter() leaves those unchanged.
        return py_call(b.iter, msel_split(model, args[0], args[1], args[2], context))
    if operation == "family_msel_n_splits":
        return msel_n_splits(model, args[0], args[1], args[2], context)
    if operation == "family_msel_split_loo_at":
        return msel_loo_at(args, context)
    if operation == "family_msel_split_lpo_at":
        return msel_lpo_at(args, context)
    if operation == "family_msel_split_logo_at":
        return msel_logo_at(args, context)
    if operation == "family_msel_split_lpgo_at":
        return msel_lpgo_at(args, context)
    if operation == "family_msel_check_cv":
        var classifier = msel_flag_truth(args[2], context)
        _ = msel_flag_truth(args[3], context)
        msel_admit(args[4], context)
        # A numeric ndarray/Array label vector goes to msel_native_discrete as
        # is (one buffer reduction); flattening boxed every label first.
        var labels = args[1]
        if classifier and labels is not None and not msel_numeric_vector(labels, False, context):
            labels = msel_flatten(labels, context)
        return msel_check_cv(args[0], labels, classifier, args[3], args[4], context)
    if operation == "family_msel_train_test_split":
        return msel_train_test_split(args[0], args[1], args[2], args[3], args[4], args[5], context)
    if operation == "family_msel_cross_validate":
        return msel_cross_validate(args, context)
    if operation == "family_msel_cross_val_score":
        return msel_cross_val_score(args, context)
    if operation == "family_msel_cross_val_predict":
        return msel_cross_val_predict(args, context)
    if operation == "family_msel_split_descriptor":
        return msel_split_descriptor(args, context)
    if operation == "family_msel_grid_iter":
        return py_call(b.iter, msel_grid_list(_msel_attr(model, "param_grid", context), context))
    if operation == "family_msel_grid_len":
        return PythonObject(msel_grid_len(_msel_attr(model, "param_grid", context), context))
    if operation == "family_msel_grid_item":
        var index = _search_index(args[0], "ParameterGrid index", context)
        # One candidate by its mixed-radix digits, not the whole grid built
        # to return one entry (lane perf-sweep-msel-nbr-svc).
        var prepared = PreparedParameterGrid(_msel_attr(model, "param_grid", context), context)
        return prepared.item(Int(py=index))
    if operation == "family_msel_sampler_iter":
        var result = msel_sampler_list(_msel_attr(model, "param_distributions", context), _msel_attr(model, "n_iter", context),
                                       _msel_attr(model, "random_state", context), context)
        return py_call(b.iter, result)
    if operation == "family_msel_sampler_len":
        return msel_sampler_len(_msel_attr(model, "param_distributions", context), _msel_attr(model, "n_iter", context), context)
    if operation == "family_msel_search_fit":
        return msel_search_fit(model, args, context)
    if operation == "family_msel_validation_curve":
        return msel_validation_curve(args, context)
    if operation == "family_msel_learning_curve":
        return msel_learning_curve(args, context)
    if operation == "family_msel_permutation_test_score":
        return msel_permutation_test_score(args, context)
    if operation == "family_msel_default_folds":
        # The default unshuffled folds cross_val_score / cross_validate take
        # for cv=None or an integer (msel_fold_source), fold-order sabotage
        # control included, without an estimator: (labels, n_splits,
        # classifier). Verify's cross-val-folds lane hashes this exact route
        # (tools/identity_break.py); last in the chain, so no public
        # operation pays its comparison.
        return msel_default_folds(args[0], args[1], msel_flag_truth(args[2], context), context)
    raise Error("RuntimeError: unknown model_selection operation " + operation)
