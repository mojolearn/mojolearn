# SPDX-License-Identifier: Apache-2.0
# Hyperparameter search and the validation tools of
# python/mojolearn/model_selection.py, natively (lane mcb2-msel):
# `ParameterGrid` / `ParameterSampler` iteration, the shared `_BaseSearch.fit`
# of GridSearchCV and RandomizedSearchCV, `validation_curve`,
# `learning_curve` and `permutation_test_score`. The folds are drawn ONCE
# and X, y and the fold indices stay resident for every candidate, value,
# size or permutation (`MselFoldRows`); the loops over candidates, folds,
# sizes and permutations are this owner's; the estimator and the scorers are
# the caller's objects. The score tables use the `row_means_f64` /
# `row_stds_f64` passes (through bindings/msel_search_stats_run.mojo) and the permutations the same counter
# draws, so every result is unchanged.
from std.memory import unsafe_memcpy
from core.admitted_convert import admitted_convert_vector
from bindings.admitted_family_door import admit_family
from bindings.native_loaded_module_door import native_loaded_module
from bindings.metrics_exp_common_unpack import mx_raise_obj
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_setattr, py_contains, py_fields
from bindings.py_attr import py_attr
from bindings.native_buffer_owner_pack import native_storage_allocate
from bindings.classical_arrays_door import classical_wrap_array
from bindings.arena_materialize_native_door import _cast
from bindings.family_common_unpack import (
    FAMILY_I64P,
    family_isinstance,
    family_is_str,
    family_repr,
    family_str,
    family_type_name,
)
from bindings.family_common_pack import (
    family_call,
)
from bindings.metrics_program_door import MetricDoorProg, metric_door_run
from bindings.msel_common_door import (
    msel_take_labels, msel_b, msel_try, msel_missed, msel_catch, msel_warn, msel_is_bool, msel_is_integral, msel_is_real,
    msel_int, msel_op, msel_cmp, msel_materialize, msel_try_materialize, msel_flatten, msel_is_array, msel_addr,
    msel_p64, msel_pf64, msel_new, msel_new2, msel_f64_list, msel_f64_block, msel_i64_list, msel_i32_list,
    msel_slice_i64, msel_index_i64, msel_get_scorer, msel_score, msel_admit, MselRng, MselGroups, msel_apply,
)
from bindings.msel_split_door import msel_truth, msel_flag_truth, msel_field, msel_cls, msel_sabotage
from bindings.msel_common_door import msel_signed32
from bindings.msel_helpers_key_unpack import msel_helpers_key
from bindings.msel_member_device_unpack import msel_member_host_folds, msel_params_name_device
from bindings.msel_helpers_released_unpack import msel_gather_i64
# lane calltable-msel: the score-table passes both base bindings exported,
# linked into this door instead of called through CPython.
from bindings.msel_search_stats_run import (
    msel_row_means, msel_row_stds, msel_score_column, msel_rank_scores, msel_rank_best, msel_pvalue,
)
from bindings.family_common_unpack import FAMILY_I32P
from bindings.msel_member_units_door import msel_member_refit, msel_member_fit_cloned, msel_text_rows
from bindings.msel_cv_door import (
    msel_clone,
    msel_cv_folds,
    msel_cv_splitter,
    msel_cv_folds_admitted,
    msel_materialize_y,
    msel_device_rows_ok,
    msel_rows_scope_open,
    msel_rows_scope_close,
    msel_unit_jobs,
    msel_cv_units,
    msel_scorers,
    MselFoldRows,
    MselScores,
    msel_cross_validate_run,
    msel_cv_by_search,
    msel_fit,
    msel_fit_planned,
    msel_error_value,
)
from bindings.msel_cv_run import msel_seconds
from bindings.msel_estimator_native_door import msel_native_clone_planned, msel_native_clone

from bindings.msel_estimator_native_door import msel_native_set_params, msel_estimator_plan
from bindings.msel_search_state_door import (
    msel_search_fields, msel_parameter_values, msel_search_update, msel_carrier_search,
)
from bindings.msel_candidates_door import (
    _search_scalar, _search_index, _search_seed, msel_grid_dicts, msel_grid_list, msel_grid_len, msel_sampler_check,
    msel_sampler_list, msel_sampler_len, msel_candidates,
)
from bindings.py_optional import py_present
from core.msel_refusal_words import MSEL_REFIT_CALLBACK_WORDS


# The candidates (grids, samplers) and the scalar admissions: bindings/msel_candidates_door.mojo.

def _set_params(est: PythonObject, params: PythonObject, context: PythonObject) raises:
    _ = msel_native_set_params(est, params, context)


def _search_host_folds(fields: PythonObject, candidates: PythonObject, b: PythonObject,
                       context: PythonObject) raises -> Bool:
    """True when a search gathers its fold rows on the host: the search's
    own `device` is `'cpu'` in a process with a GPU set, or a candidate sets
    a `device`. `fields` is the search's admitted instance dictionary; its
    `device` reads through the same rule as a member's."""
    if msel_member_host_folds(fields, b, context):
        return True
    for k in range(len(candidates)):  # driver-loop: over the candidates' parameter dicts, names only
        if msel_params_name_device(candidates[k], b):
            return True
    return False


def _door_folds_cv(cv: PythonObject, context: PythonObject, b: PythonObject) raises -> Bool:
    """True for a search cv whose folds THE DOOR draws and hands to the
    search kind as given folds (lane mf-msel5c, as the cv tools hand theirs:
    bindings/msel_cv_door.mojo `msel_cv_by_search`): one of our splitter
    objects (every class `family_msel_splitter_layouts` connects, the group
    splitters with the fit's `groups`) or a user's index list. None and an
    integer are the kind's own unshuffled folds (no splitter object; a
    user's own splitter object is refused by the kind in the door's words,
    never split here: it is a Python callback)."""
    if cv is None or py_type(cv) is b.bool or py_type(cv) is b.int:
        return False
    if py_type(cv) is b.list or py_type(cv) is b.tuple:
        return True
    if msel_is_integral(cv, context):
        return False
    var layouts = py_get(context, "family_msel_splitter_layouts")
    return layouts is not None and py_contains(layouts, py_type(cv))


def _search_settings_check(fields: PythonObject, context: PythonObject, b: PythonObject) raises:
    """scikit-learn's `fit` order (sklearn 1.9 model_selection/_search.py:
    `_get_scorers` and the multi-metric refit check come before the split):
    the search's scorers and refit word, refused in the door's words. Read
    only on the refusal path of a door split (`_search_member_fit`)."""
    var refit = fields["refit"]
    var multi = msel_scorers(fields["scoring"], context)
    if py_type(refit) is not b.bool and py_type(refit) is not b.str:
        raise Error(MSEL_REFIT_CALLBACK_WORDS)
    if (multi is not None and refit is not PythonObject(False)
            and (py_type(refit) is not b.str or not py_contains(multi, refit))):
        raise Error("For multi-metric scoring, refit must be False or a scorer key.")


def _search_member_fit(model: PythonObject, fields: PythonObject, x_in: PythonObject, y_in: PythonObject,
                       groups: PythonObject, context: PythonObject, b: PythonObject,
                       drawn: PythonObject) raises -> Bool:
    """The search fitted as ONE native member of its own META kind: UNPACK
    (X and y read once; `msel_member_refit` admits the search's clone through
    the member boundary, encodes the labels, releases the GIL around the
    member fit = `msel_search_run` and packs the clone), then the clone's
    fitted attributes are the model's. Always True (lane mf-r9a, THE REFUSAL
    RULE of bindings/msel_member_units_door.mojo): a search, base or data the
    kind does not fit is refused natively, before any fit, in the words of
    whoever refuses it; the Bool stays for the callers whose fallback branch
    lane mf-r9b deletes. y None fits an unsupervised base without a target.
    A label table y (rows of Python labels, a text or object 2-D array) is
    handed as given, as the cv tools hand it (bindings/msel_cv_door.mojo
    `msel_cv_by_search`): the kind's UNPACK reads it per output.

    THE DOOR'S FOLDS (lane mf-msel5c): a splitter object or an index list
    (`_door_folds_cv`) is split ONCE here, with the fit's `groups`, by the
    door's own checked split (`msel_cv_folds`; its words, after the search's
    scorer and refit words as scikit-learn orders them); the search's clone
    carries them as the cv tools' carrier does (`_msel_cv_tool` with no
    response, estimators or indices asked) and is fitted in place
    (`msel_member_fit_cloned`: the clone admitted its parameters, so no
    second admission clone; the old route also made one clone). A
    Mojolearn base with no member is refused by the carrier's admission; a
    class that is no Mojolearn estimator by its plan, first. The split (X, y, folds) is appended to `drawn`. An integer
    cv: the kind's own folds; scikit-learn ignores `groups` there."""
    # A base class that is no Mojolearn estimator: the plan's words, before
    # any split reads it.
    _ = msel_estimator_plan(fields["estimator"], context)
    var cv = py_get(fields, "cv")
    var table = y_in is not None and msel_text_rows(y_in, context, b)
    if _door_folds_cv(cv, context, b):
        var got = PythonObject(None)
        try:
            got = msel_cv_folds(fields["estimator"], x_in, y_in, cv, groups, True, context)
        except error:
            _search_settings_check(fields, context, b)
            raise error^
        _ = py_call(drawn.append, got)
        var carrier = msel_native_clone(model, context)
        var carrier_fields = msel_search_fields(carrier, context)
        py_setitem(carrier_fields, "_msel_cv_tool", Python.tuple(PythonObject(False), PythonObject(None),
                                                                 PythonObject(False), got[2],
                                                                 Python.list(PythonObject(None)), PythonObject(0),
                                                                 got[0].shape[0]))
        _ = msel_member_fit_cloned(carrier, y_in if table else got[1], got[0], context, b)
        _search_adopt(fields, msel_search_fields(carrier, context), context)
        return True
    var xa = msel_materialize(x_in, "X", context)
    var ya = PythonObject(None)
    if table:
        ya = y_in
    elif y_in is not None:
        ya = msel_materialize_y(y_in, context)
    var fitted = msel_member_refit(model, Python.dict(), ya, xa, context, b)
    _search_adopt(fields, msel_search_fields(fitted, context), context)
    return True


def _search_adopt(fields: PythonObject, source: PythonObject, context: PythonObject) raises:
    """The fitted attributes of the kind's fitted search (`source`, its
    instance dictionary) written into the caller's search (`fields`)."""
    var updates = Python.dict()
    # The fitted attributes the search RUN's PACK writes (providers/python/
    # member_ensemble_boundary_door.mojo `_search_pack`): the door's own set.
    var names: List[String] = ["cv_results_", "n_splits_", "multimetric_", "best_index_", "best_score_",
                               "best_params_", "best_estimator_", "scorer_"]
    for i in range(len(names)):  # driver-loop: over the fitted attribute names (eight, once per fit)
        var name = names[i]
        var value = py_get(source, name)
        if value is None:
            # refit=False writes no best_estimator_ / scorer_ (and a
            # multi-metric one no best_*; lane mf-search4b); the first three
            # are always written.
            if i < 3:
                raise Error("RuntimeError: search member fit invariant: the search kind's PACK wrote no " + name)
            if name == "scorer_" and py_contains(source, PythonObject(name)):
                py_setitem(updates, name, value)  # scoring=None: the door's scorer_ None
            continue
        py_setitem(updates, name, value)
    msel_search_update(fields, updates, context)


def msel_search_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_BaseSearch.fit(X, y, groups=...)`."""
    var b = msel_b()
    var fields = msel_search_fields(model, context)
    var x_in = args[0]
    var y_in = args[1]
    var groups = args[2]
    var estimator = fields["estimator"]
    var error_score = fields["error_score"]
    var jobs = msel_unit_jobs(fields["n_jobs"], error_score, String(py=py_type(model).__name__), context)
    _ = msel_estimator_plan(estimator, context)
    msel_admit(fields["n_jobs"], context)
    msel_admit(error_score, context)
    # THE ONE SEARCH BODY (lane mf-search4 follow-up): a search its member
    # kind admits (core/ensemble_member.mojo META_SEARCH_*: an int cv or one
    # of our splitters or an index list; scoring None or scorer names
    # core/msel_scorer_run.mojo serves, one or several (lane mf-search4b);
    # refit True, False or a scorer name; its own `device`; every candidate a
    # serving member) is fitted by the typed search RUN
    # (core/msel_search_run.mojo) through `_search_member_fit`; the units
    # below run only what that kind does not take (module note of
    # providers/python/member_ensemble_boundary_door.mojo `_admit_search`).
    # TOMBSTONE: MOJOLEARN_MSEL_MEMBER_UNITS_OFF (removed without measurement: plumbing, rule 2026-11-22) deleted 2026-11-23 by lane/ab-prep-classical; code recoverable at 4d76a9e16e.
    # THE RESIDENT FOLD ROWS (lane mf-search5): where the old units handed a
    # device member DeviceRows (the same rule as below: the member's own
    # device, the search's own `device`, no candidate naming a `device`),
    # the search's own resident scope is opened around the member fit, as
    # `MselFoldRows` opens it: the RUN's units then read every side as the
    # whole X plus the fold's rows, X resident once in that scope and the
    # rows taken on the device by the member's own program (the kernels of
    # the DeviceRows route; only who gathers the rows moved). Read once per
    # search fit; the candidates are drawn once and reused below.
    var candidates = PythonObject(None)
    var scope = PythonObject(None)
    if msel_device_rows_ok(estimator, fields["scoring"], context):
        candidates = msel_candidates(model, context)
        if not _search_host_folds(fields, candidates, b, context):
            scope = msel_rows_scope_open(context)
    var fitted = False
    var drawn = Python.list()
    try:
        fitted = _search_member_fit(model, fields, x_in, y_in, groups, context, b, drawn)
    finally:
        if scope is not None:
            msel_rows_scope_close(scope, context)
    if fitted:
        return model
    if candidates is None:
        candidates = msel_candidates(model, context)
    if len(candidates) == 0:
        raise Error("No fits were performed. Was the CV iterator empty? Were there no candidates?")
    var scoring_in = fields["scoring"]
    var refit = fields["refit"]
    var multi = msel_scorers(scoring_in, context)
    var refit_false = refit is PythonObject(False)
    if py_type(refit) is not b.bool and py_type(refit) is not b.str:
        raise Error(MSEL_REFIT_CALLBACK_WORDS)  # one body with the search kind's admission (lane mf-ens6b)
    if multi is not None and not refit_false and (py_type(refit) is not b.str or not py_contains(multi, refit)):
        raise Error("For multi-metric scoring, refit must be False or a scorer key.")
    var scoring = multi if multi is not None else scoring_in
    var return_train = msel_flag_truth(fields["return_train_score"], context)
    # The door's split, drawn once: the kind's UNPACK may already have drawn it (`drawn`).
    var got = drawn[0] if len(drawn) > 0 else msel_cv_folds(estimator, x_in, y_in, fields["cv"], groups, True, context)
    var xf = got[0]
    var yf = got[1]
    var folds = got[2]
    var per = List[List[List[Float64]]]()
    var trains = List[List[List[Float64]]]()
    var names = List[String]()
    var n_splits = 0
    # The search's own fold rows (device wave, 2026-11-21), for the searches
    # the member kind does not take (above; lane mf-search5: a device member's
    # search the kind takes runs the resident route inside the typed RUN).
    # They stay resident on the device only when the base estimator runs there
    # (msel_device_rows_ok reads the member's device), the search itself
    # does not say device="cpu", and no candidate sets a `device`: a
    # candidate that moves a member to the CPU must not be handed rows that
    # live on the device. Read once per search fit, and only when the rows
    # would be resident.
    var device_rows = msel_device_rows_ok(estimator, scoring, context)
    if device_rows and _search_host_folds(fields, candidates, b, context):
        device_rows = False
    var rows = MselFoldRows(xf, yf, folds, device_rows, context)
    try:
        # Every (candidate, fold) is one unit (msel_cv_units): the base
        # estimator cloned once per unit with the candidate's parameters set
        # on the clone; no candidate clone is made first.
        var run = msel_cv_units(estimator, candidates, yf, rows, scoring, return_train, False, error_score, jobs,
                                context)
        for c in range(len(candidates)):  # driver-loop: over candidates (results only)
            var cvr = run.candidate(c, False, context)
            if c == 0:
                names = cvr.names.copy()
                for _ in range(len(names)):  # driver-loop: over scorer names
                    per.append(List[List[Float64]]())
                    trains.append(List[List[Float64]]())
            for k in range(len(names)):  # driver-loop: over scorer names
                per[k].append(cvr.test[k].copy())
                if return_train:
                    trains[k].append(cvr.train[k].copy())
            n_splits = len(cvr.fit_time)
    finally:
        rows.close(context)
    var results = Python.dict()
    py_setitem(results, "params", candidates)
    var itertools = native_loaded_module("itertools")
    var keys = family_call(b.sorted, py_call(b.set, py_call(itertools.chain.from_iterable, candidates)))
    for i in range(len(keys)):  # driver-loop: over candidate parameter names
        var getter = py_call(native_loaded_module("operator").methodcaller, PythonObject("get"), keys[i])
        py_setitem(results, family_call(msel_op("add"), PythonObject("param_"), family_call(b.str, keys[i])),
                   py_call(b.list, py_call(b.map, getter, candidates)))
    var best_index = -1
    var best_score = 0.0
    var key_index = -1
    if multi is None:
        key_index = 0
    elif family_is_str(refit):
        for k in range(len(names)):  # driver-loop: over scorer names
            if names[k] == String(py=refit):
                key_index = k
    for k in range(len(names)):  # driver-loop: over scorer names
        var suffix = String("") if multi is None else "_" + names[k]
        for i in range(n_splits):  # driver-loop: over folds
            py_setitem(results, PythonObject("split" + String(i) + "_test_score" + suffix),
                       msel_f64_list(msel_score_column(per[k], i), context))
        var means = msel_row_means(per[k])
        var stds = msel_row_stds(per[k], means)
        var ranks = msel_rank_scores(means)
        py_setitem(results, PythonObject("mean_test_score" + suffix), msel_f64_list(means, context))
        py_setitem(results, PythonObject("std_test_score" + suffix), msel_f64_list(stds, context))
        py_setitem(results, PythonObject("rank_test_score" + suffix), msel_i32_list(ranks, context))
        if return_train:
            # scikit-learn's split<i>_train_score columns, as the kind's PACK writes them (lane mf-msel5)
            for i in range(n_splits):  # driver-loop: over folds
                py_setitem(results, PythonObject("split" + String(i) + "_train_score" + suffix),
                           msel_f64_list(msel_score_column(trains[k], i), context))
            py_setitem(results, PythonObject("mean_train_score" + suffix), msel_f64_list(msel_row_means(trains[k]), context))
        if k == key_index:
            best_index = msel_rank_best(ranks)
            best_score = means[best_index]
    var updates = Python.dict()
    py_setitem(updates, "cv_results_", results)
    py_setitem(updates, "n_splits_", PythonObject(n_splits))
    py_setitem(updates, "multimetric_", PythonObject(multi is not None))
    var best_params = PythonObject(None)
    if key_index >= 0:
        py_setitem(updates, "best_index_", PythonObject(best_index))
        py_setitem(updates, "best_score_", PythonObject(best_score))
        best_params = candidates[best_index]
        py_setitem(updates, "best_params_", best_params)
    if not refit_false:
        var xa = msel_materialize(x_in, "X", context)
        # A member whose family published its member entry is refitted over
        # it (bindings/msel_member_units_door.mojo `msel_member_refit`: typed
        # fit on a view of X with the GIL released, then the family's pack; the
        # best estimator HOLDS its fitted member state). None: any other
        # member, data or failing fit takes the estimator's own fit door below,
        # as before (the fallback rule; a failing fit raises its own words).
        var best = PythonObject(None)
        # TOMBSTONE: MOJOLEARN_MSEL_MEMBER_UNITS_OFF (removed without measurement: plumbing, rule 2026-11-22) deleted 2026-11-23 by lane/ab-prep-classical; code recoverable at 4d76a9e16e.
        if best_params is not None and yf is not None:
            best = msel_member_refit(estimator, best_params, yf, xa, context, b)
        if best is None:
            best = msel_clone(estimator, False, context)
            _set_params(best, best_params, context)
            msel_fit(best, xa, y_in, context)
        py_setitem(updates, "best_estimator_", best)
        py_setitem(updates, "scorer_", msel_get_scorer(scoring_in, context) if multi is None else multi)
    msel_search_update(fields, updates, context)
    return model


# ---------------------------------------------------------------- validation_curve


def _curve_by_search(estimator: PythonObject, args: PythonObject, param_range: PythonObject, context: PythonObject,
                     b: PythonObject) raises -> PythonObject:
    """UNPACK of a validation curve as its grid search (lane mf-cluster5):
    a GridSearchCV over {param_name: param_range} with the curve's cv,
    scoring, n_jobs and error_score, refit=False and train scores, fitted by
    the search kind's typed RUN; PACK: its split<i>_train_score /
    split<i>_test_score columns as the curve's (values x folds) blocks.
    Never None (lane mf-r9b): what the kind does not fit is refused by
    `_search_member_fit` in its words; y None fits an unsupervised base."""
    var grid = Python.dict()
    py_setitem(grid, args[3], py_call(b.list, param_range))
    var keywords = Python.dict()
    py_setitem(keywords, "scoring", args[7])
    py_setitem(keywords, "cv", args[6])
    py_setitem(keywords, "n_jobs", args[8])
    py_setitem(keywords, "error_score", args[9])
    py_setitem(keywords, "refit", PythonObject(False))
    py_setitem(keywords, "return_train_score", PythonObject(True))
    py_setitem(keywords, "estimator", estimator)
    py_setitem(keywords, "param_grid", grid)
    var search = msel_carrier_search(msel_cls("GridSearchCV", context), keywords, context)
    var fields = msel_search_fields(search, context)
    # THE RESIDENT FOLD ROWS (lane mf-r9b), as `msel_search_fit` opens them
    # for a device member (the units' DeviceRows rule): none when the curve
    # moves `device` (one of its values is the CPU) or the base says
    # device="cpu". Read once per curve.
    var scope = PythonObject(None)
    if msel_device_rows_ok(estimator, args[7], context) and not msel_member_host_folds(fields, b, context):
        var name = String(py=args[3])
        if name != "device" and not name.endswith("__device"):
            scope = msel_rows_scope_open(context)
    try:
        # raises every refusal in its words (THE REFUSAL RULE)
        _ = _search_member_fit(search, fields, args[1], args[2], args[5], context, b, Python.list())
    finally:
        if scope is not None:
            msel_rows_scope_close(scope, context)
    var fitted = msel_search_fields(search, context)
    var results = py_get(fitted, "cv_results_")
    var splits = py_get(fitted, "n_splits_")
    if results is None or splits is None:
        raise Error("RuntimeError: validation_curve invariant: the search kind's PACK wrote no cv_results_ or"
                    + " n_splits_")
    var k = Int(py=splits)
    var count = len(param_range)
    var tr = List[Float64](capacity=count * k)
    var te = List[Float64](capacity=count * k)
    var train_cols = List[List[Float64]]()
    var test_cols = List[List[Float64]]()
    for i in range(k):  # driver-loop: over the folds (one cv_results_ column each)
        train_cols.append(_f64_values(results["split" + String(i) + "_train_score"]))
        test_cols.append(_f64_values(results["split" + String(i) + "_test_score"]))
    for v in range(count):  # driver-loop: over the param_range values (their rows of the blocks)
        for i in range(k):  # small-loop(one value's folds)
            tr.append(train_cols[i][v])
            te.append(test_cols[i][v])
    return Python.tuple(msel_f64_block(tr, count, k, context), msel_f64_block(te, count, k, context))


def _f64_values(column: PythonObject) raises -> List[Float64]:
    """PACK: a cv_results_ float64 column (the search PACK's own list)."""
    var out = List[Float64](capacity=len(column))
    for i in range(len(column)):  # driver-loop: over the candidates of one column
        out.append(Float64(py=column[i]))
    return out^


def msel_validation_curve(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`validation_curve(estimator, X, y, param_name, param_range, groups,
    cv, scoring, n_jobs, error_score)`."""
    var b = msel_b()
    var estimator = args[0]
    if py_type(args[3]) is not b.str:
        raise Error("TypeError: param_name must be an exact string")
    var param_range = msel_parameter_values(args[4], context)
    # ONE SCORER (lane mf-r9b): validate_params' scoring constraint, in
    # signature order (after param_name and param_range, before n_jobs and
    # error_score); several scorers were the units' KeyError 'train_score'.
    _one_scorer_check("validation_curve", args[7], context)
    _ = msel_unit_jobs(args[8], args[9], "validation_curve", context)  # n_jobs and error_score words
    if len(param_range) == 0:
        # scikit-learn 1.9 validation_curve (model_selection/_validation.py
        # 2484-2510): the split runs, no fit, then `_aggregate_score_dicts([])`
        # reads `scores[0]`. The door's split checks first, then those words.
        _ = msel_cv_folds(estimator, args[1], args[2], args[6], args[5], True, context)
        raise Error("IndexError: list index out of range")
    # THE TYPED SEARCH RUN (lane mf-cluster5): a validation curve IS the grid
    # search over {param_name: param_range} with return_train_score=True and
    # refit=False, so the search kind's typed RUN (core/msel_search_run.mojo,
    # through `_search_member_fit`) fits and scores it (every value a serving
    # member; a scorer it serves; lane mf-msel5c: a splitter with the curve's
    # groups split by the door, as a search's); its split scores are the
    # curve's. Lane mf-r9b: the only route, y None included (an unsupervised
    # base); the kind refuses what it does not fit in its words. Its
    # resident fold rows follow the units' device rule (`_curve_by_search`).
    return _curve_by_search(estimator, args, param_range, context, b)


# ---------------------------------------------------------------- learning_curve


def _invalid_parameter_class(context: PythonObject) raises -> PythonObject:
    """scikit-learn's `InvalidParameterError` (sklearn/utils/_param_validation.py:
    a ValueError and a TypeError) when sklearn is already loaded (no import
    here), else one native class with the same name and both bases, made
    once and kept in the context (lane mf-tail5b)."""
    var b = msel_b()
    var module = py_call(native_loaded_module("sys").modules.get, PythonObject("sklearn.utils._param_validation"))
    if module is not None:
        var cls = py_call(b.getattr, module, PythonObject("InvalidParameterError"), PythonObject(None))
        if cls is not None:
            return cls
    var kept = py_call(context.get, PythonObject("_msel_invalid_parameter_error"), PythonObject(None))
    if kept is not None:
        return kept
    var body = Python.dict()
    py_setitem(body, PythonObject("__module__"), PythonObject("mojolearn.model_selection"))
    var made = py_call(b.type, PythonObject("InvalidParameterError"), Python.tuple(b.ValueError, b.TypeError), body)
    py_setitem(context, PythonObject("_msel_invalid_parameter_error"), made)
    return made


def _one_scorer_check(caller: String, scoring_in: PythonObject, context: PythonObject) raises:
    """SEVERAL SCORERS (lane mf-tail5; lane mf-r9b: validation_curve too):
    scikit-learn 1.9's `validate_params` of learning_curve and
    validation_curve allows `scoring` only as a scorer name, a callable or
    None ([StrOptions(set(get_scorer_names())), callable, None],
    model_selection/_validation.py 1763 and 2274): a list, tuple, set or
    dict is its InvalidParameterError, raised before any fold or fit. Lane
    mf-tail5b: its CLASS (`InvalidParameterError`, a ValueError and a
    TypeError: scikit-learn's own when sklearn is loaded, else
    `_invalid_parameter_class`'s) and its words as `StrOptions.__str__`
    writes them: the names' reprs in the order of the Python set it holds."""
    var b = msel_b()
    var many = py_type(scoring_in)
    if many is b.list or many is b.tuple or many is b.set or many is b.dict:
        var names = py_call(b.list, py_call(b.set, py_call(b.list, context["family_msel_scorers"])))
        var shown = String("{")
        for i in range(len(names)):  # driver-loop: over the scorer names (the refusal's words, once)
            if i > 0:
                shown += ", "
            shown += family_repr(names[i])
        mx_raise_obj(_invalid_parameter_class(context),
                     PythonObject("The 'scoring' parameter of " + caller + " must be a str among " + shown
                                  + "}, a callable or None. Got " + family_repr(scoring_in) + " instead."))


def msel_learning_curve(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`learning_curve(estimator, X, y, groups, train_sizes, cv, scoring,
    n_jobs, shuffle, random_state, error_score, return_times)`."""
    var b = msel_b()
    var estimator = args[0]
    # One scorer (`_one_scorer_check`), in validate_params' signature order:
    # scoring comes before n_jobs, shuffle, random_state and error_score.
    _one_scorer_check("learning_curve", args[6], context)
    _ = msel_unit_jobs(args[7], args[10], "learning_curve", context)  # n_jobs and error_score words
    var shuffle = msel_flag_truth(args[8], context)
    var seed = _search_seed(args[9], context)
    var return_times = msel_flag_truth(args[11], context)
    var got = msel_cv_folds(estimator, args[1], args[2], args[5], args[3], False, context)
    var x = got[0]
    var y = got[1]
    var folds = got[2]
    var nf = len(folds)
    var n_max = Int(py=folds[0][0].size)
    for f in range(nf):  # driver-loop: over folds
        n_max = min(n_max, Int(py=folds[f][0].size))
    var chosen = Python.list()
    var requested = msel_parameter_values(args[4], context)
    var math = native_loaded_module("math")
    for i in range(len(requested)):  # driver-loop: over the requested train sizes
        var s = _search_scalar(requested[i], "train_sizes", context)
        var a = 0
        if py_type(s) is b.float:
            a = msel_int(family_call(math.floor, family_call(msel_op("mul"), s, PythonObject(n_max))))
        else:
            a = msel_int(s)
        if not (0 < a and a <= n_max):
            raise Error("train_sizes has been interpreted as absolute numbers of training samples and "
                        + "must be within (0, " + String(n_max) + "], but is within [" + String(a) + ", " + String(a) + "].")
        _ = py_call(chosen.append, PythonObject(a))
    var sizes_py = family_call(b.sorted, py_call(b.set, chosen))
    var sizes = List[Int]()
    for i in range(len(sizes_py)):  # driver-loop: over train sizes
        sizes.append(Int(py=sizes_py[i]))
    var orders = Python.list()
    if shuffle:
        var rng = MselRng(seed, context)
        var lengths = List[Int]()
        for f in range(nf):  # driver-loop: over folds
            lengths.append(Int(py=folds[f][0].size))
        var perms = rng.permutation_rows(lengths, context)
        var key = msel_helpers_key(context)
        for f in range(nf):  # driver-loop: over folds (one device gather each)
            var tr = msel_index_i64(folds[f][0], context)
            var size = lengths[f]
            if size > 0:
                var out = msel_new(size, "<i8", context)
                _ = msel_gather_i64(key, msel_addr(tr), size, msel_addr(perms[f]), size, msel_addr(out))
                _ = tr  # keep the Python object alive through the native call above
                _ = out  # keep the Python object alive through the native call above
                tr = out
            _ = py_call(orders.append, tr)
        # Each gather reads a raw address into a permutation owned by this
        # list. Keep every permutation alive through the final fold's call.
        _ = perms^
    else:
        for f in range(nf):  # driver-loop: over folds
            _ = py_call(orders.append, msel_index_i64(folds[f][0], context))
    var scoring = args[6]
    var error_score = args[10]
    # Lane p3-msel-parallel-and-process-pool: every (train size, fold) is a
    # unit of msel_cv_units, size-major, so the results keep the old order.
    # The fold rows stay on the host gather (device_rows False), as before:
    # each unit's fit sees the same arrays, so no bits change. REMOVED: the
    # serial size x fold loop and its own resident plan; the test rows are
    # now gathered per unit (they were gathered once per fold for every
    # size): the price of independent units, RUN OWED in the A/B.
    var unit_folds = Python.list()
    for si in range(len(sizes)):  # driver-loop: over train sizes
        for f in range(nf):  # driver-loop: over folds
            _ = py_call(unit_folds.append, Python.tuple(msel_slice_i64(orders[f], 0, sizes[si], context), folds[f][1]))
    var tr_s = List[Float64]()
    var te_s = List[Float64]()
    var ft = List[Float64]()
    var st = List[Float64]()
    # THE ONE TYPED SEARCH RUN (lane mf-msel5, bindings/msel_cv_door.mojo
    # `msel_cv_by_search`): the size-major units as one candidate's given
    # folds (each size's train rows the fold's first `size` of its order),
    # one gathered test block per fold (the RUN's `_same_rows`, as
    # `reuse_test` below), the units' fit and test-score seconds. One
    # scorer only (several: the units' TypeError below).
    var by = PythonObject(None)
    if msel_scorers(scoring, context) is None:
        by = msel_cv_by_search(estimator, x, y, unit_folds, scoring, args[7], error_score, True, False,
                               PythonObject(None), False, context)
    if by is not None:
        var te_by = by["test"][0]
        var tr_by = by["train"][0]
        var ft_by = by["fit_time"]
        var st_by = by["score_time"]
        for u in range(len(unit_folds)):  # driver-loop: over size x fold units (results only)
            tr_s.append(Float64(py=tr_by[u]))
            te_s.append(Float64(py=te_by[u]))
            ft.append(Float64(py=ft_by[u]))
            st.append(Float64(py=st_by[u]))
    else:
        var rows = MselFoldRows(x, y, unit_folds, False, context)
        # Every size of a fold scores on the same test rows; they are gathered
        # once per fold (bindings/msel_cv_door.mojo `_take_reusing_test`, the
        # default since 2026-11-20; its _OFF old path was deleted 2026-11-23).
        rows.reuse_test = True
        try:
            var r = msel_cross_validate_run(estimator, y, rows, scoring, True, False, False, error_score, context, jobs)
            if len(r.names) != 1:
                raise Error("TypeError: learning_curve takes one scorer")
            for u in range(len(r.fit_time)):  # driver-loop: over size x fold units
                tr_s.append(r.train[0][u])
                te_s.append(r.test[0][u])
                ft.append(r.fit_time[u])
                st.append(r.score_time[u])
        finally:
            rows.close(context)
    var out = Python.list()
    _ = py_call(out.append, msel_i64_list(sizes, context))
    _ = py_call(out.append, msel_f64_block(tr_s, len(sizes), nf, context))
    _ = py_call(out.append, msel_f64_block(te_s, len(sizes), nf, context))
    if return_times:
        _ = py_call(out.append, msel_f64_block(ft, len(sizes), nf, context))
        _ = py_call(out.append, msel_f64_block(st, len(sizes), nf, context))
    return py_call(b.tuple, out)


# ---------------------------------------------------------------- permutation_test_score


def _y_free(splitter: PythonObject, context: PythonObject) raises -> Bool:
    """`_y_free(splitter)`: its split cannot read y and draws the same folds
    on every call."""
    var b = msel_b()
    var t = py_type(splitter)
    var fixed: List[String] = ["LeaveOneGroupOut", "LeavePGroupsOut", "TimeSeriesSplit", "LeaveOneOut", "LeavePOut",
                                "PredefinedSplit", "_IterableCV"]
    for i in range(len(fixed)):  # driver-loop: over the y-free splitter classes
        if t is msel_cls(fixed[i], context):
            return True
    if t is msel_cls("KFold", context) or t is msel_cls("GroupKFold", context):
        var seeded = msel_is_integral(msel_field(splitter, "random_state", context), context)
        return not msel_flag_truth(msel_field(splitter, "shuffle", context), context) or seeded
    if t is msel_cls("ShuffleSplit", context) or t is msel_cls("GroupShuffleSplit", context):
        return msel_is_integral(msel_field(splitter, "random_state", context), context)
    return False


def _perm_base(y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_NativePermutation._base(y)`: y as the definition's first
    `mean_score` array (int32 or float32), or None."""
    var b = msel_b()
    var array_mod = native_loaded_module("array")
    msel_admit(y, context)
    if py_type(y) is b.list or py_type(y) is b.tuple:
        var kinds = py_call(b.set, py_call(b.map, b.type, y))
        var code = String("")
        var dtype = String("")
        var ints = py_call(b.set, Python.tuple(b.int, b.bool))
        if len(kinds) > 0 and msel_truth(py_call(kinds.issubset, ints)):
            code = "i"
            dtype = "<i4"
        elif len(kinds) == 1 and py_contains(kinds, b.float):
            code = "f"
            dtype = "<f4"
        else:
            return PythonObject(None)
        var store = msel_try(array_mod.array, Python.tuple(PythonObject(code), y), b.OverflowError, context)
        if msel_missed(store, context):
            return PythonObject(None)
        return classical_wrap_array(store, Python.tuple(len(store)), dtype, "C", context)
    var a = msel_try_materialize(y, "y", Python.tuple(b.TypeError, b.ValueError), context)
    if msel_missed(a, context):
        return PythonObject(None)
    if Int(py=a.ndim) != 1 or Int(py=a.size) < 1:
        return PythonObject(None)
    var d = String(py=a.dtype)
    if d == "<f2" or d == "<f4" or d == "<f8":
        return admit_family(a, 1, "y", context).array
    if d == "<i4":
        return a
    if d == "<i8" or d == "<u1" or d == "<u4":
        var narrow = msel_new(Int(py=a.size), "<i4", context)
        # The typed parallel conversion (core/admitted_convert.mojo), not a
        # py_call into the base binding's cast_elements; a label outside
        # int32 is the conversion's refusal, which keeps the labels as they are.
        try:
            if not admitted_convert_vector(msel_addr(a), d, msel_addr(narrow), "<i4", Int(py=a.size)):
                return PythonObject(None)
        except:
            return PythonObject(None)
        _ = a  # keep the Python object alive through the native calls above
        _ = narrow  # keep the Python object alive through the native calls above
        return narrow
    if d == "<i2" or d == "<u2" or d == "<i1":
        var store = family_call(array_mod.array, PythonObject("i"), a._mv)
        return classical_wrap_array(store, Python.tuple(len(store)), "<i4", "C", context)
    return PythonObject(None)


def _mean_test(estimator: PythonObject, x: PythonObject, y: PythonObject, folds: PythonObject, scoring: PythonObject,
               mut rows: MselFoldRows, jobs: Int, context: PythonObject) raises -> Float64:
    """`_row_means([cross_validate(...)['test_score']])[0]` on `rows`, the
    folds as units on `jobs` threads.

    THE ONE TYPED SEARCH RUN (lane mf-msel5c): the folds are the carrier
    search's given folds (`msel_cv_by_search`, error_score nan as the door's
    units score them), fitted and scored by the search kind's RUN; the
    door's units only for what that kind does not take (resident fold rows
    included, lane mf-tail5: the carrier reads their scope)."""
    var one = List[List[Float64]]()
    # (lane mf-tail5) Resident fold rows too (`rows.dev`): `rows` opened the
    # same resident scope the carrier opens (`msel_rows_scope_open`), and the
    # carrier's RUN reads it (its resident route: the whole X plus each
    # side's rows, X pinned once in that scope for every permutation), so
    # the permutations' folds run in the typed RUN; the units only for what
    # the kind does not take.
    var by = msel_cv_by_search(estimator, x, y, folds, scoring, PythonObject(jobs), PythonObject(_nan()), False,
                               False, PythonObject(None), False, context)
    if by is not None:
        var te = by["test"][0]
        var values = List[Float64](capacity=len(te))
        for i in range(len(te)):  # driver-loop: over the folds (their test scores, once per call)
            values.append(Float64(py=te[i]))
        one.append(values^)
        return msel_row_means(one)[0]
    var r = msel_cross_validate_run(estimator, y, rows, scoring, False, False, False, PythonObject(_nan()),
                                    context, jobs)
    one.append(r.test[0].copy())
    return msel_row_means(one)[0]


def _nan() -> Float64:
    return Float64(0.0) / Float64(0.0)


def _mean_own(estimator: PythonObject, x: PythonObject, y: PythonObject, folds: PythonObject, scoring: PythonObject,
              device: Bool, jobs: Int, context: PythonObject) raises -> Float64:
    """`_mean_test` on rows of its own (the folds drawn for this y)."""
    var rows = MselFoldRows(x, y, folds, device, context)
    var value: Float64 = 0.0
    try:
        value = _mean_test(estimator, x, y, folds, scoring, rows, jobs, context)
    finally:
        rows.close(context)
    return value


def _group_order(groups: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_NativePermutation._group_order`: (sizes, group-sorted rows,
    inverse) of the groups in sorted-group order, in one program."""
    var gc = MselGroups(groups, n, "groups", context)
    var m = gc.m
    var prog = MetricDoorProg()
    var key = prog.put_i32(gc.codes)
    var off_slot = prog.alloc(m + 1)
    var off = prog.want(off_slot, m + 1)
    var ordr = prog.alloc(n)
    prog.stage("group_sort", 1, [key, n, m, off, ordr])
    var off2 = prog.alloc(n + 1)
    var inv = prog.alloc(n)
    prog.stage("group_sort", 1, [ordr, n, n, off2, inv])
    var w_ord_slot = prog.alloc(2 * n)
    var w_ord = prog.want(w_ord_slot, 2 * n)
    var w_inv_slot = prog.alloc(2 * n)
    var w_inv = prog.want(w_inv_slot, 2 * n)
    prog.stage("rows64", n, [ordr, w_ord])
    prog.stage("rows64", n, [inv, w_inv])
    # N056: the int32 words of the group order and its inverse, read back
    # once per call for the per-permutation program (`_group_index`).
    var w_ord32 = prog.want(ordr, n)
    var w_inv32 = prog.want(inv, n)
    var arena = metric_door_run(prog, PythonObject(None), context)
    var base = arena.base()
    var offsets = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=base + 4 * off)
    var sizes = Python.list()
    for g in range(m):  # driver-loop: over groups
        _ = py_call(sizes.append, PythonObject(Int(offsets[g + 1]) - Int(offsets[g])))
    var rows_sorted = msel_new(n, "<i8", context)
    var inverse = msel_new(n, "<i8", context)
    unsafe_memcpy(dest=msel_p64(rows_sorted), src=FAMILY_I64P(unsafe_from_address=base + 4 * w_ord), count=n)
    _ = rows_sorted  # keep the Python object alive through the native call above
    unsafe_memcpy(dest=msel_p64(inverse), src=FAMILY_I64P(unsafe_from_address=base + 4 * w_inv), count=n)
    _ = inverse  # keep the Python object alive through the native call above
    var ord32 = msel_new(n, "<i4", context)
    var inv32 = msel_new(n, "<i4", context)
    var off32 = msel_new(m + 1, "<i4", context)
    unsafe_memcpy(dest=FAMILY_I32P(unsafe_from_address=msel_addr(ord32)), src=FAMILY_I32P(unsafe_from_address=base + 4 * w_ord32), count=n)
    _ = ord32  # keep the Python object alive through the native copy above
    unsafe_memcpy(dest=FAMILY_I32P(unsafe_from_address=msel_addr(inv32)), src=FAMILY_I32P(unsafe_from_address=base + 4 * w_inv32), count=n)
    _ = inv32  # keep the Python object alive through the native copy above
    unsafe_memcpy(dest=FAMILY_I32P(unsafe_from_address=msel_addr(off32)), src=FAMILY_I32P(unsafe_from_address=base + 4 * off), count=m + 1)
    _ = off32  # keep the Python object alive through the native copy above
    # Both raw source addresses still point into this arena after the final
    # offset read. Allocating the result Arrays must not release its storage.
    _ = arena^
    _ = prog^
    # (sizes, rows_sorted, inverse) as before, then the int32 order, inverse,
    # group offsets and row group codes the one-program `_group_index` reads.
    return Python.tuple(sizes, rows_sorted, inverse, ord32, inv32, off32, gc.codes)


def _group_sizes(order: PythonObject) raises -> List[Int]:
    """The group sizes of `_group_order`, read once per call."""
    var sizes_py = order[0]
    var sizes = List[Int]()
    for g in range(len(sizes_py)):  # driver-loop: over groups
        sizes.append(Int(py=sizes_py[g]))
    return sizes^


def _group_index(mut rng: MselRng, order: PythonObject, sizes: List[Int], n: Int,
                 context: PythonObject) raises -> PythonObject:
    """`_NativePermutation._group_index`: each group's rows permuted by its
    draw over the group-sorted rows, put back in row order. `sizes` is
    `_group_sizes(order)`, read once per call, not once per permutation.

    N056 (lane ne2-b-search-workers): ONE program per permutation. The
    groups' draws are `permute` stages at each group's offset (the draws in
    group order, one per group, as `permutation_rows` made them), then
    `group_take` permutes each group's rows by its draw and `take_rows64`
    puts them back in row order: the words the per-group `gather_i64` calls
    and the final inverse gather gave, read back once. Same draws, same
    rows, same order: no bit changes."""
    var prog = MetricDoorProg()
    var key = prog.put_i32(order[6])
    var ordr = prog.put_i32(order[3])
    var off = prog.put_i32(order[5])
    var inv = prog.put_i32(order[4])
    var draws = prog.alloc(max(n, 1))
    var at = 0
    for g in range(len(sizes)):  # driver-loop: over groups (one permute stage each, all in one program)
        var s = rng.salt()
        prog.stage("permute", 1, [sizes[g], draws + at, msel_signed32(s), msel_signed32(s >> 32)])
        at += sizes[g]
    var src = prog.alloc(max(n, 1))
    var out_slot = prog.alloc(2 * n)
    var out = prog.want(out_slot, 2 * n)
    if n > 0:
        prog.stage("group_take", n, [key, ordr, off, draws, src])
        prog.stage("take_rows64", n, [src, inv, out])
    var arena = metric_door_run(prog, PythonObject(None), context)
    var idx = msel_new(n, "<i8", context)
    if n > 0:
        unsafe_memcpy(dest=msel_p64(idx), src=FAMILY_I64P(unsafe_from_address=arena.base() + 4 * out), count=n)
    _ = idx  # keep the Python object alive through the native copy above
    # The copied address points into this arena: keep it through the copy.
    _ = arena^
    _ = prog^
    return idx


def msel_permutation_test_score(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`permutation_test_score(estimator, X, y, groups, cv, n_permutations,
    n_jobs, random_state, verbose, scoring, fit_params, params)`."""
    var b = msel_b()
    var estimator = args[0]
    var y = args[2]
    var groups = args[3]
    var cv = args[4]
    var jobs = msel_unit_jobs(args[6], PythonObject(_nan()), "permutation_test_score", context)
    var n_permutations = msel_int(_search_scalar(args[5], "n_permutations", context))
    var random_state = _search_seed(args[7], context)
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var base = Optional[PythonObject]()
    if not msel_sabotage():
        # admission of the helper table the permutation gathers use
        _ = msel_helpers_key(context)
        base = py_present(_perm_base(y, context))
        if Bool(base) and Int(py=base.value().size) < 2:
            base = Optional[PythonObject]()
    if not base:
        return _permutation_definition(args, n_permutations, random_state, jobs, context)
    var x = msel_materialize(args[1], "X", context)
    var sc = msel_get_scorer(args[9], context)
    var n = Int(py=base.value().size)
    var reuse = not (cv is None or msel_is_integral(cv, context)) and _y_free(cv, context)
    var folds0 = msel_cv_folds(estimator, x, base.value(), cv, groups, False, context)[2]
    var device = msel_device_rows_ok(estimator, sc, context)
    var rows0 = MselFoldRows(x, PythonObject(None), folds0, device, context)
    var score: Float64 = 0.0
    var perm_scores = List[Float64]()
    try:
        score = _mean_test(estimator, x, base.value(), folds0, sc, rows0, jobs, context)
        var rng = MselRng(random_state, context)
        var order = PythonObject(None) if groups is None else _group_order(groups, n, context)
        var group_sizes = List[Int]() if groups is None else _group_sizes(order)
        # Redrawn folds (lane ne2-b-members, N055): the splitter and the
        # groups are resolved once per call and every permutation's folds
        # split the X materialized above (`msel_cv_folds_admitted`); before,
        # each permutation's msel_cv_folds copied X again, re-read the groups
        # and resolved the splitter again. check_cv reads y for its target
        # type only, which a permutation keeps. Same folds.
        var splitter = PythonObject(None)
        var group_values = PythonObject(None)
        if not reuse:
            group_values = PythonObject(None) if groups is None else msel_materialize_y(groups, context)
            splitter = msel_cv_splitter(estimator, base.value(), cv, False, context)
        for _ in range(n_permutations):  # driver-loop: over permutations
            var idx = PythonObject(None)
            if order is None:
                idx = rng.permutation_rows([n], context)[0]
            else:
                idx = _group_index(rng, order, group_sizes, n, context)
            # the permuted y: one device gather out of the resident base y
            var yp = rows0.resident_take(base.value(), idx, context)
            rows0.resident_release(idx, context)
            if reuse:
                perm_scores.append(_mean_test(estimator, x, yp, folds0, sc, rows0, jobs, context))
            else:
                var folds = msel_cv_folds_admitted(splitter, x, yp, group_values, context)
                # pass 3 (rank 107): the permutation's folds run on rows0 (X
                # stored once for the whole call), as the reused folds do;
                # only the fold index sets change. `_mean_own` built a new
                # MselFoldRows, so X was copied and stored again and freed
                # for every permutation. The permutation's index entries are
                # released after its scores (release ignores an index the
                # store never held).
                rows0.folds = folds
                try:
                    perm_scores.append(_mean_test(estimator, x, yp, folds, sc, rows0, jobs, context))
                finally:
                    rows0.folds = folds0
                    for f in range(len(folds)):  # driver-loop: over this permutation's folds
                        rows0.resident_release(folds[f][0], context)
                        rows0.resident_release(folds[f][1], context)
    finally:
        rows0.close(context)
    return Python.tuple(PythonObject(score), msel_f64_list(perm_scores, context),
                        PythonObject(msel_pvalue(perm_scores, score, n_permutations)))


def _permutation_definition(args: PythonObject, n_permutations: Int, random_state: PythonObject,
                            jobs: Int, context: PythonObject) raises -> PythonObject:
    """`permutation_test_score`'s definition route (label objects the
    native permutation cannot hold): y rebuilt from a list per permutation,
    the permutations by Python's own list operations."""
    var b = msel_b()
    var itertools = native_loaded_module("itertools")
    var operator = native_loaded_module("operator")
    var estimator = args[0]
    var groups = args[3]
    var cv = args[4]
    var xa = msel_materialize(args[1], "X", context)
    var yl = msel_flatten(args[2], context)
    var sc = msel_get_scorer(args[9], context)
    var score = _definition_mean(estimator, xa, yl, groups, cv, sc, jobs, context)
    var rng = MselRng(random_state, context)
    var perm_scores = List[Float64]()
    # The groups, their sorted values and each group's rows depend on the
    # groups alone: flattened and selected once per call, not once per
    # permutation (lane perf-sweep-msel-nbr-svc). The draws are unchanged.
    var group_rows = Python.list()
    if groups is not None:
        var g = msel_flatten(groups, context)
        var values = family_call(b.sorted, family_call(b.set, g))
        for gi in range(len(values)):  # driver-loop: over groups
            _ = py_call(group_rows.append,
                        py_call(b.list, py_call(itertools.compress, py_call(b.range, PythonObject(len(g))),
                                                py_call(b.map, operator.eq, g, py_call(itertools.repeat, values[gi])))))
    for _ in range(n_permutations):  # driver-loop: over permutations
        var yp = PythonObject(None)
        if groups is None:
            var perm = rng.permutation(len(yl), context)
            yp = py_call(b.list, py_call(b.map, py_call(b.getattr, yl, PythonObject("__getitem__")), perm))
        else:
            yp = py_call(b.list, yl)
            for gi in range(len(group_rows)):  # driver-loop: over groups
                var rows = group_rows[gi]
                var perm = rng.permutation(len(rows), context)
                var src = py_call(b.map, py_call(b.getattr, yl, PythonObject("__getitem__")),
                                  py_call(b.map, py_call(b.getattr, rows, PythonObject("__getitem__")), perm))
                _ = py_call(b.list, py_call(b.map, py_call(b.getattr, yp, PythonObject("__setitem__")), rows, src))
        perm_scores.append(_definition_mean(estimator, xa, yp, groups, cv, sc, jobs, context))
    return Python.tuple(PythonObject(score), msel_f64_list(perm_scores, context),
                        PythonObject(msel_pvalue(perm_scores, score, n_permutations)))


def _definition_mean(estimator: PythonObject, xa: PythonObject, yv: PythonObject, groups: PythonObject, cv: PythonObject,
                     sc: PythonObject, jobs: Int, context: PythonObject) raises -> Float64:
    """The definition's `mean_score(yv)`: y as a float32 / int32 Array by
    `Array.from_list`, then one cross_validate."""
    var b = msel_b()
    var first = _search_scalar(yv[0], "y", context)
    var dtype = String("<f4") if py_type(first) is b.float else String("<i4")
    var yarr = _cast(msel_materialize(yv, "y", context), dtype, context["family_msel_resident_context"])
    var folds = msel_cv_folds(estimator, xa, yarr, cv, groups, False, context)[2]
    return _mean_own(estimator, xa, yarr, folds, sc, msel_device_rows_ok(estimator, sc, context), jobs, context)
