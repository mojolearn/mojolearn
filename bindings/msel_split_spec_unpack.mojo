# SPDX-License-Identifier: Apache-2.0
"""UNPACK of THE SPLITTER SPEC (lane mf-split5, 2026-11-23): one of OUR
splitters, given as a meta-estimator's or an x_linear CV member's `cv`, read
ONCE into typed words (bindings/msel_split_spec_run.mojo `MselSplitSpec`),
exactly as the public split door reads it for `split`
(bindings/msel_split_door.mojo `msel_split`: `MselSplitter`, its stored
fields through `field`, the seed by `msel_seed`, the sizes by
`msel_size_word` / `msel_size_sum`, PredefinedSplit's test_fold by
`msel_label_codes`, the entries by `msel_split_route`, the sabotage word).
Each fit then computes its folds natively from these words
(`msel_spec_sides`). A user's own splitter object never reaches here: the
callers refuse it first, before it is invoked (AGENTS.md "No calls back into
Python").
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_type, py_contains
from bindings.msel_common_door import msel_int, msel_cmp, msel_b
from bindings.msel_split_door import (
    msel_cls, MselSplitter, msel_seed, msel_truth, msel_sabotage, msel_check_splits, SPLITTER_OTHER,
    SPLITTER_ITERABLE,
)
from bindings.msel_split_unpack import msel_split_route, MselRoutePin, msel_label_codes, msel_size_word, msel_size_sum
from bindings.msel_split_plan_run import msel_repeated_check
from bindings.msel_split_spec_run import (
    MselSplitSpec, MSEL_SPEC_KFOLD, MSEL_SPEC_STRATIFIED_KFOLD, MSEL_SPEC_GROUP_KFOLD,
    MSEL_SPEC_STRATIFIED_GROUP_KFOLD, MSEL_SPEC_TIME_SERIES, MSEL_SPEC_LOO, MSEL_SPEC_LPO, MSEL_SPEC_LOGO,
    MSEL_SPEC_LPGO, MSEL_SPEC_PREDEFINED, MSEL_SPEC_SHUFFLE, MSEL_SPEC_GROUP_SHUFFLE, MSEL_SPEC_STRATIFIED_SHUFFLE,
    MSEL_SPEC_REPEATED_KFOLD, MSEL_SPEC_REPEATED_STRATIFIED_KFOLD, msel_spec_has_n_splits,
)


def msel_split_spec_of(cv: PythonObject, context: PythonObject, mut spec: MselSplitSpec,
                       mut pin: MselRoutePin) -> Bool:
    """`cv`, one of our splitter objects, as typed words in `spec` (its
    entries' connections in `pin`, which the caller keeps for every fit).
    False (the caller keeps its door, which splits natively with X and y and
    raises the splitter's own words) for: an iterable cv (`_IterableCV`,
    whose pairs the caller reads as an index list), a class the table does
    not connect, a stored field that is not an exact integer where the door
    reads one, or any value the door refuses. A stratified splitter on a
    regressor is taken (lane mf-del5): its fit stratifies on the targets'
    values as the door's split does (bindings/msel_split_spec_run.mojo
    `msel_spec_sides_targets`; its `classifier` argument is removed).
    `context` is the
    model-selection context the splitter door reads."""
    try:
        return msel_split_spec_read(cv, context, spec, pin, False)
    except:
        return False


def msel_split_spec_read(cv: PythonObject, context: PythonObject, mut spec: MselSplitSpec,
                         mut pin: MselRoutePin, strict: Bool) raises -> Bool:
    """The body of `msel_split_spec_of`. False: `cv` is not one of our
    splitter classes, or an `_IterableCV` (whose pairs the caller reads as
    an index list). `strict` False (`msel_split_spec_of`): every case the
    spec does not take answers False or raises. `strict` True (lane
    mf-lin6, an x_linear CV member): every splitter the public split door
    splits is taken, and every refusal is the door's own, in its words: a
    field the door reads raises here as it raises there (`msel_int`,
    `msel_seed`, the sizes, the stored fields); a KFold-family n_splits
    below 2 is taken and its plan raises the door's words at fit (the same
    `msel_plan_*` call with the same k); a TimeSeriesSplit max_train_size
    the door cannot read is kept as text and raised at fit only when the
    plan makes a split (the door's order, `msel_time_series_split`); a
    repeated splitter refuses its seed before its `cv` and reads its repeats
    before its per-repeat folds, as `msel_repeated_split`; a connected class
    that is none of the sixteen raises the door's `NotImplementedError`."""
    var layouts = py_get(context, "family_msel_splitter_layouts")
    if layouts is None or not py_contains(layouts, py_type(cv)):
        return False
    var sp = MselSplitter(cv, context)
    var kind = sp.kind
    if kind == SPLITTER_ITERABLE:
        return False
    if kind == SPLITTER_OTHER:
        if strict:
            raise Error("NotImplementedError: ")  # `msel_split`'s last line
        return False
    var s = MselSplitSpec()
    if (kind == MSEL_SPEC_KFOLD or kind == MSEL_SPEC_STRATIFIED_KFOLD or kind == MSEL_SPEC_GROUP_KFOLD
            or kind == MSEL_SPEC_STRATIFIED_GROUP_KFOLD):
        s.k = msel_int(sp.field("n_splits", context))
        if s.k < 2 and not strict:
            return False
        s.shuffle = msel_truth(sp.field("shuffle", context))
        s.seed = msel_seed(sp.field("random_state", context), s.shuffle, context)
    elif kind == MSEL_SPEC_TIME_SERIES:
        s.k = msel_int(sp.field("n_splits", context))
        var ts_field = sp.field("test_size", context)
        s.has_ts = ts_field is not None
        s.ts = msel_int(ts_field) if s.has_ts else 0
        s.gap = msel_int(sp.field("gap", context))
        var max_train_field = sp.field("max_train_size", context)
        if strict:
            # The door reads it after its fold-count refusals and only for a
            # plan with a split: its words are kept for that moment.
            try:
                s.max_train = msel_int(max_train_field) if msel_truth(max_train_field) else 0
            except problem:
                s.max_train_unread = True
                s.max_train_refusal = String(problem)
        else:
            # An unreadable max_train_size is raised by the door after its
            # fold-count refusals: the door keeps that case.
            s.max_train = msel_int(max_train_field) if msel_truth(max_train_field) else 0
    elif kind == MSEL_SPEC_LPO:
        s.k = msel_int(sp.field("p", context))
    elif kind == MSEL_SPEC_LPGO:
        s.k = msel_int(sp.field("n_groups", context))
    elif kind == MSEL_SPEC_PREDEFINED:
        var folds = sp.field("_folds", context)
        var labels = msel_label_codes(folds, Int(py=folds.size), "test_fold", context, True, True)
        var view = labels.view
        s.folds = List[Int32](length=view.n, fill=Int32(0))
        var src = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=view.address)
        for i in range(view.n):  # driver-loop: over the test_fold codes (UNPACK, once per admission)
            s.folds[i] = src[i]
        _ = labels  # the codes stay alive through the copy above
        s.fold_classes = view.m
        for c in range(len(labels.classes)):  # driver-loop: over the fold values (every one but -1)
            if msel_cmp("ne", labels.classes[c], PythonObject(-1)):
                s.wants.append(c)
    elif kind == MSEL_SPEC_SHUFFLE or kind == MSEL_SPEC_GROUP_SHUFFLE or kind == MSEL_SPEC_STRATIFIED_SHUFFLE:
        s.k = msel_int(sp.field("n_splits", context))
        var test = sp.field("test_size", context)
        var train = sp.field("train_size", context)
        if test is None and train is None:
            test = sp.field("_default_test_size", context)
        s.test = msel_size_word(test, context)
        s.train = msel_size_word(train, context)
        var over = msel_size_sum(test, train, context)
        s.sum_over = over[0]
        s.sum_text = over[1]
        s.seed = msel_seed(sp.field("random_state", context), True, context)
    elif kind == MSEL_SPEC_REPEATED_KFOLD or kind == MSEL_SPEC_REPEATED_STRATIFIED_KFOLD:
        s.seed = msel_seed(sp.field("random_state", context), True, context)
        if strict:
            msel_repeated_check(s.seed)  # the door refuses the seed before it reads `cv`
        var base = sp.field("cv", context)
        # The door decides by the stored class, as `msel_repeated_split`.
        if base is msel_cls("StratifiedKFold", context):
            kind = MSEL_SPEC_REPEATED_STRATIFIED_KFOLD
        elif base is msel_cls("KFold", context):
            kind = MSEL_SPEC_REPEATED_KFOLD
        else:
            return False
        var cvargs = sp.field("cvargs", context)
        if strict:
            # The door's order: n_repeats, no split at all for n_repeats <= 0
            # (then no fold-count check), else the per-repeat fold count.
            s.repeats = msel_int(sp.field("n_repeats", context))
            if s.repeats > 0:
                s.k = msel_check_splits(py_get(cvargs, PythonObject("n_splits"), PythonObject(5)), context)
        else:
            s.k = msel_check_splits(py_get(cvargs, PythonObject("n_splits"), PythonObject(5)), context)
            s.repeats = msel_int(sp.field("n_repeats", context))
    elif kind != MSEL_SPEC_LOO and kind != MSEL_SPEC_LOGO:
        return False  # LeaveOneOut and LeaveOneGroupOut store no setting
    s.kind = kind
    # Every entry a splitter may call (the base table past the stratified
    # program bounds and for the group plans; the program runner for a
    # draw or a sort).
    pin = msel_split_route(context, True, True)
    s.route = pin.route
    s.sabotage = msel_sabotage()
    spec = s^
    return True


def msel_cv_fold_words(cv: PythonObject, context: PythonObject) raises -> Tuple[Int, Bool]:
    """UNPACK (lane mf-del5): what scikit-learn 1.9 calibration.py's cv
    checks read of a cv, for `cal_cv_check`: (its fold count: `cv` when it is
    an exact int, else the stored `n_splits` of one of our splitters whose
    scikit-learn class keeps that attribute, `msel_spec_has_n_splits`; 0
    for None, an index list and every other splitter; whether it is a
    LeaveOneOut). `context` is the model-selection context the splitter
    door reads."""
    var b = msel_b()
    if cv is None:
        return (0, False)
    if py_type(cv) is b.int:
        return (Int(py=cv), False)
    var layouts = py_get(context, "family_msel_splitter_layouts")
    if layouts is None or not py_contains(layouts, py_type(cv)):
        return (0, False)
    var sp = MselSplitter(cv, context)
    if msel_spec_has_n_splits(sp.kind):
        return (msel_int(sp.field("n_splits", context)), False)
    return (0, sp.kind == MSEL_SPEC_LOO)
