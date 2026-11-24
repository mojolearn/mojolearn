# SPDX-License-Identifier: Apache-2.0
"""Model selection's ONE-FIT door over a NATIVE MEMBER (core/member_abi.mojo):
a search fitted as one member of its own META kind (core/ensemble_member.mojo
`_fit_search`, whose RUN core/msel_search_run.mojo runs every candidate and
fold), the cv tools' carrier search, and a search's refit of its best
estimator.

LANE mf-msel5c (2026-11-23): the (candidate, fold) units this door used to
run over native members (`msel_member_units`, the route
bindings/msel_cv_door.mojo `_msel_member_route` called) are gone with that
route: every search and cv tool a member serves is the search kind's typed
RUN; what the kind does not take runs the door's Python units. What is left
is one fit:
  * UNPACK (GIL held, once): the estimator (or its clone with the given
    parameters) admitted as a member (`member_admit_serving_*`); X read once
    as an address (host float32, C order); y as the member's fit reads it:
    float32 targets (`classical_check_y`), or a CLASSIFIER's int32 codes by
    model selection's one label encoder (`msel_encode`: code c is the
    estimator's own `classes_[c]`), a 2-D y PER OUTPUT (THE 2-D y, layout 4:
    each column encoded on its own, rows x outputs codes and each output's
    class count in the plan's `class_counts`). A SEARCH over a classifier
    encodes y BEFORE its admission and hands it the labels (the search kind's
    scorers read them: positive classes, label words, Brier marks).
  * RUN: one member fit on the whole X (a view, nothing gathered or copied),
    with the GIL released for a kind with `MEMBER_GIL_FREE` or
    `MEMBER_GIL_DOOR`, else held (THE GIL RULE). In the caller's resident
    scope X's own object is pinned once for the fit (THE PIN RULE).
  * PACK (GIL held): the fitted state packed into the estimator by its
    family (`member_pack_labeled` with the labels the estimator's own fit
    publishes; a list of per-output labels for a 2-D y) and held by it
    (bindings/member_held_door.mojo); its fit notices raised.

THE REFUSAL RULE (lane mf-r9a, 2026-11-23; was THE FALLBACK RULE): no door
fits behind this one, so it never answers None / False. Every input either
fits here or is refused here, natively and before any fit, in the words of
whoever refuses it in scikit-learn 1.9 (the module's exits, one by one):
  * a class with no member: the estimator plan's own words (a class that is
    no Mojolearn estimator), Python's TypeError for `fit(X, y)` of a class
    whose fit takes no target (member_set_refusal's words);
  * X: an Array of C-order float32 words is read as it is; any other X
    (float64, another order, a list, an ndarray) is admitted ONCE by the
    product's X admission (`admit_f32`, its rank / empty / dtype words), as
    the estimator's own fit converts it;
  * y: its rows against X's (check_consistent_length's words, before any
    fit, as `indexable` raises them); a classifier's labels: NaN / infinity
    (`_assert_all_finite`'s words), non-whole floats ('continuous' /
    'continuous-multioutput', check_classification_targets' words), a ragged
    label table (NumPy's inhomogeneous-shape words), a one-column 2-D y read
    as its 1-D column; one class is the member fit's to take or refuse;
  * the admission: the never-raising serving admission is run once more,
    raising, on the refusal path only (the family's or the search kind's
    words), else the missing capability's words;
  * non-finite X for a kind without `MEMBER_CAN_TEST_ROWS`: check_array's
    words (`admitted_finite_wording`), naming the base estimator;
  * a failing fit or pack: the member's own words, its state released.
A refusal every fit of a search would raise (non-finite X, continuous labels,
a target the regressor check refuses, a class_weight dict) is, for a search
with a numeric error_score, scikit-learn's all-fits-failed words
(`_fit_words`): `_warn_or_raise_about_fit_failures` over its candidates x
folds fits.

CLASSIFIER RULE: y as long as X; float labels whole finite numbers; a 2-D y
of a numeric dtype in C order, or a label table (rows of Python labels, a
text or object 2-D array) read once per output.

WHO REFUSES A NON-FINITE X: a member whose KIND has `MEMBER_CAN_TEST_ROWS`
refuses non-finite rows in its own fit; for any other X's verdict is read
once (the held block's recorded verdict, else one scan across the host
cores) and a non-finite X is refused here.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.classical_validate_unpack import classical_check_y, classical_estimator_name
from providers.python.member_xneighbors_boundary_door import xneighbors_kind_reads_marker, xneighbors_marker_index
from core.xneighbors_member import LP_MARKER_WORD
from bindings.member_boundary_door import (
    member_admit_labeled, member_admit_serving_deferred, member_admit_serving_labeled, member_pack,
    member_pack_labeled,
)
from bindings.admitted_input_unpack import admit_f32, admit_weight_words
from core.admitted_refusals import admitted_finite_wording
from bindings.msel_candidates_door import msel_grid_len, msel_sampler_check, msel_sampler_len
from bindings.native_loaded_module_door import native_loaded_module
from providers.python.member_ensemble_boundary_door import _all_fits_failed, _no_member_fit_words
from bindings.member_class_weight_unpack import member_cw_fill_plan
from bindings.member_owner_unpack import MemberOwner
from bindings.member_held_door import member_held_adopt
from bindings.member_fit_notices_pack import member_fit_notices_raise
from core.member_notices import member_notices_take
from bindings.member_scope_door import member_scope_word, member_pin_take, member_pin_return
from bindings.classifier_public_lifecycle_door import _public_labels
from bindings.msel_common_door import (
    msel_is_array, msel_addr, msel_encode, msel_new, msel_is_integral, msel_try_materialize, msel_missed,
)
from bindings.msel_split_door import msel_cls
from bindings.msel_estimator_native_door import (
    msel_estimator_plan, msel_native_clone, msel_native_set_params, msel_clone_parameter,
)
from bindings.py_balanced import py_call, py_type, py_get
from bindings.py_admitted_registry import admitted_held_index
from bindings.admitted_meta_scope_unpack import admitted_held_verify
from core.host_finite_scan import host_all_finite_f32
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free, member_kind_capabilities,
    MEMBER_GIL_FREE, MEMBER_GIL_DOOR, MEMBER_F32, MEMBER_I32, MEMBER_CAN_EXPORT, MEMBER_CAN_TEST_ROWS,
    MEMBER_OP_FIT, MEMBER_CAN_FIT, MEMBER_CAN_CLASSIFIER, MEMBER_CAN_SCOPE, MEMBER_SCOPE_NONE, member_pin_settle,
    MEMBER_READS_Y_GIVEN, MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_CAN_MULTI_OUTPUT, MEMBER_CAN_STEPS,
)
from bindings.member_given_y_unpack import member_given_y

# REMOVED: msel_member_units (the native (candidate, fold) units of bindings/msel_cv_door.mojo
# `_msel_member_route`; not an experiment define: a door path) deleted 2026-11-23 by lane/mf-msel5c; code
# recoverable at b3ca4be7a9 (this file before the change: its MselMemberRun / MselMemberScores, the fold
# shapes, gathers, responses and UNIT_* words).


struct _FitOne(Movable):
    """The one fit's typed words: built by UNPACK, read by the RUN through
    one address, its results read back by PACK."""
    var plan: MemberPlan
    var x: Int
    var rows: Int
    var cols: Int
    # THE SEQUENCES (lane w3f5-c): 0, a plain matrix; T, each row of X is T
    # steps of `cols / T` features (a (rows, T, F) X of a kind with
    # MEMBER_CAN_STEPS; core/member_abi.mojo `MemberMatrix.steps`).
    var steps: Int
    # The target: 0 (a fit without one), float32 values, or int32 codes
    # (rows x `y_cols` words of `y_dtype`).
    var y: Int
    var y_dtype: Int
    var y_cols: Int
    # LANE w3f3-b: the fit's float32 sample weights (rows words; 0: none),
    # for a kind with MEMBER_CAN_SAMPLE_WEIGHT.
    var weights: Int
    # THE GIVEN y (lane mf-tail5; core/member_abi.mojo `MemberCall.y_given`):
    # a regressor's y as the caller was given it, for a kind that reads it.
    var given: MemberMatrix
    # THE RESIDENT SCOPE: the caller's scope word and X's pin token (0: none
    # offered); `taken` is 1 when the fit's cache took the reference.
    var scope: Int
    var pin: Int
    var taken: Int
    var state: Int
    var failed: Bool
    var error: String
    var notices: List[Float64]

    def __init__(out self, var plan: MemberPlan, x: Int, rows: Int, cols: Int):
        self.plan = plan^
        self.x = x
        self.rows = rows
        self.cols = cols
        self.steps = 0
        self.y = 0
        self.y_dtype = MEMBER_F32
        self.y_cols = 1
        self.weights = 0
        self.given = MemberMatrix()
        self.scope = MEMBER_SCOPE_NONE
        self.pin = 0
        self.taken = 0
        self.state = 0
        self.failed = False
        self.error = String("")
        self.notices = List[Float64]()


def _fit_one(address: Int):
    """RUN: the member's fit on every row of X (a view). Touches no CPython;
    a failure is recorded in the run's words, never raised."""
    var run = MutPointer[_FitOne, MutUntrackedOrigin](unsafe_from_address=address)
    try:
        var call = member_new(run[].plan)
        call.scope = run[].scope
        call.x = MemberMatrix(run[].x, run[].rows, run[].cols)
        call.x.steps = run[].steps
        if run[].y != 0:
            call.y = MemberMatrix(run[].y, run[].rows, run[].y_cols, run[].y_dtype)
            call.y_given = run[].given
        if run[].weights != 0:
            call.weights = MemberMatrix(run[].weights, run[].rows, 1)
        call.pin = run[].pin
        var kept = False
        try:
            try:
                _ = member_call(run[].plan.member, MEMBER_OP_FIT, call)
            finally:
                if run[].pin != 0:
                    run[].taken += member_pin_settle(call)
            member_notices_take(run[].plan, call.state, run[].notices)
            run[].state = call.state
            kept = True
        finally:
            if not kept:
                member_free(run[].plan, call)
    except error:
        run[].error = String(error)
        run[].failed = True


comptime _LABELS_WHOLE = 0
comptime _LABELS_CONTINUOUS = 1
comptime _LABELS_NAN = 2
comptime _LABELS_INFINITE = 3


def _label_kind(classes: PythonObject, b: PythonObject) raises -> Int:
    """How scikit-learn's `type_of_target` reads a classifier's float labels
    (sklearn 1.9 utils/multiclass.py): `_LABELS_NAN` / `_LABELS_INFINITE`
    when one is not finite (its `_assert_all_finite` raises, NaN first),
    `_LABELS_CONTINUOUS` when one is not a whole number, else
    `_LABELS_WHOLE`. Int, bool and text labels are whole. One read per
    CLASS, never per row. Every float beyond 2^53 is a whole number."""
    var kind = _LABELS_WHOLE
    for i in range(len(classes)):  # driver-loop: over the distinct classes
        var label = classes[i]
        if py_type(label) is b.float:
            var value = Float64(py=label)
            if value != value:
                return _LABELS_NAN
            if not (value - value == 0.0):
                kind = _LABELS_INFINITE
            elif kind == _LABELS_WHOLE and value < 9.0e15 and value > -9.0e15 and Float64(Int(value)) != value:
                kind = _LABELS_CONTINUOUS
    return kind


def _y_float_name(y: PythonObject, context: PythonObject) raises -> String:
    """The NumPy name of a float y's dtype in `_assert_all_finite`'s words:
    'float32' for a float32 Array, 'float64' otherwise (a label list is read
    by NumPy as float64)."""
    if msel_is_array(y, context) and String(py=y.dtype) == "<f4":
        return "float32"
    return "float64"


def _search_fits(est: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject) raises -> Int:
    """The fits a search's `_fit_and_score` would make: candidates x folds
    (scikit-learn's `n_candidates * n_splits`), counted without drawing a
    candidate (a RandomizedSearchCV's random_state is not advanced); 0 when
    the folds are not countable without splitting (the caller then raises
    the one fit's words). The folds: a cv tool carrier's own folds, 5 for
    cv=None, the integer, or a user's index list."""
    var folds = 0
    var tool = py_get(fields, "_msel_cv_tool")
    var cv = py_get(fields, "cv")
    if tool is not None:
        folds = len(tool[3])
    elif cv is None:
        folds = 5
    elif py_type(cv) is b.list or py_type(cv) is b.tuple:
        folds = len(cv)
    elif py_type(cv) is b.int or msel_is_integral(cv, context):
        folds = Int(py=py_call(b.int, cv))
    else:
        return 0
    var candidates = 0
    if py_type(est) is msel_cls("GridSearchCV", context):
        candidates = msel_grid_len(fields["param_grid"], context)
    else:
        var dists = msel_sampler_check(fields["param_distributions"], context)
        candidates = Int(py=msel_sampler_len(dists, fields["n_iter"], context))
    return candidates * folds


def _fit_words(est: PythonObject, fields: PythonObject, var words: String, context: PythonObject,
               b: PythonObject) raises -> String:
    """THE REFUSAL RULE's words (module note) for a refusal EVERY fit would
    raise: `words` itself, or for a search (`est`, its instance dictionary
    `fields`) with a numeric error_score scikit-learn's all-fits-failed
    words over its candidates x folds fits (sklearn 1.9
    model_selection/_validation.py `_warn_or_raise_about_fit_failures`, the
    search kind's `_all_fits_failed`). Read on the refusal path only."""
    if not _is_search(est, context):
        return words^
    var error_score = py_get(fields, "error_score")
    if error_score is None or py_type(error_score) is b.str:
        return words^
    if not Bool(py=py_call(b.isinstance, error_score, native_loaded_module("numbers").Real)):
        return words^
    var fits = _search_fits(est, fields, context, b)
    if fits <= 0:
        return words^
    var errors: List[String] = [words]
    var counts: List[Int] = [fits]
    return _all_fits_failed(fits, errors, counts)


def _check_labels(classes: PythonObject, multi: Bool, y: PythonObject, est: PythonObject, fields: PythonObject,
                  context: PythonObject, b: PythonObject) raises:
    """A classifier's labels (one output's classes) as scikit-learn reads
    them before its fit (`_label_kind`): NaN or infinity raise
    `_assert_all_finite`'s words (from `check_cv`'s `type_of_target`, before
    any fit); non-whole floats check_classification_targets' words, raised by
    every fit (`_fit_words`)."""
    var kind = _label_kind(classes, b)
    if kind == _LABELS_NAN:
        raise Error("ValueError: Input y contains NaN.")
    if kind == _LABELS_INFINITE:
        raise Error("ValueError: Input y contains infinity or a value too large for dtype('"
                    + _y_float_name(y, context) + "').")
    if kind == _LABELS_CONTINUOUS:
        var target = String("continuous-multioutput") if multi else String("continuous")
        raise Error(_fit_words(est, fields, "ValueError: Unknown label type: " + target + ". Maybe you are trying"
                               + " to fit a classifier, which expects discrete classes on a regression target with"
                               + " continuous values.", context, b))


def _unknown_target_words(est: PythonObject, fields: PythonObject, context: PythonObject,
                          b: PythonObject) raises -> String:
    """check_classification_targets' words for a target `type_of_target`
    reads as 'unknown' (a 2-D y with no column, a y of rank 3 or more),
    raised by every fit."""
    return _fit_words(est, fields, "ValueError: Unknown label type: unknown. Maybe you are trying to fit a"
                      + " classifier, which expects discrete classes on a regression target with continuous values.",
                      context, b)


def _inconsistent_words(rows: Int, y_rows: Int) -> String:
    """check_consistent_length's words (sklearn 1.9 utils/validation.py)."""
    return ("ValueError: Found input variables with inconsistent numbers of samples: [" + String(rows) + ", "
            + String(y_rows) + "]")


def _invariant(what: String) -> String:
    return "RuntimeError: model selection member fit invariant: " + what


def _numeric_width(dtype: String) -> Int:
    """The bytes of one element of a numeric NumPy-style typestr ('<i8',
    '<f4', '|b1', '<u2', ...), 0 for any other kind."""
    if dtype.byte_length() != 3:
        return 0
    var kind = String(dtype[byte=1])
    if kind != "i" and kind != "u" and kind != "f" and kind != "b":
        return 0
    var size = String(dtype[byte=2])
    if size == "1":
        return 1
    if size == "2":
        return 2
    if size == "4":
        return 4
    if size == "8":
        return 8
    return 0


def _encode_outputs(y: PythonObject, rows: Int, est: PythonObject, fields: PythonObject, context: PythonObject,
                    b: PythonObject) raises -> PythonObject:
    """THE 2-D y of a classifier (layout 4), PER OUTPUT: each column of a C
    order numeric rows x outputs y copied once into its own vector (its
    elements' bytes, no conversion) and encoded by the one label encoder.
    Returns (int32 rows x outputs codes, [classes of each output]); for a
    ONE-COLUMN y (None, (classes, codes)): its column read as the 1-D y
    (scikit-learn's classifiers ravel it). Every other target is refused in
    scikit-learn's words (THE REFUSAL RULE; y's rows were checked against
    X's by the caller)."""
    if Int(py=y.ndim) != 2:
        raise Error(_unknown_target_words(est, fields, context, b))
    if Int(py=y.shape[0]) != rows or rows == 0:
        raise Error(_invariant("a 2-D y reaches the per-output encode with other rows than X's"))
    if String(py=y.order) != "C":
        raise Error(_invariant("msel_materialize_y answers a 2-D y in another order than C"))
    var outputs = Int(py=y.shape[1])
    var dtype = String(py=y.dtype)
    var width = _numeric_width(dtype)
    if outputs == 0:
        raise Error(_unknown_target_words(est, fields, context, b))
    if width == 0:
        raise Error("TypeError: mojolearn: a 2-D classifier target of dtype " + dtype + " is not read as class"
                    + " labels; pass numbers, booleans or strings")
    # The codes block of two outputs or more (a one-column y answers its
    # column's 1-D encode and needs none).
    var codes = PythonObject(None)
    var dst_at = 0
    if outputs > 1:
        codes = msel_new(rows * outputs, "<i4", context)
        dst_at = msel_addr(codes)
    var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=dst_at)
    var src = msel_addr(y)
    var classes = Python.list()
    for j in range(outputs):  # driver-loop: over the outputs (one column copy and one encode each)
        var column = msel_new(rows, dtype, context)
        var to = msel_addr(column)
        for i in range(rows):  # driver-loop: over the rows of one output column (its elements' bytes)
            unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=to + i * width),
                          src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=src + (i * outputs + j) * width),
                          count=width)
        var encoded = msel_encode(column, context)
        _ = column  # copied by the encode above
        if outputs == 1:
            _ = y  # read through its address above
            return Python.tuple(PythonObject(None), encoded)
        _check_labels(encoded[0], True, y, est, fields, context, b)
        var column_codes = encoded[1]
        if Int(py=column_codes.size) != rows or String(py=column_codes.dtype) != "<i4":
            raise Error(_invariant("the label encoder answers other than one int32 code per row"))
        var got = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=msel_addr(column_codes))
        for i in range(rows):  # driver-loop: over the rows (output j's codes into its column of the block)
            dst[i * outputs + j] = got[i]
        _ = column_codes  # read through the pointer above
        _ = py_call(classes.append, encoded[0])
    _ = y  # read through its address above
    return Python.tuple(codes, classes)


def msel_text_rows(y: PythonObject, context: PythonObject, b: PythonObject) raises -> Bool:
    """True for a 2-D target that is not an admitted Array: a list or tuple
    of rows, or any other object with `ndim` 2 (a NumPy text or object
    array). Read once per fit, before any label (lane mf-tail5b)."""
    if py_type(y) is b.list or py_type(y) is b.tuple:
        return len(y) > 0 and (py_type(y[0]) is b.list or py_type(y[0]) is b.tuple)
    var ndim = py_call(b.getattr, y, PythonObject("ndim"), PythonObject(None))
    return ndim is not None and py_type(ndim) is b.int and Int(py=ndim) == 2


def _encode_text_outputs(y: PythonObject, rows: Int, est: PythonObject, fields: PythonObject,
                         context: PythonObject, b: PythonObject) raises -> PythonObject:
    """THE LABEL TABLE of a classifier (lane mf-tail5b; a list or tuple of
    rows, or a text or object 2-D array), PER OUTPUT: its rows read ONCE at
    UNPACK (boundary work: the labels are Python objects), each output's
    labels gathered into one list and encoded by the one label encoder
    (`msel_encode`, the 1-D text route's), written as int32 rows x outputs
    codes. Returns (codes, [classes of each output]) as `_encode_outputs`
    does, its classes kept for PACK; (None, (classes, codes)) for a table of
    ONE column (its column read as the 1-D y). Numeric labels in a table are
    read the same way (lane mf-r9a: they kept the caller's door before). A
    ragged table is refused in NumPy's words (scikit-learn's `check_array`
    of y raises them); y's rows were checked against X's by the caller."""
    var listed = py_call(b.list, y)
    if len(listed) != rows or rows == 0:
        raise Error(_invariant("a label table reaches the per-output encode with other rows than X's"))
    var first = py_call(b.list, listed[0])
    var outputs = len(first)
    if outputs == 0:
        raise Error(_unknown_target_words(est, fields, context, b))
    var columns = Python.list()
    for j in range(outputs):  # driver-loop: over the outputs (one label list each)
        _ = py_call(columns.append, Python.list())
    for i in range(rows):  # driver-loop: over the rows (each row's labels, read once)
        var row = py_call(b.list, listed[i])
        if len(row) != outputs:
            raise Error("ValueError: setting an array element with a sequence. The requested array has an"
                        + " inhomogeneous shape after 1 dimensions. The detected shape was (" + String(rows)
                        + ",) + inhomogeneous part.")
        for j in range(outputs):  # small-loop(the outputs)
            _ = py_call(columns[j].append, row[j])
    if outputs == 1:
        return Python.tuple(PythonObject(None), msel_encode(columns[0], context))
    var codes = msel_new(rows * outputs, "<i4", context)
    var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=msel_addr(codes))
    var classes = Python.list()
    for j in range(outputs):  # driver-loop: over the outputs (one encode each)
        var encoded = msel_encode(columns[j], context)
        _check_labels(encoded[0], True, y, est, fields, context, b)
        var column_codes = encoded[1]
        if Int(py=column_codes.size) != rows or String(py=column_codes.dtype) != "<i4":
            raise Error(_invariant("the label encoder answers other than one int32 code per row"))
        var got = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=msel_addr(column_codes))
        for i in range(rows):  # driver-loop: over the rows (output j's codes into its column of the block)
            dst[i * outputs + j] = got[i]
        _ = column_codes  # read through the pointer above
        _ = py_call(classes.append, encoded[0])
    _ = codes  # written through its address above
    return Python.tuple(codes, classes)


def _is_search(model: PythonObject, context: PythonObject) raises -> Bool:
    var cls = py_type(model)
    return cls is msel_cls("GridSearchCV", context) or cls is msel_cls("RandomizedSearchCV", context)


def msel_member_refit(base: PythonObject, parameters: PythonObject, y: PythonObject, x: PythonObject,
    context: PythonObject, b: PythonObject) raises -> PythonObject:
    """A search's refit over the native member: a clone of `base` with
    `parameters` set, fitted ONCE on every row of X (a view: nothing is
    gathered or copied) with the GIL released, packed by its family's
    `member_pack` and holding its fitted member state
    (bindings/member_held_door.mojo), so a caller that uses the best
    estimator as a member imports nothing. Returns the fitted clone; a
    member, data or fit the route does not take is refused natively, before
    any fit, in the words of whoever refuses it (THE REFUSAL RULE, module
    note). y None: a fit without a target (an unsupervised search)."""
    return _member_fit(base, parameters, True, False, y, x, PythonObject(None), context, b)


def msel_member_fit_in_place(model: PythonObject, y: PythonObject, x: PythonObject, context: PythonObject,
    b: PythonObject) raises:
    """`msel_member_refit` of `model` ITSELF (lane mf-msel5): its own
    instance dictionary is admitted (no clone, so a field only its door set,
    the carrier's `_msel_cv_tool`, is read), fitted once on every row of X
    with the GIL released and packed into it in place. Every refusal
    raises (lane mf-r9a, THE REFUSAL RULE; lane mf-r9b: no Bool, the
    callers' fallback branches are deleted)."""
    _ = _member_fit(model, PythonObject(None), False, False, y, x, PythonObject(None), context, b)


def msel_member_fit_weighted(model: PythonObject, y: PythonObject, x: PythonObject, sample_weight: PythonObject,
    context: PythonObject, b: PythonObject) raises:
    """`msel_member_fit_in_place` with the fit's `sample_weight` (lane
    w3f3-b: an owner's fit call of a member-only class). A kind that reads
    no weights (no MEMBER_CAN_SAMPLE_WEIGHT) refuses in words before any
    fit; the weights are admitted as the estimator's own fit admits them."""
    _ = _member_fit(model, PythonObject(None), False, False, y, x, sample_weight, context, b)


def msel_member_fit_cloned(model: PythonObject, y: PythonObject, x: PythonObject, context: PythonObject,
    b: PythonObject) raises:
    """`msel_member_fit_in_place` of a `model` the caller made itself with
    `msel_native_clone` (lane mf-msel5c: the search door's carrier), whose
    parameters that clone already admitted: no second admission clone.
    Every refusal raises, as `msel_member_fit_in_place`."""
    _ = _member_fit(model, PythonObject(None), False, True, y, x, PythonObject(None), context, b)


def _admission_refusal(est: PythonObject, fields: PythonObject, plan: PythonObject, member_context: PythonObject,
                       b: PythonObject, needs: Int, labels: PythonObject, context: PythonObject) raises -> String:
    """The words of a serving admission that answered `published` False
    (THE REFUSAL RULE; bindings/member_set_door.mojo `member_set_refusal`'s
    order): the raising admission runs ONCE more, on this refusal path only,
    and raises the family's own words (for a search, the search kind's:
    its scorers, refit, cv, error_score, base and candidates); else a family
    that keeps no member for these parameters (`_no_member_fit_words`), a
    member with no fit (AttributeError, as `getattr(est, 'fit')` raises), or
    one that exports no fitted state (`needs`: the PACK hands it back).

    A MEMBER-ONLY base (python/mojolearn/_native_msel_more.py
    connect_members: a marked class registered from its mark, without
    method recipes) is refused like any other: the search kind's raising
    admission says why (scikit-learn's check_scoring TypeError for a base
    with no `score` under scoring=None, `_check_response_method`'s
    AttributeError for a scorer whose response the class lacks, its family's
    own fit words). Lane w3f2-cv (gate 14f1 group B): REMOVED the
    "schema is pending migration" words lane mf-r9b raised here FIRST for
    every member-only base of a search or cv tool its member did not serve:
    they hid the real cause of every such refusal (a member-only plan's
    empty method table answered no scorer response, msel_scorer_native_door
    `_scorer_response`, now read off its class).

    THE BASE'S OWN WORDS (lane w3f2-words): a search's kind admits its
    candidates SERVING (providers/python/member_ensemble_boundary_door.mojo
    `_admit_nested`: a base its member refuses answers False, silently), so
    before the search's own admission words (a member-only base) or
    `_no_member_fit_words` the base's RAISING admission runs once
    (`_base_refusal`): a base its own fit
    door refuses (a host class's fit, HostLasso; a Python callback,
    DictionaryLearning's) raises that door's words, as scikit-learn's search
    raises its base's fit error. On this refusal path only."""
    var search = _is_search(est, context)
    var inner = PythonObject(None)
    var inner_plan = PythonObject(None)
    if search:
        inner = py_get(fields, "estimator")
        if inner is not None:
            inner_plan = msel_estimator_plan(inner, context)
            if py_get(inner_plan, "member_only") is not None:
                _base_refusal(inner, inner_plan, b, labels)
    var name = String(py=py_type(est).__name__)
    var owner = member_admit_labeled(est, fields, plan, member_context, b, labels)
    if not owner.published:
        if inner is not None:
            _base_refusal(inner, inner_plan, b, labels)
        return _no_member_fit_words(est)
    if not owner.plan.member.has(MEMBER_CAN_FIT):
        return "AttributeError: '" + name + "' object has no attribute 'fit'"
    if not owner.plan.member.has(needs):
        return ("NotImplementedError: " + name + "'s native member exports no fitted state, so a Mojolearn search"
                + " cannot hand it back fitted. Fit " + name + " on its own.")
    return _invariant("the serving admission refused " + name + " and the raising one took it (a class_weight"
                      + " tail is read over the labels handed to both)")


def _base_refusal(inner: PythonObject, inner_plan: PythonObject, b: PythonObject, labels: PythonObject) raises:
    """A search's base admitted ONCE by its family's RAISING admission
    (`member_admit_labeled`, never the serving one): a base its own fit door
    refuses raises those words here (`_admission_refusal`, its refusal path
    only); a base with no member mark, or one the family admits, returns and
    the caller raises its own words. The base as the caller gave it (a
    one-point grid's candidate is that base)."""
    if py_get(inner_plan, "member") is None:
        return
    var inner_fields = py_call(inner_plan["fields_get"], inner, py_type(inner))
    var owner = member_admit_labeled(inner, inner_fields, inner_plan, inner_plan["context"], b, labels)
    _ = owner.published
    _ = inner_fields  # read by the admission above


def _base_name(est: PythonObject, fields: PythonObject, context: PythonObject) raises -> String:
    """The class scikit-learn's check words name: a search's base estimator
    (its fits raise them), else the estimator itself."""
    if _is_search(est, context):
        var inner = py_get(fields, "estimator")
        if inner is not None:
            return String(py=py_type(inner).__name__)
    return String(py=py_type(est).__name__)


def _encode_target(y: PythonObject, rows: Int, text_y: Bool, label_list: Bool, est: PythonObject,
                   fields: PythonObject, context: PythonObject, b: PythonObject) raises -> PythonObject:
    """A classifier's target in the one label encoding: (encoded, None) for
    a 1-D y, a label list or a one-column y ((classes, int32 codes), its
    labels checked: `_check_labels`), (None, outputs) for a 2-D y or a label
    table (`_encode_outputs` / `_encode_text_outputs`: (codes block,
    [classes per output]), each output's labels checked). Raises every
    refusal in scikit-learn's words (THE REFUSAL RULE)."""
    var outputs = PythonObject(None)
    if text_y:
        outputs = _encode_text_outputs(y, rows, est, fields, context, b)
    elif label_list or Int(py=y.ndim) == 1:
        var encoded = PythonObject(None)
        try:
            encoded = msel_encode(y, context)
        except error:
            # The shared encoder refuses a NaN label first, in its own words ("y contains a NaN label",
            # bindings/classical_labels_door.mojo:392), so `_check_labels`' NaN row was unreachable: scikit-learn's
            # `type_of_target` raises `_assert_all_finite`'s words before any fit. The refusal path only.
            if "NaN label" in String(error):
                raise Error("ValueError: Input y contains NaN.")
            raise error^
        _check_labels(encoded[0], False, y, est, fields, context, b)
        return Python.tuple(encoded, PythonObject(None))
    else:
        outputs = _encode_outputs(y, rows, est, fields, context, b)
    if outputs[0] is None:
        # a one-column y: its column's 1-D encode
        _check_labels(outputs[1][0], False, y, est, fields, context, b)
        return Python.tuple(outputs[1], PythonObject(None))
    return Python.tuple(PythonObject(None), outputs)


def _member_fit(base: PythonObject, parameters: PythonObject, clone: Bool, vetted: Bool, y: PythonObject,
    x: PythonObject, sample_weight: PythonObject, context: PythonObject, b: PythonObject) raises -> PythonObject:
    """UNPACK, RUN and PACK of the one fit (module note): the fitted
    estimator (`base` itself, or its clone with `parameters` set). Every
    input it does not fit is refused natively, before any fit, in the words
    of whoever refuses it (THE REFUSAL RULE); it never answers None."""
    # A class that is no Mojolearn estimator: the plan raises its words.
    var plan = msel_estimator_plan(base, context)
    if py_get(plan, "member") is None:
        # A Mojolearn class with no member mark (its fit takes no target; the
        # mf-member8 sweep found Cholesky only): member_set_refusal's words.
        if y is not None:
            raise Error("TypeError: " + String(py=py_type(base).__name__)
                        + ".fit() takes 2 positional arguments but 3 were given")
        raise Error(_no_member_fit_words(base))
    var member_context = plan["context"]
    # THE TARGET as a door hands it: msel_materialize_y's Array or flat
    # label list (lane w3-fix-b, row 6: a classifier admits the list at
    # UNPACK into the one label encoding), or a label table (lane
    # mf-tail5b: rows of Python labels, a text or object 2-D array; a search
    # over a classifier reads it per output). A table of numeric words (a
    # numeric 2-D array, rows of numbers) is materialized here as the door's
    # y, an Array (lane mf-r9a: it kept the caller's door before); a table
    # the materialize refuses with TypeError (text or object labels, as
    # msel_materialize_y reads it) stays a table.
    var yv = y
    var text_y = yv is not None and not msel_is_array(yv, context) and msel_text_rows(yv, context, b)
    if text_y:
        var got = msel_try_materialize(yv, "y", b.TypeError, context)
        if not msel_missed(got, context):
            yv = got
            text_y = False
    var label_list = (yv is not None and not text_y and not msel_is_array(yv, context)
                      and (py_type(yv) is b.list or py_type(yv) is b.tuple))
    if yv is not None and not msel_is_array(yv, context) and not text_y and not label_list:
        raise Error(_invariant("y is neither an Array, a label list nor a label table (the doors hand"
                               + " msel_materialize_y's answer or a label table)"))
    # X: an Array of C-order float32 words is read as it is (a view); any
    # other X is admitted ONCE by the product's X admission, as the
    # estimator's own fit converts it (its rank, empty and dtype words; lane
    # mf-r9a: it kept the caller's door before). The finite test follows.
    var xv = x
    # THE SEQUENCES (lane w3f5-c; 7626d7be07 admitted every X as 2-D, so a
    # recurrent or CNN base's (n_samples, n_timesteps, n_features) X was
    # refused before its member ran): a 3-D Array is read as rows x (T * F)
    # words with `steps` = T, and only a kind with MEMBER_CAN_STEPS takes it
    # (checked below, once its mask is known). The doors' materialized X is
    # an Array; a float64 one gets the recurrent door's own words.
    var steps = 0
    var sequence_shape = String("")
    if msel_is_array(xv, context) and Int(py=xv.ndim) == 3:
        sequence_shape = String(py=py_call(b.str, xv.shape))
        if String(py=xv.dtype) == "<f8":
            raise Error(_fit_words(base, py_call(plan["fields_get"], base, py_type(base)),
                                   "X: float64 is refused; pass float32 (no float64 on the device)", context, b))
        if String(py=xv.dtype) != "<f4" or String(py=xv.order) != "C" or Int(py=xv.size) == 0:
            raise Error(_fit_words(base, py_call(plan["fields_get"], base, py_type(base)),
                                   "TypeError: a sequence X is read as a non-empty C-order float32 (n_samples,"
                                   + " n_timesteps, n_features) array, got " + String(py=xv.dtype) + " shape "
                                   + sequence_shape, context, b))
        steps = Int(py=xv.shape[1])
    elif (not msel_is_array(xv, context) or String(py=xv.dtype) != "<f4" or Int(py=xv.ndim) != 2
            or String(py=xv.order) != "C" or Int(py=xv.size) == 0):
        try:
            xv = admit_f32(xv, 2, "X", member_context, as_array=True, finite=False).array
        except error:
            raise Error(_fit_words(base, py_call(plan["fields_get"], base, py_type(base)), String(error), context,
                                   b))
    var rows = Int(py=xv.shape[0])
    var cols = Int(py=xv.shape[1]) if steps == 0 else steps * Int(py=xv.shape[2])
    if yv is not None:
        # y's rows against X's: scikit-learn's `indexable`
        # (check_consistent_length), before any fit.
        var y_rows = 0
        if msel_is_array(yv, context):
            if Int(py=yv.ndim) == 0:
                raise Error("TypeError: Singleton array cannot be considered a valid collection.")
            y_rows = Int(py=yv.shape[0])
        else:
            y_rows = len(yv)
        if y_rows != rows:
            raise Error(_inconsistent_words(rows, y_rows))
    var est = base
    if clone:
        est = msel_native_clone(base, context)
        _ = msel_native_set_params(est, msel_clone_parameter(parameters, context), context)
    elif not vetted:
        # The caller's estimator is admitted as its clone admits it (every
        # constructor parameter an exact native value, refused in words
        # before any Python protocol of it runs), then the member reads it in
        # place (lane sr3-smoke: a member's own admission must not read a
        # caller's object first).
        _ = msel_native_clone(base, context)
    var fields = py_call(plan["fields_get"], est, py_type(est))
    var needs = MEMBER_CAN_FIT | MEMBER_CAN_EXPORT
    # Everything that must stay alive until PACK is done: the admission's
    # pins, the encoded or converted target, X and y.
    var pins = Python.list()
    # THE LABELS (module note): `encoded` (classes, codes) of a 1-D y, or
    # `outputs` (codes block, [classes per output]) of a 2-D y.
    var encoded = PythonObject(None)
    var outputs = PythonObject(None)
    var early = False
    if yv is not None and _is_search(est, context):
        # A SEARCH over a classifier: its admission reads the labels.
        var search_base = py_get(fields, "estimator")
        if search_base is not None:
            try:
                early = Bool(py=msel_estimator_plan(search_base, context)["classifier"])
            except:
                early = False  # a base no plan holds: the search kind's admission raises its words below
    if early:
        var got = _encode_target(yv, rows, text_y, label_list, est, fields, context, b)
        encoded = got[0]
        outputs = got[1]
    # One admission: with the labels for a search over a classifier, else
    # deferred (a class_weight tail is completed over the encode below).
    var owner: MemberOwner
    var admitted_labels = PythonObject(None)
    if early:
        admitted_labels = encoded[0] if encoded is not None else outputs[1]
        owner = member_admit_serving_labeled(est, fields, plan, member_context, b, needs, admitted_labels)
    else:
        admitted_labels = Python.list()  # what the deferred admission hands its family
        owner = member_admit_serving_deferred(est, fields, plan, member_context, b, needs)
    if not owner.published:
        raise Error(_admission_refusal(est, fields, plan, member_context, b, needs, admitted_labels, context))
    _ = py_call(pins.append, owner.pins)
    var kind_mask = member_kind_capabilities(owner.plan)  # raises its own words
    if (kind_mask & MEMBER_CAN_FIT) == 0:
        raise Error(_no_member_fit_words(est))  # (the export is the admission's `needs`, as before)
    var classifier = (kind_mask & MEMBER_CAN_CLASSIFIER) != 0
    if steps > 0 and (kind_mask & MEMBER_CAN_STEPS) == 0:
        # a 3-D X for a kind that reads matrices: the product's rank words, as
        # the X admission raised them for it before (lane w3f5-c)
        raise Error(_fit_words(est, fields, "mojolearn: X must be 2-D, got 3-D shape " + sequence_shape, context, b))
    var run = _FitOne(owner.plan.copy(), msel_addr(xv), rows, cols)
    run.steps = steps
    if sample_weight is not None:
        # LANE w3f3-b: an owner's fit with sample_weight (`msel_member_fit_weighted`)
        if (kind_mask & MEMBER_CAN_SAMPLE_WEIGHT) == 0:
            raise Error("NotImplementedError: " + String(py=py_type(est).__name__) + "'s native member reads no "
                        + "sample_weight, so another estimator that fits it cannot hand one. Fit "
                        + String(py=py_type(est).__name__) + " on its own.")
        var w = admit_weight_words(sample_weight, rows, member_context)
        _ = py_call(pins.append, w.array)
        run.weights = w.address
    var gil_free = (kind_mask & MEMBER_GIL_FREE) != 0
    var gil_door = not gil_free and (kind_mask & MEMBER_GIL_DOOR) != 0
    if (kind_mask & MEMBER_CAN_SCOPE) != 0:
        # THE RESIDENT SCOPE, read once; a read that raises raises its
        # doors' words.
        run.scope = member_scope_word(context["family_msel_rows_context"])
    if (kind_mask & MEMBER_CAN_TEST_ROWS) == 0:
        var finite = False
        var held_x = admitted_held_index(xv)
        if held_x >= 0:
            finite = admitted_held_verify(held_x)
        else:
            finite = host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=run.x),
                                         rows * cols)
        _ = xv  # keep the Python object alive through the scan above
        if not finite:
            # check_array's words, as the base estimator's own fit raises them
            raise Error(_fit_words(est, fields, admitted_finite_wording(_base_name(est, fields, context), "X"),
                                   context, b))
    var labels = PythonObject(None)
    if classifier and yv is None:
        # validate_data's words (sklearn 1.9 utils/validation.py), raised by
        # every fit of a classifier without a target
        raise Error(_fit_words(est, fields, "ValueError: This " + _base_name(est, fields, context)
                               + " estimator requires y to be passed, but the target y is None.", context, b))
    if classifier:
        if encoded is None and outputs is None:
            var got = _encode_target(yv, rows, text_y, label_list, est, fields, context, b)
            encoded = got[0]
            outputs = got[1]
        if encoded is not None:
            var classes = encoded[0]
            var codes = encoded[1]
            if Int(py=codes.size) != rows:
                raise Error(_invariant("the label encoder answers other than one code per row"))
            # A kind that reads -1 in y as unlabeled (LabelPropagation,
            # LabelSpreading): the marker's code in its word (lane mf-meta5).
            if xneighbors_kind_reads_marker(run.plan.family, run.plan.kind):
                var marker = xneighbors_marker_index(classes)
                if marker >= 0:
                    run.plan.params[LP_MARKER_WORD] = Float64(marker)
            # One class is the member fit's to take or refuse in its words
            # (lane mf-r9a: it kept the caller's door before).
            run.plan.classes = len(classes)
            _ = py_call(pins.append, codes)
            run.y = msel_addr(codes)
            labels = _public_labels(classes, member_context)
            # CLASS WEIGHTS over the encode's classes (lane mf-left4): a
            # dict the estimator refuses raises its own words here.
            if not member_cw_fill_plan(run.plan, est, fields, classes, member_context, b):
                raise Error(_no_member_fit_words(est))  # a family that reads no class_weight dict natively
        else:
            # THE 2-D y: rows x outputs codes, each output's class count.
            var per = outputs[1]
            run.plan.classes = 0
            run.plan.class_counts = List[Int]()
            labels = Python.list()
            for j in range(len(per)):  # driver-loop: over the outputs (their counts and public labels)
                run.plan.class_counts.append(len(per[j]))
                _ = py_call(labels.append, _public_labels(per[j], member_context))
            _ = py_call(pins.append, outputs[0])
            run.y = msel_addr(outputs[0])
            run.y_cols = len(per)
        run.y_dtype = MEMBER_I32
    elif yv is not None:
        var target = PythonObject(None)
        # THE 2-D y OF A REGRESSOR (lane w3f3-k; layout 4): a kind that takes
        # a multi-output target (a search whose every candidate does, as
        # scikit-learn's KNeighborsRegressor and the multi-output members do)
        # reads rows x outputs float32 words, admitted once as the target
        # check admits one column (real, finite, C order; the target check's
        # NaN words). Before, every such fit refused "currently requires one
        # target" here. A one-column y keeps the target check (raveled).
        var y_out = 1
        var two_d = ((kind_mask & MEMBER_CAN_MULTI_OUTPUT) != 0 and msel_is_array(yv, context)
                     and Int(py=yv.ndim) == 2 and Int(py=yv.shape[1]) > 1)
        try:
            if two_d:
                var admitted_y = admit_f32(yv, 2, "y", member_context, as_array=True)
                target = admitted_y.array
                y_out = admitted_y.cols
            else:
                target = classical_check_y(yv, rows, classical_estimator_name(est), member_context)
        except error:
            # the target check's words, raised by every fit of a search
            var words = String(error)
            if two_d and words == "Input y contains NaN or infinity":
                words = "Input y contains NaN or infinity (" + classical_estimator_name(est) + ")."
            raise Error(_fit_words(est, fields, words, context, b))
        _ = py_call(pins.append, target)
        run.y = msel_addr(target)
        run.y_cols = y_out
        if (kind_mask & MEMBER_READS_Y_GIVEN) != 0:
            # THE GIVEN y (lane mf-tail5): y's own words (a view; `yv` lives
            # to PACK), for a kind whose stratified splitter codes them.
            run.given = member_given_y(yv, rows, y_out, context["array_class"])
    # THE PIN RULE: one reference on X's own object for the fit, taken with
    # the interpreter held; what the fit's cache did not take comes back.
    if run.scope >= 0:
        run.pin = member_pin_take(xv, 1)
    var address = Int(UnsafePointer(to=run))
    var py = Python()
    try:
        if gil_free or gil_door:
            with GILReleased(py):
                _fit_one(address)
        else:
            # THE GIL RULE: a kind without either bit fits with the GIL held.
            _fit_one(address)
    finally:
        if run.pin != 0:
            member_pin_return(run.pin, 1 - run.taken)
            run.pin = 0
        _ = pins
        _ = xv
        _ = yv
    if run.failed:
        raise Error(run.error.copy())  # the member fit's own words (its state was released)
    # PACK: the fitted state into the estimator, held by it. A pack the
    # family refuses releases the state and raises its words.
    try:
        fields = py_call(plan["fields_get"], est, py_type(est))
        if classifier:
            member_pack_labeled(est, fields, member_context, b, run.plan, run.state, labels)
        else:
            member_pack(est, fields, member_context, b, run.plan, run.state)
        var state = run.state
        run.state = 0
        if not member_held_adopt(est, plan, member_context, owner.pins, run.plan, state):
            var call = MemberCall(run.plan.route, state)
            member_free(run.plan, call)
    except error:
        if run.state != 0:
            var call = MemberCall(run.plan.route, run.state)
            member_free(run.plan, call)
            run.state = 0
        raise error^
    # THE LIFETIME RULE: X and the admitted target were handed to the fit
    # and are read again by a borrowing kind's export at PACK: they live to
    # here.
    _ = pins
    _ = xv
    _ = yv
    member_fit_notices_raise(run.notices, member_context)
    return est
