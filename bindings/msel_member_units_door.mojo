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

THE FALLBACK RULE (MEMBER-ABI-V2.md): a member mark never turns a call that
worked into a refusal. A class this door cannot serve (no member, an
admission the family refuses, a target the check refuses, labels outside the
CLASSIFIER rule, a failing fit) answers None / False having changed nothing:
the caller fits through the estimator's own door, which raises its own words.

CLASSIFIER RULE: y as long as X with at least two classes per output; float
labels whole finite numbers; a 2-D y of a numeric dtype in C order (labels of
another dtype keep the caller's door: a column of Python label objects is
not read here).

WHO REFUSES A NON-FINITE X: a member whose KIND has `MEMBER_CAN_TEST_ROWS`
refuses non-finite rows in its own fit; for any other X's verdict is read
once (the held block's recorded verdict, else one scan across the host
cores) and a non-finite X keeps the caller's door.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.classical_validate_unpack import classical_check_y, classical_estimator_name
from providers.python.member_xneighbors_boundary_door import xneighbors_kind_reads_marker, xneighbors_marker_index
from core.xneighbors_member import LP_MARKER_WORD
from bindings.member_boundary_door import (
    member_admit_serving_deferred, member_admit_serving_labeled, member_pack, member_pack_labeled,
)
from bindings.member_class_weight_unpack import member_cw_fill_plan
from bindings.member_owner_unpack import MemberOwner
from bindings.member_held_door import member_held_adopt
from bindings.member_fit_notices_pack import member_fit_notices_raise
from core.member_notices import member_notices_take
from bindings.member_scope_door import member_scope_word, member_pin_take, member_pin_return
from bindings.classifier_public_lifecycle_door import _public_labels
from bindings.msel_common_door import msel_is_array, msel_addr, msel_encode, msel_new
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
    MEMBER_READS_Y_GIVEN,
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
    # The target: 0 (a fit without one), float32 values, or int32 codes
    # (rows x `y_cols` words of `y_dtype`).
    var y: Int
    var y_dtype: Int
    var y_cols: Int
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
        self.y = 0
        self.y_dtype = MEMBER_F32
        self.y_cols = 1
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
        if run[].y != 0:
            call.y = MemberMatrix(run[].y, run[].rows, run[].y_cols, run[].y_dtype)
            call.y_given = run[].given
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


def _whole_classes(classes: PythonObject, b: PythonObject) raises -> Bool:
    """False when a class label is a float that is not a whole finite number
    (a continuous target, which a classifier's own fit may refuse). Int, bool
    and text labels pass. One read per CLASS, never per row."""
    for i in range(len(classes)):  # driver-loop: over the distinct classes
        var label = classes[i]
        if py_type(label) is b.float:
            var value = Float64(py=label)
            if not (value - value == 0.0) or value > 9.0e15 or value < -9.0e15 or Float64(Int(value)) != value:
                return False
    return True


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


def _encode_outputs(y: PythonObject, rows: Int, context: PythonObject, b: PythonObject) raises -> PythonObject:
    """THE 2-D y of a classifier (layout 4), PER OUTPUT: each column of a C
    order numeric rows x outputs y copied once into its own vector (its
    elements' bytes, no conversion) and encoded by the one label encoder.
    Returns (int32 rows x outputs codes, [classes of each output]), or None
    when y is not one this door reads (the caller's door then fits it)."""
    if Int(py=y.ndim) != 2 or Int(py=y.shape[0]) != rows or String(py=y.order) != "C":
        return PythonObject(None)
    var outputs = Int(py=y.shape[1])
    var dtype = String(py=y.dtype)
    var width = _numeric_width(dtype)
    if outputs < 2 or width == 0 or rows == 0:
        return PythonObject(None)
    var codes = msel_new(rows * outputs, "<i4", context)
    var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=msel_addr(codes))
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
        if len(encoded[0]) < 2 or not _whole_classes(encoded[0], b):
            return PythonObject(None)
        var column_codes = encoded[1]
        if Int(py=column_codes.size) != rows or String(py=column_codes.dtype) != "<i4":
            return PythonObject(None)
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


def _encode_text_outputs(y: PythonObject, rows: Int, context: PythonObject, b: PythonObject) raises -> PythonObject:
    """THE TEXT 2-D y of a classifier (lane mf-tail5b), PER OUTPUT: its rows
    read ONCE at UNPACK (boundary work: the labels are Python objects), each
    output's labels gathered into one list and encoded by the one label
    encoder (`msel_encode`, the 1-D text route's), written as int32 rows x
    outputs codes. Returns (codes, [classes of each output]) as
    `_encode_outputs` does, its classes kept for PACK; None when y is not
    such a target or holds no string label (a numeric y takes
    `_encode_outputs`)."""
    var listed = py_call(b.list, y)
    if len(listed) != rows or rows == 0:
        return PythonObject(None)
    var first = py_call(b.list, listed[0])
    var outputs = len(first)
    if outputs < 2:
        return PythonObject(None)
    var columns = Python.list()
    for j in range(outputs):  # driver-loop: over the outputs (one label list each)
        _ = py_call(columns.append, Python.list())
    var text = False
    for i in range(rows):  # driver-loop: over the rows (each row's labels, read once)
        var row = py_call(b.list, listed[i])
        if len(row) != outputs:
            return PythonObject(None)
        for j in range(outputs):  # small-loop(the outputs)
            var label = row[j]
            text = text or Bool(py=py_call(b.isinstance, label, b.str))
            _ = py_call(columns[j].append, label)
    if not text:
        return PythonObject(None)
    var codes = msel_new(rows * outputs, "<i4", context)
    var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=msel_addr(codes))
    var classes = Python.list()
    for j in range(outputs):  # driver-loop: over the outputs (one encode each)
        var encoded = msel_encode(columns[j], context)
        if len(encoded[0]) < 2:
            return PythonObject(None)
        var column_codes = encoded[1]
        if Int(py=column_codes.size) != rows or String(py=column_codes.dtype) != "<i4":
            return PythonObject(None)
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
    estimator as a member imports nothing. None when this member, data or
    fit is not one the route runs: the caller then fits the estimator through
    its own door, which raises its own exception (THE FALLBACK RULE). y
    None: a fit without a target (an unsupervised search)."""
    return _member_fit(base, parameters, True, False, y, x, context, b)


def msel_member_fit_in_place(model: PythonObject, y: PythonObject, x: PythonObject, context: PythonObject,
    b: PythonObject) raises -> Bool:
    """`msel_member_refit` of `model` ITSELF (lane mf-msel5): its own
    instance dictionary is admitted (no clone, so a field only its door set,
    the carrier's `_msel_cv_tool`, is read), fitted once on every row of X
    with the GIL released and packed into it in place. False (nothing
    changed): the route did not take it (THE FALLBACK RULE)."""
    return _member_fit(model, PythonObject(None), False, False, y, x, context, b) is not None


def msel_member_fit_cloned(model: PythonObject, y: PythonObject, x: PythonObject, context: PythonObject,
    b: PythonObject) raises -> Bool:
    """`msel_member_fit_in_place` of a `model` the caller made itself with
    `msel_native_clone` (lane mf-msel5c: the search door's carrier), whose
    parameters that clone already admitted: no second admission clone."""
    return _member_fit(model, PythonObject(None), False, True, y, x, context, b) is not None


def _member_fit(base: PythonObject, parameters: PythonObject, clone: Bool, vetted: Bool, y: PythonObject,
    x: PythonObject, context: PythonObject, b: PythonObject) raises -> PythonObject:
    """UNPACK, RUN and PACK of the one fit (module note): the fitted
    estimator (`base` itself, or its clone with `parameters` set), or None
    having changed nothing."""
    var plan = msel_estimator_plan(base, context)
    if py_get(plan, "member") is None:
        return PythonObject(None)
    # A TEXT 2-D y (lane mf-tail5b): not an Array; a search over a
    # classifier admits it below, per output.
    var text_y = y is not None and not msel_is_array(y, context) and msel_text_rows(y, context, b)
    if not msel_is_array(x, context) or (y is not None and not msel_is_array(y, context) and not text_y):
        return PythonObject(None)
    if String(py=x.dtype) != "<f4" or Int(py=x.ndim) != 2 or String(py=x.order) != "C" or Int(py=x.size) == 0:
        return PythonObject(None)
    var member_context = plan["context"]
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
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
    if y is not None and _is_search(est, context):
        # A SEARCH over a classifier: its admission reads the labels.
        var search_base = py_get(fields, "estimator")
        if search_base is not None:
            try:
                early = Bool(py=msel_estimator_plan(search_base, context)["classifier"])
            except:
                return PythonObject(None)  # a base no plan holds: the search door refuses it in its words
    if text_y and not early:
        return PythonObject(None)  # a text 2-D y is read by a search over a classifier only
    if early:
        try:
            if text_y:
                outputs = _encode_text_outputs(y, rows, context, b)
                if outputs is None:
                    return PythonObject(None)
            elif Int(py=y.ndim) == 1:
                if Int(py=y.size) != rows:
                    return PythonObject(None)
                encoded = msel_encode(y, context)
            else:
                outputs = _encode_outputs(y, rows, context, b)
                if outputs is None:
                    return PythonObject(None)
        except:
            return PythonObject(None)
    # One admission: with the labels for a search over a classifier, else
    # deferred (a class_weight tail is completed over the encode below).
    var owner: MemberOwner
    if early:
        owner = member_admit_serving_labeled(est, fields, plan, member_context, b, needs,
                                             encoded[0] if encoded is not None else outputs[1])
    else:
        owner = member_admit_serving_deferred(est, fields, plan, member_context, b, needs)
    if not owner.published:
        return PythonObject(None)
    _ = py_call(pins.append, owner.pins)
    var kind_mask = 0
    try:
        kind_mask = member_kind_capabilities(owner.plan)
    except:
        return PythonObject(None)
    if (kind_mask & MEMBER_CAN_FIT) == 0:
        return PythonObject(None)  # (the export is the admission's `needs`, as before)
    var classifier = (kind_mask & MEMBER_CAN_CLASSIFIER) != 0
    var run = _FitOne(owner.plan.copy(), msel_addr(x), rows, cols)
    var gil_free = (kind_mask & MEMBER_GIL_FREE) != 0
    var gil_door = not gil_free and (kind_mask & MEMBER_GIL_DOOR) != 0
    if (kind_mask & MEMBER_CAN_SCOPE) != 0:
        # THE RESIDENT SCOPE, read once; a read that raises is the caller's
        # door's to refuse.
        try:
            run.scope = member_scope_word(context["family_msel_rows_context"])
        except:
            return PythonObject(None)
    if (kind_mask & MEMBER_CAN_TEST_ROWS) == 0:
        var finite = False
        try:
            var held_x = admitted_held_index(x)
            if held_x >= 0:
                finite = admitted_held_verify(held_x)
            else:
                finite = host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=run.x),
                                             rows * cols)
            _ = x  # keep the Python object alive through the scan above
        except:
            finite = False
        if not finite:
            return PythonObject(None)
    var labels = PythonObject(None)
    if classifier and y is None:
        return PythonObject(None)  # a classifier fits class codes
    if classifier:
        try:
            if encoded is None and outputs is None:
                if Int(py=y.ndim) == 1:
                    if Int(py=y.size) != rows:
                        return PythonObject(None)
                    encoded = msel_encode(y, context)
                else:
                    outputs = _encode_outputs(y, rows, context, b)
                    if outputs is None:
                        return PythonObject(None)
            if encoded is not None:
                var classes = encoded[0]
                if len(classes) < 2 or not _whole_classes(classes, b):
                    return PythonObject(None)
                var codes = encoded[1]
                if Int(py=codes.size) != rows:
                    return PythonObject(None)
                # A kind that reads -1 in y as unlabeled (LabelPropagation,
                # LabelSpreading): the marker's code in its word (lane mf-meta5).
                if xneighbors_kind_reads_marker(run.plan.family, run.plan.kind):
                    var marker = xneighbors_marker_index(classes)
                    if marker >= 0:
                        run.plan.params[LP_MARKER_WORD] = Float64(marker)
                run.plan.classes = len(classes)
                _ = py_call(pins.append, codes)
                run.y = msel_addr(codes)
                labels = _public_labels(classes, member_context)
                # CLASS WEIGHTS over the encode's classes (lane mf-left4): a
                # dict the estimator refuses raises its own words here.
                if not member_cw_fill_plan(run.plan, est, fields, classes, member_context, b):
                    return PythonObject(None)
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
        except:
            return PythonObject(None)
        run.y_dtype = MEMBER_I32
    elif y is not None:
        try:
            var target = classical_check_y(y, rows, classical_estimator_name(est), member_context)
            _ = py_call(pins.append, target)
            run.y = msel_addr(target)
            if (kind_mask & MEMBER_READS_Y_GIVEN) != 0:
                # THE GIVEN y (lane mf-tail5): y's own words (a view; `y` lives
                # to PACK), for a kind whose stratified splitter codes them.
                run.given = member_given_y(y, rows, 1, context["array_class"])
        except:
            return PythonObject(None)
    # THE PIN RULE: one reference on X's own object for the fit, taken with
    # the interpreter held; what the fit's cache did not take comes back.
    if run.scope >= 0:
        run.pin = member_pin_take(x, 1)
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
        _ = x
        _ = y
    if run.failed:
        return PythonObject(None)
    # PACK: the fitted state into the estimator, held by it. A pack the
    # family refuses publishes nothing: the caller's door fits it.
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
    except:
        if run.state != 0:
            var call = MemberCall(run.plan.route, run.state)
            member_free(run.plan, call)
            run.state = 0
        return PythonObject(None)
    # THE LIFETIME RULE: X and the admitted target were handed to the fit
    # and are read again by a borrowing kind's export at PACK: they live to
    # here.
    _ = pins
    _ = x
    _ = y
    member_fit_notices_raise(run.notices, member_context)
    return est
