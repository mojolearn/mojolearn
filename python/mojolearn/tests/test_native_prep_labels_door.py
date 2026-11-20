# SPDX-License-Identifier: Apache-2.0
"""LabelEncoder, LabelBinarizer and MultiLabelBinarizer run each public
method as ONE native call (bindings/prep_labels_native_door.mojo through the
family door bindings/prep_owner_native_door.mojo). The runtime audit checks
that each public method runs with no Python frame and no wrapper work; the
oracle checks compare every public method's output bytes and fitted
attributes with the former Python routes (tests/_former_prep_labels.py).
Both need the built bindings."""
import pytest

# Deleted test_pure_native_owner_dispatches_every_operation: a source-text check
# of bindings/prep_owner_native.mojo, which fed1b4e384 renamed to
# prep_owner_native_door.mojo (and which no installed wheel carries). Every
# public operation of the three classes is checked against the former route
# by the oracle tests below; no Mojo check covers the dispatch text.


# ---------------------------------------------------------------- oracle (bindings: RUN OWED)
def _bytes(value):
    if hasattr(value, "tobytes") and hasattr(value, "dtype"):
        return (str(value.dtype), tuple(value.shape), bytes(value.tobytes()))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, [_bytes(v) for v in value])
    return (type(value).__name__, value)


def _state(est):
    keep = ("_classes", "classes_", "y_type_", "numeric_mode_", "_given")
    out = {k: _bytes(getattr(est, k)) for k in keep if hasattr(est, k)}
    cats = getattr(est, "_cats", None)
    out["_cats"] = None if cats is None else _bytes(cats)
    return out


def _outcome(fn, owner=None):
    try:
        result = fn()
        # fit returns the estimator: compare "returned self", not two objects
        return ("ok", "self") if owner is not None and result is owner else ("ok", _bytes(result))
    except Exception as error:  # noqa: BLE001 - the refusal class and text are the result
        return ("raised", type(error).__name__, str(error))


@pytest.fixture(scope="module")
def pair():
    np = pytest.importorskip("numpy")
    mojolearn = pytest.importorskip("mojolearn")
    try:
        mojolearn.LabelEncoder().fit(np.array([1, 2]))
    except Exception as error:  # noqa: BLE001 - no built bindings here
        pytest.skip(f"bindings unavailable: {error}")
    from mojolearn.tests import _former_prep_labels as former
    return np, mojolearn, former


# ---------------------------------------------------------------- runtime audit (bindings)
# Replaces the former source checks (an ast parse of _expansion_prep.py) with
# the behaviour they stood for: every public method of the three classes runs
# as one native operation, with no Python frame and no wrapper work.
def _audit_calls(np):
    y = np.array([10, 30, 20, 10], dtype=np.int64)
    sets = [[1, 2], [3], [2, 3]]
    return {
        "LabelEncoder": (y, [("fit", (y,)), ("transform", (y,)), ("inverse_transform", (np.array([0, 1]),)),
                             ("fit_transform", (y,))]),
        "LabelBinarizer": (y, [("fit", (y,)), ("transform", (y,)),
                               ("inverse_transform", (np.eye(3, dtype=np.float32), 0.5)), ("fit_transform", (y,))]),
        "MultiLabelBinarizer": (sets, [("fit", (sets,)), ("transform", (sets,)),
                                       ("inverse_transform", (np.array([[1, 0, 1], [0, 1, 0]]),)),
                                       ("fit_transform", (sets,))]),
    }


@pytest.mark.parametrize("cls", ["LabelEncoder", "LabelBinarizer", "MultiLabelBinarizer"])
def test_each_public_method_is_one_native_operation(pair, cls):
    from mojolearn._verify_native_runtime import audit_native_call
    np, mojolearn, _ = pair
    y, calls = _audit_calls(np)[cls]
    est = getattr(mojolearn, cls)()
    est.fit(y)
    for name, args in calls:
        audit_native_call(getattr(est, name), *args).assert_native()


def _compare(new, old, calls):
    for name, args in calls:
        a = _outcome(lambda: getattr(new, name)(*args), new)
        b = _outcome(lambda: getattr(old, name)(*args), old)
        assert a == b, (type(new).__name__, name)
        assert _state(new) == _state(old), (type(new).__name__, name)


def _label_inputs(np):
    return [
        np.array([10, 30, 20, 10], dtype=np.int64),
        np.array([3, 1, 2, 1], dtype=np.int32),
        np.array([0.5, -1.25, 0.5, 2.0], dtype=np.float32),
        np.array([0.5, -1.25, 0.5, 2.0], dtype=np.float64),
        np.array([4000, 7, 4095, 7], dtype=np.int64),      # the presence route's range
        np.array([2 ** 40, 5, 2 ** 40], dtype=np.int64),   # no exact float32 word: the input step
        np.arange(12, dtype=np.int64)[::3],                # a strided buffer
        [3, 1, 2, 3],
        [1.5, 0.5, 1.5],
        ["b", "a", "c", "a"],
        ("x", "y"),
        [True, False, True],
    ]


def test_label_encoder_matches_former_route(pair):
    np, mojolearn, former = pair
    for y in _label_inputs(np):
        new, old = mojolearn.LabelEncoder(), former.LabelEncoder()
        _compare(new, old, [("fit", (y,)), ("transform", (y,)), ("inverse_transform", ([0, 1],)),
                            ("transform", (np.array([999]),)), ("transform", (["zz"],)),
                            ("inverse_transform", (np.array([0, 7]),)), ("inverse_transform", ([-1],))])
        new, old = mojolearn.LabelEncoder(), former.LabelEncoder()
        _compare(new, old, [("fit_transform", (y,)), ("inverse_transform", (np.array([0], dtype=np.int64),))])
    _compare(mojolearn.LabelEncoder(), former.LabelEncoder(), [("transform", ([1],)), ("fit", ([],)),
                                                               ("fit", ([1, "a"],)), ("fit", ([float("nan")],))])


def test_label_binarizer_matches_former_route(pair):
    np, mojolearn, former = pair
    for kw in ({}, {"neg_label": -1, "pos_label": 2}):
        for y in _label_inputs(np) + [np.array([7, 7, 7]), ["only", "only"]]:
            new, old = mojolearn.LabelBinarizer(**kw), former.LabelBinarizer(**kw)
            _compare(new, old, [("fit", (y,)), ("transform", (y,)), ("transform", (np.array([12345]),)),
                                ("transform", (["unseen"],))])
            try:
                Y = old.transform(y)
            except Exception:  # noqa: BLE001 - the refusal was compared above
                continue
            _compare(new, old, [("inverse_transform", (Y,)), ("inverse_transform", (Y, 0.25))])
            new, old = mojolearn.LabelBinarizer(**kw), former.LabelBinarizer(**kw)
            _compare(new, old, [("fit_transform", (y,))])
    ind = np.array([[1, 0, 1], [0, 0, 1], [1, 1, 0]])
    for y in (ind, ind.tolist(), np.asfortranarray(ind.astype(np.float64))):
        new, old = mojolearn.LabelBinarizer(), former.LabelBinarizer()
        _compare(new, old, [("fit", (y,)), ("transform", (y,)), ("inverse_transform", (ind.astype(np.float32),)),
                            ("transform", (np.array([1, 2]),)), ("transform", (np.array([[1, 0], [0, 1]]),))])
    for bad in (np.array([[1, 2, 3], [0, 1, 2]]), [[1, 0], [1]], np.array([[True, False], [False, True]])):
        _compare(mojolearn.LabelBinarizer(), former.LabelBinarizer(), [("fit", (bad,))])
    for kw in ({"sparse_output": True}, {"neg_label": 1, "pos_label": 0}, {"neg_label": 0.5}):
        _compare(mojolearn.LabelBinarizer(**kw), former.LabelBinarizer(**kw), [("fit", ([1, 2],))])
    _compare(mojolearn.LabelBinarizer(), former.LabelBinarizer(), [("transform", ([1],))])


def test_multilabel_binarizer_matches_former_route(pair):
    np, mojolearn, former = pair
    inputs = [
        [[1, 2], [3], [2, 3]],
        [(4, 1), {9, 1}, frozenset({2})],
        [[1, 2], [], [2 ** 40]],
        [["b", "a"], ["c"], []],
        [[1.5], [0.5, 1.5]],
        [[True], [False]],
    ]
    for y in inputs:
        for kw in ({}, {"classes": [3, 2, 1, 9, "b", "a", "c"][:4]}):
            new, old = mojolearn.MultiLabelBinarizer(**kw), former.MultiLabelBinarizer(**kw)
            _compare(new, old, [("fit", (y,)), ("transform", (y,)), ("transform", ([[777], [1]],))])
            try:
                Yt = old.transform(y)
            except Exception:  # noqa: BLE001 - the refusal was compared above
                continue
            _compare(new, old, [("inverse_transform", (Yt,)), ("inverse_transform", (Yt.tolist(),))])
            new, old = mojolearn.MultiLabelBinarizer(**kw), former.MultiLabelBinarizer(**kw)
            _compare(new, old, [("fit_transform", (y,))])
    _compare(mojolearn.MultiLabelBinarizer(sparse_output=True), former.MultiLabelBinarizer(sparse_output=True),
             [("fit", ([[1]],))])
    _compare(mojolearn.MultiLabelBinarizer(), former.MultiLabelBinarizer(), [("transform", ([[1]],))])
