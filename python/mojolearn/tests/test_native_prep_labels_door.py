# SPDX-License-Identifier: Apache-2.0
"""LabelEncoder, LabelBinarizer and MultiLabelBinarizer run each public
method as ONE native call (bindings/prep_labels_native.mojo through the
family door bindings/prep_owner_native.mojo). The source checks run
without bindings; the oracle checks compare every public method's output
bytes and fitted attributes with the former Python routes
(tests/_former_prep_labels.py) and need the built bindings."""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
PREP = ROOT / "python" / "mojolearn" / "_expansion_prep.py"
OWNER = ROOT / "bindings" / "prep_labels_native.mojo"
DOOR = ROOT / "bindings" / "prep_owner_native.mojo"

CLASSES = {
    "LabelEncoder": ("label_encoder", {"fit": "(y,)", "fit_transform": "(y,)", "transform": "(y,)",
                                       "inverse_transform": "(y,)"}),
    "LabelBinarizer": ("label_binarizer", {"fit": "(y,)", "fit_transform": "(y,)", "transform": "(y,)",
                                           "inverse_transform": "(Y, threshold)"}),
    "MultiLabelBinarizer": ("multilabel_binarizer", {"fit": "(y,)", "fit_transform": "(y,)",
                                                     "transform": "(y,)", "inverse_transform": "(yt,)"}),
}


def _class_nodes():
    tree = ast.parse(PREP.read_text())
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


# ---------------------------------------------------------------- source (no bindings)
@pytest.mark.parametrize("cls", sorted(CLASSES))
def test_pure_each_method_is_one_encode_call(cls):
    node = _class_nodes()[cls]
    methods = {f.name: f for f in node.body if isinstance(f, ast.FunctionDef)}
    for name, args in CLASSES[cls][1].items():
        body = [s for s in methods[name].body
                if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        assert len(body) == 1 and isinstance(body[0], ast.Return), (cls, name)
        call = body[0].value
        assert isinstance(call, ast.Call) and ast.unparse(call.func) == "_ENCODE_CALL", (cls, name)
        assert ast.unparse(call.args[0]) == "self"
        assert ast.literal_eval(call.args[1]) == name
        assert ast.unparse(call.args[2]) == args
        assert ast.unparse(call.args[3]) == "_ENCODE_CONTEXT"


def test_pure_owner_kinds_are_connected():
    text = PREP.read_text()
    for cls, (kind, _) in CLASSES.items():
        assert f'{cls}: "{kind}"' in text
    assert '"x_prep_label_present"' in (ROOT / "python" / "mojolearn" / "_native_prep_encode_bridge.py").read_text()


def test_pure_native_owner_dispatches_every_operation():
    owner = OWNER.read_text()
    door = DOOR.read_text()
    assert "prep_labels_call(model, kind, operation, args, context)" in door
    for cls, (kind, ops) in CLASSES.items():
        assert f'kind == "{kind}"' in owner
        for op in ops:
            assert f'operation == "{op}"' in owner
    # the x_prep stages the former Python routes staged
    for stage in ("OP_LAB_LOAD = 119", "OP_SORT_COLS = 0", "OP_UNIQ_COUNT = 120", "OP_UNIQ_SCAN = 121",
                  "OP_UNIQ_WRITE = 122", "OP_CHUNK_NEG = 123", "OP_LOOKUP = 7", "OP_F2I = 11",
                  "OP_LABEL_BINARIZE = 51", "OP_SCATTER_ONES = 52", "OP_ROW_ONES = 135",
                  "OP_INDICATOR = 87", "OP_BLOCK_ARGMAX = 76", "OP_F2_CODE_GATHER = 241",
                  "OP_C2_BIN_CODE = 242", "OP_CAT_ZERO = 157", "OP_CAT_PRESENT = 158",
                  "OP_PRES_COUNT = 159", "OP_PRES_WRITE = 160", "OP_COUNT_NEG = 8"):
        assert stage in owner, stage


def test_pure_former_routes_left_the_package():
    text = PREP.read_text()
    for name in ("_label_fit_device", "_label_codes_device", "_mlb_flat", "_multilabel_indicator",
                 "_label_classes", "_stage_class_gather", "_inverse_device", "_transform_flat"):
        assert f"def {name}(" not in text, name


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


def _outcome(fn):
    try:
        return ("ok", _bytes(fn()))
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


def _compare(new, old, calls):
    for name, args in calls:
        a = _outcome(lambda: getattr(new, name)(*args))
        b = _outcome(lambda: getattr(old, name)(*args))
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
