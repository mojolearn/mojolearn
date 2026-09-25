"""lane/python-hotpath (DEVIATIONS 3100-3107): every compiled or C-speed
seam answers EXACTLY as the Python routine it stands in for.

The Python routines are the definitions and stay in the package as the
fallback. `MOJOLEARN_HOTPATH=python` sends every seam down its Python
routine (the REFERENCE arm); unset, the seam takes the core helper of
`bindings/hotpath_helpers.mojo` or the C builtin (the NEW arm). Each case
runs both arms and holds them to the SAME RESULT, byte for byte, OR THE
SAME REFUSAL, type and words.

A comparison of a routine with itself passes and proves nothing, so every
case PINS WHICH PATH ANSWERED: the helpers a case expects are wrapped with a
counter in `_buffer._NATIVE`, the new arm must call them and the reference
arm must not. A build with `-D MOJOLEARN_HOST_SABOTAGE=1` answers wrong on
purpose in every helper; this file run against it (with
MOJOLEARN_HOST_ALLOW_SABOTAGE=1 and MOJOLEARN_HOTPATH_EXPECT_SABOTAGE=1)
must see every group diverge, which `test_sabotage_build_diverges` asserts.

NumPy is the input factory here and nothing else. Skips when the loaded
binary has no hotpath helpers.
"""
import math
import os
import struct
import warnings

import numpy as np
import pytest

from mojolearn import _array, _buffer, _labels, _metrics_impl as M
from mojolearn import model_selection as MS
from mojolearn._array import Array

_HELPERS = (
    "cast_elements", "reduce_stat", "equal_elements", "gather_i32",
    "check_indices_i64", "indices_overlap_i64", "fold_ids", "select_fold_i64",
    "encode_labels_i64", "encode_labels_f64", "encode_labels_i32",
    "encode_labels_u8", "encode_labels_f32", "encode_labels_u32",
    "gather_i64", "gather_f64",
)
os.environ.pop("MOJOLEARN_HOTPATH", None)
try:
    for _key in _HELPERS:
        _buffer._native(_key)
except ImportError as exc:  # pragma: no cover - build state
    pytest.skip(f"binary without the hotpath helpers: {exc}", allow_module_level=True)

_EXPECT_SABOTAGE = os.environ.get("MOJOLEARN_HOTPATH_EXPECT_SABOTAGE") == "1"
_RNG = np.random.default_rng(3100)
_BIG = 70_000  # above the helpers' one-task threshold (65,536)


# ----------------------------------------------------------------- harness


def _canon(value):
    """A value as something `==` compares EXACTLY: Array bytes and layout,
    float bits (so -0.0, 0.0 and NaN payloads are told apart), types."""
    if isinstance(value, Array):
        return ("Array", value.shape, value.dtype, value.order, value.tobytes())
    if isinstance(value, M._EncodedLabels):
        return ("list", [_canon(v) for v in value])  # it stands for that list
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, float):
        return ("float", struct.pack("<d", value))
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, [_canon(v) for v in value])
    if isinstance(value, dict):
        return ("dict", sorted((k, _canon(v)) for k, v in value.items()))
    return (type(value).__name__, value)


class _Spy:
    def __init__(self):
        self.calls = {}

    def __enter__(self):
        self.saved = dict(_buffer._NATIVE)
        for key in _HELPERS:
            real = _buffer._native(key)

            def wrapped(*args, _real=real, _key=key):
                self.calls[_key] = self.calls.get(_key, 0) + 1
                return _real(*args)

            _buffer._NATIVE[key] = wrapped
        return self

    def __exit__(self, *exc):
        _buffer._NATIVE.clear()
        _buffer._NATIVE.update(self.saved)
        return False


def _outcome(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = ("ok", _canon(fn()))
        except Exception as exc:  # the refusal IS the result
            result = ("refused", type(exc).__name__, str(exc))
    return result, [(w.category.__name__, str(w.message)) for w in caught]


def _both(fn, expect=()):
    """Run `fn` on the reference arm and the new arm. `expect` names the
    helpers the new arm must call; () means the new arm is a C builtin (or a
    deliberate fall back) and only the reference arm's silence is pinned."""
    os.environ["MOJOLEARN_HOTPATH"] = "python"
    try:
        with _Spy() as ref_spy:
            ref = _outcome(fn)
    finally:
        os.environ.pop("MOJOLEARN_HOTPATH", None)
    hot = set(ref_spy.calls) - {"gather_i64", "gather_f64", "encode_labels_i64",
                                "encode_labels_f64", "encode_labels_i32",
                                "encode_labels_u8", "encode_labels_f32",
                                "encode_labels_u32"}
    assert not hot, f"the reference arm reached a hotpath helper: {ref_spy.calls}"
    with _Spy() as new_spy:
        new = _outcome(fn)
    for key in expect:
        assert new_spy.calls.get(key), (
            f"the new arm never called {key}: {new_spy.calls}; the comparison "
            "would pass vacuously")
    return ref, new


_DIVERGED = []


def _same(fn, expect=(), group=""):
    ref, new = _both(fn, expect)
    if _EXPECT_SABOTAGE:
        if ref != new:
            _DIVERGED.append(group)
        return ref
    assert new == ref, f"{group}: new arm {new!r:.300} != reference {ref!r:.300}"
    return ref


def _arr(a):
    return Array.from_buffer(a)


# ----------------------------------------------------------------- astype

_NP = {"<f4": np.float32, "<f8": np.float64, "<i4": np.int32, "<i8": np.int64,
       "<u4": np.uint32, "<u1": np.uint8}


def _values_for(src, n, flavor):
    dt = _NP[src]
    if src in ("<f4", "<f8"):
        if flavor == "small":
            a = _RNG.integers(0, 200, n).astype(dt) + dt(0.75)
        elif flavor == "signed":
            a = (_RNG.standard_normal(n) * 1e3).astype(dt)
        elif flavor == "huge":
            a = (_RNG.standard_normal(n) * 1e30).astype(dt)
        else:  # awkward
            a = _RNG.integers(0, 100, n).astype(dt)
            special = [np.nan, np.inf, -np.inf, -0.0, 0.0, 255.0, 255.5, 256.0, -0.5,
                       -1.0, 2147483647.0, 2147483648.0, -2147483648.0, -2147483649.0,
                       4294967295.0, 4294967296.0, 9.2e18, -9.3e18, 1e300, 3.5e38, -3.5e38,
                       9007199254740993.0]
            a[:len(special)] = np.array(special, dtype=np.float64).astype(dt)[:n]
        return a
    info = np.iinfo(dt)
    if flavor == "small":
        return _RNG.integers(0, 200, n).astype(dt)
    if flavor == "signed":
        return _RNG.integers(max(info.min, -1000), min(info.max, 1000), n).astype(dt)
    if flavor == "huge":
        return _RNG.integers(info.min, info.max, n, dtype=dt, endpoint=True)
    a = _RNG.integers(0, 100, n).astype(dt)
    special = [info.min, info.max, 0, 1, 255, 256 % (info.max + 1)]
    if src == "<i8":
        special += [2**53 + 1, -(2**53) - 1, 2**62 + 12345, 2**31, -(2**31) - 1, 2**32]
    a[:len(special)] = np.array(special, dtype=dt)
    return a


@pytest.mark.parametrize("dst", list(_NP))
@pytest.mark.parametrize("src", list(_NP))
def test_astype_matches_the_item_setter(src, dst):
    for flavor in ("small", "signed", "huge", "awkward"):
        for n in (0, 1, 255, 256, 1001, _BIG):
            with np.errstate(all="ignore"):
                a = _arr(_values_for(src, max(n, 64), flavor)[:n].copy())
            native = n >= _array._NATIVE_MIN and src != dst
            _same(lambda: a.astype(dst), ("cast_elements",) if native else (),
                  group="astype")


def test_astype_keeps_layout_and_reads_a_readonly_source():
    block = np.asfortranarray(_RNG.integers(0, 9, (300, 7)).astype(np.int32))
    a = _arr(block)
    assert a.order == "F"
    _same(lambda: a.astype("<f4"), ("cast_elements",), group="astype")
    frozen = _arr(_RNG.integers(0, 9, 5000).astype(np.int64).tobytes())
    assert frozen.dtype == "<u1" and frozen._readonly
    _same(lambda: frozen.astype("<i8"), ("cast_elements",), group="astype")


def test_astype_quiets_a_signaling_nan_as_the_item_setter_does():
    bits = np.full(1000, 0x7FA00001, np.uint32)
    bits[::3] = 0xFFC12345
    bits[1::7] = 0x3F800000
    a = _arr(bits.view(np.float32).copy())
    _same(lambda: a[0:900], ("cast_elements",), group="getitem")
    _same(lambda: a.astype("<f8"), ("cast_elements",), group="astype")


# ------------------------------------------------------------- reductions


def _reduction_inputs():
    out = []
    for dt in _NP.values():
        base = _RNG.integers(0, 50, 4000).astype(dt)
        out.append(base)
        out.append(base[:1])
        out.append(base[:0])
        if np.issubdtype(dt, np.floating):
            for where in (0, 1999, 3999):
                c = (base - dt(25.5)).copy()
                c[where] = np.nan
                out.append(c)
            z = np.zeros(1000, dt)
            z[::2] = -0.0
            out.append(z)
            out.append(-z)
            out.append((_RNG.standard_normal(_BIG) * 1e20).astype(dt))
            c = base.copy()
            c[7] = np.inf
            c[9] = -np.inf
            out.append(c)
        else:
            info = np.iinfo(dt)
            out.append(_RNG.integers(info.min, info.max, 3000, dtype=dt, endpoint=True))
    return out


@pytest.mark.parametrize("what", ["min", "max", "sum", "argmax"])
def test_reductions_match_python(what):
    for raw in _reduction_inputs():
        a = _arr(raw.copy())
        native = (a.size >= _array._NATIVE_MIN
                  and (what != "sum" or a.dtype in ("<f4", "<f8")))
        _same(lambda: getattr(a, what)(), ("reduce_stat",) if native else (),
              group="reduce")
    f = _arr(np.asfortranarray(_RNG.standard_normal((400, 5)).astype(np.float32)))
    _same(lambda: f.argmax(), ("reduce_stat",), group="reduce")


# --------------------------------------------------------------- equality


def test_equality_matches_python():
    for dt in _NP.values():
        x = _RNG.integers(0, 3, 5000).astype(dt)
        y = _RNG.integers(0, 3, 5000).astype(dt)
        if np.issubdtype(dt, np.floating):
            x[::11] = np.nan
            y[::11] = np.nan
            x[1::13] = -0.0
            y[1::13] = 0.0
        a, b = _arr(x), _arr(y)
        _same(lambda: a == b, ("equal_elements",), group="equal")
        _same(lambda: a != b, ("equal_elements",), group="equal")
        _same(lambda: a == a, ("equal_elements",), group="equal")
        _same(lambda: a[:100] == b[:100], (), group="equal")
        _same(lambda: a == 1, (), group="equal")
    a = _arr(_RNG.integers(0, 3, 5000).astype(np.int64))
    b = _arr(_RNG.integers(0, 3, 5000).astype(np.float64))
    _same(lambda: a == b, (), group="equal")  # two dtypes: Python compares exactly
    _same(lambda: a == b[:4999], (), group="equal")  # the shape refusal
    f = _arr(np.asfortranarray(_RNG.integers(0, 3, (300, 9)).astype(np.int32)))
    c = _arr(np.ascontiguousarray(_RNG.integers(0, 3, (300, 9)).astype(np.int32)))
    _same(lambda: f == c, ("equal_elements",), group="equal")


# --------------------------------------------------------------- indexing


def test_getitem_matches_the_per_run_copy():
    blocks = [
        _RNG.integers(0, 99, 5000).astype(np.int64),
        _RNG.standard_normal((700, 6)).astype(np.float32),
        np.asfortranarray(_RNG.standard_normal((700, 6)).astype(np.float64)),
        _RNG.integers(0, 255, (40, 30, 5)).astype(np.uint8),
        _RNG.standard_normal((500, 4)).astype(np.float16),
    ]
    keys = [
        slice(None), slice(10, 600), slice(600, 10, -1), slice(0, 5000, 2), slice(3, 3),
        5, -1, (slice(None), 2), (slice(5, 650), slice(None)), (slice(5, 650), slice(1, 3)),
        (3, slice(None)), (slice(None), slice(None)), (2, 3), (slice(1, 39), slice(None), slice(None)),
        (4, slice(2, 28)), (4, slice(2, 28), slice(None)), (4, 5, slice(1, 4)), 100000,
    ]
    for block in blocks:
        a = _arr(block)
        for key in keys:
            _same(lambda: a[key], (), group="getitem")
    a = _arr(blocks[1])
    ref, new = _both(lambda: a[5:650], ("cast_elements",))
    assert _EXPECT_SABOTAGE or ref == new


# ------------------------------------------------------------- list input

_LISTS = [
    [1, 2, 3], [1.5, 2, True], [True, False], [[1, 2], [3, 4]], [[1.0, 2], [3, 4]],
    ((1, 2), (3, 4)), [(1, 2), [3, 4]], [[], []], [[1, 2], [3]], [[1, 2], 3], [1, [2, 3]],
    [[[1, 2]], [[3, 4]]], [1, "a"], [[1, None]], [None], [2**70, 1], [[2**63], [1]],
    [np.float32(1.5), 2.0], [np.int64(3), 4], [[np.float64(1.0), 2.0]], [1.0, float("nan"), -0.0],
    [], [[]], "abc", [b"x"], [[1.0, 2.0], (3.0, 4.0)], [-0.0, 0.0], [[1, 2.5], [True, 4]],
]


@pytest.mark.parametrize("index", range(len(_LISTS)))
def test_list_input_matches_the_recursive_flatten(index):
    value = _LISTS[index]
    _same(lambda: _array._flatten(value), (), group="lists")
    for convert in (_buffer.as_f32_c, _buffer.as_i64_c, _buffer.as_f64_c):
        for ndim in (1, 2, None):
            _same(lambda: convert(value, ndim=ndim, name="X"), (), group="lists")
    _same(lambda: Array.from_list(value, "<f8"), (), group="lists")


def test_big_list_input():
    rows = _RNG.standard_normal((3000, 7)).tolist()
    _same(lambda: _buffer.as_f32_c(rows, ndim=2, name="X"), (), group="lists")
    ints = _RNG.integers(-9, 9, 5000).tolist()
    _same(lambda: _buffer.as_f32_c(ints, ndim=1, name="y"), ("cast_elements",), group="lists")
    _same(lambda: _buffer.as_i32_c(ints, ndim=1, name="y"), ("cast_elements",), group="lists")


# ----------------------------------------------------------- buffer input


def _awkward_buffers():
    base = _RNG.standard_normal((600, 8))
    ints = _RNG.integers(-1000, 1000, (600, 8))
    out = []
    for dt in (np.float32, np.float64, np.int32, np.int64, np.uint32, np.uint8, np.int16,
               np.uint16, np.int8, np.uint64, np.float16, bool):
        block = (ints if np.issubdtype(dt, np.integer) else base)
        block = np.abs(block).astype(dt) if dt in (np.uint32, np.uint8, np.uint16, np.uint64, bool) \
            else block.astype(dt)
        out += [block, block[::2], block[:, ::3], block[::-1], block.T, np.asfortranarray(block),
                block[:, 0], block[::5, 1]]
        ro = block.copy()
        ro.setflags(write=False)
        out += [ro, ro[::2]]
        if dt not in (bool, np.uint8, np.int8):
            out += [block.astype(block.dtype.newbyteorder(">")),
                    block.astype(block.dtype.newbyteorder(">"))[::2]]
    snan = np.full(2000, 0x7FA00001, np.uint32).view(np.float32)
    out += [snan[::2], snan.reshape(1000, 2)[:, 0]]
    return out


def test_buffer_input_matches():
    for block in _awkward_buffers():
        for convert in (_buffer.as_f32_c, _buffer.as_f64_c, _buffer.as_i64_c, _buffer.as_i32_c):
            _same(lambda: convert(block, ndim=None, name="X"), (), group="buffers")
        if block.ndim == 2:
            _same(lambda: _buffer.as_f32_colmajor(block, name="X"), (), group="buffers")


def test_a_strided_float32_view_takes_the_helper():
    block = _RNG.standard_normal((4000, 6)).astype(np.float32)
    _same(lambda: _buffer.as_f32_c(block[::2], ndim=2, name="X"), ("cast_elements",),
          group="buffers")
    ints = _RNG.integers(0, 9, (4000, 6)).astype(np.int32)
    _same(lambda: _buffer.as_f32_c(ints, ndim=2, name="X"), ("cast_elements",), group="buffers")


# ------------------------------------------------------------------ labels


def _label_lists():
    n = 3000
    ints = _RNG.integers(-5, 5, n).tolist()
    floats = (_RNG.integers(-5, 5, n) * 0.5).tolist()
    zeros_a = [0.0, -0.0] * 200
    zeros_b = [-0.0, 0.0] * 200
    return {
        "ints": ints, "tuple": tuple(ints), "floats": floats, "zeros a": zeros_a,
        "zeros b": zeros_b, "nan": floats[:500] + [float("nan")] + floats[:500],
        "bools mixed": [True, 1, 0] * 200, "int and float": [1, 1.0, 2] * 200,
        "strs": ["b", "a", "c"] * 200, "str and int": ["a", 1] * 200,
        "big ints": [2**70, 1, 2] * 200, "int64 edge": [2**63 - 1, -(2**63), 0] * 200,
        "many classes": list(range(5000)), "nested": [[v] for v in ints],
        "short": [3, 1, 2], "numpy ints": [np.int64(v) for v in ints[:600]],
        "none": [None] * 300, "one class": [7] * 400, "huge ints": [2**2000, 1, -5] * 200,
    }


@pytest.mark.parametrize("name", list(_label_lists()))
def test_encode_labels_list_matches_the_order_rule(name):
    y = _label_lists()[name]
    native = name in ("ints", "tuple", "one class", "int64 edge")
    _same(lambda: _labels.encode_labels(y), ("encode_labels_i64",) if native else (),
          group="labels")


def test_encode_labels_float_list_takes_the_helper():
    for name in ("floats", "zeros a", "zeros b"):
        y = _label_lists()[name]
        ref = _same(lambda: _labels.encode_labels(y), ("encode_labels_f64",), group="labels")
        assert ref[0][0] == "ok"
    classes, _ = _labels.encode_labels(_label_lists()["zeros b"])
    assert math.copysign(1.0, classes[0]) == -1.0  # the first-seen zero is kept


@pytest.mark.parametrize("classes", [[3, 5, 8], [0.5, 1.5, 2.5], ["a", "b", "c"],
                                     [False, True, 2], [2**70, 1, 2]])
def test_decode_labels_int32_codes(classes):
    codes = _RNG.integers(0, 3, 4000).astype(np.int32)
    numeric = _labels.label_kind(classes) in ("int", "float") and classes[0] != 2**70
    _same(lambda: _labels.decode_labels(classes, _arr(codes)),
          ("cast_elements",) if numeric else (), group="labels")
    negative = codes.copy()
    negative[17] = -1  # Python indexes from the end; the gather refuses; same answer
    _same(lambda: _labels.decode_labels(classes, _arr(negative)), (), group="labels")
    outside = codes.copy()
    outside[17] = 3
    _same(lambda: _labels.decode_labels(classes, _arr(outside)), (), group="labels")


# ----------------------------------------------------------------- metrics


def _metric_label_inputs():
    n = 3000
    out = {}
    for dt in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64):
        lo = -4 if np.issubdtype(dt, np.signedinteger) else 0
        out[np.dtype(dt).name] = _RNG.integers(lo, 5, n).astype(dt)
    out["bool"] = _RNG.integers(0, 2, n).astype(bool)
    out["strided"] = _RNG.integers(0, 5, 2 * n)[::2]
    out["big endian"] = _RNG.integers(0, 5, n).astype(">i4")
    out["Array"] = _arr(_RNG.integers(0, 5, n).astype(np.int32))
    out["float32"] = _RNG.integers(0, 5, n).astype(np.float32)
    out["list"] = _RNG.integers(0, 5, n).tolist()
    out["strs"] = [("c%d" % v) for v in _RNG.integers(0, 5, n)]
    out["column"] = _RNG.integers(0, 5, (n, 1))
    out["empty"] = np.zeros(0, np.int64)
    out["short"] = _RNG.integers(0, 5, 100)
    out["many classes"] = np.arange(n) % 5000
    out["one class"] = np.full(n, 3)
    out["binary"] = _RNG.integers(0, 2, n)
    out["minus one one"] = _RNG.integers(0, 2, n) * 2 - 1
    out["uint64 huge"] = np.full(n, 2**63 + 5, np.uint64)
    return out


_NATIVE_LABELS = {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool",
                  "strided", "big endian", "Array", "one class", "binary", "minus one one"}


@pytest.mark.parametrize("name", list(_metric_label_inputs()))
def test_classification_labels_match(name):
    y = _metric_label_inputs()[name]
    expect = ("encode_labels_i64",) if name in ("int64", "strided", "one class", "binary",
                                                "minus one one", "uint64") else ()
    ref = _same(lambda: M._classification_encoded(y, "y_true"), expect, group="metrics")
    if name in _NATIVE_LABELS and not _EXPECT_SABOTAGE:
        assert isinstance(M._classification_encoded(y, "y_true")[0], M._EncodedLabels), name
        assert ref[0][0] == "ok"

    other = _RNG.integers(-4, 5, 3000)[:len(y)]

    def prepared():
        true, pred, kind, observed = M._classification_pair(y, other, None)
        selected = observed[1:] + [99]
        return (kind, observed, len(true), sorted(M._label_set(true)),
                M._encode_classification(true, pred, observed),
                M._encode_classification(true, pred, selected),
                M._label_map(true, lambda v: int(v == observed[-1])))

    _same(prepared, ("gather_i32",) if name in _NATIVE_LABELS else (), group="metrics")


@pytest.mark.parametrize("name", list(_metric_label_inputs()))
def test_cluster_label_union_matches(name):
    y = _metric_label_inputs()[name]
    other = _RNG.integers(-2, 9, 3000)
    native = name in _NATIVE_LABELS and name != "bool"  # a bool is not an integer label here
    _same(lambda: M._prepare_cluster_labels(y, other[:len(y)]),
          ("gather_i32", "encode_labels_i32") if native else (), group="metrics")
    _same(lambda: M._as_i32_1d(y, "labels"), (), group="metrics")


def test_int32_range_refusal_and_weights():
    wide = _RNG.integers(0, 5, 3000)
    wide[11] = 2**31
    _same(lambda: M._as_i32_1d(wide, "labels"), ("reduce_stat",), group="metrics")
    for w in (_RNG.random(3000, dtype=np.float32), np.zeros(3000, np.float32),
              -_RNG.random(3000, dtype=np.float32),
              np.where(np.arange(3000) == 5, np.nan, 1.0).astype(np.float32),
              np.where(np.arange(3000) == 5, -0.0, 0.0).astype(np.float32)):
        # a NaN weight is refused by the finiteness scan before any reduction
        expect = () if np.isnan(w).any() else ("reduce_stat",)
        _same(lambda: M._sample_weight_f32(w, 3000, "r2_score"), expect, group="metrics")


# --------------------------------------------------------- model_selection


def _fold_lists(folds):
    return [(list(train), list(test)) for train, test in folds]


def _fold_labels():
    n = 3000
    return {
        "int64": _arr(_RNG.integers(0, 4, n)),
        "int32 signed": _arr(_RNG.integers(-3, 3, n).astype(np.int32)),
        "uint8 bool": _arr(_RNG.integers(0, 2, n).astype(np.uint8)),
        "float integral": _arr(_RNG.integers(0, 4, n).astype(np.float32)),
        "float zeros": _arr(np.where(_RNG.integers(0, 2, n) == 0, -0.0, 0.0)),
        "float regression": _arr(_RNG.standard_normal(n).astype(np.float32)),
        "float nan": _arr(np.where(np.arange(n) == 9, np.nan, 1.0)),
        "rare class": _arr(np.where(np.arange(n) < 3, 9, _RNG.integers(0, 3, n))),
        "all rare": _arr(np.arange(n)),
        "many classes": _arr(np.arange(n) % 1500),
        "sorted": _arr(np.sort(_RNG.integers(0, 4, n))),
        "list": _RNG.integers(0, 4, n).tolist(),
        "strs": [("c%d" % v) for v in _RNG.integers(0, 4, n)],
        "short": _arr(_RNG.integers(0, 2, 40)),
        "odd length": _arr(_RNG.integers(0, 4, 2999)),
    }


@pytest.mark.parametrize("classifier", [True, False])
@pytest.mark.parametrize("name", list(_fold_labels()))
def test_default_folds_match(name, classifier):
    y = _fold_labels()[name]
    for splits in (2, 3, 5, 7, 1, True, 2.0, len(y) + 1):
        python_only = (name in ("short",) or (classifier and name in ("list", "strs"))
                       or splits in (1, True, 2.0) or splits > len(y)
                       or (classifier and name == "all rare"))
        # a refusal (a class with fewer members than folds) is raised between
        # the two helpers, so only `fold_ids` is pinned for every case
        expect = () if python_only else ("fold_ids",)
        _same(lambda: _fold_lists(MS._default_fold_arrays(y, splits, classifier)), expect,
              group="folds")
        if not _EXPECT_SABOTAGE and not python_only:
            clean = _outcome(lambda: _fold_lists(MS._default_folds(y, splits, classifier)))
            fast = _outcome(lambda: _fold_lists(MS._default_fold_arrays(y, splits, classifier)))
            assert clean == fast


def test_one_row_per_fold():
    y = _arr(_RNG.integers(0, 2, 300))
    for classifier in (True, False):
        _same(lambda: _fold_lists(MS._default_fold_arrays(y, 300, classifier)), ("fold_ids",),
              group="folds")


def test_the_fold_sabotage_control_still_moves_the_folds():
    y = _arr(_RNG.integers(0, 4, 3000))
    clean = _fold_lists(MS._default_fold_arrays(y, 4, True))
    os.environ["MOJOLEARN_FOLD_ORDER_SABOTAGE"] = "1"
    os.environ["MOJOLEARN_HOST_ALLOW_SABOTAGE_SAVED"] = os.environ.get("MOJOLEARN_HOST_ALLOW_SABOTAGE", "")
    os.environ["MOJOLEARN_HOST_ALLOW_SABOTAGE"] = "1"
    try:
        with _Spy() as spy:
            rotated = _fold_lists(MS._default_fold_arrays(y, 4, True))
        assert "fold_ids" not in spy.calls
        assert rotated == _fold_lists(MS._default_folds(y, 4, True))
        if not _EXPECT_SABOTAGE:
            assert rotated != clean
    finally:
        os.environ.pop("MOJOLEARN_FOLD_ORDER_SABOTAGE")
        saved = os.environ.pop("MOJOLEARN_HOST_ALLOW_SABOTAGE_SAVED")
        if saved:
            os.environ["MOJOLEARN_HOST_ALLOW_SABOTAGE"] = saved
        else:
            os.environ.pop("MOJOLEARN_HOST_ALLOW_SABOTAGE")


def test_indices_and_overlap_match():
    n = 5000
    perm = _RNG.permutation(n)
    cases = {
        "ok int64": perm[:3000], "ok int32": perm[:3000].astype(np.int32),
        "ok uint32": perm[:3000].astype(np.uint32), "ok uint8": np.arange(200, dtype=np.uint8),
        "strided": perm[::2], "list": perm[:3000].tolist(), "short": perm[:10],
        "negative": np.where(np.arange(3000) == 7, -1, perm[:3000]),
        "too big": np.where(np.arange(3000) == 7, n, perm[:3000]),
        "duplicate": np.where(np.arange(3000) == 7, perm[8], perm[:3000]),
        "duplicate and too big": np.concatenate([perm[:3000], [perm[0], n]]),
        "floats": perm[:3000].astype(np.float32), "two dim": perm[:3000].reshape(1500, 2),
        "empty": np.zeros(0, np.int64),
    }
    for name, value in cases.items():
        native = name not in ("short", "floats", "two dim", "empty", "ok uint8")
        _same(lambda: MS._indices(value, n, "train"),
              ("check_indices_i64",) if native else (), group="indices")
    a = MS._indices(perm[:2500], n, "train")
    for other, hit in ((perm[2500:], False), (perm[2499:], True), (perm[:2500], True)):
        b = MS._indices(other, n, "test")
        ref = _same(lambda: MS._overlap(a, b, n), ("indices_overlap_i64",), group="indices")
        assert ref[0] == ("ok", ("bool", hit))


# ------------------------------------------- the C-builtin seams, pinned


class _Count:
    """Counts calls of `module.name`, for the seams that are C builtins and
    not core helpers: the new arm must reach the function and return its
    answer, the reference arm must not reach it at all."""

    def __init__(self, module, name):
        self.module, self.name, self.answered = module, name, 0

    def __enter__(self):
        self.real = getattr(self.module, self.name)

        def wrapped(*args, **kwargs):
            try:
                out = self.real(*args, **kwargs)
            except Exception:
                self.answered += 1  # a refusal raised there is its answer
                raise
            if out is not None:
                self.answered += 1
            return out

        setattr(self.module, self.name, wrapped)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.real)
        return False


def _pinned(module, name, fn, answers=True):
    os.environ["MOJOLEARN_HOTPATH"] = "python"
    try:
        with _Count(module, name) as ref_count:
            ref = _outcome(fn)
    finally:
        os.environ.pop("MOJOLEARN_HOTPATH", None)
    assert ref_count.answered == 0, f"the reference arm was answered by {name}"
    with _Count(module, name) as new_count:
        new = _outcome(fn)
    if answers is not None:
        assert bool(new_count.answered) == answers, (name, new_count.answered, answers)
    assert new == ref, f"{name}: new arm {new!r:.300} != reference {ref!r:.300}"
    return ref


@pytest.mark.parametrize("name", list(_label_lists()))
def test_plain_label_lists_match_the_label_loop(name):
    y = _label_lists()[name]
    plain = name in ("ints", "floats", "zeros a", "zeros b", "nan", "bools mixed",
                     "int and float", "strs", "int64 edge", "many classes", "short",
                     "one class", "big ints", "huge ints")
    # plain types; the NaN test is `v != v`, which holds 2**2000 too
    big = name == "huge ints"
    if isinstance(y, list):
        _pinned(_labels, "_sorted_plain_classes", lambda: _labels.sorted_classes(y),
                answers=plain)
    _pinned(_labels, "_plain_labels", lambda: _labels.flatten_labels(y),
            answers=plain or big or name == "tuple")
    if plain and name != "nan":
        classes, codes = _labels.sorted_classes(list(y))
        assert all(classes[c] == v for c, v in zip(codes, y))


def test_first_seen_object_is_kept_by_the_plain_path():
    for y in ([True, 1, 1.0, 0, False, 0.0] * 50, [1.0, 1, True, -0.0, 0, False] * 50,
              [0.0, -0.0] * 99, [-0.0, 0.0] * 99):
        ref = _pinned(_labels, "_sorted_plain_classes", lambda: _labels.sorted_classes(y))
        assert ref[0][0] == "ok"


@pytest.mark.parametrize("index", range(len(_LISTS)))
def test_flatten_fast_is_pinned(index):
    value = _LISTS[index]
    fast = _array._flatten_fast(value) is not None
    assert fast or index not in (0, 1, 2, 3, 4, 5, 6, 7, 20, 25, 26, 27)
    # a block the fast path declines is walked recursively, and the walk may
    # still be answered for a plain SUB-list; only the top-level yes is pinned
    _pinned(_array, "_flatten_fast", lambda: _array._flatten(value),
            answers=True if fast else None)


def test_block_copy_is_pinned():
    a = _arr(_RNG.integers(0, 99, (700, 6)).astype(np.int64))
    _pinned(_array, "_block_store", lambda: a[5:650], answers=True)
    _pinned(_array, "_block_store", lambda: a[5:650:2], answers=False)
    _pinned(_array, "_block_store", lambda: a[5:20], answers=False)  # a short run
    block = _RNG.integers(0, 99, (4000, 3)).astype(np.int64)
    _pinned(_buffer, "_same_dtype_store", lambda: _buffer.as_i64_c(block[::2], ndim=2, name="X"),
            answers=True)


def test_sabotage_build_diverges():
    """Runs last. Against a clean build it is vacuous by construction;
    against a `-D MOJOLEARN_HOST_SABOTAGE=1` build every group above must
    have recorded a divergence, or that group compares nothing."""
    if not _EXPECT_SABOTAGE:
        assert not _DIVERGED
        return
    groups = {"astype", "getitem", "reduce", "equal", "lists", "buffers", "labels",
              "metrics", "folds", "indices"}
    missing = groups - set(_DIVERGED)
    assert not missing, f"no divergence seen under sabotage in: {sorted(missing)}"
