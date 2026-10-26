"""The public Array operations are one native call each
(bindings/array_native.mojo); run only against the integrated native build.

Each operation is held to its retained Python definition
(`_array._ArrayReference`, never called by the product) on C- and F-order
blocks of every dtype, 0-d and empty blocks: the same result (Array bytes,
layout and dtype; Python scalars by type and bits) or the same refusal (type
and words). The audit checks that the public call itself runs no Python frame.
"""
import array
import pickle
import struct
import sys

import numpy as np
import pytest

from mojolearn import _array
from mojolearn._array import Array, _ArrayReference as R
from mojolearn._verify_native_runtime import audit_native_call

_RNG = np.random.default_rng(859)


def canon(value):
    if isinstance(value, Array):
        return ("Array", value.shape, value.dtype, value.order, value.strides, value.tobytes())
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, float):
        return ("float", struct.pack("<d", value))
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, [canon(v) for v in value])
    if isinstance(value, dict):
        return ("dict", sorted((k, canon(v)) for k, v in value.items()))
    if isinstance(value, np.ndarray):
        return ("ndarray", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, memoryview):
        return ("memoryview", value.format, value.shape, value.tobytes())
    return (type(value).__name__, value)


def outcome(fn):
    try:
        return ("ok", canon(fn()))
    except Exception as exc:  # the refusal is the result
        return ("refused", type(exc).__name__, str(exc))


def same(native, reference):
    assert outcome(native) == outcome(reference)


def clean(operation, *args):
    audit = audit_native_call(operation, *args)
    report = audit.report()
    assert not any(report[key] for key in ('python_frames', 'forbidden_native_calls',
        'unresolved_native_calls', 'wrapper_work', 'unobserved_existing_threads')), report
    return audit


def blocks():
    out = []
    for dt in (np.float32, np.float64, np.float16, np.int32, np.int64, np.uint32, np.uint8, np.uint16, np.int8):
        base = (_RNG.standard_normal((6, 5)) * 40).astype(dt)
        out += [Array.from_buffer(np.ascontiguousarray(base)), Array.from_buffer(np.asfortranarray(base)),
                Array.from_buffer(np.ascontiguousarray(base[:, 0])), Array.from_buffer(np.ascontiguousarray(base[:0]))]
        out.append(Array.from_buffer(np.ascontiguousarray((_RNG.standard_normal((3, 4, 300)) * 9).astype(dt))))
    out.append(Array.from_buffer(np.float64(2.5).reshape(())))
    return out


@pytest.mark.parametrize('index', range(46))
def test_metadata_and_conversions_match_the_definitions(index):
    a = blocks()[index]
    for name in ('__len__', '__repr__', '__bool__', 'tobytes', 'copy', 'tolist', 'ravel', 'min', 'max', 'sum', 'argmax'):
        same(lambda: getattr(a, name)(), lambda: getattr(R, name)(a))
    same(lambda: a.flags, lambda: R.flags(a))
    same(lambda: a.__array_interface__, lambda: R.__array_interface__(a))
    same(lambda: list(a), lambda: list(R.__iter__(a)))
    same(lambda: a.reshape(-1), lambda: R.reshape(a, -1))
    same(lambda: a.reshape((a.size, 1)), lambda: R.reshape(a, (a.size, 1)))
    for bad in ((-1, -1), (7, 9), 2.5, "x"):
        same(lambda: a.reshape(bad), lambda: R.reshape(a, bad))
    for dtype in ("<f4", "<f8", "<f2", "<i4", "<i8", "<u4", "<u1", "<u2", "<i1", float, int, bool, np.float32, "float32", "bad"):
        same(lambda: a.astype(dtype), lambda: R.astype(a, dtype))
    same(lambda: pickle.loads(pickle.dumps(a)), lambda: pickle.loads(pickle.dumps(a)))
    assert canon(pickle.loads(pickle.dumps(a)))[1:] == canon(a)[1:]
    same(lambda: a.__reduce__()[1], lambda: R.__reduce__(a)[1])
    same(lambda: np.asarray(a), lambda: R.__array__(a))
    same(lambda: a.__array__(np.float64, None), lambda: R.__array__(a, np.float64, None))
    same(lambda: a.__array__(np.float64, False), lambda: R.__array__(a, np.float64, False))
    if sys.version_info >= (3, 12):
        same(lambda: memoryview(a), lambda: R.__buffer__(a, 0))
    for other in (a, a.copy(), 0, 1.5, True, "x", a.ravel()):
        # The methods themselves: the operator turns NotImplemented into an
        # identity comparison on both sides.
        same(lambda: a.__eq__(other), lambda: R.__eq__(a, other))
        same(lambda: a.__ne__(other), lambda: R.__ne__(a, other))


def test_indexing_matches_the_definition():
    class Index:
        def __index__(self):
            return 2
    a = Array.from_buffer(np.ascontiguousarray((_RNG.standard_normal((40, 30, 5)) * 9).astype(np.float32)))
    f = Array.from_buffer(np.asfortranarray(_RNG.integers(0, 255, (300, 7)).astype(np.uint8)))
    keys = [slice(None), slice(3, 30), slice(30, 3, -2), 5, -1, Index(), True, (2, 3), (2, 3, 4), (slice(None), 2),
            (4, slice(2, 28)), (slice(1, 39), slice(None), slice(None)), (), (1, 2, 3, 4), 99, -41, 1.5, "x",
            slice(None, None, 0), (slice(None), Index())]
    for block in (a, f):
        for key in keys:
            same(lambda: block[key], lambda: R.__getitem__(block, key))


def test_astype_refusals_match_the_definition():
    floats = Array.from_list([1.5, float("nan"), float("inf"), 3e10], "<f8")
    big = Array.from_list([1, 2**40, -5], "<i8")
    half = Array.from_list([1.0, 70000.0], "<f8")
    for a, dtype in ((floats, "<i4"), (floats, "<u1"), (floats, "<f2"), (big, "<i4"), (big, "<u4"), (big, "<u1"),
                     (half, "<f2"), (Array.from_list([300.0, -1.0], "<f4"), "<u1")):
        same(lambda: a.astype(dtype), lambda: R.astype(a, dtype))


def test_construction_matches_the_definition():
    for shape, dtype in (((2, 3), "<f4"), (4, "<i8"), ((), "<f8"), ((0, 3), "<u1"), ((2, -1), "<f4"),
                         ((2.7, 3), "<f4"), ((2, 3), "bad"), ((2, 3), np.float16)):
        def native():
            return Array(shape, dtype)

        def reference():
            out = object.__new__(Array)
            R.__init__(out, shape, dtype)
            return out
        same(native, reference)


def test_rows_iterate_lazily_as_views():
    a = Array.from_buffer(np.ascontiguousarray(_RNG.standard_normal((5, 3)).astype(np.float32)))
    rows = iter(a)
    first = next(rows)
    assert first._base is a and first.shape == (3,) and first._addr == a._addr
    assert [canon(r) for r in rows] == [canon(r) for r in list(R.__iter__(a))[1:]]


@pytest.mark.parametrize('name', ['__len__', '__repr__', 'tolist', 'copy', 'ravel', 'sum', 'argmax'])
def test_public_operations_run_no_python_frame(name):
    a = Array.from_buffer(np.asfortranarray(_RNG.standard_normal((300, 4)).astype(np.float32)))
    clean(getattr(a, name))
    clean(a.astype, '<f8')
    clean(a.__getitem__, (slice(3, 200), 1))
    clean(a.reshape, (4, 300))
    clean(a.__eq__, a)


def test_length_exception_is_only_the_wrapper_frame():
    from mojolearn import _runtime_fallbacks
    assert _runtime_fallbacks._LENGTH_CODE is Array.__len__.__code__
    assert Array.__len__.__code__.co_names == ('_ARRAY_NATIVE', '_ARRAY_CONTEXT')


def test_helpers_are_connected_once():
    helpers = _array._ARRAY_CONTEXT['helpers']
    assert set(helpers) == {'cast_elements', 'reduce_stat', 'strided_copy_bytes', 'equal_elements'}
    assert _array._ARRAY_CONTEXT['entry'] is _array._ARRAY_NATIVE
    assert isinstance(array.array('B'), _array._ARRAY_CONTEXT['array_type'])


_LISTS = [
    [1, 2, 3], [1.5, 2, True], [True, False], [[1, 2], [3, 4]], [[1.0, 2], [3, 4]],
    ((1, 2), (3, 4)), [(1, 2), [3, 4]], [[], []], [[1, 2], [3]], [[1, 2], 3], [1, [2, 3]],
    [[[1, 2]], [[3, 4]]], [1, "a"], [[1, None]], [None], [2**70, 1], [[2**63], [1]],
    [np.float32(1.5), 2.0], [np.int64(3), 4], [1.0, float("nan"), -0.0], [], [[]], "abc",
    [b"x"], [[1.0, 2.0], (3.0, 4.0)], [[1, 2.5], [True, 4]], 3.5, [70000.0, 1.0],
]


@pytest.mark.parametrize("index", range(len(_LISTS)))
def test_from_list_matches_the_definition(index):
    value = _LISTS[index]
    for dtype in ("<f8", "<f4", "<f2", "<i4", "<i8", "<u1", "<i1", "bad"):
        same(lambda: Array.from_list(value, dtype), lambda: R.from_list(Array, value, dtype))
    nested = Array.from_list([[1.0, 2.0], [3.0, 4.0]], "<f4")
    same(lambda: Array.from_list(nested, "<f8"), lambda: R.from_list(Array, nested, "<f8"))
