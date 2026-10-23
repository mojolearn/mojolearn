# SPDX-License-Identifier: Apache-2.0
"""Public split input boundaries: values/bytes preserved, caller hooks refused."""
import array
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from mojolearn import Array, model_selection as ms
from mojolearn._verify_native_runtime import audit_native_call


def clean(call, *args, error=None, **kwargs):
    observed = audit_native_call(call, *args, **kwargs)
    report = observed.report()
    assert not any(report[key] for key in (
        'python_frames', 'forbidden_native_calls', 'unresolved_native_calls',
        'wrapper_work', 'unobserved_existing_threads')), report
    if error is not None:
        assert isinstance(observed.exception, error), report
        return observed.exception
    return observed.assert_native()


@pytest.mark.parametrize('dtype', ('f4', 'f8', 'i4', 'i8', 'u1', 'u2'))
@pytest.mark.parametrize('layout', ('c', 'f', 'strided', 'reversed', 'readonly', 'endian'))
def test_split_preserves_native_numeric_values_and_materializer_width(dtype, layout):
    source = np.arange(60, dtype=dtype).reshape(12, 5)
    if layout == 'f':
        source = np.asfortranarray(source)
    elif layout == 'strided':
        source = source[:, ::2]
    elif layout == 'reversed':
        source = source[::-1]
    elif layout == 'readonly':
        source.setflags(write=False)
    elif layout == 'endian':
        source = source.astype(source.dtype.newbyteorder('>'))
    # Independent expected rows and the existing natural-width copy ABI:
    # noncontiguous/endian uint16 buffers widen to uint32.
    expected_dtype = np.dtype('u4') if dtype == 'u2' and layout in ('strided', 'reversed', 'endian') else np.dtype(dtype)
    before = source.tobytes()
    train, test = clean(ms.train_test_split, source, test_size=3, shuffle=False)
    for actual, expected in ((train, source[:9]), (test, source[9:])):
        result = np.asarray(actual)
        assert result.dtype == expected_dtype
        np.testing.assert_array_equal(result, expected)
        assert result.tobytes() == expected.astype(expected_dtype).tobytes()
    assert source.tobytes() == before


@pytest.mark.parametrize('factory', (
    lambda: Array.from_list([[float(i), float(i + 1)] for i in range(12)], '<f4'),
    lambda: array.array('d', (i + .125 for i in range(12))),
    lambda: memoryview(array.array('h', range(12))),
    lambda: list(range(12)),
    lambda: tuple(range(12)),
    lambda: range(12),
))
def test_split_native_containers(factory):
    source = factory()
    expected = np.asarray(source)
    train, test = clean(ms.train_test_split, source, test_size=3, shuffle=False)
    np.testing.assert_array_equal(np.asarray(train), expected[:9])
    np.testing.assert_array_equal(np.asarray(test), expected[9:])
    if type(source) in (list, tuple):
        assert type(train) is list and type(test) is list


def test_no_runtime_buffer_or_binding_loader(monkeypatch):
    from mojolearn import _backend, _buffer
    source = np.arange(36, dtype='f8').reshape(12, 3)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append('Python preparation')
        raise AssertionError('Python input helper or binding loader reached')

    monkeypatch.setattr(_buffer, '_materialize', forbidden)
    monkeypatch.setattr(_buffer, '_native', forbidden)
    monkeypatch.setattr(_backend, 'binding', forbidden)
    monkeypatch.setattr(_backend, 'load_host_module', forbidden)
    train, test = clean(ms.train_test_split, source, test_size=3, shuffle=False)
    assert np.asarray(train).shape == (9, 3) and np.asarray(test).shape == (3, 3)
    assert calls == []


def test_protocol_refusal_precedes_every_input_hook():
    calls = []

    class Protocol:
        def __len__(self):
            calls.append('len')
            return 12

        def __array__(self, *args, **kwargs):
            calls.append('array')
            return np.arange(12, dtype='f4')

        def __iter__(self):
            calls.append('iter')
            return iter(range(12))

        @property
        def shape(self):
            calls.append('shape')
            return (12,)

    clean(ms.train_test_split, Protocol(), test_size=.25, error=TypeError)
    clean(ms.LeavePOut(2).get_n_splits, Protocol(), error=TypeError)
    clean(ms.check_cv, Protocol(), error=TypeError)
    assert calls == []


def test_list_subclass_and_object_scalar_hooks_are_refused():
    calls = []

    class Rows(list):
        def __len__(self):
            calls.append('list len')
            return super().__len__()

    class Scalar:
        def __float__(self):
            calls.append('float')
            return 1.0

    source = np.empty((12, 2), dtype=object)
    source[:] = Scalar()
    clean(ms.train_test_split, Rows(range(12)), shuffle=False, error=TypeError)
    clean(ms.train_test_split, source, shuffle=False, error=TypeError)
    assert calls == []


def test_splitter_subclass_refuses_overridden_property():
    calls = []

    class Custom(ms.KFold):
        @property
        def n_splits(self):
            calls.append('n_splits')
            return 3

    splitter = object.__new__(Custom)
    clean(ms.check_cv, splitter, error=TypeError)
    clean(ms.KFold.get_n_splits, splitter, error=TypeError)
    assert calls == []


def test_splitter_metaclass_hash_is_never_called():
    calls = []

    class Meta(type):
        def __hash__(cls):
            calls.append('metaclass hash')
            return type.__hash__(cls)

    class Custom(metaclass=Meta):
        pass

    clean(ms.check_cv, Custom(), error=TypeError)
    clean(ms.KFold.get_n_splits, Custom(), error=TypeError)
    assert calls == []


def test_splitter_field_property_mutation_is_refused_before_construction(monkeypatch):
    splitter = ms.KFold(3)
    calls = []

    def forbidden(*args):
        calls.append('field descriptor')
        raise AssertionError('splitter property ran')

    monkeypatch.setattr(ms.KFold, 'n_splits', property(forbidden, forbidden), raising=False)
    clean(ms.check_cv, splitter, error=TypeError)
    clean(ms.check_cv, 3, error=TypeError)
    assert calls == []


def test_explicit_native_range_folds_remain_supported():
    splitter = clean(ms.check_cv, [(range(9), range(9, 12))])
    actual = list(splitter.split(np.arange(12, dtype='f4')))
    assert len(actual) == 1
    np.testing.assert_array_equal(np.asarray(actual[0][0]), np.arange(9))
    np.testing.assert_array_equal(np.asarray(actual[0][1]), np.arange(9, 12))


def test_preconnected_helper_rejects_python_replacement(monkeypatch):
    calls = []

    def callback(*args):
        calls.append(args)
        raise AssertionError('compiled helper replacement ran')

    monkeypatch.setitem(ms._NATIVE_CONTEXT['family_msel_entries'], 'arange_i64', callback)
    clean(ms.train_test_split, np.arange(12, dtype='f4'), shuffle=False, error=TypeError)
    assert calls == []


@pytest.mark.parametrize('kind', ('float64', 'mixed', 'int16_float32', 'series'))
def test_pandas_storage_avoids_array_protocol_and_preserves_promoted_values(kind, monkeypatch):
    pd = pytest.importorskip('pandas')
    if kind == 'float64':
        source = pd.DataFrame(np.arange(36, dtype='f8').reshape(12, 3) + 2 ** -30)
    elif kind == 'mixed':
        source = pd.DataFrame({'float': np.arange(12, dtype='f8') + 2 ** -30,
                               'int': np.arange(12, dtype='i4'), 'flag': np.arange(12) % 2 == 0})
    elif kind == 'int16_float32':
        source = pd.DataFrame({'small': np.arange(12, dtype='i2'), 'single': np.arange(12, dtype='f4') + .125})
    else:
        source = pd.Series(np.arange(12, dtype='f8') + 2 ** -30)
    expected = source.to_numpy()
    if expected.dtype == object:
        expected = expected.astype('f8')
    calls = []

    def forbidden(*args, **kwargs):
        calls.append('pandas array hook')
        raise AssertionError('pandas Python conversion ran')

    monkeypatch.setattr(type(source), '__array__', forbidden)
    train, test = clean(ms.train_test_split, source, test_size=3, shuffle=False)
    assert np.asarray(train).dtype == expected.dtype
    np.testing.assert_array_equal(np.asarray(train), expected[:9])
    np.testing.assert_array_equal(np.asarray(test), expected[9:])
    assert calls == []


def test_numpy_free_import_and_standard_buffers():
    program = textwrap.dedent('''
        import array, importlib.abc, sys
        class NoArrays(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('numpy', 'scipy', 'sklearn'):
                    raise AssertionError('optional array dependency imported: ' + fullname)
        sys.meta_path.insert(0, NoArrays())
        from mojolearn.model_selection import train_test_split
        from mojolearn._verify_native_runtime import audit_native_call
        for code in ('h', 'q', 'd'):
            source = memoryview(array.array(code, range(12)))
            train, test = audit_native_call(train_test_split, source, test_size=3, shuffle=False).assert_native()
            assert train.tolist() == list(range(9))
            assert test.tolist() == list(range(9, 12))
        assert not any(name.split('.')[0] in ('numpy', 'scipy', 'sklearn') for name in sys.modules)
    ''')
    completed = subprocess.run([sys.executable, '-c', program], env=os.environ.copy(), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
