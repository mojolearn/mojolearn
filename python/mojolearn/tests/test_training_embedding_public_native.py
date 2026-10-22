"""Embedding preparation and native kernel calls preserve the public contract."""
import array
import warnings

import numpy as np
import pytest

from mojolearn import _backend, _training_impl as T
from mojolearn._array import Array
from mojolearn._verify_native_runtime import audit_native_call


INDEX_CASES = ['?', 'i1', 'u1', 'i2', 'u2', 'i4', 'u4', 'i8', 'u8', 'q', 'Q', 'f2', 'f4', 'f8', '>i4', '>f8',
               'matrix', 'fortran', 'negative', 'strided', 'readonly', 'array', 'array-fortran',
               'list', 'tuple', 'float-list', 'numpy-float-list', 'object-int', 'object-float',
               'object-numpy-int', 'array-array', 'memoryview', 'bytearray', 'scalar-int',
               'scalar-float', 'scalar-numpy-int', 'scalar-numpy-float', 'scalar-ndarray',
               'scalar-array', 'list-array', 'list-npfloat64']


def indices(case):
    if case in INDEX_CASES[:16]:
        return np.array([2, 0, 2, 1], dtype=case)
    base = np.array([2, 0, 2, 1], dtype='i4')
    if case in ('matrix', 'fortran', 'array-fortran'):
        base = base.reshape(2, 2)
        if case != 'matrix':
            base = np.asfortranarray(base)
        return Array.from_buffer(base) if case == 'array-fortran' else base
    if case == 'negative':
        return base[::-1]
    if case == 'strided':
        return np.array([2, 9, 0, 9, 2, 9, 1, 9], dtype='i8')[::2]
    if case == 'readonly':
        return np.frombuffer(base.tobytes(), dtype='i4')
    if case == 'array':
        return Array.from_buffer(base)
    if case == 'list-array':
        return [Array.from_buffer(base[:2]), Array.from_buffer(base[2:])]
    if case == 'list-npfloat64':
        return [base[:2].astype('f8'), base[2:].astype('f8')]
    values = {
        'list': [2, 0, 2, 1], 'tuple': ((2, 0), (2, 1)),
        'float-list': [2.9, 0.9, 2.1, 1.1], 'numpy-float-list': [np.float64(2), 0, 2, 1],
        'object-int': np.array([2, 0, 2, 1], dtype=object),
        'object-float': np.array([2.9, 0.9, 2.1, 1.1], dtype=object),
        'object-numpy-int': np.array([np.int64(2), 0, 2, 1], dtype=object),
        'array-array': array.array('i', [2, 0, 2, 1]),
        'memoryview': memoryview(array.array('h', [2, 0, 2, 1])),
        'bytearray': bytearray([2, 0, 2, 1]), 'scalar-int': 2, 'scalar-float': 2.9,
        'scalar-numpy-int': np.int64(2), 'scalar-numpy-float': np.float32(2),
        'scalar-ndarray': np.array(2, dtype='i4'),
        'scalar-array': Array.from_buffer(np.array(2, dtype='i4')),
    }
    return values[case]


def reference(operation, source, ids, vocab=4):
    # Test-controller preparation and unchanged native entry are a routing /
    # byte compatibility reference. They are outside the measured product call.
    if isinstance(ids, bytearray):
        ids = list(ids)
    ids = np.asarray(ids).astype('i4').ravel()
    source = np.ascontiguousarray(source, dtype='f4')
    rows, width = source.shape
    output = np.empty((vocab if operation == 'embedding_backward' else ids.size, width), dtype='f4')
    binding = _backend.binding('_mojolearn_training', 'identical')
    getattr(binding, operation)([output.ctypes.data, source.ctypes.data, ids.ctypes.data], [ids.size, vocab if operation == 'embedding_backward' else rows, width])
    return output


def assert_output(result, expected):
    result = result.assert_native()
    assert type(result) is Array
    assert result.shape == expected.shape
    assert result.dtype == '<f4'
    assert np.asarray(result).tobytes() == expected.tobytes()


def assert_refused(result, category):
    assert type(result.exception) is category, result.report()
    for key in ('python_frames', 'forbidden_native_calls', 'wrapper_work', 'unobserved_existing_threads'):
        assert not result.report()[key], result.report()


@pytest.mark.parametrize('operation', ['embedding_forward', 'embedding_backward'])
@pytest.mark.parametrize('case', INDEX_CASES)
def test_public_embedding_index_preparation_matches_retained_kernel(operation, case):
    ids = indices(case)
    count = 1 if case.startswith('scalar-') else 4
    source = (np.arange((count if operation == 'embedding_backward' else 4) * 3, dtype='f4').reshape(-1, 3) - 2) / 8
    expected = reference(operation, source, ids)
    args = (source, ids, 4) if operation == 'embedding_backward' else (source, ids)
    assert_output(audit_native_call(getattr(T, operation), *args, numeric_mode='identical'), expected)


@pytest.mark.parametrize('operation', ['embedding_forward', 'embedding_backward'])
@pytest.mark.parametrize('layout', ['fortran', 'negative', 'readonly', 'array'])
def test_public_embedding_float_buffers_and_duplicate_fold_order(operation, layout):
    # Cancellation-sensitive repeated ids exercise the retained ascending fold.
    source = np.array([[2**24, 1, -1], [1, -2, 0.5], [-2**24, 3, 2], [0.125, -4, -1]], dtype='f4')
    ids = np.array([2, 2, 2, 1], dtype='i4')
    if layout in ('fortran', 'array'):
        source = np.asfortranarray(source)
    elif layout == 'negative':
        source = source[::-1, ::-1]
    else:
        source.flags.writeable = False
    expected = reference(operation, source, ids)
    if layout == 'array':
        source = Array.from_buffer(source)
    args = (source, ids, 4) if operation == 'embedding_backward' else (source, ids)
    assert_output(audit_native_call(getattr(T, operation), *args, numeric_mode='identical'), expected)


REFUSALS = ['longdouble', 'complex', 'object-str', 'str', 'empty', 'ragged', 'nan', 'inf',
            'overflow', 'u64-overflow', 'negative-id', 'past-vocab', 'numpy-int-list',
            'numpy-float32-list', 'large-int-list', 'bad-float-dtype', 'bad-rank', 'empty-source']


@pytest.mark.parametrize('operation', ['embedding_forward', 'embedding_backward'])
@pytest.mark.parametrize('case', REFUSALS)
def test_public_embedding_preserves_retained_refusal_categories(operation, case):
    source = np.arange(12, dtype='f4').reshape(4, 3)
    ids, category = np.array([2, 0, 2, 1], dtype='i4'), TypeError
    if case == 'longdouble': ids = ids.astype('g')
    elif case == 'complex': ids, category = ids.astype('c8'), ValueError
    elif case == 'object-str': ids = np.array(['2', '0', '2', '1'], dtype=object)
    elif case == 'str': ids = np.array(['2', '0', '2', '1'])
    elif case == 'empty': ids, category = [], ValueError
    elif case == 'ragged': ids, category = [[2, 0], [1]], ValueError
    elif case == 'nan': ids, category = np.array([np.nan] * 4, dtype='f4'), ValueError
    elif case == 'inf': ids, category = np.array([np.inf] * 4), OverflowError
    elif case == 'overflow': ids, category = np.array([2**31] * 4, dtype='i8'), OverflowError
    elif case == 'u64-overflow': ids, category = np.array([2**63] * 4, dtype='u8'), OverflowError
    elif case == 'negative-id': ids, category = np.array([-1] * 4, dtype='i4'), Exception
    elif case == 'past-vocab': ids, category = np.array([4] * 4, dtype='i4'), Exception
    elif case == 'numpy-int-list': ids = [np.int64(2), 0, 2, 1]
    elif case == 'numpy-float32-list': ids = [np.float32(2), 0, 2, 1]
    elif case == 'large-int-list': ids = [2**63] * 4
    elif case == 'bad-float-dtype': source = source.astype('f8')
    elif case == 'bad-rank': source, category = source.ravel(), ValueError
    elif case == 'empty-source': source, category = source[:0], ValueError
    args = (source, ids, 4) if operation == 'embedding_backward' else (source, ids)
    assert_refused(audit_native_call(getattr(T, operation), *args, numeric_mode='identical'), category)


VOCAB_CASES = ['int', 'float', 'str', 'bytes', 'bool', 'numpy-int', 'numpy-float', 'numpy-complex',
               'longdouble', 'object', 'text', 'complex-array', 'str-decimal', 'none', 'negative', 'zero', 'nan', 'inf', 'list',
               'array-void-text', 'void-text', 'array-void-badtext', 'void-badtext', 'memoryview', 'array-1d']
if hasattr(np.dtypes, 'StringDType'):
    VOCAB_CASES.append('array-string')


def vocab_value(case):
    if case == 'array-string':
        return np.array('4', dtype=np.dtypes.StringDType())
    return {'int': 4, 'float': 4.9, 'str': '4', 'bytes': b'4', 'bool': True,
            'numpy-int': np.int64(4), 'numpy-float': np.float32(4.9), 'numpy-complex': np.complex64(4 + 1j),
            'longdouble': np.longdouble(4.9), 'object': np.array(4, dtype=object), 'text': np.array('4'),
            'complex-array': np.array(4 + 1j), 'str-decimal': '4.0', 'none': None, 'negative': -1,
            'zero': 0, 'nan': float('nan'), 'inf': float('inf'), 'list': [4],
            'array-void-text': np.array(b'4', dtype='V1'), 'void-text': np.void(b'4'),
            'array-void-badtext': np.array(b'bad', dtype='V3'), 'void-badtext': np.void(b'bad'),
            'memoryview': memoryview(b'4'), 'array-1d': np.array([4])}[case]


@pytest.mark.parametrize('case', VOCAB_CASES)
def test_public_embedding_vocab_native_unboxing_matches_errors_and_warnings(case):
    vocab = vocab_value(case)
    source, ids = np.ones((1, 3), dtype='f4'), [0]
    category = None
    with warnings.catch_warnings(record=True) as old:
        warnings.simplefilter('always')
        try:
            count = int(vocab)
            if count < 0: raise ValueError('negative dimension')
            # Old wrapper converted twice before entering the compiled entry.
            count = int(vocab)
            if count == 0: raise Exception('null output buffer')
        except Exception as error:
            category = type(error)
    with warnings.catch_warnings(record=True) as new:
        warnings.simplefilter('always')
        result = audit_native_call(T.embedding_backward, source, ids, vocab, numeric_mode='identical')
    if category:
        assert_refused(result, category)
    else:
        assert_output(result, reference('embedding_backward', source, ids, count))
    assert [(type(w.message), str(w.message)) for w in new] == [(type(w.message), str(w.message)) for w in old]


@pytest.mark.parametrize('slot', ['source', 'indices', 'nested-index', 'object-index', 'vocab', 'object-vocab'])
def test_public_embedding_refuses_python_hooks_before_execution(slot):
    calls = []
    class Foreign:
        def __array__(self, *args, **kwargs): calls.append('array'); return np.ones(1, dtype='i4')
        def __float__(self): calls.append('float'); return 1.
        def __int__(self): calls.append('int'); return 1
        def __index__(self): calls.append('index'); return 1
    source, ids, vocab = np.ones((1, 3), dtype='f4'), [0], 4
    if slot == 'source': source = Foreign()
    elif slot == 'indices': ids = Foreign()
    elif slot == 'nested-index': ids = [Foreign()]
    elif slot == 'object-index':
        ids = np.empty(1, dtype=object)
        ids[0] = Foreign()
    elif slot == 'vocab': vocab = Foreign()
    else:
        vocab = np.empty((), dtype=object)
        vocab[()] = Foreign()
    assert_refused(audit_native_call(T.embedding_backward, source, ids, vocab, numeric_mode='identical'), TypeError)
    assert not calls


@pytest.mark.parametrize('entry', ['embedding_forward', 'embedding_backward', 'cast_elements', 'array_type'])
def test_public_embedding_refuses_replaced_native_entries(monkeypatch, entry):
    calls = []
    def hook(*args): calls.append(args)
    if entry == 'array_type': monkeypatch.setitem(T._OPT_CONTEXT, 'array_type', hook)
    elif entry == 'cast_elements': monkeypatch.setitem(T._OPT_CONTEXT, 'embedding_index_cast', hook)
    else: monkeypatch.setitem(T._OPT_CONTEXT['bindings']['identical'], entry, hook)
    operation = 'embedding_backward' if entry == 'embedding_backward' else 'embedding_forward'
    source, ids = np.ones((4, 3), dtype='f4'), np.array([2, 0, 2, 1], dtype='i8')
    if entry == 'array_type': ids = [2, 0, 2, 1]
    args = (source, ids, 4) if operation == 'embedding_backward' else (source, ids)
    assert_refused(audit_native_call(getattr(T, operation), *args, numeric_mode='identical'), TypeError)
    assert not calls


@pytest.mark.parametrize('operation', ['embedding_forward', 'embedding_backward'])
def test_public_embedding_does_not_reenter_python_preparation(monkeypatch, operation):
    calls = []
    def hook(*args, **kwargs): calls.append(args); raise AssertionError('old preparation entered')
    source, ids = np.ones((4, 3), dtype='f4'), [2, 0, 2, 1]
    expected = reference(operation, source, ids)
    for name in ('_c32', '_load', 'as_i32_c'):
        monkeypatch.setattr(T, name, hook)
    args = (source, ids, 4) if operation == 'embedding_backward' else (source, ids)
    assert_output(audit_native_call(getattr(T, operation), *args, numeric_mode='identical'), expected)
    assert not calls


def test_public_embedding_backward_row_count_guard():
    assert_refused(audit_native_call(T.embedding_backward, np.ones((2, 3), dtype='f4'), [0], 4, numeric_mode='identical'), ValueError)


@pytest.mark.parametrize('operation', ['embedding_forward', 'embedding_backward'])
@pytest.mark.parametrize('layout', ['c', 'fortran', 'negative', 'strided', 'array'])
def test_public_embedding_preserves_nonfinite_source_refusal(operation, layout):
    source = np.ones((4, 3), dtype='f4')
    source.view('u4')[0, 0] = 0x7f800001
    if layout == 'fortran': source = np.asfortranarray(source)
    elif layout == 'negative': source = source[::-1]
    elif layout == 'strided':
        base = np.ones((4, 6), dtype='f4')
        base[:, ::2] = source
        source = base[:, ::2]
    elif layout == 'array': source = Array.from_buffer(source)
    ids = np.array([2, 0, 2, 1], dtype='i4')
    args = (source, ids, 4) if operation == 'embedding_backward' else (source, ids)
    # Existing CPU and GPU entries reject every nonfinite source. Layout
    # preparation must preserve that refusal, not publish NaN output bits.
    assert_refused(audit_native_call(getattr(T, operation), *args, numeric_mode='identical'), Exception)


@pytest.mark.parametrize('operation', ['embedding_forward', 'embedding_backward'])
def test_public_embedding_object_indices_require_real_scalar_leaves(operation):
    ids = np.empty(4, dtype=object)
    ids[:] = [np.array(2), np.array(0), np.array(2), np.array(1)]
    source = np.ones((4, 3), dtype='f4')
    args = (source, ids, 4) if operation == 'embedding_backward' else (source, ids)
    assert_refused(audit_native_call(getattr(T, operation), *args, numeric_mode='identical'), TypeError)


def test_existing_embedding_primitive_contract():
    from mojolearn.tests import test_training_primitives_surface as existing
    report = existing.Report("native_embedding_existing_primitive")
    existing.arm_embedding(report)
    assert report.rows and not report.failures, report.failures


@pytest.mark.parametrize('scalar', [np.longlong, np.ulonglong])
def test_native_optimizer_preserves_connected_longlong_aliases(scalar):
    params = np.ones(3, dtype='f4')
    owner = audit_native_call(T.AdamW, [params], lr=scalar(0), numeric_mode='identical').assert_native()
    audit_native_call(owner.step, [np.ones(3, dtype='f4')]).assert_native()
    assert np.array_equal(params, np.ones(3, dtype='f4'))
