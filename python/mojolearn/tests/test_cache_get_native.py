# SPDX-License-Identifier: Apache-2.0
"""Whole public cache lookup: the native admission of a buffer's key.

`DeviceCache.get` admits the caller's `_addr`, `size` and `itemsize`
natively (builtin and NumPy scalars, 0-d arrays, temporal and void words,
inherited builtins; refusals of every Python hook) and looks the key up in
the native entries. Each test holds one REAL upload on the x_prep binding's
store (a GPU package): a 3-word buffer whose metadata the cases restate in
other scalar forms, so a hit proves the converted key equals the buffer's.

Deleted with the instance dictionary (2026-11-21, arena unit E): the cases
that seeded `cache._entries` with list or malformed entries, replaced it
with a hooked mapping or key, or installed a `_key` callback on the cache
or a subclass; none of that state exists any more.
"""
import json
import os
import subprocess
import sys
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
from mojolearn import _arena_io as arena
from mojolearn._array import Array
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_device_cache_native_state import _real

#: Words of the one held upload.
WORDS = 3


@pytest.fixture
def held():
    cache = arena.DeviceCache(_real(), 'x_prep')
    buffer = Array.from_list([1., 2., 3.], '<f4')
    slot = cache.id_of(buffer)
    yield SimpleNamespace(cache=cache, slot=slot, address=int(buffer._addr), buffer=buffer)
    cache.close()


def _source(held, address=None, size=WORDS, itemsize=4):
    return SimpleNamespace(_addr=held.address if address is None else address, size=size, itemsize=itemsize)


def _expected(held, source):
    # The independent test oracle (Python conversion), outside the product.
    if getattr(source, 'itemsize', 4) != 4:
        raise ValueError('DeviceCache: a store slot holds 4-byte words')
    key = (int(source._addr), int(source.size))
    return held.slot if key == (held.address, WORDS) else None


def _refused(audit, calls):
    assert isinstance(audit.exception, TypeError), audit.report()
    assert calls == [], audit.report()
    assert not audit.report()['python_frames'], audit.report()
    assert not audit.report()['forbidden_native_calls'], audit.report()


def _clean(audit):
    assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls'], audit.report()


@pytest.mark.parametrize('kind', ['array', 'namespace', 'dictionary', 'slots', 'class_fields',
                                  'default_itemsize', 'shadowed_dict_property'])
def test_cache_get_preserves_ordinary_metadata_layouts(kind, held):
    address = held.address
    if kind == 'array':
        source = held.buffer
    elif kind == 'namespace':
        source = _source(held)
    elif kind == 'slots':
        class Buffer:
            __slots__ = ('_addr', 'size', 'itemsize')
        source = Buffer()
        source._addr, source.size, source.itemsize = address, WORDS, 4
    else:
        class Buffer:
            if kind == 'shadowed_dict_property':
                @property
                def __dict__(self):
                    raise AssertionError('unrelated __dict__ property was read')
        if kind == 'class_fields':
            Buffer._addr, Buffer.size, Buffer.itemsize = address, WORDS, 4
        source = Buffer()
        if kind != 'class_fields':
            source._addr, source.size = address, WORDS
            if kind != 'default_itemsize':
                source.itemsize = 4
    assert audit_native_call(held.cache.get, source).assert_native() == held.slot
    assert audit_native_call(held.cache.__len__).assert_native() == 1
    # `get` counts no hit and uploads nothing.
    assert (held.cache.uploads, held.cache.hits) == (1, 0)
    assert audit_native_call(held.cache.get, _source(held, address + 4)).assert_native() is None


@pytest.mark.parametrize('kind', ['exact', 'subclass', 'nested_subclass'])
def test_cache_get_preserves_native_namespace_attribute_slots(kind, held):
    class Namespace(SimpleNamespace):
        pass

    class Nested(Namespace):
        pass
    cls = {'exact': SimpleNamespace, 'subclass': Namespace, 'nested_subclass': Nested}[kind]
    source = cls(_addr=held.address, size=WORDS, itemsize=4)
    assert audit_native_call(held.cache.get, source).assert_native() == held.slot
    assert audit_native_call(held.cache.get, cls(_addr=held.address + 4, size=WORDS)).assert_native() is None
    assert audit_native_call(held.cache.__len__).assert_native() == 1


@pytest.mark.parametrize('hook', ['__getattribute__', '__getattr__', '_addr'])
def test_cache_get_refuses_namespace_subclass_hooks_before_execution(hook, held):
    calls = []

    def forbidden(*args):
        calls.append(hook)
        raise AssertionError('namespace application hook ran')
    namespace = type('Namespace', (SimpleNamespace,),
                     {hook: property(forbidden) if hook == '_addr' else forbidden})
    source = namespace(_addr=held.address, size=WORDS, itemsize=4)
    _refused(audit_native_call(held.cache.get, source), calls)


def test_cache_get_keeps_foreign_namespace_descriptor_owner_check(held):
    class Foreign:
        __getattribute__ = SimpleNamespace.__getattribute__
    source = Foreign()
    source._addr, source.size = held.address, WORDS
    # 3.11's namespace-owned wrapper requires the correct owner type;
    # newer CPython exposes object's ordinary inherited descriptor.
    if SimpleNamespace.__getattribute__ is object.__getattribute__:
        assert audit_native_call(held.cache.get, source).assert_native() == held.slot
    else:
        with pytest.raises(TypeError):
            _ = source._addr
        _refused(audit_native_call(held.cache.get, source), [])


@pytest.mark.parametrize('kind', ['int', 'float', 'str', 'bytes', 'numpy_int', 'numpy_float',
                                  'numpy_longlong', 'numpy_0d', 'numpy_object_0d', 'raw_void'])
def test_cache_get_preserves_native_integer_conversion(kind, held):
    convert = {'int': int, 'float': float, 'str': str, 'bytes': lambda x: str(x).encode(),
               'numpy_int': np.int64, 'numpy_float': np.float64, 'numpy_longlong': np.longlong,
               'numpy_0d': np.array, 'numpy_object_0d': lambda x: np.array(x, dtype=object),
               'raw_void': lambda x: np.void(str(x).encode())}[kind]
    source = _source(held, convert(held.address), convert(WORDS))
    assert audit_native_call(held.cache.get, source).assert_native() == held.slot


@pytest.mark.parametrize('field,dtype', [('_addr', 'g'), ('size', 'g'),
                                        ('itemsize', 'g'), ('itemsize', 'G')])
def test_cache_get_decodes_non_native_extended_metadata(field, dtype, held):
    source = _source(held)
    value = np.asarray(getattr(source, field), dtype=np.dtype(dtype).newbyteorder('>'))
    setattr(source, field, value)
    before = value.tobytes()
    assert audit_native_call(held.cache.get, source).assert_native() == held.slot
    assert getattr(source, field) is value and value.tobytes() == before


def test_cache_get_on_a_subclass_reads_the_same_native_state():
    class Cache(arena.DeviceCache):
        __slots__ = ()
    cache = Cache(_real(), 'x_prep')
    buffer = Array.from_list([1., 2., 3.], '<f4')
    try:
        slot = cache.id_of(buffer)
        assert audit_native_call(cache.get, buffer).assert_native() == slot
        assert audit_native_call(cache.__len__).assert_native() == 1
    finally:
        cache.close()


@pytest.mark.parametrize('itemsize', [4.0, 4+0j, np.int64(4), np.float32(4),
                                      np.array(4), np.array([4]), np.array(4, dtype=object)])
def test_cache_itemsize_equality_preserves_native_values(itemsize, held):
    assert audit_native_call(held.cache.get, _source(held, itemsize=itemsize)).assert_native() == held.slot


def test_cache_itemsize_comparison_avoids_numpy_numeric_dispatch(held):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append('numeric_ops')
        raise AssertionError('NumPy numeric dispatch called Python')
    # NumPy 2 removed this mutable 1.x dispatch table. Exercise the same
    # scalar cases there; only 1.x receives the explicit callback poison.
    configure = getattr(np, 'set_numeric_ops', None)
    saved = None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        if configure is not None:
            saved = configure(equal=forbidden, not_equal=forbidden)
        try:
            for dtype in ('i1', 'u1', 'i2', 'u2', 'i4', 'u4', 'i8', 'u8',
                          'q', 'Q', 'f2', 'f4', 'f8', 'g', 'c8', 'c16', 'G'):
                value = np.dtype(dtype).type(4)
                assert audit_native_call(held.cache.get, _source(held, itemsize=value)).assert_native() == held.slot
            assert calls == []
        finally:
            if saved is not None:
                configure(**saved)


@pytest.mark.parametrize('itemsize', [8, 4.1, '4', b'4', np.str_('4'), np.dtype('i4'), None])
def test_cache_itemsize_refusal_preserves_value_error(itemsize, held):
    audit = audit_native_call(held.cache.get, _source(held, itemsize=itemsize))
    assert isinstance(audit.exception, ValueError), audit.report()
    assert str(audit.exception) == 'DeviceCache: a store slot holds 4-byte words'
    _clean(audit)


@pytest.mark.parametrize('kind', ['empty_itemsize', 'timedelta_itemsize', 'vector_address'])
def test_cache_metadata_preserves_versioned_warning_and_error_contracts(kind, held):
    source = (_source(held, itemsize=np.array([])) if kind == 'empty_itemsize' else
              _source(held, itemsize=np.timedelta64(4, 's')) if kind == 'timedelta_itemsize' else
              _source(held, address=np.array([held.address])))
    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter('always')
        try:
            expected, expected_error = _expected(held, source), None
        except Exception as error:
            expected, expected_error = None, error
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter('always')
        audit = audit_native_call(held.cache.get, source)
    if expected_error is None:
        assert audit.assert_native() == expected
    else:
        assert type(audit.exception) is type(expected_error), audit.report()
        _clean(audit)
    assert [(w.category, str(w.message)) for w in actual_warnings] == [
        (w.category, str(w.message)) for w in expected_warnings]


@pytest.mark.parametrize('kind', ['complex_address', 'timedelta_itemsize'])
def test_cache_scalar_warning_error_action_preserves_exact_category(kind, held):
    source = (_source(held, address=np.complex128(held.address + 1j)) if kind == 'complex_address' else
              _source(held, itemsize=np.timedelta64(4, 's')))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        try:
            _expected(held, source)
            expected_error = None
        except Exception as error:
            expected_error = error
        audit = audit_native_call(held.cache.get, source)
    if expected_error is None:
        assert audit.assert_native() == held.slot
    else:
        assert type(audit.exception) is type(expected_error), audit.report()
        assert str(audit.exception) == str(expected_error)
        _clean(audit)


def test_cache_scalar_warning_refuses_application_display_hook(monkeypatch, held):
    calls = []

    def display(*args, **kwargs):
        calls.append('display')
    with warnings.catch_warnings(), monkeypatch.context() as patch:
        warnings.simplefilter('always')
        patch.setattr(warnings, 'showwarning', display)
        audit = audit_native_call(held.cache.get, _source(held, address=np.complex128(held.address + 1j)))
    _refused(audit, calls)


@pytest.mark.parametrize('kind', ['attribute', 'int', 'itemsize_eq', 'row_key'])
def test_cache_get_refuses_application_hooks_before_execution(kind, held):
    calls = []
    source = _source(held)
    address = held.address

    class Hook:
        def __int__(self):
            calls.append('int')
            return address

        def __eq__(self, other):
            calls.append('eq')
            return True

        def __hash__(self):
            calls.append('hash')
            return hash((address, WORDS))
    if kind == 'attribute':
        class Foreign:
            itemsize, size = 4, WORDS

            @property
            def _addr(self):
                calls.append('attribute')
                return address
        source = Foreign()
    elif kind == 'int':
        source._addr = Hook()
    elif kind == 'itemsize_eq':
        source.itemsize = Hook()
    else:
        class Key(str):
            __hash__ = str.__hash__

            def __eq__(self, other):
                calls.append('key_eq')
                return str.__eq__(self, other)
        fields = vars(source)
        fields.pop('_addr')
        fields[Key('_addr')] = address
        # Keep the exact poisoned key in place; no normalization by an
        # equal existing ordinary string during fixture construction.
        assert any(type(key) is Key for key in fields)
    calls.clear()
    _refused(audit_native_call(held.cache.get, source), calls)


@pytest.mark.parametrize('kind', ['missing_address', 'missing_size', 'invalid_integer'])
def test_cache_native_errors_preserve_existing_exception_classes(kind, held):
    source = _source(held)
    if kind == 'missing_address':
        del source._addr
    elif kind == 'missing_size':
        del source.size
    else:
        source._addr = 'not-an-int'
    expected = {'missing_address': AttributeError, 'missing_size': AttributeError,
                'invalid_integer': ValueError}[kind]
    audit = audit_native_call(held.cache.get, source)
    assert isinstance(audit.exception, expected), audit.report()
    _clean(audit)


def test_cache_reads_preserve_numpy_free_startup():
    _real()
    # NumPy is blocked in the shared child (installed NumPy would otherwise be
    # importable by package dependencies); the case builds its own cache and
    # changes no process-wide state, so it shares one child per test process.
    code = r'''
import sys
from types import SimpleNamespace
from mojolearn import _arena_io as a, _backend
from mojolearn import model_selection  # connects the arena row/cache doors (_native_arena_rows.connect); families load on first touch
from mojolearn._array import Array
from mojolearn._verify_native_runtime import audit_native_call
assert 'numpy' not in sys.modules
c=a.DeviceCache(_backend.binding('_mojolearn_x_prep','identical'),'x_prep')
b=Array.from_list([1.,2.,3.],'<f4')
slot=c.id_of(b)
x=SimpleNamespace(_addr=str(b._addr),size='3')
assert audit_native_call(c.get,x).assert_native()==slot
assert audit_native_call(c.__len__).assert_native()==1
assert audit_native_call(a.resident.stats).assert_native() is None
c.close()
assert 'numpy' not in sys.modules
print('native reads remain NumPy-free')
'''
    from mojolearn.tests._fresh_process import BLOCK_NUMPY_IMPORTED, shared_child
    assert shared_child(BLOCK_NUMPY_IMPORTED).run(code).strip() == 'native reads remain NumPy-free'


@pytest.mark.parametrize('dtype', ['i1', 'u1', 'i2', 'u2', 'i4', 'u4', 'i8', 'u8', 'q', 'Q',
                                  'f2', 'f4', 'f8', 'g', 'c8', 'c16', 'G'])
@pytest.mark.parametrize('array', [False, True])
def test_lookup_numeric_itemsize_uses_exact_native_equality(dtype, array, held):
    scalar = np.dtype(dtype).type(4)
    value = np.asarray(scalar) if array else scalar
    assert audit_native_call(held.cache.get, _source(held, itemsize=value)).assert_native() == held.slot
    if np.dtype(dtype).kind in 'fc':
        near = np.nextafter(np.dtype(dtype).type(4).real, np.dtype(dtype).type(5).real)
        supplied = np.asarray(near) if array else near
        audit = audit_native_call(held.cache.get, _source(held, itemsize=supplied))
        assert type(audit.exception) is ValueError, audit.report()
        _clean(audit)


@pytest.mark.parametrize('dtype', ['i8', 'u8', 'q', 'Q', 'f8', 'g'])
@pytest.mark.parametrize('array', [False, True])
def test_lookup_native_integer_conversion_preserves_full_precision_of_the_address(dtype, array, held):
    dt = np.dtype(dtype)
    # The held address converted exactly names the buffer.
    exact = np.asarray(held.address, dtype=dt) if array else dt.type(held.address)
    assert int(exact) == held.address
    assert audit_native_call(held.cache.get, _source(held, address=exact)).assert_native() == held.slot
    # One step below truncates to another address: no rounding back up.
    neighbor = (dt.type(held.address - 1) if dt.kind in 'iu' else
                np.nextafter(dt.type(held.address), dt.type(0)))
    value = np.asarray(neighbor, dtype=dt) if array else dt.type(neighbor)
    assert int(value) != held.address
    assert audit_native_call(held.cache.get, _source(held, address=value)).assert_native() is None
    if dt.kind in 'iu':
        top = np.asarray(np.iinfo(dt).max, dtype=dt) if array else dt.type(np.iinfo(dt).max)
        assert audit_native_call(held.cache.get, _source(held, address=top)).assert_native() is None


@pytest.mark.parametrize('dtype', ['f2', 'f4'])
@pytest.mark.parametrize('array', [False, True])
def test_lookup_native_integer_conversion_truncates_short_floats(dtype, array, held):
    dt = np.dtype(dtype)
    exact = np.asarray(WORDS, dtype=dt) if array else dt.type(WORDS)
    assert audit_native_call(held.cache.get, _source(held, size=exact)).assert_native() == held.slot
    below = np.nextafter(dt.type(WORDS), dt.type(0))
    value = np.asarray(below, dtype=dt) if array else below
    assert int(value) == WORDS - 1
    assert audit_native_call(held.cache.get, _source(held, size=value)).assert_native() is None


@pytest.mark.parametrize('dtype', ['f2', 'f4', 'f8', 'g'])
@pytest.mark.parametrize('value', ['nan', 'inf', '-inf'])
def test_lookup_native_nonfinite_integer_errors_keep_exact_contract(dtype, value, held):
    source = _source(held, address=np.asarray(value, dtype=dtype))
    with pytest.raises((ValueError, OverflowError)) as expected:
        int(source._addr)
    audit = audit_native_call(held.cache.get, source)
    assert type(audit.exception) is type(expected.value), audit.report()
    assert str(audit.exception) == str(expected.value)
    _clean(audit)


@pytest.mark.parametrize('name', ['list', 'tuple', 'dict', 'int', 'float', 'bytes', 'object', 'str'])
def test_lookup_uses_retained_native_builtin_identities(name):
    _real()
    code = r"""
import builtins, json
from types import SimpleNamespace
import numpy as np
from mojolearn import _arena_io as a, _backend
from mojolearn import model_selection  # connects the arena row/cache doors (_native_arena_rows.connect); families load on first touch
from mojolearn._array import Array
from mojolearn._verify_native_runtime import audit_native_call
c=a.DeviceCache(_backend.binding('_mojolearn_x_prep','identical'),'x_prep')
b=Array.from_list([1.,2.,3.],'<f4')
slot=c.id_of(b)
# Prime no native dtype registry: the first numeric lookup must also retain
# fixed builtin metadata identities when NumPy was imported before mojolearn.
x=SimpleNamespace(_addr=np.int64(b._addr),size=np.asarray(3),itemsize=np.float32(4))
y=SimpleNamespace(_addr=str(b._addr),size=b'3',itemsize=4)
name=NAME
saved=getattr(builtins,name)
calls=[]
def forbidden(*args,**kwargs):
    calls.append(True)
    raise AssertionError('mutable builtin invoked')
setattr(builtins,name,forbidden)
try:
    values=(c.get(x),c.get(y))
finally:
    setattr(builtins,name,saved)
assert values==(slot,slot) and not calls
assert audit_native_call(c.get,x).assert_native()==slot
assert audit_native_call(c.get,y).assert_native()==slot
c.close()
print(json.dumps({'calls':calls,'values':values}))
""".replace('NAME', repr(name))
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True,
                            env=dict(os.environ))
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['calls'] == []


@pytest.mark.parametrize('dtype,word', [('m8[ns]', 8), ('m8[Y]', 8), ('m8[M]', 8),
    ('m8[us]', 8), ('m8[D]', 1000000000), ('m8[2ns]', -8),
    ('M8[ns]', 8), ('M8[Y]', -1970), ('M8[Y]', -1969),
    ('M8[D]', -719163), ('M8[D]', -719162), ('M8[D]', 2932897),
    ('M8[3D]', -239721), ('M8[us]', 8), ('M8[ns]', -(2**63)), ('m8[ns]', None), ('M8[ns]', None)])
@pytest.mark.parametrize('array', [False, True])
def test_lookup_temporal_integer_metadata_preserves_storage_and_error_contract(dtype, word, array, held):
    # `word=None` stands for the held address itself: a positive hit.
    source = np.asarray(held.address if word is None else word, dtype=dtype)
    value = source if array else source[()]
    candidate = _source(held, address=value)
    try:
        expected, failure = _expected(held, candidate), None
    except Exception as error:
        expected, failure = None, error
    if word is None:
        assert failure is None and expected == held.slot
    audit = audit_native_call(held.cache.get, candidate)
    if failure is None:
        assert audit.assert_native() == expected
    else:
        assert type(audit.exception) is type(failure), audit.report()
        assert str(audit.exception) == str(failure)
        _clean(audit)


@pytest.mark.parametrize('kind', ['object_array_address', 'object_array_itemsize', 'dtype_module_key'])
def test_lookup_rejects_transitive_numpy_metadata_hooks(kind, held):
    calls = []
    source = _source(held)
    address = held.address

    class Hook:
        def __int__(self):
            calls.append('int')
            return address

        def __eq__(self, other):
            calls.append('eq')
            return True

        def __ne__(self, other):
            calls.append('ne')
            return False
    restore = None
    if kind == 'object_array_address':
        source._addr = np.asarray(Hook(), dtype=object)
    elif kind == 'object_array_itemsize':
        source.itemsize = np.asarray(Hook(), dtype=object)
    else:
        class Key(str):
            __hash__ = str.__hash__

            def __eq__(self, other):
                calls.append('module_key')
                return str.__eq__(self, other)
        fields = vars(np)
        saved = fields.pop('dtype')
        fields[Key('dtype')] = saved
        assert any(type(key) is Key for key in fields)
        restore = (fields, saved)
        source.itemsize = np.int64(4)
    calls.clear()
    try:
        audit = audit_native_call(held.cache.get, source)
    finally:
        if restore is not None:
            fields, saved = restore
            # Removing the retained exact key avoids a fixture equality call.
            for key in tuple(fields):
                if type(key) is Key:
                    del fields[key]
                    break
            fields['dtype'] = saved
    _refused(audit, calls)


@pytest.mark.parametrize('base', [int, float, complex, str, bytes, bytearray])
@pytest.mark.parametrize('field', ['_addr', 'itemsize'])
def test_lookup_preserves_benign_inherited_builtin_metadata(base, field, held):
    class Native(base):
        pass

    class Nested(Native):
        pass
    number = 4 if field == 'itemsize' else held.address
    value = Nested(str(number).encode()) if base in (bytes, bytearray) else Nested(number)
    source = _source(held)
    setattr(source, field, value)
    try:
        expected, failure = _expected(held, source), None
    except Exception as error:
        expected, failure = None, error
    audit = audit_native_call(held.cache.get, source)
    if failure is None:
        assert audit.assert_native() == expected
    else:
        assert type(audit.exception) is type(failure), audit.report()
        assert str(audit.exception) == str(failure)
        _clean(audit)


@pytest.mark.parametrize('base,hook,field', [(int, '__int__', '_addr'),
    (float, '__int__', '_addr'), (int, '__index__', '_addr'),
    (str, '__int__', '_addr'), (str, '__repr__', '_addr'),
    (bytes, '__repr__', '_addr'), (float, '__eq__', 'itemsize'),
    (complex, '__ne__', 'itemsize')])
def test_lookup_refuses_overridden_primitive_protocols_before_execution(base, hook, field, held):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(hook)
        raise AssertionError('primitive protocol callback ran')
    Native = type('Native', (base,), {hook: forbidden})
    value = Native(b'bad') if base is bytes else Native('bad') if base is str else Native(4)
    source = _source(held)
    setattr(source, field, value)
    _refused(audit_native_call(held.cache.get, source), calls)
