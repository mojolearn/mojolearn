# SPDX-License-Identifier: Apache-2.0
"""Whole public cache lookup; cache construction is test setup."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
from mojolearn import _arena_io as arena
from mojolearn._array import Array
from mojolearn._buffer import _native
from mojolearn._verify_native_runtime import audit_native_call


def _cache():
    # Metadata fixture only: no put/free entry executes and no device slot is
    # acquired. This suite does not qualify the still-Python constructor.
    return arena.DeviceCache(SimpleNamespace(x_dev_put=_native('msel_put'),
                                            x_dev_free=_native('msel_free')), 'x')


def _source(address=8, size=3, itemsize=4):
    return SimpleNamespace(_addr=address, size=size, itemsize=itemsize)


def _refused(audit, calls):
    assert isinstance(audit.exception, TypeError), audit.report()
    assert calls == [], audit.report()
    assert not audit.report()['python_frames'], audit.report()
    assert not audit.report()['forbidden_native_calls'], audit.report()


@pytest.mark.parametrize('kind', ['array', 'namespace', 'dictionary', 'slots', 'class_fields',
                                  'default_itemsize', 'shadowed_dict_property'])
def test_cache_get_preserves_ordinary_metadata_layouts(kind):
    cache = _cache()
    if kind == 'array':
        source = Array.from_list([1, 2, 3], '<f4')
    elif kind == 'namespace':
        source = _source()
    elif kind == 'slots':
        class Buffer:
            __slots__ = ('_addr', 'size', 'itemsize')
        source = Buffer()
        source._addr, source.size, source.itemsize = 8, 3, 4
    else:
        class Buffer:
            if kind == 'shadowed_dict_property':
                @property
                def __dict__(self):
                    raise AssertionError('unrelated __dict__ property was read')
        if kind == 'class_fields':
            Buffer._addr, Buffer.size, Buffer.itemsize = 8, 3, 4
        source = Buffer()
        if kind != 'class_fields':
            source._addr, source.size = 8, 3
            if kind != 'default_itemsize':
                source.itemsize = 4
    key = (int(source._addr), int(source.size))
    owner = object()
    cache._entries[key] = (19, owner)
    before = cache._entries
    assert audit_native_call(cache.get, source).assert_native() == 19
    assert audit_native_call(cache.__len__).assert_native() == 1
    assert cache._entries is before and cache._entries[key][1] is owner
    assert (cache.uploads, cache.hits) == (0, 0)
    assert audit_native_call(cache.get, _source(999)).assert_native() is None


@pytest.mark.parametrize('kind', ['exact', 'subclass', 'nested_subclass'])
def test_cache_get_preserves_native_namespace_attribute_slots(kind):
    class Namespace(SimpleNamespace):
        pass
    class Nested(Namespace):
        pass
    cls = {'exact': SimpleNamespace, 'subclass': Namespace, 'nested_subclass': Nested}[kind]
    source = cls(_addr=8, size=3, itemsize=4)
    cache = _cache()
    cache._entries[(8, 3)] = (19, source)
    assert audit_native_call(cache.get, source).assert_native() == 19
    assert audit_native_call(cache.get, cls(_addr=9, size=3)).assert_native() is None
    assert cache._entries[(8, 3)][1] is source
    assert audit_native_call(cache.__len__).assert_native() == 1


@pytest.mark.parametrize('hook', ['__getattribute__', '__getattr__', '_addr'])
def test_cache_get_refuses_namespace_subclass_hooks_before_execution(hook):
    calls = []
    def forbidden(*args):
        calls.append(hook)
        raise AssertionError('namespace application hook ran')
    namespace = type('Namespace', (SimpleNamespace,),
                     {hook: property(forbidden) if hook == '_addr' else forbidden})
    source = namespace(_addr=8, size=3, itemsize=4)
    audit = audit_native_call(_cache().get, source)
    _refused(audit, calls)


def test_cache_get_keeps_foreign_namespace_descriptor_owner_check():
    class Foreign:
        __getattribute__ = SimpleNamespace.__getattribute__
    source = Foreign()
    source._addr, source.size = 8, 3
    cache = _cache()
    cache._entries[(8, 3)] = (19, None)
    # 3.11's namespace-owned wrapper requires the correct owner type;
    # newer CPython exposes object's ordinary inherited descriptor.
    if SimpleNamespace.__getattribute__ is object.__getattribute__:
        assert audit_native_call(cache.get, source).assert_native() == 19
    else:
        with pytest.raises(TypeError):
            _ = source._addr
        _refused(audit_native_call(cache.get, source), [])


@pytest.mark.parametrize('kind', ['int', 'float', 'str', 'bytes', 'numpy_int', 'numpy_float',
                                  'numpy_longlong', 'numpy_0d', 'numpy_object_0d', 'raw_void'])
def test_cache_get_preserves_native_integer_conversion(kind):
    convert = {'int': int, 'float': float, 'str': str, 'bytes': lambda x: str(x).encode(),
               'numpy_int': np.int64, 'numpy_float': np.float64, 'numpy_longlong': np.longlong,
               'numpy_0d': np.array, 'numpy_object_0d': lambda x: np.array(x, dtype=object),
               'raw_void': lambda x: np.void(str(x).encode())}[kind]
    cache = _cache()
    cache._entries[(8, 3)] = (19, None)
    assert audit_native_call(cache.get, _source(convert(8), convert(3))).assert_native() == 19


@pytest.mark.parametrize('field,dtype', [('_addr', 'g'), ('size', 'g'),
                                        ('itemsize', 'g'), ('itemsize', 'G')])
def test_cache_get_decodes_non_native_extended_metadata(field, dtype):
    cache, source = _cache(), _source()
    cache._entries[(8, 3)] = (19, source)
    value = np.asarray(getattr(source, field), dtype=np.dtype(dtype).newbyteorder('>'))
    setattr(source, field, value)
    before = value.tobytes()
    assert audit_native_call(cache.get, source).assert_native() == 19
    assert cache._entries[(8, 3)][1] is source
    assert getattr(source, field) is value and value.tobytes() == before


@pytest.mark.parametrize('kind', ['subclass', 'retained_instance_key'])
def test_cache_get_preserves_ordinary_cache_variants(kind):
    cache = _cache()
    if kind == 'subclass':
        class Cache(arena.DeviceCache):
            pass
        cache.__class__ = Cache
    else:
        cache._key = arena.DeviceCache._key
    cache._entries[(8, 3)] = [19, None]
    assert audit_native_call(cache.get, _source()).assert_native() == 19
    assert audit_native_call(cache.__len__).assert_native() == 1


@pytest.mark.parametrize('itemsize', [4.0, 4+0j, np.int64(4), np.float32(4),
                                      np.array(4), np.array([4]), np.array(4, dtype=object)])
def test_cache_itemsize_equality_preserves_native_values(itemsize):
    cache = _cache()
    cache._entries[(8, 3)] = (19, None)
    assert audit_native_call(cache.get, _source(itemsize=itemsize)).assert_native() == 19


def test_cache_itemsize_comparison_avoids_numpy_numeric_dispatch():
    cache, calls = _cache(), []
    cache._entries[(8, 3)] = (19, None)
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
                assert audit_native_call(cache.get, _source(itemsize=value)).assert_native() == 19
            assert calls == []
        finally:
            if saved is not None:
                configure(**saved)


@pytest.mark.parametrize('itemsize', [8, 4.1, '4', b'4', np.str_('4'), np.dtype('i4'), None])
def test_cache_itemsize_refusal_preserves_value_error(itemsize):
    audit = audit_native_call(_cache().get, _source(itemsize=itemsize))
    assert isinstance(audit.exception, ValueError), audit.report()
    assert str(audit.exception) == 'DeviceCache: a store slot holds 4-byte words'
    assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']


@pytest.mark.parametrize('kind', ['empty_itemsize', 'timedelta_itemsize', 'vector_address'])
def test_cache_metadata_preserves_versioned_warning_and_error_contracts(kind):
    cache = _cache()
    cache._entries[(8, 3)] = (19, None)
    source = (_source(itemsize=np.array([])) if kind == 'empty_itemsize' else
              _source(itemsize=np.timedelta64(4, 's')) if kind == 'timedelta_itemsize' else
              _source(address=np.array([8])))
    # The independent test oracle executes outside the product boundary.
    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter('always')
        try:
            if getattr(source, 'itemsize', 4) != 4:
                raise ValueError('DeviceCache: a store slot holds 4-byte words')
            expected = cache._entries.get((int(source._addr), int(source.size)))
            expected = None if expected is None else expected[0]
            expected_error = None
        except Exception as error:
            expected, expected_error = None, error
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter('always')
        audit = audit_native_call(cache.get, source)
    if expected_error is None:
        assert audit.assert_native() == expected
    else:
        assert type(audit.exception) is type(expected_error), audit.report()
        assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']
    assert [(w.category, str(w.message)) for w in actual_warnings] == [
        (w.category, str(w.message)) for w in expected_warnings]


@pytest.mark.parametrize('kind', ['complex_address', 'timedelta_itemsize'])
def test_cache_scalar_warning_error_action_preserves_exact_category(kind):
    cache = _cache()
    cache._entries[(8, 3)] = (19, None)
    source = (_source(address=np.complex128(8+1j)) if kind == 'complex_address' else
              _source(itemsize=np.timedelta64(4, 's')))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        try:
            if source.itemsize != 4:
                raise ValueError('DeviceCache: a store slot holds 4-byte words')
            int(source._addr)
            expected_error = None
        except Exception as error:
            expected_error = error
        audit = audit_native_call(cache.get, source)
    if expected_error is None:
        assert audit.assert_native() == 19
    else:
        assert type(audit.exception) is type(expected_error), audit.report()
        assert str(audit.exception) == str(expected_error)
        assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']


def test_cache_scalar_warning_refuses_application_display_hook(monkeypatch):
    cache, calls = _cache(), []
    def display(*args, **kwargs):
        calls.append('display')
    with warnings.catch_warnings(), monkeypatch.context() as patch:
        warnings.simplefilter('always')
        patch.setattr(warnings, 'showwarning', display)
        audit = audit_native_call(cache.get, _source(address=np.complex128(8+1j)))
    _refused(audit, calls)


@pytest.mark.parametrize('kind', ['attribute', 'getitem', 'int', 'itemsize_eq',
                                  'row_key', 'stored_key', 'instance_key', 'subclass_key'])
def test_cache_get_refuses_application_hooks_before_execution(kind):
    calls = []
    cache, source = _cache(), _source()
    cache._entries[(8, 3)] = (19, None)
    class Hook:
        def __int__(self):
            calls.append('int')
            return 8
        def __eq__(self, other):
            calls.append('eq')
            return True
        def __hash__(self):
            calls.append('hash')
            return hash((8, 3))
        def __getitem__(self, key):
            calls.append('getitem')
            return 19
    if kind == 'attribute':
        class Foreign:
            itemsize, size = 4, 3
            @property
            def _addr(self):
                calls.append('attribute')
                return 8
        source = Foreign()
    elif kind == 'int':
        source._addr = Hook()
    elif kind == 'itemsize_eq':
        source.itemsize = Hook()
    elif kind == 'getitem':
        cache._entries[(8, 3)] = Hook()
    elif kind == 'stored_key':
        cache._entries = {Hook(): (19, None)}
    elif kind == 'row_key':
        class Key(str):
            __hash__ = str.__hash__
            def __eq__(self, other):
                calls.append('key_eq')
                return str.__eq__(self, other)
        fields = vars(source)
        fields.pop('_addr')
        fields[Key('_addr')] = 8
        # Keep the exact poisoned key in place; no normalization by an
        # equal existing ordinary string during fixture construction.
        assert any(type(key) is Key for key in fields)
    elif kind == 'instance_key':
        cache._key = lambda value: calls.append('key') or (8, 3)
    else:
        class Cache(arena.DeviceCache):
            @staticmethod
            def _key(value):
                calls.append('subclass_key')
                return 8, 3
        cache.__class__ = Cache
    calls.clear()
    _refused(audit_native_call(cache.get, source), calls)


@pytest.mark.parametrize('operation', ['get'])
def test_cache_field_descriptor_is_refused_before_execution(operation, monkeypatch):
    cache, calls = _cache(), []
    def forbidden(self):
        calls.append('fields')
        return {}
    with monkeypatch.context() as patch:
        patch.setattr(arena.DeviceCache, '_entries', property(forbidden), raising=False)
        audit = (audit_native_call(cache.get, _source()) if operation == 'get' else
                 audit_native_call(cache.__len__))
    _refused(audit, calls)


@pytest.mark.parametrize('kind', ['missing_address', 'missing_size', 'empty_entry', 'invalid_integer'])
def test_cache_native_errors_preserve_existing_exception_classes(kind):
    cache, source = _cache(), _source()
    if kind == 'missing_address':
        del source._addr
    elif kind == 'missing_size':
        del source.size
    elif kind == 'empty_entry':
        cache._entries[(8, 3)] = ()
    else:
        source._addr = 'not-an-int'
    expected = {'missing_address': AttributeError, 'missing_size': AttributeError,
                'empty_entry': IndexError, 'invalid_integer': ValueError}[kind]
    audit = audit_native_call(cache.get, source)
    assert isinstance(audit.exception, expected), audit.report()
    assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']


def test_cache_reads_preserve_numpy_free_startup():
    code = r'''
import sys, importlib.abc
# Package dependencies may import installed NumPy. Model actual absence rather
# than assuming a fresh subprocess makes an available optional dependency absent.
class BlockNumPy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'numpy' or fullname.startswith('numpy.'):
            raise ModuleNotFoundError('NumPy intentionally unavailable', name='numpy')
sys.meta_path.insert(0, BlockNumPy())
from types import SimpleNamespace
from mojolearn import _arena_io as a
from mojolearn._buffer import _native
from mojolearn._verify_native_runtime import audit_native_call
assert 'numpy' not in sys.modules
c=a.DeviceCache(SimpleNamespace(x_dev_put=_native('msel_put'),x_dev_free=_native('msel_free')),'x')
c._entries[(8,3)]=(19,None)
x=SimpleNamespace(_addr='8',size='3')
assert audit_native_call(c.get,x).assert_native()==19
assert audit_native_call(c.__len__).assert_native()==1
assert audit_native_call(a.resident.stats).assert_native() is None
assert 'numpy' not in sys.modules
print('native reads remain NumPy-free')
'''
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True,
                            env=dict(os.environ))
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'native reads remain NumPy-free'



@pytest.mark.parametrize('dtype', ['i1', 'u1', 'i2', 'u2', 'i4', 'u4', 'i8', 'u8', 'q', 'Q',
                                  'f2', 'f4', 'f8', 'g', 'c8', 'c16', 'G'])
@pytest.mark.parametrize('array', [False, True])
def test_lookup_numeric_itemsize_uses_exact_native_equality(dtype, array):
    cache = _cache()
    cache._entries[(8, 3)] = (19, None)
    scalar = np.dtype(dtype).type(4)
    value = np.asarray(scalar) if array else scalar
    assert audit_native_call(cache.get, _source(itemsize=value)).assert_native() == 19
    if np.dtype(dtype).kind in 'fc':
        near = np.nextafter(np.dtype(dtype).type(4).real, np.dtype(dtype).type(5).real)
        supplied = np.asarray(near) if array else near
        audit = audit_native_call(cache.get, _source(itemsize=supplied))
        assert type(audit.exception) is ValueError, audit.report()
        assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']


@pytest.mark.parametrize('dtype', ['i8', 'u8', 'q', 'Q', 'f2', 'f4', 'f8', 'g'])
@pytest.mark.parametrize('array', [False, True])
def test_lookup_native_integer_conversion_preserves_full_precision(dtype, array):
    dt = np.dtype(dtype)
    scalar = (np.iinfo(dt).max if dt.kind in 'iu' else
              np.nextafter(dt.type(8), dt.type(0)))
    value = np.asarray(scalar, dtype=dt) if array else dt.type(scalar)
    expected = int(value)
    cache = _cache()
    sentinel = object()
    cache._entries[(expected, 3)] = (sentinel, None)
    assert audit_native_call(cache.get, _source(address=value)).assert_native() is sentinel


@pytest.mark.parametrize('dtype', ['f2', 'f4', 'f8', 'g'])
@pytest.mark.parametrize('value', ['nan', 'inf', '-inf'])
def test_lookup_native_nonfinite_integer_errors_keep_exact_contract(dtype, value):
    source = _source(address=np.asarray(value, dtype=dtype))
    with pytest.raises((ValueError, OverflowError)) as expected:
        int(source._addr)
    audit = audit_native_call(_cache().get, source)
    assert type(audit.exception) is type(expected.value), audit.report()
    assert str(audit.exception) == str(expected.value)
    assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']


@pytest.mark.parametrize('name', ['list', 'tuple', 'dict', 'int', 'float', 'bytes', 'object', 'str'])
def test_lookup_uses_retained_native_builtin_identities(name):
    code = r"""
import builtins, json
from types import SimpleNamespace
import numpy as np
from mojolearn import _arena_io as a
from mojolearn._buffer import _native
from mojolearn._verify_native_runtime import audit_native_call
c=a.DeviceCache(SimpleNamespace(x_dev_put=_native('msel_put'),x_dev_free=_native('msel_free')),'x')
c._entries[(8,3)]=(19,None)
# Prime no native dtype registry: the first numeric lookup must also retain
# fixed builtin metadata identities when NumPy was imported before mojolearn.
x=SimpleNamespace(_addr=np.int64(8),size=np.asarray(3),itemsize=np.float32(4))
y=SimpleNamespace(_addr='8',size=b'3',itemsize=4)
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
assert values==(19,19) and not calls
assert audit_native_call(c.get,x).assert_native()==19
assert audit_native_call(c.get,y).assert_native()==19
# Metadata fixture entries never acquired a device slot.
c._entries.clear()
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
    ('M8[3D]', -239721), ('M8[us]', 8), ('M8[ns]', -(2**63))])
@pytest.mark.parametrize('array', [False, True])
def test_lookup_temporal_integer_metadata_preserves_storage_and_error_contract(dtype, word, array):
    source = np.asarray(word, dtype=dtype)
    value = source if array else source[()]
    cache = _cache()
    try:
        expected = int(value)
        failure = None
    except Exception as error:
        failure = error
    if failure is None:
        cache._entries[(expected, 3)] = (19, None)
    audit = audit_native_call(cache.get, _source(address=value))
    if failure is None:
        assert audit.assert_native() == 19
    else:
        assert type(audit.exception) is type(failure), audit.report()
        assert str(audit.exception) == str(failure)
        assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']


@pytest.mark.parametrize('kind', ['object_array_address', 'object_array_itemsize', 'dtype_module_key'])
def test_lookup_rejects_transitive_numpy_metadata_hooks(kind):
    cache, calls = _cache(), []
    source = _source()
    class Hook:
        def __int__(self):
            calls.append('int')
            return 8
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
        audit = audit_native_call(cache.get, source)
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
def test_lookup_preserves_benign_inherited_builtin_metadata(base, field):
    class Native(base):
        pass
    class Nested(Native):
        pass
    number = 4 if field == 'itemsize' else 8
    value = Nested(str(number).encode()) if base in (bytes, bytearray) else Nested(number)
    source = _source()
    setattr(source, field, value)
    cache = _cache()
    cache._entries[(8, 3)] = (19, None)
    try:
        if source.itemsize != 4:
            raise ValueError('DeviceCache: a store slot holds 4-byte words')
        expected = cache._entries.get((int(source._addr), int(source.size)))
        expected = None if expected is None else expected[0]
        failure = None
    except Exception as error:
        failure = error
    audit = audit_native_call(cache.get, source)
    if failure is None:
        assert audit.assert_native() == expected
    else:
        assert type(audit.exception) is type(failure), audit.report()
        assert str(audit.exception) == str(failure)
        assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']


@pytest.mark.parametrize('base,hook,field', [(int, '__int__', '_addr'),
    (float, '__int__', '_addr'), (int, '__index__', '_addr'),
    (str, '__int__', '_addr'), (str, '__repr__', '_addr'),
    (bytes, '__repr__', '_addr'), (float, '__eq__', 'itemsize'),
    (complex, '__ne__', 'itemsize')])
def test_lookup_refuses_overridden_primitive_protocols_before_execution(base, hook, field):
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(hook)
        raise AssertionError('primitive protocol callback ran')
    Native = type('Native', (base,), {hook: forbidden})
    value = Native(b'bad') if base is bytes else Native('bad') if base is str else Native(4)
    source = _source()
    setattr(source, field, value)
    _refused(audit_native_call(_cache().get, source), calls)
