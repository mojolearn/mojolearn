# SPDX-License-Identifier: Apache-2.0
"""Actual native array allocation retains bytes and never calls audit hooks."""
import json
import subprocess
import sys
import textwrap

import pytest


def test_native_unicode_storage_strict_warning_setup_and_wire_bytes():
    program = textwrap.dedent('''\
        import array, gc, json, sys, warnings
        original_filters = warnings.filters
        before_filters = list(original_filters)
        from mojolearn import model_selection as ms
        from mojolearn import _native_array_storage as storage
        assert warnings.filters is original_filters
        assert warnings.filters == before_filters, (warnings.filters, before_filters)
        # Our import-only filter must not hide later application warnings,
        # including the identical message attributed to this private module.
        for message in ("The 'u' type code is deprecated and will be removed in Python 3.16",
                        'independent deprecation'):
            try:
                warnings.warn_explicit(message, DeprecationWarning, '<probe>', 1,
                                       module=storage.__name__)
            except DeprecationWarning as error:
                assert str(error) == message
            else:
                raise AssertionError('setup warning filter escaped its scope')
        active = False
        events = []
        def hook(event, args):
            if active:
                events.append(event)
        sys.addaudithook(hook)
        active = True
        control = array.array('i', [1])
        active = False
        assert events == ['array.__new__'], events
        events.clear()
        cases = 0
        for code in ('u', 'w'):
            if code not in array.typecodes:
                continue  # this interpreter never supported the typecode
            for text in ('', 'A\\0Ω😀'):
                # Test data uses its public constructor; only the operation
                # below is audited. Deprecation handling stays in test setup.
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', DeprecationWarning)
                    source = array.array(code, text)
                expected = source.tobytes()
                active = True
                result = ms._NATIVE(None, 'family_msel_parallel_wire', (source,), ms._NATIVE_CONTEXT)
                active = False
                assert events == [], (code, text, events)
                assert type(result) is array.array and result.typecode == code
                assert result is not source and result.tobytes() == expected
                assert result.tounicode() == text
                view = memoryview(result)
                del result, source
                gc.collect()
                assert view.tobytes() == expected
                view.release()
                cases += 1
        assert cases == 2 * sum(code in array.typecodes for code in ('u', 'w'))
        assert cases >= 2
        print(json.dumps(dict(cases=cases, audit_events=events, filters_restored=True)))
    ''')
    result = subprocess.run([sys.executable, '-Werror', '-c', program],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report['cases'] in (2, 4)
    assert report['audit_events'] == [] and report['filters_restored'] is True


@pytest.mark.parametrize('numpy_connected', [False, True])
def test_native_array_storage_audit_bytes_and_export_lifetimes(numpy_connected):
    # Audit hooks cannot be removed; every probe owns an independent process.
    program = textwrap.dedent('''\
        import array, gc, json, sys
        numpy_connected = sys.argv[1] == 'True'
        if numpy_connected:
            import numpy
        from mojolearn import Array, _backend
        from mojolearn import _native_array_storage as storage
        codec = _backend.load_host_module('_mojolearn_tokenizer_host').native_codec_call
        codec('initialize', (Array, None, None))
        events = []
        active = False
        def hook(event, args):
            if active:
                events.append(event)
        sys.addaudithook(hook)

        # Positive observer control: the old allocation mechanism is visible.
        active = True
        control = array.array('i', [7])
        active = False
        assert events == ['array.__new__'], events
        events.clear()
        seeds_before = {code: value.tobytes() for code, value in storage.ZERO_STORES.items()}

        cases = 0
        for dtype, width, code in (('<f4', 4, 'f'), ('<f8', 8, 'd'), ('<i4', 4, 'i'),
                                    ('<i8', 8, 'q'), ('<u4', 4, 'I'), ('<u1', 1, 'B'),
                                    ('<u2', 2, 'H'), ('<f2', 2, 'H'), ('<i1', 1, 'b')):
            for shape in ([2, 3], [0, 3]):
                for order in ('C', 'F'):
                    for readonly in (False, True):
                        count = shape[0] * shape[1]
                        # Independent raw bytes include arbitrary float bit patterns;
                        # decoding storage must preserve them without conversion.
                        raw = bytes((i * 37 + 11) % 256 for i in range(count * width))
                        expected = bytes(bytearray(raw))
                        header = dict(kind='array', begin=0, size=len(raw), shape=shape,
                                      dtype=dtype, order=order, readonly=readonly)
                        gc.collect()
                        active = True
                        result = codec('unpack', (header, raw, 1024))
                        active = False
                        assert events == [], (dtype, shape, order, readonly, events)
                        assert type(result) is Array
                        assert type(result._store) is array.array
                        assert result._store.typecode == code
                        assert result.shape == tuple(shape) and result.dtype == dtype
                        assert result.order == order and result._readonly is readonly
                        assert result.itemsize == width and result.nbytes == count * width
                        assert result.strides == ((shape[1] * width, width) if order == 'C'
                                                   else (width, shape[0] * width))
                        assert result._mv.tobytes() == expected
                        assert result._store is not storage.ZERO_STORES[code]
                        view = memoryview(result._store)
                        del result, header, raw
                        gc.collect()
                        assert view.tobytes() == expected  # export owns storage after result release
                        if count and not readonly:
                            view[0] = 7
                            assert view[0] == 7
                        view.release()
                        cases += 1
        assert cases == 72, cases
        assert {code: value.tobytes() for code, value in storage.ZERO_STORES.items()} == seeds_before

        # Python-visible metadata cannot authorize Python protocols in a
        # native operation, even when both the type and seed are replaced.
        hooks = []
        class FakeStorage:
            def __len__(self):
                hooks.append('len')
                return 1
            @property
            def typecode(self):
                hooks.append('typecode')
                return 'i'
            def __getitem__(self, index):
                hooks.append('getitem')
                return 0
            def __mul__(self, count):
                hooks.append('repeat')
                return self
        original_type, original_seeds = storage.ARRAY_TYPE, storage.ZERO_STORES
        storage.ARRAY_TYPE, storage.ZERO_STORES = FakeStorage, {'i': FakeStorage()}
        header = dict(kind='array', begin=0, size=4, shape=[1], dtype='<i4', order='C', readonly=False)
        try:
            active = True
            try:
                codec('unpack', (header, b'\\0' * 4, 1024))
            except ValueError as error:
                message = str(error)
                assert ('native module connection export changed: mojolearn._native_array_storage.ARRAY_TYPE' in message
                        or 'requires its extension type' in message), message
            else:
                raise AssertionError('substituted Python storage was admitted')
        finally:
            active = False
            storage.ARRAY_TYPE, storage.ZERO_STORES = original_type, original_seeds
        assert hooks == [] and events == [], (hooks, events)

        # Numeric zero equality is insufficient: negative zero must not seed
        # buffers that previously contained all-zero bytes.
        for code, dtype, width in (('f', '<f4', 4), ('d', '<f8', 8)):
            original = storage.ZERO_STORES[code]
            storage.ZERO_STORES[code] = array.array(code, [-0.0])
            header = dict(kind='array', begin=0, size=width, shape=[1], dtype=dtype, order='C', readonly=False)
            try:
                active = True
                try:
                    codec('unpack', (header, b'\\0' * width, 1024))
                except ValueError as error:
                    assert 'storage seed changed' in str(error), str(error)
                else:
                    raise AssertionError('negative-zero storage seed was admitted')
            finally:
                active = False
                storage.ZERO_STORES[code] = original
            assert events == [], events
        assert ('numpy' in sys.modules) is numpy_connected
        print(json.dumps(dict(cases=cases, audit_events=events,
                              numpy_loaded='numpy' in sys.modules, positive_control=True,
                              refused_mutations=3)))
    ''')
    result = subprocess.run([sys.executable, '-c', program, str(numpy_connected)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report == dict(cases=72, audit_events=[], numpy_loaded=numpy_connected,
                          positive_control=True, refused_mutations=3)
