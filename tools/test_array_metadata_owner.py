"""Ownership regression for ArrayMetadata's real native replacement methods.

Set MOJOLEARN_ARRAY_METADATA_OWNER_PROBE to the compiled test-only shared library
from tests/array_metadata_owner_probe_pyapi.mojo to run the native lifetime controls.
The default source guards do not claim those native controls ran.
"""
from array import array
import gc
import importlib.util
import os
from pathlib import Path
import re
import weakref

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _method(source, name):
    return source.split('    def '+name+'(', 1)[1].split('\n    def ', 1)[0].split('\n\ndef ', 1)[0]


def test_replacement_keeps_original_owner_and_current_storage_separate():
    source = (ROOT/'providers/python/array_metadata_pyapi.mojo').read_text()
    assert len(re.findall(r'^\s+self\.owner\s*=', source, re.M)) == 1
    assert 'self.owner = owner' in _method(source, '__init__')
    for name in ('replace_f32', 'replace_f64'):
        body = _method(source, name)
        assert 'self.keep = keep' in body
        assert 'self.owner =' not in body
        assert 'array_owner_once(' not in body
        assert 'array_slot_once(' not in body
        assert 'self.copied = True' in body
        assert 'self.owned_writable = True' in body


def test_original_mutable_array_slot_owners_are_snapshotted_independently():
    source = (ROOT/'providers/python/array_metadata_pyapi.mojo').read_text()
    owner = source.split('def array_owner_once(', 1)[1].split('\ndef ', 1)[0]
    for slot in ('_store', '_base', '_pin', '_mv'):
        assert f'array_slot_once(value, "{slot}", context)' in owner
    assert 'var owner = array_owner_once(value, context)' in source


@pytest.fixture(scope='module')
def probe():
    path = os.environ.get('MOJOLEARN_ARRAY_METADATA_OWNER_PROBE')
    if not path:
        pytest.skip('native owner probe has not been compiled for this test run')
    spec = importlib.util.spec_from_file_location('array_metadata_owner_probe', path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('route', [0, 32, 64, 3264, 6432])
def test_native_replacement_retains_original_owner_and_finished_metadata(probe, route):
    source = array('f', range(6))
    exporter = array('f', range(6))
    original_owner = (exporter, None, None, memoryview(exporter))
    converted_f32 = array('f', range(6))
    converted_f64 = array('d', range(6))
    actual = probe.replace(source, original_owner, converted_f32, converted_f64, route)
    assert actual[0] is original_owner
    expected_keep = source if route == 0 else converted_f32 if route in (32, 6432) else converted_f64
    assert actual[1] is expected_keep
    assert actual[2:4] == (route != 0, route != 0)
    expected_words = (256, 2, 6, 2) if route == 0 else (512, 4, 12, 4) if route in (32, 6432) else (768, 8, 24, 8)
    assert actual[4:8] == expected_words
    assert actual[8:] == (6, (2, 3))


def test_native_result_retains_exporter_after_original_slots_change(probe):
    original = array('f', range(6))
    converted = array('d', range(6))
    original_ref, converted_ref = weakref.ref(original), weakref.ref(converted)
    source_slots = {'_store': original, '_base': None, '_pin': None, '_mv': memoryview(original)}
    owner_snapshot = tuple(source_slots.values())
    actual = probe.replace(source_slots, owner_snapshot, array('f'), converted, 64)
    source_slots.clear()
    del owner_snapshot, original, converted
    gc.collect()
    assert original_ref() is not None
    assert converted_ref() is not None
    del actual
    gc.collect()
    assert original_ref() is None
    assert converted_ref() is None
