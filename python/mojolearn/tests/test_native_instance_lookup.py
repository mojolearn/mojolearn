# SPDX-License-Identifier: Apache-2.0
"""Public regression cases for native instance reads and result ownership."""
import gc
import sys
import weakref

import pytest

from mojolearn import KMeans
from mojolearn._verify_native_runtime import audit_native_call


class Payload:
    pass


def clean(operation, *args):
    return audit_native_call(operation, *args).assert_native()


@pytest.mark.parametrize('value', [None, False, True, 0, 'field'])
def test_existing_parameter_and_attribute_preserve_identity(value):
    model = KMeans(numeric_mode='identical')
    fields = object.__getattribute__(model, '__dict__')
    fields.update(init_centroids=value, user_note=value)
    assert clean(model.get_params, False)['init_centroids'] is value
    assert clean(getattr, model, 'user_note') is value


@pytest.mark.parametrize('route', ['params', 'attribute'])
def test_result_keeps_payload_alive_after_parent_deletion_then_releases(route):
    model = KMeans(numeric_mode='identical')
    payload = Payload()
    witness = weakref.ref(payload)
    object.__getattribute__(model, '__dict__')['init_centroids'] = payload
    if route == 'params':
        result = clean(model.get_params, False)
        assert result['init_centroids'] is payload
    else:
        result = clean(getattr, model, 'init_centroids')
        assert result is payload
    del payload, model
    gc.collect()
    assert witness() is not None
    del result
    gc.collect()
    assert witness() is None


@pytest.mark.parametrize('route', ['params', 'attribute'])
def test_repeated_reads_do_not_accumulate_payload_references(route):
    model = KMeans(numeric_mode='identical')
    payload = Payload()
    object.__getattribute__(model, '__dict__')['init_centroids'] = payload
    getter = model.get_params
    before = sys.getrefcount(payload)
    for _ in range(200):  # testing repetition, outside every product operation
        result = getter(False)['init_centroids'] if route == 'params' else model.init_centroids
        assert result is payload
        del result
    gc.collect()
    assert sys.getrefcount(payload) == before


def test_instance_string_subclass_key_is_refused_before_equality_hook():
    calls = []
    class Key(str):
        def __hash__(self):
            calls.append('hash')
            return super().__hash__()
        def __eq__(self, other):
            calls.append('equal')
            return super().__eq__(other)
    model = KMeans(numeric_mode='identical')
    getter = model.get_params
    fields = object.__getattribute__(model, '__dict__')
    del fields['init_centroids']
    fields[Key('init_centroids')] = None
    calls.clear()
    for operation, args in ((getter, (False,)), (getattr, (model, 'init_centroids'))):
        audit = audit_native_call(operation, *args)
        assert isinstance(audit.exception, TypeError), audit.report()
        report = audit.report()
        assert not report['python_frames'], report
        assert not calls
