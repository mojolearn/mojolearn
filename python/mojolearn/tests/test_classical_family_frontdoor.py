# SPDX-License-Identifier: Apache-2.0
"""Public family storage controls; runtime execution is pending integration.

These use public state/metadata operations and independent callback counters.
They do not add algorithm identity lanes or claim other family work is native.
"""
import pytest


@pytest.mark.parametrize('hook', ['dict_property', 'dict_value', 'getter', 'getattr', 'setter', 'parameter_property'])
def test_public_state_operation_refuses_changed_storage_before_hooks(hook):
    from mojolearn import KMeans

    model = KMeans(n_clusters=2)
    operation = model.__getstate__
    original = type(model)
    calls = []

    def read(self, name):
        calls.append(name)
        return object.__getattribute__(self, name)

    def missing(self, name):
        calls.append(name)
        raise AttributeError(name)

    def write(self, name, value):
        calls.append(name)
        object.__setattr__(self, name, value)

    def property_read(self):
        calls.append('property')
        return {}

    class MappingHook:
        def __getitem__(self, name):
            calls.append('getitem')
            return None

    members = {'__slots__': ()}
    if hook == 'dict_property':
        members['__dict__'] = property(property_read)
    elif hook == 'dict_value':
        members['__dict__'] = MappingHook()
    elif hook == 'getter':
        members['__getattribute__'] = read
    elif hook == 'getattr':
        members['__getattr__'] = missing
    elif hook == 'setter':
        members['__setattr__'] = write
    else:
        members['n_clusters'] = property(property_read)
    changed = type('Changed', (original,), members)
    object.__setattr__(model, '__class__', changed)
    calls.clear()  # Class-definition metadata is test setup, outside execution.
    try:
        with pytest.raises(TypeError):
            operation()
        assert calls == []
    finally:
        object.__setattr__(model, '__class__', original)


def test_public_state_operation_refuses_replaced_connected_descriptor(monkeypatch):
    from mojolearn import KMeans

    model = KMeans(n_clusters=2)
    operation = model.__getstate__
    calls = []
    with monkeypatch.context() as patch:
        patch.setattr(KMeans, '__getstate__', property(lambda self: calls.append('property')))
        with pytest.raises(TypeError):
            operation()
    assert calls == []


def test_public_state_operation_refuses_nonstring_state_keys_before_comparison():
    from mojolearn import KMeans

    model = KMeans(n_clusters=2)
    operation = model.__getstate__
    fields = vars(model)
    calls = []

    class Key:
        def __hash__(self):
            calls.append('hash')
            return hash('_kmeans_resident')

        def __eq__(self, other):
            calls.append('eq')
            return False

    key = Key()
    fields[key] = 1
    calls.clear()
    try:
        with pytest.raises(TypeError):
            operation()
        assert calls == []
    finally:
        fields.pop(key)


def test_passive_subclass_keeps_state_and_unrelated_application_methods():
    from mojolearn import KMeans

    class Passive(KMeans):
        label = 'plain'

        def own_method(self):
            raise AssertionError('the product must not invoke this method')

    model = Passive(n_clusters=2)
    state = model.__getstate__()
    assert state['n_clusters'] == 2
    assert model.label == 'plain'


def test_functional_none_owner_remains_supported():
    from mojolearn.lowbit import is_packed

    assert is_packed(object()) is False
