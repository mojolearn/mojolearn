"""Public Sequential operations and legacy C-entry forwarding share one owner."""
from types import BuiltinFunctionType

import pytest

from mojolearn import Sequential, _backend, _x_sequence_model
from mojolearn._verify_native_runtime import audit_native_call


def legacy_entry():
    return _backend.load_host_module('_mojolearn_tokenizer_host').sequential_native_call


def test_sequential_public_entry_is_retained_from_the_new_owner():
    owner = _backend.load_host_module('_mojolearn_runtime_sequential_host')
    assert isinstance(owner.sequential_native_call, BuiltinFunctionType)
    assert _x_sequence_model._SEQUENTIAL_ENTRY is owner.sequential_native_call
    assert _x_sequence_model._SEQUENTIAL_CONTEXT['sequential_owner_entry'] is owner.sequential_native_call
    assert legacy_entry() is not owner.sequential_native_call


def test_public_and_legacy_descriptions_match_with_native_public_execution():
    public = Sequential.__new__(Sequential)
    audit_native_call(public.__init__, input_shape=(4,)).assert_native()
    audit_native_call(public.linear, 3, bias=False).assert_native()
    audit_native_call(public.relu).assert_native()
    other = Sequential.__new__(Sequential)
    entry, context = legacy_entry(), _x_sequence_model._SEQUENTIAL_CONTEXT
    entry(other, 'construct', ('identical', (4,)), context)
    entry(other, 'add', ('linear', (3,), {'bias': False}), context)
    entry(other, 'add', ('relu', (), {}), context)
    assert public.layers == other.layers
    assert public._program(4) == other._program(4)


def test_legacy_owner_exception_is_not_remapped_and_next_call_still_works():
    model = Sequential()
    context = _x_sequence_model._SEQUENTIAL_CONTEXT
    direct, legacy = _x_sequence_model._SEQUENTIAL_ENTRY, legacy_entry()
    with pytest.raises(ValueError) as first:
        direct(model, 'add', ('unknown-layer', (), {}), context)
    with pytest.raises(ValueError) as second:
        legacy(model, 'add', ('unknown-layer', (), {}), context)
    assert type(first.value) is type(second.value)
    assert str(first.value) == str(second.value)
    legacy(model, 'add', ('linear', (2,), {}), context)
    assert model.layers == [('linear', 2, 1, 0.0)]


def test_forwarder_refuses_python_entry_and_mapping_protocol_before_callbacks():
    calls = []

    def callback(*args):
        calls.append(args)

    class Mapping(dict):
        def __getitem__(self, key):
            calls.append(key)
            return super().__getitem__(key)

    context = _x_sequence_model._SEQUENTIAL_CONTEXT
    for invalid in (dict(context, sequential_owner_entry=callback), Mapping(context)):
        with pytest.raises(TypeError, match='native'):
            legacy_entry()(None, 'initialize', (), invalid)
    assert calls == []
