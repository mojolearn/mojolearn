# SPDX-License-Identifier: Apache-2.0
"""Public Mamba hook refusals and native state/session boundary controls.

These tests require the compiled boundary under test. Source syntax checks alone
are not runtime evidence; the peer lane leaves execution to final integration.
"""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_mamba_native_public import _block, _tokens


class _ScalarHook:
    def __init__(self, calls):
        self.calls = calls

    def __int__(self):
        self.calls.append("int")
        return 0

    def __bool__(self):
        self.calls.append("bool")
        return False


@pytest.mark.parametrize("kind", [1, 2, 3])
def test_state_constructors_store_references_in_one_native_operation(kind):
    value = np.zeros((1,), np.float32)
    cls = getattr(ml, f"Mamba{kind}State")
    count = {1: 2, 2: 4, 3: 10}[kind]
    owner = object.__new__(cls)
    audit_native_call(cls.__init__, owner, *([value] * count)).assert_native()
    assert owner.h is value
    if kind == 1:
        assert owner.conv_window is value and owner.profile_id == 1
    else:
        assert owner.buffered_tokens == 0
    if kind == 3:
        assert owner.pending is False


@pytest.mark.parametrize("kind,keyword", [(1, "profile_id"), (1, "absolute_position"),
                                          (2, "buffered_tokens"), (3, "buffered_tokens"), (3, "pending")])
def test_state_constructor_refuses_scalar_hooks_before_conversion(kind, keyword):
    calls = []
    value = np.zeros((1,), np.float32)
    cls = getattr(ml, f"Mamba{kind}State")
    count = {1: 2, 2: 4, 3: 10}[kind]
    with pytest.raises(TypeError, match="native scalar|conversion"):
        cls(*([value] * count), **{keyword: _ScalarHook(calls)})
    assert calls == []


@pytest.mark.parametrize("field", ["buffered_tokens", "pending"])
def test_mamba3_carried_state_refuses_mutated_scalar_hooks(field):
    block, dm = _block("mamba3")
    state = block.allocate_state(1)
    calls = []
    setattr(state, field, _ScalarHook(calls))
    with pytest.raises(TypeError, match="native scalar|conversion"):
        block.step(_tokens(1, 1, dm), state)
    assert calls == []


@pytest.mark.parametrize("field", ["__dict__", "h"])
def test_mamba3_carried_state_refuses_descriptors_before_reading_them(monkeypatch, field):
    block, dm = _block("mamba3")
    state = block.allocate_state(1)
    calls = []

    def forbidden(_):
        calls.append(field)
        raise AssertionError("descriptor was invoked before admission")

    monkeypatch.setattr(type(state), field, property(forbidden), raising=False)
    with pytest.raises(TypeError, match="dictionary|descriptor"):
        block.step(_tokens(1, 1, dm), state)
    assert calls == []


def test_mamba3_restore_refuses_mapping_hooks_before_update():
    block, _ = _block("mamba3")
    calls = []

    class MappingHook(dict):
        def keys(self):
            calls.append("keys")
            return super().keys()

    with pytest.raises(TypeError, match="ordinary dictionary"):
        block.__setstate__(MappingHook(block.__getstate__()))
    assert calls == []


def test_decode_session_refuses_replaced_handle_descriptor(monkeypatch):
    block, _ = _block("mamba1")
    state = block.allocate_state(1)
    session = block.decode_session(state)
    calls = []

    def forbidden(_):
        calls.append("handle")
        raise AssertionError("handle descriptor was invoked")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(type(session), "_handle", property(forbidden), raising=False)
            with pytest.raises(TypeError, match="descriptor"):
                _ = session.is_open
        assert calls == []
    finally:
        session.close()
