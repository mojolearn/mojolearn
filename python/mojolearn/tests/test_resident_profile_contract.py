# SPDX-License-Identifier: Apache-2.0
"""Bounded installed GPU admission regressions; not identity/performance evidence."""
import os
import numpy as np
import pytest
from mojolearn.tests.test_generation_native import load_fixture

pytestmark = pytest.mark.skipif(
    os.environ.get("MOJOLEARN_GENERATION_DEVICE") != "gpu",
    reason="explicit installed GPU profile contract run required",
)


@pytest.fixture(autouse=True)
def resident_route(monkeypatch):
    monkeypatch.setenv("MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD", "1")
    monkeypatch.delenv("MOJOLEARN_TRANSFORMER_LEGACY_SETUP", raising=False)


def test_default_profile_publishes_with_resident_state(tmp_path):
    model = load_fixture(tmp_path, "llama", uniform=True)
    assert model._native_entries["resident_read_v1"] is True
    expected = [block._arithmetic_profile for block in model._blocks]
    assert all(isinstance(value, str) for value in expected)
    state = model.allocate_state(1, 5)
    model.forward(np.array([[1, 3]], dtype="i4"), state)
    assert model._resident_sessions is not None
    assert [layer.arithmetic_profile for layer in state.layers] == expected
    model.forward(np.array([[4]], dtype="i4"), state)
    assert [layer.cached_tokens for layer in state.layers] == [3] * len(expected)
    assert [layer.arithmetic_profile for layer in state.layers] == expected


def test_resident_profile_mismatch_keeps_legacy_diagnostic(tmp_path):
    model = load_fixture(tmp_path, "llama", uniform=True)
    state = model.allocate_state(1, 5)
    state.layers[0].arithmetic_profile = "incompatible-profile"
    with pytest.raises(Exception, match="state arithmetic profile differs"):
        model.forward(np.array([[1]], dtype="i4"), state)
    assert state.layers[0].cached_tokens == 0
    assert model._resident_sessions is None


def test_changed_profile_unlabelled_cache_keeps_legacy_diagnostic(tmp_path):
    model = load_fixture(tmp_path, "llama", uniform=True)
    state = model.allocate_state(1, 5)
    state.positions = 1
    state.layers[0].cached_tokens = 1
    model._blocks[0]._arithmetic_profile_changed = True
    with pytest.raises(Exception, match="cached state requires arithmetic profile metadata"):
        model.forward(np.array([[1]], dtype="i4"), state)
    assert model._resident_sessions is None


@pytest.mark.parametrize("family,entry,value", [
    ("llama", "resident_read_v1", 1),
    ("llama", "resident_read_v1", lambda: True),
    ("mamba2", "mamba2_workspace_provider", object()),
])
def test_resident_registry_replacement_is_refused(tmp_path, family, entry, value):
    model = load_fixture(tmp_path, family, uniform=(family == "llama"))
    assert entry in model._native_entries
    model._native_entries[entry] = value
    with pytest.raises(Exception, match="native execution entry changed after admission"):
        model.allocate_state(1, 5)
