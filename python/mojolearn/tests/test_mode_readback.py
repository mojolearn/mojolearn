"""Actual connected native readbacks take precedence over module path hints."""
import pytest
from mojolearn import KMeans, _mode


@pytest.fixture
def connected():
    model = KMeans(n_clusters=2, numeric_mode='identical')
    connection = _mode._MODE_METADATA_CONTEXT['plans'][type(model)]['modes']['identical']
    assert connection['error'] is None
    assert connection['mode'] is not None
    return model, connection


def test_compiled_readback_wins_over_misleading_module_path(connected, monkeypatch):
    model, connection = connected
    # This is an identical readback with misleading legacy metadata, not a
    # fast operation or a fast validation campaign.
    monkeypatch.setitem(connection, 'legacy_mode', 'fast')
    assert model.numeric_mode_used() == 'identical'


def test_unknown_compiled_mode_is_not_inferred_from_path(connected, monkeypatch):
    model, connection = connected
    monkeypatch.setitem(_mode._MODE_METADATA_CONTEXT, 'mode_codes', {})
    with pytest.raises(RuntimeError, match='unknown numeric mode'):
        model.numeric_mode_used()


def test_compiled_mode_must_match_selected_mode(connected, monkeypatch):
    model, connection = connected
    code = connection['mode']()
    monkeypatch.setitem(_mode._MODE_METADATA_CONTEXT, 'mode_codes', {code: 'fast'})
    with pytest.raises(RuntimeError, match='does not match'):
        model.numeric_mode_used()


def test_legacy_artifact_keeps_path_hint(connected, monkeypatch):
    model, connection = connected
    monkeypatch.setitem(connection, 'mode', None)
    monkeypatch.setitem(connection, 'legacy_mode', 'identical')
    assert model.numeric_mode_used() == 'identical'


def test_python_binding_override_is_refused_before_invocation(connected):
    model, connection = connected
    calls = []
    def forbidden():
        calls.append('called')
        return None
    model.__dict__['_bind'] = forbidden
    with pytest.raises(TypeError, match='binding overrides'):
        model.numeric_mode_used()
    assert not calls
