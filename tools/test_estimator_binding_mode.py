"""Source checks for native explicit-mode binding admission; runtime is separate."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'providers/python/estimator_binding_pyapi.mojo'


def _body(name):
    return SOURCE.read_text().split('def ' + name + '(', 1)[1].split('\n\ndef ', 1)[0]


def test_model_route_reads_admitted_storage_then_delegates():
    source = _body('estimator_binding')
    assert 'py_get(_fields(model), "numeric_mode")' in source
    assert 'return estimator_binding_mode(requested, name, context, check_mode)' in source
    assert '.strip(' not in source
    assert 'py_fields(model)' not in source


def test_explicit_mode_is_not_reread_from_model_or_fitted_metadata():
    source = _body('estimator_binding_mode')
    assert 'var mode = requested' in source
    assert 'model' not in source.split('    var bi =', 1)[1]
    assert 'numeric_mode_' not in source.split('    var bi =', 1)[1]
    assert 'mode = backend["_DEFAULT_MODE"]' in source


def test_scalar_validation_precedes_normalization_and_registry_selection():
    source = _body('estimator_binding_mode')
    admission = source.index('if py_type(mode) is not bi.str:')
    assert admission < source.index('mode.strip().lower()')
    assert admission < source.index('context["family_binding_connections"]')
    assert 'if selected_mode != "identical" and selected_mode != "fast":' in source


def test_explicit_mode_preserves_checked_live_route_selection():
    source = _body('estimator_binding_mode')
    assert 'context["family_binding_offerings"][name][selected_mode]' in source
    assert 'py_get(native_loaded_module("sys").modules, record[1])' in source
    assert 'py_get(sets, selected_mode)' in source
    assert source.count('estimator_checked_mode(') == 2
    assert source.count('if check_mode else module') == 2
