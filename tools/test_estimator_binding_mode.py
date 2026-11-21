"""Source checks for native explicit-mode binding admission; runtime is separate."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'providers/python/estimator_binding_pyapi_door.mojo'


def _body(name):
    return SOURCE.read_text().split('def ' + name + '(', 1)[1].split('\n\ndef ', 1)[0]


def test_model_route_reads_admitted_storage_then_delegates():
    # Lane dw-core (2026-11-18): the door admits the instance storage once, reads the mode from it and hands the
    # admitted fields to the estimator's route (the device record is read from the same fields on a miss); a None
    # model still goes straight to the mode-only selector.
    source = _body('estimator_binding')
    assert 'var fields = _fields(model)' in source
    assert source.count('_fields(model)') == 1
    assert ('return _estimator_binding_routed(model, fields, py_get(fields, "numeric_mode"), name, context, '
            'check_mode)') in source
    assert 'return estimator_binding_mode(PythonObject(None), name, context, check_mode)' in source
    assert '.strip(' not in source
    assert 'py_fields(model)' not in source


def _all(name):
    text = SOURCE.read_text()
    return [part.split('\n\ndef ', 1)[0] for part in text.split('def ' + name + '(')[1:]]


# The mode check and the registry lookup are split into _selected_mode and _estimator_binding_selected, and the
# AdmittedCall variant of estimator_binding_mode resolves a binding once per call (find_binding / remember_binding).
def test_explicit_mode_is_not_reread_from_model_or_fitted_metadata():
    source = _body('_selected_mode')
    assert 'var mode = requested' in source
    assert 'model' not in source.split('    var mode = requested', 1)[1]
    assert 'numeric_mode_' not in source.split('    var mode = requested', 1)[1]
    assert 'mode = context["backend_fields"]["_DEFAULT_MODE"]' in source
    variants = _all('estimator_binding_mode')
    assert len(variants) == 2
    for variant in variants:
        assert 'var selected_mode = _selected_mode(bi, requested, context)' in variant
        assert 'model' not in variant.split('raises -> PythonObject:', 1)[1]


def test_scalar_validation_precedes_normalization_and_registry_selection():
    source = _body('_selected_mode')
    admission = source.index('if py_type(mode) is not bi.str:')
    assert admission < source.index('mode.strip().lower()')
    assert source.count('if selected_mode != "identical" and selected_mode != "fast":') == 2
    for variant in _all('estimator_binding_mode'):
        assert variant.index('_selected_mode(') < variant.index('_estimator_binding_selected(')
    cached = _all('estimator_binding_mode')[0]
    assert cached.index('_selected_mode(') < cached.index('call.find_binding(name, selected_mode, check_mode)')
    assert 'call.remember_binding(name, selected_mode, check_mode, module)' in cached


def test_explicit_mode_preserves_checked_live_route_selection():
    source = _body('_estimator_binding_selected')
    assert source.index('context["family_binding_offerings"][name][selected_mode]') < source.index('backend["_CPU_ONLY"]')
    assert 'py_get(sys_module.modules, record[1])' in source
    assert 'py_get(sets, selected_mode)' in source
    assert source.count('estimator_checked_mode(') == 2
    assert source.count('if check_mode else module') == 2
