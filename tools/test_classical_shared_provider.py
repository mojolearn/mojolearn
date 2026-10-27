"""SVC native settings/dispatch source contracts; no compiled-runtime claim."""
from pathlib import Path
import re
from check_generation_native_boundary import audit

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = (ROOT / 'bindings/svc_kernel_native.mojo').read_text()
NATIVE = (ROOT / 'svm/native_descriptor_dispatch.mojo').read_text()
SETTINGS = (ROOT / 'svm/operation_metadata.mojo').read_text()


def body(name):
    return re.search(rf'^def {name}\(.*?(?=^def |\Z)', PUBLIC, re.M | re.S).group()


def test_native_settings_and_numerical_dispatch_have_no_provider_backedge():
    _, errors = audit(ROOT, ('svm.operation_metadata', 'svm.native_descriptor_dispatch'))
    assert not errors, errors
    for forbidden in ('PythonObject', 'py_call', 'family_call', 'providers.python'):
        assert forbidden not in NATIVE and forbidden not in SETTINGS


def test_repeated_binary_calls_do_not_reread_model_settings_or_query_shape():
    for name in ('_solve', '_machine'):
        src = body(name)
        for forbidden in ('svc_field(', '_kernel_code(', 'q.shape', '_svc_provider(', '_svc_fit_settings(', '_svc_predict_settings('):
            assert forbidden not in src
        assert 'neural_descriptor_execute(native,' in src
    for name in ('_run', '_pair_decisions'):
        src = body(name)
        assert src.count('_svc_predict_settings(') == 1
        assert src.count('_svc_provider(') == 1
    assert body('_svc_fit').count('_svc_fit_settings(') == 1
    assert body('_svc_fit').count('_svc_provider(') == 1


def test_common_provider_is_selected_without_new_provider_type():
    src = body('_svc_provider')
    assert 'MOJOLEARN_CLASSICAL_NATIVE_DISPATCH' in src
    assert 'neural_descriptor_entry(binding, "svm_binary_api_v1", False)' in src
    assert 'neural_descriptor_connect(factory, String(SVC_ABI_NAME), 2)' in src
    assert 'return PythonObject(None)' in src
    assert 'unavailable for this binding' in src
    assert 'struct ' not in PUBLIC
    assert 'neural_descriptor_info(native)' in body('_svc_publish_dispatch')


def test_original_numerical_owners_and_output_layout_are_preserved():
    assert 'svc_fit_host_borrowed(xp, y, n, d,' in NATIVE
    assert 'svc_predict_host(x, n, d, support, dual, count,' in NATIVE
    assert 'if p[3] != 4:' in NATIVE  # precomputed does not read support rows
    for slot, field in enumerate(('b', 'n_support', 'n_iter', 'label0', 'label1')):
        assert f'info[{slot}] = Float64(result.{field})' in NATIVE
    assert 'bounds = read_f32(a[6], n)' in NATIVE
    assert 'native_descriptor_report(problem, context, sink)' in NATIVE


def test_all_solve_and_machine_call_sites_receive_operation_snapshot():
    # Calls span physical lines, so inspect to the next matched closing call
    # delimiter rather than line numbers. There are3solver callers and3predict.
    solves = re.findall(r'(?<!def )_solve\(fields, binding,.*?context, settings, native\)', PUBLIC, re.S)
    machines = re.findall(r'(?<!def )_machine\(fields, binding,.*?context, settings, native,', PUBLIC, re.S)
    assert len(solves) == 3
    assert len(machines) == 3
    assert 'settings, native)' in body('_fit_ovo')
    assert 'settings, native,' in body('_platt')
