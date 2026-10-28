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
        assert 'svc_descriptor_execute(native,' in src
    for name in ('_run', '_pair_decisions'):
        src = body(name)
        assert src.count('_svc_predict_settings(') == 1
        assert '_svc_provider(' not in src
    assert body('_svc_fit').count('_svc_fit_settings(') == 1
    assert '_svc_provider(' not in body('_svc_fit')
    assert body('svc_kernel_operation').count('_svc_provider(') == 1


def test_common_provider_is_selected_without_new_provider_type():
    src = body('_svc_provider')
    assert 'MOJOLEARN_CLASSICAL_NATIVE_DISPATCH' in src
    assert 'neural_descriptor_entry(binding, "svm_binary_api_v1", False)' in src
    assert 'neural_descriptor_connect(factory, String(SVC_ABI_NAME), 12)' in src
    assert 'return PythonObject(None)' in src
    assert 'unavailable for this binding' in src
    assert 'struct ' not in PUBLIC
    assert 'neural_descriptor_info(native.downcast_value_ptr[CompiledBindingSnapshot]()[].provider)' in body('_svc_publish_dispatch')


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


def test_pair_fold_and_calibration_calls_use_same_native_provider():
    for operation in ('svc_pair_epilogue', 'svc_platt_train', 'svc_splitmix_perm'):
        assert f'family_call(_entry(binding, "{operation}"' not in PUBLIC
    for function in ('_c_rows', '_gather', '_gather_i32', '_pair_select', '_epilogue', '_platt'):
        src = body(function)
        assert 'native: PythonObject' in src.split('raises', 1)[0]
        assert 'svc_epilogue_execute(binding, native,' in src
    assert 'neural_descriptor_execute(native, 5, [], [], [])' in body('_svc_provider')
    assert 'py_call(mode)' not in body('_svc_provider')


def test_epilogue_typed_bridge_retains_numerical_owners():
    old = (ROOT / 'providers/python/svm_impl_svc_epilogue_pyapi.mojo').read_text()
    new = (ROOT / 'svm/native_epilogue_dispatch.mojo').read_text()
    numerical = r'\b(?:device_gather|device_select|device_fold_split|device_fold_finish|device_c_rows|device_pair_epilogue|device_platt_train|device_splitmix_perm)\('
    assert re.findall(numerical, old) == re.findall(numerical, new)
    assert 'Python' not in new and 'py=' not in new
    abi = re.search(r'^def svc_binary_v1\(.*?(?=^struct )', NATIVE, re.M | re.S).group()
    assert abi.index('svc_binary_spans(op, na, np, nf)') < abi.index('native_descriptor_words(aa, na)')
    schema=(ROOT/'svm/native_descriptor_schema.mojo').read_text()
    assert 'na != expected_a or np != expected_p or nf != expected_f' in schema
    for message in ('svc gather: an identity axis must keep its length', 'svc fold: bad fold range',
                    'svc_pair_epilogue: needs a pair and two classes', 'svc_splitmix_perm: n exceeds int32'):
        assert message in old and message in new


def test_svr_uses_same_provider_and_retains_native_solver():
    source = (ROOT / 'svm/native_svr_dispatch.mojo').read_text()
    assert 'Python' not in source and 'py=' not in source
    assert 'svr_fit_host(' in source and 'svr_predict_host(' in source
    assert 'if kernel != 4:' in source
    for name in ('_svr_fit', '_svr_predict'):
        assert 'svr_execute(binding, native,' in body(name)
        assert 'family_call(_entry(binding, "svr_' not in body(name)
    assert 'ip.unsafe_store(2, Float64(res.n_iter))' in source
    assert 'if len(params) != 9 and len(params) != 10 and len(params) != 12:' in source
    assert 'if len(params) != 7 and len(params) != 9:' in source


def test_svc_auxiliary_dispatch_retains_native_owners_and_operation_metadata():
    aux=(ROOT/'svm/native_auxiliary_dispatch.mojo').read_text()
    for name in ('scale_gamma_limbs_device','ovo_layout_device','ovo_dual_gemv_device','ovo_check_result'):
        assert name+'(' in aux
    assert 'svc_descriptor_flags(native)' in body('_resolve_gamma')
    assert 'svc_descriptor_execute(native, 9,' in body('_resolve_gamma')
    assert 'svc_descriptor_execute(native, 11,' in body('_linear_coef')
    assert 'svc_ovo_execute(binding, native,' in body('_fit_ovo')
    admit=body('_svc_provider')
    assert admit.count('neural_descriptor_execute(native, 8,')==1
    assert 'CompiledBindingSnapshot(native, binding' in admit
    adapter=(ROOT/'providers/python/svc_descriptor_pyapi.mojo').read_text()
    assert 'neural_descriptor_execute(owner[].provider, op, a, p, f)' in adapter
    assert 'neural_descriptor_connect' not in adapter


def test_svm_shared_inputs_are_snapshotted_outside_pair_and_fold_loops():
    for name in ('_resolve_gamma', '_fit_ovo', '_platt'):
        src=body(name)
        assert 'x_meta: ArrayMetadata' in src
        for repeated in ('x.shape', 'x.size', '_addr(x)', 'classical_address(x)'):
            assert repeated not in src
    for name in ('_svc_fit', '_svr_fit'):
        src=body(name)
        assert src.count('svc_snapshot(args[1],') == 1
        assert '_ = x_meta' in src
    for name in ('_run', '_pair_decisions', '_svr_predict'):
        assert 'q_meta.source.addr' in body(name)
        assert '_ = q_meta' in body(name)
    assert '_addr(q)' not in body('_machine')
    metadata=(ROOT/'providers/python/array_metadata_pyapi.mojo').read_text()
    for slot in ('_store', '_base', '_pin', '_mv'):
        assert f'array_slot_once(value, "{slot}", context)' in metadata
    owned=(ROOT/'bindings/family_common.mojo').read_text().split('def family_owned_snapshot(',1)[1].split('\ndef ',1)[0]
    assert 'value.owned_writable' in owned
    assert 'value.keep.' not in owned


def test_shared_weight_snapshot_keeps_cast_order_and_reuses_admitted_address():
    source=(ROOT/'bindings/family_common.mojo').read_text()
    take=lambda name: re.search(rf'^def {name}\(.*?(?=^def |\Z)', source,re.M|re.S).group()
    admission=take('_family_weight_buffer')
    assert admission.count('array_shape_once(view, context)') == 1
    for old in ('view.shape', 'view.ndim', 'view.size', 'b.hasattr', 'b.float'):
        assert old not in admission
    assert 'family_cast_rows_f64(raw.addr' in admission
    assert 'family_cast_rows(raw.addr' in admission
    validate=take('family_sample_weight_snapshot')
    assert 'snapshot.source.addr' in validate
    assert validate.index('FAMILY_WEIGHTS_NONFINITE') < validate.index('FAMILY_WEIGHTS_NEGATIVE') < validate.index('FAMILY_WEIGHTS_ALL_ZERO')
    weights=body('_c_rows')
    assert 'family_sample_weight_snapshot(' in weights
    assert 'sw_owner = Python.tuple(weights.keep, weights.owner)' in weights
    assert 'classical_address(sw)' not in weights


def test_cpu_svm_table_uses_same_schema_and_existing_host_owners():
    host=(ROOT/'svm/host/native_descriptor_dispatch.mojo').read_text()
    assert 'String(SVC_ABI_NAME)' in host
    assert 'svc_binary_spans(op, na, np, nf)' in host
    assert host.index('svc_binary_spans(op, na, np, nf)') < host.index('native_descriptor_words(aa, na)')
    for name in ('svc_fit_host','svc_predict_host','svr_fit_host','svr_predict_host'):
        assert name+'(' in host
    assert 'svm.estimator' not in host
    assert 'neural_descriptor_capsule(table[].name, table[].entries)' in (ROOT/'bindings/_mojolearn_svm_host.mojo').read_text()
    native=(ROOT/'svm/host/native_epilogue_dispatch.mojo').read_text()
    old=(ROOT/'providers/python/svm_host_svc_proba_pyapi.mojo').read_text()
    calls=r'\bhost_(?:gather|select|fold_split|fold_finish|c_rows|check_pairs|pair_epilogue|platt_train|splitmix_perm)\('
    assert re.findall(calls,native)==re.findall(calls,old)
    assert 'dec_addr if nsr * nsc > 0 else dst' in native
    assert 'pairs_addr if mode != EPI_BINARY_CODES else dst' in native
    _,errors=audit(ROOT,('svm.host.native_descriptor_dispatch',))
    assert not errors,errors
