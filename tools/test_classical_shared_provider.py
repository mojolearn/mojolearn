"""SVC native settings/dispatch source contracts; no compiled-runtime claim."""
from pathlib import Path
import re
from check_generation_native_boundary import audit

ROOT = Path(__file__).resolve().parents[1]
# The kernel SVM door and its UNPACK / PACK files (lane dw-svc-kernel split the door).
PUBLIC = '\n'.join((ROOT / 'bindings' / name).read_text() for name in (
    'svc_kernel_native_door.mojo', 'svc_kernel_params_unpack.mojo', 'svc_kernel_rows_unpack.mojo',
    'svc_kernel_svr_unpack.mojo', 'svc_kernel_svr_pack.mojo', 'svc_kernel_provider_unpack.mojo',
    'svc_kernel_query_unpack.mojo', 'svc_kernel_query_pack.mojo', 'svc_kernel_fit_unpack.mojo',
    'svc_kernel_fit_pack.mojo'))
RUN = (ROOT / 'svm/svc_kernel_native_run.mojo').read_text()
FIT = (ROOT / 'svm/svc_kernel_fit_run.mojo').read_text()
TRANSPORT = (ROOT / 'core/svc_native_api.mojo').read_text()
DEVICE = (ROOT / 'svm/svc_native_device.mojo').read_text()
NATIVE = (ROOT / 'svm/native_descriptor_dispatch.mojo').read_text()
SETTINGS = (ROOT / 'svm/operation_metadata.mojo').read_text()


def body(name, text=None):
    return re.search(rf'^def {name}\(.*?(?=^def |^struct |\Z)', PUBLIC if text is None else text, re.M | re.S).group()


def test_native_settings_and_numerical_dispatch_have_no_provider_backedge():
    _, errors = audit(ROOT, ('svm.operation_metadata', 'svm.native_descriptor_dispatch'))
    assert not errors, errors
    for forbidden in ('PythonObject', 'py_call', 'family_call', 'providers.python'):
        assert forbidden not in NATIVE and forbidden not in SETTINGS


def test_repeated_binary_calls_do_not_reread_model_settings_or_query_shape():
    # RUN is typed and holds no Python object: a solve or a machine cannot re-read a model field, a
    # provider or the query's shape. The settings are read once per operation at UNPACK.
    for text in (RUN, FIT, TRANSPORT, DEVICE):
        for forbidden in ('PythonObject', 'std.python', 'svc_field(', 'py_call', 'family_call'):
            assert forbidden not in text
    assert body('svc_query_unpack').count('svc_predict_settings(') == 1
    assert body('svc_fit_unpack').count('hp.fit_operation(') == 1 and body('svc_fit_unpack').count('svm_settings_unpack(') == 1
    assert body('svr_fit_unpack').count('svm_settings_unpack(') == 1
    # One provider admission per public operation on each of the two exclusive paths: the
    # algorithm-only early return (coef_), and the public/fit lifecycle after it.
    operation = body('svc_kernel_operation')
    early, lifecycle = operation.split('        return plain\n', 1)
    assert early.count('svc_connection(') == 1
    assert lifecycle.count('svc_connection(') == 1
    assert '_svc_provider' not in PUBLIC


def test_common_provider_is_selected_without_new_provider_type():
    src = body('svc_connection')
    # One route, no environment read on the call path (the B arm -D MOJOLEARN_CLASSICAL_NATIVE_DISPATCH was
    # removed 2026-11-23 as plumbing).
    assert 'CLASSICAL_NATIVE_DISPATCH' not in src
    assert 'getenv' not in PUBLIC
    assert 'neural_descriptor_entry(binding, "svm_binary_api_v1", False)' in src
    assert 'neural_descriptor_connect_factory_once(factory, String(SVC_ABI_NAME), SVC_TABLE_COUNT)' in src
    assert 'SVC_TABLE_COUNT = 12' in TRANSPORT
    assert 'has no svm_binary_api_v1 table; rebuild it' in src
    # The device table is connected when the binding registers one.
    assert 'neural_descriptor_entry(binding, "svc_device_api_v1", False)' in src
    # No provider type of its own: the only structs are the door's records (the query record, the settings
    # and weights records, the connection's owners, the UNPACK records `*Unpacked`).
    structs = re.findall(r'^struct (\w+)', PUBLIC, re.M)
    assert '_SvmQuery' in structs and all(name in ('_SvmQuery', 'SvmSettings', 'SvcConnection') or name.endswith('Unpacked') for name in structs), structs
    assert 'svc_connection_publish' not in PUBLIC


def test_original_numerical_owners_and_output_layout_are_preserved():
    assert 'svc_fit_host_borrowed(Int(xp), QueryFold.none(), y, n, d,' in NATIVE
    assert 'svc_predict_host(x, n, d, support, dual, count,' in NATIVE
    assert 'if p[3] != 4:' in NATIVE  # precomputed does not read support rows
    for slot, field in enumerate(('b', 'n_support', 'n_iter', 'label0', 'label1')):
        assert f'info[{slot}] = Float64(result.{field})' in NATIVE
    assert 'bounds = read_f32(a[6], n)' in NATIVE
    assert 'native_descriptor_report(problem, context, sink)' in NATIVE


def test_all_solve_and_machine_call_sites_receive_operation_snapshot():
    # Every solve and every machine query takes the operation's settings record (SvcOperationSettings):
    # the binary solve, the pair solves and the fold solves; the inference machines and Platt's held rows.
    assert FIT.count('_solve(drivers, settings,') == 2  # one-vs-one, Platt
    assert FIT.count('drivers.fit(SvcFitRequest(') == 2  # _solve (short form) and the binary machine
    assert FIT.count('drivers.query(SvcQueryRequest(') == 1 and RUN.count('drivers.query(SvcQueryRequest(') == 1
    for name in ('_fit_ovo', '_platt'):
        assert 'settings: SvcOperationSettings' in body(name, FIT).split('raises', 1)[0]


def test_pair_fold_and_calibration_calls_use_same_native_provider():
    # No by-name entry is called through CPython by the kernel SVM door: every pair, fold, calibration and
    # epilogue launch goes through the one typed driver record (core/svc_native_api.mojo SvcDrivers).
    for operation in ('svc_pair_epilogue', 'svc_platt_train', 'svc_splitmix_perm', 'svc_fit', 'svc_predict',
                      'svc_predict_resident', 'svc_ovo_layout', 'svc_dual_gemv', 'scale_gamma_limbs'):
        assert f'"{operation}"' not in PUBLIC
    assert 'svc_epilogue_execute' not in PUBLIC and 'svc_descriptor_execute' not in PUBLIC
    for function in ('svc_bounds_run', '_gather', '_fit_ovo', '_platt'):
        src = body(function, FIT)
        assert 'drivers: SvcDrivers' in src.split('raises', 1)[0]
    assert 'drivers.epilogue(SVC_GLUE_SELECT,' in body('_fit_ovo', FIT) and 'drivers.epilogue(SVC_GLUE_SELECT,' in body('_platt', FIT)
    assert 'drivers.epilogue(SVC_GLUE_FOLD,' in body('_platt', FIT) and 'drivers.epilogue(SVC_GLUE_FOLD_FINISH,' in body('_platt', FIT)
    assert 'drivers.splitmix(' in body('_platt', FIT) and 'drivers.platt(' in body('_platt', FIT)
    assert 'drivers.epilogue(SVC_GLUE_C_ROWS,' in body('svc_bounds_run', FIT)
    assert RUN.count('drivers.epilogue(') == 3  # the cross-kernel column gather (SVC machines, SVR) and the per-row epilogue


def test_epilogue_typed_bridge_retains_numerical_owners():
    old = (ROOT / 'providers/python/svm_impl_svc_epilogue_pyapi_door.mojo').read_text()
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
    assert 'svr_fit_host_borrowed(' in source and 'svr_predict_host(' in source
    assert 'if kernel != 4:' in source
    # e69fd3d769 / 7e20f47df1: SVR fit and query go through the typed native SVR boundary of the same binding.
    assert 'svr_connection(binding)' in body('svr_fit_unpack') and 'unpacked.connection.driver.fit(unpacked.request)' in body('_svr_fit')
    assert 'svr_connection(binding)' in body('svr_query_unpack') and 'unpacked.connection.driver.query(unpacked.request, unpacked.resident)' in body('_svr_predict')
    for name in ('_svr_fit', '_svr_predict', 'svr_fit_unpack', 'svr_query_unpack'):
        assert 'family_call(' not in body(name)
    assert 'ip.unsafe_store(2, Float64(res.n_iter))' in source
    assert 'if len(params) != 9 and len(params) != 10 and len(params) != 12:' in source
    assert 'if len(params) != 7 and len(params) != 9:' in source


def test_svc_auxiliary_dispatch_retains_native_owners_and_operation_metadata():
    aux=(ROOT/'svm/native_auxiliary_dispatch.mojo').read_text()
    for name in ('scale_gamma_limbs_device','ovo_layout_device','ovo_dual_gemv_device','ovo_check_result'):
        assert name+'(' in aux
    # gamma='scale': the binding's flag word is read at UNPACK (the refusal), the limbs launched by RUN.
    assert 'drivers.flags()' in body('svm_scale_gamma_admit')
    assert 'drivers.scale_gamma_limbs(' in body('svc_gamma_run', FIT) and 'scale_gamma_from_limbs(limbs, cells, cols)' in body('svc_gamma_run', FIT)
    for name, code in (('FLAGS', 8), ('SCALE_GAMMA', 9), ('OVO', 10)):
        assert f'SVC_TABLE_{name} = {code}' in TRANSPORT
    assert 'svc_coef_run(drivers,' in body('_linear_coef') and RUN.count('drivers.dual_gemv(') == 1
    assert 'SVC_TABLE_GEMV = 11' in TRANSPORT
    assert 'drivers.ovo_layout(addresses, words)' in body('_fit_ovo', FIT)
    # The Python-list adapter of the table (providers/python/svc_descriptor_pyapi_door.mojo) is gone: the
    # typed drivers call the table's entry directly.
    assert not (ROOT/'providers/python/svc_descriptor_pyapi_door.mojo').exists()
    assert 'native_descriptor_call_fixed(self.table, SVC_TABLE_EPILOGUE, a, p, f)' in TRANSPORT


def test_svm_shared_inputs_are_snapshotted_outside_pair_and_fold_loops():
    # X's address, rows, columns and cell count enter the plan once (SvcFitPlan); the pair and fold loops read the plan.
    for field in ('plan.x = x_meta.source.addr', 'plan.rows = n_rows', 'plan.cols = n_cols', 'plan.cells = x_meta.count'):
        assert field in body('svc_fit_unpack')
    for name in ('_fit_ovo', '_platt'):
        assert 'plan: SvcFitPlan' in body(name, FIT).split('raises', 1)[0]
    for name in ('svc_fit_unpack', 'svr_fit_unpack'):
        src=body(name)
        assert src.count('_svm_fit_admit(args[1],') == 1
        # the admitted X outlives the solve: named after it, or moved into the unpacked record the door holds
        assert '_ = x_meta' in src or 'x_meta^' in src
    assert body('_svm_fit_admit').count('admit_family_meta(source,') == 1
    for name, record in (('svc_query_unpack', 'q'), ('svr_query_unpack', 'q_meta')):
        # the query record (_SvmQuery) holds the admitted address, read once; moved into the unpacked record the door holds past RUN
        assert record + '.address' in body(name)
        assert record + '^' in body(name)
    metadata=(ROOT/'providers/python/array_metadata_pyapi_door.mojo').read_text()
    for slot in ('_store', '_base', '_pin', '_mv'):
        assert f'array_slot_once(value, "{slot}", context)' in metadata
    owned=(ROOT/'bindings/admitted_family_door.mojo').read_text().split('def admit_family_owned(value: ArrayMetadata',1)[1].split('\ndef ',1)[0]
    assert 'value.owned_writable' in owned
    assert 'value.keep.' not in owned


def test_shared_weight_snapshot_keeps_cast_order_and_reuses_admitted_address():
    source=(ROOT/'bindings/admitted_family_door.mojo').read_text()
    take=lambda name: re.search(rf'^def {name}\(.*?(?=^def |\Z)', source,re.M|re.S).group()
    admission=take('_weight_buffer')
    assert admission.count('array_shape_once(view, context)') == 1
    for old in ('view.shape', 'view.ndim', 'view.size', 'b.hasattr', 'b.float'):
        assert old not in admission
    assert 'admitted_cast_f64(raw.addr' in admission
    assert 'admitted_cast_f32(raw.addr' in admission
    validate=take('admit_family_weights')
    assert 'snapshot.source.addr' in validate
    assert validate.index('WEIGHTS_NONFINITE') < validate.index('WEIGHTS_NEGATIVE') < validate.index('WEIGHTS_ALL_ZERO')
    weights=body('svm_weights_unpack')
    assert 'admit_family_weights(' in weights
    # both owners of the admitted weights are held for the operation (ArrayPin since c7893b3b8d)
    assert 'owners.append, weights_pin.keep' in weights and 'owners.append, weights_pin.owner' in weights
    assert 'sw_address = weights.source.addr' in weights


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
    old=(ROOT/'providers/python/svm_host_svc_proba_pyapi_door.mojo').read_text()
    calls=r'\bhost_(?:gather|select|fold_split|fold_finish|c_rows|check_pairs|pair_epilogue|platt_train|splitmix_perm)\('
    assert re.findall(calls,native)==re.findall(calls,old)
    assert 'dec_addr if nsr * nsc > 0 else dst' in native
    assert 'pairs_addr if mode != EPI_BINARY_CODES else dst' in native
    _,errors=audit(ROOT,('svm.host.native_descriptor_dispatch',))
    assert not errors,errors
