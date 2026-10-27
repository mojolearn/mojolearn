"""Architectural source checks; actual public probes are retained separately."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def test_device_execution_is_shared_and_native():
    owner = (ROOT / 'autograd/program_device_execution.mojo').read_text()
    for forbidden in ('PythonObject', 'std.python', 'bindings.', 'providers.'):
        assert forbidden not in owner
    packed = (ROOT / 'providers/python/autograd_program_device_pyapi.mojo').read_text()
    assert 'program_device_execute[KIND](' in packed
    assert 'replay_values(s[]' not in packed
    table = (ROOT / 'sequence/model_device_dispatch.mojo').read_text()
    assert 'if op == PROGRAM_DEVICE_RUN:' in table and 'program_device_request(' in table
    host = (ROOT / 'sequence/model_host_dispatch.mojo').read_text()
    assert 'if op == PROGRAM_DEVICE_RUN:' in host
    assert 'this session binding takes no device buffers' in host


def test_torch_uses_common_provider_without_internal_packed_execution():
    frontend = (ROOT / 'providers/python/autograd_torch_program_pyapi.mojo').read_text()
    assert 'imported_device_provider_run(program, descriptors, addresses)' in frontend
    assert 'py_call(run,' not in frontend
    provider = (ROOT / 'providers/python/autograd_device_provider_pyapi.mojo').read_text()
    assert 'neural_descriptor_execute(owner[].provider, PROGRAM_DEVICE_RUN,' in provider
    assert 'NativeDescriptorResource' in provider and 'neural_descriptor_entry(provider[].module,' in provider
    assert 'List[Float64](), False)' in provider  # preserve GIL-held Session semantics
    assert 'struct ' not in provider  # no algorithm/framework-specific provider type


def test_device_metadata_captured_once_and_kept_natively():
    frontend = (ROOT / 'providers/python/autograd_torch_program_pyapi.mojo').read_text()
    assert frontend.count('native_torch_device(inputs[') == 1
    assert 'struct TorchDeviceSnapshot(Movable):' in frontend
    route = frontend.split('def _torch_route_with_devices(')[1].split('def _torch_run_host[')[0]
    assert 'native_torch_device(' not in route
    assert 'target = devices[0].owner' in frontend
    assert 'native_torch_storage_metadata(value)' in frontend


def test_python_export_debt_is_not_declared_migrated():
    frontend = (ROOT / 'python/mojolearn/adapters/_torch_compile.py').read_text()
    assert 'Still prohibited runtime debt' in frontend
    assert 'self._program, self._versions = self._import(gm)' in frontend


def test_import_and_cleanup_use_the_same_session_provider():
    source = (ROOT / 'providers/python/autograd_import_frontend_pyapi.mojo').read_text()
    body = source.split('def _import(')[1].split('def ops_import_model_c(')[0]
    assert 'neural_descriptor_execute(owner[].provider, PROGRAM_IMPORT,' in body
    assert 'py_call(entries["onnx_import' not in body
    assert 'py_call(entries["onnx_buffers"]' not in body
    assert 'py_call(session._session_native' not in body
    assert 'downcast_value_ptr[NativeDescriptorResource]()[].close()' in body
    owner = (ROOT / 'autograd/program_import_request.mojo').read_text()
    assert 'onnx_import_file[E, KIND]' in owner and 'onnx_buffers(request[].reader)' in owner
    assert 'ops_session_registry_token()' in owner
    for forbidden in ('PythonObject', 'std.python', 'bindings.', 'providers.'):
        assert forbidden not in owner
    for filename in ('sequence/model_host_dispatch.mojo', 'sequence/model_device_dispatch.mojo'):
        assert 'if op == PROGRAM_IMPORT:' in (ROOT / filename).read_text()


def test_host_byte_lm_uses_common_provider_and_shared_native_execution():
    frontend = (ROOT / 'bindings/byte_lm_host_trainer_native.mojo').read_text()
    assert 'py_call_typed' not in frontend
    assert 'neural_descriptor_execute(provider, 3,' in frontend
    assert 'neural_descriptor_reuse(' in frontend
    for filename in ('training/byte_lm_host_execution.mojo', 'training/byte_lm_host_dispatch.mojo'):
        source = (ROOT / filename).read_text()
        for forbidden in ('PythonObject', 'std.python', 'bindings.', 'providers.'):
            assert forbidden not in source
    binding = (ROOT / 'bindings/_mojolearn_byte_lm_host.mojo').read_text()
    for name in ('logits', 'next', 'loss', 'train'):
        assert f'byte_host_{name}_native(' in binding
    assert 'byte_lm_host_api_v1' in binding
    ids = frontend.split('def _ids_i32(')[1].split('def _f32_array(')[0]
    assert ids.count('value.shape') == 1 and ids.count('value.strides') == 1


def test_host_trainer_public_frontdoors_and_shared_metadata():
    import ast
    tree = ast.parse((ROOT / 'python/mojolearn/_byte_lm_host.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LanguageModelHostTrainer')
    expected = {'__init__', 'from_state', 'shape', 'profile', 'completed_steps', 'parameters_',
                'm_', 'v_', 'gradient_', 'parameters_sha256', 'train_step', 'loss'}
    for method in cls.body:
        if not isinstance(method, ast.FunctionDef) or method.name not in expected:
            continue
        body = method.body
        if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        assert len(body) == 1 and isinstance(body[0], ast.Return), method.name
        assert body[0].value.func.id == '_HOST_TRAIN', method.name
        expected.remove(method.name)
    assert not expected
    for name in ('bindings/byte_lm_host_trainer_native.mojo', 'bindings/byte_lm_trainer_native.mojo'):
        assert 'from providers.python.byte_lm_metadata_pyapi import _Shape, _shape_of' in (ROOT / name).read_text()
    shared = (ROOT / 'providers/python/training_buffer_pyapi.mojo').read_text().split('def training_f32_cast_owned(')[1]
    assert shared.count('value.shape') == 1 and shared.count('value.strides') == 1
    assert 'native_strides, item, kind, True, pointer)' in shared
