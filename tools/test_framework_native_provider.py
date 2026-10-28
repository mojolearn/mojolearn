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
    assert 'imported_device_provider_run(snapshot, descriptors, addresses)' in frontend
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


def test_python_export_is_replaced_by_explicit_artifact_admission():
    frontend = (ROOT / 'python/mojolearn/adapters/_torch_compile.py').read_text()
    assert 'return _IMPORT_NATIVE("torch_capture",' in frontend
    assert 'self._import(gm)' not in frontend
    assert 'torch.onnx' not in frontend
    assert 'torch_dynamo_backends' not in (ROOT / 'python/pyproject.toml').read_text()
    public = (ROOT / 'python/mojolearn/importers/_program.py').read_text()
    assert 'return self._run_torch_native(self, tensors, None, False)' in public
    assert '_export_torch' not in public


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


def test_host_inference_operations_are_one_native_call():
    import ast
    tree = ast.parse((ROOT / 'python/mojolearn/_byte_lm_host.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LanguageModelInference')
    expected = {'shape', 'profile', 'parameters_sha256', 'logits', 'loss_bits', 'loss', 'next_bytes'}
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
    native = (ROOT / 'bindings/byte_lm_host_trainer_native.mojo').read_text().split('def byte_lm_host_infer(')[1].split('def byte_lm_host_train_step(')[0]
    assert native.count('value.shape') == 1 and native.count('value.strides') == 1
    assert 'neural_descriptor_execute(provider, 0,' in native
    assert 'neural_descriptor_execute(provider, 1,' in native
    assert 'neural_descriptor_execute(provider, 2,' in native
    assert 'byte_lm_lengths(' in native and 'neural_copy_row_prefixes(' in native
    assert 'argmax_rows_f32(' in native


def test_byte_config_keeps_frozen_dataclass_and_native_public_methods():
    import ast
    source = (ROOT / 'python/mojolearn/_byte_lm_config.py').read_text()
    assert '@dataclass(frozen=True, init=False, repr=False, eq=False)' in source
    assert 'ByteLanguageModelConfig.__setattr__ = _config_setattr' in source
    assert 'ByteLanguageModelConfig.__delattr__ = _config_delattr' in source
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    for method in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
        body = method.body
        if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        assert len(body) == 1 and isinstance(body[0], ast.Return), method.name
        assert body[0].value.func.id == '_CONFIG', method.name
    native = (ROOT / 'providers/python/byte_lm_metadata_pyapi.mojo').read_text()
    assert 'byte_lm_config_frozen_error' in native and 'stash_python_error()' in native
    assert 'native_storage_allocate(context["byte_lm_config_class"])' in native
    assert 'training_byte_shape_validate(shape.d, shape.chunked)' in native


def test_inference_construction_uses_native_registry_and_shared_lowbit_provider():
    source = (ROOT / 'python/mojolearn/_byte_lm_host.py').read_text()
    constructor = source.split('class LanguageModelInference:')[1].split('    def __init__(')[1].split('    @classmethod')[0]
    assert '_HOST_TRAIN(self, "inference_construct"' in constructor
    assert '_flatten_parameters(' not in constructor and '_load()' not in constructor
    native = (ROOT / 'bindings/byte_lm_host_trainer_native.mojo').read_text()
    assert 'linalg_ops_operation(PythonObject(None), "family_lin_from_bf16"' in native
    assert 'linalg_ops_operation(PythonObject(None), "family_lin_dequantize_int8"' in native
    assert 'List[String](length=2' in native
    assert 'py_call(py_fields(model).update, pending)' in native


def test_host_checkpoint_reuses_native_codec_and_full_state_validator():
    source = (ROOT / 'python/mojolearn/_byte_lm_host.py').read_text()
    body = source.split('def from_checkpoint(cls, path')[1].split('    @property')[0]
    assert '_HOST_TRAIN(cls, "inference_from_checkpoint"' in body
    assert '_decode_checkpoint' not in body
    native = (ROOT / 'bindings/byte_lm_host_trainer_native.mojo').read_text()
    assert 'byte_json_payload(byte_json_read(args[0], io_error))' in native
    assert '_validated_state(source, state_context)' in native
    assert 'if io_error.pending:' in native and 'return io_error^.restore()' in native


def test_imported_program_constructor_enters_native_metadata_boundary():
    source = (ROOT / 'python/mojolearn/importers/_program.py').read_text()
    constructor = source.split('def __init__(self, session, info, inputs, outputs):')[1].split('    def run(')[0]
    assert '_IMPORT_NATIVE(self, session, info, inputs, outputs, _IMPORT_CONTEXT)' in constructor
    assert 'super()' not in constructor and 'getattr' not in constructor
    native = (ROOT / 'providers/python/autograd_import_frontend_pyapi.mojo').read_text()
    assert 'program_admit_value(info)' in native
    assert 'neural_descriptor_entry(binding, "ops_imported_run")' in native
    assert '_initialize(pending, session, info, inputs, outputs, entries, context)' in native


def test_cpu_byte_lm_owner_uses_same_native_transport_without_python_proxy():
    for name in ('training/byte_lm_host_session.mojo', 'training/byte_lm_host_trainer_dispatch.mojo'):
        source = (ROOT / name).read_text()
        for forbidden in ('PythonObject', 'std.python', 'bindings.', 'providers.'):
            assert forbidden not in source
    table = (ROOT / 'training/byte_lm_host_trainer_dispatch.mojo').read_text()
    assert '"mojolearn.training.byte_lm.v1"' in table
    assert 'BYTE_HOST_TRAINER_COUNT = 12' in table
    assert 'host_trainer_recover(owner, completed)' in table
    bridge = (ROOT / 'python/mojolearn/_byte_lm_trainer_host.py').read_text()
    assert 'class _HostTrainerBinding' not in bridge and 'class _Session' not in bridge
    assert 'return host' in bridge
    owner = (ROOT / 'bindings/byte_lm_trainer_native.mojo').read_text()
    entry = owner.split('def _entry(')[1].split('def _has(')[0]
    assert 'return neural_descriptor_entry(binding, name, required)' in entry
    assert 'getattr' not in entry


def test_framework_setup_helpers_enter_native_metadata_directly():
    import ast
    for filename, names in (('max.py', ('custom_extensions', 'native_library_path', 'parameters')),
                            ('torch.py', ('library_path', 'native_library_path')),
                            ('_shipped.py', ('directory', 'manifest', 'path', 'series'))):
        parsed = ast.parse((ROOT / 'python/mojolearn/adapters' / filename).read_text())
        for node in parsed.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                statements = [s for s in node.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
                assert len(statements) == 1 and isinstance(statements[0], ast.Return)
                assert ast.unparse(statements[0].value.func).endswith('_NATIVE')
    source = (ROOT / 'providers/python/framework_metadata_pyapi.mojo').read_text()
    assert 'native_json_parse(raw)' in source
    assert 'native_storage_allocate(layout["class"])' in source
    assert 'Path.__init__' in source  # explanatory prohibition, not an invocation


def test_onnx_reader_and_resolver_use_common_native_resource():
    table = (ROOT / 'autograd/import_reader_dispatch.mojo').read_text()
    assert 'onnx_read(request[].path)' in table and 'onnx_resolve(addresses[0]' in table
    assert 'words[0] != onnx_registry_token()' in table
    assert 'PythonObject' not in table and 'std.python' not in table
    owner = (ROOT / 'providers/python/autograd_reader_frontend_pyapi.mojo').read_text()
    assert 'neural_descriptor_adopt(selected[1], handle, 3, scope.copy())' in owner
    assert 'py_setattr(array, "owner", resource)' in owner
    assert 'resource.downcast_value_ptr[NativeDescriptorResource]()[].close()' in owner
    assert 'weakref' not in owner and 'ops_session_initialize' not in owner
    reader = (ROOT / 'python/mojolearn/importers/onnx.py').read_text()
    assert 'return _READER_NATIVE("reader_read", path, binding, _READER_CONTEXT)' in reader
    graph = (ROOT / 'python/mojolearn/importers/_graph.py').read_text()
    assert 'return _READER_NATIVE("reader_resolve", self, shapes, _READER_CONTEXT)' in graph
    for name in ('resolve', 'resolve_shapes'):
        assert f'return _READER_NATIVE("{name}", g, shapes, _READER_CONTEXT)' in graph


def test_cpu_profile_preserves_public_default_shape_independently_of_build_toggle():
    source = (ROOT / 'providers/python/byte_lm_host_compat_pyapi.mojo').read_text()
    assert 'ByteConfig(chunked_lm_head_v2=False).profile()' in source
