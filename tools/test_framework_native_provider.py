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
