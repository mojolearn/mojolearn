"""Both CPU and GPU probes must witness settings that seal checks after timing."""
import pytest
import bench_board_evidence as E
import bench_board_probe as P


@pytest.mark.parametrize('device', ['cpu', 'gpu'])
def test_probe_and_timed_worker_environments_match(device, monkeypatch):
    monkeypatch.setenv('MOJOLEARN_MEASUREMENT_RECEIPTS', '/test/receipts')
    monkeypatch.setenv('MOJOLEARN_NUMERIC_MODE', 'identical')
    monkeypatch.setenv('MOJOLEARN_BENCH_CPU_POLICY', 'full-allocation')
    monkeypatch.setattr(E, 'hardware_context', lambda: {'status': 'verified'})
    info = P.library_identity(dict(device=device, library='mojolearn', version='test', device_name='test'))
    assert info['runtime_environment'] == E.runtime_environment()
    assert info['runtime_environment']['MOJOLEARN_NUMERIC_MODE'] == 'identical'
    assert ('hardware_receipt' in info) == (device == 'gpu')
