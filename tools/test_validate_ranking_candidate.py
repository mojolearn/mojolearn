"""CPU-only checks for candidate evidence, comparators and ownership guards."""
import json
from pathlib import Path
import time

import numpy as np
import pytest

import validate_ranking_candidate as V


@pytest.mark.parametrize('name', ['libc.so.6', 'libm.so.6', 'libpthread.so.0', 'libdl.so.2',
    'librt.so.1', 'ld-linux-x86-64.so.2', 'libstdc++.so.6', 'libgcc_s.so.1',
    'libcuda.so.1', 'libnvidia-ptxjitcompiler.so.1', 'libamdhip64.so.6', 'libhsa-runtime64.so.1'])
def test_system_and_driver_libraries_never_override_worker(name):
    assert not V.runtime_allowed(dict(path='/compiler/env/lib/' + name, retained='runtime/' + name))


def test_compiler_library_allowed_but_system_origin_excluded():
    row = dict(path='/compiler/env/lib/libKGENCompilerRTShared.so', retained='runtime/libKGENCompilerRTShared.so')
    assert V.runtime_allowed(row)
    assert not V.runtime_allowed(dict(row, path='/usr/lib/libKGENCompilerRTShared.so'))


def report(path):
    path.mkdir()
    np.save(path / 'predict.npy', np.array([0., 1.], dtype=np.float32))
    np.save(path / 'loss_curve.npy', np.array([2., 1.], dtype=np.float64))
    np.savez(path / 'model.npz', model=np.array([1, 2], dtype=np.uint8), numeric_mode=np.array(['identical']))
    outputs = {key: dict(path=str(path / file), sha256=V.sha(path / file)) for key, file in
               [('predict', 'predict.npy'), ('loss_curve', 'loss_curve.npy'), ('model', 'model.npz')]}
    data = dict(complete=True, exit_status=0, failures=[], numeric_mode='identical', data_sha256=V.FIXTURE,
                cases=[dict(loss='test', tag=str(i), params=dict(a=1), outputs=outputs) for i in range(24)])
    V.write(path / 'summary.json', data)
    return path / 'summary.json'


def test_compare_complete_arrays_and_all_model_members(tmp_path):
    a, b = report(tmp_path / 'a'), report(tmp_path / 'b')
    result = V.compare(a, b)
    assert result['verdict'] == 'PASS'
    assert len(result['compared']) == 24 * 4


def test_compare_rejects_one_changed_bit_even_when_float_equal(tmp_path):
    a, b = report(tmp_path / 'a'), report(tmp_path / 'b')
    np.save(b.parent / 'predict.npy', np.array([-0., 1.], dtype=np.float32))
    data = V.read(b)
    for case in data['cases']:
        case['outputs']['predict']['sha256'] = V.sha(b.parent / 'predict.npy')
    V.write(b, data)
    result = V.compare(a, b)
    assert result['verdict'] == 'FAIL'
    assert all('predict/predict differs' in error for error in result['failures'])


def test_compare_rejects_missing_coverage_and_failed_run(tmp_path):
    a, b = report(tmp_path / 'a'), report(tmp_path / 'b')
    data = V.read(b); data['cases'].pop(); data['complete'] = False
    V.write(b, data)
    result = V.compare(a, b)
    assert result['verdict'] == 'FAIL'
    assert len(result['failures']) == 2


def test_compare_checks_artifact_hashes(tmp_path):
    a, b = report(tmp_path / 'a'), report(tmp_path / 'b')
    (b.parent / 'model.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='output artifact changed'):
        V.compare(a, b)


def boundary(tmp_path):
    campaign = tmp_path / 'campaign.json'
    V.write(campaign, dict(status='paused between cells', vendor='nvidia', heartbeat=time.time(), provider_deadline=time.time()+3600))
    (tmp_path / 'PAUSE').touch()
    ack = tmp_path / 'ack.json'
    V.write(ack, dict(authorized_by='root', vendor='nvidia', created_epoch=time.time(), native_commit=V.FREEZE, campaign=str(campaign)))
    proc = tmp_path / 'proc'; proc.mkdir()
    return ack, proc


def test_boundary_needs_live_explicit_pause(tmp_path):
    ack, proc = boundary(tmp_path)
    assert V.check_boundary(ack, 'nvidia', 90, proc)['sessions'] == []
    (tmp_path / 'PAUSE').unlink()
    with pytest.raises(ValueError, match='PAUSE marker missing'):
        V.check_boundary(ack, 'nvidia', 90, proc)


def test_boundary_refuses_stale_ack(tmp_path):
    ack, proc = boundary(tmp_path)
    data = V.read(ack); data['created_epoch'] -= 301; V.write(ack, data)
    with pytest.raises(ValueError, match='acknowledgement expired'):
        V.check_boundary(ack, 'nvidia', 90, proc)


def test_boundary_refuses_orphan_measurement_worker(tmp_path):
    ack, proc = boundary(tmp_path)
    session = tmp_path / 'measured/cell/timed-session'; session.mkdir(parents=True)
    V.write(session / 'session.json', dict(status='finished', remaining_workers=[], session='fixture'))
    worker = proc / '1234'; worker.mkdir()
    (worker / 'environ').write_bytes(b'OTHER=ignored\0MOJOLEARN_BOARD_SESSION=fixture\0')
    with pytest.raises(ValueError, match='measurement workers still active'):
        V.check_boundary(ack, 'nvidia', 90, proc)
