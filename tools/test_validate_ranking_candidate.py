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


def test_completed_campaign_releases_without_synthetic_pause(tmp_path):
    ack, proc = boundary(tmp_path)
    (tmp_path / 'PAUSE').unlink()
    campaign = tmp_path / 'campaign.json'
    data = V.read(campaign)
    data.update(status='finished', expected=1, recorded=1, recipes={'one': {'status': 'complete'}})
    V.write(campaign, data)
    exit_path = tmp_path / 'controller.exit'; exit_path.write_text('0\n')
    a = V.read(ack); a['campaigns'] = [dict(path=str(campaign), controller_exit=str(exit_path))]
    V.write(ack, a)
    result = V.check_boundary(ack, 'nvidia', 90, proc)
    assert V.boundary_still_held(result)
    assert result['campaigns'][0]['gate'] == 'finished'


def test_finished_campaign_missing_work_is_not_released(tmp_path):
    ack, proc = boundary(tmp_path)
    campaign = tmp_path / 'campaign.json'
    data = V.read(campaign)
    data.update(status='finished', expected=2, recorded=1, recipes={'one': {'status': 'complete'}})
    V.write(campaign, data)
    with pytest.raises(ValueError, match='incomplete expected coverage'):
        V.check_boundary(ack, 'nvidia', 90, proc)


def test_finished_campaign_requires_actual_successful_controller_exit(tmp_path):
    ack, proc = boundary(tmp_path)
    campaign = tmp_path / 'campaign.json'
    data = V.read(campaign)
    data.update(status='finished', expected=0, recorded=0, recipes={})
    V.write(campaign, data)
    with pytest.raises(ValueError, match='retained controller exit status 0'):
        V.check_boundary(ack, 'nvidia', 90, proc)


def test_all_queued_campaigns_must_acknowledge_boundary(tmp_path):
    ack, proc = boundary(tmp_path)
    second = tmp_path / 'second-campaign.json'
    V.write(second, dict(status='running', vendor='nvidia', provider_deadline=time.time()+3600))
    a = V.read(ack); a['campaigns'] = [dict(path=a['campaign']), dict(path=str(second))]
    V.write(ack, a)
    with pytest.raises(ValueError, match='neither paused between cells nor fully finished'):
        V.check_boundary(ack, 'nvidia', 90, proc)


def failed_boundary(tmp_path):
    ack, proc = boundary(tmp_path)
    (tmp_path / 'PAUSE').unlink()
    campaign = tmp_path / 'campaign.json'
    data = V.read(campaign)
    data.update(status='stopped', expected=1, recorded=1,
                recipes={'one': {'status': 'failed', 'returncode': 1}},
                stop_reason='timed driver failed rc1')
    V.write(campaign, data)
    exit_path = tmp_path / 'controller.exit'; exit_path.write_text('1\n')
    failure = tmp_path / 'timed-driver.log'; failure.write_text('retained actual failure\n')
    a = V.read(ack)
    a.update(no_pending_queue_launch=True, campaigns=[dict(path=str(campaign), controller_exit=str(exit_path),
             failure_acknowledged=True, failure_evidence=[str(failure)])])
    V.write(ack, a)
    return ack, proc


def test_acknowledged_stopped_failure_releases_without_claiming_success(tmp_path):
    ack, proc = failed_boundary(tmp_path)
    result = V.check_boundary(ack, 'nvidia', 90, proc)
    item = result['campaigns'][0]
    assert item['gate'] == 'stopped'
    assert item['controller_exit_status'] == 1
    assert item['state']['recipes']['one']['status'] == 'failed'
    assert item['failure_evidence'][0]['sha256'] == V.sha(tmp_path / 'timed-driver.log')
    assert V.boundary_still_held(result)


@pytest.mark.parametrize('missing', ['failure_acknowledged', 'no_pending_queue_launch', 'failure_evidence', 'nonzero_exit'])
def test_stopped_failure_requires_explicit_retained_acknowledgement(tmp_path, missing):
    ack, proc = failed_boundary(tmp_path)
    a = V.read(ack)
    if missing == 'no_pending_queue_launch':
        a.pop(missing)
    elif missing == 'nonzero_exit':
        (tmp_path / 'controller.exit').write_text('0\n')
    else:
        a['campaigns'][0].pop(missing)
    V.write(ack, a)
    with pytest.raises(ValueError):
        V.check_boundary(ack, 'nvidia', 90, proc)


def test_acknowledged_stopped_failure_still_refuses_live_workers(tmp_path):
    ack, proc = failed_boundary(tmp_path)
    session = tmp_path / 'measured/cell/timed-session'; session.mkdir(parents=True)
    V.write(session / 'session.json', dict(status='finished', remaining_workers=[], session='failed-worker', pid=4321))
    process = proc / '4321'; process.mkdir()
    (process / 'environ').write_bytes(b'MOJOLEARN_BOARD_SESSION=failed-worker\0')
    with pytest.raises(ValueError, match='measurement workers still active'):
        V.check_boundary(ack, 'nvidia', 90, proc)


def fixture_wheel(tmp_path, name, payload):
    import base64
    import csv
    import hashlib
    import io
    import zipfile
    info = name.replace('-', '_') + '-0.8.45.dist-info'
    files = dict(payload)
    files[info + '/METADATA'] = ('Metadata-Version: 2.1\nName: ' + name + '\nVersion: 0.8.45\n').encode()
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    for file, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
        writer.writerow([file, 'sha256=' + digest, len(data)])
    writer.writerow([info + '/RECORD', '', ''])
    files[info + '/RECORD'] = buffer.getvalue().encode()
    path = tmp_path / (name + '.whl')
    with zipfile.ZipFile(path, 'w') as archive:
        for file, data in files.items():
            archive.writestr(file, data)
    digest = V.sha(path)
    return dict(distribution=name, path=str(path), pypi_digest_match=True, sha256=digest, pypi_sha256=digest)


@pytest.mark.parametrize('vendor,foreign', [('nvidia', 'mojolearn-amd'), ('amd', 'mojolearn-nvidia')])
def test_overlay_extracts_only_verified_target_when_both_vendor_metadata_installed(tmp_path, monkeypatch, vendor, foreign):
    from types import SimpleNamespace
    _, directory, arch, plugin = V.VENDORS[vendor]
    installed = [SimpleNamespace(metadata={'Name': name}, version='0.8.45')
                 for name in ['mojolearn', 'mojolearn-nvidia', 'mojolearn-amd']]
    monkeypatch.setattr(V.metadata, 'distributions', lambda: installed)
    rel = 'mojolearn/' + directory + '/' + arch + '/identical/_mojolearn_gbdt.so'
    wheels = [fixture_wheel(tmp_path, 'mojolearn', {'mojolearn/__init__.py': b'verified core', 'mojolearn_diagnostics.py': b'verified diagnostic helper'}),
              fixture_wheel(tmp_path, plugin, {rel: b'verified released GBDT',
                  'mojolearn/' + directory + '/' + arch + '/identical/_mojolearn_linalg.so': b'unchanged linalg'})]
    manifest = tmp_path / 'wheel-manifest.json'; V.write(manifest, dict(all_digests_match=True, files=wheels))
    candidate = tmp_path / 'candidate.so'; candidate.write_bytes(b'raw candidate GBDT')
    out = tmp_path / 'out'; out.mkdir()
    site, target = V.package_overlay(out, vendor, candidate, manifest)
    assert target.read_bytes() == b'raw candidate GBDT'
    evidence = V.read(out / 'overlay-manifest.json')
    assert {d['distribution'] for d in evidence['base_distributions']} == {'mojolearn', plugin}
    assert [d['distribution'] for d in evidence['excluded_distributions']] == [foreign]
    assert not (site / (foreign.replace('-', '_') + '-0.8.45.dist-info')).exists()
    assert evidence['changed'] == [rel]
    assert (site / 'mojolearn/__init__.py').read_bytes() == b'verified core'
    unchanged = 'mojolearn/' + directory + '/' + arch + '/identical/_mojolearn_linalg.so'
    assert evidence['before'][unchanged] == evidence['after'][unchanged]
