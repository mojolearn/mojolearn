"""Selection-only tests: no native imports, device discovery, or GPU work."""
import importlib.util
import json
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('py_job_check', HERE/'check.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def manifest(tmp_path, rows):
    p = tmp_path/'python/mojolearn/host_surface.py'
    p.parent.mkdir(parents=True)
    p.write_text('LANE_NOT_APPLICABLE = "NOT APPLICABLE"\n'
                 'def lane_exposure(lanes, device_class):\n'
                 f'    return {rows!r}\n')


def test_physical_lanes_excluded_before_any_cpu_pool_even_on_two_gpu_host(tmp_path, monkeypatch):
    # Visible devices never promote this explicitly single-device driver's scope.
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    manifest(tmp_path, {'par-gmm': {'status':'NOT APPLICABLE', 'reason':'needs two devices'},
                        'gmm': {'status':'EXPOSED'}})
    assert check.single_device_plan(tmp_path, ['par-gmm','gmm']) == (['gmm'], {'par-gmm':'needs two devices'})


def test_undeclared_physical_lane_refuses_instead_of_gpu_reference_pool(tmp_path):
    manifest(tmp_path, {'par-gmm': {'status':'EXPOSED'}})
    with pytest.raises(ValueError, match='separate device-axis'):
        check.single_device_plan(tmp_path, ['par-gmm'])


def test_unavailable_is_not_structural_exclusion(tmp_path):
    manifest(tmp_path, {'unknown': {'status':'UNAVAILABLE', 'reason':'missing binding'}})
    assert check.single_device_plan(tmp_path, ['unknown']) == (['unknown'], {})


def saved_exclusion(path, declared=True):
    path.mkdir()
    row = {'gpu_vs_cpu':'NOT APPLICABLE','excluded':True,'reason':'needs two devices'}
    (path/'lanes.json').write_text(json.dumps({'par-gmm':row}))
    if declared:
        (path/'coverage.json').write_text(json.dumps(dict(scope='single-device', requested=['par-gmm'],
            selected=[], excluded={'par-gmm':'needs two devices'})))


def test_cross_declared_exclusions_are_reported_without_numeric_pass(tmp_path, monkeypatch, capsys):
    base,new=tmp_path/'base',tmp_path/'new'
    saved_exclusion(base);saved_exclusion(new)
    monkeypatch.setenv('NO_PROBE','1')
    assert check.cross(base,new,['par-gmm']) == 0
    text=capsys.readouterr().out
    assert 'no numerical comparison' in text and '1 lanes explicitly excluded' in text
    assert 'SAME' not in text.splitlines()[1]


@pytest.mark.parametrize('declared', [False, True])
def test_cross_cannot_turn_missing_or_one_sided_exclusion_into_success(tmp_path, monkeypatch, declared):
    base,new=tmp_path/'base',tmp_path/'new'
    saved_exclusion(base,declared);saved_exclusion(new,False)
    monkeypatch.setenv('NO_PROBE','1')
    assert check.cross(base,new,['par-gmm']) == 1
