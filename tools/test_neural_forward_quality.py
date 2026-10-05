"""Saved-output forward quality uses only untimed own host inference."""
import types
import numpy as np
import pytest
import bench_board_neural as n


@pytest.fixture
def host(monkeypatch):
    calls = []
    class Reference:
        def __init__(self, weights, **kwargs):calls.append(('construct', weights, kwargs))
        def forward(self, x):calls.append(('forward', x.copy()));return x.copy()
    names = ['TransformerBlockInference','Mamba1BlockInference','Mamba2BlockInference','Mamba3BlockInference']
    mod = types.SimpleNamespace(**{name: type(name, (Reference,), {}) for name in names})
    monkeypatch.setattr(n, '_ours_module', lambda: mod)
    return calls


def data():
    return {'x': np.array([[[0., 1., -2.]]], dtype=np.float32), 'w:weight': np.array([3.],dtype=np.float32)}


@pytest.mark.parametrize('lane', n.FORWARD_REFERENCE_LANES)
def test_fast_only_has_real_host_quality(lane, host):
    d=data();q=n.quality(lane,d,{'ours-fast':{'y':d['x'].copy()}},shape='small')['ours-fast']
    assert q['host_reference_passed'] and q['host_reference_finite']
    assert q['max_abs_diff_vs_host']==0 and q['host_reference_tolerance_ratio']==0
    assert [call[0] for call in host]==['construct','forward']
    assert host[0][2]['numeric_mode']=='identical'
    np.testing.assert_array_equal(host[0][1]['weight'],d['w:weight'])
    if lane=='transformer-forward':
        expected=n._dims_of(lane,'small')
        assert host[0][2]['n_heads']==expected['n_heads']
        assert host[0][2]['norm_eps']==n.BLOCK_NORM_EPS


@pytest.mark.parametrize('change', ['wrong', 'nan', 'infinity', 'shape'])
def test_bad_saved_outputs_cannot_pass(change,host):
    d=data();y=d['x'].copy()
    if change=='shape':y=y[..., :2]
    else:y.flat[0]={'wrong':.01,'nan':np.nan,'infinity':np.inf}[change]
    quality=n.quality('mamba2-forward',d,{'ours-fast':{'y':y}},shape='small')
    assert not quality['ours-fast']['host_reference_passed']
    report={'lane':'mamba2-forward','arms':{'ours-fast':{'status':'ok','ms':[2.]}},'quality':quality}
    n.mark_forward_quality_failures(report)
    assert report['arms']['ours-fast']['status']=='quality_failed'
    assert report['arms']['ours-fast']['ms']==[2.]


def test_elementwise_tolerance_does_not_hide_small_element_error(host):
    d=data();d['x'][0,0,2]=1e5;y=d['x'].copy();y.flat[0]=2e-6
    q=n.quality('mamba1-forward',d,{'ours-fast':{'y':y}},shape='small')['ours-fast']
    assert not q['host_reference_passed'] and q['host_reference_tolerance_ratio']>1


def test_host_runs_once_for_both_saved_arms(host):
    d=data();q=n.quality('mamba3-forward',d,{a:{'y':d['x']} for a in ('ours','ours-fast')},shape='small')
    assert all(q[a]['host_reference_passed'] for a in q)
    assert sum(call[0]=='forward' for call in host)==1


def test_missing_host_or_saved_output_is_a_quality_failure():
    for quality in ({'error':'host binding unavailable'}, {}):
        report={'lane':'transformer-forward','arms':{'ours-fast':{'status':'ok'},'torch-cpu':{'status':'ok'}},'quality':quality}
        n.mark_forward_quality_failures(report)
        assert report['arms']['ours-fast']['status']=='quality_failed'
        assert report['arms']['torch-cpu']['status']=='ok'


def test_no_host_run_for_opponents_only(host):
    d=data();q=n.quality('mamba1-forward',d,{'torch-cpu':{'y':d['x']}},shape='small')
    assert host==[] and q=={'torch-cpu':{}}
