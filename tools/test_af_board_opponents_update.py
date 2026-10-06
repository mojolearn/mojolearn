import copy
import pytest
from af_board_opponents_update import merge


def fixtures():
    box={'gpu':{'name':'Apple M3 Ultra'}}
    own={'arm':'ours','library':'mojolearn','mode':'identical','device':'gpu','shape':'100x3','status':'ok','median_ms':10}
    opp={'arm':'sklearn-cpu','library':'sklearn','device':'cpu','mode':'opponent','shape':'100x3','status':'ok','rounds':1,'times_ms':[20],'median_ms':20,'warmup_ms':22}
    race={'family':'classical','lane':'x','dataset':'taxi','cells':[own]}
    board={'box':box,'races':{'r':race}}
    source={'box':box,'races':{'r':dict(race,cells=[opp])}}
    resources={'cpu_count':28,'thread_caps':{'OMP_NUM_THREADS':None},'harness_sha':'frozen'}
    return board,source,resources


def test_preserves_own_and_keeps_failed_attempt_visible():
    board,source,resources=fixtures();before=copy.deepcopy(board)
    source['races']['r']['cells'][0].update(status='HOST-MEMORY',median_ms=None,times_ms=[],rounds=0)
    result,counts=merge(board,source,resources,'hash','snapshot.json')
    assert board==before
    saved=before['races']['r']['cells'][0]
    assert all(result['races']['r']['cells'][0][k]==v for k,v in saved.items())
    assert counts['failed']==1 and result['races']['r']['cells'][-1]['median_ms'] is None


def test_stored_cell_not_relabelled_uncapped():
    board,source,resources=fixtures();source['races']['r']['cells'][0]['stored']={'commit':'old'}
    result,counts=merge(board,source,resources,'hash','snapshot.json')
    assert counts['stored_skipped']==1 and counts['imported']==0
    assert result['races']==board['races']


def test_shape_mismatch_refused():
    board,source,resources=fixtures();source['races']['r']['cells'][0]['shape']='200x3'
    with pytest.raises(AssertionError,match='shape mismatch'):merge(board,source,resources,'hash','snapshot.json')


def test_uncapped_repair_keeps_historical_comparison_and_is_idempotent():
    board,source,resources=fixtures();before=copy.deepcopy(board)
    source['races']['r']['cells'][0]['shape']='200x3'
    result,count=merge(board,source,resources,'hash','snapshot.json',separate_workloads=True)
    assert result['races']==before['races']
    assert count['successful']==1 and len(result['extra_races'])==1
    key,race=next(iter(result['extra_races'].items()))
    assert key.startswith('r/workload=') and race['original_race_id']=='r'
    assert all(c['library']!='mojolearn' for c in race['cells'])
    assert race['cells'][0].get('ratio_ours_identical_over') is None
    again,count=merge(result,source,resources,'hash','snapshot.json',separate_workloads=True)
    assert count['already_current']==1 and again['extra_races']==result['extra_races']


def test_full_failed_attempt_keeps_its_own_workload_scope():
    board,source,resources=fixtures()
    source['races']['r']['cells'][0].update(shape='200x3',status='REFUSED(solver overflow)',
                                         median_ms=None,times_ms=[],rounds=0)
    result,count=merge(board,source,resources,'hash','snapshot.json',separate_workloads=True)
    assert result['races']==board['races'] and count['failed']==1
    assert len(result['extra_races'])==1
    assert next(iter(result['extra_races'].values()))['cells'][0]['median_ms'] is None


def test_idempotent_cells_and_resource_proof():
    board,source,resources=fixtures();first,_=merge(board,source,resources,'hash','snapshot.json')
    second,count=merge(first,source,resources,'hash','snapshot.json')
    assert first['races']==second['races'] and count['already_current']==1
    assert second['races']['r']['cells'][-1]['resource_policy']['available_logical_cpus']==28
    resources['thread_caps']['OMP_NUM_THREADS']='1'
    with pytest.raises(AssertionError):merge(board,source,resources,'hash','snapshot.json')


def test_failed_missing_shape_is_evidence_without_comparison():
    board,source,resources=fixtures()
    board['races']['r']['cells'][0].pop('shape')
    source['races']['r']['cells'][0].update(status='REFUSED(unsupported MPS)',median_ms=None,times_ms=[],rounds=0)
    source['races']['r']['cells'][0].pop('shape')
    result,count=merge(board,source,resources,'hash','snapshot.json')
    assert count['failed']==1 and count['successful']==0
    assert result['races']['r']['cells'][-1]['median_ms'] is None
