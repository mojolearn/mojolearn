import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import compare_ptx_vendor_evidence as c

SOURCE = 'a' * 40
HASH = 'b' * 16


def column(vendor, repeats):
    cell = dict(verdict='STABLE', hashes=[HASH] * repeats,
                parts=[{'predict': HASH}] * repeats, reload=[HASH] * repeats)
    for part in c.PARTS[1:]:
        cell[part] = [HASH] * repeats
        cell[part + '_verdict'] = 'STABLE'
    result = dict(commit=SOURCE, mode='identical', complete=True, skipped=[], repeats=repeats,
                  vendor=vendor, heldout_seed=1, parts_collected=list(c.PARTS[3:]),
                  fixtures={f:dict(X=HASH, y_clf=HASH, y_reg=HASH) for f in c.FIXTURES},
                  heldout={f:dict(X=HASH) for f in c.FIXTURES},
                  lane_revisions={'lane': 1}, batch_revisions={'lane': 1},
                  package={'bindings':[{'module':'test','sha256':'c'*64}]},
                  cells={'lane/'+f:copy.deepcopy(cell) for f in c.FIXTURES})
    for part in c.PARTS[3:]:
        result[part + '_protocol'] = {'rows': 16}
    return result


@pytest.fixture
def columns():
    with patch.object(c.identity_columns.vref(), 'record_device_class', side_effect=lambda d, p:(d['vendor'], None)):
        yield column('nvidia', 2), column('apple', 1)


def test_equal_raw_scope_and_no_mutation(columns):
    a,b=columns; before=copy.deepcopy(columns)
    result=c.compare_columns(a,b,SOURCE,'apple')
    assert result['passed'] and result['compared_hashes']==27
    assert result['excluded']['parts']==['batch','rlpair']
    assert columns==before


@pytest.mark.parametrize('field', ['commit','heldout_seed','lane_revisions','batch_revisions','batchgrad_protocol','fixtures','heldout'])
def test_metadata_tamper(columns,field):
    a,b=columns;b[field]={} if isinstance(b[field],dict) else 'wrong'
    with pytest.raises(ValueError):c.compare_columns(a,b,SOURCE,'apple')


@pytest.mark.parametrize('field', ['hashes','infer','model','parts','reload','ragged'])
def test_missing_hash_rejected(columns,field):
    a,b=columns;del b['cells']['lane/base'][field]
    with pytest.raises(ValueError):c.compare_columns(a,b,SOURCE,'apple')


def test_actual_hash_mismatch_is_reported(columns):
    a,b=columns;b['cells']['lane/base']['batchgrad']=['d'*16]
    result=c.compare_columns(a,b,SOURCE,'apple')
    assert not result['passed'] and len(result['mismatches'])==1


def test_constituent_hash_mismatch_is_reported(columns):
    a,b=columns;b['cells']['lane/base']['parts']=[{'predict':'d'*16}]
    assert not c.compare_columns(a,b,SOURCE,'apple')['passed']


def test_bilateral_structural_na_only(columns):
    a,b=columns
    for doc in (a,b):
        for cell in doc['cells'].values():
            cell['ragged']=['n/a:no-sequence-axis']*doc['repeats'];cell['ragged_verdict']='N/A'
    assert c.compare_columns(a,b,SOURCE,'apple')['structural_na']==3
    b['cells']['lane/base']['ragged']=['n/a:skipped (--no-ragged)']
    with pytest.raises(ValueError):c.compare_columns(a,b,SOURCE,'apple')


def test_repeat_disagreement_rejected(columns):
    a,b=columns;a['cells']['lane/base']['infer'][1]='d'*16
    with pytest.raises(ValueError):c.compare_columns(a,b,SOURCE,'apple')


def test_reference_lane_omission_rejected(columns):
    a,b=columns;del b['cells']['lane/base']
    with pytest.raises(ValueError):c.compare_columns(a,b,SOURCE,'apple')


def test_receipt_column_tamper_rejected(tmp_path):
    manifest=tmp_path/'manifest.json';receipt=tmp_path/'run.json';column_path=tmp_path/'column.json'
    manifest.write_text(json.dumps({'source_commit':SOURCE}));column_path.write_text('{}')
    receipt.write_text(json.dumps({'role':'baseline','column_file':'column.json','column_sha256':'0'*64}))
    with patch.object(c.baseline,'manifest_files',return_value={}):
        with pytest.raises(ValueError,match='Column hash changed'):
            c.report(manifest,receipt,{'apple':column_path,'amd':column_path},SOURCE,'f'*64,
                     {'apple':'e'*64,'amd':'e'*64})


def test_whole_report_pins_inputs_and_never_admits(tmp_path, columns):
    a,b=columns
    amd=copy.deepcopy(b);amd['vendor']='amd'
    paths={name:tmp_path/(name+'.json') for name in ('manifest','receipt','ptx','apple','amd')}
    docs={'manifest':{'source_commit':SOURCE}, 'ptx':a, 'apple':b, 'amd':amd}
    for name,doc in docs.items():paths[name].write_text(json.dumps(doc))
    receipt=dict(role='baseline',column_file='ptx.json',column_sha256=c.baseline.sha(paths['ptx']),
                 lanes=['lane'],fixtures=list(c.FIXTURES),hardware={'uuid':'witness'},driver_version='driver')
    paths['receipt'].write_text(json.dumps(receipt))
    refs={name:paths[name] for name in ('apple','amd')}
    hashes={name:c.baseline.sha(path) for name,path in refs.items()}
    with patch.object(c.baseline,'manifest_files',return_value={}), patch.object(c.baseline,'validate_receipt') as validate:
        result=c.report(paths['manifest'],paths['receipt'],refs,SOURCE,'f'*64,hashes)
        validate.assert_called_once()
        assert result['passed'] and not result['identical_qualified'] and not result['release_qualified']
        assert len(result['inputs'])==5
        paths['apple'].write_text(paths['apple'].read_text()+' ')
        with pytest.raises(ValueError,match='Reference file hash changed'):
            c.report(paths['manifest'],paths['receipt'],refs,SOURCE,'f'*64,hashes)


def test_extra_ptx_fixture_explicitly_excluded(columns):
    a,b=columns;a['fixtures']['large']=a['fixtures']['base'];a['cells']['lane/large']=copy.deepcopy(a['cells']['lane/base'])
    result=c.compare_columns(a,b,SOURCE,'apple')
    assert result['excluded']['ptx_fixtures']==['large'] and result['compared_hashes']==27


def test_unexpected_shared_lane_cannot_be_intersected_away(columns):
    a,b=columns;a['cells']['new/base']=copy.deepcopy(a['cells']['lane/base'])
    with pytest.raises(ValueError,match='Shared lane coverage differs'):c.compare_columns(a,b,SOURCE,'apple')
