import copy
import pytest

from admit_cluster_identity_repair import admit, validate_bindings
from test_compare_installed_identity import records


def inputs():
    reports, _ = records()
    repair = reports['cuda']
    repair['lanes'] = ['x-cluster-bgmm']
    for row in repair['cells']:
        row['lane'] = 'x-cluster-bgmm'
    initial = copy.deepcopy(repair)
    initial.update(exit=4, verdict='INCOMPLETE', lanes=['x-cluster-bgmm', 'ridge'])
    initial['execution'].update(completed_cells=2, total_cells=2)
    initial['cells'].extend([dict(row, lane='ridge') for row in initial['cells']])
    initial['cells'][0].update(state='REFUSED', value=None, error='memoryview(Array)')
    table = {'cells': {lane + '/base': {row['part']: {'ref': row['value']} for row in repair['cells']}
                       for lane in initial['lanes']}}
    return initial, repair, table


def test_preserves_failed_original_and_accounts_for_repair():
    initial, repair, table = inputs()
    original = copy.deepcopy(initial)
    result = admit(initial, repair, table, initial['lanes'], 'a' * 40)
    assert result['lanes'] == 2 and result['carried_lanes'] == 1
    assert result['originally_failing_lanes'] == ['x-cluster-bgmm']
    assert result['numeric_parts'] == 10 and not result['synthesized_raw_column']
    assert not result['release_qualified'] and initial == original


def test_missing_decode_reference_is_explicit_structural_only():
    initial, repair, table = inputs()
    initial['cells'][-1].update(state='N/A', value='n/a:no-decode-state')
    del table['cells']['ridge/base']['stepfull']
    result = admit(initial, repair, table, initial['lanes'], 'a' * 40)
    assert result['unreferenced_structural_parts'] == ['ridge/base/stepfull']
    assert result['numeric_parts'] == 9 and result['structural_na_parts'] == 1
    del table['cells']['ridge/base']['train']
    with pytest.raises(ValueError):
        admit(initial, repair, table, initial['lanes'], 'a' * 40)


@pytest.mark.parametrize('mutation', [
    lambda old, new, table: old['cells'].pop(),
    lambda old, new, table: old['cells'].append(old['cells'][0]),
    lambda old, new, table: old['execution'].update(completed_cells=1),
    lambda old, new, table: old['cells'][-1].update(state='REFUSED', error='unfixed'),
    lambda old, new, table: new['cells'][0].update(state='REFUSED'),
    lambda old, new, table: new['cells'][1].update(value='f' * 16, reference='f' * 16),
    lambda old, new, table: table['cells']['ridge/base']['train'].update(ref='f' * 16),
    lambda old, new, table: new['lanes'].append('unreviewed'),
])
def test_rejects_gaps_unfixed_failures_and_changed_passes(mutation):
    initial, repair, table = inputs()
    mutation(initial, repair, table)
    with pytest.raises(ValueError):
        admit(initial, repair, table, ['x-cluster-bgmm', 'ridge'], 'a' * 40)


def test_native_witness_must_match_packaged_module_and_bytes():
    report = {'bindings': [{'module': 'mojolearn._host._example', 'sha256': 'a' * 64}]}
    validate_bindings(report, {'mojolearn/host/_example.so': 'a' * 64})
    with pytest.raises(ValueError):
        validate_bindings(report, {'mojolearn/host/_example.so': 'b' * 64})
    with pytest.raises(ValueError):
        validate_bindings(report, {'mojolearn/host/_wrong.so': 'a' * 64})
    report['bindings'] *= 2
    with pytest.raises(ValueError):
        validate_bindings(report, {'mojolearn/host/_example.so': 'a' * 64})
