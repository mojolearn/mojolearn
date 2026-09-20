"""Coverage is driven by the current contract, never by table keys alone."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mojolearn._crossvendor_coverage import audit, parallel_contract
from mojolearn import _verify_reference as vref

HASH = '0123456789abcdef'
CONTRACT = {'par-demo': {'train': 'numeric'}}


def table(value=HASH):
    return {'cells': {'par-demo/base': {'train': {'ref': value,
            'cols': {'amd': 0, 'apple': 1, 'nvidia': 2}}}}}


def test_complete_and_deterministic():
    t = table()
    assert audit(t, CONTRACT, ['base'])['complete']
    assert audit(t, CONTRACT, ['base'], ['nvidia', 'apple', 'amd']) == audit(t, CONTRACT, ['base'])


@pytest.mark.parametrize('reason,mutate', [
    ('missing_vendor', lambda t: t['cells']['par-demo/base']['train']['cols'].pop('amd')),
    ('missing_part', lambda t: t['cells']['par-demo/base'].pop('train')),
    ('missing_cell', lambda t: t['cells'].clear()),
    ('stale_na', lambda t: t['cells']['par-demo/base']['train']['cols'].update(amd=[0, 'n/a:no-save'])),
    ('value_mismatch', lambda t: t['cells']['par-demo/base']['train']['cols'].update(amd=[0, 'fedcba9876543210'])),
    ('table_conflict', lambda t: t['cells']['par-demo/base']['train'].update(conflict=True, ref=None)),
])
def test_mutated_table_gaps_are_named_and_planned(reason, mutate):
    t = table()
    mutate(t)
    before = copy.deepcopy(t)
    result = audit(t, CONTRACT, ['base'])
    assert not result['complete'] and t == before
    assert result['gaps'][0]['reasons']['amd'] == reason
    assert result['collection_plan']['amd']['par-demo']['base'][0]['reason'] == reason


def test_new_part_and_fixture_cannot_disappear_when_all_records_omit_them():
    contract = {'par-demo': {'train': 'numeric', 'future': 'numeric'}}
    result = audit(table(), contract, ['base', 'wide'])
    assert result['counts']['expected_parts'] == 4
    assert result['counts']['incomplete_parts'] == 3
    assert result['gaps'][0]['part'] == 'future'
    assert result['gaps'][0]['reasons'] == dict.fromkeys(('amd', 'apple', 'nvidia'), 'missing_part')


def test_cpu_numeric_reference_rejects_unanimous_gpu_na():
    t = table()
    ent = t['cells']['par-demo/base']['train']
    ent['cols'] = {v: [0, 'n/a:no-save'] for v in ('amd', 'apple', 'nvidia')}
    ent['cols']['cpu'] = 3
    result = audit(t, {'par-demo': {'train': 'recorded'}}, ['base'])
    assert result['gaps'][0]['reasons'] == dict.fromkeys(('amd', 'apple', 'nvidia'), 'stale_na')


def test_static_na_needs_no_duplicate_vendor_recording():
    t = table('n/a:transductive')
    contract = {'par-demo': {'train': 'n/a:transductive (current detailed explanation)'}}
    result = audit(t, contract, ['base'])
    assert result['complete'] and result['counts']['not_applicable_parts'] == 1
    t['cells']['par-demo/base']['train']['cols'].pop('amd')
    result = audit(t, contract, ['base'])
    assert result['complete']
    assert result['counts']['not_applicable_parts'] == 1
    assert result['numeric_missing_values_by_vendor']['amd'] == 0


def test_numeric_requirement_overrides_all_old_na_and_future_declared_parts_exist():
    harness = SimpleNamespace(LANES={'par-demo': lambda: None}, BATCH={}, RLPAIR={},
        EXTRA_PARTS={'future': ({'par-demo': lambda: None}, 'n/a:absent')},
        REQUIRED_NUMERIC_PARTS={'par-demo': ('model',)})
    contract = parallel_contract(harness)
    assert contract['par-demo']['model'] == contract['par-demo']['future'] == 'numeric'
    result = audit(table('n/a:no-save'), contract, ['base'])
    assert not result['complete']
    assert any(g['part'] == 'model' and g['kind'] == 'numeric' for g in result['gaps'])


def test_cli_strict_exit_and_saved_plan(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[3] / 'tools/audit_parallel_coverage.py'
    spec = importlib.util.spec_from_file_location('audit_parallel_cli', path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, 'load_harness', lambda **kw: SimpleNamespace(FIXTURES=['base']))
    reference, contract, output = (tmp_path / n for n in ('table.json', 'contract.json', 'out.json'))
    reference.write_text(json.dumps(dict(format=vref.FORMAT, records=[], fixtures={}, heldout={}, cells={})))
    contract.write_text(json.dumps(CONTRACT))
    args = ['--reference-table', str(reference), '--contract', str(contract), '--output', str(output)]
    assert cli.main(args) == 0
    assert cli.main(args + ['--fail-on-incomplete']) == 5
    assert json.loads(output.read_text())['gaps'][0]['reasons']['amd'] == 'missing_cell'


def test_core_na_needs_evidence_but_not_duplicate_vendor_declarations():
    t = table('n/a:no-save')
    t['cells']['par-demo/base']['train']['cols'] = {'cpu': 0}
    contract = {'par-demo': {'train': 'recorded'}}
    assert audit(t, contract, ['base'])['complete']
    t['cells'].clear()
    result = audit(t, contract, ['base'])
    assert not result['complete']
    assert result['gaps'][0]['kind'] == 'undeclared_value'


@pytest.mark.parametrize('value', ['n/a:skipped', 'n/a:UNDECLARED'])
def test_skipped_run_is_not_an_applicability_declaration(value):
    result = audit(table(value), {'par-demo': {'train': 'recorded'}}, ['base'])
    assert not result['complete']
    assert set(result['gaps'][0]['reasons'].values()) == {'invalid_value'}


def test_statically_inapplicable_part_cannot_hide_numeric_contradiction():
    result = audit(table(), {'par-demo': {'train': 'n/a:no-backward'}}, ['base'])
    assert not result['complete']
    assert set(result['gaps'][0]['reasons'].values()) == {'contract_mismatch'}
