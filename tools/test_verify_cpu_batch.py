import copy
from pathlib import Path
import pytest
from verify_cpu_batch import arm_environment, evaluate_pair, expected_oracle_failure


def test_direct_loaders_use_the_selected_arm_and_clean_clears_sabotage_permissions():
    original = {'MOJOLEARN_FOREST_HOST_BINARY': '/old/forest.so',
                'MOJOLEARN_BYTE_LM_HOST_BINARY': '/old/byte.so',
                'MOJOLEARN_HOST_ALLOW_SABOTAGE': '1',
                'MOJOLEARN_FOREST_HOST_ALLOW_SABOTAGE': '1',
                'MOJOLEARN_BYTE_LM_HOST_ALLOW_SABOTAGE': '1'}
    for sabotage in (False, True):
        env = arm_environment(original, Path('/selected'), sabotage)
        assert env['MOJOLEARN_HOST_DIR'] == '/selected'
        for family in ('FOREST', 'BYTE_LM'):
            assert env[f'MOJOLEARN_{family}_HOST_BINARY'] == f'/selected/_mojolearn_{family.lower()}_host.so'
        for key in original:
            if key.endswith('ALLOW_SABOTAGE'):
                assert (env.get(key) == '1') == sabotage
    assert original['MOJOLEARN_FOREST_HOST_BINARY'] == '/old/forest.so'


def records():
    clean = dict(commit='a' * 40, mode='identical', complete=True,
                 fixtures={'base': {'X': 'input'}}, heldout={'base': {'X': 'held'}},
                 cells={'x/base': {'hashes': ['a' * 16] * 2, 'verdict': 'STABLE'}},
                 host={'families': {'native': {'sabotage': False}}})
    bad = copy.deepcopy(clean)
    bad['cells']['x/base']['hashes'] = ['b' * 16] * 2
    bad['host']['families']['native']['sabotage'] = True
    return clean, bad


def test_stable_changed_native_result_passes():
    assert evaluate_pair(*records(), 'x', ['base'])['passed']


def test_only_repeated_explicit_oracle_failures_allow_nonzero_sabotage_exit():
    clean, bad = records()
    cell = bad['cells']['x/base']
    cell.update(verdict='DIVERGENT', oracle_errors=['wrong result'] * 2)
    assert expected_oracle_failure(bad)
    assert evaluate_pair(clean, bad, 'x', ['base'])['passed']
    cell['oracle_errors'].pop()
    assert not expected_oracle_failure(bad)
    cell['oracle_errors'].append('wrong result')
    cell['hashes'][1] = 'c' * 16
    assert not expected_oracle_failure(bad)
    cell['verdict'] = 'REFUSED'
    assert not expected_oracle_failure(bad)


@pytest.mark.parametrize('part', ['infer', 'model', 'batch', 'stepfull', 'batchgrad', 'rlpair'])
def test_clean_property_refusal_blocks_training_control_success(part):
    clean, bad = records()
    clean['cells']['x/base'][part + '_verdict'] = 'REFUSED'
    result = evaluate_pair(clean, bad, 'x', ['base'])
    assert result['cells'][0]['detected']
    assert not result['passed']
    assert result['clean_failures'] == [dict(cell='x/base', part=part + '_verdict', verdict='REFUSED')]


@pytest.mark.parametrize('failure', ['same', 'unstable', 'refused', 'missing-fixture', 'input', 'incomplete', 'not-native'])
def test_invalid_native_control_never_passes(failure):
    clean, bad = records()
    fixtures = ['base']
    if failure == 'same':
        bad['cells'] = copy.deepcopy(clean['cells'])
    elif failure == 'unstable':
        bad['cells']['x/base']['hashes'][1] = 'c' * 16
    elif failure == 'refused':
        bad['cells']['x/base'] = {'hashes': ['error'] * 2, 'verdict': 'REFUSED'}
    elif failure == 'missing-fixture':
        fixtures.append('wide')
    elif failure == 'input':
        bad['fixtures']['base']['X'] = 'other'
    elif failure == 'incomplete':
        bad['complete'] = False
    elif failure == 'not-native':
        bad['host']['families']['native']['sabotage'] = False
    assert not evaluate_pair(clean, bad, 'x', fixtures)['passed']
