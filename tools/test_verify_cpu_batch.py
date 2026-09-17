import copy
import pytest
from verify_cpu_batch import evaluate_pair


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
