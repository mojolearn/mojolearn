import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('par_gmm_recipe', Path(__file__).with_name('run.py'))
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


def candidate():
    return dict(lane_revisions={'par-gmm': 'classic-kmeanspp-init-1'},
                records=[{'class': 'amd'}, {'class': 'nvidia'}], cells={
                    'par-gmm/' + f: {p: dict(ref=recipe.NA.get(p, 'a' * 16), cols=dict(amd=0, nvidia=1))
                                    for p in recipe.PARTS} for f in recipe.FIXTURES})


def test_candidate_requires_all_fixture_properties_and_independent_classes():
    table = candidate()
    recipe.strict_candidate(table)
    for change in ('missing', 'single_vendor', 'conflict', 'numeric_na'):
        bad = copy.deepcopy(table)
        cell = bad['cells']['par-gmm/odd']
        if change == 'missing':
            del cell['batchgrad']
        elif change == 'single_vendor':
            del cell['train']['cols']['amd']
        elif change == 'conflict':
            cell['train']['cols']['amd'] = [0, 'b' * 16]
        else:
            cell['train']['ref'] = 'n/a:unavailable'
        with pytest.raises(ValueError):
            recipe.strict_candidate(bad)


def physical():
    pool = dict(devices=[0, 1], cooperative=True, workers=[dict(devices=['GPU-a', 'GPU-b'])])
    witness = dict(placement='physical', devices=[0, 1], detail=[pool])
    return dict(state='VERIFIED', passed=True, vendor='cuda', devices=[0, 1], fixtures=['base'],
                witness=witness, cell_witnesses=[dict(fixture='base', witness=witness)],
                cells=[dict(verdict='IDENTICAL')])


def test_physical_witness_cannot_be_reused_or_alias_one_gpu():
    result = physical()
    recipe.strict_physical(result, 'base')
    for change in ('other_fixture', 'same_gpu', 'process_only', 'missing'):
        bad = copy.deepcopy(result)
        if change == 'other_fixture':
            bad['cell_witnesses'][0]['fixture'] = 'odd'
        elif change == 'same_gpu':
            bad['witness']['detail'][0]['workers'][0]['devices'] = ['GPU-a', 'GPU-a']
        elif change == 'process_only':
            bad['witness']['placement'] = 'process-only'
        else:
            bad['cell_witnesses'] = []
        with pytest.raises(ValueError):
            recipe.strict_physical(bad, 'base')
