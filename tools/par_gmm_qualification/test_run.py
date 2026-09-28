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
    return dict(state='VERIFIED', passed=True, vendor='cuda', devices=[0, 1], fixtures=['base'], lanes=['par-gmm'],
                witness=witness, cell_witnesses=[dict(lane='par-gmm', fixture='base', witness=witness)],
                cells=[dict(lane='par-gmm', fixture='base', part=p, verdict='N/A' if p in recipe.NA else 'IDENTICAL')
                       for p in recipe.PHYSICAL_PARTS])


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


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'wrong_lane', 'report_lane', 'witness_lane'])
def test_physical_requires_exact_part_scope(change):
    report = physical()
    if change == 'missing':
        report['cells'].pop()
    elif change == 'duplicate':
        report['cells'][-1] = report['cells'][0].copy()
    elif change == 'wrong_lane':
        report['cells'][0]['lane'] = 'gmm'
    elif change == 'report_lane':
        report['lanes'] = ['gmm']
    else:
        report['cell_witnesses'][0]['lane'] = 'gmm'
    with pytest.raises(ValueError):
        recipe.strict_physical(report, 'base')


def test_timeout_kills_residual_group_after_leader_exits(tmp_path, monkeypatch):
    class Child:
        pid = 123
        calls = 0
        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise recipe.subprocess.TimeoutExpired('fixture', timeout)
            return -15
    child = Child()
    signals = []
    monkeypatch.setattr(recipe.subprocess, 'Popen', lambda *a, **kw: child)
    monkeypatch.setattr(recipe.os, 'killpg', lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(recipe.subprocess.TimeoutExpired):
        recipe.bounded(['fixture'], tmp_path, {}, tmp_path / 'fixture.log')
    assert signals == [(123, recipe.signal.SIGTERM), (123, recipe.signal.SIGKILL)]


def test_manifest_preserves_binding_bytes_and_source_closure(tmp_path):
    package = tmp_path / 'python/mojolearn/identical'
    package.mkdir(parents=True)
    (package / '_mojolearn_mixture.so').write_bytes(b'native-fixture')
    stamps = tmp_path / 'python/.binding-stamps'
    stamps.mkdir()
    (stamps / 'identical___mojolearn_mixture.so.json').write_text('{"digest":"source-closure"}')
    row = recipe.native_manifest(tmp_path)['bindings']['identical/_mojolearn_mixture.so']
    assert row['sha256'] == recipe.hashlib.sha256(b'native-fixture').hexdigest()
    assert row['source_closure_stamp'] == {'digest': 'source-closure'}
