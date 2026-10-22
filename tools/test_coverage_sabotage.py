"""Fault evidence must establish detection and restoration, not merely difference."""
import copy
import hashlib
import json
import subprocess

import pytest

import coverage_sabotage as gate


@pytest.fixture
def trial(tmp_path):
    arm = dict(arm='coverage-test', patch='unit.patch', bindings=['build_core_host'], lanes=['one'])
    (tmp_path / 'unit.patch').write_text('a retained injected value fault\n')
    clean = dict(commit='a' * 40, vendor='cpu-test', platform='test', mode='identical',
                 repeats=1, complete=True, fixtures={'base': {'X': 'same'}, 'odd': {'X': 'odd'}},
                 package={'version': 'alpha', 'python': '3', 'numpy': '2', 'bindings': [
                     {'module': '_mojolearn_core_host', 'sha256': 'a' * 64, 'size': 10}]},
                 resume_signature={'source_sha256': 'c' * 64, 'environment_sha256': 'd' * 64,
                                   'options': {'lanes': 'one', 'fixtures': 'base,odd', 'repeats': 1}},
                 cells={f'one/{fx}': {'verdict': 'STABLE', 'hashes': ['abc'],
                                     'infer_verdict': 'STABLE', 'infer': ['answer']}
                        for fx in ('base', 'odd')})
    sab = copy.deepcopy(clean)
    sab['package']['bindings'][0]['sha256'] = 'b' * 64
    sab['resume_signature']['source_sha256'] = 'e' * 64
    sab['cells']['one/base']['hashes'] = ['def']
    restored = copy.deepcopy(clean)

    def run(evidence=None):
        paths = []
        for label, data in zip(('clean', 'sabotaged', 'restored'), (clean, sab, restored)):
            path = tmp_path / (label + '.json')
            path.write_text(json.dumps(data))
            paths.append(path)
        review = None
        if evidence is not None:
            evidence.update(arm=arm['arm'], patch_sha256=hashlib.sha256((tmp_path / arm['patch']).read_bytes()).hexdigest(),
                            records={key: hashlib.sha256(path.read_bytes()).hexdigest()
                                     for key, path in zip(('clean', 'sabotaged', 'restored'), paths)})
            review = tmp_path / 'review.json'
            review.write_text(json.dumps(evidence))
        return gate.check(arm, *paths, evidence_path=review, root=tmp_path)

    return arm, clean, sab, restored, run


def test_complete_changed_result_and_exact_restoration_pass(trial):
    assert trial[-1]() == 0


@pytest.mark.parametrize('failure', [
    'unstable-clean', 'wrong-commit', 'wrong-vendor', 'wrong-fixture', 'missing-fault-cell',
    'missing-clean-cell', 'missing-restored-cell', 'missing-part', 'incomplete',
    'restore-moved', 'restore-binary', 'unchanged-binary', 'unrelated-binary',
    'no-build-provenance', 'wrong-options', 'wrong-source-restoration', 'unstable-part',
    'missing-fixture-everywhere', 'empty-fault-hash', 'failed-probe',
])
def test_invalid_evidence_never_counts_as_a_catch(trial, failure):
    _, clean, sab, restored, run = trial
    if failure == 'unstable-clean': clean['cells']['one/base']['verdict'] = 'DIVERGENT'
    elif failure == 'wrong-commit': sab['commit'] = 'b' * 40
    elif failure == 'wrong-vendor': sab['vendor'] = 'other-device'
    elif failure == 'wrong-fixture': sab['fixtures']['base']['X'] = 'changed'
    elif failure == 'missing-fault-cell': del sab['cells']['one/odd']
    elif failure == 'missing-clean-cell': del clean['cells']['one/odd']
    elif failure == 'missing-restored-cell': del restored['cells']['one/odd']
    elif failure == 'missing-part': del sab['cells']['one/base']['infer']
    elif failure == 'incomplete': sab['complete'] = False
    elif failure == 'restore-moved': restored['cells']['one/base']['hashes'] = ['def']
    elif failure == 'restore-binary': restored['package']['bindings'][0]['sha256'] = 'b' * 64
    elif failure == 'unchanged-binary': sab['package'] = copy.deepcopy(clean['package'])
    elif failure == 'unrelated-binary':
        for r in (clean, sab, restored):
            r['package']['bindings'].append(dict(module='unrelated', size=10, sha256='f' * 64))
        sab['package']['bindings'][1]['sha256'] = 'e' * 64
    elif failure == 'no-build-provenance': clean['package']['bindings'] = []
    elif failure == 'wrong-options': sab['resume_signature']['options']['repeats'] = 2
    elif failure == 'wrong-source-restoration': restored['resume_signature']['source_sha256'] = 'f' * 64
    elif failure == 'unstable-part': clean['cells']['one/base']['infer_verdict'] = 'REFUSED'
    elif failure == 'missing-fixture-everywhere':
        for r in (clean, sab, restored):
            del r['fixtures']['odd']
            del r['cells']['one/odd']
    elif failure == 'empty-fault-hash': sab['cells']['one/base']['hashes'] = ['']
    elif failure == 'failed-probe': sab['cells']['one/base']['infer_verdict'] = 'REFUSED'
    assert run() == 1


@pytest.mark.parametrize('error', ['ImportError: missing extension', 'ModuleNotFoundError: numpy',
                                  'MemoryError: out of memory', 'ValueError: unnamed refusal'])
def test_unreviewed_or_infrastructure_refusal_is_not_detection(trial, error):
    _, _, sab, _, run = trial
    sab['cells']['one/base'] = dict(verdict='REFUSED', hashes=[], error=error)
    assert run() == 1
    if error.startswith('ValueError'):
        return
    assert run({'refusals': {'one/base': dict(error=error, product_check='kernel.validate',
                                             why_fault_reaches_check='reviewed')}}) == 1


def test_reviewed_product_refusal_requires_exact_record_bytes(trial):
    _, _, sab, _, run = trial
    error = 'ValueError: weights must have positive total'
    sab['cells']['one/base'] = dict(verdict='REFUSED', hashes=[], error=error)
    review = {'refusals': {'one/base': dict(error=error, product_check='weights.validate',
                                          why_fault_reaches_check='injected weight became zero')}}
    assert run(review) == 0
    review['refusals']['one/base']['error'] = 'different'
    assert run(review) == 1


def test_no_restore_record_is_incomplete(trial):
    arm = trial[0]
    assert gate.check(arm, 'absent-clean', 'absent-sab') == 1


def test_equivalence_reads_git_diff_and_rejects_runtime_changes(tmp_path, trial):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args], text=True).strip()
    git('init', '-q')
    git('config', 'user.name', 'Test')
    git('config', 'user.email', 'test@example.invalid')
    (tmp_path / 'runtime.mojo').write_text('original\n')
    git('add', 'runtime.mojo'); git('commit', '-qm', 'base')
    base = git('rev-parse', 'HEAD')
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs' / 'receipt.md').write_text('retained evidence\n')
    git('add', 'docs'); git('commit', '-qm', 'evidence')
    documentation = git('rev-parse', 'HEAD')
    _, clean, sab, restored, run = trial
    clean['commit'] = restored['commit'] = base
    sab['commit'] = documentation
    assert run({'source_equivalence': {'commits': [base, documentation], 'reason': 'documentation only'}}) == 0
    (tmp_path / 'runtime.mojo').write_text('changed\n')
    git('add', 'runtime.mojo'); git('commit', '-qm', 'runtime change')
    changed = git('rev-parse', 'HEAD')
    sab['commit'] = changed
    assert run({'source_equivalence': {'commits': [base, changed], 'reason': 'claimed equivalent'}}) == 1


def test_recorded_batch_movement_is_value_evidence_not_infrastructure_failure(trial):
    _, clean, sab, restored, run = trial
    for record in (clean, sab, restored):
        for cell in record['cells'].values():
            cell.update(batch=['abc'], batch_verdict='STABLE')
    sab['cells']['one/base'].update(batch=['BATCH_MOVED:row:0'], batch_verdict='BATCH_MOVED')
    assert run() == 0


def test_reviewed_independent_oracle_failure_needs_every_repeat(trial):
    _, _, sab, _, run = trial
    errors = ['distributed distances and the independent plain distances differ: 512 bytes']
    sab['cells']['one/base'] = dict(verdict='DIVERGENT', hashes=[], oracle_errors=errors)
    assert run() == 1
    review = {'oracles': {'one/base': dict(errors=errors, product_check='identity_break._same_bytes',
                                         why_fault_reaches_check='fault changes the distributed merge distances')}}
    assert run(review) == 0
    sab['cells']['one/base']['oracle_errors'] = []
    assert run(review) == 1


def test_base_build_script_maps_to_the_base_extension(trial):
    arm, clean, sab, restored, run = trial
    arm['bindings'] = ['build']
    for record in (clean, sab, restored):
        record['package']['bindings'][0]['module'] = '_mojolearn'
    assert run() == 0
