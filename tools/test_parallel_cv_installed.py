"""Installed CV entry point and complete receipt failure contracts; no GPU claims."""
import copy
import pytest
from mojolearn import _verify_parallel_cv as cv
from mojolearn._parallel_cv_witness import score_with_witness
from test_parallel_cv_witness import records


def receipt():
    return dict(protocol=cv.PROTOCOL, status='NUMERICS_AND_PLACEMENT_PASS', vendor='cuda',
                folds=5, repeats=2, devices=[0, 1],
                source={'source_sha256': {name: 'a' * 64 for name in cv.PROFILE_FILES}},
                bindings={'_mojolearn_gbdt': {'vendor': 'cuda', 'sha256': 'e' * 64}},
                inputs={name: 'f' * 64 for name in ('X', 'classifier', 'regressor')},
                runs=[dict(model=model, devices=order, repeat=repeat,
                           scores_hex='000000000000f03f' * 5, fold_records=records(len(order)))
                      for model in ('classifier', 'regressor')
                      for order in ([0], [0, 1], [1, 0]) for repeat in range(2)],
                controls=[model + ':' + fault for model in ('classifier', 'regressor') for fault in cv.FAULTS])


def test_shipped_scorer_and_complete_receipt():
    assert score_with_witness.__module__ == 'mojolearn._parallel_cv_witness'
    assert cv.validate_receipt(receipt())
    assert cv.compare(receipt(), receipt())


@pytest.mark.parametrize('fault', ['missing-run', 'duplicate-run', 'missing-control', 'bad-profile',
                                   'bad-input', 'bad-native', 'bad-replay', 'bad-scores', 'aliased'])
def test_complete_receipt_rejects_missing_or_changed_evidence(fault):
    report = receipt()
    if fault == 'missing-run': report['runs'].pop()
    elif fault == 'duplicate-run': report['runs'][-1] = copy.deepcopy(report['runs'][0])
    elif fault == 'missing-control': report['controls'].pop()
    elif fault == 'bad-profile': report['source']['source_sha256'].pop('model_selection.py')
    elif fault == 'bad-input': report['inputs'].pop('X')
    elif fault == 'bad-native': report['bindings']['_mojolearn_gbdt']['sha256'] = '0' * 64
    elif fault == 'bad-replay': next(iter(report['runs'][0]['fold_records'].values()))['reload_predict'] = '0' * 64
    elif fault == 'bad-scores': report['runs'][0]['scores_hex'] = '0' * 80
    else: report['runs'][2]['fold_records'] = records(1)
    with pytest.raises((ValueError, RuntimeError)):
        cv.validate_receipt(report)


def test_cross_vendor_comparison_ignores_native_but_compares_numerics():
    left, right = receipt(), receipt()
    right['vendor'] = 'hip'
    right['bindings']['_mojolearn_gbdt'] = {'vendor': 'hip', 'sha256': 'f' * 64}
    for run in right['runs']:
        for row in run['fold_records'].values():
            row['inventory']['vendor'] = 'hip'
            row['binding_sha256'] = 'f' * 64
    assert cv.compare(left, right)
    for run in right['runs']:
        for row in run['fold_records'].values(): row['model'] = '0' * 64
    assert not cv.compare(left, right)


def test_existing_output_preserved_before_backend_access(tmp_path):
    witness = tmp_path / 'keep.txt'
    witness.write_text('prior capture')
    with pytest.raises(SystemExit):
        cv.main(['--out', str(tmp_path), '--require-installed'])
    assert witness.read_text() == 'prior capture'
