# SPDX-License-Identifier: Apache-2.0
"""Receipt comparison negative controls; these are not hardware records."""
import copy
import json
from pathlib import Path

import pytest

from parallel_cv_witness import compare_records, read_records


def records(workers=2):
    result = {}
    for fold in range(5):
        worker = fold % workers
        key = f'{fold + 1:064x}'
        result[key] = dict(fixture=key, model='a' * 64, predict='b' * 64,
                           raw_predict='c' * 64, reload_predict='c' * 64,
                           loss_curve='d' * 64, score='000000000000f03f', binding_sha256='e' * 64,
                           inventory=dict(kind='visible-device-inventory', vendor='cuda', pid=101 + worker,
                                          devices=[dict(ordinal=0, uuid=f'{worker + 1:032x}',
                                                        pci_bus_id=f'0000:{worker + 1:02x}:00.0')]))
    return result


def test_one_vs_two_worker_records_keep_every_numerical_part():
    compare_records(records(1), records(2), vendor='cuda', workers=2)


@pytest.mark.parametrize('part', ['model', 'predict', 'raw_predict', 'reload_predict', 'loss_curve', 'score'])
@pytest.mark.parametrize('mutation', ['missing', 'changed'])
def test_each_numerical_part_is_required(part, mutation):
    expected, actual = records(1), records(2)
    row = next(iter(actual.values()))
    if mutation == 'missing': row.pop(part)
    else: row[part] = 'f' * 64
    with pytest.raises(ValueError, match=part):
        compare_records(expected, actual, vendor='cuda', workers=2)


@pytest.mark.parametrize('fault', ['dropped', 'other-fixture', 'same-gpu', 'unused-worker',
                                  'binding', 'changed-worker-identity'])
def test_missing_work_aliased_hardware_and_changed_artifacts_fail(fault):
    expected, actual = records(1), records(2)
    keys = list(actual)
    if fault == 'dropped': actual.pop(keys[-1])
    elif fault == 'other-fixture': actual['f' * 64] = actual.pop(keys[-1])
    elif fault == 'same-gpu':
        for row in actual.values():
            row['inventory']['devices'] = copy.deepcopy(actual[keys[0]]['inventory']['devices'])
    elif fault == 'unused-worker': actual = records(1)
    elif fault == 'binding': actual[keys[-1]]['binding_sha256'] = '0' * 64
    else: actual[keys[-1]]['inventory']['devices'][0]['uuid'] = 'f' * 32
    with pytest.raises((ValueError, RuntimeError)):
        compare_records(expected, actual, vendor='cuda', workers=2)


def test_reader_refuses_missing_and_duplicate_folds(tmp_path):
    for i, row in enumerate(records().values()):
        (tmp_path / f'{i}.json').write_text(json.dumps(row))
    assert len(read_records(tmp_path, 5)) == 5
    with pytest.raises(ValueError, match='missing or duplicate'):
        read_records(tmp_path, 6)
    (tmp_path / '4.json').write_text((tmp_path / '0.json').read_text())
    with pytest.raises(ValueError, match='missing or duplicate'):
        read_records(tmp_path, 5)


@pytest.mark.parametrize('fault', [None, 'reload', 'vendor', 'mode', 'duplicate'])
def test_scorer_retains_model_and_output_witnesses_or_fails(tmp_path, monkeypatch, fault):
    import hashlib
    from types import SimpleNamespace
    import numpy as np
    import mojolearn as ml
    from mojolearn import _backend, _gpu_witness
    from parallel_cv_witness import score_with_witness, array_digest

    values = np.asarray([.25, -.5], dtype='<f4')
    binding_path = tmp_path / 'fake.so'
    binding_path.write_bytes(b'fake binding for an isolated unit test')
    binding = SimpleNamespace(__file__=str(binding_path),
                              gbdt_vendor=lambda: 'cpu' if fault == 'vendor' else 'cuda',
                              gbdt_numeric_mode=lambda: 0 if fault == 'mode' else 1)
    learner = SimpleNamespace(model_='synthetic model fixture', loss_curve_=values,
                              predict=lambda X: values, save=lambda path: Path(path).write_bytes(b'fixture'))
    restored = SimpleNamespace(predict=lambda X: values + np.float32(1) if fault == 'reload' else values)
    estimator = SimpleNamespace(_learner_=learner, predict=lambda X: values, score=lambda X, y: .5)
    monkeypatch.setattr(_backend, 'vendor', lambda: 'cuda')
    monkeypatch.setattr(_backend, 'binding', lambda *args: binding)
    monkeypatch.setattr(_backend, 'gpu_arch', lambda: 'simulated')
    monkeypatch.setattr(_backend, 'gpu_arch_how', lambda: 'unit test')
    monkeypatch.setattr(_gpu_witness, 'visible_gpu_inventory', lambda v: next(iter(records(1).values()))['inventory'])
    monkeypatch.setattr(ml, 'GradientBoosting', SimpleNamespace(load=lambda path: restored))
    X = values.reshape(2, 1)
    if fault in ('reload', 'vendor', 'mode'):
        with pytest.raises(RuntimeError):
            score_with_witness(estimator, X, values, directory=tmp_path)
        assert not list(tmp_path.glob('*.json'))
    else:
        assert score_with_witness(estimator, X, values, directory=tmp_path) == .5
        result = next(iter(read_records(tmp_path, 1).values()))
        assert result['predict'] == array_digest(values)
        assert result['raw_predict'] == result['reload_predict']
        assert result['model'] == hashlib.sha256(learner.model_.encode()).hexdigest()
        assert result['binding_sha256'] == hashlib.sha256(binding_path.read_bytes()).hexdigest()
        if fault == 'duplicate':
            with pytest.raises(RuntimeError, match='overwrite'):
                score_with_witness(estimator, X, values, directory=tmp_path)
