"""Small-profile inputs and evidence admission; these tests perform no fits."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from mojolearn import _verify_all as suite, _verify_small as small


@pytest.fixture(scope='module')
def harness():
    return suite.load_harness()


def test_all_cases_are_small_and_repeatable_with_distinct_heldout(harness):
    for name, kind, rows, columns in small.CASES:
        data = small.fixture(harness, name)
        again = small.fixture(harness, name)
        heldout = small.fixture(harness, name, heldout=True)
        assert rows <= 257
        assert data[0].shape == heldout[0].shape == (rows, columns)
        assert data[0].dtype == np.float32
        assert all(a.tobytes() == b.tobytes() for a, b in zip(data, again))
        assert data[0].tobytes() != heldout[0].tobytes()
        assert data[0].nbytes <= 257 * 17 * 4
    assert {n for _, _, n, _ in small.CASES} >= {31, 32, 33, 63, 64, 65, 255, 256, 257}
    assert 'odd' not in {kind for _, kind, *_ in small.CASES}
    with pytest.raises(ValueError, match='unknown'):
        small.fixture(harness, 'full-base')


def test_small_boundary_pathologies_are_real(harness):
    duplicate = small.fixture(harness, 'dupes')[0]
    # Odd length leaves one tail row; the first 32 rows repeat exactly.
    assert np.array_equal(duplicate[32:64], duplicate[:32])
    assert np.all(duplicate[:, -2] == np.float32(3.5))
    assert np.all(duplicate[:, -1] == 0)
    ties = small.fixture(harness, 'ties')[0]
    assert len(np.unique(ties)) <= 6
    denormal = small.fixture(harness, 'denormal')
    flushed = small.fixture(harness, 'denormal_ftz')
    subnormal = (denormal[0] != 0) & (np.abs(denormal[0]) < np.finfo(np.float32).tiny)
    assert np.any(subnormal)
    assert np.all(flushed[0][subnormal] == 0)
    assert np.array_equal(np.signbit(denormal[0][subnormal]), np.signbit(flushed[0][subnormal]))
    assert denormal[1].tobytes() == flushed[1].tobytes()
    assert denormal[2].tobytes() == flushed[2].tobytes()


def capture(harness):
    return dict(format='mojolearn.small-training-capture.v1', complete=True,
        status='CAPTURED_UNQUALIFIED', repeats=2, contract=small.contract(harness),
        device=dict(vendor='cpu', numeric_mode='identical'),
        bindings=[dict(module='synthetic-test-binding', sha256='a' * 64, size=1)],
        inputs={case: dict(X='a' * 16, y_clf='b' * 16, y_reg='c' * 16, heldout='d' * 16)
                for case, *_ in small.CASES},
        cells={f'{lane}/{case}': {part: ['a' * 16, 'a' * 16]
               for part in ('train', 'infer', 'model', 'batch')}
               for lane in small.LANES for case, *_ in small.CASES})


def test_numerical_agreement_does_not_admit_reference_or_invent_gpu(harness):
    left = capture(harness)
    right = copy.deepcopy(left)
    result = small.compare_captures(left, right)
    assert result['status'] == 'NUMERICAL_MATCH_UNQUALIFIED'
    assert result['compared_parts'] == 180
    assert not result['reference_admitted'] and not result['independent_backends']
    right['device']['vendor'] = 'cuda'
    assert small.compare_captures(left, right)['independent_backends']
    right['cells']['ridge/rows-31']['infer'] = ['b' * 16] * 2
    assert small.compare_captures(left, right)['differences'] == ['ridge/rows-31/infer']


@pytest.mark.parametrize('mutation', ['missing-cell', 'missing-repeat', 'unstable',
    'refused', 'na', 'error', 'incomplete', 'no-bindings', 'no-inputs', 'fast', 'big-shape'])
def test_incomplete_or_failed_evidence_is_never_a_match(harness, mutation):
    left = capture(harness)
    right = copy.deepcopy(left)
    cell = right['cells']['ridge/rows-31']
    if mutation == 'missing-cell':
        del right['cells']['ridge/rows-31']
    elif mutation == 'missing-repeat':
        cell['train'].pop()
    elif mutation == 'unstable':
        cell['train'][1] = 'b' * 16
    elif mutation in ('refused', 'na'):
        cell['infer'] = [None, None] if mutation == 'refused' else ['n/a:function'] * 2
    elif mutation == 'error':
        cell['errors'] = ['native failure']
    elif mutation == 'incomplete':
        right['complete'] = False
    elif mutation == 'no-bindings':
        right['bindings'] = []
    elif mutation == 'no-inputs':
        right['inputs'] = {}
    elif mutation == 'fast':
        right['device']['numeric_mode'] = 'fast'
    else:
        right['contract']['cases'][0]['rows'] = 20000
    with pytest.raises(ValueError):
        small.compare_captures(left, right)


@pytest.mark.parametrize('field', ['inputs', 'contract'])
def test_mismatched_inputs_or_source_cannot_be_compared(harness, field):
    left = capture(harness)
    right = copy.deepcopy(left)
    if field == 'inputs':
        right[field]['rows-31']['X'] = 'b' * 16
    else:
        right[field]['harness_sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='different profile'):
        small.compare_captures(left, right)


def test_retained_native_pilot_matches_and_catches_estimator_fault():
    root = Path(__file__).resolve().parents[3]
    evidence = root / 'bench/results/small_training/2026-09-18-apple-m4/provenance-v2'
    if not evidence.is_dir():
        pytest.skip('source capture evidence is not installed in wheels')
    read = lambda name: json.loads((evidence / name).read_text())
    cpu = read('cpu.json')
    assert cpu['capture_commit'] == cpu['device']['commit']
    for name, independent in [('cpu-replay.json', False), ('metal.json', True)]:
        assert read(name)['capture_commit'] == cpu['capture_commit']
        result = small.compare_captures(cpu, read(name))
        assert not result['differences']
        assert result['compared_parts'] == 180
        assert result['independent_backends'] is independent
        assert not result['reference_admitted']
    fault = read('cpu-ridge-sabotage.json')
    assert fault['host']['families']['_mojolearn_estimators_host']['sabotage'] is True
    result = small.compare_captures(cpu, fault)
    expected = {f'ridge/{case}/{part}' for case, *_ in small.CASES
                for part in ('train', 'infer', 'model', 'batch')}
    assert set(result['differences']) == expected
