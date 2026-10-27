"""The tiny verifier must reject changed words and incomplete evidence."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import struct

import pytest

SPEC = importlib.util.spec_from_file_location('light_verifier_tested',
    Path(__file__).resolve().parents[1]/'python/mojolearn/_verify_light.py')
light = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(light)


def record():
    raw = struct.pack('<f', 0.25)
    output = dict(shape=[1], dtype='<f4', sha256=hashlib.sha256(raw).hexdigest(), bytes_hex=raw.hex())
    return dict(format=light.FORMAT, numeric_mode=1, status='CAPTURED',
                expected_cases=list(light.EXPECTED_CASES),
                digest_protocol='sha256-raw-shape-dtype-v1',
                case_source_protocol='python-tokens-v1',
                case_sources={n:'a'*64 for n in light.EXPECTED_CASES},
                bindings=[dict(sha256='b'*64)],
                cases={n:dict(status='captured',outputs=[dict(output) for _ in range(light.OUTPUT_COUNTS[n])])
                       for n in light.EXPECTED_CASES})


@pytest.mark.parametrize('case', light.EXPECTED_CASES)
def test_every_consumed_output_rejects_a_one_bit_fault(case):
    clean = record()
    assert not light.compare(clean, clean)
    for index in range(light.OUTPUT_COUNTS[case]):
        faulty = copy.deepcopy(clean)
        output = faulty['cases'][case]['outputs'][index]
        raw = bytes.fromhex(output['bytes_hex'])
        changed = bytes([raw[0] ^ 1]) + raw[1:]
        output.update(bytes_hex=changed.hex(), sha256=hashlib.sha256(changed).hexdigest())
        assert light.compare(faulty, clean) == ['bits differ: ' + case]


@pytest.mark.parametrize('fault', ['missing-case','missing-output','failed','mode','source','bindings','corrupt-bytes'])
def test_incomplete_or_inconsistent_records_never_match_themselves(fault):
    bad = record()
    first = light.EXPECTED_CASES[0]
    if fault == 'missing-case':
        del bad['cases'][first]
    elif fault == 'missing-output':
        bad['cases'][first]['outputs'].pop()
    elif fault == 'failed':
        bad['cases'][first]['status'] = 'failed'
    elif fault == 'mode':
        bad['numeric_mode'] = 0
    elif fault == 'source':
        bad['case_sources'] = {}
    elif fault == 'bindings':
        bad['bindings'] = []
    else:
        bad['cases'][first]['outputs'][0]['bytes_hex'] = 'bad'
    assert light.compare(bad, bad)


def test_source_changes_require_new_fixture_evidence_but_labels_do_not():
    original, renamed = record(), record()
    original['source_sha256'], renamed['source_sha256'] = 'x', 'y'
    assert not light.compare(original, renamed)
    renamed['case_sources'][light.EXPECTED_CASES[0]] = 'c'*64
    assert 'different numerical fixture source' in light.compare(original, renamed)


@pytest.mark.parametrize('field,value', [('shape', [1, 1]), ('dtype', '<i4')])
def test_metadata_change_is_not_a_caught_numerical_fault(field, value):
    clean, faulty = record(), record()
    case = light.EXPECTED_CASES[0]
    faulty['cases'][case]['outputs'][0][field] = value
    assert light.compare(faulty, clean) == ['output contract differs: ' + case]


def test_fingerprint_ignores_comments_but_catches_changed_arithmetic(monkeypatch):
    monkeypatch.setattr(light.inspect, 'getsource', lambda f: 'def f(x):\n    return x + 1\n')
    baseline = light.case_fingerprint(None)
    monkeypatch.setattr(light.inspect, 'getsource', lambda f: 'def f(x):\n    # explanation\n    return x+1\n')
    assert light.case_fingerprint(None) == baseline
    monkeypatch.setattr(light.inspect, 'getsource', lambda f: 'def f(x):\n    return x + 2\n')
    assert light.case_fingerprint(None) != baseline
