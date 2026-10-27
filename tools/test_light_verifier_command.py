"""Reference admission and shipped command must fail closed, without a GPU."""
import copy

import pytest
from light_verifier_reference import load

command = load('_verify_light_command')
light = load('_verify_light')
reference = load('_verification_light_reference').DATA


def test_complete_three_vendor_reference_matches_current_fixture_sources():
    assert command.reference_errors(reference, light) == []


@pytest.mark.parametrize('fault', ['vendor', 'version', 'missing-vendor', 'partial', 'stale', 'missing-output'])
def test_reference_admission_rejects_incomplete_or_inconsistent_witnesses(fault):
    bad = copy.deepcopy(reference)
    row = bad['witnesses']['cuda']
    if fault == 'vendor':
        row['vendor'] = 'metal'
    elif fault == 'version':
        row['version'] = 'another-release'
    elif fault == 'missing-vendor':
        del bad['witnesses']['hip']
    elif fault == 'partial':
        bad['witnesses'] = {v: command.subset(r, [light.EXPECTED_CASES[0]])
                            for v, r in bad['witnesses'].items()}
    elif fault == 'stale':
        row['case_sources'][light.EXPECTED_CASES[0]] = 'e'*64
    else:
        row['cases'][light.EXPECTED_CASES[0]]['outputs'].pop()
    assert command.reference_errors(bad, light)


def test_each_tiny_case_has_a_public_operation_fault():
    faults = load('_verify_light_faults').FAULTS
    assert {case for case, _ in faults.values()} == set(light.EXPECTED_CASES)


def test_subset_preserves_only_requested_case_contract():
    report = command.subset(reference['witnesses']['metal'], ['tape_gradients'])
    assert not light.compare(report, report)
    assert set(report['cases']) == {'tape_gradients'}


def test_version_and_installation_path_alone_do_not_require_numerical_reruns():
    witness = reference['witnesses']['metal']
    repackaged = copy.deepcopy(witness)
    repackaged['version'] = 'next-release'
    for binding in repackaged['bindings']:
        binding['file'] = '/another/install/' + binding['module']
    assert command.unchanged_artifacts(repackaged, witness)
    repackaged['bindings'][0]['sha256'] = 'f'*64
    assert not command.unchanged_artifacts(repackaged, witness)
