import copy
import pytest
from controller_local_failures import validated_failure


def report():
    return dict(device=dict(device_class='apple'), lanes=['arima'], native_provenance_complete=True,
                self_test=dict(passed=True), execution=dict(crashed_cells=[dict(error='cell/arima/base: worker exited -10; ')]), cells=[])


def check(doc, receipt=None, validate=None):
    return validated_failure(doc, receipt or {'exit_code': 4}, 'hash', 'apple', 'arima',
                             validate=validate or (lambda *args: None))


def test_known_native_worker_crash_is_recorded_as_failed_not_passed():
    result = check(report())
    assert result['worker_signals'] == [-10] and not result['release_qualified']
    assert result['state'] == 'failed-pending-repair' and not result['numerical_mismatch_asserted']


def test_property_failure_has_explicit_count():
    doc = report()
    doc['execution']['crashed_cells'] = []
    doc['cells'] = [{'local_check': 'failed'}]
    assert check(doc, {'exit_code': 1})['property_failures'] == 1


@pytest.mark.parametrize('change', ['vendor', 'lane', 'provenance', 'self-test', 'oom', 'terminated', 'unknown', 'missing'])
def test_infrastructure_and_provenance_failures_stop(change):
    doc = report()
    if change == 'vendor': doc['device']['device_class'] = 'cpu'
    if change == 'lane': doc['lanes'] = ['different']
    if change == 'provenance': doc['native_provenance_complete'] = False
    if change == 'self-test': doc['self_test']['passed'] = False
    if change == 'oom': doc['execution']['crashed_cells'][0]['error'] = 'worker exited -9; '
    if change == 'terminated': doc['execution']['crashed_cells'][0]['error'] = 'worker exited -15; '
    if change == 'unknown': doc['execution']['crashed_cells'][0]['error'] = 'transport failure'
    if change == 'missing': doc['execution']['crashed_cells'] = []
    with pytest.raises(ValueError): check(doc)


def test_invalid_original_producer_receipt_cannot_be_skipped():
    def invalid(*args): raise ValueError('installed manifest changed')
    with pytest.raises(ValueError, match='installed manifest changed'):
        check(report(), validate=invalid)


@pytest.mark.parametrize('code', [0, 124, -9])
def test_unexpected_exit_cannot_be_classified_as_local(code):
    with pytest.raises(ValueError, match='producer exit'):check(report(), {'exit_code': code})
