"""Narrow opt-in continuation after a provenance-valid local algorithm failure."""
import re


class LocalObservationFailure(RuntimeError):
    def __init__(self, detail):
        self.detail = detail
        super().__init__(str(detail))


def validated_failure(report, receipt, report_sha, vendor, lane, validate=None):
    if validate is None:
        from identity_verify_launch import validate_run
        validate = validate_run
    validate(receipt, report, report_sha)
    if report.get('device', {}).get('device_class') != vendor or report.get('lanes') != [lane]:
        raise ValueError('local failure belongs to a different vendor or lane')
    if report.get('native_provenance_complete') is not True or report.get('self_test', {}).get('passed') is not True:
        raise ValueError('provenance or verifier self-test failure cannot be continued')
    if receipt.get('exit_code') not in (1, 4):
        raise ValueError('unexpected producer exit is not a local algorithm failure')
    crashes = report.get('execution', {}).get('crashed_cells', [])
    signals = []
    for crash in crashes:
        match = re.search(r'worker exited (-?[0-9]+);', str(crash.get('error', '')))
        if not match or int(match.group(1)) not in (-6, -10, -11):
            raise ValueError('unknown, OOM or terminated worker failure cannot be continued')
        signals.append(int(match.group(1)))
    property_failures = sum(cell.get('local_check') == 'failed' for cell in report.get('cells', []))
    if not crashes and not property_failures:
        raise ValueError('incomplete observation lacks a demonstrated local algorithm failure')
    return dict(lane=lane, vendor=vendor, report_sha256=report_sha, exit_code=receipt['exit_code'],
                worker_signals=signals, property_failures=property_failures,
                state='failed-pending-repair', numerical_mismatch_asserted=False,
                release_qualified=False)
