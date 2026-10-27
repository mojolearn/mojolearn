"""Command controller for tiny identity checks (testing only)."""
import copy
import json
from pathlib import Path


def reference_errors(reference, verifier):
    """Admission is complete agreement from three actual GPU captures."""
    captures = reference.get('witnesses', {})
    if set(captures) != {'metal', 'cuda', 'hip'}:
        return ['reference needs Apple, NVIDIA and AMD witnesses']
    errors = []
    canonical = captures['metal']
    for vendor, report in captures.items():
        if report.get('vendor') != vendor or report.get('expected_cases') != list(verifier.EXPECTED_CASES):
            errors.append('incorrect vendor or incomplete scope: ' + vendor)
        if not report.get('version') or report.get('version') != canonical.get('version'):
            errors.append('reference witnesses must belong to one release')
        errors.extend(verifier.compare(report, canonical))
        current = {n: verifier.case_fingerprint(getattr(verifier, n)) for n in verifier.EXPECTED_CASES}
        if report.get('case_sources') != current:
            errors.append('reference fixture source is stale: ' + vendor)
    return errors


def subset(report, names):
    report = copy.deepcopy(report)
    report['expected_cases'] = list(names)
    report['cases'] = {n: report['cases'][n] for n in names}
    report['case_sources'] = {n: report['case_sources'][n] for n in names}
    return report


def unchanged_artifacts(actual, witness):
    # Release labels and installation paths are not numerical inputs. Reuse
    # accepted evidence when the loaded native bytes and fixtures are unchanged.
    # Changed artifacts need new witnesses, never preservation of old bits.
    def binaries(report):
        return {b.get('module'): b.get('sha256') for b in report.get('bindings', ())}
    return (bool(binaries(actual)) and binaries(actual) == binaries(witness)
            and actual.get('case_sources') == witness.get('case_sources'))


def verify(reference, *, self_test=False, retain=None):
    from . import _verify_light as verifier, __version__
    from ._verify_light_faults import FAULTS, inject
    errors = reference_errors(reference, verifier)
    result = dict(format='mojolearn.light-verification.v1', status='OWED' if errors else 'FAILED',
                  scope=list(verifier.EXPECTED_CASES), release_qualified=False,
                  errors=errors, faults={},
                  limitations=['Only the listed tiny fixtures; not whole-library qualification.',
                               'Faults alter public operations that execute native kernels, not kernel source.'])
    if errors:
        return result
    clean = verifier.capture()
    if retain:
        retain('clean', clean)
    result['vendor'] = clean['vendor']
    if reference['witnesses']['metal']['version'] != __version__:
        witness = reference['witnesses'].get(clean['vendor'], {})
        if not unchanged_artifacts(clean, witness):
            result.update(status='OWED', errors=['changed native artifacts need current-release witnesses; prior-release bits are not a contract'])
            return result
        result['evidence_reuse'] = 'unchanged native bytes and numerical fixtures'
    result['errors'] = verifier.compare(clean, reference['witnesses']['metal'])
    if result['errors']:
        return result
    if self_test:
        for name, (case, _) in FAULTS.items():
            baseline = subset(clean, [case])
            with inject(name):
                faulty = verifier.capture([case], artifacts=clean['bindings'])
            restored = verifier.capture([case], artifacts=clean['bindings'])
            if retain:
                retain(name + '-faulty', faulty)
                retain(name + '-restored', restored)
            differences = verifier.compare(faulty, baseline)
            restoration = verifier.compare(restored, baseline)
            caught = bool(differences) and all(e.startswith('bits differ:') for e in differences)
            passed = caught and not restoration
            result['faults'][name] = dict(status='CAUGHT_AND_RESTORED' if passed else 'FAILED',
                                          differences=differences, restoration_errors=restoration,
                                          kind='public-operation', native_kernel_fault=False)
            if not passed:
                result['errors'].append('fault not caught and restored: ' + name)
    result['status'] = 'MATCH' if not result['errors'] else 'FAILED'
    return result
