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
    # BOTH SIDES IN THIS PROCESS (2026-11-24): the CPU side (device="cpu")
    # and, with a GPU, the GPU side (device="gpu"). Each side's outputs are
    # held to the one admitted witness: identical bits on every column, the
    # host column included. The fault controls run on the last side.
    from . import runtime_report
    sides = ['cpu'] if runtime_report().get('cpu_only') else ['cpu', 'gpu']
    result['sides'] = {}
    for side in sides:
        clean = verifier.capture(side=side)
        if retain:
            retain('clean-' + side, clean)
        result['vendor'] = clean['vendor']
        changed = False
        if reference['witnesses']['metal']['version'] != __version__:
            witness = reference['witnesses'].get(clean['vendor'], {})
            changed = not unchanged_artifacts(clean, witness)
            result['evidence_reuse'] = ('changed native bytes reproduce the admitted witness' if changed
                                        else 'unchanged native bytes and numerical fixtures')
        errors = verifier.compare(clean, reference['witnesses']['metal'])
        result['sides'][side] = dict(vendor=clean['vendor'], errors=errors,
                                     status='MATCH' if not errors else 'OWED' if changed else 'FAILED')
        if errors:
            # Changed native bytes are judged by their outputs; only moved outputs owe new witnesses.
            result['errors'] = [f'{side} side: {e}' for e in errors]
            if changed:
                result.pop('evidence_reuse', None)
                result.update(status='OWED', errors=['changed native artifacts moved these outputs; they need current-release witnesses'] + result['errors'])
            return result
    side = sides[-1]
    if self_test:
        for name, (case, _) in FAULTS.items():
            baseline = subset(clean, [case])
            with inject(name):
                faulty = verifier.capture([case], artifacts=clean['bindings'], side=side)
            restored = verifier.capture([case], artifacts=clean['bindings'], side=side)
            if retain:
                retain(name + '-faulty', faulty)
                retain(name + '-restored', restored)
            differences = verifier.compare(faulty, baseline)
            restoration = verifier.compare(restored, baseline)
            caught = bool(differences) and all(e.startswith('bits differ:') for e in differences)
            passed = caught and not restoration
            result['faults'][name] = dict(status='CAUGHT_AND_RESTORED' if passed else 'FAILED',
                                          differences=differences, restoration_errors=restoration,
                                          kind='public-operation', native_kernel_fault=False, side=side)
            if not passed:
                result['errors'].append('fault not caught and restored: ' + name)
    result['status'] = 'MATCH' if not result['errors'] else 'FAILED'
    return result
