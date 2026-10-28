#!/usr/bin/env python3
"""Capture tiny identity fixtures or prove a named public-operation fault.

No builds, timing, benchmarking or background services. Run against an
installed package, with the candidate verifier module staged beside it.
"""
import argparse
import hashlib
import json
from pathlib import Path

from retain_run_summary import retain_verification, try_retain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--vendor', choices=('cpu', 'metal', 'cuda', 'hip'), required=True)
    parser.add_argument('--cases', help='comma-separated tiny cases; default all')
    parser.add_argument('--fault', help='run clean, injected and restored for this fault')
    parser.add_argument('--expect', type=Path)
    args = parser.parse_args()
    from mojolearn import _verify_light as verifier
    from light_verifier_faults import FAULTS, inject
    args.out.mkdir(parents=True, exist_ok=True)

    def capture(label, cases):
        report = verifier.capture(cases)
        if report['vendor'] != args.vendor:
            report.update(status='FAILED', error='unexpected vendor: ' + report['vendor'])
        path = args.out / (label + '.json')
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        return report

    if args.fault:
        if args.fault not in FAULTS or args.cases or args.expect:
            parser.error('a fault selects its own case and clean reference')
        cases = [FAULTS[args.fault][0]]
        clean = capture('clean', cases)
        with inject(args.fault):
            faulty = capture('faulty', cases)
        restored = capture('restored', cases)
        differences = verifier.compare(faulty, clean)
        restoration = verifier.compare(restored, clean)
        # A crash, refusal, missing output or bad provenance is NOT detection.
        caught = bool(differences) and all(e.startswith('bits differ:') for e in differences)
        passed = caught and not restoration and clean['vendor'] == args.vendor
        result = dict(fault=args.fault, kind='public-operation', native_kernel_fault=False,
                      status='CAUGHT_AND_RESTORED' if passed else 'FAILED',
                      differences=differences, restoration_errors=restoration,
                      records={name: hashlib.sha256((args.out/(name+'.json')).read_bytes()).hexdigest()
                               for name in ('clean', 'faulty', 'restored')})
    else:
        cases = args.cases.split(',') if args.cases else verifier.EXPECTED_CASES
        report = capture('capture', cases)
        errors = verifier.compare(report, report)
        if args.expect:
            errors += verifier.compare(report, json.loads(args.expect.read_text()))
        passed = not errors and report['vendor'] == args.vendor
        result = dict(status=('MATCH' if args.expect else 'CAPTURED_UNQUALIFIED') if passed else 'FAILED', errors=errors)
    (args.out/'result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    names = ('clean', 'faulty', 'restored') if args.fault else ('capture',)
    try_retain(retain_verification, 'light-' + args.vendor + ('-fault-' + args.fault if args.fault else ''),
               [args.out / (name + '.json') for name in (*names, 'result')],
               metadata=dict(vendor=args.vendor, cases=cases, fault=args.fault, exit_code=0 if passed else 1))
    print(result['status'])
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
