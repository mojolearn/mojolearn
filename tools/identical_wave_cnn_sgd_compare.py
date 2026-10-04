#!/usr/bin/env python3
"""Compare complete CNN/SGD gate receipts within one source/suite/arm."""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED = {
    'cnn-primitives': {'xent-33', 'xent-1024', 'xent-1025', 'epoch-1', 'epoch-2', 'epoch-3',
                       'epoch-1000', 'epoch-65537', 'adam-hyper-0', 'adam-hyper-1'},
    'cnn-training': {'cnn-sgd', 'cnn-adam'},
    'sgd': {'sgd-ovr-sgd', 'sgd-ovr-perceptron', 'sgd-ovr-pa', 'sgd-nan-refusal', 'sgd-overflow-refusal'},
}


def compare(paths, required):
    if not required or not set(required) <= {'cuda', 'hip', 'metal', 'cpu'}:
        raise ValueError('invalid required vendor inventory')
    baseline = None; vendors = set(); evidence = []
    for path in paths:
        raw = path.read_bytes(); report = json.loads(raw)
        core = {key: report[key] for key in ('sha', 'arm', 'suite')}
        if report.get('status') != 'PASS' or set(report['cases']) != EXPECTED[core['suite']]:
            raise ValueError('failed or incomplete gate receipt: '+str(path))
        if report.get('timing_samples') != 0 or report.get('opponents_executed') != 0:
            raise ValueError('receipt is not an untimed own-build gate')
        digests = {key: row['digest'] for key, row in report['cases'].items() if row['status'] == 'PASS'}
        if set(digests) != EXPECTED[core['suite']]: raise ValueError('case status failed')
        value = {'identity': core, 'digests': digests}
        if baseline is None: baseline = value
        elif value != baseline: raise ValueError('source/arm/suite/case digest mismatch: '+str(path))
        vendors.add(report['vendor'])
        evidence.append({'path': str(path), 'vendor': report['vendor'], 'sha256': hashlib.sha256(raw).hexdigest()})
    if not set(required) <= vendors: raise ValueError('required vendor receipts missing')
    return {'status': 'PASS', **baseline, 'required_vendors': sorted(required), 'observed_vendors': sorted(vendors), 'receipts': evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipts', nargs='+', type=Path, required=True)
    parser.add_argument('--require-vendors', default='cuda,hip,metal,cpu', help='Explicit subset is an interim check only')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try: result = compare(args.receipts, args.require_vendors.split(','))
    except (KeyError, ValueError, TypeError, OSError) as exc: result = {'status': 'FAIL', 'error': str(exc)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix+'.new')
    temporary.write_text(json.dumps(result, indent=2)+'\n'); temporary.replace(args.out)
    print('CNN_SGD_COMPARE', result['status'])
    return 0 if result['status'] == 'PASS' else 1

if __name__ == '__main__': raise SystemExit(main())
