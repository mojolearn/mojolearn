#!/usr/bin/env python3
"""Audit parallel reference coverage and emit per-vendor collection requirements."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from mojolearn._crossvendor_coverage import audit, parallel_contract
from mojolearn import _verify_reference as vref
from mojolearn._verify_all import load_harness


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-table', type=Path, default=Path(vref.table_path()))
    parser.add_argument('--contract', type=Path, help='independent JSON lane -> part -> numeric/recorded/n/a declaration')
    parser.add_argument('--fixtures', help='comma-separated fixture override; default all harness fixtures')
    parser.add_argument('--vendors', default='amd,apple,nvidia')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--fail-on-incomplete', action='store_true')
    args = parser.parse_args(argv)
    harness = load_harness(par_axis=True)
    contract = json.loads(args.contract.read_text()) if args.contract else parallel_contract(harness)
    report = audit(vref.load_table(args.reference_table), contract,
                   args.fixtures.split(',') if args.fixtures else harness.FIXTURES,
                   args.vendors.split(','))
    report['reference_table_sha256'] = vref.sha256_file(args.reference_table)
    payload = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end='')
    return 5 if args.fail_on_incomplete and not report['complete'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
