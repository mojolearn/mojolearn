#!/usr/bin/env python3
"""Audit parallel reference coverage and emit per-vendor collection requirements."""
import argparse
import json
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    """Read-only modules must not initialize mojolearn or load native bindings."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


vref = _load('parallel_audit_reference', ROOT / 'python/mojolearn/_verify_reference.py')
coverage = _load('parallel_audit_coverage', ROOT / 'python/mojolearn/_crossvendor_coverage.py')


def load_harness(**_kwargs):
    return _load('parallel_audit_harness', ROOT / 'tools/identity_break.py')


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
    contract = json.loads(args.contract.read_text()) if args.contract else coverage.parallel_contract(harness, reference=vref)
    report = coverage.audit(vref.load_table(args.reference_table), contract,
                   args.fixtures.split(',') if args.fixtures else harness.FIXTURES,
                   args.vendors.split(','),
                   lane_revisions=getattr(harness, 'LANE_REVISIONS', {}))
    report['reference_table_sha256'] = vref.sha256_file(args.reference_table)
    payload = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end='')
    return 5 if args.fail_on_incomplete and not report['complete'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
