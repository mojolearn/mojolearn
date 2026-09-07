#!/usr/bin/env python3
"""File-only admission of one vendor's bounded real-text training run.

No numerical libraries, model execution, or cross-vendor certification.
Root runs this after fetching the complete guarded campaign artifacts.
"""
import argparse
from pathlib import Path
import re

from byte_lm_state_compare import (
    COUNTS, PROFILE, array_read, load_capture, oracle, parse, read, receipt, require, sha,
)
from byte_lm_gradient_oracle import TOLERANCES

JOBS = ('byte-venv', 'byte-dependencies', 'dependencies', 'dependency-freeze',
        'guard-checks', 'comparator-fixtures', 'byte-build', 'retain-binding',
        'byte-host-mocks', 'byte-step1', 'byte-gradient-oracle', 'byte-full128')


def admit(root):
    root = Path(root)
    require(read(root / 'exit_code').strip() == b'0', 'campaign did not finish successfully')
    rows = [line.split('\t') for line in read(root / 'results.tsv').decode().splitlines()]
    require(rows == [[name, '0'] for name in JOBS], 'missing, duplicate, reordered or failed job')
    leg = dict(line.split('=', 1) for line in read(root.parent / 'leg.txt').decode().splitlines()
               if '=' in line)
    vendor = {'nvidia': 'cuda', 'amd': 'hip'}.get(leg.get('vendor'))
    require(vendor is not None, 'remote vendor missing')
    provenance = read(root / 'provenance.txt').decode().splitlines()
    require('vendor=' + vendor in provenance and 'source=' + leg.get('commit', '') in provenance,
            'campaign vendor/source differs from transported leg')
    for name in JOBS:
        log = read(root / (name + '.log'), limit=8 * 1024 * 1024)
        require(log.strip(), 'empty job log: ' + name)
        terminal = parse(log.rstrip().split(b'\n')[-1])
        cores = str(terminal.get('cpu_affinity', '')).split(',')
        require(terminal.get('guard') == ('nvidia-root-serial-v1' if vendor == 'cuda' else 'amd-root-serial-v1')
                and terminal.get('reason') is None and type(terminal.get('returncode')) is int
                and terminal['returncode'] == 0 and terminal.get('thread_limit') == 2
                and 1 <= len(cores) <= 2 and all(core.isdigit() for core in cores),
                'job missing successful bounded guard exit: ' + name)
        require(read(root / (name + '.command.txt')).strip(), 'missing retained job command')
    for name in ('comparator-fixtures', 'byte-host-mocks'):
        log = read(root / (name + '.log')).decode()
        require(re.search(r'\b[1-9][0-9]* passed\b', log) is not None,
                'no executed passing tests: ' + name)
        require(re.search(r'\b[1-9][0-9]* (?:skipped|failed|errors?|xfailed|xpassed)\b', log) is None,
                'incomplete test qualification: ' + name)
    capture = load_capture(root / 'full128', 'continuous', vendor)
    frozen = parse(read(root.parent / 'source_inventory.json'))['files']
    require(all(frozen.get(name) == value for name, value in capture['source'].items()),
            'training sources differ from frozen archive')
    binding = sha(read(root / 'bindings/_mojolearn_byte_lm.so'))
    require(binding == capture['runtime']['binding_sha256'], 'retained binding mismatch')
    full_receipt = receipt(root / 'byte-full128.receipt.json', capture['summary_sha256'], vendor)
    first = root / 'step1'
    first_summary_raw = read(first / 'summary.json')
    first_summary = parse(first_summary_raw)
    first_runtime = parse(read(first / 'runtime.json', limit=65536))
    require(first_summary['schema'] == 'mojolearn.byte-lm.real-text-capture.v1'
            and first_summary['completed_steps'] == first_summary['expected_steps'] == 1
            and first_summary['action'] == 'continuous'
            and first_summary['runtime'] == first_runtime
            and first_runtime['binding_sha256'] == binding
            and first_runtime['native_vendor'] == vendor and first_runtime['native_profile'] == PROFILE
            and first_runtime['native_numeric_mode'] == 1
            and first_runtime['source_sha256'] == capture['runtime']['source_sha256']
            and parse(read(first / 'source.json')) == capture['source']
            and first_summary['schedule'] == capture['summary']['schedule'],
            'independent one-step capture profile mismatch')
    first_receipt = receipt(root / 'byte-step1.receipt.json', sha(first_summary_raw), vendor)
    # The independent oracle ran on a separate process's first step. Require
    # every byte to match the first step of the actual 128-step training run.
    a, b = first / 'step000001', root / 'full128/step000001'
    first_manifest = parse(read(a / 'capture.json'))
    require(read(a / 'capture.json') == read(b / 'capture.json'), 'first-step manifests differ')
    require(len(first_summary['records']) == 1
            and first_summary['records'][0]['step'] == 1
            and first_summary['records'][0]['capture'] == 'step000001'
            and first_summary['records'][0]['capture_sha256'] == sha(read(a / 'capture.json')),
            'first summary does not bind its actual step')
    for key in COUNTS:
        require(array_read(a, key, first_manifest['arrays'][key]) ==
                array_read(b, key, first_manifest['arrays'][key]), 'first-step bytes differ: ' + key)
    report = parse(read(root / 'gradient-oracle.json'))
    require(report['tolerances'] == {key: dict(atol=value[0], rtol=value[1])
                                     for key, value in TOLERANCES.items()},
            'independent reference tolerances changed')
    independent = oracle(root / 'gradient-oracle.json', capture,
                         root / 'byte-gradient-oracle.receipt.json')
    learning = capture['summary']['learning']
    require(learning['eligible'] is True and learning['observed_gate_passed'] is True
            and capture['ratio'] <= .9, 'predeclared heldout learning gate failed')
    return dict(schema='mojolearn.byte-lm.single-vendor-admission.v1', passed=True,
                vendor=vendor, source_commit=leg['commit'], binding_sha256=binding,
                completed_steps=128, heldout_loss_ratio=capture['ratio'],
                receipts=dict(first=first_receipt, full=full_receipt), independent_oracle=independent,
                scope='fixed small real-text model: independent first-step FP64 correctness and '
                      '128-step heldout learning on this vendor; cross-vendor identity remains separate')


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', type=Path)
    args = parser.parse_args()
    print(json.dumps(admit(args.artifacts), sort_keys=True, indent=2, allow_nan=False))
