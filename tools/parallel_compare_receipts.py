#!/usr/bin/env python3
"""Compare cloud receipts on the cloud host; does not execute a training model."""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--left', type=Path, required=True)
    p.add_argument('--right', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('Run this comparison on the RunPod host')
    checks = []
    failures = []
    for lane in ('byte-lm', 'mlp', 'forest', 'samba', 'samba-attention', 'kmeans'):
        left = json.loads((args.left / (lane + '.json')).read_text())
        right = json.loads((args.right / (lane + '.json')).read_text())
        keys = ('gradient_sha256', 'state_sha256') if lane == 'byte-lm' else ('hashes',)
        if lane in ('mlp', 'samba', 'samba-attention'):
            keys += ('state_hashes',)
        if left['status'] != 'PASS' or right['status'] != 'PASS':
            failures.append(lane + ': gate did not pass on both hosts')
        for key in keys:
            if not left.get(key) or left[key] != right.get(key):
                failures.append(lane + ': ' + key + ' mismatch')
            else:
                checks.append(lane + ':' + key)
        if left.get('corpus_sha256') != right.get('corpus_sha256'):
            failures.append(lane + ': corpus mismatch')
    left_source = (args.left / 'source.sha256').read_text()
    right_source = (args.right / 'source.sha256').read_text()
    if left_source != right_source:
        failures.append('source manifests differ')
    else:
        checks.append('source_manifest')
    report = dict(status='FAIL' if failures else 'PASS', checks=checks, failures=failures,
                  scope='Two RTX 4090s versus two H100s; tested fixtures only. Not AMD/Apple qualification.')
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(args.report.read_text())
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
