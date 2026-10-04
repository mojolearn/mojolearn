#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Manager-only SVGP quality pair / quality-gated timing on verified A/B arms
(tools/target_scratch_pair.py's pattern, binding x_neighbors, lane w2-svgp).

quality SOURCE QUALITY_TAG
 timing SOURCE QUALITY_TAG TIMING_TAG taxi|istella
The quality job runs tools/svgp_fast_quality.py dump on both datasets with
each arm installed, then compare (the gate is in that file's docstring), and
writes PASS.json. The timing job refuses without a matching PASS.json, then
hands off to ~/mq/verified_arms.py + tools/afc_ab_def.sh (A = "", B = the
manifest's define). No builds, SSH, queue edits, or opponent runs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

DEFINES = ("MOJOLEARN_SVGP_FAST_BLKCHOL", "MOJOLEARN_SVGP_FAST_RBFTILE", "MOJOLEARN_SVGP_FAST_BSPLIT")
DATASETS = ("taxi", "istella")
FIXTURE = "svgp-board-v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_logged(command, path, check=True):
    with path.open('x') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode and check:
        print('\n'.join(path.read_text(errors='replace').splitlines()[-12:]))
        raise RuntimeError('command failed: ' + str(path))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('quality', 'timing'))
    parser.add_argument('source')
    parser.add_argument('quality_tag')
    parser.add_argument('timing_tag', nargs='?')
    parser.add_argument('dataset', nargs='?', choices=DATASETS)
    args = parser.parse_args()
    assert re.fullmatch('[0-9a-f]{40}', args.source)
    for tag in (args.quality_tag, args.timing_tag):
        assert tag is None or re.fullmatch('[A-Za-z0-9_.-]+', tag)
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == args.source
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'x_neighbors/', 'bindings/', 'python/',
                    'tools/svgp_fast_quality.py', 'tools/svgp_fast_pair.py'], check=True)
    home = Path.home()
    arms = home / 'mq/verified-arms' / args.source / 'x_neighbors'
    manifest = json.loads((arms / 'manifest.json').read_text())
    assert manifest['source_sha'] == args.source and manifest['binding'] == 'x_neighbors'
    assert manifest['numeric_mode'] == 'fast'
    assert manifest['defines_A'] == ''
    tokens = manifest['defines_B'].split()
    assert len(tokens) == 2 and tokens[0] == '-D' and tokens[1] in DEFINES, manifest['defines_B']
    define = tokens[1]
    hashes = {arm: digest(arms / (arm + '.so')) for arm in ('A', 'B')}
    assert hashes == manifest['hashes']
    out = home / 'mq/out' / (args.quality_tag + '-quality')
    receipt_path = out / 'PASS.json'
    receipt_want = dict(source_sha=args.source, hashes=hashes, status='PASS', fixture=FIXTURE,
                        datasets=list(DATASETS), define=define)
    if args.action == 'timing':
        assert args.timing_tag and args.dataset
        assert json.loads(receipt_path.read_text()) == receipt_want
        os.execv(sys.executable, [sys.executable, str(home / 'mq/verified_arms.py'),
                 args.source, 'x_neighbors', define, args.timing_tag,
                 'bash', 'tools/afc_ab_def.sh', args.timing_tag, 'x_neighbors', 'svgp',
                 args.dataset, '1', '1', '', '-D ' + define])
    assert args.timing_tag is None and args.dataset is None
    out.mkdir(parents=True, exist_ok=False)  # never silently reuse partial captures
    os.environ.update(MOJOLEARN_NUMERIC_MODE='fast', MOJOLEARN_VENDOR='apple',
                      MOJOLEARN_BENCH_INSTALLED='0', PYTHONPATH=str(root / 'python'),
                      OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    installed = root / 'python/mojolearn/_mojolearn_x_neighbors.so'
    shutil.copy2(installed, out / 'original.so')
    passed = True
    try:
        for arm in ('A', 'B'):
            temporary = installed.with_suffix('.so.next')
            shutil.copy2(arms / (arm + '.so'), temporary)
            os.replace(temporary, installed)
            for ds in DATASETS:
                log = out / (arm + '-' + ds + '.log')
                run_logged([sys.executable, 'tools/svgp_fast_quality.py', 'dump',
                            str(out / (arm + '-' + ds + '.npz')), '--dataset', ds], log)
                caps = [json.loads(line.split(' ', 1)[1]) for line in log.read_text().splitlines()
                        if line.startswith('SVGP-FAST-CAPTURE ')]
                assert len(caps) == 1 and caps[0]['binding_sha256'] == hashes[arm], caps
                print('SVGP-FAST-PAIR arm=%s ds=%s capture=PASS' % (arm, ds), flush=True)
        for ds in DATASETS:
            clog = out / ('compare-' + ds + '.log')
            rc = run_logged([sys.executable, 'tools/svgp_fast_quality.py', 'compare',
                             str(out / ('A-' + ds + '.npz')), str(out / ('B-' + ds + '.npz'))], clog, check=False)
            line = [x for x in clog.read_text().splitlines() if x.startswith('SVGP-FAST-AB ')]
            print('SVGP-FAST-PAIR ds=%s %s' % (ds, line[-1] if line else 'NO-RESULT rc=%d' % rc), flush=True)
            passed = passed and rc == 0 and bool(line) and 'status=PASS' in line[-1]
        if passed:
            with receipt_path.open('x') as stream:
                json.dump(receipt_want, stream, sort_keys=True)
        print('SVGP-FAST-PAIR status=%s define=%s source=%s' % ('PASS' if passed else 'FAIL', define, args.source))
    finally:
        temporary = installed.with_suffix('.so.restore')
        shutil.copy2(out / 'original.so', temporary)
        os.replace(temporary, installed)
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
