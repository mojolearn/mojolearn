#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Manager-only quality pair / quality-gated timing for MOJOLEARN_EIGH_FAST_TRIDIAG
(lane w4-eigh), on verified prebuilt x_decomp arms (tools/target_scratch_pair.py
pattern).

quality SOURCE QUALITY_TAG
 timing SOURCE QUALITY_TAG TIMING_TAG
No builds, SSH, queue edits, or opponent runs. SOURCE is the exact full SHA.
Timing refuses unless QUALITY_TAG wrote a matching PASS.json.
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

BIND = 'x_decomp'
DEFINE = 'MOJOLEARN_EIGH_FAST_TRIDIAG'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_logged(command, path):
    with path.open('x') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print('\n'.join(path.read_text(errors='replace').splitlines()[-12:]))
        raise RuntimeError('Quality command failed: ' + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('quality', 'timing'))
    parser.add_argument('source')
    parser.add_argument('quality_tag')
    parser.add_argument('timing_tag', nargs='?')
    args = parser.parse_args()
    assert re.fullmatch('[0-9a-f]{40}', args.source)
    for tag in (args.quality_tag, args.timing_tag):
        assert tag is None or re.fullmatch('[A-Za-z0-9_.-]+', tag)
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == args.source
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'x_decomp/', 'bindings/', 'python/',
                    'tools/eigh_w4_quality.py', 'tools/eigh_w4_pair.py'], check=True)
    home = Path.home()
    arms = home / 'mq/verified-arms' / args.source / BIND
    manifest = json.loads((arms / 'manifest.json').read_text())
    assert manifest['source_sha'] == args.source and manifest['binding'] == BIND
    assert manifest['numeric_mode'] == 'fast'
    assert manifest['defines_A'] == ''
    assert manifest['defines_B'] == '-D ' + DEFINE
    hashes = {arm: digest(arms / (arm + '.so')) for arm in ('A', 'B')}
    assert hashes == manifest['hashes']
    out = home / 'mq/out' / (args.quality_tag + '-quality')
    receipt_path = out / 'PASS.json'
    expect = dict(source_sha=args.source, hashes=hashes, status='PASS', fixture='eigh-w4-v1', define=DEFINE)
    if args.action == 'timing':
        assert args.timing_tag
        assert json.loads(receipt_path.read_text()) == expect, 'SKIP: no matching quality PASS'
        os.execv(sys.executable, [sys.executable, str(home / 'mq/verified_arms.py'),
                 args.source, BIND, DEFINE, args.timing_tag,
                 'bash', 'tools/afc_ab_def.sh', args.timing_tag, BIND, 'eigh',
                 'synthetic', '1', '1', '', '-D ' + DEFINE])
    assert args.timing_tag is None
    out.mkdir(parents=True, exist_ok=False)  # never silently reuse partial captures
    os.environ.update(MOJOLEARN_NUMERIC_MODE='fast', MOJOLEARN_VENDOR='apple',
                      MOJOLEARN_BENCH_INSTALLED='0', PYTHONPATH=str(root / 'python'),
                      OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    installed = root / 'python/mojolearn/_mojolearn_x_decomp.so'
    shutil.copy2(installed, out / 'original.so')
    try:
        for arm in ('A', 'B'):
            temporary = installed.with_suffix('.so.next')
            shutil.copy2(arms / (arm + '.so'), temporary)
            os.replace(temporary, installed)
            run_logged([sys.executable, 'tools/eigh_w4_quality.py', 'dump', str(out / (arm + '.json'))],
                       out / (arm + '.log'))
            got = json.loads((out / (arm + '.json')).read_text())
            assert got['binding_sha256'] == hashes[arm]
            print('EIGH-W4-PAIR arm=' + arm + ' capture=PASS', flush=True)
        run_logged([sys.executable, 'tools/eigh_w4_quality.py', 'compare',
                    str(out / 'A.json'), str(out / 'B.json')], out / 'compare.log')
        assert 'EIGH-W4-AB status=PASS' in (out / 'compare.log').read_text()
        with receipt_path.open('x') as stream:
            json.dump(expect, stream, sort_keys=True)
        print('EIGH-W4-PAIR status=PASS source=' + args.source + ' receipt=' + str(receipt_path))
    finally:
        temporary = installed.with_suffix('.so.restore')
        shutil.copy2(out / 'original.so', temporary)
        os.replace(temporary, installed)


if __name__ == '__main__':
    main()
