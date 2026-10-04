#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Manager-only quality pair / quality-gated timing using verified A/B arms.

quality SOURCE QUALITY_TAG
 timing SOURCE QUALITY_TAG TIMING_TAG taxi|istella
No builds, SSH, queue edits, or opponent runs. SOURCE is the exact full SHA.
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
    parser.add_argument('dataset', nargs='?', choices=('taxi', 'istella'))
    args = parser.parse_args()
    assert re.fullmatch('[0-9a-f]{40}', args.source)
    for tag in (args.quality_tag, args.timing_tag):
        assert tag is None or re.fullmatch('[A-Za-z0-9_.-]+', tag)
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == args.source
    # Refuse uncommitted source/checker changes that a source SHA cannot attest.
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'x_prep/', 'bindings/', 'python/',
                    'tools/target_scratch_quality.py', 'tools/target_scratch_pair.py'], check=True)
    home = Path.home()
    arms = home / 'mq/verified-arms' / args.source / 'x_prep'
    manifest = json.loads((arms / 'manifest.json').read_text())
    assert manifest['source_sha'] == args.source and manifest['binding'] == 'x_prep'
    assert manifest['numeric_mode'] == 'fast'
    assert manifest['defines_A'] == ''
    assert manifest['defines_B'] in ('-D MOJOLEARN_TARGET_SCRATCH', '-D MOJOLEARN_TARGET_SCRATCH_OFF')
    define = manifest['defines_B'].removeprefix('-D ')
    default_on = define.endswith('_OFF')
    scratch_expected = {'A': default_on, 'B': not default_on}
    hashes = {arm: digest(arms / (arm + '.so')) for arm in ('A', 'B')}
    assert hashes == manifest['hashes']
    out = home / 'mq/out' / (args.quality_tag + '-quality')
    receipt_path = out / 'PASS.json'
    if args.action == 'timing':
        assert args.timing_tag and args.dataset
        receipt = json.loads(receipt_path.read_text())
        assert receipt == dict(source_sha=args.source, hashes=hashes, status='PASS',
                               fixture='target-scratch-v1-seed553', arrays=108, define=define)
        os.execv(sys.executable, [sys.executable, str(home / 'mq/verified_arms.py'),
                 args.source, 'x_prep', define, args.timing_tag,
                 'bash', 'tools/afc_ab_def.sh', args.timing_tag, 'x_prep', 'target-encoder',
                 args.dataset, '1', '1', '', '-D ' + define])
    assert args.timing_tag is None and args.dataset is None
    out.mkdir(parents=True, exist_ok=False)  # never silently reuse partial captures
    os.environ.update(MOJOLEARN_NUMERIC_MODE='fast', MOJOLEARN_VENDOR='apple',
                      MOJOLEARN_BENCH_INSTALLED='0', PYTHONPATH=str(root / 'python'),
                      OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    installed = root / 'python/mojolearn/_mojolearn_x_prep.so'
    shutil.copy2(installed, out / 'original.so')
    try:
        for arm in ('A', 'B'):
            temporary = installed.with_suffix('.so.next')
            shutil.copy2(arms / (arm + '.so'), temporary)
            os.replace(temporary, installed)
            log = out / (arm + '.log')
            run_logged([sys.executable, 'tools/target_scratch_quality.py', 'dump', str(out / (arm + '.npz'))], log)
            records = [json.loads(line.split(' ', 1)[1]) for line in log.read_text().splitlines()
                       if line.startswith('TARGET-SCRATCH-CAPTURE ')]
            assert len(records) == 1
            assert records[0] == dict(binding_sha256=hashes[arm], scratch_enabled=scratch_expected[arm],
                                      fixture='target-scratch-v1-seed553', arrays=108)
            print('TARGET-SCRATCH-PAIR arm=' + arm + ' capture=PASS', flush=True)
        run_logged([sys.executable, 'tools/target_scratch_quality.py', 'compare',
                    str(out / 'A.npz'), str(out / 'B.npz')], out / 'compare.log')
        assert 'TARGET-SCRATCH-AB status=PASS exact_arrays=108' in (out / 'compare.log').read_text()
        with receipt_path.open('x') as stream:
            json.dump(dict(source_sha=args.source, hashes=hashes, status='PASS',
                           fixture='target-scratch-v1-seed553', arrays=108, define=define), stream, sort_keys=True)
        print('TARGET-SCRATCH-PAIR status=PASS source=' + args.source + ' receipt=' + str(receipt_path))
    finally:
        temporary = installed.with_suffix('.so.restore')
        shutil.copy2(out / 'original.so', temporary)
        os.replace(temporary, installed)


if __name__ == '__main__':
    main()
