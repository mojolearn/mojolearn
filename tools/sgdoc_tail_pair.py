#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Manager-only quality pair / quality-gated timing for the SGDOneClassSVM
tail replay (lane/apple-fast-w2-sgdoc), on verified prebuilt x_linear arms
(tools/target_scratch_pair.py's pattern).

quality SOURCE QUALITY_TAG
 timing SOURCE QUALITY_TAG TIMING_TAG taxi|istella
No builds, SSH, queue edits, or opponent runs. SOURCE is the exact full SHA.
Arms: ~/mq/verified-arms/SOURCE/x_linear/{A,B}.so + manifest.json, A = main
(no define), B = -D MOJOLEARN_SGDOC_FAST_TAIL or -D MOJOLEARN_SGDOC_FAST_TAIL_LONG.
The quality job writes ~/mq/out/QUALITY_TAG-quality/PASS.json only when
tools/sgdoc_tail_quality.py compare passes every gate; timing refuses without it.
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

DEFINES = ('-D MOJOLEARN_SGDOC_FAST_TAIL', '-D MOJOLEARN_SGDOC_FAST_TAIL_LONG')
FIXTURE = 'sgdoc-tail-v1-rows-small'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_logged(command, path, ok_codes=(0,)):
    with path.open('x') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode not in ok_codes:
        print('\n'.join(path.read_text(errors='replace').splitlines()[-12:]))
        raise RuntimeError('Quality command failed: ' + str(path))
    return result.returncode


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
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'x_linear/', 'bindings/', 'python/',
                    'tools/sgdoc_tail_quality.py', 'tools/sgdoc_tail_pair.py', 'tools/bench_board_algos.py'],
                   check=True)
    home = Path.home()
    arms = home / 'mq/verified-arms' / args.source / 'x_linear'
    manifest = json.loads((arms / 'manifest.json').read_text())
    assert manifest['source_sha'] == args.source and manifest['binding'] == 'x_linear'
    assert manifest['numeric_mode'] == 'fast'
    assert manifest['defines_A'] == ''
    assert manifest['defines_B'] in DEFINES
    define = manifest['defines_B'].removeprefix('-D ')
    hashes = {arm: digest(arms / (arm + '.so')) for arm in ('A', 'B')}
    assert hashes == manifest['hashes']
    out = home / 'mq/out' / (args.quality_tag + '-quality')
    receipt_path = out / 'PASS.json'
    receipt_want = dict(source_sha=args.source, hashes=hashes, status='PASS', fixture=FIXTURE, define=define)
    if args.action == 'timing':
        assert args.timing_tag and args.dataset
        assert json.loads(receipt_path.read_text()) == receipt_want
        os.execv(sys.executable, [sys.executable, str(home / 'mq/verified_arms.py'),
                 args.source, 'x_linear', define, args.timing_tag,
                 'bash', 'tools/afc_ab_def.sh', args.timing_tag, 'x_linear', 'sgd-ocsvm',
                 args.dataset, '1', '1', '', '-D ' + define])
    assert args.timing_tag is None and args.dataset is None
    out.mkdir(parents=True, exist_ok=False)  # never silently reuse partial captures
    for b in ('board-0834', 'board-0833'):
        vp = home / b / 'cache/venv/bin/python'
        if vp.exists():
            break
    os.environ.update(MOJOLEARN_NUMERIC_MODE='fast', MOJOLEARN_VENDOR='apple',
                      MOJOLEARN_BENCH_INSTALLED='0', PYTHONPATH=str(root / 'python'),
                      OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    installed = root / 'python/mojolearn/_mojolearn_x_linear.so'
    shutil.copy2(installed, out / 'original.so')
    try:
        for arm in ('A', 'B'):
            temporary = installed.with_suffix('.so.next')
            shutil.copy2(arms / (arm + '.so'), temporary)
            os.replace(temporary, installed)
            log = out / (arm + '.log')
            run_logged([str(vp), 'tools/sgdoc_tail_quality.py', 'dump', str(out / (arm + '.npz'))], log)
            records = [json.loads(line.split(' ', 1)[1]) for line in log.read_text().splitlines()
                       if line.startswith('SGDOC-TAIL-CAPTURE ')]
            assert len(records) == 1 and records[0]['binding_sha256'] == hashes[arm], records
            assert records[0]['fixture'] == FIXTURE
            print('SGDOC-TAIL-PAIR arm=' + arm + ' capture=PASS', flush=True)
        rc = run_logged([str(vp), 'tools/sgdoc_tail_quality.py', 'compare',
                         str(out / 'A.npz'), str(out / 'B.npz')], out / 'compare.log', ok_codes=(0, 1))
        text = (out / 'compare.log').read_text()
        print('\n'.join(l for l in text.splitlines() if l.startswith(('SGDOC-TAIL-GATE', 'SGDOC-TAIL-ROW', 'SGDOC-TAIL-AB'))))
        if rc == 0 and 'SGDOC-TAIL-AB status=PASS' in text:
            with receipt_path.open('x') as stream:
                json.dump(receipt_want, stream, sort_keys=True)
            print('SGDOC-TAIL-PAIR status=PASS source=' + args.source + ' define=' + define + ' receipt=' + str(receipt_path))
        else:
            print('SGDOC-TAIL-PAIR status=FAIL source=' + args.source + ' define=' + define)
            sys.exit(1)
    finally:
        temporary = installed.with_suffix('.so.restore')
        shutil.copy2(out / 'original.so', temporary)
        os.replace(temporary, installed)


if __name__ == '__main__':
    main()
