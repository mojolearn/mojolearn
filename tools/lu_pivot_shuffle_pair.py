#!/usr/bin/env python3
"""Manager-only M3 verified-arm pair; quality SOURCE TAG or timing SOURCE QTAG TTAG.

No builds, transfers, queue edits or opponent runs. Exclusive artifacts and
SHA/hash-matched quality receipts prevent accidental scored repeats.
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

DEFINE = 'MOJOLEARN_LU_FAST_PIVOT_SHUFFLE'
BIND = 'x_decomp'
ROOT = Path(__file__).resolve().parents[1]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(command, log):
    with log.open('x') as stream:
        code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode
    if code:
        print('\n'.join(log.read_text(errors='replace').splitlines()[-8:]))
        raise RuntimeError(f'Worker failed rc={code}: {log}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('quality', 'timing'))
    p.add_argument('source')
    p.add_argument('quality_tag')
    p.add_argument('timing_tag', nargs='?')
    args = p.parse_args()
    assert re.fullmatch('[0-9a-f]{40}', args.source)
    for tag in (args.quality_tag, args.timing_tag):
        assert tag is None or re.fullmatch('[A-Za-z0-9_.-]+', tag)
    assert (args.action == 'timing') == (args.timing_tag is not None)
    os.chdir(ROOT)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == args.source
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'x_decomp/', 'bindings/',
                    'gemm/', 'python/', 'tools/lu_fast_mma_quality.py', 'tools/lu_pivot_shuffle.py', 'tools/lu_pivot_shuffle_pair.py'], check=True)
    arms = Path.home() / 'mq/verified-arms' / args.source / BIND
    manifest = json.loads((arms / 'manifest.json').read_text())
    assert manifest['source_sha'] == args.source and manifest['binding'] == BIND
    assert manifest['numeric_mode'] == 'fast'
    assert manifest['defines_A'] == '' and manifest['defines_B'] == '-D ' + DEFINE
    hashes = {arm: digest(arms / (arm + '.so')) for arm in ('A', 'B')}
    assert hashes == manifest['hashes']
    expect = dict(source_sha=args.source, hashes=hashes, define=DEFINE,
                  fixture='lu-pivot-shuffle-v1', status='PASS')
    base = Path.home() / 'mq/out'
    quality = base / (args.quality_tag + '-quality')
    if args.action == 'timing':
        assert json.loads((quality / 'PASS.json').read_text()) == expect
    tag = args.quality_tag if args.action == 'quality' else args.timing_tag
    out = base / (tag + '-' + args.action)
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(MOJOLEARN_VENDOR='apple', MOJOLEARN_NUMERIC_MODE='fast',
                      MOJOLEARN_BENCH_INSTALLED='0', PYTHONPATH=str(ROOT / 'python'),
                      OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    installed = ROOT / 'python/mojolearn/_mojolearn_x_decomp.so'
    shutil.copy2(installed, out / 'original.so')
    try:
        for arm in ('A', 'B'):
            temporary = installed.with_suffix('.so.next')
            shutil.copy2(arms / (arm + '.so'), temporary)
            os.replace(temporary, installed)
            run([sys.executable, 'tools/lu_pivot_shuffle.py', args.action, str(out / arm)], out / (arm + '.log'))
            metrics = json.loads((out / arm / 'metrics.json').read_text())
            assert metrics['binding_sha256'] == hashes[arm]
            assert metrics['reach'] == (1 if arm == 'B' else 0)
            print(f'LU-PIVOT-SHUFFLE-PAIR arm={arm} action={args.action} output={out / arm}', flush=True)
        if args.action == 'quality':
            run([sys.executable, 'tools/lu_pivot_shuffle.py', 'compare', str(out / 'A'), str(out / 'B')], out / 'compare.log')
            with (out / 'PASS.json').open('x') as stream:
                json.dump(expect, stream, sort_keys=True)
        print(f'LU-PIVOT-SHUFFLE-PAIR action={args.action} status=PASS source={args.source}', flush=True)
    finally:
        temporary = installed.with_suffix('.so.restore')
        shutil.copy2(out / 'original.so', temporary)
        os.replace(temporary, installed)


if __name__ == '__main__':
    main()
