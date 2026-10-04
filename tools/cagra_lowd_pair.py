#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Manager-only CAGRA quality pair / quality-gated timing on verified A/B arms
(lane/apple-fast-w2-cagra; pattern of tools/target_scratch_pair.py).

quality SOURCE QUALITY_TAG
 timing SOURCE QUALITY_TAG TIMING_TAG taxi|istella

Arms: ~/mq/verified-arms/SOURCE/x_ann/{A,B}.so + manifest.json, A = main
(defines ''), B = '-D MOJOLEARN_CAGRA_FAST_IVFG_LOWD_SEEDS4_OFF' (LOWD +
SEEDS4 is default since its promotion; before it B was '-D ..._IVFG_LOWD' or
'-D ..._IVFG_LOWD_SEEDS4'). The quality job dumps both arms
on taxi and istella (tools/cagra_lowd_quality.py) and writes PASS.json only
when B's recall@10 >= A's on both (and, for IVFG_LOWD, istella's graph and ids
are byte-identical). The timing job refuses without that receipt. No builds,
SSH, queue edits, or opponent runs. SOURCE is the exact full SHA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

# LOWD_SEEDS4 is default since its promotion: A ('') is LOWD + SEEDS4, and
# its _OFF form builds the old arm. The recall gate (B >= A) was written for
# the pre-promotion direction.
DEFINES = ('-D MOJOLEARN_CAGRA_FAST_IVFG_LOWD_SEEDS4_OFF',)
DATASETS = ('taxi', 'istella')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_logged(command, path, check=True):
    with path.open('x') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode and check:
        print('\n'.join(path.read_text(errors='replace').splitlines()[-12:]))
        raise RuntimeError('Quality command failed: ' + str(path))
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
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'x_ann/', 'bindings/', 'python/',
                    'tools/cagra_lowd_quality.py', 'tools/cagra_lowd_pair.py'], check=True)
    home = Path.home()
    arms = home / 'mq/verified-arms' / args.source / 'x_ann'
    manifest = json.loads((arms / 'manifest.json').read_text())
    assert manifest['source_sha'] == args.source and manifest['binding'] == 'x_ann'
    assert manifest['numeric_mode'] == 'fast'
    assert manifest['defines_A'] == ''
    assert manifest['defines_B'] in DEFINES
    define = manifest['defines_B'].removeprefix('-D ')
    istella_identical = False  # every arm vs the LOWD_SEEDS4 default moves the search
    hashes = {arm: digest(arms / (arm + '.so')) for arm in ('A', 'B')}
    assert hashes == manifest['hashes']
    out = home / 'mq/out' / (args.quality_tag + '-quality')
    receipt_path = out / 'PASS.json'
    expect = dict(source_sha=args.source, hashes=hashes, status='PASS', fixture='cagra-board-ann-v1',
                  datasets=list(DATASETS), define=define)
    if args.action == 'timing':
        assert args.timing_tag and args.dataset
        receipt = json.loads(receipt_path.read_text())
        recalls = receipt.pop('recall_at_10')
        assert receipt == expect, receipt
        print('CAGRA-LOWD-PAIR receipt recall_at_10=' + json.dumps(recalls, sort_keys=True), flush=True)
        os.execv(sys.executable, [sys.executable, str(home / 'mq/verified_arms.py'),
                 args.source, 'x_ann', define, args.timing_tag,
                 'bash', 'tools/afc_ab_def.sh', args.timing_tag, 'x_ann', 'cagra',
                 args.dataset, '1', '1', '', '-D ' + define])
    assert args.timing_tag is None and args.dataset is None
    out.mkdir(parents=True, exist_ok=False)  # never silently reuse partial captures
    os.environ.update(MOJOLEARN_NUMERIC_MODE='fast', MOJOLEARN_VENDOR='apple',
                      MOJOLEARN_BENCH_INSTALLED='0', PYTHONPATH=str(root / 'python'),
                      OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    installed = root / 'python/mojolearn/_mojolearn_x_ann.so'
    shutil.copy2(installed, out / 'original.so')
    recalls = {}
    ok = True
    try:
        for arm in ('A', 'B'):
            temporary = installed.with_suffix('.so.next')
            shutil.copy2(arms / (arm + '.so'), temporary)
            os.replace(temporary, installed)
            for ds in DATASETS:
                log = out / (arm + '-' + ds + '.log')
                run_logged([sys.executable, 'tools/cagra_lowd_quality.py', 'dump', ds,
                            str(out / (arm + '-' + ds + '.npz'))], log)
                records = [json.loads(line.split(' ', 1)[1]) for line in log.read_text().splitlines()
                           if line.startswith('CAGRA-LOWD-CAPTURE ')]
                assert len(records) == 1 and records[0]['binding_sha256'] == hashes[arm]
                recalls.setdefault(ds, {})[arm] = records[0]['recall_at_10']
                print('CAGRA-LOWD-PAIR arm=%s ds=%s recall_at_10=%.6f graph=%s' % (
                    arm, ds, records[0]['recall_at_10'], records[0]['graph_sha256'][:16]), flush=True)
        for ds in DATASETS:
            log = out / ('compare-' + ds + '.log')
            cmd = [sys.executable, 'tools/cagra_lowd_quality.py', 'compare',
                   str(out / ('A-' + ds + '.npz')), str(out / ('B-' + ds + '.npz'))]
            if istella_identical:
                cmd.append('--istella-identical')
            rc = run_logged(cmd, log, check=False)
            print(log.read_text().strip().splitlines()[-1], flush=True)
            ok = ok and rc == 0
        if ok:
            with receipt_path.open('x') as stream:
                json.dump(dict(expect, recall_at_10=recalls), stream, sort_keys=True)
        print('CAGRA-LOWD-PAIR status=%s source=%s define=%s' % ('PASS' if ok else 'FAIL', args.source, define))
    finally:
        temporary = installed.with_suffix('.so.restore')
        shutil.copy2(out / 'original.so', temporary)
        os.replace(temporary, installed)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
