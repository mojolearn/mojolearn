#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Manager-only quality pair / quality-gated timing for MOJOLEARN_EIGH_FAST_PANEL_DF
(lane w4-eigh), on verified prebuilt x_decomp arms (tools/target_scratch_pair.py
pattern).

quality SOURCE QUALITY_TAG
No builds, SSH, queue edits, or opponent runs. SOURCE is the exact full SHA.
Quality only, no timing receipt. Status follows the FAST rule
(tools/fast_quality_rule.py): `eigh_w4_quality.py compare` (within noise of
FAST main plus the absolute bounds) decides PASS; the strict zero-tolerance
no-regression result is saved as info only.
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
DEFINE = 'MOJOLEARN_EIGH_FAST_PANEL_DF'


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
    parser.add_argument('action', choices=('quality',))
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
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', '*.mojo', 'bindings/', 'python/', 'pixi.toml', 'pixi.lock',
                    'tools/eigh_w4_quality.py', 'tools/eigh_panel_df_pair.py',
                    'tools/fast_quality_rule.py'], check=True)
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
    assert args.timing_tag is None
    out.mkdir(parents=True, exist_ok=False)  # never silently reuse partial captures
    os.environ.update(MOJOLEARN_NUMERIC_MODE='fast', MOJOLEARN_VENDOR='apple',
                      MOJOLEARN_BENCH_INSTALLED='0', PYTHONPATH=str(root / 'python'),
                      OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    installed = root / 'python/mojolearn/_mojolearn_x_decomp.so'
    if not installed.exists():
        shutil.copy2(arms / 'A.so', installed)
    original_hash = digest(installed)
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
        run_logged([sys.executable, 'tools/eigh_w4_quality.py', 'no-regression',
                    str(out / 'A.json'), str(out / 'B.json')], out / 'no-regression.log')
        line = (out / 'no-regression.log').read_text().strip()
        prefix = 'EIGH-MAIN-NO-REGRESSION '
        assert line.startswith(prefix)
        no_reg = json.loads(line[len(prefix):])
        # The FAST-rule comparator over all eight cases decides the status.
        with (out / 'compare.log').open('x') as stream:
            absolute = subprocess.run([sys.executable, 'tools/eigh_w4_quality.py', 'compare',
                                       str(out / 'A.json'), str(out / 'B.json')],
                                      stdout=stream, stderr=subprocess.STDOUT)
        report = dict(source_sha=args.source, hashes=hashes, define=DEFINE,
                      fixture='eigh-panel-df-v1', baseline='current panel default',
                      strict_no_regression_info=no_reg['status'],
                      status='PASS' if absolute.returncode == 0 else 'HOLD',
                      rule='fast-quality-v1 (tools/fast_quality_rule.py)',
                      opponent_status='HOLD: separate opponent admission required',
                      timing_authorized=False, promotion_authorized=False,
                      quality_script_sha256=digest(root / 'tools/eigh_w4_quality.py'))
        (out / 'REPORT.json').write_text(json.dumps(report, indent=2) + '\n')
        print('EIGH-PANEL-DF-QUALITY ' + json.dumps(report, sort_keys=True))
        if absolute.returncode != 0:
            raise RuntimeError('HOLD: quality outside FAST main noise or the absolute bounds; see saved report')
    finally:
        temporary = installed.with_suffix('.so.restore')
        shutil.copy2(out / 'original.so', temporary)
        os.replace(temporary, installed)
        assert digest(installed) == original_hash


if __name__ == '__main__':
    main()
