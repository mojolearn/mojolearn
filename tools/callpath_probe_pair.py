#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""M3 test-only probe from verified M2 A/B binaries. SOURCE TAG.
Imports each verified .so directly in a fresh process; installed custom binding
need not exist. No install, backup, build fallback, model fit, or timing.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    source, tag = sys.argv[1:]
    assert re.fullmatch('[0-9a-f]{40}', source)
    assert re.fullmatch('[A-Za-z0-9_.-]+', tag)
    root = Path(__file__).resolve().parents[1]
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() == source
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'experiments/apple_callpath',
                    'bench/apple_callpath_quality.mojo', 'bindings/_mojolearn_callpath_probe.mojo',
                    'tools/callpath_probe_pair.py'], cwd=root, check=True)
    arms = Path.home() / 'mq/verified-arms' / source / 'callpath_probe'
    manifest = json.loads((arms / 'manifest.json').read_text())
    assert manifest['source_sha'] == source and manifest['binding'] == 'callpath_probe'
    assert manifest['numeric_mode'] == 'fast' and manifest['defines_A'] == ''
    assert manifest['defines_B'].split() == ['-D', 'MOJOLEARN_APPLE_FAST_CALLPATH_CANDIDATES']
    hashes = {a: digest(arms / (a + '.so')) for a in ('A', 'B')}
    assert hashes == manifest['hashes']
    out = Path.home() / 'mq/out' / (tag + '-quality')
    out.mkdir(parents=True, exist_ok=False)
    code = '''
import importlib.util, json, sys
spec=importlib.util.spec_from_file_location('_mojolearn_callpath_probe',sys.argv[1])
b=importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
expected=int(sys.argv[2])
assert int(b.numeric_mode()) == 0 and str(b.vendor()) == 'apple'
assert int(b.reach()) == expected
assert int(b.run_quality()) == expected
print('CALLPATH-PROBE '+json.dumps(dict(mode='fast',vendor='apple',reach=expected,status='PASS')))
'''
    for arm, reach in (('A', 0), ('B', 1)):
        with (out / (arm + '.log')).open('x') as stream:
            result = subprocess.run([sys.executable, '-c', code, str(arms / (arm + '.so')), str(reach)],
                                    cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            print('\n'.join((out / (arm + '.log')).read_text(errors='replace').splitlines()[-12:]))
            raise RuntimeError('callpath quality failed: ' + arm)
        print(f'CALLPATH-PROBE arm={arm} mode=fast vendor=apple reach={reach} status=PASS')
    receipt = dict(source_sha=source, hashes=hashes, status='PASS', fixture='callpath-bits-lifecycle-v1',
                   mode='fast', vendor='apple', variants=['C1','C2','C3','C4'],
                   baseline='same-binary direct transfers and unchanged minmax kernel; A verifies opt-in off',
                   timing=False)
    with (out / 'PASS.json').open('x') as stream:
        json.dump(receipt, stream, sort_keys=True)
    print('CALLPATH-PAIR ' + json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
