#!/usr/bin/env python3
"""Reproduce the two runner failures without network access or a rental."""
import os
from pathlib import Path
import subprocess
import tempfile

source = Path('tools/gemm_remote_leg.sh').read_text()
a = source.index('    MOJOLEARN_STAGE_KEYS="${MOJOLEARN_STAGE_KEYS-$_stage_default}"')
b = source.index('    # lane/r2-binding-cache', a)
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp); (p/'tools').mkdir()
    stage = p/'tools/stage_from_r2.sh'
    script = ('set -eu\nOUT=.\nSSH_TARGET=dummy\n_stage_default=corpus\n'
              'leg_say() { echo "$*"; }\n'
              'leg_die() { echo "$*"; exit 27; }\n'
              + source[a:b] + 'echo PAYLOAD_STARTED\n')
    for code in (17, 0):
        stage.write_text(f'#!/bin/sh\necho stage_injected_exit={code}\nexit {code}\n')
        r = subprocess.run(['sh', '-c', script], cwd=p, text=True, capture_output=True,
                           env={**os.environ, 'MOJOLEARN_STAGE_STRICT': '1'})
        print(r.stdout, end='')
        assert r.returncode == (27 if code else 0), r
        assert ('PAYLOAD_STARTED' in r.stdout) == (code == 0)
        print('EXPECTED FAILURE BEFORE PAYLOAD' if code else 'PASS STAGING CONTINUES')

line = next(x.strip() for x in source.splitlines() if x.strip().startswith('POD_NAME="mojolearn-gemm-'))
for bad in (True, False):
    spelling = line.replace('-$$', '') if bad else line
    script = 'VENDOR=nvidia\nSTAMP=fixed\n' + spelling + '\nprintf "%s\\n" "$POD_NAME"\n'
    children = [subprocess.Popen(['sh', '-c', script], stdout=subprocess.PIPE, text=True) for _ in range(2)]
    names = [p.communicate()[0].strip() for p in children]
    print('POD NAMES', names)
    if bad:
        try: assert names[0] != names[1], 'recovery names collide'
        except AssertionError as exc: print('EXPECTED FAIL', str(exc))
        else: raise AssertionError('did not reproduce old collision')
    else:
        assert names[0] != names[1]
        print('PASS distinct recovery names in the same timestamp second')
