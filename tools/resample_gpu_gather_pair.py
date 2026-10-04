#!/usr/bin/env python3
"""Quality only: SOURCE TAG IBASE_SOURCE. M3 pinned runner, no native builds.
Pair resample artifacts and IDENTICAL base prerequisite must be preverified.
No timing command exists until this full exact-output gate passes.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

DEFINE = '-D MOJOLEARN_RESAMPLE_FAST_GATHER'
FIXTURE = 'resample-gpu-gather-v1'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def verify_artifacts(source, ibase_source):
    assert re.fullmatch('[0-9a-f]{40}', source)
    assert re.fullmatch('[0-9a-f]{40}', ibase_source)
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    harness = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    subprocess.run(['git', 'merge-base', '--is-ancestor', source, harness], check=True)
    drift = subprocess.check_output(['git', 'diff', '--name-only', source, harness], text=True).splitlines()
    assert all(p.startswith(('tools/', 'docs/')) for p in drift), drift
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--'], check=True, stdout=subprocess.DEVNULL)
    assert not subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], text=True).strip()
    assert 'Apple M3 Ultra' in subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True)
    arms = Path.home() / 'mq/verified-arms' / source / 'resample'
    manifest = json.loads((arms / 'manifest.json').read_text())
    assert manifest['source_sha'] == source and manifest['binding'] == 'resample'
    assert manifest['numeric_mode'] == 'fast'
    assert manifest['defines_A'] == '' and manifest['defines_B'] == DEFINE
    hashes = {arm: digest(arms / (arm + '.so')) for arm in ('A', 'B')}
    assert manifest['hashes'] == hashes
    ibase = Path.home() / 'mq/verified-arms' / ibase_source / 'ibase'
    bm = json.loads((ibase / 'manifest.json').read_text())
    assert bm['source_sha'] == ibase_source and bm['binding'] == 'ibase'
    assert bm['numeric_mode'] == 'identical' and bm['defines'] == ''
    assert bm['artifact'] == '_mojolearn.so'
    base_binary = ibase / '_mojolearn.so'
    assert bm['sha256'] == digest(base_binary)
    # Base dependency must be built from exactly this production source.
    subprocess.run(['git', 'merge-base', '--is-ancestor', ibase_source, source], check=True)
    changed = subprocess.check_output(['git', 'diff', '--name-only', ibase_source, source], text=True).splitlines()
    allowed = {'resample/estimator.mojo', 'resample/gather_fast.mojo',
               'bindings/_mojolearn_resample.mojo', 'python/mojolearn/resample.py'}
    assert all(p in allowed or p.startswith(('tools/', 'docs/')) for p in changed), changed
    return root, arms, hashes, bm, base_binary, harness


def main():
    source, tag, ibase_source = sys.argv[1:]
    assert re.fullmatch('[A-Za-z0-9_.-]+', tag)
    root, arms, hashes, bm, base_binary, harness = verify_artifacts(source, ibase_source)
    out = Path.home() / 'mq/out' / (tag + '-quality')
    out.mkdir(parents=True, exist_ok=False)
    pkg = root / 'python/mojolearn'
    installed = pkg / '_mojolearn_resample.so'
    base_installed = pkg / 'identical/_mojolearn.so'
    # Fresh pinned tree only; never replace another job's installed artifacts.
    assert not installed.exists() and not base_installed.exists()
    base_installed.parent.mkdir(exist_ok=True)
    shutil.copy2(base_binary, base_installed)
    env = os.environ.copy()
    env.update(MOJOLEARN_NUMERIC_MODE='fast', MOJOLEARN_VENDOR='apple',
               PYTHONPATH=str(root / 'python'), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    records = {}
    try:
        for arm in ('A', 'B'):
            shutil.copy2(arms / (arm + '.so'), installed)
            with (out / (arm + '.log')).open('x') as stream:
                result = subprocess.run([sys.executable, 'tools/resample_gpu_gather_quality.py',
                                         str(out / (arm + '.npz')), str(int(arm == 'B'))],
                                        env=env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                print('\n'.join((out / (arm + '.log')).read_text().splitlines()[-12:]))
                raise RuntimeError('quality failed: ' + arm)
            rec = json.loads((out / (arm + '.npz.json')).read_text())
            assert rec['status'] == 'PASS' and rec['fixture'] == FIXTURE
            assert rec['binding_sha256'] == hashes[arm]
            assert rec['enabled'] == int(arm == 'B')
            assert (rec['successful_gpu_calls'] > 0) == (arm == 'B')
            records[arm] = rec
        import numpy as np
        with np.load(out / 'A.npz') as a, np.load(out / 'B.npz') as b:
            assert a.files == b.files
            for key in a.files:
                assert a[key].shape == b[key].shape and a[key].dtype == b[key].dtype
                assert a[key].tobytes() == b[key].tobytes(), key
        assert records['A']['refusals'] == records['B']['refusals']
        receipt = dict(status='PASS', fixture=FIXTURE, source_sha=source, harness_source=harness, hashes=hashes,
                       defines_A='', defines_B=DEFINE, ibase_source=ibase_source,
                       ibase_sha256=bm['sha256'], arrays=records['A']['arrays'], records=records,
                       quality_script_sha256=digest(root / 'tools/resample_gpu_gather_quality.py'))
        (out / 'PASS.json').write_text(json.dumps(receipt, sort_keys=True))
        print('RESAMPLE_GPU_PAIR PASS ' + str(out / 'PASS.json'))
    finally:
        installed.unlink(missing_ok=True)
        base_installed.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
