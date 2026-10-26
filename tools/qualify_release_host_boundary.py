#!/usr/bin/env python3
"""Qualify final installed Linux wheels at the changed host boundary only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

from candidate_wheel_install import wheel_record


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    bundle, out = args.bundle.resolve(), args.out.resolve()
    manifest = json.loads((bundle / 'manifest.json').read_text())
    out.mkdir(parents=True, exist_ok=False)
    work = out / 'work'
    work.mkdir()
    wheels = sorted(bundle.glob('*.whl'))
    assert {x.name for x in wheels} == set(manifest['wheels']) and len(wheels) == 5
    for wheel in wheels:
        assert sha(wheel) == manifest['wheels'][wheel.name]
        wheel_record(wheel, manifest['version'], manifest['source_commit'])
    env = {k: v for k, v in os.environ.items() if not k.startswith(('MOJOLEARN_', 'PYTHON'))}
    env.update(MOJOLEARN_VENDOR='cpu', MOJOLEARN_NUMERIC_MODE='identical',
               MOJOLEARN_CPU_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONNOUSERSITE='1')
    checks = {}

    def run(name, command):
        log = out / (name + '.log')
        with log.open('w') as stream:
            result = subprocess.run(list(map(str, command)), cwd=work, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=300)
        checks[name] = dict(exit_code=result.returncode, log=log.name, sha256=sha(log))
        (out / (name + '.exit')).write_text(str(result.returncode) + '\n')
        assert result.returncode == 0, name

    run('venv', [sys.executable, '-m', 'venv', out / 'venv'])
    python = out / 'venv/bin/python'
    run('install', [python, '-m', 'pip', 'install', *wheels, 'numpy', 'pytest'])
    run('dependencies', [python, '-m', 'pip', 'check'])
    run('self-test', [python, '-m', 'mojolearn', 'verify', '--self-test'])
    test = root / 'python/mojolearn/tests/test_parallel_forest_first_call.py'
    shutil.copy2(test, out / test.name)
    run('forest', [python, '-m', 'pytest', '--noconftest', '--rootdir', out,
                   '--junitxml', out / 'tests.xml', out / test.name, '-q'])
    cases = list(ET.parse(out / 'tests.xml').getroot().iter('testcase'))
    assert len(cases) == 4 and not any(c.find(tag) is not None
                                      for c in cases for tag in ('failure', 'error', 'skipped'))
    site = Path(subprocess.check_output([str(python), '-c',
        'import sysconfig; print(sysconfig.get_path("purelib"))'], cwd=work, env=env, text=True).strip())
    report = dict(schema='mojolearn.release.host-boundary.v1', passed=True, exit_code=0,
                  version=manifest['version'], source_commit=manifest['source_commit'],
                  wheels=manifest['wheels'], checks=checks, cases=len(cases),
                  junit=dict(path='tests.xml', sha256=sha(out / 'tests.xml')),
                  test_sha256=sha(test),
                  host_native_sha256=sha(site / 'mojolearn/host/_mojolearn_native_tokenizer_host.so'),
                  verifier_sha256=sha(site / 'mojolearn/_verify_distributed.py'),
                  scope='Final installed wheel, dependencies, comparator self-test and four public CPU forest cases. GPU arithmetic and adapter checks are reused separately.')
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(passed=True, cases=len(cases))))


if __name__ == '__main__':
    main()
