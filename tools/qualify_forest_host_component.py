#!/usr/bin/env python3
"""Test the changed Linux forest host component without rebuilding kernels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host-build', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--baseline-version', required=True)
    parser.add_argument('--baseline-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    out.mkdir(exist_ok=False, parents=True)
    (out / 'work').mkdir()
    (out / 'wheels').mkdir()
    host = args.host_build.resolve()
    proof = json.loads((host / 'host-build-proof.json').read_text())
    assert proof['complete'] and proof['source_commit'] == args.source_commit
    changed = {name: row for name, row in proof['group_plan'].items()
               if row['action'] == 'compile'}
    assert len(changed) == 1
    name, group = next(iter(changed.items()))
    assert '_mojolearn_tokenizer_host' in group['members']
    native = host / 'set/host' / name
    assert sha(native) == proof['host_extension']['mojolearn/cpu/host/' + name]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('MOJOLEARN_', 'PYTHON'))}
    env.update(MOJOLEARN_VENDOR='cpu', MOJOLEARN_NUMERIC_MODE='identical',
               MOJOLEARN_CPU_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONNOUSERSITE='1')

    def run(label, argv):
        with (out / (label + '.log')).open('w') as log:
            result = subprocess.run(list(map(str, argv)), cwd=out / 'work', env=env,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=300)
        (out / (label + '.exit')).write_text(str(result.returncode) + '\n')
        return result.returncode

    assert run('venv', [sys.executable, '-m', 'venv', out / 'venv']) == 0
    python = out / 'venv/bin/python'
    assert run('download', [python, '-m', 'pip', 'download', '--no-deps', '--only-binary=:all:',
                           '--dest', out / 'wheels', 'mojolearn-core==' + args.baseline_version]) == 0
    wheel, = (out / 'wheels').glob('*.whl')
    assert sha(wheel) == args.baseline_sha256
    assert run('install', [python, '-m', 'pip', 'install', wheel, 'numpy', 'pytest']) == 0
    site = Path(subprocess.check_output(
        [str(python), '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'],
        cwd=out / 'work', env=env, text=True).strip())
    destination = site / 'mojolearn/host' / name
    assert destination.is_file()
    baseline_native = sha(destination)
    shutil.copy2(native, destination)
    test = root / 'python/mojolearn/tests/test_parallel_forest_first_call.py'
    shutil.copy2(test, out / test.name)
    code = run('tests', [python, '-m', 'pytest', '--noconftest', '--rootdir', out,
                        '--junitxml', out / 'tests.xml', out / test.name, '-q'])
    suites = ET.parse(out / 'tests.xml').getroot()
    cases = list(suites.iter('testcase'))
    passed = code == 0 and len(cases) == 4 and all(
        not any(c.find(tag) is not None for tag in ('failure', 'error', 'skipped')) for c in cases)
    report = dict(schema='mojolearn.targeted-host-component.v1', passed=passed,
                  exit_code=code, cases=len(cases), native_source_commit=args.source_commit,
                  tooling_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                  baseline_wheel_sha256=sha(wheel), baseline_native_sha256=baseline_native,
                  native_sha256=sha(destination), native_members=group['members'],
                  host_proof_sha256=sha(host / 'host-build-proof.json'), test_sha256=sha(test),
                  qualification='Four fresh-process public CPU regressions with the separately compiled host component over an installed baseline. No final-wheel RECORD or GPU-runtime claim.')
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(passed=passed, cases=len(cases), exit_code=code)))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
