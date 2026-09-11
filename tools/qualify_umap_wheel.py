#!/usr/bin/env python3
"""Main-only UMAP qualification of one exact wheel in a disposable venv.

No builds or publication. Only invocation by the main lane performs installs
and GPU checks. Tests/harnesses come from --source-root; mojolearn itself must
come from the installed wheel. Every subprocess has a 180-second deadline.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


GUARD = r'''
import hashlib, json, os, pathlib, runpy, sys
import mojolearn
from mojolearn import _umap_impl

prefix = pathlib.Path(sys.prefix).resolve()
package = pathlib.Path(mojolearn.__file__).resolve()
wrapper = pathlib.Path(_umap_impl.__file__).resolve()
assert package.is_relative_to(prefix) and 'site-packages' in package.parts, package
assert wrapper.is_relative_to(package.parent), wrapper
assert mojolearn.__version__ == os.environ['UMAP_EXPECT_VERSION'], mojolearn.__version__
wrapper_sha = hashlib.sha256(wrapper.read_bytes()).hexdigest()
assert wrapper_sha == os.environ['UMAP_EXPECT_WRAPPER_SHA'], 'Installed wrapper differs from frozen source'
mode = os.environ['MOJOLEARN_NUMERIC_MODE']
if mode != 'identical':
    # DEVIATION 2490: UMAP is IDENTICAL only. The installed wheel must refuse
    # the lower tiers BY NAME before touching the disk; the refusal is the
    # evidence for this arm and the target never runs.
    try:
        mojolearn.UMAP(numeric_mode=mode)._bind()
    except ValueError as exc:
        refusal = str(exc)
    else:
        raise AssertionError('UMAP accepted numeric_mode=' + mode + '; it is IDENTICAL only (DEVIATION 2490)')
    assert '_mojolearn_metrics' in refusal and 'IDENTICAL only' in refusal, refusal
    record = {'package': str(package), 'version': mojolearn.__version__,
              'wrapper': str(wrapper), 'wrapper_sha256': wrapper_sha,
              'mode': mode, 'refused': True, 'refusal': refusal, 'python': sys.version}
    pathlib.Path(os.environ['UMAP_GUARD_OUTPUT']).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    raise SystemExit(0)
binding = mojolearn.UMAP(numeric_mode=mode)._bind()
binary = pathlib.Path(binding.__file__).resolve()
assert binary.is_relative_to(prefix) and 'site-packages' in binary.parts, binary
code = int(binding.umap_numeric_mode())
assert code == {'fast': 0, 'identical': 1, 'deterministic': 2}[mode], code
record = {'package': str(package), 'version': mojolearn.__version__,
          'wrapper': str(wrapper), 'wrapper_sha256': wrapper_sha,
          'binding_file': str(binary), 'binding_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
          'mode': mode, 'binding_mode_code': code, 'python': sys.version}
pathlib.Path(os.environ['UMAP_GUARD_OUTPUT']).write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
target = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(target, run_name='__main__')
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--python', default='/opt/homebrew/bin/python3.12')
    parser.add_argument('--expected-version', required=True)
    parser.add_argument('--device', default='Apple M4')
    parser.add_argument('--wheelhouse', type=Path,
                        help='Install dependencies only from this local directory, without an index')
    args = parser.parse_args()
    wheel, root, output = args.wheel.resolve(), args.source_root.resolve(), args.output.resolve()
    wheelhouse = args.wheelhouse.resolve() if args.wheelhouse else None
    if wheelhouse is not None and not wheelhouse.is_dir():
        raise SystemExit('Missing dependency wheelhouse: ' + str(wheelhouse))
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'results.json').exists():
        raise SystemExit('Refusing to overwrite previous qualification results')
    wrapper = root / 'python/mojolearn/_umap_impl.py'
    tests = {
        'fit': root / 'python/mojolearn/tests/test_umap_surface.py',
        'transform': root / 'python/mojolearn/tests/test_umap_transform.py',
        'quality': root / 'tools/umap_transform_quality_check.py',
    }
    for path in (wheel, wrapper, *tests.values()):
        if not path.is_file():
            raise SystemExit('Missing required file: ' + str(path))
    manifest = {
        'status': 'INCOMPLETE', 'started_at': datetime.now(timezone.utc).isoformat(),
        'wheel': str(wheel), 'wheel_sha256': sha(wheel),
        'source_root': str(root), 'expected_version': args.expected_version,
        'device': args.device, 'python_requested': args.python,
        'source_files': {str(p.relative_to(root)): sha(p) for p in (wrapper, *tests.values())},
        'timeout_seconds': 180, 'jobs': [],
        'dependency_wheelhouse': str(wheelhouse) if wheelhouse else None,
        'dependency_wheels': {p.name: sha(p) for p in sorted(wheelhouse.glob('*.whl'))}
                             if wheelhouse else {},
    }
    source_commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                   capture_output=True, text=True, timeout=10)
    manifest['source_commit'] = source_commit.stdout.strip() or 'unavailable'
    env = dict(os.environ)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'MOJOLEARN_VENDOR'):
        env.pop(name, None)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    env['UMAP_EXPECT_VERSION'] = args.expected_version
    env['UMAP_EXPECT_WRAPPER_SHA'] = manifest['source_files'][str(wrapper.relative_to(root))]

    def save():
        (output / 'results.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def run(name, command, cwd, extra=None):
        local_env = dict(env, **(extra or {}))
        log = output / (name + '.log')
        begin = time.monotonic()
        status = None
        timed_out = False
        try:
            with log.open('w') as handle:
                proc = subprocess.Popen(command, cwd=cwd, env=local_env,
                                        stdout=handle, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                try:
                    status = proc.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    status = 124
        except Exception as exc:
            status = 125
            with log.open('a') as handle:
                handle.write('\nHARNESS ERROR: ' + repr(exc) + '\n')
        record = {'name': name, 'command': command, 'exit_code': status,
                  'timed_out': timed_out, 'seconds': time.monotonic() - begin,
                  'log': str(log)}
        manifest['jobs'].append(record)
        save()
        print(json.dumps({'job': name, 'exit_code': status, 'timed_out': timed_out}), flush=True)
        return status == 0

    save()
    try:
        with tempfile.TemporaryDirectory(prefix='mojolearn-umap-wheel-') as work:
            work = Path(work)
            venv = work / 'venv'
            manifest['temporary_venv'] = str(venv)
            if not run('create-venv', [args.python, '-m', 'venv', str(venv)], work):
                raise RuntimeError('Could not create isolated venv')
            python = str(venv / 'bin/python')
            install_options = (['--no-index', '--find-links', str(wheelhouse)]
                               if wheelhouse else [])
            if not run('install-wheel', [python, '-m', 'pip', 'install', '--disable-pip-version-check',
                       '--no-input', '--only-binary=:all:', *install_options, str(wheel)], work):
                raise RuntimeError('Exact wheel installation failed')
            if not run('check-dependencies', [python, '-m', 'pip', 'check'], work):
                raise RuntimeError('Installed wheel dependencies are inconsistent')
            # The wheel's runtime is dependency-free; NumPy belongs to the
            # test harness (the three targets import it), not to the wheel.
            if not run('install-test-dependencies', [python, '-m', 'pip', 'install',
                       '--disable-pip-version-check', '--no-input', '--only-binary=:all:',
                       *install_options, 'numpy>=1.24'], work):
                raise RuntimeError('Test dependency installation failed')
            if not run('installed-packages', [python, '-m', 'pip', 'list', '--format=json',
                       '--disable-pip-version-check'], work):
                raise RuntimeError('Could not retain installed dependency versions')
            guard = work / 'run_installed.py'
            guard.write_text(GUARD)
            passed = True
            for mode in ('fast', 'deterministic', 'identical'):
                for surface, target in tests.items():
                    name = surface + '-' + mode
                    extra = {'MOJOLEARN_NUMERIC_MODE': mode,
                             'UMAP_GUARD_OUTPUT': str(output / (name + '.installed.json'))}
                    command = [python, str(guard), str(target)]
                    if surface == 'quality':
                        command += ['--mode', mode, '--device', args.device,
                                    '--source-root', str(root),
                                    '--output', str(output / (name + '.json'))]
                    job_passed = run(name, command, work, extra)
                    if job_passed:
                        installed_record = json.loads(Path(extra['UMAP_GUARD_OUTPUT']).read_text())
                        manifest['jobs'][-1]['installed'] = installed_record
                        if mode != 'identical':
                            if installed_record.get('refused') is not True:
                                raise RuntimeError('Lower tier did not refuse UMAP: ' + name)
                        elif surface == 'quality':
                            quality = json.loads((output / (name + '.json')).read_text())
                            if quality.get('status') != 'PASS' or not quality.get('results'):
                                raise RuntimeError('Quality job returned without complete passing evidence')
                            for result in quality['results']:
                                if (not result.get('passed') or
                                    result.get('binding_sha256') != installed_record['binding_sha256']):
                                    raise RuntimeError('Quality binding provenance/admission mismatch')
                    passed = job_passed and passed
            if sha(wheel) != manifest['wheel_sha256']:
                raise RuntimeError('Wheel changed during qualification')
            if wheelhouse is not None:
                for name, expected in manifest['dependency_wheels'].items():
                    if sha(wheelhouse / name) != expected:
                        raise RuntimeError('Dependency wheel changed during qualification: ' + name)
            for relative, expected in manifest['source_files'].items():
                if sha(root / relative) != expected:
                    raise RuntimeError('Frozen qualification source changed: ' + relative)
            manifest['status'] = 'PASSED' if passed else 'FAILED'
    except Exception as exc:
        manifest['status'] = 'FAILED'
        manifest['reason'] = repr(exc)
    finally:
        manifest['temporary_venv_removed'] = not Path(
            manifest.get('temporary_venv', '/nonexistent-umap-qualification-venv')
        ).exists()
        manifest['completed_at'] = datetime.now(timezone.utc).isoformat()
        save()
    return 0 if manifest['status'] == 'PASSED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
