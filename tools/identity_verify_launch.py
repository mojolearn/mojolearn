#!/usr/bin/env python3
"""Run installed full verifier once and retain truthful before/after provenance.

This controller performs no algorithm of its own. The ordinary installed CLI
produces observations; identity_columns converts them without another fit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

SCHEMA = 'mojolearn.installed-verifier-run.v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def source_state(root):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    return dict(root=str(root), commit=git('rev-parse', 'HEAD'),
                dirty_tracked=git('status', '--porcelain', '--untracked-files=no'))


def installed_files(package):
    files = {}
    for path in sorted(package.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            files[path.relative_to(package).as_posix()] = dict(size=path.stat().st_size, sha256=sha(path))
    return files


def locate(python, cwd, env):
    code = "import importlib.util,json; s=importlib.util.find_spec('mojolearn'); print(json.dumps(list(s.submodule_search_locations)[0] if s else None))"
    value = json.loads(subprocess.check_output([str(python), '-c', code], cwd=cwd, env=env, text=True))
    require(value, 'mojolearn is not installed for this interpreter')
    return Path(value).resolve()


def execute(root, python, out, lanes, profile, cpu_threads, *, runner=subprocess.run):
    root, python, out = Path(root).resolve(), Path(python).absolute(), Path(out).resolve()
    require(not out.exists(), 'output already exists; preserve completed observations instead of rerunning')
    out.mkdir(parents=True)
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    # The release columns cover defaults. The sampling experiment needs its own
    # focused runtime/A-B evidence, never implicit promotion through this run.
    env.update(MOJOLEARN_NUMERIC_MODE='identical', MOJOLEARN_EXPERIMENT_SAMPLING_KV_UPLOAD_REUSE='0')
    for key, value in env.items():
        if key.startswith('MOJOLEARN_') and 'SABOTAGE' in key:
            require(value in ('', '0'), 'sabotage environment is not reference evidence')
    source = source_state(root)
    require(not source['dirty_tracked'], 'source checkout has tracked changes')
    package = locate(python, out, env)
    require(not package.is_relative_to(root), 'verifier must load an installed package outside the source checkout')
    files = installed_files(package)
    commit = (package / 'identity_columns/COMMIT').read_text().strip()
    require(commit == source['commit'], 'installed package commit differs from clean source')
    require(sha(package / '_identity_break.py') == sha(root / 'tools/identity_break.py'), 'installed harness differs from frozen source')
    import identity_columns as columns
    harness = columns.harness()
    require(sha(harness.__file__) == sha(root / "tools/identity_break.py"), "controller harness differs from frozen source")
    require(lanes and len(lanes) == len(set(lanes)) and set(lanes) <= set(harness.LANES), 'invalid lane selection')
    require(profile in ('routine', 'neural-training'), 'invalid profile')
    require(set(lanes) <= set(columns.profile_lanes(harness, profile)), 'lanes outside selected profile')
    require(cpu_threads > 0, 'CPU thread request must be positive')
    options = dict(lanes=list(lanes), profile=profile, repeats=1, cpu_threads=cpu_threads,
                   fixtures=list(harness.FIXTURES), parts=list(columns.vref().PARTS + columns.vref().OPTIONAL_PARTS))
    options.update({'all': True, 'include_pending': True, 'no_models': True})
    report = out / 'verify.json'
    command = [str(python), '-m', 'mojolearn', 'verify', '--all', '--include-pending', '--no-models', '--lanes', ','.join(lanes),
               '--repeats', '1', '--cpu-threads', str(cpu_threads), '--json-out', str(report), '--json']
    if profile == 'neural-training':
        command.append('--neural-training')
    receipt = dict(schema=SCHEMA, phase='before-execution', status='running',
                   started_at=datetime.now(timezone.utc).isoformat(), source=source,
                   install=dict(package=str(package), commit=commit, files=files, manifest_sha256=digest(files)),
                   options=options, command=command, environment=dict(numeric_mode='identical', sampling_kv_upload_reuse='0'))
    path = out / 'verifier-run.json'
    save(path, receipt)
    try:
        with (out / 'verify.log').open('w') as stream:
            result = runner(command, cwd=out, env=env, stdout=stream, stderr=subprocess.STDOUT)
        receipt['exit_code'] = result.returncode
        (out / 'verify.exit').write_text(str(result.returncode) + '\n')
        receipt['after_source'] = source_state(root)
        receipt['after_install_manifest_sha256'] = digest(installed_files(package))
        receipt['report_sha256'] = sha(report) if report.is_file() else None
        receipt['status'] = 'completed'  # Completion is not runtime or identity PASS.
    except BaseException as error:
        receipt.update(status='interrupted', error=str(error))
        raise
    finally:
        receipt['finished_at'] = datetime.now(timezone.utc).isoformat()
        save(path, receipt)
    return receipt['exit_code']


def validate_run(receipt, report, report_sha):
    require(receipt.get('schema') == SCHEMA and receipt.get('phase') == 'before-execution', 'missing pre-run verifier receipt')
    require(receipt.get('status') == 'completed' and isinstance(receipt.get('exit_code'), int), 'verifier producer did not complete')
    require(receipt.get('report_sha256') == report_sha, 'verifier report differs from completed run')
    source, after = receipt.get('source', {}), receipt.get('after_source', {})
    require(source and source == after and source.get('dirty_tracked') == '', 'clean source witness missing or changed')
    install = receipt.get('install', {})
    files = install.get('files', {})
    require(files and install.get('manifest_sha256') == digest(files) == receipt.get('after_install_manifest_sha256'), 'installed package changed or lacks manifest')
    device = report.get('device', {})
    require(source.get('commit') == install.get('commit') == device.get('commit'), 'source/install/report commit mismatch')
    require(report.get('binding_provenance_protocol') == 'loaded-native-v2', 'report lacks loaded-native-v2 protocol')
    options = receipt.get('options', {})
    require(options.get('all') is True and options.get('repeats') == report.get('repeats') == 1, 'full one-repeat verifier options missing')
    require(isinstance(options.get('lanes'), list) and len(options['lanes']) == len(set(options['lanes']))
            and len(report.get('lanes', [])) == len(options['lanes'])
            and set(options['lanes']) == set(report.get('lanes', []))
            and set(options.get('fixtures', [])) == set(report.get('fixtures', [])), 'verifier selection differs from pre-run options')
    require(options.get('profile') == report.get('selection', {}).get('profile', {}).get('name'), 'verifier profile differs from launch')
    require(options.get('include_pending') is True and report.get('selection', {}).get('include_pending') is True,
            'current pending/stale fixture observations were not requested')
    require(options.get('no_models') is True and report.get('models_checked') == 0,
            'column producer must defer portable models to their separate once-only smoke')
    command = receipt.get('command', [])
    require(len(command) in (16, 17), 'missing verifier command')
    expected = [command[0], '-m', 'mojolearn', 'verify', '--all', '--include-pending', '--no-models',
                '--lanes', ','.join(options['lanes']), '--repeats', '1', '--cpu-threads',
                str(options.get('cpu_threads')), '--json-out', command[14], '--json']
    if options.get('profile') == 'neural-training':
        expected.append('--neural-training')
    require(command == expected, 'verifier command differs from bound options')
    harness = files.get('_identity_break.py', {})
    require(harness.get('sha256') == report.get('harness', {}).get('sha256'), 'installed harness witness differs')
    package = Path(install.get('package', '/__missing__'))
    for binding in report.get('bindings', []):
        file = Path(binding.get('file', ''))
        require(file.is_absolute() and file.is_relative_to(package), 'native binding lies outside retained install')
        witness = files.get(file.relative_to(package).as_posix(), {})
        require(witness.get('sha256') == binding.get('sha256') and witness.get('size') == binding.get('size'), 'native binding differs from pre-run install')
    return options


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--python', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lanes-file', type=Path, required=True)
    p.add_argument('--profile', choices=('routine', 'neural-training'), required=True)
    p.add_argument('--cpu-threads', type=int, required=True)
    a = p.parse_args()
    return execute(a.root, a.python, a.out, a.lanes_file.read_text().strip().split(','), a.profile, a.cpu_threads)


if __name__ == '__main__':
    sys.exit(main())
