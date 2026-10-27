#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One local source/runtime verification command; never compiles or benchmarks.

Runtime mode requires an explicit candidate interpreter, package and build record.
Each independent check gets its own process/log/result. No source pass qualifies
runtime, and no bounded public pass qualifies all 29 recipes or other devices.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback
import types
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUITES = (
    'test_generation_metadata_boundary.py', 'test_neural_metadata_boundary.py',
    'test_generation_native_abi.py', 'test_neural_native_descriptors.py',
    'test_neural_provider_layout.py', 'test_sequential_native_provider.py', 'test_xseq_native_provider.py',
    'test_neural_provider_reuse.py', 'test_native_execution_boundary.py',
    'test_native_metadata_snapshot.py', 'test_native_prep_plans.py',
    'test_runtime_fallback_scope.py', 'test_verify_neural_provider.py',
)
SOURCE_AUDITS = ('check_neural_native_boundary.py', 'check_generation_native_boundary.py',
                 'check_neural_provider_layout.py')
RUNTIME_SUITES = ('test_neural_metadata_snapshot.py', 'test_generation_metadata_snapshot.py',
                  'test_sequential_native_provider.py', 'test_xseq_native_provider.py')
RUNTIME_PROBES = (
    'experiments/native_dispatch_20261027/public_runtime_probe.py',
    'experiments/neural_native_dispatch_20261027/public_runtime_probe.py',
    'experiments/samba_native_dispatch_20261027/public_provider_probe.py',
)
PENDING = [
    'Full 29-recipe runtime qualification (mapping/source checks are not execution)',
    'Other vendors/devices and within-release cross-vendor identity',
    'Full-workload performance, concurrency and remaining mixed/provider paths',
    'Resident, fixed15, distributed and optional training paths beyond selected probes',
]
ENV_KEYS = ('MOJOLEARN_VENDOR', 'MOJOLEARN_GENERATION_DEVICE',
            'MOJOLEARN_NEURAL_NATIVE_DISPATCH', 'MOJOLEARN_GENERATION_NATIVE_SESSION',
            'MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD', 'MOJOLEARN_TRANSFORMER_LEGACY_SETUP',
            'MOJOLEARN_BUILD_ROOT', 'MOJOLEARN_NATIVE_ROOT', 'DYLD_LIBRARY_PATH',
            'LD_LIBRARY_PATH', 'PYTHONPATH')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def source_provenance(root):
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=root).decode().split('\0')
    files = {}
    for name in names:
        path = root / name
        if path.is_file() and path.suffix in {'.mojo', '.py', '.h', '.cpp', '.toml', '.lock'}:
            files[name] = digest(path)
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                git_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True),
                files=files, scope='tracked source bytes; not proof these bytes produced a binary')


def pin_candidate(package, binary_roots):
    """Pin the explicitly selected tree before launching any candidate process."""
    pinned = {}
    for base in (package, *binary_roots):
        if not base.is_dir():
            raise ValueError(f'candidate directory missing: {base}')
        for path in base.rglob('*'):
            if path.is_file() and path.suffix in {'.py', '.so', '.dylib', '.pyd'}:
                canonical = str(path.resolve())
                if canonical not in pinned:
                    pinned[canonical] = digest(path)
    if str((package / '__init__.py').resolve()) not in pinned:
        raise ValueError('candidate package must be the mojolearn directory containing __init__.py')
    if not any(Path(path).suffix in {'.so', '.dylib', '.pyd'} for path in pinned):
        raise ValueError('candidate has no native binaries; supply their explicit --binary-root')
    return pinned


def retain_build_records(paths, out):
    summaries = []
    for index, path in enumerate(paths):
        data = json.loads(path.read_text())
        target = out / f'build-record-{index}.json'
        target.write_bytes(path.read_bytes())
        # Keep incomplete selected-group builds visible without disqualifying
        # unrelated successful groups. Actual loaded bytes are verified below.
        groups = [{key: item.get(key) for key in ('key', 'mode', 'stem', 'state', 'exit_code',
                   'sha256', 'log', 'execution_receipt')} for item in data.get('results', [])]
        references = set()
        for item in data.get('results', []):
            if item.get('key'):
                references.add(path.parent / 'artifacts' / item['key'] / 'receipt.json')
            receipt = item.get('execution_receipt')
            if isinstance(receipt, str):
                references.add(Path(receipt) if Path(receipt).is_absolute() else path.parent / receipt)
        for item in data.get('replacements', []):
            receipt = item.get('build_receipt')
            if isinstance(receipt, str):
                references.add(Path(receipt) if Path(receipt).is_absolute() else path.parent / receipt)
        retained_receipts = []
        for receipt_index, receipt in enumerate(sorted(references)):
            item = dict(path=str(receipt), retained=False)
            if receipt.is_file():
                destination = out / f'build-record-{index}-receipt-{receipt_index}.json'
                destination.write_bytes(receipt.read_bytes())
                item.update(retained=str(destination), sha256=digest(destination))
            retained_receipts.append(item)
        summaries.append(dict(path=str(path), retained=str(target), sha256=digest(target),
                              referenced_receipts=retained_receipts,
                              schema=data.get('schema'), groups=groups,
                              baseline_wheel=data.get('baseline_wheel'),
                              replacements=data.get('replacements', [])))
    return summaries


def validate_loaded(observed, pinned):
    rows, failures = [], []
    for name, filename in sorted(observed.items()):
        path = Path(filename).resolve()
        try:
            actual = digest(path)
        except OSError as error:
            failures.append(f'{name}: {error}')
            continue
        expected = pinned.get(str(path))
        row = dict(module=name, path=str(path), sha256=actual, expected_sha256=expected)
        row['verified'] = expected is not None and actual == expected
        rows.append(row)
        if not row['verified']:
            failures.append(f'{name}: unpinned loaded path or changed bytes: {path}')
    native = [row for row in rows if Path(row['path']).suffix in {'.so', '.dylib', '.pyd'}]
    if not native:
        failures.append('no loaded native product binary was observed')
    return rows, failures


def junit_result(path):
    root = ET.parse(path).getroot()
    cases = []
    for case in root.iter('testcase'):
        problem = next((child for child in case if child.tag in {'failure', 'error', 'skipped'}), None)
        cases.append(dict(name=case.get('name'), classname=case.get('classname'),
                          outcome='passed' if problem is None else problem.tag,
                          message=None if problem is None else problem.get('message')))
    if not cases:
        raise ValueError('pytest collected no test cases')
    return dict(cases=cases, passed=sum(c['outcome'] == 'passed' for c in cases),
                failed=sum(c['outcome'] != 'passed' for c in cases),
                status='passed' if all(c['outcome'] == 'passed' for c in cases) else 'failed')


class PytestEvents:
    def __init__(self):
        self.events = []

    def pytest_runtest_logreport(self, report):
        self.events.append(dict(nodeid=report.nodeid, phase=report.when, outcome=report.outcome,
                                expected_failure=bool(getattr(report, 'wasxfail', False))))

    def pytest_collectreport(self, report):
        if report.outcome != 'passed':
            self.events.append(dict(nodeid=report.nodeid, phase='collection', outcome=report.outcome))


def worker(config_path):
    """Runs under the supplied candidate Python; this function is test tooling."""
    config = json.loads(config_path.read_text())
    result = dict(status='failed', interpreter=dict(executable=sys.executable, prefix=sys.prefix,
                  real_executable=str(Path(sys.executable).resolve()), sha256=digest(sys.executable),
                  version=sys.version), platform=platform.platform(),
                  environment={key: os.environ.get(key) for key in ENV_KEYS})
    observed = {}
    package = Path(config['package'])
    pinned = json.loads(Path(config['pinned']).read_text())

    def observe(name, filename):
        if filename and isinstance(filename, str) and Path(filename).is_file():
            observed[name] = str(Path(filename).resolve())

    def audit(event, args):
        if event == 'import' and args and str(args[0]).startswith(('mojolearn', '_mojolearn')):
            if not str(args[0]).startswith('mojolearn.tests'):
                observe(str(args[0]), args[1] if len(args) > 1 else None)

    sys.addaudithook(audit)
    try:
        # -I ignores ambient PYTHONPATH/site user settings. Admit only the
        # requested candidate package; test fixtures get a tests-only namespace.
        sys.path.insert(0, str(package.parent))
        if Path(sys.executable).resolve() != Path(config['python']).resolve() or digest(sys.executable) != config['python_sha256']:
            raise RuntimeError('candidate interpreter path/hash differs from supplied interpreter')
        if platform.system() != 'Darwin':
            raise RuntimeError('selected runtime probes are scoped to the local Mac')
        import mojolearn
        if Path(mojolearn.__file__).resolve() != (package / '__init__.py').resolve():
            raise RuntimeError('loaded mojolearn package is not the supplied candidate')
        tests = types.ModuleType('mojolearn.tests')
        tests.__path__ = [str(Path(config['root']) / 'python/mojolearn/tests')]
        sys.modules['mojolearn.tests'] = tests
        mojolearn.tests = tests
        if config['kind'] == 'pytest':
            import pytest
            events = PytestEvents()
            code = pytest.main(['-q', '--import-mode=importlib', '-o', 'addopts=',
                                '--junitxml=' + config['junit'], config['path']], plugins=[events])
            result['pytest_exit_code'] = int(code)
            result['pytest'] = junit_result(config['junit'])
            result['events'] = events.events
            unexpected = any(e['outcome'] == 'skipped' or e.get('expected_failure') for e in events.events)
            result['status'] = 'passed' if code == 0 and result['pytest']['status'] == 'passed' and not unexpected else 'failed'
        else:
            spec = importlib.util.spec_from_file_location('provider_public_probe', config['path'])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            try:
                result['probe'] = module.qualify()
            finally:
                # Preserve structured audit reports even if assert_native raises.
                result['runtime_audits'] = getattr(module, 'RUNTIME_AUDITS', [])
            result['status'] = 'passed' if result['probe'].get('status') == 'passed' else 'failed'
    except BaseException:
        result['exception'] = traceback.format_exc()
        traceback.print_exc()
        result['status'] = 'failed'
    finally:
        for name, module in tuple(sys.modules.items()):
            if name.startswith(('mojolearn', '_mojolearn')) and not name.startswith('mojolearn.tests'):
                if isinstance(module, types.ModuleType):
                    observe(name, types.ModuleType.__getattribute__(module, '__dict__').get('__file__'))
        try:
            result['loaded_files'], result['provenance_failures'] = validate_loaded(observed, pinned)
            if result['provenance_failures']:
                result['status'] = 'failed'
        except BaseException:
            result['provenance_exception'] = traceback.format_exc()
            result['status'] = 'failed'
        result['final_environment'] = {key: os.environ.get(key) for key in ENV_KEYS}
        write(config['result'], result)
    return int(result['status'] != 'passed')


def execute(command, log, cwd, env, timeout):
    start = time.monotonic()
    result = dict(command=command, log=str(log), status='failed')
    with log.open('w') as stream:
        try:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                result['exit_code'] = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                result.update(exit_code=process.returncode, timed_out=True)
                stream.write(f'\nVerifier timeout after {timeout} seconds\n')
            result['status'] = 'passed' if result['exit_code'] == 0 else 'failed'
        except Exception:
            result['exception'] = traceback.format_exc()
            stream.write(result['exception'])
    result['elapsed_seconds'] = time.monotonic() - start
    return result


def verify(args):
    out = args.output.absolute()
    out.mkdir(parents=True, exist_ok=False)
    root = ROOT
    source = source_provenance(root)
    write(out / 'source-provenance.json', source)
    report = dict(schema='mojolearn.neural-provider-verification.v1', mode=args.mode,
                  source_commit=source['commit'], source_provenance=str(out / 'source-provenance.json'),
                  checks=[], expected_source_checks=list(SOURCE_AUDITS + SOURCE_SUITES),
                  expected_runtime_checks=list(RUNTIME_SUITES + RUNTIME_PROBES),
                  pending=list(PENDING), status='running',
                  selected_runtime_checks_passed=False, runtime_scope='selected bounded public probes on local Mac only')
    write(out / 'result.json', report)
    env = os.environ.copy()
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    env.pop('PYTEST_ADDOPTS', None)
    for name in SOURCE_AUDITS + SOURCE_SUITES:
        test = name.startswith('test_')
        xml = out / (name + '.xml')
        command = [sys.executable, '-m', 'pytest', '-q', '-o', 'addopts=', '--junitxml=' + str(xml), str(root / 'tools' / name)] if test else [sys.executable, str(root / 'tools' / name)]
        result = execute(command, out / (name + '.log'), root, env, args.timeout)
        result.update(name=name, phase='source')
        if test:
            try:
                result['pytest'] = junit_result(xml)
                if result['pytest']['status'] != 'passed':
                    result['status'] = 'failed'
            except Exception:
                result['report_error'] = traceback.format_exc()
                result['status'] = 'failed'
        report['checks'].append(result)
        write(out / 'result.json', report)
        print(f"{result['status']}: source {name}", flush=True)
    if args.mode == 'runtime':
        try:
            pinned = pin_candidate(args.package, args.binary_root)
            write(out / 'candidate-files.json', pinned)
            report['build_records'] = retain_build_records(args.build_record, out)
            report['build_provenance_scope'] = ('Records retained as declared. Per-check loaded paths/hashes are verified against pinned candidate bytes; '
                'unrelated incomplete build groups remain visible and do not invalidate successful exercised groups. '
                'No inference of binary/source correspondence from checkout HEAD.')
            report['candidate'] = dict(python=str(args.python), package=str(args.package),
                                       pinned_files=str(out / 'candidate-files.json'))
            config_base = dict(root=str(root), package=str(args.package), pinned=str(out / 'candidate-files.json'),
                               python=str(args.python), python_sha256=digest(args.python))
        except Exception:
            problem = traceback.format_exc()
            admission_log = out / 'candidate-admission.log'
            admission_log.write_text(problem)
            report['checks'].append(dict(name='candidate-admission', phase='runtime', status='failed',
                                         exception=problem, log=str(admission_log)))
            config_base = None
        if config_base is not None:
            env.update(MOJOLEARN_GENERATION_DEVICE='gpu', MOJOLEARN_NEURAL_NATIVE_DISPATCH='1',
                       MOJOLEARN_GENERATION_NATIVE_SESSION='0', MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD='0')
            env.pop('MOJOLEARN_TRANSFORMER_LEGACY_SETUP', None)
            for index, relative in enumerate(tuple('python/mojolearn/tests/' + name for name in RUNTIME_SUITES) + RUNTIME_PROBES):
                name = f'runtime-{index}-{Path(relative).stem}'
                config = dict(config_base, kind='pytest' if relative.startswith('python/') else 'probe',
                              path=str(root / relative), result=str(out / (name + '.json')),
                              junit=str(out / (name + '.xml')))
                config_path = out / (name + '-config.json')
                write(config_path, config)
                check_env = env.copy()
                # Each A/B probe owns its switches; the older session-only
                # probe requires an unmodified baseline before enabling itself.
                if config['kind'] == 'probe':
                    check_env['MOJOLEARN_NEURAL_NATIVE_DISPATCH'] = '0'
                result = execute([str(args.python), '-I', str(Path(__file__).resolve()), '--worker', str(config_path)],
                                 out / (name + '.log'), out, check_env, args.timeout)
                result.update(name=name, phase='runtime', result=config['result'])
                try:
                    details = json.loads(Path(config['result']).read_text())
                    result['runtime_status'] = details['status']
                    if 'pytest' in details:
                        result['cases_passed'] = details['pytest']['passed']
                        result['cases_failed'] = details['pytest']['failed']
                    probe = details.get('probe', {})
                    for pending in probe.get('pending', []) + probe.get('not_qualified', []):
                        report['pending'].append(name + ': ' + pending)
                    loaded_hashes = {row['sha256'] for row in details.get('loaded_files', [])
                                     if Path(row['path']).suffix in {'.so', '.dylib', '.pyd'}}
                    result['loaded_native_binaries'] = len(loaded_hashes)
                    result['matching_build_groups'] = [dict(record=record['path'], **group)
                        for record in report['build_records'] for group in record['groups']
                        if group.get('sha256') in loaded_hashes]
                    result['matching_replacements'] = [dict(record=record['path'], **replacement)
                        for record in report['build_records'] for replacement in record['replacements']
                        if replacement.get('sha256') in loaded_hashes]
                    if details['status'] != 'passed':
                        result['status'] = 'failed'
                except Exception:
                    result['report_error'] = traceback.format_exc()
                    result['status'] = 'failed'
                report['checks'].append(result)
                write(out / 'result.json', report)
                print(f"{result['status']}: {name}", flush=True)
    else:
        report['pending'].insert(0, 'ALL candidate compiled/public runtime checks: source-only mode requested')
    failed = [item['name'] for item in report['checks'] if item['status'] != 'passed']
    report['failed_checks'] = failed
    report['status'] = 'failed' if failed else ('source_passed_runtime_pending' if args.mode == 'source-only' else 'selected_runtime_passed')
    runtime_checks = [c for c in report['checks'] if c['phase'] == 'runtime']
    report['selected_runtime_checks_passed'] = args.mode == 'runtime' and not failed and len(runtime_checks) == len(RUNTIME_SUITES) + len(RUNTIME_PROBES)
    write(out / 'result.json', report)
    print(f"{report['status']}: {out / 'result.json'}", flush=True)
    return int(bool(failed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('source-only', 'runtime'), default='source-only')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--python', type=lambda p: Path(p).absolute(), help='candidate interpreter, preserving venv symlink')
    parser.add_argument('--package', type=lambda p: Path(p).resolve(), help='candidate mojolearn package directory')
    parser.add_argument('--binary-root', type=lambda p: Path(p).resolve(), action='append', default=[])
    parser.add_argument('--build-record', type=lambda p: Path(p).resolve(), action='append', default=[],
                        help='development-build result/receipt or DEVELOPMENT_PROVENANCE.json; repeatable')
    parser.add_argument('--timeout', type=int, default=600, help='per-check wall time in seconds')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if args.output is None:
        parser.error('--output is required and must name a new directory')
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    if args.mode == 'runtime' and (args.python is None or args.package is None or not args.build_record):
        parser.error('runtime requires --python, --package and at least one --build-record')
    return verify(args)


if __name__ == '__main__':
    raise SystemExit(main())
