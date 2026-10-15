#!/usr/bin/env python3
"""Diagnostic test controller: never install, build, publish or claim a wheel pass.

Keep logs out of context: save complete output to files, inspect exit status,
coverage, failures and evidence paths. A filtered excerpt never proves success.
"""
import argparse
import base64
import csv
import fcntl
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback

FREEZE = 'f63ef2235d8e84339ffafb16c973610a03dcf85a'
CHECKER = 'b2ad3802db4e011413ac3c613791d5510c99aa211ae5e51766d7415caa9af6fe'
FIXTURE = 'd2962b249671027d8fd896078ac9c91f25bdce40e443dca49ae05ef61a89311a'
VENDORS = {'nvidia': ('cuda', 'cuda_native', 'sm_89', 'mojolearn-nvidia'),
           'amd': ('hip', 'hip_native', 'gfx942', 'mojolearn-amd')}
# The compiler environment's libc/loader, host C++ ABI and GPU driver are NOT
# a deployable runtime. Resolve these against the GPU machine's own installation.
HOST_LIB = re.compile(r'^(ld[-.]|lib(?:c|m|dl|pthread|rt|resolv|util|anl|nss_[^.]+|stdc\+\+|gcc_s|cuda|nvidia[^.]*|amdhip64|hsa-runtime64|drm[^.]*)\.so)')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def required(condition, message):
    if not condition:
        raise ValueError(message)


def runtime_allowed(row):
    name = Path(row['retained']).name
    source = Path(row['path'])
    return (not HOST_LIB.match(name) and '.so' in name and
            not str(source).startswith(('/usr/lib/', '/lib/', '/lib64/', '/usr/lib64/')))


def check_boundary(ack_path, vendor, seconds, proc_root=Path('/proc')):
    ack = read(ack_path)
    required(ack.get('authorized_by') == 'root' and ack.get('vendor') == vendor,
             'fresh root boundary acknowledgement required')
    required(0 <= time.time() - ack['created_epoch'] <= 300, 'boundary acknowledgement expired')
    required(ack.get('native_commit') == FREEZE, 'boundary acknowledges a different candidate')
    campaign = Path(ack['campaign']).resolve()
    state = read(campaign)
    required((campaign.parent / 'PAUSE').is_file(), 'measurement PAUSE marker missing')
    required(state.get('status') == 'paused between cells', 'measurement is not paused between cells')
    required(state.get('vendor') == vendor, 'measurement vendor differs')
    required(time.time() - state.get('heartbeat', 0) < 30, 'measurement pause heartbeat is stale')
    required(time.time() + seconds + 900 < state['provider_deadline'], 'insufficient window plus 900s retention reserve')
    sessions = []
    for path in sorted(campaign.parent.glob('measured/*/*-session/session.json')):
        item = read(path)
        required(item.get('status') == 'finished' and item.get('remaining_workers') == [],
                 'unfinished measurement session: ' + str(path))
        sessions.append(dict(path=str(path), session=item['session'], sha256=sha(path)))
    # Retain only the relevant marker, never unrelated /proc environment secrets.
    markers = {(b'MOJOLEARN_BOARD_SESSION=' + s['session'].encode()) for s in sessions}
    workers = []
    for p in proc_root.iterdir():
        if p.name.isdigit():
            try:
                if markers.intersection((p / 'environ').read_bytes().split(b'\0')):
                    workers.append(int(p.name))
            except (FileNotFoundError, ProcessLookupError):
                pass
    required(not workers, 'measurement workers still active: ' + str(workers))
    return dict(acknowledgement=ack, campaign=state, sessions=sessions, checked_epoch=time.time())


def verify_candidate(artifact, vendor):
    _, _, arch, _ = VENDORS[vendor]
    config, source, execution, binary, linkage = [read(artifact / n) for n in
        ('build-config.json', 'source.json', 'execution.json', 'binary.json', 'linkage.json')]
    required(config.get('source_commit') == source.get('source_commit') == FREEZE, 'wrong native source freeze')
    required(not source.get('source_working_tree_changed'), 'candidate native source was dirty')
    required(config.get('name') == '_mojolearn_gbdt' and config.get('tier') == 'identical' and
             config.get('arch') == arch, 'candidate binding/tier/target mismatch')
    expected_flags = dict(MOJOLEARN_NUMERIC_MODE='identical', MOJOLEARN_TARGET_COLUMN=vendor,
                          MOJOLEARN_GPU_ARCHS=arch, MOJOLEARN_LINUX_CPU='x86-64-v3', MOJOLEARN_BUILD_DEFINES='')
    required(all(config.get('environment', {}).get(k) == v for k, v in expected_flags.items()),
             'candidate canonical flags differ from reviewed freeze')
    required(execution.get('complete') and execution.get('exit_status') == 0 and
             execution.get('verdict') == 'PASS', 'candidate compile did not pass')
    required(linkage['ldd']['exit_status'] == 0 and not linkage['ldd']['unresolved'], 'incomplete retained runtime closure')
    path = artifact / '_mojolearn_gbdt.so'
    required(sha(path) == binary['sha256'], 'candidate binary hash mismatch')
    required(sha(artifact / 'source.tar.gz') == source['archive_sha256'], 'candidate source archive hash mismatch')
    for row in linkage['ldd']['runtime']:
        retained = (artifact / row['retained']).resolve()
        required(retained.is_relative_to(artifact.resolve()), 'runtime artifact escapes candidate directory')
        required(sha(retained) == row['sha256'], 'retained runtime hash mismatch: ' + str(retained))
    return dict(config=config, source=source, execution=execution, binary=binary, linkage=linkage)


def package_overlay(out, vendor, candidate, manifest_path):
    """Copy all package bytes and adjacent plugin metadata; verify wheel RECORDs."""
    _, directory, arch, plugin = VENDORS[vendor]
    manifest = read(manifest_path)
    required(manifest.get('all_digests_match') is True, 'released-wheel verification missing')
    all_dists = [d for d in metadata.distributions() if
                 d.metadata['Name'].lower().replace('_', '-') in ('mojolearn', 'mojolearn-nvidia', 'mojolearn-amd')]
    dists = {d.metadata['Name'].lower().replace('_', '-'): d for d in all_dists}
    required('mojolearn' in dists and plugin in dists, 'installed core or vendor plugin missing')
    required(all(d.version == '0.8.45' for d in dists.values()), 'installed overlay base must be 0.8.45')
    root = Path(dists['mojolearn'].locate_file('')).resolve()
    site = out / 'overlay-site'
    site.mkdir()
    verified = []
    for name, dist in dists.items():
        required(Path(dist.locate_file('')).resolve() == root, 'plugin metadata must be adjacent to core')
        wheel = next((r for r in manifest['files'] if r['distribution'] == name), None)
        required(wheel and wheel.get('pypi_digest_match') is True, 'missing verified wheel: ' + name)
        required(sha(wheel['path']) == wheel['sha256'] == wheel['pypi_sha256'], 'wheel bytes changed: ' + name)
        checked = 0
        for row in csv.reader((dist.read_text('RECORD') or '').splitlines()):
            if not row[1]:
                continue
            algorithm, encoded = row[1].split('=', 1)
            required(algorithm == 'sha256', 'unsupported installed RECORD digest')
            path = Path(dist.locate_file(row[0]))
            expected = base64.urlsafe_b64decode(encoded + '=' * (-len(encoded) % 4)).hex()
            required(sha(path) == expected, 'installed wheel bytes changed: ' + str(path))
            checked += 1
        required(checked > 0, 'empty installed RECORD: ' + name)
        info = Path(dist._path)
        shutil.copytree(info, site / info.name)
        verified.append(dict(distribution=name, version=dist.version, record_files_verified=checked,
                             wheel_sha256=wheel['sha256'], metadata=str(info)))
    shutil.copytree(root / 'mojolearn', site / 'mojolearn', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    before = {str(p.relative_to(site)): sha(p) for p in site.rglob('*') if p.is_file()}
    rel = Path('mojolearn') / directory / arch / 'identical/_mojolearn_gbdt.so'
    target = site / rel
    required(target.is_file(), 'installed vendor GBDT slot is missing: ' + str(rel))
    shutil.copy2(candidate, target)
    after = {str(p.relative_to(site)): sha(p) for p in site.rglob('*') if p.is_file()}
    required({k for k in after if before.get(k) != after[k]} == {str(rel)}, 'overlay changed more than identical GBDT')
    write(out / 'overlay-manifest.json', dict(base_distributions=verified, installed_site=str(root),
          before=before, after=after, changed=[str(rel)], base_version='0.8.45',
          qualification='raw candidate overlay; NOT a released 0.8.45 execution'))
    return site, target


def compare(left_summary, right_summary):
    """Compare raw array payloads and every serialized model member, not ZIP bytes."""
    import numpy as np
    left_summary, right_summary = Path(left_summary), Path(right_summary)
    left, right = read(left_summary), read(right_summary)
    result = dict(verdict='FAIL', left=str(left_summary), right=str(right_summary), compared=[], failures=[])
    for label, report in [('left', left), ('right', right)]:
        if not report.get('complete') or report.get('exit_status') != 0 or report.get('failures'):
            result['failures'].append(label + ': numerical run incomplete or failed')
        if report.get('numeric_mode') != 'identical' or report.get('data_sha256') != FIXTURE:
            result['failures'].append(label + ': fixture or numeric mode mismatch')
    a = {(c['loss'], c['tag']): c for c in left['cases']}
    b = {(c['loss'], c['tag']): c for c in right['cases']}
    if len(a) != 24 or set(a) != set(b) or len(b) != 24:
        result['failures'].append('expected 24 identical covered cases on both vendors')
    for key in sorted(a.keys() & b.keys()):
        first, second = a[key], b[key]
        if first.get('params') != second.get('params'):
            result['failures'].append(str(key) + ': settings differ')
        if 'expected_refusal' in first or 'expected_refusal' in second:
            if not all('requires Newton or Gradient' in row.get('expected_refusal', '') for row in (first, second)):
                result['failures'].append(str(key) + ': expected refusal differs')
            continue
        if set(first['outputs']) != set(second['outputs']):
            result['failures'].append(str(key) + ': missing output coverage')
        for name in sorted(set(first['outputs']) & set(second['outputs'])):
            paths = []
            for summary, case in [(left_summary, first), (right_summary, second)]:
                item = case['outputs'][name]
                # References are copied intact; original absolute paths are
                # provenance, while bytes must be adjacent to this summary.
                path = summary.parent / Path(item['path']).name
                required(sha(path) == item['sha256'], 'output artifact changed: ' + str(path))
                paths.append(path)
            values = [np.load(p, allow_pickle=False) for p in paths]
            try:
                aa, bb = values
                if name == 'model':
                    if set(aa.files) != set(bb.files):
                        result['failures'].append(str(key) + ': model member coverage differs')
                    pairs = [(member, aa[member], bb[member]) for member in sorted(set(aa.files) & set(bb.files))]
                else:
                    pairs = [(name, aa, bb)]
                for member, x, y in pairs:
                    equal = x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
                    result['compared'].append(dict(loss=key[0], tag=key[1], output=name, member=member,
                        dtype=str(x.dtype), shape=list(x.shape), left_payload_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                        right_payload_sha256=hashlib.sha256(y.tobytes()).hexdigest(), equal=equal))
                    if not equal:
                        result['failures'].append(str(key) + ': ' + name + '/' + member + ' differs')
            finally:
                for value in values:
                    if hasattr(value, 'close'):
                        value.close()
    result['verdict'] = 'PASS' if not result['failures'] else 'FAIL'
    return result


def worker(config_path):
    """Fresh child only: actual imported binding and process mappings are evidence."""
    config = read(config_path)
    out = Path(config['out'])
    import mojolearn
    from mojolearn import _mojolearn_gbdt as binding
    path = Path(binding.__file__).resolve()
    loaded = dict(package_file=mojolearn.__file__, binary=str(path), sha256=sha(path),
                  vendor=binding.gbdt_vendor(), numeric_mode=binding.gbdt_numeric_mode(),
                  native_commit=FREEZE, base_distribution_version='0.8.45',
                  qualification='raw candidate overlay; not a released-wheel result')
    write(out / 'loaded-binding.json', loaded)
    required(path == Path(config['binary']).resolve() and loaded['sha256'] == config['binary_sha256'], 'loaded wrong candidate binary')
    required(loaded['vendor'] == VENDORS[config['vendor']][0] and loaded['numeric_mode'] == 1, 'loaded wrong vendor or mode')
    import importlib.util
    spec = importlib.util.spec_from_file_location('querywise_checker', config['checker'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = [config['checker'], '--data', config['fixture'], '--out', str(out / 'numerical'),
                '--vendor', config['vendor'], '--numeric-mode', 'identical']
    try:
        return module.main()
    finally:
        maps = Path('/proc/self/maps').read_text()
        (out / 'loaded-maps.txt').write_text(maps)
        files = sorted({line.split(None, 5)[-1] for line in maps.splitlines() if len(line.split(None, 5)) == 6
                        and line.split(None, 5)[-1].startswith('/') and '.so' in line.split(None, 5)[-1]})
        write(out / 'loaded-library-hashes.json', [dict(path=p, sha256=sha(p)) for p in files if Path(p).is_file()])


def run(args):
    required(sys.platform == 'linux', 'run only on the Linux GPU worker')
    required(1 <= args.seconds <= 600, 'diagnostic duration must be 1..600 seconds')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    state = dict(complete=False, exit_status=None, native_commit=FREEZE, vendor=args.vendor,
                 qualification='raw diagnostic candidate; not release or performance qualification', failures=[])
    write(out / 'validation.json', state)
    try:
        artifact = Path(args.artifact).resolve()
        candidate = verify_candidate(artifact, args.vendor)
        write(out / 'candidate-provenance.json', candidate)
        required(sha(args.fixture) == FIXTURE, 'wrong querywise fixture bytes')
        required(sha(args.checker) == CHECKER, 'wrong frozen numerical checker bytes')
        required(args.reference_native_commit == FREEZE, 'reference native commit not acknowledged')
        reference_provenance = read(args.reference_provenance)
        required(reference_provenance['native_commit'] == reference_provenance['compile']['commit'] == FREEZE,
                 'reference compile source mismatch')
        required(reference_provenance['compile']['exit_code'] == 0 and
                 reference_provenance['compile']['binary_sha256'] == reference_provenance['loaded_binding']['sha256'],
                 'reference loaded binary not bound to successful compile')
        required(reference_provenance['loaded_binding']['mode'] == 1 and reference_provenance['loaded_binding']['vendor'] == 'metal',
                 'reference is not Apple identical')
        required(reference_provenance['summary_sha256'] == sha(args.reference), 'reference summary changed')
        required(reference_provenance['checker_sha256'] == CHECKER, 'reference checker differs')
        write(out / 'reference-provenance.json', reference_provenance)
        shutil.copytree(artifact, out / 'retained-build')
        shutil.copytree(Path(args.reference).parent, out / 'reference')
        shutil.copy2(args.fixture, out / 'querywise-fixture.npz')
        shutil.copy2(args.checker, out / 'querywise-checker.py')
        shutil.copy2(__file__, out / 'controller.py')
        shutil.copy2(args.wheel_manifest, out / 'base-wheel-manifest.json')
        shutil.copy2(args.boundary_ack, out / 'boundary-ack.json')
        site, binary = package_overlay(out, args.vendor, artifact / '_mojolearn_gbdt.so', args.wheel_manifest)
        runtime = out / 'candidate-runtime'; runtime.mkdir()
        chosen = []
        for row in candidate['linkage']['ldd']['runtime']:
            include = runtime_allowed(row)
            if include:
                shutil.copy2(artifact / row['retained'], runtime / Path(row['retained']).name)
            chosen.append(dict(row, selected=include, reason='compiler runtime' if include else 'resolve against worker system/driver'))
        write(out / 'runtime-selection.json', chosen)
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONEXECUTABLE', 'MOJO_PYTHON_LIBRARY', 'LD_PRELOAD', 'LD_LIBRARY_PATH')}
        released_runtime = out / 'released-runtime'; released_runtime.mkdir()
        for libdir in sorted(p for p in site.rglob('.libs') if p.is_dir()):
            for path in sorted(libdir.iterdir()):
                if path.is_file() and '.so' in path.name and not HOST_LIB.match(path.name):
                    dest = released_runtime / path.name
                    if dest.exists():
                        required(sha(dest) == sha(path), 'released runtime basename collision: ' + path.name)
                    else:
                        shutil.copy2(path, dest)
        released_libs = [str(released_runtime)]
        inherited_library_path = os.environ.get('LD_LIBRARY_PATH', '')
        env.update(PYTHONPATH=str(site), LD_LIBRARY_PATH=':'.join([str(runtime), *released_libs] + ([inherited_library_path] if inherited_library_path else [])),
                   MOJOLEARN_NUMERIC_MODE='identical', MOJOLEARN_VENDOR=VENDORS[args.vendor][0],
                   MOJOLEARN_GPU_ARCH=VENDORS[args.vendor][2], PYTHONDONTWRITEBYTECODE='1')
        if args.vendor == 'amd':
            env.setdefault('ROCM_HOME', '/opt/rocm'); env.setdefault('HIP_PATH', '/opt/rocm')
        write(out / 'execution-environment.json', {k: env[k] for k in ('PYTHONPATH', 'LD_LIBRARY_PATH', 'MOJOLEARN_NUMERIC_MODE', 'MOJOLEARN_VENDOR', 'MOJOLEARN_GPU_ARCH')})
        for tag, cmd in [('candidate-ldd', ['ldd', str(binary)]), ('candidate-versions', ['readelf', '--version-info', str(binary)])]:
            with (out / (tag + '.log')).open('w') as log:
                proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=30)
            (out / (tag + '.exit')).write_text(str(proc.returncode) + '\n')
            required(proc.returncode == 0, tag + ' failed')
        required('not found' not in (out / 'candidate-ldd.log').read_text(), 'candidate runtime cannot resolve on this box')
        # This separate lock excludes duplicate diagnostics without contending
        # with the campaign's intentionally held measurement-owner.lock.
        lock = Path(args.boundary_ack).parent.joinpath('candidate-validation.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        boundary = check_boundary(args.boundary_ack, args.vendor, args.seconds)
        write(out / 'boundary-check.json', boundary)
        cfg = dict(out=str(out), binary=str(binary), binary_sha256=sha(binary),
                   vendor=args.vendor, checker=str(out / 'querywise-checker.py'), fixture=str(out / 'querywise-fixture.npz'))
        write(out / 'worker-config.json', cfg)
        cmd = [sys.executable, str(out / 'controller.py'), 'worker', str(out / 'worker-config.json')]
        state['command'] = cmd; write(out / 'validation.json', state)
        proc = None
        try:
            with (out / 'numerical.log').open('wb') as log:
                proc = subprocess.Popen(cmd, env=env, cwd=out, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + args.seconds
                while proc.poll() is None:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('candidate numerical timeout')
                    campaign = Path(boundary['acknowledgement']['campaign'])
                    if not (campaign.parent / 'PAUSE').exists() or read(campaign).get('status') != 'paused between cells':
                        raise RuntimeError('measurement pause removed during diagnostics')
                    time.sleep(.2)
        finally:
            if proc is not None:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                state['numerical_exit_status'] = proc.returncode
                (out / 'numerical.exit').write_text(str(proc.returncode) + '\n')
        required(proc.returncode == 0, 'candidate numerical checker failed')
        numerical = read(out / 'numerical/summary.json')
        required(numerical.get('complete') and len(numerical['cases']) == 24 and not numerical['failures'], 'incomplete numerical coverage')
        result = compare(out / 'reference/summary.json', out / 'numerical/summary.json')
        write(out / 'crossvendor-comparison.json', result)
        state['comparison'] = dict(verdict=result['verdict'], members=len(result['compared']), failures=result['failures'])
        required(result['verdict'] == 'PASS', 'cross-vendor output/model bits differ')
        state.update(complete=True, exit_status=0, cases=24)
    except Exception:
        state['failures'].append(traceback.format_exc())
        state['exit_status'] = 1
        (out / 'controller-error.log').write_text(state['failures'][-1])
    finally:
        state['finished_epoch'] = time.time()
        write(out / 'validation.json', state)
        write(out / 'evidence-manifest.json', [dict(path=str(p.relative_to(out)), sha256=sha(p), bytes=p.stat().st_size)
              for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'evidence-manifest.json'])
    print(json.dumps(dict(complete=state['complete'], exit_status=state['exit_status'], evidence=str(out / 'validation.json'))))
    return state['exit_status']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    worker_parser = sub.add_parser('worker'); worker_parser.add_argument('config')
    comp = sub.add_parser('compare'); comp.add_argument('--left', required=True); comp.add_argument('--right', required=True); comp.add_argument('--out', required=True)
    r = sub.add_parser('run')
    r.add_argument('--vendor', choices=VENDORS, required=True)
    for name in ('artifact', 'fixture', 'checker', 'reference', 'reference-native-commit', 'reference-provenance', 'boundary-ack', 'out'):
        r.add_argument('--' + name, required=True)
    r.add_argument('--wheel-manifest', default='/root/gpu-run/setup/wheel-manifest.json')
    r.add_argument('--seconds', type=int, default=90)
    args = p.parse_args()
    if args.action == 'worker':
        return worker(args.config)
    if args.action == 'compare':
        result = compare(args.left, args.right); write(args.out, result)
        print(json.dumps(dict(verdict=result['verdict'], compared=len(result['compared']), failures=len(result['failures']))))
        return int(result['verdict'] != 'PASS')
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
