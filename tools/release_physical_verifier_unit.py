#!/usr/bin/env python3
"""Finite physical-lane glue over the existing installed verifier producers.

One current-source one-device observation supplies a provisional reference;
the shipped --par producer owns per-cell physical placement and comparison.
This does not replace release_multi_gpu_parallel or cross-vendor admission.
An outer controller must retain this whole directory on Mac and R2 before
starting its next unit. No background service or automatic retry is installed.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def require(ok, why):
    if not ok:
        raise ValueError(why)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def physical_inventory(rows):
    selected = [r for r in rows if r['index'] in (0, 1)]
    require(len(selected) == 2 and {r['index'] for r in selected} == {0, 1}, 'physical devices 0 and 1 absent')
    require(len({r['uuid'] for r in selected}) == 2 and all(r['uuid'] for r in selected), 'distinct physical UUIDs required')
    require(len({r['pci_bus_id'] for r in selected}) == 2, 'distinct physical PCI devices required')
    return selected


def strict_table(table, lane, fixtures, parts, source=None):
    require(table.get('records'), 'reference lacks original one-device records')
    for fixture in fixtures:
        cell = table.get('cells', {}).get(lane + '/' + fixture, {})
        for part in parts:
            row = cell.get(part, {})
            value = row.get('ref')
            require(isinstance(value, str) and (re.fullmatch('[0-9a-f]{16}', value) or value.startswith('n/a:')) and not row.get('conflict')
                    and row.get('cols') and not any(isinstance(v, list) for v in row['cols'].values()),
                    'missing or conflicting shared reference: ' + '/'.join((lane, fixture, part)))
            for index in row['cols'].values():
                require(type(index) is int and 0 <= index < len(table['records']), 'invalid reference record index')
                require(source is None or table['records'][index].get('commit') == source, 'reference is not this frozen source')


def compare_tables(shared, local, lane, fixtures, parts, source=None):
    strict_table(shared, lane, fixtures, parts, source)
    strict_table(local, lane, fixtures, parts, source)
    for fixture in fixtures:
        for part in parts:
            key = lane + '/' + fixture
            require(shared['cells'][key][part]['ref'] == local['cells'][key][part]['ref'],
                    'current one-device output differs from shared provisional reference: ' + key + '/' + part)


def strict_physical(report, lane, fixtures, backend):
    require(report.get('format') == 'mojolearn.verify-par.v1' and report.get('vendor') == backend,
            'wrong physical producer or vendor')
    require(report.get('devices') == [0, 1] and report.get('lanes') == [lane]
            and set(report.get('fixtures', [])) == set(fixtures), 'physical scope differs')
    cells = report.get('cells', [])
    parts = ('train', 'infer', 'model', 'batch', 'stepfull')
    require(len(cells) == len(fixtures) * len(parts) and
            {(c.get('lane'), c.get('fixture'), c.get('part')) for c in cells}
            == {(lane, fixture, part) for fixture in fixtures for part in parts}
            and all(c.get('verdict') in ('IDENTICAL', 'N/A') for c in cells),
            'incomplete or failed physical cell parts')
    declared = lane == 'par-byte-lm-offload' and report.get('state') == 'NOTHING COMPARED' and report.get('passed') is None
    require(((report.get('state') == 'VERIFIED' and report.get('passed') is True) or declared)
            and report.get('perturbed') is False and not report.get('witness_refusal'), 'physical comparison is not verified or declared inapplicable')
    rows = report.get('cell_witnesses', [])
    require(len(rows) == len(fixtures) and {(r['lane'], r['fixture']) for r in rows}
            == {(lane, f) for f in fixtures} and not any(r.get('witness_refusal') for r in rows),
            'missing, duplicate or refused physical cell witness')
    require(report.get('witness', {}).get('placement') == 'physical', 'process-only evidence is not physical')
    if declared:
        require(report['witness'].get('one_device_by_design') == [lane] and report.get('cells')
                and all(c.get('verdict') == 'N/A' for c in report['cells']), 'one-device declaration is not witnessed')
        return 'declared-one-device-only'
    return 'local-physical-complete'


def bounded(argv, out, name, env, timeout, cwd):
    start = time.time()
    with (out / (name + '.stdout')).open('wb') as stdout, (out / (name + '.stderr')).open('wb') as stderr:
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        timed_out = False
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
    write(out / (name + '.execution.json'), dict(argv=argv, exit_code=code, timeout=timed_out,
          elapsed_seconds=time.time() - start, stdout_sha256=sha(out / (name + '.stdout')),
          stderr_sha256=sha(out / (name + '.stderr'))))
    require(not timed_out, name + ' timed out; preserve partial evidence')
    return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--python', type=Path, required=True)
    p.add_argument('--vendor', choices=('amd', 'nvidia'), required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--lane', required=True)
    p.add_argument('--cpu-threads', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reference-table', type=Path)
    p.add_argument('--provisional-first-vendor', action='store_true')
    p.add_argument('--timeout', type=int, default=900)
    a = p.parse_args()
    root, out = a.root.resolve(), a.out.resolve()
    require(not out.exists(), 'output already exists; no automatic rerun')
    require(bool(a.reference_table) != a.provisional_first_vendor, 'supply shared table or explicitly request provisional first-vendor generation')
    require(not a.provisional_first_vendor or a.vendor == 'nvidia', 'AMD must use the retained shared first-vendor table')
    require(a.cpu_threads > 0 and a.timeout > 0, 'positive resource and time bounds required')
    sys.path.insert(0, str(root / 'tools'))
    import identity_verify_launch as launch
    import identity_columns as columns
    from release_multi_gpu_parallel import driver_inventory
    require(launch.source_state(root) == dict(root=str(root), commit=a.source_commit, dirty_tracked=''), 'wrong or dirty frozen source')
    h = columns.harness()
    require(a.lane in h.LANES and a.lane.startswith('par-'), 'not a physical driver lane')
    profile = 'neural-training' if a.lane in columns.profile_lanes(h, 'neural-training') else 'routine'
    fixtures = list(h.FIXTURES)
    out.mkdir(parents=True)
    shutil.copyfile(__file__, out / 'producer.py')
    command, raw, inventory = driver_inventory(a.vendor)
    write(out / 'driver-inventory.json', dict(command=command, raw=raw, parsed=inventory, selected=physical_inventory(inventory)))
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.update(MOJOLEARN_VENDOR=a.vendor, MOJOLEARN_NUMERIC_MODE='identical', MOJOLEARN_PAR_DEVICES='0',
               MOJOLEARN_VERIFY_CPU_THREADS=str(a.cpu_threads), OMP_NUM_THREADS=str(a.cpu_threads),
               OPENBLAS_NUM_THREADS=str(a.cpu_threads), MOJOLEARN_EXPERIMENT_SAMPLING_KV_UPLOAD_REUSE='0',
               MOJOLEARN_EXPERIMENT_RESIDENT_SAMPLING_FP32='0')
    from release_identity_run import policy_environment
    policy_environment(env)
    for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
        env.pop(key, None)
    env['CUDA_VISIBLE_DEVICES' if a.vendor == 'nvidia' else 'ROCR_VISIBLE_DEVICES'] = '0,1'
    write(out / 'request.json', dict(source_commit=a.source_commit, lane=a.lane, profile=profile, fixtures=fixtures,
          vendor=a.vendor, cpu_threads=a.cpu_threads, timeout=a.timeout,
          environment={k: v for k, v in env.items() if k.startswith(('MOJOLEARN_', 'CUDA_VISIBLE', 'HIP_VISIBLE', 'ROCR_VISIBLE'))},
          provisional_first_vendor=a.provisional_first_vendor, producer_sha256=sha(__file__)))
    (out / 'lane.txt').write_text(a.lane + '\n')
    argv = [str(a.python), str(root / 'tools/identity_verify_launch.py'), '--root', str(root), '--python', str(a.python),
            '--out', str(out / 'one'), '--lanes-file', str(out / 'lane.txt'), '--profile', profile, '--cpu-threads', str(a.cpu_threads)]
    bounded(argv, out, 'one-producer', env, a.timeout, out)
    # Reference mismatches can be old provisional hashes. Conversion still
    # refuses interrupted work, changed installs, invalid native provenance,
    # incomplete cells and unusable output parts; no exit code becomes a pass.
    column = out / 'one.identical.json'
    code = bounded([str(a.python), str(root / 'tools/identity_columns.py'), 'from-verify', str(out / 'one/verify.json'),
          '--harness', str(root / 'tools/identity_break.py'), '--launch', str(out / 'one/verifier-run.json'),
          '--out', str(column)], out, 'convert', env, a.timeout, out)
    require(code == 0, 'one-device conversion failed')
    report = json.loads((out / 'one/verify.json').read_text())
    backend = 'hip' if a.vendor == 'amd' else 'cuda'
    require(report['device']['vendor'] == backend and report['device']['requested_parallel_devices'] == [0], 'one-device route differs')
    vref = columns.vref()
    logs = []
    table = vref.build_table([str(column)], h, str(root), lanes=[a.lane], log=logs.append, parts=vref.PARTS + vref.OPTIONAL_PARTS)
    write(out / 'reference-build-log.json', logs)
    strict_table(table, a.lane, fixtures, vref.PARTS + vref.OPTIONAL_PARTS, a.source_commit)
    vref.write_table(table, out / 'one-candidate-table.json')
    if a.reference_table:
        shared = json.loads(a.reference_table.read_text())
        compare_tables(shared, table, a.lane, fixtures, vref.PARTS + vref.OPTIONAL_PARTS, a.source_commit)
        shutil.copyfile(a.reference_table, out / 'shared-table.json')
    else:
        shutil.copyfile(out / 'one-candidate-table.json', out / 'shared-table.json')
    before = json.loads((out / 'one/verifier-run.json').read_text())
    package = Path(before['install']['package'])
    require(launch.digest(launch.installed_files(package)) == before['install']['manifest_sha256'], 'install changed before physical run')
    env['MOJOLEARN_PAR_DEVICES'] = '0,1'
    code = bounded([str(a.python), '-m', 'mojolearn', 'verify', '--par', 'all', '--lanes', a.lane,
            '--par-devices', '0,1', '--fixtures', ','.join(fixtures), '--repeats', '1',
            '--reference-table', str(out / 'shared-table.json'), '--json'], out, 'physical', env, a.timeout, out)
    require(launch.source_state(root) == before['source'] and launch.digest(launch.installed_files(package))
            == before['install']['manifest_sha256'], 'source or install changed during physical run')
    physical = json.loads((out / 'physical.stdout').read_text())
    status = strict_physical(physical, a.lane, fixtures, backend)
    require(code == 0 or status == 'declared-one-device-only', 'physical producer refused or disagreed')
    write(out / 'summary.json', dict(status=status, lane=a.lane, source=a.source_commit,
          vendor=a.vendor, fixtures=len(fixtures), compared_parts=list(vref.PARTS),
          optional_parts='one-device column retained; existing physical producer compares its registered PARTS only',
          provisional_first_vendor=a.provisional_first_vendor, release_admitted=False,
          registered_parallel_cv_gate='separate required producer',
          source_unchanged=True, install_manifest_sha256=before['install']['manifest_sha256'],
          shared_reference_sha256=sha(out / 'shared-table.json'), physical_report_sha256=sha(out / 'physical.stdout')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
