#!/usr/bin/env python3
"""Untimed GPU quality diagnostics with separate pinned library/harness sources.

This does not produce timing permission or release admission. Final qualification
must rebuild and rerun against the final common source commit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--harness', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--vendor', required=True, choices=('nvidia', 'amd'))
    p.add_argument('--python', required=True)
    a = p.parse_args()
    if sys.platform != 'linux':
        p.error('Run on the authorized Linux GPU boxes only')
    a.source = a.source.resolve()
    a.harness = a.harness.resolve()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.harness / 'tools'))
    from identical_wave_native_build import command, save

    def source_info(root):
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=root, check=True)
        return {'path': str(root), 'sha': sha}

    report = {'status': 'RUNNING', 'purpose': 'untimed quality diagnostics',
              'source': source_info(a.source), 'harness': source_info(a.harness),
              'vendor': a.vendor, 'builds': {}, 'gates': {}}
    receipt = a.out / 'quality.json'
    save(receipt, report)
    backend, arch, column = ('cuda', 'sm_89', 'NVIDIA') if a.vendor == 'nvidia' else ('hip', 'gfx942', 'AMD')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('MOJOLEARN_', 'MODULAR_MOJO_', 'MOJO_COMPILE_'))}
    env.update(PATH='/root/.pixi/bin:/opt/rocm/bin:' + env.get('PATH', ''),
               PYTHONPATH=str(a.source / 'python'), MOJOLEARN_VENDOR=backend,
               MOJOLEARN_NUMERIC_MODE='identical', MOJOLEARN_TARGET_COLUMN=a.vendor,
               MOJOLEARN_GPU_ARCHS=arch, MOJOLEARN_COMPILE_JOBS='1',
               MOJOLEARN_SKIP_BUILD_GATE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1',
               LD_LIBRARY_PATH=str(a.source / 'python/mojolearn/.libs') + ':' + env.get('LD_LIBRARY_PATH', ''))
    slot = '/root/mojolearn-evidence/compile_slot.sh'
    if not Path(slot).is_file():
        raise RuntimeError('compile semaphore missing')
    for name in ('solver', 'solver_host', 'mixture', 'mixture_host'):
        build_env = dict(env)
        if name.endswith('_host'):
            build_env.pop('MOJOLEARN_GPU_ARCHS')
            build_env['MOJOLEARN_TARGET_COLUMN'] = 'cpu'
            build_env['MOJOLEARN_VENDOR'] = 'cpu'
        report['builds'][name] = command(['bash', slot, 'bash', 'bindings/build_' + name + '.sh'],
                                        a.source, build_env, a.out / ('build-' + name + '.log'), 3600)
        save(receipt, report)

    gates = [
        ('routes', 'mojo', a.harness / 'tools/identical_wave_route_gate.mojo', [], []),
        ('gbdt_depth', 'mojo', a.harness / 'tools/identical_wave_gbdt_depth_gate.mojo', [], []),
        ('small_eigh', 'python', a.harness / 'tools/identical_wave_quality.py', ['small_eigh'], []),
        ('gram_cd', 'python', a.harness / 'tools/identical_wave_quality.py', ['gram_cd'], []),
        ('pca', 'python', a.harness / 'tools/identical_wave_quality.py', ['pca'], []),
        ('gmm', 'mojo', a.source / 'mixture/checks/gmm_check.mojo', [], []),
        ('prophet', 'script', a.source / 'python/mojolearn/tests/test_x_sequence_prophet.py', [], []),
        ('dbscan_split', 'mojo', a.source / 'dbscan/checks/dbscan_edge_split_check.mojo', [], []),
        ('dbscan_scale', 'mojo', a.source / 'dbscan/checks/dbscan_edge_split_scale_check.mojo', [], []),
        ('kde_chunked', 'mojo', a.harness / 'kde/checks/kde_chunked_check.mojo', [], []),
        ('eigh_large', 'python', a.harness / 'tools/identical_wave_quality.py', ['eigh'], []),
        ('dart', 'python', a.harness / 'tools/identical_wave_dart_gate.py',
         ['--data', '/root/board-0833/cache/algos-data/rows-full'], []),
    ]
    report['expected_gates'] = [g[0] for g in gates]
    save(receipt, report)
    for name, kind, path, args, defines in gates:
        row = {'status': 'RUNNING', 'harness_file': str(path),
               'harness_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        report['gates'][name] = row
        save(receipt, report)
        if kind == 'mojo':
            binary = a.out / (name + '.bin')
            build = ['bash', slot, 'pixi', 'run', 'mojo', 'build', '-j', '1',
                     '-I', str(a.source), '-I', str(a.source / 'bindings'),
                     '--target-accelerator', arch, '-D', 'MOJOLEARN_COLUMN_' + column,
                     '-D', 'MOJOLEARN_NUMERIC_IDENTICAL=1']
            for define in defines:
                build += ['-D', define]
            row['compile'] = command(build + [str(path), '-o', str(binary)], a.source, env,
                                     a.out / (name + '-build.log'), 3600)
            if row['compile']['rc']:
                row['status'] = 'COMPILE_FAILED'
                save(receipt, report)
                continue
            argv = [str(binary)]
        else:
            argv = [a.python, str(path), *args]
            if kind == 'python':
                argv += ['--source', str(a.source), '--report', str(a.out / (name + '.json'))]
        row['run'] = command(argv, a.source, env, a.out / (name + '.log'), 7200)
        row['status'] = 'PASS' if row['run']['rc'] == 0 else 'RUN_FAILED'
        save(receipt, report)
        print('QUALITY_GATE', name, row['status'], flush=True)
    report['binary_sha256'] = {str(p.relative_to(a.source)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (a.source / 'python/mojolearn').rglob('*.so')}
    report['status'] = 'PASS' if all(r['rc'] == 0 for r in report['builds'].values()) and all(
        r['status'] == 'PASS' for r in report['gates'].values()) else 'FAILED'
    save(receipt, report)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
