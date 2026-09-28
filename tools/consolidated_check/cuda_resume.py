#!/usr/bin/env python3
"""Print the pinned CUDA continuation; --execute runs it on an already provided host."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess

MANIFEST = Path(__file__).with_name('cuda_resume_20260928.json')


def load_runs():
    runs = json.loads(MANIFEST.read_text())['runs']
    lanes = [lane for run in runs for lane in run['lanes']]
    if len(lanes) != 445 or len(set(lanes)) != 445:
        raise ValueError('pinned continuation must cover 445 unique lanes exactly once')
    if [len(run['lanes']) for run in runs] != [337, 4, 2, 97, 3, 1, 1]:
        raise ValueError('unexpected pinned lane groups')
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, required=True, help='existing Git clone containing all five commits')
    ap.add_argument('--workspace', type=Path, required=True, help='dedicated worktree/cache/evidence directory')
    ap.add_argument('--gpu-arch', required=True, help='CUDA architecture, e.g. sm_90')
    ap.add_argument('--execute', action='store_true', help='run locally on the already provided CUDA host')
    args = ap.parse_args()
    if not re.fullmatch(r'sm_[0-9]+[a-z]?', args.gpu_arch):
        ap.error('--gpu-arch must be a CUDA architecture such as sm_90')
    repo, workspace = args.repo.resolve(), args.workspace.resolve()
    runs = load_runs()
    print(json.dumps(dict(execute=args.execute, repo=str(repo), workspace=str(workspace),
                         gpu_arch=args.gpu_arch, runs=[dict(name=r['name'], commit=r['commit'],
                         fixtures=r['fixtures'], lanes=len(r['lanes'])) for r in runs]), indent=2), flush=True)
    if not args.execute:
        return 0
    if platform.system() != 'Linux':
        ap.error('execution requires the provided Linux CUDA host')
    pixi = shutil.which('pixi') or str(Path.home() / '.pixi/bin/pixi')
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE='identical', MOJOLEARN_GPU_ARCHS=args.gpu_arch,
               MOJOLEARN_COMPILE_JOBS='2', BUILD_JOBS='2', ARM_TIMEOUT='120', CPU_THREADS='default')
    # Do not inherit binding overrides or CPU-only selectors from a prior task.
    overrides = [k for k in env if k.startswith('MOJOLEARN_') and k not in
                 {'MOJOLEARN_NUMERIC_MODE', 'MOJOLEARN_GPU_ARCHS', 'MOJOLEARN_COMPILE_JOBS'}]
    if overrides:
        ap.error('clear inherited execution overrides before running: ' + ', '.join(overrides))
    subprocess.run(['nvidia-smi', '-L'], check=True)
    caps = subprocess.check_output(['nvidia-smi', '--query-gpu=compute_cap',
                                    '--format=csv,noheader'], text=True).splitlines()
    expected = re.match(r'sm_([0-9]+)', args.gpu_arch)[1]
    if not caps or any(cap.strip().replace('.', '') != expected for cap in caps):
        ap.error('--gpu-arch does not match the provided CUDA hardware')
    for commit in {r['commit'] for r in runs}:
        subprocess.run(['git', '-C', str(repo), 'cat-file', '-e', commit + '^{commit}'], check=True)
    workspace.mkdir(parents=True, exist_ok=True)
    # One invocation owns all these trees; all GPU/CPU arms execute serially.
    with (workspace / 'cuda-resume.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = workspace / ('native-store-' + platform.machine() + '-' + args.gpu_arch)
        prepared = set()
        failed = []
        for run in runs:
            commit = run['commit']
            tree = workspace / ('tree-' + commit[:12])
            if commit not in prepared:
                if not tree.exists():
                    subprocess.run(['git', '-C', str(repo), 'worktree', 'add', '--detach', str(tree), commit], check=True)
                actual = subprocess.check_output(['git', '-C', str(tree), 'rev-parse', 'HEAD'], text=True).strip()
                dirty = subprocess.check_output(['git', '-C', str(tree), 'status', '--porcelain', '--untracked-files=no'], text=True)
                if actual != commit or dirty:
                    raise RuntimeError('existing worktree does not match clean pinned source: ' + str(tree))
                subprocess.run([pixi, 'install', '-e', 'default'], cwd=tree, env=env, check=True)
                # Native objects are copied, never symlinked. The store validates
                # source closures; ordinary build checks still rebuild stale/missing objects.
                subprocess.run([pixi, 'run', '-e', 'default', 'python', 'tools/steward_build.py',
                                'seed', '--root', '.', '--store', str(store)], cwd=tree, env=env, check=True)
                prepared.add(commit)
            output = workspace / 'evidence' / (commit[:12] + '-' + run['name'])
            run_env = dict(env, LANES=','.join(run['lanes']), FIXTURES=run['fixtures'],
                           RUN_RADIX='1' if run['radix'] else '0')
            result = subprocess.run(['bash', 'tools/consolidated_check/mac_job.sh', '0/1', str(output)],
                                    cwd=tree, env=run_env)
            subprocess.run([pixi, 'run', '-e', 'default', 'python', 'tools/steward_build.py',
                            'publish', '--root', '.', '--store', str(store)], cwd=tree, env=env, check=True)
            if result.returncode:
                failed.append(run['name'])
        print(json.dumps(dict(failed=failed, evidence=str(workspace / 'evidence'))), flush=True)
        return bool(failed)


if __name__ == '__main__':
    raise SystemExit(main())
