#!/usr/bin/env python3
"""Build the toggle-round bindings once and run the EXPERIMENTS.md sweep on one GPU.

Measurement overlay (2026-09-30), not a released wheel: mojolearn 0.8.31
installed in an isolated venv, its Python sources and three IDENTICAL
bindings replaced from this tree. Order: build, session checks, fresh
prefill check (NVIDIA), `neural_experiments.py --set <vendor>`, then the
GEMM plan arms on transformer-forward. Failed logs are kept.
"""
import argparse, datetime, hashlib, json, os, pathlib, shutil, subprocess
P = pathlib.Path
ap = argparse.ArgumentParser()
ap.add_argument('--vendor', choices=['nvidia', 'amd'], required=True)
ap.add_argument('--calls', type=int, default=10)
a = ap.parse_args()
root = P.cwd()
out = P('/root/neural-toggle-sweep'); out.mkdir(exist_ok=True)
backend, arch = ('cuda', 'sm_89') if a.vendor == 'nvidia' else ('hip', 'gfx942')
env = dict(os.environ, PATH='/root/.pixi/bin:/opt/rocm/bin:' + os.environ['PATH'],
           MOJOLEARN_TARGET_COLUMN=a.vendor, MOJOLEARN_GPU_ARCHS=arch,
           MOJOLEARN_NUMERIC_MODE='identical', MOJOLEARN_COMPILE_JOBS='2',
           MOJOLEARN_BENCH_INSTALLED='1', PYTHONUNBUFFERED='1')
env.pop('PYTHONPATH', None)
phase = 'setup'
GEMM_ARMS = 'shipped,tuned128,half,quarter,kpack,kfoldv'


def stamp(state, **more):
    (out / 'status.json').write_text(json.dumps(dict(
        state=state, phase=phase, vendor=a.vendor,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **more), indent=2) + '\n')


def run(cmd, log, timeout=3600):
    stamp('running')
    with (out / log).open('a') as f:
        f.write('COMMAND ' + json.dumps([str(x) for x in cmd]) + '\n'); f.flush()
        subprocess.run([str(x) for x in cmd], cwd=root, env=env, stdout=f,
                       stderr=subprocess.STDOUT, timeout=timeout, check=True)


def sha(p):
    return hashlib.sha256(P(p).read_bytes()).hexdigest()


try:
    if (out / 'complete.json').exists():
        raise RuntimeError('Refusing to repeat a completed sweep')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    (out / 'source.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=root))
    run(['pixi', 'install'], 'pixi.log', 1800)
    base = subprocess.check_output(['pixi', 'run', 'python3', '-c', 'import sys;print(sys.executable)'],
                                   cwd=root, env=env, text=True).strip().splitlines()[-1]
    venv = out / 'venv'
    if not (venv / 'bin/python').exists():
        run([base, '-m', 'venv', venv], 'venv.log')
    py = venv / 'bin/python'
    run([py, '-m', 'pip', 'install', 'mojolearn==0.8.31', 'numpy==2.5.2', 'scipy==1.18.0'], 'packages-install.log', 1800)
    run([py, '-m', 'pip', 'freeze'], 'packages.txt')
    for module in ['transformer', 'mamba', 'byte_lm']:
        phase = 'build-' + module
        run(['bash', 'bindings/build_' + module + '.sh'], phase + '.log', 3600)
    site = P(subprocess.check_output([str(py), '-c', 'import sysconfig;print(sysconfig.get_paths()["purelib"])'],
                                     text=True).strip()) / 'mojolearn'
    for src in (root / 'python/mojolearn').rglob('*.py'):
        dst = site / src.relative_to(root / 'python/mojolearn'); dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for cache in site.rglob('__pycache__'):
        shutil.rmtree(cache)
    target = site / backend / arch / 'identical'
    receipts = {'head': head, 'vendor': a.vendor, 'arch': arch, 'calls': a.calls,
                'source_patch_sha256': sha(out / 'source.patch'), 'artifacts': {},
                'base_distribution': 'mojolearn 0.8.31; Python sources and three IDENTICAL bindings replaced from head'}
    (out / 'native').mkdir(exist_ok=True)
    for module in ['transformer', 'mamba', 'byte_lm']:
        name = '_mojolearn_' + module + '.so'
        built = root / 'python/mojolearn/identical' / name
        if not (target / name).exists():
            raise RuntimeError('no installed binding to replace at ' + str(target / name))
        shutil.copy2(built, target / name); shutil.copy2(built, out / 'native' / name)
        receipts['artifacts'][name] = sha(built)
    (out / 'manifest.json').write_text(json.dumps(receipts, indent=2) + '\n')
    env['MOJOLEARN_REPO_COMMIT'] = head
    for group in ['reuse', 'refusals', 'lifetime', 'budget']:
        phase = 'session-' + group
        run([py, root / 'tools/transformer_session_check.py', '--binding', target / '_mojolearn_transformer.so',
             '--backend', backend, '--group', group, '--out', out / (phase + '.json')], phase + '.log', 600)
    if a.vendor == 'nvidia':
        phase = 'fresh-prefill'
        run([py, root / 'tools/transformer_fresh_prefill_check.py'], phase + '.log', 600)
    phase = 'sweep-' + a.vendor
    run([py, root / 'tools/neural_experiments.py', '--set', a.vendor, '--calls', str(a.calls),
         '--json', out / (phase + '.json')], phase + '.log', 7200)
    phase = 'gemm-arms-transformer-forward'
    run([py, root / 'tools/neural_experiments.py', '--lane', 'transformer-forward', '--only', 'baseline',
         '--gemm-arms', GEMM_ARMS, '--calls', str(a.calls), '--json', out / (phase + '.json')], phase + '.log', 3600)
    (out / 'complete.json').write_text(json.dumps(receipts, indent=2) + '\n')
    phase = 'complete'; stamp('finished')
except Exception as exc:
    stamp('failed', error=str(exc)); raise
