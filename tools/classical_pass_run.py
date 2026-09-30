#!/usr/bin/env python3
"""One box, one build: the neural toggle round (stage reset now off by default)
and the classical pass (LU, SGD, LARS, IVF), each old path vs new path.

Measurement overlay (2026-09-30), not a released wheel: mojolearn 0.8.31 in an
isolated venv, Python sources and six IDENTICAL bindings replaced from this tree.
"""
import argparse, datetime, hashlib, json, os, pathlib, shutil, subprocess
P = pathlib.Path
ap = argparse.ArgumentParser(); ap.add_argument('--vendor', choices=['nvidia', 'amd'], required=True)
a = ap.parse_args()
root = P.cwd(); out = P('/root/classical-pass'); out.mkdir(exist_ok=True)
backend, arch = ('cuda', 'sm_89') if a.vendor == 'nvidia' else ('hip', 'gfx942')
env = dict(os.environ, PATH='/root/.pixi/bin:/opt/rocm/bin:' + os.environ['PATH'],
           MOJOLEARN_TARGET_COLUMN=a.vendor, MOJOLEARN_GPU_ARCHS=arch, MOJOLEARN_NUMERIC_MODE='identical',
           MOJOLEARN_COMPILE_JOBS='2', MOJOLEARN_BENCH_INSTALLED='1', PYTHONUNBUFFERED='1')
env.pop('PYTHONPATH', None)
MODULES = ['transformer', 'mamba', 'byte_lm', 'x_decomp', 'x_linear', 'ivf']
phase = 'setup'


def stamp(state, **more):
    (out / 'status.json').write_text(json.dumps(dict(state=state, phase=phase, vendor=a.vendor,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **more), indent=2) + '\n')


def run(cmd, log, timeout=3600, check=True):
    stamp('running')
    with (out / log).open('a') as f:
        f.write('COMMAND ' + json.dumps([str(x) for x in cmd]) + '\n'); f.flush()
        return subprocess.run([str(x) for x in cmd], cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT,
                              timeout=timeout, check=check).returncode


def sha(p):
    return hashlib.sha256(P(p).read_bytes()).hexdigest()


def built(module):
    c = [p for p in (root / 'python/mojolearn/identical' / ('_mojolearn_%s.so' % module),
                     root / 'python/mojolearn' / ('_mojolearn_%s.so' % module)) if p.exists()]
    if not c:
        raise RuntimeError('no built _mojolearn_%s.so' % module)
    return max(c, key=lambda p: p.stat().st_mtime)


try:
    if (out / 'complete.json').exists():
        raise RuntimeError('Refusing to repeat a completed run')
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
    # torch only generates the Mamba weights (mamba/corpus/gen_corpus.py) on the CPU
    run([py, '-m', 'pip', 'install', 'torch==2.13.0', '--index-url', 'https://download.pytorch.org/whl/cpu'], 'torch-install.log', 1800)
    run([py, '-m', 'pip', 'freeze'], 'packages.txt')
    native = out / 'native'; native.mkdir(exist_ok=True)
    phase = 'build-ivf-scan-off'
    env['MOJOLEARN_BUILD_EXTRA_DEFINES'] = '-D MOJOLEARN_IVF_IDENTICAL_SCAN_OFF'
    run(['bash', 'bindings/build_ivf.sh'], phase + '.log')
    env.pop('MOJOLEARN_BUILD_EXTRA_DEFINES')
    ivf_off = native / '_mojolearn_ivf.scan-off.so'; shutil.move(str(built('ivf')), ivf_off)
    for m in MODULES:
        phase = 'build-' + m
        run(['bash', 'bindings/build_%s.sh' % m], phase + '.log')
        shutil.copy2(built(m), native / ('_mojolearn_%s.so' % m))
    site = P(subprocess.check_output([str(py), '-c', 'import sysconfig;print(sysconfig.get_paths()["purelib"])'],
                                     text=True).strip()) / 'mojolearn'
    for src in (root / 'python/mojolearn').rglob('*.py'):
        dst = site / src.relative_to(root / 'python/mojolearn'); dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for cache in site.rglob('__pycache__'):
        shutil.rmtree(cache)
    target = site / backend / arch / 'identical'
    receipts = {'head': head, 'vendor': a.vendor, 'arch': arch, 'source_patch_sha256': sha(out / 'source.patch'),
                'artifacts': {'_mojolearn_ivf.scan-off.so': sha(ivf_off)},
                'base_distribution': 'mojolearn 0.8.31; Python sources and six IDENTICAL bindings replaced from head'}
    for m in MODULES:
        name = '_mojolearn_%s.so' % m
        if not (target / name).exists():
            raise RuntimeError('no installed binding to replace at %s' % (target / name))
        shutil.copy2(native / name, target / name); receipts['artifacts'][name] = sha(native / name)
    (out / 'manifest.json').write_text(json.dumps(receipts, indent=2) + '\n')
    env['MOJOLEARN_REPO_COMMIT'] = head
    results = {}
    for group in ['reuse', 'refusals', 'lifetime', 'budget']:
        phase = 'session-' + group
        results[phase] = run([py, root / 'tools/transformer_session_check.py', '--binding', target / '_mojolearn_transformer.so',
                              '--backend', backend, '--group', group, '--out', out / (phase + '.json')], phase + '.log', 600, False)
    if a.vendor == 'nvidia':
        phase = 'fresh-prefill'
        results[phase] = run([py, root / 'tools/transformer_fresh_prefill_check.py'], phase + '.log', 600, False)
    phase = 'neural-sweep'
    results[phase] = run([py, root / 'tools/neural_experiments.py', '--set', a.vendor, '--calls', '10',
                          '--json', out / 'neural-sweep.json'], phase + '.log', 7200, False)
    phase = 'classical-ab'
    env['CLASSICAL_IVF_SO'] = str(target / '_mojolearn_ivf.so')
    results[phase] = run([py, root / 'tools/classical_pass_ab.py', 'all', '--out', out / 'classical',
                          '--ivf-off-so', ivf_off], phase + '.log', 6 * 3600, False)
    receipts['returncodes'] = results
    (out / 'complete.json').write_text(json.dumps(receipts, indent=2) + '\n')
    phase = 'complete'; stamp('finished', returncodes=results)
except Exception as exc:
    stamp('failed', error=str(exc)); raise
