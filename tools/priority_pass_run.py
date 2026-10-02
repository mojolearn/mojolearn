#!/usr/bin/env python3
"""One box, one build: the priority-list pass, each row's
new route against its restore env, output hashes compared.

Measurement overlay (2026-09-30), not a released wheel: mojolearn 0.8.31 in an
isolated venv, Python sources and the touched IDENTICAL bindings replaced from
this tree. Needs gbm-bench taxi and istella npz staged under /root/datasets.
"""
import argparse, datetime, hashlib, json, os, pathlib, shutil, subprocess, sys
P = pathlib.Path
ap = argparse.ArgumentParser(); ap.add_argument('--vendor', choices=['nvidia', 'amd'], required=True)
a = ap.parse_args()
root = P.cwd(); out = P('/root/priority-pass'); out.mkdir(exist_ok=True)
backend, arch = ('cuda', 'sm_89') if a.vendor == 'nvidia' else ('hip', 'gfx942')
env = dict(os.environ, PATH='/root/.pixi/bin:/opt/rocm/bin:' + os.environ['PATH'],
           MOJOLEARN_TARGET_COLUMN=a.vendor, MOJOLEARN_GPU_ARCHS=arch, MOJOLEARN_NUMERIC_MODE='identical',
           MOJOLEARN_COMPILE_JOBS='2', MOJOLEARN_BENCH_INSTALLED='1', PYTHONUNBUFFERED='1')
env.pop('PYTHONPATH', None)
MODULES = ['transformer', 'mamba', 'byte_lm', 'x_decomp', 'x_sequence', 'training', 'linalg']
# (lane, restore env) for the board's algos races; each runs ours with and without it
ROWS = [
    ('cholesky', {'MOJOLEARN_XD_CHOL_SERIAL': '99999'}),
    ('svd', {'MOJOLEARN_LINALG_LEGACY_SIGN': '1'}),
    ('qr', {'MOJOLEARN_LINALG_LEGACY_SIGN': '1'}),
    ('lr-warmup-cosine', {'MOJOLEARN_LR_EXACT_ONLY': '1'}),
    ('clip-grad-norm', {'MOJOLEARN_CLIP_PACKED': '1'}),
]
phase = 'setup'


def stamp(state, **more):
    (out / 'status.json').write_text(json.dumps(dict(state=state, phase=phase, vendor=a.vendor,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **more), indent=2) + '\n')


def run(cmd, log, timeout=3600, check=True, extra=None):
    stamp('running')
    e = dict(env, **(extra or {}))
    with (out / log).open('a') as f:
        f.write('COMMAND ' + json.dumps([str(x) for x in cmd]) + ' ENV ' + json.dumps(extra or {}) + '\n'); f.flush()
        try:
            return subprocess.run([str(x) for x in cmd], cwd=root, env=e, stdout=f, stderr=subprocess.STDOUT,
                                  timeout=timeout, check=check).returncode
        except subprocess.TimeoutExpired:
            f.write('TIMEOUT %d s\n' % timeout)
            return 'timeout'


def sha(p):
    return hashlib.sha256(P(p).read_bytes()).hexdigest()


def built(module):
    c = [p for p in (root / 'python/mojolearn/identical' / ('_mojolearn_%s.so' % module),
                     root / 'python/mojolearn' / ('_mojolearn_%s.so' % module)) if p.exists()]
    if not c:
        raise RuntimeError('no built _mojolearn_%s.so' % module)
    return max(c, key=lambda p: p.stat().st_mtime)


def ours_cell(race_dir):
    """(hash, median_ms) of the ours arm in the race JSON under race_dir."""
    for f in sorted(P(race_dir).rglob('*.json')):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        for c in d.get('cells', []) if isinstance(d, dict) else []:
            if c.get('arm') == 'ours':
                return c.get('hash'), c.get('median_ms')
    return None, None


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
    run([py, '-m', 'pip', 'install', 'torch==2.13.0', '--index-url', 'https://download.pytorch.org/whl/cpu'], 'torch-install.log', 1800)
    run([py, '-m', 'pip', 'freeze'], 'packages.txt')
    native = out / 'native'; native.mkdir(exist_ok=True)
    phase = 'build-linalg-int8-reference'
    for stale in (root / 'python/mojolearn/identical/_mojolearn_linalg.so', root / 'python/mojolearn/_mojolearn_linalg.so'):
        stale.unlink(missing_ok=True)
    run(['bash', 'bindings/build_linalg.sh'], phase + '.log', extra={'MOJOLEARN_MOJO_BUILD_FLAGS': '-D MOJOLEARN_INT8_MMA_REFERENCE=1'})
    lin_ref = native / '_mojolearn_linalg.int8-reference.so'; shutil.move(str(built('linalg')), lin_ref)
    for m in MODULES:
        phase = 'build-' + m
        for stale in (root / 'python/mojolearn/identical' / ('_mojolearn_%s.so' % m), root / 'python/mojolearn' / ('_mojolearn_%s.so' % m)):
            stale.unlink(missing_ok=True)   # build_byte_lm.sh refuses an existing output
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
                'artifacts': {lin_ref.name: sha(lin_ref)},
                'base_distribution': 'mojolearn 0.8.31; Python sources and seven IDENTICAL bindings replaced from head'}
    for m in MODULES:
        name = '_mojolearn_%s.so' % m
        if not (target / name).exists():
            raise RuntimeError('no installed binding to replace at %s' % (target / name))
        shutil.copy2(native / name, target / name); receipts['artifacts'][name] = sha(native / name)
    (out / 'manifest.json').write_text(json.dumps(receipts, indent=2) + '\n')
    env['MOJOLEARN_REPO_COMMIT'] = head
    rcs = {}
    for group in ['reuse', 'refusals', 'lifetime', 'budget']:
        phase = 'session-' + group
        rcs[phase] = run([py, root / 'tools/transformer_session_check.py', '--binding', target / '_mojolearn_transformer.so',
                          '--backend', backend, '--group', group, '--out', out / (phase + '.json')], phase + '.log', 600, False)
    if a.vendor == 'nvidia':
        phase = 'fresh-prefill'
        rcs[phase] = run([py, root / 'tools/transformer_fresh_prefill_check.py'], phase + '.log', 600, False)
    for rep in (1, 2):
        phase = 'neural-priority-%d' % rep
        rcs[phase] = run([py, root / 'tools/neural_experiments.py', '--set', 'priority', '--calls', '10',
                          '--json', out / (phase + '.json')], phase + '.log', 7200, False)
    # gemm-int8: the board's neural race, ours only, tiled plan then the reference plan
    results = {}
    lin = target / '_mojolearn_linalg.so'
    for arm in ('new', 'reference'):
        phase = 'gemm-int8-' + arm
        if arm == 'reference':
            shutil.copy2(lin_ref, lin)
        rcs[phase] = run([py, root / 'tools/bench_board_neural.py', 'race', '--lane', 'gemm-int8', '--shape', 'full',
                          '--arms', 'ours', '--rounds', '5', '--out', out / phase, '--work', out / 'work',
                          '--ours-python', py], phase + '.log', 3600, False)
        results[phase] = ours_cell(out / phase)
    shutil.copy2(native / '_mojolearn_linalg.so', lin)
    # the algos rows
    sys.path.insert(0, str(root / 'tools'))
    import bench_board_algos as bba
    data = out / 'algos-data'
    todo = {}
    for lane, _ in ROWS:
        ds = [d for d in bba.LANES[lane].get('datasets', ('taxi', 'istella'))]
        todo[lane] = ds
    phase = 'algos-prep'
    rcs[phase] = run([py, root / 'tools/bench_board_algos.py', 'prep', '--data', data, '--lanes', ','.join(todo),
                      '--datasets', ','.join(sorted({d for v in todo.values() for d in v}))], phase + '.log', 3600, False)
    for lane, restore in ROWS:
        for ds in todo[lane]:
            for arm, extra in (('new', {}), ('old', restore)):
                phase = 'algos-%s-%s-%s' % (lane, ds, arm)
                rcs[phase] = run([py, root / 'tools/bench_board_algos.py', 'race', '--lane', lane, '--dataset', ds,
                                  '--data', data, '--arms', 'ours', '--rounds', '3', '--out', out / phase,
                                  '--work', out / 'work', '--ours-python', py], phase + '.log', 2700, False, extra)
                results[phase] = ours_cell(out / phase)
            n, o = results['algos-%s-%s-new' % (lane, ds)], results['algos-%s-%s-old' % (lane, ds)]
            results['verdict-%s-%s' % (lane, ds)] = {'same_hash': bool(n[0]) and n[0] == o[0],
                                                     'new_ms': n[1], 'old_ms': o[1]}
            (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    receipts['returncodes'] = rcs; receipts['results'] = results
    (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    (out / 'complete.json').write_text(json.dumps(receipts, indent=2) + '\n')
    phase = 'complete'; stamp('finished')
except Exception as exc:
    stamp('failed', error=str(exc)); raise
