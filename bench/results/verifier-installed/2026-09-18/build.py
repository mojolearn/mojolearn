from pathlib import Path
import hashlib, json, os, shutil, subprocess, zipfile

base = Path(__file__).resolve().parent
root = Path('/Users/andrewhendel/mojolearn-wt/verification-coverage-continuation')
prior = Path('/Users/andrewhendel/mojolearn-evidence/verifier-broad-scope/fresh-wheel-stage/mojolearn')
kernels = Path('/Users/andrewhendel/mojolearn-evidence/next-wheel-coverage/cpu-kernel-identity/clean')
python = '/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/pkg/bin/python'
stage = base / 'stage'
stage.mkdir()
pkg = stage / 'mojolearn'
shutil.copytree(root/'python/mojolearn', pkg, ignore=shutil.ignore_patterns('*.so', '*.dylib', '__pycache__', 'tests', '.dylibs'))
for name in ('setup.py', 'pyproject.toml', 'mojolearn_diagnostics.py'):
    shutil.copy2(root/'python'/name, stage/name)
for name in ('README.md', 'LICENSE', 'NOTICE'):
    shutil.copy2(root/name, stage/name)
shutil.copy2(root/'CITATION.cff', pkg/'CITATION.cff')
shutil.copytree(prior/'host', pkg/'host', dirs_exist_ok=True)
native = {}
for target in sorted((pkg/'host').glob('*.so')):
    source = prior/'host'/target.name
    if target.name in ('_mojolearn_kernel_methods_host.so', '_mojolearn_estimators_host.so'):
        source = kernels/target.name
        shutil.copy2(source, target)
    native[target.name] = {'source': str(source.resolve()), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
for module in ('identity_break', 'identity_trace_diff'):
    shutil.copy2(root/'tools'/f'{module}.py', pkg/f'_{module}.py')
commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip()
(pkg/'identity_columns').mkdir(exist_ok=True)
(pkg/'identity_columns/COMMIT').write_text(commit+'\n')
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MOJOLEARN_CPU_THREADS='1')
env.pop('PYTHONPATH', None)
with (base/'build.log').open('w') as log:
    anchor = pkg/'_stage_anchor.so'
    shutil.copy2(next((pkg/'host').glob('*.so')), anchor)
    subprocess.run([python, str(root/'packaging/macos/stage_dylibs.py'), str(anchor), *map(str, sorted((pkg/'host').glob('*.so'))), '/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/default/lib'], check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    anchor.unlink()
    subprocess.run([python, 'setup.py', 'bdist_wheel', '--dist-dir', str(base/'dist')], cwd=stage, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
wheel = next((base/'dist').glob('*.whl'))
with zipfile.ZipFile(wheel) as z:
    files = {n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.endswith(('.so','.dylib'))}
receipt = dict(scope='Development CPU installed replay only; retained mixed-source native bindings; NOT a fresh expanded release build', source_commit=commit, wheel=str(wheel), wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(), reused_native=native, packaged_native=files)
(base/'wheel-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(wheel)
