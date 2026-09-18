from pathlib import Path
import shutil,subprocess,json,hashlib,runpy
base=Path(__file__).resolve().parent;source=base/'fresh-host-source-7f5b786ae';stage=base/'fresh-wheel-stage';pkg=stage/'mojolearn'
manifest=runpy.run_path(str(source/'python/mojolearn/host_surface.py'))
receipt=json.loads((base/'fresh-host-build-7f5b786ae/build-receipt.json').read_text())
assert set(receipt['families'])==set(manifest['wheel_families']())
for family,row in receipt['families'].items():
 assert row['exit_code']==0
 assert hashlib.sha256((base/f'fresh-host-build-7f5b786ae/_mojolearn_{family}_host.so').read_bytes()).hexdigest()==row['sha256']
shutil.copytree(base/'wheel-stage',stage,ignore=shutil.ignore_patterns('build','*.egg-info','__pycache__'))
shutil.rmtree(pkg/'host');(pkg/'host').mkdir()
for so in (base/'fresh-host-build-7f5b786ae').glob('*.so'):shutil.copy2(so,pkg/'host'/so.name)
assert len(list((pkg/'host').glob('*.so')))==32
shutil.copy2(pkg/'host/_mojolearn_core_host.so',pkg/'_stage_core.so')
with (base/'fresh-host-runtime-staging.log').open('w') as log:
 subprocess.run(['python3',str(source/'packaging/macos/stage_dylibs.py'),str(pkg/'_stage_core.so'),*map(str,sorted((pkg/'host').glob('*.so'))),'/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/default/lib'],stdout=log,stderr=subprocess.STDOUT,check=True)
(pkg/'_stage_core.so').unlink()
with (base/'fresh-host-wheel-build.log').open('w') as log:
 subprocess.run(['/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/pkg/bin/python','setup.py','bdist_wheel','--dist-dir',str(base/'fresh-dist')],cwd=stage,stdout=log,stderr=subprocess.STDOUT,check=True)
subprocess.run(['/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/test/bin/python','-m','venv',str(base/'fresh-installed-env')],check=True)
with (base/'fresh-host-wheel-install.log').open('w') as log:
 subprocess.run([str(base/'fresh-installed-env/bin/python'),'-m','pip','install',str(next((base/'fresh-dist').glob('*.whl')))],cwd=base,stdout=log,stderr=subprocess.STDOUT,check=True)
with (base/'fresh-host-exports.json').open('w') as log:
 subprocess.run([str(base/'fresh-installed-env/bin/python'),str(base/'audit_installed_exports.py')],cwd=base,stdout=log,check=True)
print('Fresh wheel installed; all 32 host export audits passed',flush=True)
