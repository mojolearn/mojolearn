from pathlib import Path
import shutil,subprocess
base=Path(__file__).resolve().parent;root=Path('/Users/andrewhendel/mojolearn-wt/verifier-broad-scope');stage=base/'wheel-stage';pkg=stage/'mojolearn'
for folder in ('host','.dylibs'):
 shutil.rmtree(pkg/folder);shutil.copytree(base/'fresh-wheel-stage/mojolearn'/folder,pkg/folder,symlinks=True)
for p in (root/'python/mojolearn').glob('*.py'):shutil.copy2(p,pkg/p.name)
shutil.copytree(root/'python/mojolearn/models',pkg/'models',dirs_exist_ok=True)
shutil.copy2(root/'python/mojolearn/verify_reference/table.json',pkg/'verify_reference/table.json')
shutil.copy2(root/'tools/identity_break.py',pkg/'_identity_break.py')
shutil.copy2(root/'tools/identity_trace_diff.py',pkg/'_identity_trace_diff.py')
(pkg/'identity_columns/COMMIT').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True))
for path in (stage/'build',stage/'mojolearn.egg-info'):
 if path.exists():shutil.rmtree(path)
with (base/'candidate-wheel-build.log').open('w') as log:
 subprocess.run(['/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/pkg/bin/python','setup.py','bdist_wheel','--dist-dir',str(base/'dist')],cwd=stage,stdout=log,stderr=subprocess.STDOUT,check=True)
with (base/'candidate-wheel-install.log').open('w') as log:
 subprocess.run([str(base/'installed-env/bin/python'),'-m','pip','install','--no-deps','--force-reinstall',str(next((base/'dist').glob('*.whl')))],cwd=base,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Updated candidate installed',flush=True)
