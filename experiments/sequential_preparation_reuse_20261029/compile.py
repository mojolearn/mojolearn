from pathlib import Path
import subprocess,json,time,hashlib,tarfile,platform
root=Path.cwd(); ev=Path('/Users/andrewhendel/mojolearn-evidence/sequential-preparation-reuse-20261029')
fixture=root/'experiments/sequential_preparation_reuse_20261029/model_compile.mojo'
mojo=Path('/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/default/bin/mojo')
cmd=['bash','tools/compile_slot.sh','pixi','run','--manifest-path','/Users/andrewhendel/CascadeProjects/mojolearn/pixi.toml','--locked','--no-install','--executable',str(mojo),'build','-j','8','--emit','shared-lib','--optimization-level','0','-D','MOJOLEARN_NUMERIC_IDENTICAL=1','-D','MOJOLEARN_COLUMN_CPU','-D','MOJOLEARN_AUTOGRAD_HAS_INFERENCE','-I',str(root),'-I',str(root/'bindings'),str(fixture),'-o',str(ev/'model.dylib')]
files=[p for p in root.rglob('*.mojo') if '.git' not in p.parts and '.pixi' not in p.parts]
files += [root/'pixi.toml',root/'pixi.lock',root/'tools/compile_slot.sh']
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(ev/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(ev/'source-envelope.tar.gz','w:gz') as archive:
 for p in files: archive.add(p,arcname=str(p.relative_to(root)))
receipt={'command':cmd,'cwd':str(root),'product_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_count':len(files),'source_scope':'Conservative all present Mojo source plus manifest/lock/slot script; sparse experiments not imported except fixture and retained apple_fast.','compiler_sha256':hashlib.sha256(mojo.read_bytes()).hexdigest(),'machine':platform.platform(),'coverage':'CPU identical actual NativeModelRequest/model_request_execute: fit, stateful fit, validation, predict; both description versions; standalone model_forward. Inference enabled. Compilation only, fixture not executed.'}
start=time.monotonic()
with (ev/'compile.log').open('w') as log: result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
receipt.update(exit_code=result.returncode,elapsed_seconds=round(time.monotonic()-start,2))
if result.returncode==0: receipt['binary_sha256']=hashlib.sha256((ev/'model.dylib').read_bytes()).hexdigest()
(ev/'compile.json').write_text(json.dumps(receipt,indent=2)+'\n');(ev/'compile.exit').write_text(str(result.returncode)+'\n')
print(json.dumps({k:receipt[k] for k in ['exit_code','elapsed_seconds','source_count']}))
raise SystemExit(result.returncode)
