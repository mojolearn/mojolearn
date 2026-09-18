import hashlib,json,os,runpy,subprocess,time
from pathlib import Path
base=Path(__file__).resolve().parent
source=base/'fresh-host-source-7f5b786ae'
out=base/'fresh-host-build-7f5b786ae';out.mkdir(exist_ok=False)
os.chdir(source)
env=os.environ.copy()
for key in ('MOJOLEARN_GPU_ARCHS','MACOSX_DEPLOYMENT_TARGET'):env.pop(key,None)
mojo='/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/default/bin/mojo'
python='/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/test/bin/python'
sdk=subprocess.check_output(['xcrun','--sdk','macosx','--show-sdk-version'],text=True).strip()
subprocess.run([python,'tokenizer/tools/gen_unicode_categories.py'],check=True,env=env)
manifest=runpy.run_path('python/mojolearn/host_surface.py')
receipt={'source_commit':'7f5b786ae','source_archive_sha256':hashlib.sha256((base/'fresh-host-source-7f5b786ae.tar').read_bytes()).hexdigest(),'toolchain':subprocess.check_output([mojo,'--version'],text=True,env=env).strip(),'families':{}}
for family in manifest['wheel_families']():
 dest=out/f'_mojolearn_{family}_host.so'
 command=[mojo,'build','-j','1','--emit','shared-lib','--target-cpu','apple-m1','-Xlinker','-platform_version','-Xlinker','macos','-Xlinker','11.0','-Xlinker',sdk,'-D','MOJOLEARN_NUMERIC_IDENTICAL=1','-D','MOJOLEARN_COLUMN_CPU','-I','.','-I','bindings',f'bindings/_mojolearn_{family}_host.mojo','-o',str(dest)]
 started=time.monotonic()
 with (out/f'{family}.log').open('w') as log:
  result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=600)
 row={'command':command,'exit_code':result.returncode,'seconds':time.monotonic()-started}
 if dest.exists():row['sha256']=hashlib.sha256(dest.read_bytes()).hexdigest()
 receipt['families'][family]=row
 (out/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(f'{family}: exit {result.returncode}, {row["seconds"]:.1f}s',flush=True)
 if result.returncode:raise SystemExit(result.returncode)
print('All 32 host families freshly built',flush=True)
