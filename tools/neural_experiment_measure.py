#!/usr/bin/env python3
"""Run a bounded parent/candidate neural comparison on one dedicated GPU."""
import argparse, datetime, hashlib, json, os, pathlib, shutil, subprocess, sys, tarfile
P=pathlib.Path
ap=argparse.ArgumentParser();ap.add_argument('--vendor',choices=['nvidia','amd'],required=True);a=ap.parse_args()
root=P.cwd();out=P('/root/neural-experiment-results');out.mkdir(exist_ok=True)
patch=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=root);(out/'candidate-source.patch').write_bytes(patch)
parent='f89f43b85';candidate='97b6e3b76';backend='cuda' if a.vendor=='nvidia' else 'hip';arch='sm_89' if a.vendor=='nvidia' else 'gfx942'
env=dict(os.environ,PATH='/root/.pixi/bin:/opt/rocm/bin:'+os.environ['PATH'],MOJOLEARN_TARGET_COLUMN=a.vendor,MOJOLEARN_GPU_ARCHS=arch,MOJOLEARN_NUMERIC_MODE='identical',MOJOLEARN_COMPILE_JOBS='2',MOJOLEARN_BENCH_INSTALLED='1')
env.pop('PYTHONPATH',None)
phase='setup'
def stamp(state,**more):
 (out/'status.json').write_text(json.dumps(dict(state=state,phase=phase,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),vendor=a.vendor,parent=parent,candidate=candidate,**more),indent=2)+'\n')
def run(cmd,log,cwd=root,timeout=3600):
 print(phase,cmd,flush=True);stamp('running')
 with (out/log).open('a') as f:
  subprocess.run([str(x) for x in cmd],cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,check=True)
try:
 if (out/'complete.json').exists():raise RuntimeError('Completed experiment already exists')
 run(['pixi','install'],'pixi.log',timeout=1800)
 pybase=subprocess.check_output(['pixi','run','python3','-c','import sys;print(sys.executable)'],cwd=root,env=env,text=True).strip().splitlines()[-1]
 venv=out/'venv';run([pybase,'-m','venv',venv],'venv.log');py=venv/'bin/python'
 run([py,'-m','pip','install','mojolearn==0.8.31','numpy==2.5.2','scipy==1.18.0'],'dependencies.log',timeout=1800)
 torch=['torch==2.13.0'] if a.vendor=='nvidia' else ['torch==2.13.0+rocm7.1','--index-url','https://download.pytorch.org/whl/rocm7.1']
 run([py,'-m','pip','install',*torch],'torch-install.log',timeout=1800)
 run([py,'-m','pip','freeze'],'packages.txt')
 run([py,'-c','import torch; print(torch.__version__,torch.version.cuda,torch.version.hip); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'],'device.txt')
 par=out/'parent-source';par.mkdir(exist_ok=True)
 archive=out/'parent.tar'
 with archive.open('wb') as f:subprocess.run(['git','archive',parent],cwd=root,stdout=f,check=True)
 with tarfile.open(archive) as t:t.extractall(par,filter='data')
 archive.unlink()
 if not (par/'.pixi').exists():(par/'.pixi').symlink_to(root/'.pixi',target_is_directory=True)
 sources={'candidate':root,'parent':par}
 # Compile the untested candidate first; fail visibly if it does not build.
 for label,src in sources.items():
  for module in ['transformer','byte_lm']:
   phase=label+'-build-'+module
   (src/'python/mojolearn/identical'/('_mojolearn_'+module+'.so')).unlink(missing_ok=True)
   run(['bash','bindings/build_'+module+'.sh'],phase+'.log',cwd=src,timeout=3600)
 site=P(subprocess.check_output([str(py),'-c','import sysconfig; print(sysconfig.get_paths()["purelib"])'],text=True).strip())/'mojolearn'
 lanes=['transformer-forward','lm-forward','samba-forward','lm-train-step','samba-train-step']
 manifest={'parent':parent,'candidate':candidate,'source_patch_sha256':hashlib.sha256(patch).hexdigest(),'base_distribution':'mojolearn 0.8.31; Python sources and two IDENTICAL bindings replaced from the named revision','vendor':a.vendor,'backend':backend,'arch':arch,'lanes':lanes,'rounds':5,'artifacts':{}}
 for label in ['parent','candidate']:
  src=sources[label]
  for p in (src/'python/mojolearn').rglob('*.py'):
   if 'tests' in p.parts:continue
   dst=site/p.relative_to(src/'python/mojolearn');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
  # Remove stale bytecode after replacing Python source in the isolated venv.
  for p in site.rglob('__pycache__'):shutil.rmtree(p)
  manifest['artifacts'][label]={}
  for module in ['transformer','byte_lm']:
   name='_mojolearn_'+module+'.so';built=src/'python/mojolearn/identical'/name
   targets=list((site/backend/arch/'identical').glob(name))
   if len(targets)!=1:raise RuntimeError('Expected exactly one installed '+str(site/backend/arch/'identical'/name))
   shutil.copy2(built,targets[0]);manifest['artifacts'][label][name]=hashlib.sha256(built.read_bytes()).hexdigest()
  (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
  env['MOJOLEARN_REPO_COMMIT']=candidate if label=='candidate' else parent
  if label=='candidate':
   for group in ['reuse','refusals','lifetime','budget']:
    phase='candidate-session-'+group
    run([py,root/'tools/transformer_session_check.py','--binding',site/backend/arch/'identical/_mojolearn_transformer.so','--backend',backend,'--group',group,'--out',out/(phase+'.json')],phase+'.log',timeout=600)
  for lane in lanes:
   phase=label+'-'+lane
   run([py,root/'tools/bench_board_neural.py','race','--lane',lane,'--shape','full','--arms','ours,torch-eager-fp32,torch-compile-fp32','--rounds','5','--out',out/label,'--work',out/'work','--ours-python',py,'--theirs-python',py],phase+'.log',timeout=3600)
 phase='summarize'
 summary={}
 for lane in lanes:
  pair={}
  for label in ['parent','candidate']:
   files=list((out/label).glob(lane+'*.json'))
   if len(files)!=1:raise RuntimeError('Missing or ambiguous result: '+label+' '+lane)
   pair[label]=json.loads(files[0].read_text())
  old=pair['parent']['arms']['ours'];new=pair['candidate']['arms']['ours']
  summary[lane]={'parent_ms':old['median_ms'],'candidate_ms':new['median_ms'],'parent_over_candidate':old['median_ms']/new['median_ms'] if old['median_ms'] and new['median_ms'] else None,'ours_digests_equal':old['digests']==new['digests'],'arms':{k:v['arms'] for k,v in pair.items()},'quality':{k:v['quality'] for k,v in pair.items()}}
 (out/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
 phase='complete';stamp('finished');(out/'complete.json').write_text(json.dumps(manifest,indent=2)+'\n')
except Exception as e:
 stamp('failed',error=str(e));raise
