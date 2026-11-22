import hashlib,json,os,pathlib,shutil,signal,subprocess,sys,time,xml.etree.ElementTree as ET
W=pathlib.Path(sys.argv[1]);R=pathlib.Path(__file__).resolve().parent;ROOT=pathlib.Path('/Users/andrewhendel/CascadeProjects/mojolearn');PY=str(ROOT/'.pixi/envs/test/bin/python');COMMIT=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=W)
if not (W/'.pixi').exists():(W/'.pixi').symlink_to(ROOT/'.pixi',target_is_directory=True)
# Dependencies are existing native build bytes. Only the linalg bindings below
# are compiled or exercised by this focused validation.
old=pathlib.Path('/Users/andrewhendel/CascadeProjects/mojolearn-apple-fast-bf16/python/mojolearn')
for folder in ['', 'identical', 'cpu']:
 dest=W/'python/mojolearn'/folder;dest.mkdir(exist_ok=True)
 for src in (old/folder).glob('*.so'):
  if src.name=='_mojolearn_linalg.so':continue
  target=dest/src.name
  if not target.exists():target.symlink_to(src)
env=dict(os.environ,MOJOLEARN_COMPILE_JOBS='1',MOJOLEARN_SKIP_BUILD_GATE='1',PYTHONPATH=str(W/'python'),MOJOLEARN_REQUIRE_FAST_BF16='1',MOJOLEARN_BENCH_SOURCE_COMMIT=COMMIT)
for k in ['MOJOLEARN_BUILD_DEFINES','MOJOLEARN_BUILD_DEFINE_FLAGS','MOJOLEARN_MOJO_BUILD_FLAGS','PYTHONHOME']:env.pop(k,None)
base=R.parent/'apple-fast-bf16-lane/local-full-4096'
probe=R/'probe.py'
probe.write_text("""import sys,json,hashlib,pathlib,numpy as np
from mojolearn import _linalg_impl,linalg,linalg_fast
binding=_linalg_impl._load();variant=sys.argv[1];base=pathlib.Path(sys.argv[2]);out=pathlib.Path(sys.argv[3]);expected=sys.argv[4]
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
record={'variant':variant,'mode':int(binding.linalg_numeric_mode()),'binary':str(pathlib.Path(binding.__file__).resolve()),'binary_sha256':sha(binding.__file__),'fast_bf16_available':bool(binding.fast_bf16_available())}
assert record['binary_sha256']==expected
assert record['mode']==(1 if variant=='identical' else 0)
assert record['fast_bf16_available']==(variant=='fast-default')
if variant=='fast-off':
 try:linalg_fast.matmul_bf16(np.ones((2,2),np.float32),np.ones((2,2),np.uint16))
 except Exception as e:record['disabled_refusal']=str(e)
 else:raise AssertionError('disabled BF16 path accepted a call')
else:
 input_path=base/'input-4096-4096-4096.npz'
 with np.load(input_path,allow_pickle=False) as data:a,b=data['a'],data['b']
 result=np.asarray(linalg.matmul_bf16(a,b));ref=base/('candidate-fast' if variant=='fast-default' else 'released-identical')/'output-4096-4096-4096-0.npy';reference=np.load(ref,mmap_mode='r')
 assert result.shape==reference.shape and result.dtype==reference.dtype
 record.update(input_sha256=sha(input_path),reference_sha256=sha(ref),shape=list(result.shape),bitwise_equal=bool(np.array_equal(result.view(np.uint32),reference.view(np.uint32))),differing_elements=int(np.count_nonzero(result.view(np.uint32)!=reference.view(np.uint32))))
 np.save(out/(variant+'-output.npy'),result)
 record['output_sha256']=sha(out/(variant+'-output.npy'))
 assert record['bitwise_equal'],record
record['status']='passed';(out/(variant+'-probe.json')).write_text(json.dumps(record,indent=2));print(json.dumps(record))
""")
def run(name,cmd,variant_env,limit=300):
 p=R/(name+'.result.json');rec={'started':time.time(),'source_commit':COMMIT,'command':cmd,'numeric_mode':variant_env.get('MOJOLEARN_NUMERIC_MODE'),'defines':variant_env.get('MOJOLEARN_BUILD_DEFINES',''),'status':'running'};p.write_text(json.dumps(rec,indent=2))
 with (R/(name+'.log')).open('xb') as log:
  child=subprocess.Popen(cmd,cwd=W,env=variant_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=child.wait(timeout=limit)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   rc=124
 rec.update(exit_code=rc,elapsed_s=time.time()-rec['started'],status='finished');p.write_text(json.dumps(rec,indent=2))
 if rc:raise SystemExit(rc)
 return rec
binaries={}
for variant,mode,defines in [('fast-default','fast',''),('fast-off','fast','MOJOLEARN_AFN_GEMM_BF16_MMA=0'),('identical','identical','')]:
 ve=dict(env,MOJOLEARN_NUMERIC_MODE=mode)
 if defines:ve['MOJOLEARN_BUILD_DEFINES']=defines
 run(variant+'-compile',['bash','/Users/andrewhendel/mojolearn-evidence/compile_slot.sh','bash','bindings/build_linalg.sh'],ve)
 binary=W/'python/mojolearn'/('identical' if mode=='identical' else '')/'_mojolearn_linalg.so';retained=R/(variant+'.so');shutil.copy2(binary,retained);digest=hashlib.sha256(binary.read_bytes()).hexdigest();binaries[variant]={'sha256':digest,'bytes':binary.stat().st_size,'path':str(retained)}
 run(variant+'-probe',[PY,str(probe),variant,str(base),str(R),digest],ve)
 if variant=='fast-default':
  run('correctness-default',[PY,'-m','pytest','-q','python/mojolearn/tests/test_linalg_fast_bf16.py','--junitxml='+str(R/'correctness-default.xml')],ve)
  suites=ET.parse(R/'correctness-default.xml').getroot().findall('testsuite');counts={key:sum(int(x.get(key,'0')) for x in suites) for key in ['tests','failures','errors','skipped']};assert counts=={'tests':23,'failures':0,'errors':0,'skipped':0},counts
shutil.copy2(R/'fast-default.so',W/'python/mojolearn/_mojolearn_linalg.so')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==COMMIT
summary={'status':'passed','source_commit':COMMIT,'variants':binaries,'correctness':counts,'scope':'default enabled, explicit off, IDENTICAL gates; saved full4096 output equivalence; no new performance measurement or cross-vendor rerace'};(R/'validation.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
