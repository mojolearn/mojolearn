"""Operational boundary check with mocked SSH; never runs a measurement."""
import ast,datetime,hashlib,json,pathlib,shlex,tempfile,types
source=pathlib.Path(__file__).with_name('after-tail-ets-repair.py')
tree=ast.parse(source.read_text());body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.Try))]
for already_done in [False,True]:
 with tempfile.TemporaryDirectory() as folder:
  root=pathlib.Path(folder);E=root/'nvidia';E.mkdir();events=[]
  (E/'default-capture/artifacts').mkdir(parents=True)
  (E/'default-capture/manager-status.json').write_text(json.dumps({'status':'MANAGING'}))
  capture=E/'default-capture/artifacts/status.json';capture.write_text(json.dumps({'phase':'GPU_OPPONENTS_RUNNING'}))
  (root/'opponent-publisher-config.json').write_text(json.dumps({'sources':[{'name':'amd','keep':'unchanged'},{'name':'nvidia-default'}]}))
  patch=E/'repair.patch';patch.write_bytes(b'patch')
  plan={'deadline':999,'owner_worker':21356,'patch':str(patch),'patch_sha256':hashlib.sha256(b'patch').hexdigest(),'commit':'fix'}
  def sleep(seconds):
   events.append(('wait',seconds));capture.write_text(json.dumps({'phase':'WAITING_FOR_NEXT_MEASUREMENTS'}))
  def run(argv,**kwargs):
   if argv[0]=='ssh':
    command=argv[-1];events.append(('ssh',command))
    if "ets_status" in command:
     data={'worker':{'pid':21356,'phase':'WAITING_FOR_NEXT_MEASUREMENTS'},'done':True,'sealed':True,'ets_status':'done' if already_done else 'failed'}
     return types.SimpleNamespace(stdout=json.dumps(data).encode(),returncode=0)
   return types.SimpleNamespace(stdout=b'',returncode=0)
  ns={'E':E,'PLAN':plan,'SSH':['ssh','owned-default'],'STATE':E/'ets-repair-status.json','THREAD':'thread','datetime':datetime,'hashlib':hashlib,'json':json,'pathlib':pathlib,'shlex':shlex,'subprocess':types.SimpleNamespace(run=run),'time':types.SimpleNamespace(time=lambda:100,sleep=sleep)}
  exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec'),ns)
  result=json.loads(ns['STATE'].read_text());commands=[v for k,v in events if k=='ssh']
  assert events[0][0]=='wait'
  assert json.loads((root/'opponent-publisher-config.json').read_text())['sources'][0]=={'name':'amd','keep':'unchanged'}
  if already_done:
   assert result['status']=='ALREADY_REPAIRED_NO_REPLAY' and len(commands)==1
  else:
   assert result['status']=='RETRY_RELEASED_TO_EXISTING_OWNER'
   assert result['selection']['prefixes']==['classical2/ets/synthetic/rows=full']
   assert sum('git apply --check' in command for command in commands)==1
   assert 'OPPONENTS_DONE' in commands[-1]
print('PASS waits for full tail, exact failed ETS only, preserves other publisher entries, completed ETS never replayed')

# Execute the exact generated remote provenance code against isolated files.
assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='provenance' for t in n.targets))
remote_code=ast.literal_eval(assignment.value)
with tempfile.TemporaryDirectory() as folder:
 root=pathlib.Path(folder)
 paths={
  '/root/campaign-results/gpu-opponents/harness-repair.json':root/'provenance.json',
  '/root/overnight-nvidia/ets-host-repair-ready.json':root/'marker.json',
  '/root/opponent-harness/tools/bench_board_more.py':root/'driver.py',
 }
 paths['/root/campaign-results/gpu-opponents/harness-repair.json'].write_text(json.dumps({'base':'preserved'}))
 paths['/root/overnight-nvidia/ets-host-repair-ready.json'].write_text(json.dumps({'commit':'scoped-fix'}))
 paths['/root/opponent-harness/tools/bench_board_more.py'].write_bytes(b'driver source')
 for original,replacement in paths.items():remote_code=remote_code.replace(original,str(replacement))
 exec(compile(remote_code,'generated-provenance-code','exec'),{})
 result=json.loads(paths['/root/campaign-results/gpu-opponents/harness-repair.json'].read_text())
 assert result['base']=='preserved'
 assert result['additional_repairs']==[{'commit':'scoped-fix','file_sha256':hashlib.sha256(b'driver source').hexdigest()}]
print('PASS generated remote provenance writes parseable JSON, preserving base and repair hash')
