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
