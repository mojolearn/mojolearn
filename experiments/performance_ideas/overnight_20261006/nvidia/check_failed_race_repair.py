"""Exercise the real harness scheduling loop only; no algorithm is executed."""
import ast,copy,contextlib,io,pathlib,runpy,sys,types,json
path=pathlib.Path('/Users/andrewhendel/mojolearn-wt/overnight-opponents-publisher-20261006/tools/bench_board.py');sys.path.insert(0,str(path.parent));ns=runpy.run_path(str(path));tree=ast.parse(path.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
loop=next(n for n in main.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='r' and isinstance(n.iter,ast.Name) and n.iter.id=='races' and n.body and isinstance(n.body[0],ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prev' for t in n.body[0].targets))
old={'status':'failed','finished':'2026-10-06T05:00:00Z','cells':[{'arm':'implicit-gpu','status':'REFUSED(missing CUDA13)'}]}
result={'races':{'algos/als/text':copy.deepcopy(old),'algos/other/text':{'status':'failed','finished':'2026-10-06T05:00:00Z'},'algos/cached/text':{'status':'done','finished':'2026-10-06T05:00:00Z','params_check':'MATCHED'},'algos/als/repaired':{'status':'done','finished':'2026-10-06T06:00:00Z','params_check':'MATCHED'}}}
before=copy.deepcopy(result['races']);races=[{'id':k,'family':'algos'} for k in result['races']]+[{'id':'algos/missing/text','family':'algos'}]
ns.update(todo=[],races=races,result=result,args=types.SimpleNamespace(skip_failed=True,rerun='algos/als/',rerun_before='2026-10-06T05:24:06Z'),ctx={})
with contextlib.redirect_stdout(io.StringIO()):exec(compile(ast.Module(body=[loop],type_ignores=[]),str(path),'exec'),ns)
assert [r['id'] for r in ns['todo']]==['algos/als/text','algos/missing/text']
assert result['superseded']==[old]
assert result['races']==before
print(json.dumps(dict(status='PASS',checks=['selected failed ALS queued despite skip_failed','other failed cell skipped','cached successful cells preserved','repaired-after-cutoff ALS not repeated','missing work remains queued','original refusal retained verbatim in superseded','original race records unchanged before execution'])))
