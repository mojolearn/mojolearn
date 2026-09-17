"""Admit complete source records only when their bytes match the independent wheel run."""
import ast,gzip,hashlib,json,os,sys
from pathlib import Path
root=Path('/Users/andrewhendel/mojolearn-wt/cpu-verification-completion')
base=Path(__file__).resolve().parent
sys.path[:0]=[str(root/'python'),str(root/'tools')]
from mojolearn import _verify_all as va,_verify_reference as vr
class WithoutDocstrings(ast.NodeTransformer):
 def visit_Expr(self,node):
  return None if isinstance(node.value,ast.Constant) and isinstance(node.value.value,str) else self.generic_visit(node)
def program(path):return ast.dump(WithoutDocstrings().visit(ast.parse(path.read_text())),include_attributes=False)
assert program(root/'tools/identity_break.py')==program(base/'fresh-host-source-7f5b786ae/tools/identity_break.py'),'The wheel and source harnesses differ beyond documentation'
harness=va.load_harness(str(root/'tools/identity_break.py'))
old=vr.load_table(str(root/'python/mojolearn/verify_reference/table.json'))
summary=json.loads((base/'fresh-cpu-wheel-replay/summary.json').read_text())
assert summary['complete']
lanes=[lane for lane,row in summary['results'].items() if not row['passed']]
reports={lane:json.loads(gzip.decompress((base/f'fresh-cpu-wheel-replay/{lane}.json.gz').read_bytes())) for lane in lanes}
for report in reports.values():
 assert report['self_test']['passed'] and report['repeats'] >= 2
 for cell in report['cells']:
  assert cell['state'] in ('IDENTICAL','N/A','OWED'),cell
  if cell['state']=='OWED':assert cell['reference'] is None or str(cell['reference']).startswith('n/a:'),cell
paths=list(map(str,(root/'bench/results/identity_break').rglob('*.json')))
with (base/'reference-repair-generation.log').open('w') as log:
 candidate=vr.build_table(paths,harness,str(root),log=lambda s:log.write(s+'\n'),lanes=lanes)
vr.write_table(candidate,str(base/'reference-repair-candidate.json'))
eligible=[];held={};changes=[]
for lane in lanes:
 problems=[]
 measured={(cell['fixture'],cell['part']):cell for cell in reports[lane]['cells']}
 for fixture in old['fixtures']:
  key=f'{lane}/{fixture}';prior=old['cells'].get(key,{})
  generated=candidate['cells'].get(key,{})
  required=({'train','infer','model','batch'}|set(prior)|set(generated)
            |{part for (fx,part),cell in measured.items() if fx==fixture and cell['state']!='N/A'})
  for part in sorted(required):
   entry=generated.get(part,{})
   ref=entry.get('ref');oldref=prior.get(part,{}).get('ref')
   if ref is None or entry.get('conflict'):
    problems.append(f'{key}/{part}: no complete qualifying reference');continue
   if oldref is not None and not str(oldref).startswith('n/a:') and oldref!=ref:
    problems.append(f'{key}/{part}: would change an existing numerical reference');continue
   actual=measured.get((fixture,part),{}).get('value')
   if ((not str(ref).startswith('n/a:') or
        (actual is not None and not str(actual).startswith('n/a:'))) and actual!=ref):
    problems.append(f'{key}/{part}: '+('no current numerical reference (source still says N/A)' if str(ref).startswith('n/a:') else 'source record disagrees with independently measured Mac wheel bytes'));continue
   if oldref!=ref:changes.append({'lane':lane,'cell':key,'part':part,'before':oldref,'after':ref})
 if problems:held[lane]=problems
 else:eligible.append(lane)
result=vr.merge_reference_lanes(old,candidate,eligible)
vr.write_table(result,str(base/'reference-repair-table.json'))
report={'eligible':eligible,'held':held,'changes':[c for c in changes if c['lane'] in eligible],'old_table_sha256':hashlib.sha256((root/'python/mojolearn/verify_reference/table.json').read_bytes()).hexdigest(),'new_table_sha256':hashlib.sha256((base/'reference-repair-table.json').read_bytes()).hexdigest(),'scope':'Strict source record admission, same numerical harness AST, all new numerical references match independent frozen Mac wheel measurements; no existing numerical reference changed.'}
(base/'reference-repair-admission.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'eligible':eligible,'held':held,'changed_parts':len(report['changes'])},indent=2))
