import json,sys
from pathlib import Path
root=Path('/Users/andrewhendel/mojolearn-wt/cpu-verification-completion')
base=Path(__file__).resolve().parent
sys.path[:0]=[str(root/'python'),str(root/'tools')]
from mojolearn import _verify_all as va,_verify_reference as vr
lanes=['mamba3','transformer','transformer-window','samba','samba-untied-dropout-accum']
h=va.load_harness(str(root/'tools/identity_break.py'))
with (base/'remaining-neural-generation.log').open('w') as log:
 t=vr.build_table(list(map(str,(root/'bench/results/identity_break').rglob('*.json'))),h,str(root),lanes=lanes,log=lambda s:log.write(s+'\n'))
vr.write_table(t,str(base/'remaining-neural-table.json'))
report={}
for lane in lanes:
 missing=[];single=[];classes=set()
 for fx in h.FIXTURES:
  for part in ('train','infer','model','batch','stepfull'):
   e=t['cells'].get(lane+'/'+fx,{}).get(part,{})
   if e.get('ref') is None or e.get('conflict'):missing.append(fx+'/'+part);continue
   agree=[c for c,v in e['cols'].items() if isinstance(v,int)];classes.update(agree)
   if not str(e['ref']).startswith('n/a:') and len(agree)<2:single.append(fx+'/'+part)
 report[lane]={'missing':missing,'single_class':single,'classes':sorted(classes),'eligible_for_independent_wheel_replay':not(missing or single)}
(base/'remaining-neural-audit.json').write_text(json.dumps(report,indent=2)+'\n')
for lane,row in report.items():print(lane,'missing',len(row['missing']),'single_class',len(row['single_class']),'classes',row['classes'],'eligible',row['eligible_for_independent_wheel_replay'])
