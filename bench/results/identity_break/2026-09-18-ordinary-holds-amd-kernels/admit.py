import json,sys,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(R/'python'),str(R/'tools')]
import identity_break as h
from mojolearn import _verify_reference as ref
old=R/'bench/results/identity_break/2026-09-18-cpu-kernel-variants'
new=R/'bench/results/identity_break/2026-09-18-ordinary-holds-amd-kernels'
lanes=[f'{f}-{k}' for f in ('kernel-ridge','nystroem') for k in ('poly','sigmoid','laplacian')]
paths=[str(old/(arm+'-'+lane+'.json')) for arm in ('cpu','apple') for lane in lanes]+[str(new/'captures-kernels'/(lane+'.json')) for lane in lanes]
log=[]
assert hashlib.sha256((R/"python/mojolearn/verify_reference/table.json").read_bytes()).hexdigest()=='1dfbc700520f88952f54be0c7f5fe545e15cb9e14308a59e04101a6b3915b535', "run against the pre-admission f771338c7 table; do not overwrite later work"
base=ref.load_table()
candidate=ref.build_table(paths,h,str(R),lanes=lanes,parts=ref.PARTS+ref.OPTIONAL_PARTS,log=log.append)
counts={};numeric=0
for lane in lanes:
 for fixture in h.FIXTURES:
  key=lane+'/'+fixture
  cell=candidate['cells'][key]
  required=set(ref.PARTS+ref.OPTIONAL_PARTS)-({'rlpair'} if lane not in h.RLPAIR else set())
  assert set(cell)==required,(key,sorted(cell))
  for part,entry in cell.items():
   assert entry.get('ref') is not None and not entry.get('conflict'),(key,part,entry)
   assert set(entry['cols'])=={'cpu','apple','amd'},(key,part,entry)
   assert all(isinstance(v,int) for v in entry['cols'].values()),(key,part,entry)
   counts[part]=counts.get(part,0)+1
   numeric+=not entry['ref'].startswith('n/a:')
merged=ref.merge_reference_lanes(base,candidate,lanes)
changed={k for k in set(base['cells'])|set(merged['cells']) if base['cells'].get(k)!=merged['cells'].get(k)}
expected={lane+'/'+f for lane in lanes for f in h.FIXTURES}
assert changed==expected,(changed^expected)
for k in set(base['cells'])-expected:assert base['cells'][k]==merged['cells'][k]
receipt=dict(status='SCOPED_REFERENCE_ADMITTED_NOT_DEFAULT_PROMOTED',lanes=lanes,classes=['cpu','apple','amd'],cells_changed=len(changed),numeric_parts=numeric,parts=counts,unchanged_other_cells=len(set(base['cells'])-expected),missing=['NVIDIA property/model recordings','installed CPU verifier replay'],log=log,source_files={str(Path(p).relative_to(R)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths})
(new/'admission.json').write_text(json.dumps(receipt,indent=2)+'\n')
ref.write_table(merged,R/'python/mojolearn/verify_reference/table.json')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('log','source_files')},indent=2))
