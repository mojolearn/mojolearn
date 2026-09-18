"""Build a scoped replacement using existing witnessed captures; no promotion."""
from pathlib import Path
import hashlib, json, sys

root=Path('/Users/andrewhendel/mojolearn-wt/verification-coverage-continuation')
sys.path.insert(0,str(root/'python'))
from mojolearn import _verify_all as va, _verify_reference as vr

base=Path(__file__).resolve().parent
table=vr.load_table(str(root/'python/mojolearn/verify_reference/table.json'))
lanes={'gp-sample-y','gp-sample-y-normalize'}
records=root/'bench/results/identity_break'
indexes=set()
for key,cell in table['cells'].items():
    if key.partition('/')[0] in lanes:
        for entry in cell.values():
            indexes.update(i[0] if isinstance(i,list) else i for i in entry['cols'].values())
paths={records/table['records'][i]['dir']/table['records'][i]['file'] for i in indexes}
paths.update(records/'2026-09-18_installed-apple-properties'/f'{lane}.json' for lane in lanes)
messages=[]
candidate=vr.build_table(list(map(str,paths)),va.load_harness(),str(root),lanes=lanes,parts=vr.PARTS+vr.OPTIONAL_PARTS,log=messages.append)
merged=vr.merge_reference_lanes(table,candidate,sorted(lanes))
changes=[]
for key,cell in table['cells'].items():
    current=merged['cells'][key]
    if key.partition('/')[0] not in lanes:
        assert current==cell, key
        continue
    assert set(cell)<=set(current), key
    for part,entry in cell.items():
        assert current[part]['ref']==entry['ref'], (key,part,'numerical reference changed')
    for part,entry in current.items():
        for cls,index in entry['cols'].items():
            assert isinstance(index,int), (key,part,cls,'disagreement remains')
            assert 'metal-transient' not in merged['records'][index]['dir']
        assert {'cpu','apple','amd'} <= set(entry['cols']), (key,part)
    changes.append(key)
vr.write_table(merged,str(base/'gp-reference-candidate.json'))
receipt=dict(status='SCOPED_CANDIDATE_CHECKED_NOT_APPLIED',lanes=sorted(lanes),selected_cells=len(changes),unselected_cells_unchanged=len(table['cells'])-len(changes),existing_reference_hashes_unchanged=True,inputs={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)},admission_log=messages,source_table_sha256=vr.sha256_file(root/'python/mojolearn/verify_reference/table.json'),candidate_sha256=vr.sha256_file(base/'gp-reference-candidate.json'))
(base/'gp-admission-review.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('inputs','admission_log')}),flush=True)
