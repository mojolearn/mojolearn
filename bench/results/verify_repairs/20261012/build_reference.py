import pathlib,sys,json,collections
root=pathlib.Path('/Users/andrewhendel/mojolearn-wt/verify-neural-20261012')
sys.path.insert(0,str(root/'tools'))
import identity_columns
h=identity_columns.harness();v=identity_columns.vref()
files=sorted(str(p) for p in (root/'bench/results/identity_break').rglob('*.json'))
p=pathlib.Path(__file__).resolve().parent
with (p/'reference-build.log').open('w') as log:
 table=v.build_table(files,h,str(root),log=lambda s: log.write(s+'\n'),parts=v.PARTS+v.OPTIONAL_PARTS)
target=root/'python/mojolearn/verify_reference/table.json'
before=json.loads(target.read_text());target.write_text(json.dumps(table,sort_keys=True,separators=(',',':'))+'\n')
old={(k,p) for k,parts in before['cells'].items() for p in parts};new={(k,p) for k,parts in table['cells'].items() for p in parts}
summary=dict(records=len(table['records']),cells=len(table['cells']),parts=len(new),removed_parts=len(old-new),added_parts=len(new-old),ptx_divergent=len(table.get('ptx_divergent',[])),conflicts=sum(bool(e.get('conflict')) for c in table['cells'].values() for e in c.values()),removed_by_lane=dict(collections.Counter(k.split('/')[0] for k,_ in old-new)),ptx_by_lane=dict(collections.Counter(r['cell'].split('/')[0] for r in table.get('ptx_divergent',[]))))
(p/'reference-build-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
