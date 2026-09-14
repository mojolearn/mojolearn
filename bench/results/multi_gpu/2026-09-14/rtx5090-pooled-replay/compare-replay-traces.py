import json
from pathlib import Path
old=Path('/root/h100-traces'); new=Path('/root/replay-out')
counts={'histogram_files':0,'histogram_bytes':0,'trace_files':0}
for directory in sorted(old.iterdir()):
    target=new/('ordered_rmse' if directory.name.startswith('ordered-') else 'pointwise')/directory.name
    assert {p.name for p in directory.iterdir()} == {p.name for p in target.iterdir()}, directory.name
    for p in directory.iterdir():
        q=target/p.name
        if p.suffix == '.bin':
            assert p.read_bytes() == q.read_bytes(), str(p)
            counts['histogram_files'] += 1
            counts['histogram_bytes'] += p.stat().st_size
        elif p.suffix == '.trace':
            records=lambda f:[s for s in f.read_text().splitlines() if s and not s.startswith('#')]
            assert records(p) == records(q), str(p)
            counts['trace_files'] += 1
        else:
            raise AssertionError('unrecognized trace artifact: '+str(p))
assert counts['histogram_files'] and counts['trace_files']
result=dict(status='IDENTICAL', scope='H100 vs RTX 5090: complete raw histogram dumps and pointwise/OrderedRMSE trace records', **counts)
(new/'trace-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
