"""Summarize retained phase diagnostics and same-process component trials."""
import json,re,statistics as st
from pathlib import Path
root=Path(__file__).resolve().parents[1]
raw=root/'raw'/'knn-loads'
report={'source_commit':'db782123','scope':'Diagnostics only; no ordinary request price or default admission.','phases':{},'trials':{}}
for k in (10,15):
    s=(raw/f'phase-k{k}.log').read_text()
    assert 'KNN REFERENCE PRICE PASS' in s and 'query_tile 512' in s
    lines=[x for x in s.splitlines() if 'KNN_PHASE_TIMERS distance_ms' in x][-10:]
    assert len(lines)==10
    report['phases'][str(k)]={a:st.median(float(re.search(a+r' ([\d.]+)',x)[1]) for x in lines) for a in ('distance_ms','select_ms','merge_ms')}
for variant in ('vector','aligned','interior'):
    s=(raw/variant/'trial.log').read_text()
    assert 'KNN INDEX LAYOUT QUALIFICATION PASS' in s
    times={a:[float(v) for v in re.findall(r'^SAMPLE \d+ '+a+r' ([\d.]+)',s,re.M)] for a in ('scalar','vector')}
    assert all(len(v)==15 for v in times.values())
    med={a:st.median(v) for a,v in times.items()}
    gates=[x for x in s.splitlines() if x.startswith('BITGATE PASS')]
    assert any('512 65536 32' in x for x in gates)
    report['trials'][variant]={'median_ms':med,'candidate_over_scalar':med['vector']/med['scalar'],'samples':times,'bitgate_cases':len(gates)}
(root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
