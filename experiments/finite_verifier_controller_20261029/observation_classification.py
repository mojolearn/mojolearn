"""Controller-only stop policy; does not admit records or alter verifier verdicts."""
import re
# Exact prior-release version files retained in historical-reference-release-witnesses.json.
# An unknown or same-release commit is never excused merely for differing from HEAD.
HISTORICAL_RELEASES = {'439199dcddf58ca68fe7c7ed2078da0a7a864326': '0.8.57', '640adbe7ea09098c644926c845d2719f05a70f34': '0.8.58', 'b25df4e213e0de66f4db2d526bf2f350889f23f9': '0.8.57'}

def classify(report, receipt):
 rows=report.get('cells',[])
 failures=sum(row.get('local_check')=='failed' or row.get('state')=='REFUSED' or not isinstance(row.get('value'),str) or not (re.fullmatch(r'[0-9a-f]{16}',row['value']) or row['value'].startswith('n/a:')) for row in rows)
 divergent=[r for r in rows if r.get('state')=='DIVERGENT']
 old_refs=report.get('reference_evidence',{}).get('columns',[])
 historical_only=bool(divergent and old_refs) and all(re.fullmatch(r'[0-9a-f]{40}',str(r.get('recorded_at_commit',''))) and r['recorded_at_commit'] in HISTORICAL_RELEASES and r['recorded_at_commit']!=report.get('device',{}).get('commit') for r in old_refs) and all(r.get('local_check')!='failed' and str(r.get('detail','')).startswith('this box ') and ', reference ' in str(r.get('detail','')) for r in divergent)
 e=report.get('execution',{});ls=report.get('lanes_summary',{})
 expected={(l,f,p) for l in report.get('lanes',[]) for f in report.get('fixtures',[]) for p in ['train','infer','model','batch','batchgrad','batchscale','ragged','rlpair','stepfull']}
 actual={(r.get('lane'),r.get('fixture'),r.get('part')) for r in rows}
 rc=receipt.get('exit_code')
 complete=(rc==0 or (rc==1 and historical_only)) and not failures and len(expected)==len(rows) and actual==expected and len(report.get('fixtures',[]))==9 and not e.get('crashed_cells') and not e.get('interrupted') and e.get('completed_cells')==e.get('total_cells')==9 and ls.get('attempted')==1 and not ls.get('not_run') and report.get('native_provenance_complete') is True and report.get('self_test',{}).get('passed') is True
 return dict(observation_complete=bool(complete),local_property_failures=failures,historical_reference_only=bool(historical_only),original_exit_code=rc,identity_admission='pending')
