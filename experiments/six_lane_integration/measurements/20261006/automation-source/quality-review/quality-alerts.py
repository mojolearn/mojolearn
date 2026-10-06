"""Saved quality failure edges, separate from reviewer/publisher process exits."""
from pathlib import Path
import datetime,json,time
ROOT=Path(__file__).resolve().parent
LEDGER=ROOT/'quality-failure-ledger.json'
STATUS=ROOT/'quality-failure-alert-status.json'
def atomic(path,value):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(path)
def update():
 ledger=json.loads(LEDGER.read_text())
 found={}
 for name in ('classical-12-pair-quality-review.json','next-reg-resample-quality-review.json'):
  source=ROOT/name
  for row in json.loads(source.read_text())['rows']:
   verdict=row.get('quality_assessment','')
   if verdict!='QUALITY_FAILED' and not verdict.startswith('FAILED_'):continue
   ident=row['receipt_sha256']
   found[ident]={'receipt_sha256':ident,'vendor':row['vendor'],'lane':row['lane'],'dataset':row['dataset'],'quality_assessment':verdict,'reason':row.get('reason'),'receipt':row['receipt'],'review':str(source)}
   if ident not in ledger['failures']:
    ledger['failures'][ident]=dict(found[ident],first_seen_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (ROOT/'automation-logs/new-quality-failures.jsonl').open('a') as log:log.write(json.dumps(ledger['failures'][ident])+'\n')
 # Keep an unacknowledged edge visible even if a later review changes; only
 # explicit acknowledgement of its exact receipt ID clears it. The watcher
 # deduplicates unchanged IDs/reasons using its persisted alert fingerprint.
 alerts=[dict(v,currently_failed=k in found) for k,v in sorted(ledger['failures'].items()) if k not in ledger['acknowledged_receipt_ids']]
 ledger['updated_at']=time.time();atomic(LEDGER,ledger)
 atomic(STATUS,{'status':'QUALITY_FAILED' if alerts else 'RUNNING','phase':'New saved quality failure requires review' if alerts else 'No unacknowledged new quality failures','heartbeat_at':time.time(),'failed':len(alerts),'errors':[{'receipt_sha256':v['receipt_sha256'],'vendor':v['vendor'],'lane':v['lane'],'dataset':v['dataset'],'reason':v['reason']} for v in alerts],'failures':alerts,'acknowledged_receipt_ids':ledger['acknowledged_receipt_ids'],'process_errors_separate':'automation-status.json','ledger':str(LEDGER),'numerical_work':False,'automatic_promotion':False})
 return len(alerts)
if __name__=='__main__':print(json.dumps({'new_unacknowledged_failures':update()}))
