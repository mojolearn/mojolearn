"""Upload a finite snapshot of explicitly queued archives; never watch for new jobs."""
import argparse
import json
from pathlib import Path
import sys
import time

from controller_offsite import upload_one, save

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--queue', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--max-items', type=int, required=True)
p.add_argument('--seconds', type=int, default=3600)
p.add_argument('--request-timeout', type=int, default=1200)
a = p.parse_args()
if min(a.max_items, a.seconds, a.request_timeout) <= 0 or a.request_timeout > 1200:
    p.error('positive bounded limits required; maximum request timeout1200')
sys.path.insert(0, str(a.root / 'tools'))
import bincache
import evidence_r2
creds = evidence_r2._creds()
paths = sorted(a.queue.glob('*.json'))[:a.max_items]
start = time.monotonic()
report = dict(status='running', snapshot=[str(x) for x in paths], results=[], qualification_claim=False)
save(a.out, report)
for path in paths:
    if time.monotonic() - start + a.request_timeout > a.seconds:
        report['status'] = 'finite-budget-exhausted'
        break
    try:
        outcome = upload_one(path, lambda method, key, **kw: bincache.r2_request(creds, method, key, **kw), a.request_timeout)
        report['results'].append(dict(path=str(path), status=outcome))
    except Exception as error:
        report['results'].append(dict(path=str(path), status='offsite-pending', error=str(error)))
        report['status'] = 'stopped-retention-failure'
        break
    save(a.out, report)
else:
    report['status'] = 'snapshot-retained'
report['elapsed_seconds'] = time.monotonic() - start
save(a.out, report)
print(json.dumps(dict(status=report['status'], selected=len(paths), completed=len(report['results']))))
raise SystemExit(0 if report['status'] == 'snapshot-retained' else 75)
