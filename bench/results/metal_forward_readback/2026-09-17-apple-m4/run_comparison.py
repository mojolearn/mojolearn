import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

root = Path(__file__).parent
scheduler = '/Users/andrewhendel/CascadeProjects/mojolearn/tools/mac_slot.py'
records = {}
for arm in ('baseline', 'candidate'):
    binary = root / arm
    card = root / f'{arm}.trace'
    env = dict(os.environ, MOJOLEARN_IDENTITY_TRACE=str(card))
    env.pop('MOJOLEARN_TRANSFORMER_TIMING', None)
    command = ['python3', scheduler, '--deadline', str(time.monotonic() + 60),
               '--timeout', '60', '--wait-timeout', '60',
               '--timing-json', str(root / f'{arm}-scheduler.json'), 'metal', str(binary)]
    with (root / f'{arm}.log').open('w') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    records[arm] = {'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(), 'exit_code': result.returncode}
    (root / 'comparison.json').write_text(json.dumps(records, indent=2) + '\n')
    if result.returncode:
        raise SystemExit(f'{arm} failed ({result.returncode}); do not expand the diagnostic')
    records[arm]['card_sha256'] = hashlib.sha256(card.read_bytes()).hexdigest()
    samples = re.findall(r'execution (\d+) waits (\d+) launches (\d+) ms ([0-9.eE+-]+) oracle_cells (\d+) host_allocs (\d+) copies (\d+)', (root / f'{arm}.log').read_text())
    if len(samples) != 2:
        raise SystemExit(f'{arm}: expected two completed executions')
    records[arm]['executions'] = [dict(repeat=int(r), waits=int(w), launches=int(l), ms=float(t), oracle_cells=int(c), host_allocs=int(a), copies=int(d)) for r,w,l,t,c,a,d in samples]
    print(arm, records[arm], flush=True)
assert records['baseline']['card_sha256'] == records['candidate']['card_sha256']
for before, after in zip(records['baseline']['executions'], records['candidate']['executions']):
    assert before['launches'] == after['launches'], (before, after)
    assert before['oracle_cells'] == after['oracle_cells'], (before, after)
    assert before['waits'] - after['waits'] == 58, (before, after)
    assert before['host_allocs'] - after['host_allocs'] == 29, (before, after)
    assert before['copies'] == after['copies'], (before, after)
records['result'] = 'PASS: 58 fewer waits and 29 fewer host allocations, unchanged copies, launches and oracle bits, matching traced card'
(root / 'comparison.json').write_text(json.dumps(records, indent=2) + '\n')
print(records['result'], flush=True)
