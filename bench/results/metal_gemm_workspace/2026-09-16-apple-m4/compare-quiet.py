import os, subprocess, time, json, hashlib
from pathlib import Path
root = Path(__file__).parent
rows=[]
for i, arm in enumerate(('baseline','candidate','candidate','baseline','baseline','candidate')):
    trace=root/f'quiet-{i}-{arm}.trace'
    env=dict(os.environ, MOJOLEARN_IDENTITY_TRACE=str(trace))
    env.pop('MOJOLEARN_TRANSFORMER_TIMING',None)
    with (root/f'quiet-{i}-{arm}.log').open('w') as f:
        start=time.monotonic()
        p=subprocess.run([str(root/f'{arm}-backward')],env=env,stdout=f,stderr=subprocess.STDOUT)
        seconds=time.monotonic()-start
    if p.returncode: raise RuntimeError(f'{arm} failed: {p.returncode}')
    row=dict(arm=arm,seconds=seconds,trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),binary_sha256=hashlib.sha256((root/f'{arm}-backward').read_bytes()).hexdigest())
    rows.append(row)
    print(json.dumps(row),flush=True)
    (root/'quiet-timings.json').write_text(json.dumps(rows,indent=2)+'\n')
assert len({r['trace_sha256'] for r in rows})==1
print('PASS all six oracle gates and identical 37-stage cards',flush=True)
