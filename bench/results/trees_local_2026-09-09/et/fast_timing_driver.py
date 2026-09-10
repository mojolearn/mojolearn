import subprocess, statistics, json
results = []
for rows in [65536, 262144]:
    timings = {'baseline': [], 'dispatch': []}
    hashes = set()
    for arm in ['baseline', 'dispatch', 'dispatch', 'baseline']:
        cmd = [f'/tmp/et_{arm}_bench', str(rows), '2', '16', '10', '3']
        run = subprocess.run(cmd, text=True, capture_output=True, check=True)
        for line in run.stdout.splitlines():
            if line.startswith('fingerprint '):
                hashes.add(line)
            elif line.startswith('fit_ms '):
                timings[arm].append(float(line.split()[1]))
    assert len(hashes) == 1, hashes
    medians = {a: statistics.median(v) for a,v in timings.items()}
    result = {'rows': rows, 'classes': 2, 'cols': 13, 'trees': 16, 'depth': 10, 'mode': 'FAST', 'fit_ms': timings, 'medians': medians, 'speedup': medians['baseline']/medians['dispatch'], 'fingerprint': hashes.pop()}
    results.append(result)
    print(json.dumps(result), flush=True)
open('/tmp/et_dispatch_timing.json','w').write(json.dumps(results,indent=2)+'\n')
