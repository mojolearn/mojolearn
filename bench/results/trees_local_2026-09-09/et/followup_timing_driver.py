import subprocess, statistics, json
results = []
cases = [('identical', 262144, 2, 16, 10), ('fast', 65536, 5, 8, 8), ('fast', 65536, 9, 8, 8), ('fast', 65536, 17, 8, 8)]
for mode, rows, classes, trees, depth in cases:
    timings = {'baseline': [], 'dispatch': []}
    hashes = set()
    suffix = '_identical' if mode == 'identical' else ''
    for arm in ['baseline', 'dispatch', 'dispatch', 'baseline']:
        cmd = [f'/tmp/et_{arm}{suffix}_bench', str(rows), str(classes), str(trees), str(depth), '3']
        run = subprocess.run(cmd, text=True, capture_output=True, check=True)
        for line in run.stdout.splitlines():
            if line.startswith('fingerprint '):
                hashes.add(line)
            elif line.startswith('fit_ms '):
                timings[arm].append(float(line.split()[1]))
    assert len(hashes) == 1, hashes
    medians = {a: statistics.median(v) for a,v in timings.items()}
    result = {'rows': rows, 'classes': classes, 'cols': 13, 'trees': trees, 'depth': depth, 'mode': mode, 'fit_ms': timings, 'medians': medians, 'speedup': medians['baseline']/medians['dispatch'], 'fingerprint': hashes.pop()}
    results.append(result)
    print(json.dumps(result), flush=True)
open('bench/results/trees_local_2026-09-09/et/followup_timing.json','w').write(json.dumps(results,indent=2)+'\n')
