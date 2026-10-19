import collections
import json
import re
from pathlib import Path

root = Path(__file__).parent
progress = json.loads((root / 'full.json.progress.json').read_text())
counts = collections.Counter()
problems = []
problem_lanes = collections.Counter()
matches_recorded_apple = 0
with (root / 'full.json.cells.jsonl').open() as journal:
    for line in journal:
        try:
            part = json.loads(line)
        except json.JSONDecodeError:
            continue  # The writer may be appending the current final line.
        counts[part['state']] += 1
        if part['state'] not in ('IDENTICAL', 'N/A'):
            problems.append({k: part.get(k) for k in ('lane', 'fixture', 'part', 'state', 'detail')})
            problem_lanes[part['lane'] + ':' + part['state']] += 1
            apple = part.get('columns', {}).get('apple', {})
            matches_recorded_apple += part['state'] == 'DIVERGENT' and apple.get('value') == part.get('value')
cross_rows = re.findall(r'^  (\S+)\s+(\S+)\s+batch=(\S+) infer=(\S+)', (root / 'full.log').read_text(), re.MULTILINE)
print(json.dumps({
    'stage': progress.get('stage'),
    'complete': progress['complete'],
    'elapsed_s': progress.get('elapsed_s'),
    'execution': progress.get('execution'),
    'part_counts': dict(counts),
    'problem_count': len(problems),
    'problem_lanes': dict(problem_lanes),
    'divergent_parts_matching_recorded_apple': matches_recorded_apple,
    'cross_check_rows': len(cross_rows),
    'cross_check_fixtures': dict(collections.Counter(row[1] for row in cross_rows)),
    'cross_check_states': dict(collections.Counter(value for row in cross_rows for value in row[2:])),
}, indent=2))
