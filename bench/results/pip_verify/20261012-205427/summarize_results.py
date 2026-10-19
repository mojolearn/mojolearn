import collections
import json
from pathlib import Path

root = Path(__file__).parent
summary = {'environment': json.loads((root / 'environment.json').read_text())}
for name in ('quick', 'full', 'diagnostic-numpy-2.5.2'):
    report = json.loads((root / (name + '.json')).read_text())
    failures = [row for row in report['cells'] if row['state'] not in ('IDENTICAL', 'N/A')]
    cross = report.get('cross_check') or {}
    result = {
        'exit': int((root / (name + '.exit')).read_text()),
        'verdict': report['verdict'],
        'elapsed_s': report['elapsed_s'],
        'device': report['device'],
        'counts': report['counts'],
        'lane_count': len(report['lanes']),
        'fixture_count': len(report['fixtures']),
        'execution': report['execution'],
        'failures_by_lane': dict(collections.Counter(row['lane'] for row in failures)),
        'failure_details': [{k: row.get(k) for k in ('lane', 'fixture', 'part', 'state', 'value', 'reference', 'detail')} for row in failures],
        'divergent_parts_matching_recorded_apple': sum(row['state'] == 'DIVERGENT' and row.get('columns', {}).get('apple', {}).get('value') == row['value'] for row in failures),
        'cross_check': {
            **{k: cross.get(k) for k in ('ran', 'passed', 'complete', 'agree', 'differ', 'compared')},
            'missing_parts': len(cross.get('missing', [])),
            'missing_parts_by_lane': dict(collections.Counter(row['lane'] for row in cross.get('missing', []))),
            'skipped_cases': len(cross.get('skipped', {})),
            'skip_reasons': dict(collections.Counter(cross.get('skipped', {}).values())),
        },
        'profile': report.get('selection', {}).get('profile', {}).get('name'),
        'evidence': {'json': str(root / (name + '.json')), 'log': str(root / (name + '.log'))},
    }
    summary[name] = result
summary['installation'] = {name: int((root / (name + '.exit')).read_text()) for name in ('pip-install', 'pip-verify-extra', 'pip-check')}
summary['commands'] = {
    'install': 'python -m pip install mojolearn',
    'verification_support': 'python -m pip install mojolearn[verify]==0.8.42',
    'quick': 'python -I -u -m mojolearn verify --quick --json-out quick.json',
    'full': 'python -I -u -m mojolearn verify --full --json-out full.json',
    'diagnostic': 'python -I -u -m mojolearn verify --full --lanes linalg-eigh,trees-oob-cv-link --json-out diagnostic-numpy-2.5.2.json',
}
summary['scope_note'] = 'Published macOS arm64 wheel in an isolated environment. Routine verification excludes neural training and physical multi-device identity. The diagnostic used a separate environment with the same package and NumPy 2.5.2. No repair or new release is claimed.'
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
for name in ('quick', 'full', 'diagnostic-numpy-2.5.2'):
    result = summary[name]
    print(json.dumps({k: result[k] for k in ('verdict', 'exit', 'elapsed_s', 'lane_count', 'fixture_count', 'counts', 'failures_by_lane', 'cross_check')}, sort_keys=True))
print('Summary:', root / 'summary.json')
