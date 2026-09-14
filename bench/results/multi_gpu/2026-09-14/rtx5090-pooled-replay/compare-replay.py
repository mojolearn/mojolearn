import json
from pathlib import Path
root = Path('/root/replay-out')
reference = Path('/root/replay-reference')
manifest = json.loads((reference / 'manifest.json').read_text())
rows = []
for name, relative in manifest.items():
    actual = root / relative
    if not actual.is_file():
        rows.append(dict(name=name, status='MISSING', file=relative))
        continue
    before = json.loads((reference / (name + '.json')).read_text())
    after = json.loads(actual.read_text())
    if before.get('status') != 'PASS' or after.get('status') != 'PASS':
        rows.append(dict(name=name, status='FAIL', reason='gate did not pass'))
        continue
    # Compare the complete structured receipt, excluding only its prose scope.
    # Shape/configuration, check names and every recorded raw-byte hash remain.
    before.pop('scope', None)
    after.pop('scope', None)
    rows.append(dict(name=name, status='IDENTICAL' if before == after else 'DIFFERENT'))
report = dict(source='same frozen H100 working tree, target sm_120 vs sm_90a',
              scope='H100 vs RTX 5090, two NVIDIA architectures; no AMD/Apple claim',
              checks=rows)
(root / 'h100-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
if any(row['status'] != 'IDENTICAL' for row in rows):
    raise SystemExit(1)
