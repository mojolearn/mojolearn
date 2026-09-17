"""Reproduce the scoped TSA admission without replacing legacy references.

Run from the checkout with its Python package on PYTHONPATH. The immutable
baseline is intentional: unrelated table changes are never silently absorbed.
"""
import copy
import json
import subprocess
from pathlib import Path

from mojolearn import _verify_all as va
from mojolearn import _verify_reference as ref

ROOT = Path(__file__).resolve().parents[4]
BASELINE = '1752ab7c4'
TABLE = 'python/mojolearn/verify_reference/table.json'
LANES = {'select-d', 'holtwinters'}
harness = va.load_harness(str(ROOT / 'tools/identity_break.py'))
base = json.loads(subprocess.check_output(['git', 'show', f'{BASELINE}:{TABLE}'], cwd=ROOT))
paths = list((ROOT / 'bench/results/identity_break').rglob('*.json'))
candidate = ref.build_table([str(p) for p in paths], harness, str(ROOT), lanes=LANES)
assert candidate['admission_policy']['min_repeats'] == 2
assert candidate['fixtures'] == base['fixtures']
assert candidate['heldout'] == base['heldout']
assert set(candidate['cells']) == {f'{lane}/{fixture}' for lane in LANES for fixture in harness.FIXTURES}
assert not (set(candidate['cells']) & set(base['cells']))
# Four applicable/reference-bearing parts. Step/full is a sequence-decoder
# property and the existing public verifier records its inapplicability.
for cell in candidate['cells'].values():
    assert set(cell) == {'train', 'infer', 'model', 'batch'}
    for entry in cell.values():
        assert entry['ref'] is not None and not entry.get('conflict')
        assert set(entry['cols']) == {'cpu'}
        assert isinstance(entry['cols']['cpu'], int)
result = copy.deepcopy(base)
offset = len(result['records'])
result['records'].extend(candidate['records'])
for key, cell in candidate['cells'].items():
    cell = copy.deepcopy(cell)
    for entry in cell.values():
        entry['cols'] = {cls: idx + offset for cls, idx in entry['cols'].items()}
    result['cells'][key] = cell
for lane in LANES:
    if lane in candidate['lane_revisions']:
        result.setdefault('lane_revisions', {})[lane] = candidate['lane_revisions'][lane]
# Keep the legacy admission status and harness witness: this is NOT a
# regeneration of all old cells under today's stricter policy.
assert 'admission_policy' not in result
assert all(result['cells'][key] == cell for key, cell in base['cells'].items())
ref.write_table(result, str(ROOT / TABLE))
audit = dict(baseline=BASELINE, lanes=sorted(LANES), added_cells=len(candidate['cells']),
             added_parts=sum(map(len, candidate['cells'].values())),
             unchanged_existing_cells=len(base['cells']),
             admission_policy=candidate['admission_policy'],
             harness_sha256=candidate['harness_sha256'], records=candidate['records'],
             scope='Strictly admitted CPU historical references; legacy table and release qualification unchanged')
Path(__file__).with_name('admission.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit, indent=2))
