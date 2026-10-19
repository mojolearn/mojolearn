import argparse
import os
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import mojolearn as ml
from mojolearn import _identity_break as h

parser = argparse.ArgumentParser()
parser.add_argument('--lanes', default='linalg-eigh,trees-oob-cv-link')
parser.add_argument('--fixtures', default=','.join(h.FIXTURES))
parser.add_argument('--reference', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
reference = json.loads(Path(args.reference).read_text())
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
rows = []
for lane in args.lanes.split(','):
    for fixture in args.fixtures.split(','):
        data = h.fixture(fixture)
        held = h.heldout(fixture)
        parts = []
        for repeat in range(2):
            from mojolearn._cpu_reference import reference_training
            with reference_training() if os.environ.get("MOJOLEARN_VENDOR") == "cpu" else nullcontext():
                result = h.LANES[lane](ml, *data, held.copy())
            parts.append(dict(result))
        expected = reference['cells'][lane + '/' + fixture]['parts'][0]
        rows.append(dict(lane=lane, fixture=fixture, parts=parts, reference=expected,
                         changed_parts=[k for k in parts[0] if parts[0][k] != expected.get(k)],
                         unstable_parts=[k for k in parts[0] if parts[0][k] != parts[1][k]]))
        (out / 'parts.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps({k: rows[-1][k] for k in ('lane', 'fixture', 'changed_parts', 'unstable_parts')}), flush=True)
