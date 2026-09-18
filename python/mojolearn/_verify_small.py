"""Pinned small training pilot. Captures are not admitted reference evidence.

Keep this profile separate from the historical full harness. Its supported
lanes consume their supplied arrays rather than constructing fixed-size fits
internally. Expansion requires reviewing each added lane's input behavior.
"""
import hashlib
from pathlib import Path

PROFILE = 'small-training-v1'
LANES = ('ridge', 'standard-scaler', 'minmax-scaler')
MAX_ROWS = 257
# Below/on/above common row boundaries, then explicit numerical pathologies.
CASES = tuple((f'rows-{n}', 'base', n, 17 if n == 257 else 16)
              for n in (31, 32, 33, 63, 64, 65, 255, 256, 257)) + tuple(
    (kind, kind, 65, 16) for kind in
    ('ties', 'dupes', 'wide', 'denormal', 'denormal_ftz', 'negative'))


def fixture(harness, case, *, heldout=False):
    spec = next((s for s in CASES if s[0] == case), None)
    if spec is None:
        raise ValueError(f'unknown {PROFILE} fixture: {case}')
    _, kind, rows, columns = spec
    # Never call the legacy "odd" kind: it ignores n/d and allocates 12345x17.
    data = harness.fixture(kind, n=rows, d=columns,
                           seed=harness.HELDOUT_SEED if heldout else 0)
    if data[0].shape != (rows, columns) or rows > MAX_ROWS:
        raise ValueError('small training fixture exceeded its declared shape')
    return data


def contract(harness):
    return dict(profile=PROFILE, lanes=list(LANES), max_rows=MAX_ROWS,
        cases=[dict(name=name, kind=kind, rows=rows, columns=columns)
               for name, kind, rows, columns in CASES],
        profile_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(harness.__file__).read_bytes()).hexdigest(),
        repeats_minimum=2, default_reference_compatible=False)


def validate_capture(record):
    """Validate completeness/repeats before comparing any pilot captures.

    Numeric matching is a separate step. Inapplicability cannot substitute for
    a Ridge/scaler train, infer, model or batch hash in this pilot.
    """
    import re
    if record.get('format') != 'mojolearn.small-training-capture.v1':
        raise ValueError('not a small training capture')
    if record.get('contract', {}).get('profile') != PROFILE:
        raise ValueError('different small training profile')
    spec = record['contract']
    expected_cases = [dict(name=n, kind=k, rows=r, columns=d) for n, k, r, d in CASES]
    if (spec.get('lanes') != list(LANES) or spec.get('cases') != expected_cases or
            spec.get('max_rows') != MAX_ROWS or spec.get('repeats_minimum') != 2 or
            spec.get('default_reference_compatible') is not False or
            any(not re.fullmatch('[0-9a-f]{64}', str(spec.get(field, '')))
                for field in ('profile_sha256', 'harness_sha256'))):
        raise ValueError('invalid small training contract')
    if record.get('status') != 'CAPTURED_UNQUALIFIED' or not record.get('complete'):
        raise ValueError('incomplete or failed small training capture')
    if not isinstance(record.get('repeats'), int) or record['repeats'] < 2:
        raise ValueError('at least two actual repeats required')
    expected = {f'{lane}/{case}' for lane in LANES for case, *_ in CASES}
    if set(record.get('cells', {})) != expected:
        raise ValueError('missing or unexpected small training cells')
    if not record.get('bindings') or any(
            not re.fullmatch('[0-9a-f]{64}', str(b.get('sha256', '')))
            or not b.get('module') or not isinstance(b.get('size'), int) or b['size'] <= 0
            for b in record['bindings']):
        raise ValueError('missing native binding witnesses')
    if record.get('device', {}).get('numeric_mode') != 'identical':
        raise ValueError('capture is not identical mode')
    if record['device'].get('vendor') not in ('cpu', 'metal', 'cuda', 'hip'):
        raise ValueError('missing or unknown backend witness')
    inputs = record.get('inputs', {})
    if set(inputs) != {case for case, *_ in CASES} or any(
            set(value) != {'X', 'y_clf', 'y_reg', 'heldout'} or
            any(not re.fullmatch('[0-9a-f]{16}', str(v)) for v in value.values())
            for value in inputs.values()):
        raise ValueError('missing fixture input witnesses')
    for key, cell in record['cells'].items():
        for part in ('train', 'infer', 'model', 'batch'):
            values = cell.get(part, [])
            if (len(values) != record['repeats'] or len(set(values)) != 1 or
                    not all(isinstance(v, str) and re.fullmatch('[0-9a-f]{16}', v)
                            for v in values)):
                raise ValueError(f'{key}/{part}: absent, inapplicable, failed or unstable')
        if cell.get('errors'):
            raise ValueError(f'{key}: native/probe error')


def compare_captures(left, right):
    """Compare measured bytes; agreement alone never promotes a reference."""
    validate_capture(left)
    validate_capture(right)
    if left['contract'] != right['contract'] or left.get('inputs') != right.get('inputs'):
        raise ValueError('different profile, source, protocol or fixture bytes')
    differences = []
    for key in sorted(left['cells']):
        for part in ('train', 'infer', 'model', 'batch'):
            if left['cells'][key][part][0] != right['cells'][key][part][0]:
                differences.append(f'{key}/{part}')
    return dict(status='DIFFERENT' if differences else 'NUMERICAL_MATCH_UNQUALIFIED',
                differences=differences, compared_parts=len(left['cells']) * 4,
                independent_backends=left['device'].get('vendor') != right['device'].get('vendor'),
                reference_admitted=False)
