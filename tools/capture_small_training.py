#!/usr/bin/env python3
"""Maintainer pilot capture; never installs references or certifies a wheel.

Run under the shared native slot / bounded remote controller. Each finished
cell is retained atomically; no full-size fixture or model bundle is run.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def save(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='new external output directory')
    parser.add_argument('--require-backend', choices=('cpu', 'metal', 'cuda', 'hip'), required=True)
    parser.add_argument('--installed', action='store_true', help='import installed package instead of checkout')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('--out must not already exist; captures are never overwritten')
    # Before importing NumPy, mojolearn or any native binding.
    for name in ('MOJOLEARN_CPU_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
                 'OMP_THREAD_LIMIT'):
        os.environ[name] = '1'
    os.environ['OMP_MAX_ACTIVE_LEVELS'] = '1'
    os.environ['MOJOLEARN_NUMERIC_MODE'] = 'identical'
    root = Path(__file__).resolve().parents[1]
    if not args.installed:
        sys.path.insert(0, str(root / 'python'))
    import mojolearn as ml
    from mojolearn import _verify_all as suite, _verify_small as profile, _verify
    from mojolearn._cpu_reference import reference_training
    if ml.vendor() != args.require_backend or ml.numeric_mode() != 'identical':
        parser.error(f'expected {args.require_backend}/identical, got {ml.vendor()}/{ml.numeric_mode()}')
    harness = suite.load_harness()
    args.out.mkdir(parents=True)
    path = args.out / 'capture.json'
    started = time.monotonic()
    record = dict(format='mojolearn.small-training-capture.v1', complete=False,
        status='INCOMPLETE', reference_admitted=False, repeats=2,
        contract=profile.contract(harness), inputs={}, cells={},
        device=suite._device_block(ml, harness), package_dir=str(Path(ml.__file__).parent),
        capture_tool_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest())
    if ml.vendor() == 'cpu':
        # Include native column/fault-define readback, not just file labels.
        record['host'] = harness.host_record(ml)
    save(path, record)
    try:
        with reference_training():
            for case, *_ in profile.CASES:
                data = profile.fixture(harness, case)
                heldout = profile.fixture(harness, case, heldout=True)[0]
                record['inputs'][case] = dict(zip(('X', 'y_clf', 'y_reg', 'heldout'),
                                                  map(harness._h, (*data, heldout))))
                for lane in profile.LANES:
                    cell = {part: [] for part in ('train', 'infer', 'model', 'batch')}
                    errors = []
                    cell_start = time.monotonic()
                    for repeat in range(2):
                        # Independent fits, each with fresh mutable inputs.
                        result = suite.run_cell(harness, ml, lane, case,
                            tuple(array.copy() for array in data), heldout.copy(), repeats=1)
                        for part in cell:
                            value, error = result[part]
                            cell[part].append(value)
                            if error:
                                errors.append(dict(repeat=repeat, part=part, error=error))
                    cell['errors'] = errors
                    cell['seconds'] = round(time.monotonic() - cell_start, 6)
                    record['cells'][f'{lane}/{case}'] = cell
                    record['elapsed_seconds'] = round(time.monotonic() - started, 6)
                    save(path, record)
                    print(f'{lane}/{case}: {cell["seconds"]:.3f}s', flush=True)
        record['bindings'] = _verify.binding_artifacts() + suite.host_binding_artifacts()
        record['complete'] = True
        record['status'] = 'CAPTURED_UNQUALIFIED'
        profile.validate_capture(record)
    except Exception as exc:
        record['status'] = 'FAILED'
        record['error'] = f'{type(exc).__name__}: {exc}'
        save(path, record)
        print(record['error'], file=sys.stderr)
        return 1
    save(path, record)
    print(f'CAPTURED_UNQUALIFIED: {len(record["cells"])} cells; {path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
