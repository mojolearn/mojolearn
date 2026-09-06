#!/usr/bin/env python3
"""Main-only diagnostic: capture Mamba3 L4 public angle/key state in one mode.

Example, in an isolated wheel environment on the named GPU:
  MOJOLEARN_NUMERIC_MODE=fast python tools/mamba3_mode_state_probe.py \
    --source-root /repo --expected-vendor cuda --output /artifacts/m3-fast.json

Run modes serially as separate processes. This does not certify correctness,
change tolerances, or infer which source produced an installed binary. Source
hashes describe the supplied checkout; the actual loaded binary has its own
hash. Compare theta_last_[1,0,11] and k_last_[1,0,22:24] across modes to locate
angle versus normalized-key differences before changing arithmetic.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

# Must precede importing NumPy or MojoLearn; the controller owns GPU deadlines.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

MODES = {'fast': 0, 'deterministic': 2, 'identical': 1}
CASE = 'mamba/corpus/mamba3/m3_base_b2_l4_d32'
WEIGHTS = {
    'block_norm.weight': (32,), 'in_proj.weight': (419, 32), 'dt_bias': (1,),
    'B_norm.weight': (128,), 'C_norm.weight': (128,), 'B_bias': (1, 128),
    'C_bias': (1, 128), 'D': (1,), 'out_proj.weight': (32, 64),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--expected-vendor', choices=('metal', 'cuda', 'hip'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.source_root.resolve(), args.output.resolve()
    require(not output.exists(), 'Refusing to overwrite diagnostic evidence')
    mode = os.environ.get('MOJOLEARN_NUMERIC_MODE', '').strip().lower()
    require(mode in MODES, 'Set MOJOLEARN_NUMERIC_MODE explicitly')
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        'schema': 'mojolearn.mamba3.mode-state-diagnostic.v1', 'status': 'INCOMPLETE',
        'started_utc': datetime.now(timezone.utc).isoformat(), 'requested_mode': mode,
        'expected_vendor': args.expected_vendor, 'fixture': CASE,
        'shape': {'batch': 2, 'length': 4, 'd_model': 32, 'heads': 1},
        'focus': {'batch': 1, 'head': 0, 'last_token': 3, 'rotary_pair': 11,
                  'components': [22, 23], 'failing_flat_key_index': 151},
        'scope': 'Diagnostic state capture only; no pass/tolerance/cross-vendor certification',
        'source_binding_relationship': 'Source inventory is descriptive; build provenance is not inferred',
        'harness_sha256': sha(Path(__file__)),
    }

    def save():
        output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')

    save()
    try:
        import numpy as np
        import mojolearn

        package = Path(mojolearn.__file__).resolve()
        require(package.is_relative_to(Path(sys.prefix).resolve()) and 'site-packages' in package.parts,
                'Run this diagnostic against an isolated installed package')
        require(mojolearn.numeric_mode() == mode, 'Package default mode differs')
        fixture_files = {CASE + '/x.f32': sha(root / CASE / 'x.f32')}

        def read(name, shape):
            path = root / CASE / (name + '.f32')
            value = np.fromfile(path, dtype='<f4')
            require(value.size == int(np.prod(shape)), 'Wrong fixture length: ' + name)
            value = value.reshape(shape)
            require(np.isfinite(value).all(), 'Nonfinite fixture: ' + name)
            fixture_files[str(path.relative_to(root))] = sha(path)
            return value

        weights = {name: read(name, shape) for name, shape in WEIGHTS.items()}
        x = read('x', (2, 4, 32))
        source_paths = set((root / 'mamba/impl').rglob('*.mojo'))
        source_paths.update((root / 'gemm/checks').glob('*.mojo'))
        for relative in ('checks/numerics.mojo', 'bindings/_mojolearn_mamba.mojo',
                         'bindings/build_mamba.sh', 'python/mojolearn/_mamba_impl.py',
                         'python/mojolearn/_mode.py', 'python/mojolearn/_backend.py', 'pixi.lock'):
            source_paths.add(root / relative)
        source_files = {str(p.relative_to(root)): sha(p) for p in sorted(source_paths)}
        record.update(source_root=str(root), source_files_sha256=source_files,
                      fixture_files_sha256=fixture_files,
                      source_inventory_sha256=hashlib.sha256(json.dumps(
                          source_files, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
        model = mojolearn.Mamba3Block(weights, numeric_mode=mode)
        binding = model._bind()
        binary = Path(binding.__file__).resolve()
        code, vendor = int(binding.mamba_numeric_mode()), str(binding.mamba_vendor())
        require(code == MODES[mode] and vendor == args.expected_vendor, 'Native mode/vendor mismatch')
        require(binary.is_relative_to(package.parent), 'Binding outside installed package')
        binary_hash = sha(binary)
        record['installed'] = dict(package=str(package), version=mojolearn.__version__,
                                   python=sys.version, binding=str(binary), binding_sha256=binary_hash,
                                   native_mode_code=code, native_vendor=vendor)
        save()
        state = model.allocate_state(2)
        model.forward(x, state)

        def bits(array, shape):
            value = np.asarray(array)
            require(value.dtype == np.float32 and value.shape == shape, 'Unexpected state shape/dtype')
            require(np.isfinite(value).all(), 'Nonfinite public state')
            raw = np.ascontiguousarray(value, dtype='<f4')
            return {'shape': list(shape), 'dtype': 'float32', 'uint32': raw.view('<u4').tolist(),
                    'float32_le_sha256': hashlib.sha256(raw.tobytes()).hexdigest()}

        record['theta_last'] = bits(model.theta_last_, (2, 1, 32))
        record['k_last'] = bits(model.k_last_, (2, 1, 128))
        record['focus']['theta_uint32'] = record['theta_last']['uint32'][1][0][11]
        record['focus']['key_pair_uint32'] = record['k_last']['uint32'][1][0][22:24]
        require(sha(binary) == binary_hash, 'Loaded binary changed during probe')
        for relative, expected in {**source_files, **fixture_files}.items():
            require(sha(root / relative) == expected, 'Source/fixture changed during probe: ' + relative)
        record['status'] = 'DIAGNOSTIC_COMPLETE'
    except Exception as exc:
        record.update(status='DIAGNOSTIC_ERROR', error={'type': type(exc).__name__, 'message': str(exc)})
    finally:
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps({'status': record['status'], 'output': str(output), 'focus': record['focus']}), flush=True)
    return 0 if record['status'] == 'DIAGNOSTIC_COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
