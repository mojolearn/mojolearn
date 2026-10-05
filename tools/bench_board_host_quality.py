"""Untimed, own-implementation host references for opponent-free boards."""
import json
import os
from pathlib import Path
import subprocess
import sys


def compare(actual, reference):
    import numpy as np
    if not actual or actual.keys() != reference.keys():
        raise ValueError('Host reference output fields differ')
    errors = []
    for key in actual:
        a, b = np.asarray(actual[key]), np.asarray(reference[key])
        if a.shape != b.shape or not a.size:
            raise ValueError('Host reference output shape differs: ' + key)
        if a.dtype.kind not in 'biuf' or b.dtype.kind not in 'biuf':
            raise ValueError('Non-numeric host reference output: ' + key)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Nonfinite host reference output: ' + key)
        # Integer outputs (tokens, indices, dimensions) have an exact contract.
        if a.dtype.kind in 'biu' and b.dtype.kind in 'biu':
            if not np.array_equal(a, b):
                raise ValueError('Integer host reference mismatch: ' + key)
            errors.append(0.)
            continue
        a, b = a.astype(np.float64), b.astype(np.float64)
        denominator = max(float(np.linalg.norm(b)), 1e-30)
        errors.append(float(np.linalg.norm(a - b)) / denominator)
    # Report numerical distance as quality, not identity qualification.
    # FAST has no cross-vendor bitwise promise. Nonzero error remains visible.
    return {'relative_error_vs_own_host': max(errors)}


def enrich(lane, inputs, outputs, quality, python, directory, timeout):
    """Only replace explicitly empty quality, never an existing quality error."""
    import numpy as np
    missing = [arm for arm in outputs if arm.startswith('ours') and quality.get(arm) == {}]
    if not missing:
        return None
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    data, ref, receipt = root / 'inputs.npz', root / 'host.npz', root / 'receipt.json'
    np.savez(data, **inputs)
    env = dict(os.environ, MOJOLEARN_VENDOR='cpu', MOJOLEARN_NUMERIC_MODE='identical')
    for key in ('PYTHONPATH', 'MOJOLEARN_CUDA_PATH', 'MOJOLEARN_EXPERIMENTAL_PTX',
                'MOJOLEARN_GPU_ARCH', 'MOJOLEARN_BOARD_ARTIFACT_MANIFEST', 'MOJOLEARN_BOARD_RECEIPTS'):
        env.pop(key, None)
    try:
        with (root / 'host.log').open('w') as log:
            result = subprocess.run([python, str(Path(__file__).resolve()), lane,
                                     str(data), str(ref), str(receipt)],
                                    env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        if result.returncode:
            raise ValueError('Own host reference failed; see ' + str(root / 'host.log'))
        proof = json.loads(receipt.read_text())
        if proof.get('vendor') != 'cpu' or proof.get('numeric_mode') != 'identical' or not proof.get('bindings'):
            raise ValueError('Missing own host reference provenance')
        with np.load(ref, allow_pickle=False) as saved:
            reference = {key: saved[key] for key in saved.files}
        for arm in missing:
            try:
                quality[arm] = compare(outputs[arm], reference)
            except Exception as error:
                quality[arm] = {'error': str(error)}
    except Exception as error:
        for arm in missing:
            quality[arm] = {'error': str(error)}
    return str(receipt)


def main():
    import numpy as np
    import mojolearn as ml
    from mojolearn._verify import binding_artifacts
    import bench_board_algos as board
    lane, data, output, receipt = sys.argv[1:]
    if ml.vendor() != 'cpu' or ml.numeric_mode() != 'identical':
        raise RuntimeError('Reference must use our explicit IDENTICAL host column')
    with np.load(data, allow_pickle=False) as saved:
        inputs = {key: saved[key] for key in saved.files}
    runner = board.build(lane, 'ours', inputs)
    runner.fit()
    runner.infer()
    np.savez(output, **runner.outputs())
    bindings = binding_artifacts()
    if not bindings or any(not row['module'].endswith('_host') for row in bindings):
        raise RuntimeError('Reference loaded a non-host binding')
    Path(receipt).write_text(json.dumps({'vendor': ml.vendor(), 'numeric_mode': ml.numeric_mode(),
                                        'lane': lane, 'bindings': bindings, 'timed': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
