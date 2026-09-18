# SPDX-License-Identifier: Apache-2.0
"""Importable scorer/receipt comparator for the physical CV validation runner."""
import hashlib
import json
from pathlib import Path
import struct

__all__ = ["array_digest", "score_with_witness", "read_records", "compare_records"]


def array_digest(value):
    import numpy as np
    a = np.asarray(value)
    if a.dtype.kind not in 'fiub':
        raise TypeError('CV witness requires numeric array bytes')
    h = hashlib.sha256()
    for part in (a.dtype.str.encode(), json.dumps(list(a.shape)).encode(), a.tobytes(order='C')):
        h.update(len(part).to_bytes(8, 'little'))
        h.update(part)
    return h.hexdigest()


def score_with_witness(estimator, X, y, *, directory):
    """Record complete GBDT model/prediction bytes before returning the score.

    Driver inventory is placement evidence. No profiler/kernel-execution
    witness is manufactured from it or from a compiled vendor label.
    """
    from mojolearn import GradientBoosting, _backend
    from mojolearn._gpu_witness import visible_gpu_inventory
    vendor = _backend.vendor()
    inventory = visible_gpu_inventory(vendor)
    binding = _backend.binding('_mojolearn_gbdt', 'identical')
    if binding.gbdt_vendor() != vendor or binding.gbdt_numeric_mode() != 1:
        raise RuntimeError('CV witness requires matching GPU IDENTICAL GBDT binding')
    directory = Path(directory)
    key = hashlib.sha256((array_digest(X) + array_digest(y)).encode()).hexdigest()
    models = directory / 'models'
    models.mkdir(exist_ok=True)
    learner = estimator._learner_
    model_path = models / (key + '.npz')
    if model_path.exists():
        raise RuntimeError('duplicate fold input would overwrite its model witness')
    learner.save(model_path)
    restored = GradientBoosting.load(model_path)
    raw, reload = array_digest(learner.predict(X)), array_digest(restored.predict(X))
    if raw != reload:
        raise RuntimeError('CV fitted model save/reload changed prediction bytes')
    score = float(estimator.score(X, y))
    record = dict(fixture=key, model=hashlib.sha256(str(learner.model_).encode()).hexdigest(),
                  predict=array_digest(estimator.predict(X)), raw_predict=raw, reload_predict=reload,
                  loss_curve=array_digest(learner.loss_curve_), score=struct.pack('<d', score).hex(),
                  inventory=inventory, binding_sha256=hashlib.sha256(Path(binding.__file__).read_bytes()).hexdigest(),
                  selected_arch=_backend.gpu_arch(), selected_arch_how=_backend.gpu_arch_how())
    with (directory / (key + '.json')).open('x') as f:
        json.dump(record, f, indent=2)
        f.write('\n')
    return score


def read_records(directory, folds):
    records = [json.loads(p.read_text()) for p in sorted(Path(directory).glob('*.json'))]
    if len(records) != folds or len({r['fixture'] for r in records}) != folds:
        raise ValueError('missing or duplicate fold witnesses')
    return {r['fixture']: r for r in records}


def compare_records(expected, actual, *, vendor, workers):
    """Exact numerical/placement comparison; deliberately not full GPU admission."""
    from mojolearn._gpu_witness import require_distinct_workers
    if not expected or expected.keys() != actual.keys():
        raise ValueError('fold inputs differ')
    inventories, bindings = {}, set()
    for key, record in actual.items():
        for part in ('model', 'predict', 'raw_predict', 'reload_predict', 'loss_curve', 'score'):
            value = record.get(part)
            if not value or value != expected[key].get(part):
                raise ValueError(f'{key}: {part} differs or is missing')
        inventory = record['inventory']
        pid = inventory['pid']
        if pid in inventories and inventory != inventories[pid]:
            raise ValueError('worker device identity changed between folds')
        inventories[pid] = inventory
        binding = record.get('binding_sha256')
        if not binding or binding != expected[key].get('binding_sha256'):
            raise ValueError('worker GBDT binding differs from baseline')
        bindings.add(binding)
    if len(bindings) != 1:
        raise ValueError('workers did not use the same native GBDT artifact')
    require_distinct_workers(list(inventories.values()), vendor, workers)
